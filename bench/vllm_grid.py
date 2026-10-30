#!/usr/bin/env python3
"""Run the `vllm bench latency` grid for Level 3 and append records to data/<tag>/vllm_latency.jsonl.

Runs ON THE GPU BOX (called by bench/run_vllm.sh). Each grid point is its own `vllm bench
latency` process (fresh engine, no state carried between points) with a timeout. A failed
point writes <raw json>.error.txt and the grid continues. Re-running skips points already
recorded in vllm_latency.jsonl, so an interrupted grid can be resumed.

Grid (Qwen/Qwen3-8B, bf16, --max-model-len 16384, warmup 3, iters 10):
    decode:  batch {1,2,4,8,16,32,64} x input_len 512 x output_len {1, 128}
             -> TPOT(batch) = (lat(out=128) - lat(out=1)) / 127
    prefill: batch 1 x input_len {128, 512, 2048, 8192} x output_len 1
             -> TTFT(input_len) ~= lat(out=1)

Engine settings (recorded in every record):
  * CUDA graphs ON: we do not pass --enforce-eager (vLLM default is enforce_eager=False).
    If this vLLM's CLI has the BooleanOptionalAction form, we pass --no-enforce-eager to
    make it explicit. (The literal `--enforce-eager false` is NOT valid vLLM syntax.)
  * Prefix caching OFF: `vllm bench latency` sets enable_prefix_caching=False itself.
  * --max-num-batched-tokens 16384 by default (MAX_NUM_BATCHED_TOKENS env / flag; empty =
    vLLM's own default). Fixing it makes chunked prefill identical on every GPU; batch 64 x
    512 = 32768 prompt tokens is then prefilled in 2 chunks (perfsim.engine.prefill_passes).
  * Random prompt token ids, temperature 1.0, ignore_eos=True (set by vllm bench latency):
    every sequence writes exactly output_len tokens.

Output record (CONTRACTS.md keys first, extras after):
  {"engine":"vllm","batch","input_len","output_len","avg_latency_s","p50_latency_s",
   "p10_latency_s","p90_latency_s","p99_latency_s","latencies_s":[...],"num_iters","num_iters_warmup",
   "model","dtype","max_model_len","max_num_batched_tokens","cuda_graphs":true,"enforce_eager":false,
   "vllm_version","torch_version","gpu_tag","raw_json","unix_time"}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-8B"
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64)
DECODE_INPUT = 512
DECODE_OUTPUTS = (1, 128)
PREFILL_INPUTS = (128, 512, 2048, 8192)


def build_grid() -> List[Dict[str, int]]:
    """Unique (batch, input_len, output_len) points; cheap ones first."""
    pts = [(b, DECODE_INPUT, o) for b in DECODE_BATCHES for o in DECODE_OUTPUTS]
    pts += [(1, s, 1) for s in PREFILL_INPUTS]
    seen, out = set(), []
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append({"batch": p[0], "input_len": p[1], "output_len": p[2]})
    return out


# ---------------------------------------------------------------------------------------
# parsing the --output-json file (defensive: key names have changed across vLLM versions)
# ---------------------------------------------------------------------------------------
def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method), q in [0, 100]."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def parse_latency_json(obj: Dict) -> Dict[str, Optional[float]]:
    """vLLM bench latency --output-json -> latency stats in seconds.

    vLLM (checked against v0.32.0 vllm/benchmarks/latency.py) writes
        {"model_id", "avg_latency", "latencies": [...], "percentiles": {"10": .., "25", "50", "75", "90", "99"}}
    (percentile keys are ints in Python, strings after JSON). We accept int/str/"p50"
    keys and fall back to computing from `latencies`; anything missing is None.
    """
    lat = [float(x) for x in (obj.get("latencies") or []) if isinstance(x, (int, float))]
    pct = obj.get("percentiles") or {}

    def pick(q: int) -> Optional[float]:
        for k in (str(q), q, f"p{q}", float(q), f"{float(q)}"):
            if isinstance(pct, dict) and k in pct and pct[k] is not None:
                return float(pct[k])
        return _percentile(sorted(lat), q) if lat else None

    avg = obj.get("avg_latency", obj.get("avg_latency_s", obj.get("mean_latency")))
    if avg is None and lat:
        avg = statistics.fmean(lat)
    return {
        "avg_latency_s": float(avg) if avg is not None else None,
        "p10_latency_s": pick(10), "p50_latency_s": pick(50),
        "p90_latency_s": pick(90), "p99_latency_s": pick(99),
        "latencies_s": lat,
    }


# ---------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------
def vllm_cmd() -> List[str]:
    """`vllm` CLI if on PATH, else the module entry point of the same CLI."""
    exe = shutil.which("vllm")
    return [exe] if exe else [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def help_text(base: List[str]) -> str:
    """Full CLI help. Recent vLLM prints only a group summary for plain --help and needs
    --help=all for every flag; older versions reject --help=all, so try both."""
    text = ""
    for flag in ("--help=all", "--help"):
        try:
            r = subprocess.run(base + ["bench", "latency", flag], capture_output=True, text=True, timeout=300)
            text = r.stdout + r.stderr
            if r.returncode == 0 and "--max-model-len" in text:
                return text
        except Exception as e:  # noqa: BLE001
            text = f"(help failed: {e})"
    return text


def versions() -> Dict[str, Optional[str]]:
    out = {}
    for mod in ("vllm", "torch"):
        try:
            r = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"],
                               capture_output=True, text=True, timeout=300)
            out[mod] = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def done_keys(jsonl: Path) -> set:
    keys = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("avg_latency_s") is not None:
                    keys.add((int(r["batch"]), int(r["input_len"]), int(r["output_len"])))
            except Exception:  # noqa: BLE001
                continue
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="gpu_tag, e.g. h100_sxm or rtx5090")
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-batched-tokens", default=os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384"),
                    help="'' = vLLM default")
    ap.add_argument("--gpu-memory-utilization", default=os.environ.get("GPU_MEM_UTIL", ""),
                    help="'' = vLLM default (0.9)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("VLLM_POINT_TIMEOUT", "1800")),
                    help="seconds per grid point (model load + torch.compile + graph capture + runs)")
    ap.add_argument("--only", default="", help="comma list of b:in:out to run, e.g. 1:512:1,8:512:128")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args(argv)

    out_dir = Path(args.data) / args.tag
    raw_dir = out_dir / "vllm"
    if not args.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "vllm_latency.jsonl"

    base = vllm_cmd()
    ver = {"vllm": None, "torch": None} if args.dry_run else versions()
    htxt = "" if args.dry_run else help_text(base)
    if htxt:
        (raw_dir / "bench_latency_help.txt").write_text(htxt)  # the exact CLI this run used
    no_eager_flag = "--no-enforce-eager" in htxt
    print(f"vllm {ver['vllm']}  torch {ver['torch']}  cli: {' '.join(base)}  "
          f"--no-enforce-eager supported: {no_eager_flag}", flush=True)

    grid = build_grid()
    if args.only:
        want = {tuple(int(x) for x in s.split(":")) for s in args.only.split(",") if s}
        grid = [g for g in grid if (g["batch"], g["input_len"], g["output_len"]) in want]
    have = done_keys(jsonl)

    n_ok = n_fail = n_skip = 0
    for g in grid:
        b, s, o = g["batch"], g["input_len"], g["output_len"]
        name = f"b{b}_in{s}_out{o}"
        raw = raw_dir / f"{name}.json"
        err = raw_dir / f"{name}.json.error.txt"
        if (b, s, o) in have:
            print(f"[{name}] already in {jsonl.name}, skipped", flush=True)
            n_skip += 1
            continue
        cmd = base + ["bench", "latency", "--model", args.model, "--dtype", args.dtype,
                      "--max-model-len", str(args.max_model_len),
                      "--batch-size", str(b), "--input-len", str(s), "--output-len", str(o),
                      "--num-iters-warmup", str(args.warmup), "--num-iters", str(args.iters),
                      "--output-json", str(raw)]
        if args.max_num_batched_tokens:
            cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
        if args.gpu_memory_utilization:
            cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        if no_eager_flag:
            cmd += ["--no-enforce-eager"]  # CUDA graphs on (also the default)
        print(f"[{name}] + {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        log = raw_dir / f"{name}.log"
        raw.unlink(missing_ok=True)
        t0 = time.time()
        try:
            with open(log, "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout)
            if r.returncode != 0:
                raise RuntimeError(f"exit code {r.returncode}")
            if not raw.exists():
                raise RuntimeError("no --output-json file written")
            stats = parse_latency_json(json.loads(raw.read_text()))
            if stats["avg_latency_s"] is None:
                raise RuntimeError(f"no latency in {raw.name}: keys {sorted(json.loads(raw.read_text()))}")
        except Exception as e:  # noqa: BLE001  (TimeoutExpired included)
            tail = ""
            try:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-60:])
            except OSError:
                pass
            err.write_text(f"{type(e).__name__}: {e}\ncmd: {' '.join(cmd)}\nelapsed_s: {time.time()-t0:.0f}\n"
                           f"--- last lines of {log.name} ---\n{tail}\n")
            print(f"[{name}] FAILED ({type(e).__name__}: {e}); see {err}", flush=True)
            n_fail += 1
            continue
        err.unlink(missing_ok=True)
        rec = {"engine": "vllm", "batch": b, "input_len": s, "output_len": o,
               "avg_latency_s": stats["avg_latency_s"], "p50_latency_s": stats["p50_latency_s"],
               "p10_latency_s": stats["p10_latency_s"], "p90_latency_s": stats["p90_latency_s"],
               "p99_latency_s": stats["p99_latency_s"], "latencies_s": stats["latencies_s"],
               "num_iters": args.iters, "num_iters_warmup": args.warmup,
               "model": args.model, "dtype": args.dtype, "max_model_len": args.max_model_len,
               "max_num_batched_tokens": int(args.max_num_batched_tokens) if args.max_num_batched_tokens else None,
               "cuda_graphs": True, "enforce_eager": False,
               "vllm_version": ver["vllm"], "torch_version": ver["torch"], "gpu_tag": args.tag,
               "raw_json": str(raw.relative_to(out_dir)), "unix_time": time.time()}
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{name}] avg {stats['avg_latency_s']:.4f} s  p50 {stats['p50_latency_s']:.4f} s  "
              f"({time.time()-t0:.0f} s wall)", flush=True)
        n_ok += 1

    print(f"vllm grid: {n_ok} ok, {n_fail} failed, {n_skip} skipped -> {jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
