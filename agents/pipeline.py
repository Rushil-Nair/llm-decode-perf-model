"""Automated model-improvement loop, callable by a Claude Code agent or by hand:

    python -m agents.pipeline [--run] [--data data] [--out results] [--threshold 15]

  1. (--run only) run the GPU benchmarks (gpu/run_all.sh) -- needs a GPU box
  2. fit calibration for every data/<gpu_tag>/ with ops.jsonl -> data/<gpu_tag>/calib.json
     (bench/fit_calib.py main() if present, else perfsim.ops_model.fit_calibration)
  3. run the correlation (correlate.correlate.run) -> results/correlation.md + CSVs
  4. flag every op/config with |error| > threshold and write a one-line hypothesis
     for each (rule-based; it names the op group that dominates the predicted time)
  5. append a timestamped entry to results/AGENT_LOG.md, and write results/flags.csv

Plain Python, no agent framework needed. Each step catches its own errors and the log
records what ran, so a failed step never hides the others. `--json` prints a
machine-readable summary on stdout for a calling agent.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from correlate import correlate as C
from correlate import loaders as L
from correlate.backend import Backend, default_backend
from correlate.metrics import flag_rows, fmt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLD = 15.0


# ---------------------------------------------------------------------------------
# steps 1-2
# ---------------------------------------------------------------------------------
def step_bench(run: bool, cmd: Optional[List[str]] = None) -> Dict:
    if not run:
        return {"step": "bench", "status": "skipped", "detail": "--run not set (using existing data/)"}
    cmd = cmd or ["bash", str(REPO / "gpu" / "run_all.sh")]
    if cmd[0] == "bash" and not Path(cmd[1]).exists():
        return {"step": "bench", "status": "skipped", "detail": f"{cmd[1]} not found"}
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=6 * 3600)
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        return {"step": "bench", "status": "ok" if p.returncode == 0 else f"exit {p.returncode}",
                "detail": " | ".join(tail)}
    except Exception as e:
        return {"step": "bench", "status": "error", "detail": f"{type(e).__name__}: {e}"}


def _load_fit_calib(path: Path):
    spec = importlib.util.spec_from_file_location("fit_calib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step_fit(data_root: Path, fit_script: Optional[Path] = None, backend: Optional[Backend] = None,
             results_dir: Optional[Path] = None) -> Dict:
    """Fit data/<gpu_tag>/calib.json from data/<gpu_tag>/ops.jsonl for every GPU.

    1. bench/fit_calib.py if it exists: main(["--data", data/<tag>, "--results", <results>]) is
       called in-process once per GPU (it writes calib.json and results/<tag>_ops_sol.md);
    2. else perfsim.ops_model.fit_calibration(records, hw, gpu_tag).save(calib.json) directly.
    """
    data_root = Path(data_root)
    fit_script = Path(fit_script) if fit_script else REPO / "bench" / "fit_calib.py"
    with_ops = [d.name for d in L.gpu_dirs(data_root) if (d / "ops.jsonl").exists()]
    if not with_ops:
        return {"step": "fit", "status": "skipped", "detail": f"no data/<gpu_tag>/ops.jsonl under {data_root}"}
    if fit_script.exists():
        try:
            mod = _load_fit_calib(fit_script)
        except Exception as e:
            return {"step": "fit", "status": "error", "detail": f"cannot import {fit_script}: {type(e).__name__}: {e}"}
        if not hasattr(mod, "main"):
            return {"step": "fit", "status": "error", "detail": f"{fit_script} has no main(argv)"}
        done, problems = [], []
        for tag in with_ops:
            argv = ["--data", str(data_root / tag)] + (["--results", str(results_dir)] if results_dir else [])
            try:
                rc = mod.main(argv)
            except SystemExit as e:
                rc = e.code
            except Exception as e:
                rc = f"{type(e).__name__}: {e}"
            if rc in (0, None) and (data_root / tag / "calib.json").exists():
                done.append(tag)
            else:
                problems.append(f"{tag}: {rc}")
        status = "ok" if done and not problems else ("partial" if done else "error")
        return {"step": "fit", "status": status,
                "detail": f"{fit_script.name} -> calib.json for {done}" + (f"; failed {problems}" if problems else "")}
    backend = backend or default_backend()
    if backend.fit_calibration is None:
        return {"step": "fit", "status": "skipped",
                "detail": f"{fit_script} not found and perfsim.ops_model.fit_calibration unavailable; existing calib.json used"}
    from perfsim.hardware import load_hardware
    hws, done, problems = load_hardware(), [], []
    for tag in with_ops:
        try:
            if tag not in hws:
                problems.append(f"{tag}: no hw/hardware.yaml entry")
                continue
            recs = L.load_ops(data_root / tag)
            calib = backend.fit_calibration(recs, hws[tag], gpu_tag=tag)
            calib.save(data_root / tag / "calib.json")
            done.append(tag)
        except Exception as e:
            problems.append(f"{tag}: {type(e).__name__}: {e}")
    status = "ok" if done and not problems else ("partial" if done else "error")
    return {"step": "fit", "status": status,
            "detail": f"ops_model.fit_calibration -> calib.json for {done}" + (f"; {problems}" if problems else "")}


# ---------------------------------------------------------------------------------
# step 4: flags and hypotheses
# ---------------------------------------------------------------------------------
def _dominant(err: Dict) -> Optional[str]:
    shares = {g: err.get(f"share_{g}") for g in ("weights", "attention", "norms", "overhead")}
    shares = {g: v for g, v in shares.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return max(shares, key=shares.get) if shares else None


def hypothesis(err: Dict) -> str:
    """One-line, testable hypothesis for a flagged error record (rule-based)."""
    e = float(err["err_pct"])
    item = err.get("item", "")
    prefix = ""
    if err.get("kind") == C.KIND_LOGO:
        prefix = (f"LOGO {err.get('calib_from')}->{err.get('gpu')}: efficiency curves do not transfer "
                  f"(target reaches a different fraction of its datasheet BW/peak); ")
    if err.get("level") == "L1":
        if e < 0:
            return (f"{item} under-predicted ({e:+.0f}%) at {fmt(err.get('bif_min'), '{:.2e}')}-"
                    f"{fmt(err.get('bif_max'), '{:.2e}')} B in flight: effective latency is below the pointer-chase "
                    f"DRAM latency (L2 hits / prefetch / row-buffer locality) or bytes_per_load undercounts "
                    f"in-flight bytes")
        return (f"{item} over-predicted ({e:+.0f}%): outstanding loads are capped below warps x ILP "
                f"(registers, MSHR/LSU slots) or loaded latency grows faster than the queueing term (alpha too small)")
    sh = lambda g: err.get(f"share_{g}") or 0.0
    if item.startswith("vLLM TPOT"):
        b, ctx = err.get("batch"), err.get("context_len")
        if e < 0:   # model too fast
            if sh("attention") >= 0.25 or (ctx or 0) >= 8192:
                return prefix + (f"decode attention under-predicted at kv_len>={ctx} (attention "
                                 f"{100 * sh('attention'):.0f}% of predicted step): model ignores paged-KV gather "
                                 f"inefficiency and split-KV reduction cost")
            if (b or 0) <= 4:
                return prefix + (f"per-step overhead under-modelled at batch {b} (overhead {100 * sh('overhead'):.0f}% "
                                 f"of predicted step): CPU scheduling, sampling and launch gaps not in the model")
            return prefix + (f"weight GEMM efficiency at M={b} below the fitted curve (weights "
                             f"{100 * sh('weights'):.0f}% of predicted step): tile quantization / split-K not captured")
        if _dominant(err) == "overhead":
            return prefix + f"fixed per-step overhead over-estimated at batch {b}: CUDA graphs hide launch cost"
        return prefix + (f"decode over-predicted at batch {b}: GEMM efficiency curve too pessimistic at M={b} "
                         f"(ops.jsonl may not cover this M; extrapolated efficiency too low)")
    if item.startswith("vLLM TTFT"):
        il, b = err.get("input_len"), err.get("batch")
        if e < 0:
            if (il or 0) * (b or 1) <= 1024:
                return prefix + (f"prefill fixed cost not modelled at {il} tokens x batch {b}: scheduling, "
                                 f"tokenization, first decode step dominate a short prefill")
            return prefix + (f"prefill under-predicted at {il} tokens x batch {b}: GEMM/attention efficiency at "
                             f"{il * (b or 1)} tokens below fit (causal attention cost or chunked prefill not modelled)")
        return prefix + (f"prefill over-predicted at {il} tokens x batch {b}: compute-bound GEMMs reach a higher "
                         f"fraction of peak than the fitted curve")
    if item.startswith("llama.cpp slot"):
        k = err.get("passes_per_step")
        if k is not None and float(k) > 1.2 and e < 0:
            return prefix + (f"llama.cpp step cost over-modelled with k={k} passes (build={err.get('build')}, "
                             f"users={err.get('users')}): extra passes are cheaper than a full weight re-read "
                             f"(L2 reuse, smaller sub-batches)")
        if e < 0:
            return prefix + (f"llama.cpp per-pass cost over-modelled at k={k} (build={err.get('build')}, "
                             f"users={err.get('users')}): Q4_K weight bytes per pass or the q4 GEMV efficiency "
                             f"proxy too pessimistic")
        if e > 0:
            return prefix + (f"llama.cpp step cost under-modelled (build={err.get('build')}, users={err.get('users')}, "
                             f"k={k}): Q4_K dequant/MMQ kernels below fitted efficiency, or the pass counter undercounts")
    if item.startswith("llama.cpp fix/master"):
        return prefix + (f"speedup mis-predicted at {err.get('users')} users: per-pass cost a vs per-sequence cost c "
                         f"split is off (k_master={err.get('k_master')}, k_fix={err.get('k_fix')})")
    return prefix + f"{item} error {e:+.0f}%: no rule matched; inspect the per-op attribution"


def flag(errors: List[Dict], threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    out = []
    for r in flag_rows(errors, threshold):
        out.append({"level": r.get("level"), "gpu": r.get("gpu"), "kind": r.get("kind"),
                    "calib_from": r.get("calib_from"), "item": r.get("item"), "config": r.get("config"),
                    "measured": r.get("measured"), "predicted": r.get("predicted"), "unit": r.get("unit"),
                    "err_pct": r.get("err_pct"), "dominant_group": _dominant(r), "hypothesis": hypothesis(r)})
    return out


# ---------------------------------------------------------------------------------
# step 5: log
# ---------------------------------------------------------------------------------
def log_entry(steps: List[Dict], res: Optional[Dict], flags: List[Dict], threshold: float, now: str) -> str:
    s = [f"## {now} pipeline run", ""]
    for st in steps:
        s.append(f"- {st['step']}: {st['status']}" + (f" ({st['detail']})" if st.get("detail") else ""))
    if res:
        s.append(f"- GPUs: {', '.join(res['gpus']) or 'none'}; backend: {res['backend']}")
        for m in res["metrics"]:
            tag = " [projection]" if m["kind"] == C.KIND_LOGO else ""
            s.append(f"  - {m['gpu']} {m['quantity']} ({m['kind']}, calib {m['calib_from']}){tag}: "
                     f"MAPE {fmt(m['mape'])}%, max {fmt(m['max_abs_pct'])}%, r {fmt(m['pearson_r'], '{:.3f}')}")
        for r in res["level1_summary"]:
            if r["kernel"] == "all":
                s.append(f"  - {r['gpu']} L1 GEMV BW: MAPE {fmt(r['mape_plain'])}% plain -> "
                         f"{fmt(r['mape_loaded'])}% with loaded latency")
    s.append(f"- flagged (|err| > {threshold:g}%): {len(flags)}")
    for f in flags:
        s.append(f"  - [{f['level']} {f['gpu']} {f['item']} {f['config']}] err {f['err_pct']:+.1f}% -> {f['hypothesis']}")
    return "\n".join(s) + "\n\n"


def run_pipeline(data_root="data", out_dir="results", run: bool = False, threshold: float = DEFAULT_THRESHOLD,
                 backend: Optional[Backend] = None, fit: bool = True, fit_script: Optional[Path] = None,
                 bench_cmd: Optional[List[str]] = None, now: Optional[str] = None) -> Dict:
    data_root, out_dir = Path(data_root), Path(out_dir)
    now = now or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    steps = [step_bench(run, bench_cmd)]
    backend = backend or default_backend()
    steps.append(step_fit(data_root, fit_script, backend, out_dir) if fit
                 else {"step": "fit", "status": "skipped", "detail": "--no-fit"})
    res, flags = None, []
    try:
        res = C.run(data_root, out_dir, backend=backend)
        steps.append({"step": "correlate", "status": "ok",
                      "detail": f"{len(res['errors'])} error records -> {out_dir}/correlation.md"})
    except Exception as e:
        steps.append({"step": "correlate", "status": "error", "detail": f"{type(e).__name__}: {e}"})
    if res is not None:
        flags = flag(res["errors"], threshold)
        steps.append({"step": "flag", "status": "ok", "detail": f"{len(flags)} over {threshold:g}%"})
        C.write_csv(out_dir / "flags.csv", flags)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "AGENT_LOG.md"
    if not log.exists():
        log.write_text("# Agent log\n\nAppend-only record of agents/pipeline.py runs "
                       "(bench -> fit -> correlate -> flag).\n\n")
    with open(log, "a") as f:
        f.write(log_entry(steps, res, flags, threshold, now))
    return {"timestamp": now, "steps": steps, "flags": flags,
            "metrics": res["metrics"] if res else [], "log": str(log)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="bench -> fit -> correlate -> flag -> AGENT_LOG.md")
    ap.add_argument("--run", action="store_true", help="run gpu/run_all.sh first (GPU box only)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="flag |error| above this %%")
    ap.add_argument("--no-fit", action="store_true", help="skip the calibration fit step")
    ap.add_argument("--json", action="store_true", help="print a JSON summary (for a calling agent)")
    args = ap.parse_args(argv)
    summary = run_pipeline(args.data, args.out, run=args.run, threshold=args.threshold, fit=not args.no_fit)
    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    else:
        for st in summary["steps"]:
            print(f"{st['step']:>10}: {st['status']}  {st.get('detail', '')}")
        for f in summary["flags"]:
            print(f"FLAG {f['gpu']} {f['item']} {f['config']}: {f['err_pct']:+.1f}% -> {f['hypothesis']}")
        print(f"log: {summary['log']}")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
