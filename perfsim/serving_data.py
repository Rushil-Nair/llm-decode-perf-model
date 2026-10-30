"""Loaders for the Level-3 serving measurements in data/<gpu_tag>/.

vLLM (`vllm_latency.jsonl`, one record per `vllm bench latency` run, see CONTRACTS.md):
    load_vllm_latency(path)          -> raw records
    tpot_records(records)            -> decode time per output token per batch:
                                        TPOT = (lat(out=OL) - lat(out=1)) / (OL - 1)
    ttft_records(records)            -> TTFT(input_len, batch) ~= lat(out=1)

    Why the difference: lat(out=1) = prefill + fixed costs; lat(out=OL) adds OL-1 decode
    steps. Subtracting cancels prefill and per-run fixed costs (engine call, tokenization),
    leaving the mean decode step time at context input_len .. input_len+OL-2.

llama.cpp (`llamacpp/`, the kit's results_h100 layout copied as-is):
    load_llamacpp_slots(dir)         -> (users, build, tok_s, passes_per_step) per slot run
    load_llamacpp_batched(dir)       -> llama-batched-bench decode/prefill speed per batch
    load_llama_bench(dir)            -> llama-bench pp/tg rows

All loaders skip malformed lines/files instead of raising, and never invent values:
a quantity that is missing is None.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------------------
# generic
# ---------------------------------------------------------------------------------------


def read_jsonl(path: Path | str) -> List[Dict]:
    """Records of a JSON-lines file; blank and unparsable lines are skipped."""
    out: List[Dict] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------------------
# vLLM bench latency
# ---------------------------------------------------------------------------------------


def load_vllm_latency(path: Path | str) -> List[Dict]:
    """vllm_latency.jsonl -> records with at least batch, input_len, output_len and a latency.
    When the same (batch, input_len, output_len) appears more than once, the LAST one wins
    (a re-run after a fix replaces the earlier record)."""
    keep: Dict[tuple, Dict] = {}
    for r in read_jsonl(path):
        try:
            key = (int(r["batch"]), int(r["input_len"]), int(r["output_len"]))
        except (KeyError, TypeError, ValueError):
            continue
        if _latency(r, "avg") is None and _latency(r, "p50") is None:
            continue
        keep[key] = r
    return [keep[k] for k in sorted(keep)]


def _latency(r: Dict, stat: str) -> Optional[float]:
    v = r.get(f"{stat}_latency_s")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def tpot_records(records: Iterable[Dict], output_len: Optional[int] = None,
                 stat: str = "avg") -> List[Dict]:
    """Per (batch, input_len): TPOT = (lat(out=OL) - lat(out=1)) / (OL - 1).

    output_len: which OL to use; default = the largest OL > 1 measured for that group.
    stat: "avg" (avg_latency_s) or "p50" (p50_latency_s).
    """
    groups: Dict[tuple, Dict[int, float]] = {}
    for r in records:
        lat = _latency(r, stat)
        if lat is None:
            continue
        groups.setdefault((int(r["batch"]), int(r["input_len"])), {})[int(r["output_len"])] = lat
    out = []
    for (b, s), by_out in sorted(groups.items()):
        if 1 not in by_out:
            continue
        longs = [o for o in by_out if o > 1 and (output_len is None or o == output_len)]
        if not longs:
            continue
        OL = max(longs)
        tpot = (by_out[OL] - by_out[1]) / (OL - 1)
        out.append({"batch": b, "input_len": s, "output_len": OL, "stat": stat,
                    "lat_out1_s": by_out[1], "lat_outOL_s": by_out[OL], "tpot_s": tpot,
                    # aggregate decode throughput of the batch (tokens/s); None if TPOT <= 0 (noise)
                    "decode_tok_s": b / tpot if tpot > 0 else None})
    return out


def ttft_records(records: Iterable[Dict], stat: str = "avg") -> List[Dict]:
    """TTFT(input_len, batch) ~= lat(out=1): prefill + one sample, plus fixed per-call costs."""
    out = []
    for r in records:
        if int(r.get("output_len", -1)) != 1:
            continue
        lat = _latency(r, stat)
        if lat is None:
            continue
        out.append({"batch": int(r["batch"]), "input_len": int(r["input_len"]), "stat": stat,
                    "ttft_s": lat, "prefill_tok_s": int(r["batch"]) * int(r["input_len"]) / lat})
    return sorted(out, key=lambda x: (x["batch"], x["input_len"]))


# ---------------------------------------------------------------------------------------
# llama.cpp kit results
# ---------------------------------------------------------------------------------------

# slot/<policy>_u<users>.json, policy = master_plain | fix_plain | master_kvu
SLOT_FILE = re.compile(r"^(?P<build>master|fix)_(?P<variant>[a-z]+)_u(?P<users>\d+)\.json$")
DIAG = re.compile(r"passes_per_step=([\d.]+)")
DIAG_CALLS = re.compile(r"decode_calls=(\d+)")


def parse_diag(text: str) -> Dict[str, Optional[float]]:
    """Last DIAG line of the patched server: cumulative forward passes per generation step."""
    k = calls = None
    for line in text.splitlines():
        m = DIAG.search(line)
        if m:
            k = float(m.group(1))
            c = DIAG_CALLS.search(line)
            calls = int(c.group(1)) if c else None
    return {"passes_per_step": k, "decode_calls": calls}


def load_llamacpp_slots(folder: Path | str) -> List[Dict]:
    """One record per slot run: users, build (master|fix), kv_unified, tok_s (median over
    the loadgen passes of wall-clock aggregate tokens/s), passes_per_step (DIAG counter;
    None if the server printed no DIAG line), latency p50, failed requests."""
    slot = Path(folder) / "slot"
    out = []
    if not slot.is_dir():
        return out
    for f in sorted(slot.glob("*.json")):
        m = SLOT_FILE.match(f.name)
        if not m:
            continue  # e.g. *.slots.json snapshots
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            passes = rec.get("passes") or []
        except (json.JSONDecodeError, OSError, AttributeError):
            continue
        tps = [p.get("throughput_tok_s") for p in passes if p.get("throughput_tok_s")]
        p50 = [p.get("latency_p50_s") for p in passes if p.get("latency_p50_s") is not None]
        diag_f = f.with_name(f.stem + ".diag.txt")
        diag = parse_diag(diag_f.read_text(errors="replace")) if diag_f.exists() else \
            {"passes_per_step": None, "decode_calls": None}
        out.append({
            "users": int(m["users"]),
            "build": m["build"],
            "kv_unified": m["variant"] == "kvu",
            "policy": f.stem.rsplit("_u", 1)[0],
            "tok_s": statistics.median(tps) if tps else None,
            "passes_per_step": diag["passes_per_step"],
            "decode_calls": diag["decode_calls"],
            "latency_p50_s": statistics.median(p50) if p50 else None,
            "failed": sum(int(p.get("failed", 0)) for p in passes),
            "requests": sum(int(p.get("requests", 0)) for p in passes),
        })
    return sorted(out, key=lambda r: (r["build"], r["kv_unified"], r["users"]))


def load_llamacpp_batched(folder: Path | str) -> List[Dict]:
    """llama-batched-bench jsonl (build-fix, one KV stream per sequence, sequences 0..B-1
    contiguous, so one forward pass per decode step is expected but NOT measured)."""
    out = []
    for r in read_jsonl(Path(folder) / "bench" / "batched_bench.jsonl"):
        try:
            B = int(r["pl"])
        except (KeyError, TypeError, ValueError):
            continue
        tg = r.get("speed_tg")
        out.append({"users": B, "build": "fix", "source": "batched_bench",
                    "pp": r.get("pp"), "tg": r.get("tg"),
                    "tok_s": float(tg) if tg is not None else None,
                    "prefill_tok_s": float(r["speed_pp"]) if r.get("speed_pp") is not None else None,
                    # speed_tg = B*tg tokens / t_tg, one step writes B tokens -> step = B / speed_tg
                    "step_s": B / float(tg) if tg else None,
                    "passes_per_step": None})
    return sorted(out, key=lambda r: r["users"])


def load_llama_bench(folder: Path | str) -> List[Dict]:
    """llama-bench -o json rows: test name (pp512 / tg128 / pp512+tg128) and avg tokens/s."""
    f = Path(folder) / "bench" / "llama_bench.json"
    try:
        rows = json.loads(f.read_text(encoding="utf-8") or "[]")
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        p, g = int(r.get("n_prompt", 0) or 0), int(r.get("n_gen", 0) or 0)
        name = f"pp{p}" if g == 0 else (f"tg{g}" if p == 0 else f"pp{p}+tg{g}")
        out.append({"test": name, "n_prompt": p, "n_gen": g,
                    "tok_s": r.get("avg_ts"), "stddev_tok_s": r.get("stddev_ts")})
    return out
