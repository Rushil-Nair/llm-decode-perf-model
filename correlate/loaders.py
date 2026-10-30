"""Readers for data/<gpu_tag>/ (layout in CONTRACTS.md).

Every reader is tolerant: a missing file returns an empty result and a note, a bad
JSON line is skipped with a note. Nothing here ever writes into data/.

The vLLM TPOT/TTFT derivation follows the contract formula exactly:
    TPOT(batch, input_len) = (lat(out=OL) - lat(out=1)) / (OL - 1)
    TTFT(batch, input_len) ~= lat(out=1)
(perfsim/serving_data.py implements the same thing for the engine side; this copy
keeps correlate runnable when that module is absent.)
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Files whose presence marks a directory as a gpu_tag results dir.
KNOWN_FILES = ("env.json", "probes.jsonl", "gemv_sweep.jsonl", "ops.jsonl",
               "vllm_latency.jsonl", "calib.json", "llamacpp")

# Bytes one GEMV lane has in flight per group load: 16 B packed int4 (one uint4 load) + 2 B fp16
# scale (kernels/q4gemv.cu; same constant as perfsim.microarch.GEMV_BYTES_PER_GROUP).
GEMV_BYTES_PER_LOAD = 18


def read_jsonl(path: Path, notes: Optional[List[str]] = None) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        if notes is not None:
            notes.append(f"missing {path} (skipped)")
        return []
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            if notes is not None:
                notes.append(f"{path}:{i}: bad JSON line skipped ({e.msg})")
    return out


def read_json(path: Path, notes: Optional[List[str]] = None, default=None):
    path = Path(path)
    if not path.exists():
        if notes is not None:
            notes.append(f"missing {path} (skipped)")
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        if notes is not None:
            notes.append(f"{path}: unreadable ({e})")
        return default


def gpu_dirs(data_root: Path) -> List[Path]:
    """Sub-directories of data/ that look like gpu_tag result dirs, sorted by name."""
    root = Path(data_root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")):
            if any((d / f).exists() for f in KNOWN_FILES):
                out.append(d)
    return out


# ---------------------------------------------------------------------------------
# Level 1 inputs: env.json + probes.jsonl -> GPUParams fields
# ---------------------------------------------------------------------------------
def measured_gpu_fields(tag_dir: Path, notes: Optional[List[str]] = None) -> Dict:
    """Fields of microarch.GPUParams derived from this GPU's own measurements.

    sm_count, l2_bytes  from env.json
    dram_latency_ns     pointer-chase latency at the largest footprint measured
                        (median over strides >= 128 B if present: smaller strides
                        share sectors/lines and read low)
    peak_bw_Bps         max GB/s of the stream probe ("measured stream max", per contract)
    Missing pieces are None; callers decide whether Level 1 can run.
    """
    tag_dir = Path(tag_dir)
    env = read_json(tag_dir / "env.json", notes, default={}) or {}
    probes = read_jsonl(tag_dir / "probes.jsonl", notes)
    chase = [p for p in probes if p.get("probe") == "pointer_chase" and p.get("latency_ns")]
    stream = [p for p in probes if p.get("probe") == "stream" and p.get("GBps")]
    lat = None
    if chase:
        big = max(p["footprint_bytes"] for p in chase)
        at_big = [p for p in chase if p["footprint_bytes"] == big]
        wide = [p for p in at_big if p.get("stride_bytes", 0) >= 128] or at_big
        lat = float(statistics.median(p["latency_ns"] for p in wide))
    peak = float(max(p["GBps"] for p in stream)) * 1e9 if stream else None
    return {
        "sm_count": env.get("sm_count"),
        "dram_latency_ns": lat,
        "peak_bw_Bps": peak,
        "l2_bytes": env.get("l2_bytes"),
    }


def load_gemv_sweep(tag_dir: Path, notes: Optional[List[str]] = None) -> List[Dict]:
    """ok==True records with a positive measured bandwidth."""
    recs = read_jsonl(Path(tag_dir) / "gemv_sweep.jsonl", notes)
    # ok, timed, positive bandwidth; ncu-mode records are profiler replays (not timings)
    good = [r for r in recs if r.get("ok", True) and (r.get("GBps") or 0) > 0 and r.get("mode", "timed") != "ncu"]
    if notes is not None and len(good) < len(recs):
        notes.append(f"{tag_dir}/gemv_sweep.jsonl: {len(recs) - len(good)} records dropped (ok=false or GBps<=0)")
    return good


def load_ops(tag_dir: Path, notes: Optional[List[str]] = None) -> List[Dict]:
    recs = read_jsonl(Path(tag_dir) / "ops.jsonl", notes)
    return [r for r in recs if (r.get("us") or 0) > 0]


# ---------------------------------------------------------------------------------
# Level 3 inputs: vLLM bench latency -> TPOT / TTFT
# ---------------------------------------------------------------------------------
def derive_vllm(records: List[Dict], latency_key: str = "avg_latency_s") -> Tuple[List[Dict], List[Dict]]:
    """(tpot_rows, ttft_rows) from vllm_latency.jsonl records (contract formula).

    Repeated records with the same (batch, input_len, output_len) are averaged.
    tpot row: {batch, input_len, output_len, tpot_s}; ttft row: {batch, input_len, ttft_s}.
    A (batch, input_len) without an out=1 record yields neither.
    """
    lat: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    for r in records:
        try:
            key = (int(r["batch"]), int(r["input_len"]), int(r["output_len"]))
            v = float(r[latency_key])
        except (KeyError, TypeError, ValueError):
            continue
        if v > 0:
            lat[key].append(v)
    mean = {k: sum(v) / len(v) for k, v in lat.items()}
    tpot, ttft = [], []
    for (b, il, ol), v in sorted(mean.items()):
        if ol == 1:
            ttft.append({"batch": b, "input_len": il, "ttft_s": v})
            continue
        base = mean.get((b, il, 1))
        if base is None:
            continue
        tpot.append({"batch": b, "input_len": il, "output_len": ol, "tpot_s": (v - base) / (ol - 1)})
    return tpot, ttft


def load_level3(tag_dir: Path, notes: Optional[List[str]] = None):
    """(tpot_rows, ttft_rows, slot_rows) for one GPU. Uses perfsim.serving_data's derivation
    when that module is importable (one implementation for the whole repo), else the local
    copies above. Lines are parsed here either way so malformed ones are reported."""
    tag_dir = Path(tag_dir)
    recs = read_jsonl(tag_dir / "vllm_latency.jsonl", notes)
    try:
        from perfsim import serving_data as sd
    except Exception:
        sd = None
    if sd is not None:
        tpot = [{k: r[k] for k in ("batch", "input_len", "output_len", "tpot_s")} for r in sd.tpot_records(recs)]
        ttft = [{k: r[k] for k in ("batch", "input_len", "ttft_s")} for r in sd.ttft_records(recs)]
        slots = []
        if (tag_dir / "llamacpp" / "slot").is_dir():
            for r in sd.load_llamacpp_slots(tag_dir / "llamacpp"):
                if r.get("tok_s") is None:
                    continue
                k = r.get("passes_per_step")
                slots.append({"policy": r.get("policy"), "build": r["build"], "users": r["users"],
                              "tok_s": float(r["tok_s"]), "passes_per_step": k,
                              "k_source": "DIAG counter" if k is not None else "missing",
                              "source": str(tag_dir / "llamacpp" / "slot" / f"{r.get('policy')}_u{r['users']}.json")})
        elif notes is not None:
            notes.append(f"missing {tag_dir / 'llamacpp' / 'slot'} (llama.cpp slot test skipped)")
        return tpot, ttft, slots
    tpot, ttft = derive_vllm(recs)
    return tpot, ttft, load_slots(tag_dir, notes)


# ---------------------------------------------------------------------------------
# Level 3 inputs: llama.cpp slot test (h100kit layout: llamacpp/slot/<policy>_u<U>.json)
# ---------------------------------------------------------------------------------
_SLOT_RE = re.compile(r"^(?P<policy>.+)_u(?P<users>\d+)\.json$")
_DIAG_RE = re.compile(r"passes_per_step=([\d.]+)")


def load_slots(tag_dir: Path, notes: Optional[List[str]] = None) -> List[Dict]:
    """Rows {policy, build, users, tok_s, passes_per_step, k_source, source}.

    tok_s = median over passes of passes[i].throughput_tok_s (loadgen.py record);
    passes_per_step from the DIAG counter line in <policy>_u<U>.diag.txt.
    """
    sd = Path(tag_dir) / "llamacpp" / "slot"
    if not sd.is_dir():
        if notes is not None:
            notes.append(f"missing {sd} (llama.cpp slot test skipped)")
        return []
    rows = []
    for f in sorted(sd.glob("*.json")):
        m = _SLOT_RE.match(f.name)
        if not m:
            continue
        rec = read_json(f, notes, default=None)
        if not isinstance(rec, dict):
            continue
        thr = [p.get("throughput_tok_s") for p in rec.get("passes", []) if p.get("throughput_tok_s")]
        if not thr:
            if notes is not None:
                notes.append(f"{f}: no throughput in passes (skipped)")
            continue
        policy, users = m.group("policy"), int(m.group("users"))
        diag = sd / f"{policy}_u{users}.diag.txt"
        k = None
        if diag.exists():
            mm = _DIAG_RE.search(diag.read_text())
            k = float(mm.group(1)) if mm else None
        rows.append({
            "policy": policy, "build": policy.split("_")[0], "users": users,
            "tok_s": float(statistics.median(thr)), "passes_per_step": k,
            "k_source": "DIAG counter" if k is not None else "missing",
            "source": str(f),
        })
    return rows


def load_rtx5090_october_slots(yaml_path: Path) -> List[Dict]:
    """The owner's Oct-8 RTX 5090 slot measurements (data/llamacpp_rtx5090_qwen3_8b.yaml),
    in the load_slots row format. Used for gpu_tag rtx5090 when no slot dir exists.

    passes_per_step: 8 users from the CPU counter (4.34 before, 1.05 after);
    1 user = 1 by construction; 16 users = 1 (all 16 slots busy -> contiguous ids)
    and fix 4 users = 1 are the assumptions already stated in perfsim/llamacpp_slots.py;
    master 4 users has no counter -> no prediction.
    """
    p = Path(yaml_path)
    if not p.exists():
        return []
    with open(p) as f:
        d = yaml.safe_load(f)
    thr = d["throughput_tok_s"]
    cpu = d["counters"]["cpu_check_master"]
    k = {
        ("master", 1): (1.0, "1 sequence"), ("fix", 1): (1.0, "1 sequence"),
        ("master", 8): (cpu["forward_passes_per_generation_step_before"], "CPU counter"),
        ("fix", 8): (cpu["forward_passes_per_generation_step_after"], "CPU counter"),
        ("master", 16): (1.0, "assumed (all slots busy)"), ("fix", 16): (1.0, "assumed (all slots busy)"),
        ("fix", 4): (1.0, "assumed (patched ordering)"), ("master", 4): (None, "missing"),
    }
    rows = []
    for users, (before, after) in sorted(thr.items()):
        for build, v in (("master", before), ("fix", after)):
            kk, src = k.get((build, int(users)), (None, "missing"))
            rows.append({"policy": f"{build}_plain", "build": build, "users": int(users), "tok_s": float(v),
                         "passes_per_step": kk, "k_source": src, "source": str(p)})
    return rows
