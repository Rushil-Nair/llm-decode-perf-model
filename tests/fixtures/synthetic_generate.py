"""Generate SYNTHETIC test fixtures for perfsim.microarch (NOT measurements; no GPU was involved).

Records have the same fields as the real probes.jsonl / gemv_sweep.jsonl, with "gpu": "SYNTHETIC" and
"synthetic": true, and are produced from known parameters so the tests can check that the model
recovers them:

    TRUE: 132 SMs, L2 50 MB, unloaded DRAM latency 650 ns, peak 3000 GB/s, alpha 0.4

Regenerate with:  python3 tests/fixtures/synthetic_generate.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))     # repo root, for `import perfsim`

from perfsim.microarch import GEMV_BYTES_PER_GROUP, GPUParams, bytes_in_flight, predict_bw  # noqa: E402

TRUE = dict(sm_count=132, l2_bytes=50_000_000, dram_latency_ns=650.0, peak_bw_Bps=3000e9, alpha=0.4)
L1_NS, L2_NS, L1_BYTES = 35.0, 280.0, 192 * 1024


def true_params() -> GPUParams:
    return GPUParams(sm_count=TRUE["sm_count"], dram_latency_ns=TRUE["dram_latency_ns"],
                     peak_bw_Bps=TRUE["peak_bw_Bps"], l2_bytes=TRUE["l2_bytes"], name="SYNTHETIC")


def _noise(rng, n, rel):
    return np.ones(n) if rel == 0 else 1.0 + rel * rng.standard_normal(n)


def pointer_chase_records():
    """L1 / L2 / DRAM plateaus; beyond L2 a random chase still hits L2 with probability L2/footprint."""
    out = []
    f = 16 * 1024.0
    while f <= 1024 ** 3 * 1.0001:
        fp = int(f) // 128 * 128
        if fp <= L1_BYTES:
            ns = L1_NS
        elif fp <= TRUE["l2_bytes"]:
            ns = L2_NS
        else:
            h = TRUE["l2_bytes"] / fp
            ns = h * L2_NS + (1 - h) * TRUE["dram_latency_ns"]
        out.append({"probe": "pointer_chase", "footprint_bytes": fp, "stride_bytes": 128,
                    "latency_cycles": round(ns * 1.755, 2), "latency_ns": round(ns, 2),
                    "gpu": "SYNTHETIC", "synthetic": True})
        f *= math.sqrt(2.0)
    return out


def stream_records(noise=0.03, seed=0):
    """Stream sweep generated from the loaded-latency model with the TRUE parameters."""
    p, rng = true_params(), np.random.default_rng(seed)
    cfgs = []
    for loads in (1, 2, 4, 8):
        for tpb in (64, 128, 256, 512, 1024):
            for bps in range(1, 2048 // tpb + 1):
                cfgs.append((bps, tpb, loads))
    bif = np.array([bps * tpb * loads * 16 * p.sm_count for bps, tpb, loads in cfgs], dtype=float)
    bw = predict_bw(p, bif, loaded_latency=True, alpha=TRUE["alpha"]) * _noise(rng, len(cfgs), noise)
    out = [{"probe": "stream", "blocks_per_sm": bps, "threads_per_block": tpb,
            "bytes_per_thread_in_flight": loads * 16, "bytes_in_flight_total": int(b),
            "GBps": round(float(g) / 1e9, 2), "gpu": "SYNTHETIC", "sm_count": p.sm_count,
            "l2_bytes": p.l2_bytes, "loads_per_thread": loads, "synthetic": True}
           for (bps, tpb, loads), b, g in zip(cfgs, bif, bw)]
    best = max(out, key=lambda r: r["GBps"])
    out.append({**best, "probe": "stream_peak"})
    return out


def gemv_records(noise=0.03, seed=1):
    """GEMV sweep records (v2/v3/v4 x ILP 1/2/4 x blocks/SM 1..6, block 256) from the same model."""
    p, rng = true_params(), np.random.default_rng(seed)
    shapes = [("q_proj", 4096, 4096), ("gate_proj", 12288, 4096), ("down_proj", 4096, 12288)]
    rows = []
    for name, N, K in shapes:
        qbytes = N * K // 2 + N * (K // 32) * 2 + K * 2 + N * 4
        for kernel in ("v2", "v3", "v4"):
            for ilp in (1, 2, 4):
                for bps in range(1, 7):
                    grid_rows = (N + 7) // 8
                    grid = grid_rows if kernel == "v2" else min(grid_rows, bps * p.sm_count)
                    warps_eff = min(grid, bps * p.sm_count) * 8 / p.sm_count
                    rows.append(dict(kernel=kernel, ilp=ilp, blocks_per_sm=bps, warps_per_sm=bps * 8, N=N, K=K,
                                     bytes=qbytes, shape=name, requested_blocks_per_sm=bps, threads_per_block=256,
                                     grid=grid, warps_per_sm_eff=round(warps_eff, 3), sm_count=p.sm_count,
                                     l2_bytes=p.l2_bytes, gpu="SYNTHETIC", mode="timed", ncu_launch=-1,
                                     ok=True, synthetic=True))
    bif = np.array([bytes_in_flight(r["warps_per_sm_eff"], r["ilp"], GEMV_BYTES_PER_GROUP, p.sm_count)
                    for r in rows])
    bw = predict_bw(p, bif, loaded_latency=True, alpha=TRUE["alpha"]) * _noise(rng, len(rows), noise)
    for r, b in zip(rows, bw):
        r["GBps"] = round(float(b) / 1e9, 1)
        r["us"] = round(r["bytes"] / float(b) * 1e6, 3)
    return rows


def write_all(outdir: Path = HERE) -> None:
    probes = pointer_chase_records() + stream_records()
    (outdir / "synthetic_probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probes))
    (outdir / "synthetic_gemv_sweep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gemv_records()))


if __name__ == "__main__":
    write_all()
    print("wrote", HERE / "synthetic_probes.jsonl", HERE / "synthetic_gemv_sweep.jsonl")
