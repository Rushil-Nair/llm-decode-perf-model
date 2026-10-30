"""Repeatability and cuBLAS baseline for one GPU run (output of bench/run_extra.sh).

    python -m perfsim.extra_report data_kernels/<tag> results/extra_<tag>.md

- Repeatability: for each kernel and shape, the best config's GB/s in each of the 3 sweep repeats
  (gemv_sweep.jsonl, gemv_sweep_rep2.jsonl, gemv_sweep_rep3.jsonl): mean, min, max, spread.
- Baseline: cuBLAS fp16 dense GEMV (kernels/q4gemv.cu, cublasGemmEx with n = 1) on the same shapes,
  as % of the stream peak and as time per call, next to the best 4-bit kernel. cuBLAS reads 2 bytes per
  weight, the 4-bit kernels 0.5625, so time is the user-facing comparison and % of peak the efficiency one.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from perfsim import microarch as M


def _load(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.startswith("{")]


def main(argv) -> None:
    run = Path(argv[1]); out = Path(argv[2]) if len(argv) > 2 else None
    params = M.params_from_probes(M.load_probes(run))
    peak = params.peak_bw_Bps / 1e9
    reps = [r for r in [_load(run / "gemv_sweep.jsonl"), _load(run / "gemv_sweep_rep2.jsonl"),
                        _load(run / "gemv_sweep_rep3.jsonl")] if r]
    reps = [[r for r in rep if r.get("ok") and r.get("GBps", 0) > 0 and r.get("mode") != "ncu"] for rep in reps]
    L = [f"# Repeatability and cuBLAS baseline: {run.name}", "", f"Stream peak {peak:.0f} GB/s; {len(reps)} sweep repeats.", ""]
    shapes = ["q_proj", "gate_proj", "down_proj"]
    kernels = sorted({r["kernel"] for rep in reps for r in rep})
    L += ["## Best config per kernel, across repeats (GB/s)", "",
          "| shape | kernel | mean | min | max | spread % |", "|---|---|---|---|---|---|"]
    best = {}
    for sh in shapes:
        for k in kernels:
            vals = []
            for rep in reps:
                xs = [r["GBps"] for r in rep if r["kernel"] == k and r["shape"] == sh]
                if xs:
                    vals.append(max(xs))
            if not vals:
                continue
            m = statistics.mean(vals)
            best[(sh, k)] = m
            L.append(f"| {sh} | {k} | {m:.0f} | {min(vals):.0f} | {max(vals):.0f} | {100 * (max(vals) - min(vals)) / m:.1f} |")
    q4 = [r for i in (1, 2, 3) for r in _load(run / f"q4gemv_rep{i}.jsonl")]
    cub = [r for r in q4 if r.get("kernel") == "cublas_fp16_dense" and r.get("ok") and r.get("GBps", 0) > 0]
    if cub:
        L += ["", "## cuBLAS fp16 dense GEMV vs the best 4-bit kernel", "",
              "| shape | cuBLAS fp16 GB/s | % of peak | cuBLAS us | best 4-bit kernel | GB/s | % of peak | us | time vs cuBLAS |",
              "|---|---|---|---|---|---|---|---|---|"]
        for sh in shapes:
            c = [r for r in cub if r["shape"] == sh]
            if not c:
                continue
            cg = statistics.mean(r["GBps"] for r in c); cu = statistics.mean(r["us"] for r in c)
            cand = [(v, k) for (s, k), v in best.items() if s == sh]
            if not cand:
                continue
            bg, bk = max(cand)
            ex = next(r for rep in reps for r in rep if r["shape"] == sh)
            bus = ex["bytes"] / (bg * 1e9) * 1e6
            L.append(f"| {sh} | {cg:.0f} | {100 * cg / peak:.0f}% | {cu:.1f} | {bk} | {bg:.0f} | {100 * bg / peak:.0f}% | "
                     f"{bus:.1f} | {cu / bus:.2f}x faster |")
    text = "\n".join(L) + "\n"
    if out:
        out.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
