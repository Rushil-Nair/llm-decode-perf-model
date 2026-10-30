"""Level 1, refined: a three-ceiling model of the 4-bit decode GEMV kernels.

Plain Little's Law (perfsim/microarch.py) says achieved bandwidth = min(peak, bytes_in_flight / latency).
On the H100 that over-predicts our GEMV kernels by 3-7x: they plateau at ~1.5 TB/s (48% of the measured
3.1 TB/s stream peak) no matter how many bytes are in flight. On the RTX 5090 the same kernels reach ~90%.
The kernels are not waiting on memory on the H100; they are limited by how fast the SMs can dequantize.

Machine balance explains why. Bytes the memory system delivers per SM per cycle:
    H100 SXM : 3.14e12 B/s / (132 SMs * 1.98e9 Hz) ~= 12.0 B/SM/cycle
    RTX 5090 : 1.70e12 B/s / (170 SMs * 2.89e9 Hz) ~=  3.5 B/SM/cycle
Each 18-byte group (16 B of int4 weights + 2 B scale) costs a fixed number of SM cycles to unpack, convert
and multiply. At ~3 cycles/group an SM consumes ~6 B/cycle: below the H100's 12 (SM-bound), above the
5090's 3.5 (memory-bound).

Model (per kernel version and ILP), bandwidth = min of three ceilings, plus a fixed launch cost:

    BW_mem  = loaded-latency Little's Law (microarch.predict_bw with alpha from the stream probe)
    BW_sm   = SMs * f_sm * bytes_per_group / cpg          cpg = SM-cycles per group (SM throughput limit)
    BW_warp = warps_per_SM * SMs * f_sm * bytes_per_group / cpw
                                                          cpw = cycles per group per warp (too few warps
                                                          to hide the ALU dependency chain)
    t = t0 + bytes / min(BW_mem, BW_sm, BW_warp)          t0 = per-launch fixed cost

cpg and cpw are properties of the compiled code (instruction mix and dependency chains); f_sm is the
measured busy SM clock (nvidia-smi 1 Hz log). cpg, cpw and t0 are fitted on one GPU's sweep and then used
unchanged on the other GPU (leave-one-GPU-out), together with that GPU's own measured latency, peak
bandwidth, SM count and clock.

    python -m perfsim.gemv_model --data data --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from perfsim import microarch as M

SMALL = 4096 * 4096          # shapes this small are dominated by launch/tail effects: used to fit t0 only
Key = Tuple[str, int]        # (kernel version, ILP)


def busy_sm_clock_hz(data_dir) -> float:
    """Median SM clock while the GPU was busy (utilization > 50%) from data/<tag>/clocks.csv."""
    vals = []
    with open(Path(data_dir) / "clocks.csv") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        try:
            sm = float(r[1].split()[0]); util = float(r[5].split()[0])
        except (IndexError, ValueError):
            continue
        if util > 50:
            vals.append(sm)
    if not vals:
        raise ValueError(f"no busy samples in {data_dir}/clocks.csv")
    return statistics.median(vals) * 1e6


class GPU:
    """Everything measured about one GPU that the model needs."""

    def __init__(self, data_dir):
        self.tag = Path(data_dir).name
        probes = M.load_probes(data_dir)
        self.params = M.params_from_probes(probes)
        stream = [r for r in probes if r.get("probe") == "stream"]
        self.alpha = M.fit_loaded_latency(self.params, stream)
        self.f_sm = busy_sm_clock_hz(data_dir)
        self.records = [r for r in M.usable_records(M.load_gemv_sweep(data_dir)) if r.get("mode") != "ncu"]

    @property
    def bytes_per_sm_cycle(self) -> float:
        return self.params.peak_bw_Bps / (self.params.sm_count * self.f_sm)


def _bytes_per_group(r: dict) -> float:
    return r["bytes"] / (r["N"] * r["K"] / 32)


def _warps(r: dict) -> float:
    return r.get("warps_per_sm_eff", r["warps_per_sm"])


def ceilings(gpu: GPU, r: dict, cpg: Dict[Key, float], cpw: Dict[Key, float]) -> Dict[str, float]:
    k = (r["kernel"], r["ilp"])
    p, f, bpg = gpu.params, gpu.f_sm, _bytes_per_group(r)
    return {
        "mem": float(M.predict_bw(p, M.record_bytes_in_flight(r, p), loaded_latency=True, alpha=gpu.alpha)),
        "sm": p.sm_count * f * bpg / cpg[k],
        "warp": _warps(r) * p.sm_count * f * bpg / cpw[k],
    }


def fit(gpu: GPU, iters: int = 3):
    """cpg, cpw per (kernel, ILP) from the large shapes; t0 from the small shape's residual.

    Cycle costs are computed from the kernel time minus t0 and t0 from the residual given the cycle costs,
    alternating a few times (starting from t0 = 0) so the launch cost is not counted twice.
    """
    t0 = 0.0
    for _ in range(iters):
        cpg: Dict[Key, float] = {}
        cpw: Dict[Key, float] = {}
        for r in gpu.records:
            if r["N"] * r["K"] <= SMALL:
                continue
            k = (r["kernel"], r["ilp"])
            bw = r["bytes"] / max(r["us"] * 1e-6 - t0, 1e-9)                       # streaming BW without t0
            cyc = gpu.params.sm_count * gpu.f_sm * _bytes_per_group(r) / bw          # SM-cycles per group
            cpg[k] = min(cpg.get(k, math.inf), cyc)              # best config = the SM throughput limit
            cpw[k] = min(cpw.get(k, math.inf), _warps(r) * cyc)  # per-warp limit seen at low occupancy
        resid = []
        for r in gpu.records:
            if r["N"] * r["K"] <= SMALL and (r["kernel"], r["ilp"]) in cpg:
                bw = min(ceilings(gpu, r, cpg, cpw).values())
                resid.append(r["us"] * 1e-6 - r["bytes"] / bw)
        t0 = max(0.0, float(np.median(resid))) if resid else 0.0
    return cpg, cpw, t0


def predict(gpu: GPU, model) -> List[dict]:
    cpg, cpw, t0 = model
    out = []
    for r in gpu.records:
        if (r["kernel"], r["ilp"]) not in cpg:
            continue
        c = ceilings(gpu, r, cpg, cpw)
        bound = min(c, key=c.get)
        t = t0 + r["bytes"] / c[bound]
        pred = r["bytes"] / t / 1e9
        out.append({"shape": r.get("shape"), "N": r["N"], "K": r["K"], "kernel": r["kernel"], "ilp": r["ilp"],
                    "blocks_per_sm": r["blocks_per_sm"], "warps_per_sm": _warps(r),
                    "measured_GBps": r["GBps"], "predicted_GBps": pred,
                    "err_pct": (pred / r["GBps"] - 1) * 100, "bound": bound})
    return out


def summary(rows: List[dict]) -> dict:
    e = np.abs([r["err_pct"] for r in rows])
    meas = np.array([r["measured_GBps"] for r in rows]); pred = np.array([r["predicted_GBps"] for r in rows])
    return {"n": len(rows), "mape": float(e.mean()), "max": float(e.max()),
            "r": float(np.corrcoef(meas, pred)[0, 1]) if len(rows) > 2 else math.nan,
            "bound": {b: sum(1 for r in rows if r["bound"] == b) for b in ("mem", "sm", "warp")}}


WHATIF = [("measured GPU", 1, 1, 1, 1), ("HBM bandwidth x2", 2, 1, 1, 1), ("SM count x1.25", 1, 1.25, 1, 1),
          ("SM clock x1.25", 1, 1, 1.25, 1), ("dequant cost / 2 (cpg, cpw halved)", 1, 1, 1, 0.5)]


def scaled(gpu: GPU, bw=1.0, sms=1.0, clk=1.0) -> GPU:
    import copy
    import dataclasses
    g = copy.copy(gpu)
    g.params = dataclasses.replace(gpu.params, peak_bw_Bps=gpu.params.peak_bw_Bps * bw,
                                   sm_count=int(round(gpu.params.sm_count * sms)))
    g.f_sm = gpu.f_sm * clk
    return g


def whatif(gpu: GPU, model, kernel="v4") -> List[dict]:
    """Projection (not measured): predicted GBps of the best measured config of `kernel` when one hardware
    or code parameter changes and the others are held."""
    cpg, cpw, t0 = model
    best = max((r for r in gpu.records if r["kernel"] == kernel), key=lambda r: r["GBps"])
    out = []
    for label, bw, sms, clk, code in WHATIF:
        g = scaled(gpu, bw, sms, clk)
        m = ({k: v * code for k, v in cpg.items()}, {k: v * code for k, v in cpw.items()}, t0)
        p = predict(g, m)
        row = next(x for x in p if (x["shape"], x["N"], x["K"], x["kernel"], x["ilp"], x["blocks_per_sm"]) ==
                   (best.get("shape"), best["N"], best["K"], best["kernel"], best["ilp"], best["blocks_per_sm"]))
        out.append({"change": label, "predicted_GBps": row["predicted_GBps"], "bound": row["bound"],
                    "pct_of_peak": 100 * row["predicted_GBps"] * 1e9 / g.params.peak_bw_Bps})
    base = out[0]["predicted_GBps"]
    for x in out:
        x["speedup"] = x["predicted_GBps"] / base
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    gpus = {d.name: GPU(d) for d in sorted(Path(a.data).iterdir())
            if (d / "gemv_sweep.jsonl").exists() and (d / "clocks.csv").exists()}
    models = {t: fit(g) for t, g in gpus.items()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = ["# Level 1: three-ceiling GEMV model (memory / SM throughput / per-warp) vs measured", "",
             "| GPU | busy SM clock | stream peak | bytes per SM-cycle | DRAM latency | alpha |",
             "|---|---|---|---|---|---|"]
    for t, g in gpus.items():
        lines.append(f"| {t} | {g.f_sm / 1e6:.0f} MHz | {g.params.peak_bw_Bps / 1e9:.0f} GB/s | "
                     f"{g.bytes_per_sm_cycle:.1f} | {g.params.dram_latency_ns:.0f} ns | {g.alpha:.2f} |")
    lines += ["", "Fitted on each GPU (cpg = SM-cycles per 18-byte group, cpw = per-warp cycles per group):", "",
              "| fitted on | kernel | ILP | cpg | cpw | t0 (us) |", "|---|---|---|---|---|---|"]
    for t, (cpg, cpw, t0) in models.items():
        for k in sorted(cpg):
            lines.append(f"| {t} | {k[0]} | {k[1]} | {cpg[k]:.2f} | {cpw[k]:.1f} | {t0 * 1e6:.2f} |")
    lines += ["", "Predicted vs measured achieved bandwidth (all shapes and configurations):", "",
              "| fitted on | predicted | n | MAPE % | max abs % | Pearson r | bound: mem / sm / warp |",
              "|---|---|---|---|---|---|---|"]
    allrows = []
    for A in gpus:
        for B in gpus:
            rows = predict(gpus[B], models[A])
            for r in rows:
                r.update({"fitted_on": A, "predicted_on": B})
            allrows += rows
            s = summary(rows)
            tag = "" if A == B else " (projection)"
            lines.append(f"| {A} | {B}{tag} | {s['n']} | {s['mape']:.1f} | {s['max']:.1f} | {s['r']:.3f} | "
                         f"{s['bound']['mem']} / {s['bound']['sm']} / {s['bound']['warp']} |")
    lines += ["", "Plain Little's Law (memory ceiling only) on the same data: see correlation.md Level 1.", "",
              "## What-if (projection, not measured): best measured v4 config, one change at a time", "",
              "| GPU | change | predicted GB/s | % of (scaled) peak | bound | speedup |", "|---|---|---|---|---|---|"]
    for t, g in gpus.items():
        for x in whatif(g, models[t]):
            lines.append(f"| {t} | {x['change']} | {x['predicted_GBps']:.0f} | {x['pct_of_peak']:.0f} | "
                         f"{x['bound']} | {x['speedup']:.2f}x |")
    lines.append("")
    (out / "level1_gemv_model.md").write_text("\n".join(lines) + "\n")
    with open(out / "level1_gemv_model.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    json.dump({t: {"cpg": {f"{k[0]}_ilp{k[1]}": v for k, v in m[0].items()},
                   "cpw": {f"{k[0]}_ilp{k[1]}": v for k, v in m[1].items()}, "t0_s": m[2]}
               for t, m in models.items()}, open(out / "level1_gemv_model_params.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
