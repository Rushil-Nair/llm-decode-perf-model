"""Level 1 (microarchitecture): Little's Law latency hiding for memory-bound kernels.

The question this module answers: *given how many bytes a kernel keeps in flight, what DRAM
bandwidth will it achieve?*  It is the model behind the occupancy x ILP sweep of our 4-bit GEMV
(kernels/gemv_sweep.cu) and the bandwidth probe (microarch/probes/stream_bw.cu).

Little's Law (J. D. C. Little, "A Proof for the Queuing Formula L = lambda W", Operations Research
1961): in any stable system, items in the system = arrival rate x time each item spends in it.
For a memory system: bytes_in_flight = bandwidth x latency, so

    achieved_bandwidth = bytes_in_flight / latency                       (1)

V. Volkov, "Understanding Latency Hiding on GPUs" (PhD thesis, UC Berkeley, 2016), shows this is
how GPUs hide memory latency: enough independent loads must be outstanding (from many warps = TLP,
or from several independent loads per warp = ILP) to cover latency x bandwidth. Once (1) exceeds what
DRAM can deliver, bandwidth saturates:

    achieved = min(peak, bytes_in_flight / L0)                            (2)  "unloaded" model

where L0 is the *unloaded* DRAM latency (pointer chase, one load at a time) and peak is the best
bandwidth any configuration of the stream probe reached. (2) is a hard corner; real memory systems
bend smoothly into saturation because latency itself grows as the DRAM gets busy (queueing in the
memory controllers). Volkov's thesis discusses this growth of latency under load; we model it with a
one-parameter, M/M/1-style queueing correction:

    L(u) = L0 * (1 + alpha * u / (1 - u)),   u = achieved / peak          (3)  "loaded" model

Plugging (3) into (1) gives a fixed-point problem u = g(u) with g(u) = D / (1 + alpha*u/(1-u)),
where D = bytes_in_flight / (L0 * peak) is the *demand* (the utilisation the unloaded model would
predict before the min()). Multiplying out gives a quadratic with exactly one root in (0, 1):

    u = 2D / ((1 + D) + sqrt((1 - D)^2 + 4 * alpha * D))                  (4)

With alpha = 0, (4) reduces to min(D, 1), i.e. exactly model (2). alpha > 0 rounds the corner.
alpha is fitted by least squares to measured sweep records (fit_loaded_latency).

All numbers come from data/<gpu_tag>/probes.jsonl and gemv_sweep.jsonl (measured on the GPU);
nothing here is a hard-coded hardware constant.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

# Bytes a GEMV lane has in flight per group load: 16 B of packed int4 weights + its 2 B fp16 scale.
GEMV_BYTES_PER_GROUP = 18
# Bytes of one vector load in the stream probe (uint4).
STREAM_BYTES_PER_LOAD = 16
WARP_SIZE = 32


@dataclass(frozen=True)
class GPUParams:
    """Measured parameters of one GPU for the Little's Law model.

    sm_count        number of SMs (multiProcessorCount)
    dram_latency_ns unloaded DRAM load-to-use latency L0 (pointer-chase plateau beyond L2)
    peak_bw_Bps     peak *achievable* read bandwidth (max over the stream_bw sweep), bytes/s
    l2_bytes        L2 cache size (used to find the DRAM plateau of the pointer chase)
    """
    sm_count: int
    dram_latency_ns: float
    peak_bw_Bps: float
    l2_bytes: int
    name: str = ""

    @property
    def latency_bandwidth_product(self) -> float:
        """Bytes that must be in flight (whole GPU) to reach peak with unloaded latency: peak x L0."""
        return self.peak_bw_Bps * self.dram_latency_ns * 1e-9


# --------------------------------------------------------------------------------------------
# Core model
# --------------------------------------------------------------------------------------------

def bytes_in_flight(warps_per_sm, ilp, bytes_per_load=16, sm_count=1):
    """Bytes of loads outstanding at once, whole GPU (pass sm_count=1 for per SM).

    Every lane of every resident warp has `ilp` independent loads of `bytes_per_load` bytes issued
    before it needs the first result (Volkov 2016: TLP = warps, ILP = independent loads per thread):

        BIF = warps_per_sm * 32 lanes * ilp * bytes_per_load * sm_count
    """
    bif = np.asarray(warps_per_sm, dtype=float) * WARP_SIZE * ilp * bytes_per_load * sm_count
    return float(bif) if bif.ndim == 0 else bif


def demand(params: GPUParams, bif):
    """D = BIF / (L0 * peak): utilisation that Little's Law with unloaded latency asks for (may exceed 1)."""
    return np.asarray(bif, dtype=float) / params.latency_bandwidth_product


def loaded_utilization(D, alpha: float):
    """Closed-form fixed point of u = D / (1 + alpha*u/(1-u)), equation (4) in the module docstring.

    Derivation: u*(1-u) + alpha*u^2 = D*(1-u)  ->  (alpha-1) u^2 + (1+D) u - D = 0. The root in (0,1)
    written in the rationalised form below is numerically safe for every alpha >= 0 (incl. alpha = 1).
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    D = np.asarray(D, dtype=float)
    return 2.0 * D / ((1.0 + D) + np.sqrt((1.0 - D) ** 2 + 4.0 * alpha * D))


def loaded_utilization_iterative(D: float, alpha: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Same fixed point as loaded_utilization(), found by iterating on u = g(u).

    g(u) = min(1, D / (1 + alpha*u/(1-u))) is *decreasing* in u, so the naive iteration u <- g(u) can
    oscillate. Instead we keep a bracket [lo, hi] around the fixed point (h(u) = u - g(u) is increasing,
    h(0) <= 0, h(1) >= 0) and halve it each step. Used to cross-check the closed form in the tests.
    """
    def g(u: float) -> float:
        if u >= 1.0:
            return 0.0 if alpha > 0 else min(1.0, D)
        return min(1.0, D / (1.0 + alpha * u / (1.0 - u)))

    lo, hi = 0.0, 1.0
    u = 0.5
    for _ in range(max_iter):
        u = 0.5 * (lo + hi)
        if u - g(u) > 0:
            hi = u
        else:
            lo = u
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def loaded_latency_ns(params: GPUParams, u, alpha: float):
    """Equation (3): latency under load, L(u) = L0 * (1 + alpha * u / (1 - u))."""
    u = np.asarray(u, dtype=float)
    return params.dram_latency_ns * (1.0 + alpha * u / (1.0 - u))


def predict_bw(params: GPUParams, bytes_in_flight, loaded_latency: bool = False, alpha: Optional[float] = None):
    """Predicted achieved bandwidth (bytes/s) for a given number of bytes in flight (whole GPU).

    loaded_latency=False:  Little's Law with unloaded latency, min(peak, BIF / L0)          (2)
    loaded_latency=True:   latency grows with utilisation per (3); solved in closed form (4)
    """
    D = demand(params, bytes_in_flight)
    if loaded_latency:
        if alpha is None:
            raise ValueError("loaded_latency=True needs alpha (see fit_loaded_latency)")
        u = loaded_utilization(D, alpha)
    else:
        u = np.minimum(D, 1.0)
    bw = u * params.peak_bw_Bps
    return float(bw) if np.ndim(bw) == 0 else bw


# --------------------------------------------------------------------------------------------
# Mapping measured records to (bytes in flight, measured bandwidth)
# --------------------------------------------------------------------------------------------

def record_bytes_in_flight(rec: dict, params: GPUParams) -> float:
    """Whole-GPU bytes in flight for one stream-probe or GEMV-sweep record."""
    if rec.get("probe") in ("stream", "stream_peak"):
        return float(rec["bytes_in_flight_total"])
    if "kernel" in rec:
        # Prefer the average resident warps (accounts for grids too small to fill every block slot).
        warps = rec.get("warps_per_sm_eff", rec["warps_per_sm"])
        return bytes_in_flight(warps, rec["ilp"], GEMV_BYTES_PER_GROUP, params.sm_count)
    raise ValueError(f"not a stream or gemv record: {rec}")


def usable_records(records: Iterable[dict]) -> List[dict]:
    """Records that carry a real timed bandwidth: stream configs, and correct + timed GEMV configs."""
    out = []
    for r in records:
        if r.get("probe") == "stream" and r.get("GBps", 0) > 0:
            out.append(r)
        elif "kernel" in r and r.get("ok") and r.get("us", 0) > 0 and r.get("GBps", 0) > 0 \
                and r.get("mode", "timed") != "ncu":
            out.append(r)
    return out


def _xy(params: GPUParams, records: Sequence[dict]):
    bif = np.array([record_bytes_in_flight(r, params) for r in records], dtype=float)
    meas = np.array([r["GBps"] * 1e9 for r in records], dtype=float)
    return bif, meas


def fit_loaded_latency(params: GPUParams, sweep_records: Iterable[dict]) -> float:
    """Fit alpha of L(u) = L0 (1 + alpha u/(1-u)) by least squares on the relative bandwidth error.

    Step 1 (initial guess, linear least squares): each measurement implies an effective latency
    L_eff = BIF / measured (Little's Law). Model (3) says L_eff/L0 - 1 = alpha * u/(1-u), linear in alpha,
    so alpha0 = sum(x*y)/sum(x*x) with x = u/(1-u), y = L_eff/L0 - 1 (points with u >= 0.98 skipped,
    where x blows up).
    Step 2 (refinement): scipy.optimize.least_squares on residuals predict/measured - 1, alpha >= 0.
    This minimises what we report (relative error), so every config counts equally.
    """
    from scipy.optimize import least_squares

    recs = usable_records(sweep_records)
    if len(recs) < 2:
        raise ValueError("need at least 2 usable sweep records to fit alpha")
    bif, meas = _xy(params, recs)
    u = meas / params.peak_bw_Bps
    keep = u < 0.98
    x = u[keep] / (1.0 - u[keep])
    y = bif[keep] / meas[keep] / (params.dram_latency_ns * 1e-9) - 1.0
    alpha0 = float(np.dot(x, y) / np.dot(x, x)) if keep.any() and np.dot(x, x) > 0 else 0.5
    alpha0 = min(max(alpha0, 1e-6), 1e3)

    def resid(a):
        return predict_bw(params, bif, loaded_latency=True, alpha=float(a[0])) / meas - 1.0

    sol = least_squares(resid, x0=[alpha0], bounds=([0.0], [1e4]))
    return float(sol.x[0])


# --------------------------------------------------------------------------------------------
# Loading data/<gpu_tag>/ and building GPUParams
# --------------------------------------------------------------------------------------------

def load_jsonl(path) -> List[dict]:
    """Read a JSONL file, skipping blank lines and lines that are not valid JSON objects
    (a GPU run that was killed mid-line must not make the analysis crash)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def load_probes(data_dir) -> List[dict]:
    return load_jsonl(Path(data_dir) / "probes.jsonl")


def load_gemv_sweep(data_dir) -> List[dict]:
    return load_jsonl(Path(data_dir) / "gemv_sweep.jsonl")


def dram_latency_from_chase(probe_records: Iterable[dict], l2_bytes: int, min_l2_multiple: float = 8.0) -> float:
    """Unloaded DRAM latency L0 (ns) from the pointer chase.

    A random chase over a footprint F still hits in L2 with probability about L2/F, which pulls the
    measured latency below the true DRAM latency: at F = 8x L2 about 12% of loads still hit. So we use
    footprints >= min_l2_multiple x L2 and take the median of the (up to) three smallest of them: the
    smallest DRAM-resident footprints have the least extra TLB-miss latency. If the chase did not go
    that far (e.g. 96 MB L2 x 8 > 1 GB max footprint), the largest footprint measured is used.
    """
    chase = sorted((r for r in probe_records if r.get("probe") == "pointer_chase"),
                   key=lambda r: r["footprint_bytes"])
    if not chase:
        raise ValueError("no pointer_chase records")
    dram = [r["latency_ns"] for r in chase if r["footprint_bytes"] >= min_l2_multiple * l2_bytes][:3]
    if not dram:
        dram = [chase[-1]["latency_ns"]]
    return float(np.median(dram))


def params_from_probes(probe_records: Sequence[dict], sm_count: Optional[int] = None,
                       l2_bytes: Optional[int] = None, name: str = "") -> GPUParams:
    """GPUParams from probes.jsonl: peak = stream_peak (or max stream GBps), L0 = DRAM chase plateau.
    sm_count / l2_bytes default to the values the stream probe recorded."""
    stream = [r for r in probe_records if r.get("probe") in ("stream", "stream_peak")]
    if not stream:
        raise ValueError("no stream records in probes")
    sm_count = sm_count or int(stream[0]["sm_count"])
    l2_bytes = l2_bytes or int(stream[0]["l2_bytes"])
    peak = max(r["GBps"] for r in stream) * 1e9
    name = name or str(stream[0].get("gpu", ""))
    return GPUParams(sm_count=sm_count, dram_latency_ns=dram_latency_from_chase(probe_records, l2_bytes),
                     peak_bw_Bps=peak, l2_bytes=l2_bytes, name=name)


# --------------------------------------------------------------------------------------------
# Predicted vs measured
# --------------------------------------------------------------------------------------------

def _label(r: dict) -> str:
    if r.get("probe") == "stream":
        return f"stream bps={r['blocks_per_sm']} tpb={r['threads_per_block']} B/thr={r['bytes_per_thread_in_flight']}"
    return f"{r.get('shape', str(r['N']) + 'x' + str(r['K']))} {r['kernel']} ilp={r['ilp']} bps={r['blocks_per_sm']}"


def prediction_table(params: GPUParams, records: Iterable[dict], alpha: float):
    """pandas DataFrame: one row per usable record with measured and predicted GB/s and % errors.

    pred_littles_GBps = min(peak, BIF/L0)        (before correction)
    pred_loaded_GBps  = loaded-latency model     (after correction, with the given alpha)
    """
    import pandas as pd

    cols = ["config", "bytes_in_flight", "measured_GBps", "pred_littles_GBps", "pred_loaded_GBps",
            "err_littles_pct", "err_loaded_pct"]
    recs = usable_records(records)
    if not recs:
        return pd.DataFrame(columns=cols)
    bif, meas = _xy(params, recs)
    p0 = np.atleast_1d(predict_bw(params, bif))
    p1 = np.atleast_1d(predict_bw(params, bif, loaded_latency=True, alpha=alpha))
    return pd.DataFrame({
        "config": [_label(r) for r in recs],
        "bytes_in_flight": bif,
        "measured_GBps": meas / 1e9,
        "pred_littles_GBps": p0 / 1e9,
        "pred_loaded_GBps": p1 / 1e9,
        "err_littles_pct": (p0 / meas - 1.0) * 100.0,
        "err_loaded_pct": (p1 / meas - 1.0) * 100.0,
    }, columns=cols)


def mape(table) -> dict:
    """Mean absolute percentage error before (Little's Law min()) and after (loaded latency)."""
    if len(table) == 0:
        return {"n": 0, "mape_littles_pct": math.nan, "mape_loaded_pct": math.nan}
    return {"n": int(len(table)),
            "mape_littles_pct": float(np.mean(np.abs(table["err_littles_pct"]))),
            "mape_loaded_pct": float(np.mean(np.abs(table["err_loaded_pct"])))}


def summarize(data_dir) -> dict:
    """Fit on the stream probe, then predict the GEMV sweep out-of-sample (and in-sample for reference)."""
    probes = load_probes(data_dir)
    sweep = load_gemv_sweep(data_dir)
    params = params_from_probes(probes)
    out = {"params": params, "alpha_stream": None, "alpha_gemv": None}
    stream = [r for r in probes if r.get("probe") == "stream"]
    if len(usable_records(stream)) >= 2:
        out["alpha_stream"] = fit_loaded_latency(params, stream)
        out["stream_table"] = prediction_table(params, stream, out["alpha_stream"])
    if len(usable_records(sweep)) >= 2:
        out["alpha_gemv"] = fit_loaded_latency(params, sweep)
        if out["alpha_stream"] is not None:
            out["gemv_table_stream_alpha"] = prediction_table(params, sweep, out["alpha_stream"])
        out["gemv_table_gemv_alpha"] = prediction_table(params, sweep, out["alpha_gemv"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Little's Law model vs measured probes / GEMV sweep")
    ap.add_argument("data_dir", help="data/<gpu_tag> containing probes.jsonl and gemv_sweep.jsonl")
    ap.add_argument("--rows", type=int, default=20, help="table rows to print")
    args = ap.parse_args(argv)
    s = summarize(args.data_dir)
    p = s["params"]
    print(f"{p.name}: SMs={p.sm_count}  L0={p.dram_latency_ns:.0f} ns  peak={p.peak_bw_Bps / 1e9:.0f} GB/s  "
          f"latency x BW = {p.latency_bandwidth_product / 1e6:.2f} MB in flight needed")
    for key, alpha_key in (("stream_table", "alpha_stream"), ("gemv_table_stream_alpha", "alpha_stream"),
                           ("gemv_table_gemv_alpha", "alpha_gemv")):
        if key in s:
            m = mape(s[key])
            print(f"\n{key}: alpha={s[alpha_key]:.3f}  n={m['n']}  MAPE before={m['mape_littles_pct']:.1f}%  "
                  f"after={m['mape_loaded_pct']:.1f}%")
            print(s[key].head(args.rows).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
