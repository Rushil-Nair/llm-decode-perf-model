"""SYNTHETIC ops.jsonl fixtures for tests/test_ops_model.py. NOT measured data.

Each record is built from the real bench plan (bench/ops_bench.py: same shapes, same
analytic FLOPs/bytes) and a KNOWN ground truth:

    us = 1e6 * ( T0_US[op]*1e-6 + SOL(spec sheet) / TRUE_EFF(op, feature, weight bytes) )

so a correct fit must recover T0_US and TRUE_EFF and reproduce every `us`.
The same ground truth is used for both GPUs, so a leave-one-GPU-out projection
(calibrate on one, predict the other from its spec sheet) must also be exact.

Regenerate:  python3 tests/fixtures/synthetic_ops_gen.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from perfsim.hardware import load_hardware  # noqa: E402
from perfsim.ops_model import sol_time  # noqa: E402

GPUS = ("h100_sxm", "rtx5090")
T0_US = {"gemm": 3.0, "attn_decode": 5.0, "attn_prefill": 8.0, "rmsnorm": 2.5}
SMALL_WEIGHT_BYTES = 16e6      # GEMMs with smaller weights get a shape penalty
SMALL_SHAPE_FACTOR = 0.85
SLOWER_BACKEND_FACTOR = 1.3    # a duplicate, slower attention backend the fit must ignore


def load_bench():
    spec = importlib.util.spec_from_file_location("ops_bench", REPO / "bench" / "ops_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lerp_log10(v, xs, ys):
    """piecewise linear in log10(v), clamped at the ends"""
    lv = math.log10(v)
    if lv <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if lv <= x1:
            return y0 + (y1 - y0) * (lv - x0) / (x1 - x0)
    return ys[-1]


def true_eff(op: str, flops: float, nbytes: float, weight_bytes: float | None = None) -> float:
    if op == "gemm":
        ai = flops / nbytes
        e = _lerp_log10(ai, [0, 2, 2.5, 3, 3.5], [0.80, 0.70, 0.55, 0.75, 0.80])  # dip near the ridge
        if weight_bytes is not None and weight_bytes < SMALL_WEIGHT_BYTES:
            e *= SMALL_SHAPE_FACTOR
        return e
    if op == "attn_decode":
        return _lerp_log10(nbytes, [6, 9], [0.35, 0.85])
    if op == "attn_prefill":
        return _lerp_log10(flops, [9, 12], [0.35, 0.65])
    if op == "rmsnorm":
        return _lerp_log10(nbytes, [4, 8], [0.10, 0.85])
    raise ValueError(op)


def true_us(op, flops, nbytes, hw, weight_bytes=None, dtype="bf16") -> float:
    return T0_US[op] + 1e6 * sol_time(flops, nbytes, hw, dtype) / true_eff(op, flops, nbytes, weight_bytes)


def make_records(gpu: str) -> list:
    bench = load_bench()
    hw = load_hardware()[gpu]
    recs = []
    for spec in bench.plan(quick=False):
        flops, nbytes = bench.spec_cost(spec)
        wb = spec["shape"].get("weight_bytes")
        us = true_us(spec["op"], flops, nbytes, hw, wb)
        impl = "synthetic"
        recs.append({"op": spec["op"], "dtype": "bf16", "shape": spec["shape"], "us": us,
                     "flops": flops, "bytes": nbytes, "impl": impl})
        if spec["op"] == "attn_decode" and not spec["shape"].get("probe"):
            recs.append({"op": spec["op"], "dtype": "bf16", "shape": spec["shape"],
                         "us": us * SLOWER_BACKEND_FACTOR, "flops": flops, "bytes": nbytes,
                         "impl": "synthetic_slower"})
    return recs


def fixture_path(gpu: str) -> Path:
    return HERE / f"synthetic_ops_{gpu}.jsonl"


def render(gpu: str) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in make_records(gpu))


def main():
    for gpu in GPUS:
        fixture_path(gpu).write_text(render(gpu))
        print("wrote", fixture_path(gpu))


if __name__ == "__main__":
    main()
