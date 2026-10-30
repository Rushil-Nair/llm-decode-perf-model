"""SYNTHETIC stub predictors for the correlate/whatif/agents tests.

These mimic the CONTRACTS.md interfaces of perfsim.microarch / ops_model / engine with
deliberately simple formulas, so the correlation plumbing can be tested without the
real modules (which are written in parallel). They are NOT the project's model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from correlate.backend import Backend
from perfsim.models import decode_ops, prefill_ops
from perfsim.roofline import op_time

STEP_OVERHEAD_S = 30e-6  # fixed per-forward-pass overhead in the stub engine


@dataclass(frozen=True)
class StubGPUParams:
    sm_count: int
    dram_latency_ns: float
    peak_bw_Bps: float
    l2_bytes: int = 0


def bytes_in_flight(warps_per_sm, ilp, bytes_per_load, sm_count):
    return float(warps_per_sm) * 32 * ilp * bytes_per_load * sm_count


def predict_bw(params, bytes_in_flight, loaded_latency=False, alpha=None):
    """min(peak, BIF / latency); with loaded latency, latency(u) = L0 (1 + alpha u / (1 - u))
    solved by damped fixed-point iteration."""
    L0, peak = params.dram_latency_ns * 1e-9, params.peak_bw_Bps
    bw = min(peak, bytes_in_flight / L0)
    if loaded_latency and alpha:
        for _ in range(200):
            u = min(bw / peak, 0.99)
            new = min(peak, bytes_in_flight / (L0 * (1 + alpha * u / (1 - u))))
            bw = 0.5 * bw + 0.5 * new
    return bw


def fit_loaded_latency(params, sweep_records):
    return 0.2


class StubCost:
    def __init__(self, hw, calib=None, mode="sol"):
        self.hw, self.mode = hw, mode
        c = calib if (calib and mode == "empirical") else {}
        self.eff_mem = float(c.get("mem_eff", 1.0))
        self.eff_compute = float(c.get("compute_eff", 1.0))

    def time(self, op, batch):
        return op_time(op, self.hw, "fp16", eff_compute=self.eff_compute, eff_memory=self.eff_mem).time


def _sum(cost, ops, batch):
    tw = sum(cost.time(o, batch) * o.count for o in ops if o.kind == "gemm")
    tr = sum(cost.time(o, batch) * o.count for o in ops if o.kind != "gemm")
    return tw, tr


def decode_step_time(model, cost, batch, context_len, engine="vllm", passes_per_step=1):
    w_bits = 4.5 if engine == "llamacpp" else 16
    tw, tr = _sum(cost, decode_ops(model, batch, context_len, w_bits=w_bits), batch)
    # every extra pass re-reads the weights (slot bug); per-sequence work is unchanged
    return passes_per_step * (tw + STEP_OVERHEAD_S) + tr


def prefill_time(model, cost, batch, input_len):
    tw, tr = _sum(cost, prefill_ops(model, batch, input_len, w_bits=16), batch)
    return tw + tr + STEP_OVERHEAD_S


def predict_vllm_latency(model, cost, batch, input_len, output_len):
    t = prefill_time(model, cost, batch, input_len)
    if output_len > 1:
        t += (output_len - 1) * decode_step_time(model, cost, batch, input_len + output_len // 2)
    return t


def make_backend() -> Backend:
    return Backend(
        name="synthetic-stub",
        make_gpu_params=lambda f: StubGPUParams(**f),
        bytes_in_flight=bytes_in_flight, predict_bw=predict_bw, fit_loaded_latency=fit_loaded_latency,
        make_cost=lambda hw, calib=None, mode="sol": StubCost(hw, calib, mode),
        load_calib=lambda p: json.loads(Path(p).read_text()),
        decode_step_time=decode_step_time, prefill_time=prefill_time,
        predict_vllm_latency=predict_vllm_latency,
    )
