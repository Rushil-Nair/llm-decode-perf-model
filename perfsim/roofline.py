"""Roofline timing on top of the analytic op costs.

For each op:
    t_compute = flops / (peak * eff_compute)
    t_memory  = bytes / (bandwidth * eff_memory)
    t_op      = max(t_compute, t_memory)          # ideal roofline bound
    attainable FLOP/s = min(peak, AI * bandwidth)

A forward pass is the *sum* of its ops' times (each op is a separate kernel,
no overlap between ops). With eff_* = 1.0 the result is a lower bound on time
(upper bound on throughput) for that hardware: it ignores kernel launch
overhead, dequantization cost, imperfect bandwidth, scheduling, sampling, etc.

Multi-device sharding (`n_devices = P > 1`) is a deliberately simple
alpha-beta model (alpha = per-hop latency, beta = 1/link bandwidth):
  * "tp" (tensor parallel over all P devices): every op's FLOPs and bytes are
    split evenly over P devices, which run it concurrently. Two all-reduces
    per layer (after o_proj and after mlp_down) of an N*hidden activation
    message. All-reduce cost = 2*ceil(log2 P)*alpha + 2*(P-1)/P * msg * beta
    (recursive halving/doubling, Rabenseifner-style).
  * "pp" (pipeline over S = min(P, n_layers) stages, with tensor parallelism
    of degree ceil(P/S) inside a stage when P > n_layers): one microbatch
    visits the stages in sequence, so per-op time is divided only by the
    intra-stage TP degree; add (S-1) activation transfers of msg*beta + alpha
    and the intra-stage all-reduces. Reported latency/throughput are for ONE
    microbatch in flight (no overlap between stages). With S microbatches in
    flight, throughput could be up to S x higher at S x the concurrent
    sequences and KV cache; that is not claimed here.
  * Interconnect bandwidth is the single per-device number in hardware.yaml.
    For the SRAM chip it is a hypothetical value (see hardware.yaml).
  * KV heads are assumed divisible across devices (no replication when
    P > n_kv_heads), which is optimistic for large P.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .hardware import Hardware
from .models import ModelConfig, Op, kv_cache_bytes, layer_ops, totals


@dataclass
class OpTime:
    name: str
    kind: str
    flops: float
    bytes: float
    ai: float
    t_compute: float
    t_memory: float
    time: float  # per instance
    count: int
    bound: str  # "compute" | "memory"

    @property
    def total_time(self) -> float:
        return self.time * self.count


def attainable_flops(ai: float, peak: float, bandwidth: float) -> float:
    return min(peak, ai * bandwidth)


def op_time(op: Op, hw: Hardware, precision: str = "fp16", eff_compute: float = 1.0,
            eff_memory: float = 1.0, scale: float = 1.0) -> OpTime:
    """Roofline time of one op instance. `scale` divides FLOPs and bytes (sharding)."""
    flops, nbytes = op.flops / scale, op.bytes / scale
    peak = hw.peak(precision) * eff_compute
    bw = hw.memory_bandwidth * eff_memory
    tc = flops / peak
    tm = nbytes / bw
    return OpTime(op.name, op.kind, flops, nbytes, op.arithmetic_intensity, tc, tm,
                  max(tc, tm), op.count, "compute" if tc >= tm else "memory")


def pass_time(ops: List[Op], hw: Hardware, precision: str = "fp16", **kw) -> Dict:
    """Sum of per-op roofline times for one forward pass on one device."""
    timed = [op_time(o, hw, precision, **kw) for o in ops]
    total = sum(t.total_time for t in timed)
    tc = sum(t.t_compute * t.count for t in timed)
    tm = sum(t.t_memory * t.count for t in timed)
    by_op = {t.name: t.total_time for t in timed}
    return {"time": total, "t_compute_sum": tc, "t_memory_sum": tm, "ops": timed, "by_op": by_op}


def allreduce_time(msg_bytes: float, P: int, bw: float, lat: float) -> float:
    if P <= 1:
        return 0.0
    return 2 * math.ceil(math.log2(P)) * lat + 2 * (P - 1) / P * msg_bytes / bw


def shard_plan(m: ModelConfig, P: int, mode: str):
    """(stages, tp_degree) for a P-device deployment."""
    if mode == "tp":
        return 1, P
    if mode == "pp":
        S = min(P, m.n_layers)
        return S, math.ceil(P / S)
    raise ValueError("mode must be 'tp' or 'pp'")


def comm_time(m: ModelConfig, hw: Hardware, n_tokens: int, P: int, mode: str,
              act_bits: float = 16) -> float:
    """Interconnect time per forward pass (see module docstring)."""
    if P <= 1:
        return 0.0
    if not hw.interconnect_bandwidth:
        raise ValueError(f"{hw.key} has no interconnect bandwidth")
    msg = n_tokens * m.hidden * act_bits / 8
    bw, lat = hw.interconnect_bandwidth, hw.hop_latency
    S, tp = shard_plan(m, P, mode)
    t = 2 * m.n_layers * allreduce_time(msg, tp, bw, lat)
    t += (S - 1) * (msg / bw + lat)
    return t


def memory_required(m: ModelConfig, B: int, L: int, w_bits: float = 16,
                    kv_bits: float = 16) -> Dict[str, float]:
    w = m.weight_bytes(w_bits, include_embedding_table=True)
    kv = kv_cache_bytes(m, B, L, kv_bits)
    return {"weights": w, "kv_cache": kv, "total": w + kv}


def devices_needed(m: ModelConfig, hw: Hardware, B: int = 1, L: int = 0,
                   w_bits: float = 16, kv_bits: float = 16,
                   usable_fraction: float = 1.0) -> int:
    """Smallest device count whose combined memory holds weights + KV cache.
    usable_fraction < 1 reserves room for activations/runtime (assumption)."""
    need = memory_required(m, B, L, w_bits, kv_bits)["total"]
    return max(1, math.ceil(need / (hw.memory_capacity * usable_fraction)))


def step(m: ModelConfig, hw: Hardware, B: int, L: int, T: int, *, w_bits: float = 16,
         kv_bits: float = 16, act_bits: float = 16, precision: str = "fp16",
         n_devices: int = 1, mode: str = "tp", eff_compute: float = 1.0,
         eff_memory: float = 1.0, weight_bytes_override: Optional[float] = None) -> Dict:
    """Time for one forward pass of B sequences x T new tokens, L cached.

    weight_bytes_override: total bytes of weights read per pass (e.g. an actual
    GGUF file size); per-op weight bytes are rescaled proportionally.
    """
    ops = layer_ops(m, B, L, T, w_bits=w_bits, kv_bits=kv_bits, act_bits=act_bits)
    if weight_bytes_override is not None:
        cur = sum(o.weight_bytes * o.count for o in ops if o.kind == "gemm")
        f = weight_bytes_override / cur
        for o in ops:
            if o.kind == "gemm":
                o.weight_bytes *= f
    P = n_devices
    kw = dict(eff_compute=eff_compute, eff_memory=eff_memory)
    if P == 1:
        pt = pass_time(ops, hw, precision, **kw)
        compute_mem = pt["time"]
    else:
        _, tp_degree = shard_plan(m, P, mode)
        timed = [op_time(o, hw, precision, scale=tp_degree, **kw) for o in ops]
        pt = {"ops": timed, "by_op": {t.name: t.total_time for t in timed},
              "t_compute_sum": sum(t.t_compute * t.count for t in timed),
              "t_memory_sum": sum(t.t_memory * t.count for t in timed)}
        # tp: all P devices work on every op in parallel, 1/P each.
        # pp: one stage at a time is busy; inside a stage, 1/tp_degree each.
        compute_mem = sum(t.total_time for t in timed)
    t_comm = comm_time(m, hw, B * T, P, mode, act_bits)
    tot = totals(ops)
    t = compute_mem + t_comm
    mem = memory_required(m, B, L + T, w_bits, kv_bits)
    if weight_bytes_override is not None:
        mem["weights"] = weight_bytes_override
        mem["total"] = mem["weights"] + mem["kv_cache"]
    return {
        "time": t,
        "t_ops": compute_mem,
        "t_comm": t_comm,
        "t_compute_sum": pt["t_compute_sum"],
        "t_memory_sum": pt["t_memory_sum"],
        "bound": "compute" if pt["t_compute_sum"] > pt["t_memory_sum"] else "memory",
        # what dominates the step: interconnect if comm time exceeds op time
        "limiter": "interconnect" if t_comm > compute_mem else
        ("compute" if pt["t_compute_sum"] > pt["t_memory_sum"] else "memory"),
        "by_op": pt["by_op"],
        "ops": pt["ops"],
        "flops": tot["flops"],
        "bytes": tot["bytes"],
        "weight_bytes_read": tot["weight_bytes"] if weight_bytes_override is None
        else sum(o.weight_bytes * o.count for o in ops),
        "kv_read_bytes": tot["kv_read_bytes"],
        "memory_needed": mem["total"],
        "fits": mem["total"] <= hw.memory_capacity * P,
    }


def decode(m: ModelConfig, hw: Hardware, B: int, L: int, **kw) -> Dict:
    """Decode step: per-token latency (s) and aggregate throughput (tokens/s)."""
    r = step(m, hw, B, L, 1, **kw)
    r["latency_per_token"] = r["time"]
    r["throughput"] = B / r["time"]
    r["per_user_tok_s"] = 1.0 / r["time"]
    return r


def prefill(m: ModelConfig, hw: Hardware, B: int, prompt_len: int, **kw) -> Dict:
    r = step(m, hw, B, 0, prompt_len, **kw)
    r["ttft"] = r["time"]
    r["throughput"] = B * prompt_len / r["time"]
    return r


def pick_precision(hw: Hardware, w_bits: float) -> str:
    """Compute precision used for the peak FLOP/s: fp8 (or int8 if no fp8) for
    <=8-bit weights, else fp16. Weight-only quantized kernels often compute in
    fp16 instead; for memory-bound decode the choice barely matters."""
    if w_bits <= 8:
        for p in ("fp8", "int8"):
            if p in hw.peak_flops:
                return p
    return "fp16"


def deployment(m: ModelConfig, hw: Hardware, B: int, L: int, *, phase: str = "decode",
               prompt_len: int = 0, w_bits: float = 16, kv_bits: float = 16,
               mode: str = "tp", usable_fraction: float = 0.9,
               n_devices: Optional[int] = None, **kw) -> Dict:
    """Size the deployment (minimum devices to hold weights + KV at context L,
    unless n_devices is given) and time one decode step or one prefill."""
    P = n_devices or devices_needed(m, hw, B, L, w_bits, kv_bits, usable_fraction)
    prec = kw.pop("precision", None) or pick_precision(hw, w_bits)
    common = dict(w_bits=w_bits, kv_bits=kv_bits, precision=prec, n_devices=P, mode=mode, **kw)
    r = decode(m, hw, B, L, **common) if phase == "decode" else prefill(m, hw, B, prompt_len, **common)
    r["n_devices"], r["precision"], r["mode"] = P, prec, mode
    return r


def bottleneck_summary(r: Dict, top: int = 3) -> List[tuple]:
    """Largest ops by total time: [(name, seconds, bound, share)]."""
    tot = r["t_ops"] or 1.0
    rows = sorted(r["ops"], key=lambda t: -t.total_time)[:top]
    return [(t.name, t.total_time, t.bound, t.total_time / tot) for t in rows]
