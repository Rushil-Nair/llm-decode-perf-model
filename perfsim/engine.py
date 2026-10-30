"""Level 3: step-level serving model (vLLM static batch + llama.cpp slots).

This is a transparent analytical step model, not an event simulator. It turns the
per-op costs of Level 2 (an OpCostModel) into the latency of whole serving steps:

    decode step (B sequences, L cached tokens each, k forward passes per step)
        t_step = sum over the k passes of [ sum_ops cost(op) + overhead_pass ]
        where the B sequences are split as evenly as possible over the k passes.

    prefill (B prompts of S tokens)
        t_prefill = sum over the prefill passes of [ sum_ops cost(op) + overhead_pass ]
        (one pass, or several if vLLM's chunked prefill splits the prompt tokens)

    vLLM `bench latency` (one static batch, B prompts, S input, O output tokens)
        latency = t_prefill(B, S) + sum_{i=1}^{O-1} t_step(B, L = S + i - 1)

What each forward pass reads (from perfsim.models.layer_ops):
  * every weight matrix ONCE per pass, however many sequences are in it
    (this is why batching helps decode, and why the llama.cpp slot bug hurt:
    k passes per step = k full weight reads);
  * each sequence's own K/V cache once per pass (grows with batch and context);
  * activations, small elementwise ops, LM head, one embedding row per token.

Assumptions (deliberate, so the model stays explainable):
  * Ops run back to back on one GPU: pass time = sum of op times (no overlap
    between kernels; no multi-stream concurrency).
  * Everything that is not an op -- kernel launch gaps, CUDA-graph replay,
    the scheduler, sampling, detokenization, Python -- is ONE constant per
    forward pass (`overhead_s`). It is fitted, not derived. We do not model
    CUDA graphs explicitly (vLLM's default captures decode batches as graphs,
    llama.cpp also uses graphs for decode); the fitted constant absorbs it.
  * Static batch: all B sequences start together and finish together
    (vLLM `bench latency` uses ignore_eos=True, so every sequence generates
    exactly output_len tokens). No preemption, no continuous-batching arrivals.
  * Prefix caching off (vLLM's bench latency disables it) and no speculative
    decoding.
  * Attention is flash-style: K/V read once per sequence per layer per pass.
  * llama.cpp: Q4_K_M weights (GGUF file size from llamacpp_slots), f16 KV,
    per-slot KV streams. A decode step that is split into k passes runs each
    group of sequences in its own pass; fractional measured k (e.g. 4.34) is
    interpolated linearly between floor(k) and ceil(k) passes.
  * The llama.cpp slot test reports wall-clock tok/s that also includes prompt
    processing of new requests; the model predicts decode-only throughput, so
    expect it to read slightly high there.

The cost model interface (Level 2, perfsim/ops_model.py):
    cost.time(op) -> seconds  or  cost.time(op, batch) -> seconds
for ONE instance of a perfsim.models.Op (the engine multiplies by op.count,
the same convention as roofline.op_time().time). If the cost object has a
`time_includes_count = True` attribute, the engine does not multiply.
The per-pass overhead comes from (first match wins):
    the `overhead_s` argument  ->  cost.pass_overhead_s (float, or dict keyed
    by engine name "vllm"/"llamacpp")  ->  0.0 (pure SOL).
Each op also gets a `dtype` attribute hint ("bf16" for vLLM, "q4_k" for
llama.cpp GEMMs) so an empirical cost model can pick the right efficiency curve.
"""
from __future__ import annotations

import inspect
import math
from typing import Dict, List, Optional

from . import llamacpp_slots
from .models import ModelConfig, Op, layer_ops

ENGINES = ("vllm", "llamacpp")

# Default numeric formats per engine (see the module docstring).
ENGINE_DEFAULTS = {
    # dtype: hint for GEMM weights; act_dtype: hint for attention / elementwise ops
    "vllm": {"w_bits": 16.0, "kv_bits": 16.0, "act_bits": 16.0, "dtype": "bf16", "act_dtype": "bf16"},
    "llamacpp": {"w_bits": llamacpp_slots.Q4_K_BITS, "kv_bits": 16.0, "act_bits": 16.0,
                 "dtype": "q4_k", "act_dtype": "f16"},
}


# ---------------------------------------------------------------------------------------
# helpers: talking to the cost model
# ---------------------------------------------------------------------------------------
def _takes_batch(cost) -> bool:
    """True if cost.time accepts a second positional argument (batch)."""
    try:
        params = [p for p in inspect.signature(cost.time).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params)


def op_seconds(cost, op: Op, batch: int) -> float:
    """Seconds for all `op.count` instances of `op` in one forward pass."""
    t = cost.time(op, batch) if _takes_batch(cost) else cost.time(op)
    return float(t) if getattr(cost, "time_includes_count", False) else float(t) * op.count


def pass_overhead(cost, engine: str, overhead_s: Optional[float] = None) -> float:
    """Fixed seconds per forward pass (launch gaps, scheduler, sampling ...)."""
    if overhead_s is not None:
        return float(overhead_s)
    o = getattr(cost, "pass_overhead_s", None)
    if isinstance(o, dict):
        return float(o.get(engine, 0.0))
    return float(o) if o is not None else 0.0


def _check_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")


def _llamacpp_weight_bytes(model: ModelConfig) -> Optional[float]:
    """Per-pass weight bytes for llama.cpp Q4_K_M. Qwen3-8B: real GGUF size minus the
    (stored but not streamed) embedding table, from llamacpp_slots. Other models:
    None -> uniform Q4_K_BITS on every GEMM weight (layer_ops default)."""
    if model.name.lower().startswith("qwen3-8b"):
        return llamacpp_slots.weight_bytes_per_pass(include_embedding=False)
    return None


def _scale_gemm_weights(ops: List[Op], total_bytes: float) -> None:
    """Rescale GEMM weight bytes so their sum is `total_bytes` (same as roofline.step)."""
    cur = sum(o.weight_bytes * o.count for o in ops if o.kind == "gemm")
    f = total_bytes / cur
    for o in ops:
        if o.kind == "gemm":
            o.weight_bytes *= f


def _formats(engine: str, w_bits, kv_bits):
    d = ENGINE_DEFAULTS[engine]
    return (d["w_bits"] if w_bits is None else w_bits,
            d["kv_bits"] if kv_bits is None else kv_bits,
            d["act_bits"], d["dtype"], d["act_dtype"])


def _pass(model: ModelConfig, cost, B: int, L: int, T: int, engine: str,
          w_bits=None, kv_bits=None, weight_bytes_override=None) -> Dict[str, float]:
    """Op time of ONE forward pass over B sequences x T new tokens with L cached
    (no overhead). Returns {op_name: seconds} for per-op attribution."""
    wb, kb, ab, dtype, act_dtype = _formats(engine, w_bits, kv_bits)
    ops = layer_ops(model, B, L, T, w_bits=wb, kv_bits=kb, act_bits=ab)
    # llama.cpp default: use the real GGUF bytes (only when the caller did not pick w_bits)
    if weight_bytes_override is None and engine == "llamacpp" and w_bits is None:
        weight_bytes_override = _llamacpp_weight_bytes(model)
    if weight_bytes_override is not None:
        _scale_gemm_weights(ops, weight_bytes_override)
    by_op: Dict[str, float] = {}
    for o in ops:
        o.dtype = dtype if o.kind == "gemm" else act_dtype  # optional hint for the cost model
        by_op[o.name] = by_op.get(o.name, 0.0) + op_seconds(cost, o, B)
    return by_op


def _split(B: int, k: int) -> List[int]:
    """B sequences over k passes, as evenly as possible (sizes differ by <= 1)."""
    q, r = divmod(B, k)
    return [q + 1] * r + [q] * (k - r)


# ---------------------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------------------
def decode_step_breakdown(model: ModelConfig, cost, batch: int, context_len: int,
                          engine: str = "vllm", passes_per_step: float = 1.0, *,
                          overhead_s: Optional[float] = None, w_bits: Optional[float] = None,
                          kv_bits: Optional[float] = None,
                          weight_bytes_override: Optional[float] = None) -> Dict:
    """One decode step (every sequence writes 1 token) with per-op attribution.

    passes_per_step = k: the step runs as k forward passes over disjoint groups of
    sequences (llama.cpp slot bug: k > 1; vLLM and fixed llama.cpp: k = 1). Each pass
    re-reads all weights and pays the per-pass overhead once. Non-integer k (a measured
    average) interpolates linearly between floor(k) and ceil(k) passes.
    """
    _check_engine(engine)
    B, L = int(batch), int(context_len)
    if B < 1 or L < 0:
        raise ValueError("need batch >= 1 and context_len >= 0")
    k = min(max(float(passes_per_step), 1.0), float(B))  # 1 <= k <= B
    oh = pass_overhead(cost, engine, overhead_s)
    kw = dict(w_bits=w_bits, kv_bits=kv_bits, weight_bytes_override=weight_bytes_override)

    def for_k(kk: int) -> Dict:
        by_op: Dict[str, float] = {}
        for b in _split(B, kk):
            for name, t in _pass(model, cost, b, L, 1, engine, **kw).items():
                by_op[name] = by_op.get(name, 0.0) + t
        return {"by_op": by_op, "overhead_s": kk * oh}

    lo, hi = math.floor(k), math.ceil(k)
    a = for_k(lo)
    if hi != lo:
        b = for_k(hi)
        f = k - lo  # interpolation weight of the ceil(k) case
        a = {"by_op": {n: (1 - f) * a["by_op"][n] + f * b["by_op"][n] for n in a["by_op"]},
             "overhead_s": (1 - f) * a["overhead_s"] + f * b["overhead_s"]}
    ops_s = sum(a["by_op"].values())
    return {"engine": engine, "batch": B, "context_len": L, "passes_per_step": k,
            "by_op": a["by_op"], "ops_s": ops_s, "overhead_s": a["overhead_s"],
            "total_s": ops_s + a["overhead_s"]}


def decode_step_time(model: ModelConfig, cost, batch: int, context_len: int,
                     engine: str = "vllm", passes_per_step: float = 1.0, **kw) -> float:
    """Seconds for one decode step of `batch` sequences with `context_len` cached tokens."""
    return decode_step_breakdown(model, cost, batch, context_len, engine, passes_per_step, **kw)["total_s"]


# ---------------------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------------------
def prefill_passes(batch: int, input_len: int, chunk_tokens: Optional[int] = None) -> List[tuple]:
    """Forward passes of a prefill as (n_seqs, cached_len, new_tokens) tuples.

    chunk_tokens = None: one pass over all B x S tokens.
    chunk_tokens = C (vLLM max_num_batched_tokens with chunked prefill):
      * S <= C: whole prompts are packed, floor(C / S) per pass;
      * S >  C: each prompt is cut into pieces of C tokens, piece j attends to the
        j*C tokens already cached (one sequence per pass; simplification).
    """
    B, S = int(batch), int(input_len)
    if B < 1 or S < 1:
        raise ValueError("need batch >= 1 and input_len >= 1")
    if not chunk_tokens or B * S <= chunk_tokens:
        return [(B, 0, S)]
    C = int(chunk_tokens)
    if S <= C:
        per = C // S
        return [(len(g), 0, S) for g in [range(B)[i:i + per] for i in range(0, B, per)]]
    out = []
    for _ in range(B):
        for start in range(0, S, C):
            out.append((1, start, min(C, S - start)))
    return out


def prefill_breakdown(model: ModelConfig, cost, batch: int, input_len: int, *,
                      engine: str = "vllm", chunk_tokens: Optional[int] = None,
                      overhead_s: Optional[float] = None, w_bits: Optional[float] = None,
                      kv_bits: Optional[float] = None,
                      weight_bytes_override: Optional[float] = None) -> Dict:
    """Prefill of `batch` prompts of `input_len` tokens; produces the first token
    (the LM head runs for the last prompt token only)."""
    _check_engine(engine)
    oh = pass_overhead(cost, engine, overhead_s)
    passes = prefill_passes(batch, input_len, chunk_tokens)
    by_op: Dict[str, float] = {}
    for n, L, T in passes:
        for name, t in _pass(model, cost, n, L, T, engine, w_bits=w_bits, kv_bits=kv_bits,
                             weight_bytes_override=weight_bytes_override).items():
            by_op[name] = by_op.get(name, 0.0) + t
    ops_s = sum(by_op.values())
    return {"engine": engine, "batch": batch, "input_len": input_len, "passes": len(passes),
            "by_op": by_op, "ops_s": ops_s, "overhead_s": len(passes) * oh,
            "total_s": ops_s + len(passes) * oh}


def prefill_time(model: ModelConfig, cost, batch: int, input_len: int, **kw) -> float:
    """Seconds to prefill `batch` prompts of `input_len` tokens (~ TTFT of a static batch)."""
    return prefill_breakdown(model, cost, batch, input_len, **kw)["total_s"]


# ---------------------------------------------------------------------------------------
# vLLM `bench latency`
# ---------------------------------------------------------------------------------------
def predict_vllm_latency(model: ModelConfig, cost, batch: int, input_len: int, output_len: int, *,
                         chunk_tokens: Optional[int] = None, overhead_s: Optional[float] = None,
                         w_bits: Optional[float] = None, kv_bits: Optional[float] = None) -> float:
    """End-to-end latency of one static batch, comparable to `vllm bench latency`
    avg_latency: prefill (writes token 1) + (output_len - 1) decode steps, decode step i
    having input_len + i - 1 tokens cached (context grows by one per step)."""
    if output_len < 1:
        raise ValueError("output_len >= 1")
    kw = dict(overhead_s=overhead_s, w_bits=w_bits, kv_bits=kv_bits)
    t = prefill_time(model, cost, batch, input_len, engine="vllm", chunk_tokens=chunk_tokens, **kw)
    for i in range(1, output_len):
        t += decode_step_time(model, cost, batch, input_len + i - 1, "vllm", 1.0, **kw)
    return t


def predict_vllm_tpot(model: ModelConfig, cost, batch: int, input_len: int, output_len: int = 128,
                      **kw) -> float:
    """Model counterpart of the measured TPOT = (lat(out=OL) - lat(out=1)) / (OL - 1):
    the mean decode step time over contexts input_len .. input_len + OL - 2."""
    return (predict_vllm_latency(model, cost, batch, input_len, output_len, **kw)
            - predict_vllm_latency(model, cost, batch, input_len, 1, **kw)) / (output_len - 1)


# ---------------------------------------------------------------------------------------
# llama.cpp slot test (master vs slot-order fix)
# ---------------------------------------------------------------------------------------
def slot_coeffs(model: ModelConfig, cost, context_len: int = 512, **kw) -> tuple:
    """(a, c) of llamacpp_slots' t_step = k*a + B*c, from this cost model:
    c = t(B=2) - t(B=1) (per-sequence cost), a = t(B=1) - c (per-pass cost, incl. overhead).
    Exact when every op is memory-bound (op time affine in B), which holds for decode."""
    t1 = decode_step_time(model, cost, 1, context_len, "llamacpp", 1.0, **kw)
    t2 = decode_step_time(model, cost, 2, context_len, "llamacpp", 1.0, **kw)
    c = t2 - t1
    return t1 - c, c


def predict_llamacpp_slot_tps(model: ModelConfig, cost, users: int, passes_per_step: float = 1.0,
                              context_len: int = 512, **kw) -> float:
    """Aggregate decode tok/s with `users` busy slots and k = passes_per_step:
    users / t_step(users, k). Uses the same decode step as everything else."""
    return users / decode_step_time(model, cost, users, context_len, "llamacpp", passes_per_step, **kw)


def predict_llamacpp_master_vs_fix(model: ModelConfig, cost, users: int = 8,
                                   k_master: Optional[float] = None, k_fix: Optional[float] = None,
                                   context_len: int = 512, **kw) -> Dict:
    """Master vs slot-order fix at `users` busy slots. Default k values are the measured
    forward passes per generation step from data/llamacpp_rtx5090_qwen3_8b.yaml
    (4.34 before, 1.05 after). Also reports llamacpp_slots.predict_throughput with
    (a, c) from this cost model, which must agree with the direct step model."""
    if k_master is None or k_fix is None:
        c = llamacpp_slots.load_measurements()["counters"]["cpu_check_master"]
        k_master = c["forward_passes_per_generation_step_before"] if k_master is None else k_master
        k_fix = c["forward_passes_per_generation_step_after"] if k_fix is None else k_fix
    tm = predict_llamacpp_slot_tps(model, cost, users, k_master, context_len, **kw)
    tf = predict_llamacpp_slot_tps(model, cost, users, k_fix, context_len, **kw)
    a, c = slot_coeffs(model, cost, context_len, **kw)
    return {
        "users": users, "k_master": k_master, "k_fix": k_fix,
        "master_tok_s": tm, "fix_tok_s": tf, "speedup": tf / tm,
        "a_s": a, "c_s": c,
        "slots_formula_speedup": llamacpp_slots.predict_throughput(a, c, users, k_fix)
        / llamacpp_slots.predict_throughput(a, c, users, k_master),
    }
