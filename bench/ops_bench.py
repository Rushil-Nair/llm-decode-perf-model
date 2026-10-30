#!/usr/bin/env python3
"""Level 2 microbenchmarks: time Qwen3-8B's operators in isolation on the GPU.

    python bench/ops_bench.py --out data/<gpu_tag> [--quick] [--families gemm,rmsnorm]

Writes (CONTRACTS.md):
    data/<tag>/ops.jsonl            one record per (op, shape, implementation)
    data/<tag>/ops.jsonl.error.txt  every failure (OOM, missing kernel, crash), run continues
    data/<tag>/ops_meta.json        torch/CUDA/device facts and the timing settings used

Record: {"op","dtype","shape":{...},"us","flops","bytes",
         "impl","us_min","us_max","n_iter", (gemm) "weight_copies",
         (attention) "finite" (output has no NaN/Inf), "rejected" (the backend/GQA-mode
         combinations that refused this shape, with the reason)}

Method (the AIConfigurator idea: measure ops in isolation on silicon, then model
everything else as speed-of-light x an empirical efficiency):
  * Timing: CUDA events around ONLY the op. `warmup` untimed calls first, then
    n_iter timed calls; we report the MEDIAN (robust to the odd clock/IRQ hiccup)
    plus min and max.
  * Cold cache: before every timed call we write a buffer of max(2 x L2, 256 MiB)
    (`flush_buf.zero_()`, the same trick as triton.testing.do_bench) so the op
    reads its inputs from DRAM, as it does inside a real forward pass where the
    other layers' weights have evicted it. The flush runs before the start event,
    so it is not timed.
  * Weight rotation (GEMM): if one weight matrix is < 4 x L2 we allocate
    ceil(4 x L2 / weight_bytes) copies (max 64) and iteration i uses copy i mod n,
    so even without the flush consecutive iterations never hit a warm L2.
  * Every available implementation is measured (SDPA backends; compiled / eager
    RMSNorm). bench/fit_calib.py keeps the FASTEST per shape, i.e. we model an
    engine that uses the best kernel. `impl` says which one won.
  * Because the flush keeps the GPU busy, the CPU enqueues the op long before the
    GPU reaches it: the events measure GPU execution time only (no CPU launch
    latency). That matches engines that replay CUDA graphs (vLLM, llama.cpp decode).
  * Isolation: each op family runs in its own Python process (--worker). A sticky
    CUDA error (illegal address) or a hang in one family cannot take down the
    others; the parent logs it and moves on.

Analytic FLOPs / bytes (same conventions as perfsim/models.py, e = 2 bytes for bf16):
  gemm  y[M,N] = x[M,K] @ W[K,N]:
      flops = 2*M*K*N                      (a multiply-accumulate = 2 FLOPs)
      bytes = (M*K + K*N + M*N) * e        (read x, read W once, write y)
  attention (flash-style, the score matrix never touches DRAM), Hq query heads,
  Hkv KV heads, head dim D, B sequences, q_len queries, kv_len keys each:
      pairs = B*q_len*kv_len               (decode, non-causal)
            = B*S*(S+1)/2                  (causal prefill, q_len = kv_len = S)
      flops = pairs * Hq * (4*D + 5)       (QK^T 2D + AV 2D + 5 for softmax)
      bytes = (2*B*q_len*Hq*D              (read Q, write O)
               + 2*B*kv_len*Hkv*D) * e     (read K and V once: GQA shares them)
      Note: if the backend needs K/V expanded to Hq heads (no native GQA), the
      kernel really reads Hq/Hkv x more K/V bytes; we still record the GQA bytes,
      so that penalty shows up as lower efficiency (impl says "+expand_kv").
  rmsnorm  y = x * rsqrt(mean(x^2) + eps) * w over T tokens x H:
      flops = 4*T*H
      bytes = (2*T*H + H) * e              (read x, write y, read w)

Probes: one deliberately tiny instance per family (shape has "probe": true). Its
time is almost pure fixed per-kernel cost; perfsim/ops_model.py uses it as the
launch-overhead term t0 and excludes probes from the efficiency curves.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_CONFIG = REPO / "configs" / "qwen3-8b.json"

DTYPE = "bf16"
ELEM_BYTES = 2  # bf16
FAMILIES = ("rmsnorm", "gemm", "attn_prefill", "attn_decode")

GEMM_M_FULL = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
GEMM_M_QUICK = (1, 8, 64, 512, 4096)
DECODE_BATCH_FULL, DECODE_KV_FULL = (1, 4, 16, 64), (512, 2048, 8192, 32768)
DECODE_BATCH_QUICK, DECODE_KV_QUICK = (1, 16), (512, 8192)
PREFILL_SEQ_FULL, PREFILL_SEQ_QUICK = (512, 2048, 8192), (512, 2048)
NORM_TOKENS = (1, 64, 4096)

MAX_WEIGHT_COPIES = 64
MIN_FLUSH_BYTES = 256 * 2**20


# ----------------------------------------------------------------------------------
# Analytic costs (pure Python; unit-tested and used by the synthetic test fixtures)
# ----------------------------------------------------------------------------------
def gemm_cost(M: int, K: int, N: int, e: int = ELEM_BYTES):
    """(flops, bytes) of y[M,N] = x[M,K] @ W[K,N]."""
    return 2.0 * M * K * N, float((M * K + K * N + M * N) * e)


def attn_cost(batch: int, q_len: int, kv_len: int, n_q_heads: int, n_kv_heads: int,
              head_dim: int, causal: bool, e: int = ELEM_BYTES):
    """(flops, bytes) of flash-style GQA attention (see module docstring)."""
    if causal:
        if q_len != kv_len:
            raise ValueError("causal cost is defined for prefill with q_len == kv_len")
        pairs = batch * q_len * (q_len + 1) / 2.0
    else:
        pairs = float(batch * q_len * kv_len)
    flops = pairs * n_q_heads * (4.0 * head_dim + 5.0)
    nbytes = (2 * batch * q_len * n_q_heads * head_dim
              + 2 * batch * kv_len * n_kv_heads * head_dim) * e
    return flops, float(nbytes)


def rmsnorm_cost(tokens: int, hidden: int, e: int = ELEM_BYTES):
    return 4.0 * tokens * hidden, float((2 * tokens * hidden + hidden) * e)


def load_model_dims(path: Path = MODEL_CONFIG) -> dict:
    """The few config fields we need, read straight from the HF config.json
    (no perfsim import, so this script needs only torch on the GPU box)."""
    with open(path) as f:
        c = json.load(f)
    n_heads = int(c["num_attention_heads"])
    hidden = int(c["hidden_size"])
    return {
        "hidden": hidden,
        "n_heads": n_heads,
        "n_kv_heads": int(c.get("num_key_value_heads", n_heads)),
        "head_dim": int(c.get("head_dim") or hidden // n_heads),
        "intermediate": int(c["intermediate_size"]),
        "vocab": int(c["vocab_size"]),
    }


def projection_shapes(d: dict) -> dict:
    """name -> (K, N) for every weight GEMM of one layer + the LM head.
    q,k,v,o,gate,up,down as separate matrices, plus the fused qkv and gate_up
    matrices that vLLM actually runs (QKVParallelLinear / MergedColumnParallelLinear),
    which are also what perfsim.models.layer_ops emits."""
    h, I = d["hidden"], d["intermediate"]
    q, kv = d["n_heads"] * d["head_dim"], d["n_kv_heads"] * d["head_dim"]
    return {
        "q": (h, q), "k": (h, kv), "v": (h, kv), "o": (q, h),
        "gate": (h, I), "up": (h, I), "down": (I, h),
        "qkv": (h, q + 2 * kv), "gate_up": (h, 2 * I),
        "lm_head": (h, d["vocab"]),
    }


def plan(quick: bool = False, families=FAMILIES, dims: dict | None = None) -> list:
    """Every (op, shape) to measure, as dicts {"op", "shape"}; probes included."""
    d = dims or load_model_dims()
    hq, hkv, D, H = d["n_heads"], d["n_kv_heads"], d["head_dim"], d["hidden"]
    specs = []
    if "rmsnorm" in families:
        specs.append({"op": "rmsnorm", "shape": {"tokens": 1, "hidden": 128, "probe": True}})
        specs += [{"op": "rmsnorm", "shape": {"tokens": t, "hidden": H}} for t in NORM_TOKENS]
    if "gemm" in families:
        specs.append({"op": "gemm", "shape": {"proj": "probe", "M": 1, "K": 128, "N": 128,
                                              "weight_bytes": 128 * 128 * ELEM_BYTES, "probe": True}})
        for name, (K, N) in projection_shapes(d).items():
            for M in (GEMM_M_QUICK if quick else GEMM_M_FULL):
                specs.append({"op": "gemm", "shape": {"proj": name, "M": M, "K": K, "N": N,
                                                      "weight_bytes": K * N * ELEM_BYTES}})

    def attn(op, b, ql, kvl, nq, nkv, causal, probe=False):
        s = {"batch": b, "q_len": ql, "kv_len": kvl, "n_q_heads": nq, "n_kv_heads": nkv,
             "head_dim": D, "causal": causal}
        if probe:
            s["probe"] = True
        return {"op": op, "shape": s}

    if "attn_prefill" in families:
        # probe: 4 q heads / 1 kv head keeps the GQA ratio but makes the work negligible
        specs.append(attn("attn_prefill", 1, 16, 16, 4, 1, True, probe=True))
        specs += [attn("attn_prefill", 1, s, s, hq, hkv, True)
                  for s in (PREFILL_SEQ_QUICK if quick else PREFILL_SEQ_FULL)]
    if "attn_decode" in families:
        specs.append(attn("attn_decode", 1, 1, 16, 4, 1, False, probe=True))
        for b in (DECODE_BATCH_QUICK if quick else DECODE_BATCH_FULL):
            for kvl in (DECODE_KV_QUICK if quick else DECODE_KV_FULL):
                specs.append(attn("attn_decode", b, 1, kvl, hq, hkv, False))
    return specs


def spec_cost(spec: dict):
    """(flops, bytes) for a plan entry."""
    s, op = spec["shape"], spec["op"]
    if op == "gemm":
        return gemm_cost(s["M"], s["K"], s["N"])
    if op in ("attn_decode", "attn_prefill"):
        return attn_cost(s["batch"], s["q_len"], s["kv_len"], s["n_q_heads"], s["n_kv_heads"],
                         s["head_dim"], s["causal"])
    if op == "rmsnorm":
        return rmsnorm_cost(s["tokens"], s["hidden"])
    raise ValueError(f"unknown op {op}")


# ----------------------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------------------
class Writer:
    """Appends records / errors line by line (flushed), so a crash loses nothing."""

    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ops_path = self.out / "ops.jsonl"
        self.err_path = self.out / "ops.jsonl.error.txt"
        self.n_ok = self.n_err = 0

    def record(self, rec: dict) -> None:
        with open(self.ops_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.n_ok += 1

    def error(self, where: str, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.err_path, "a") as f:
            f.write(f"[{stamp}] {where}\n{msg.rstrip()}\n\n")
        self.n_err += 1
        print(f"  ERROR {where}: {msg.strip().splitlines()[-1] if msg.strip() else msg}", flush=True)


# ----------------------------------------------------------------------------------
# GPU side (torch imported lazily so the analytic part above imports anywhere)
# ----------------------------------------------------------------------------------
class Timer:
    """CUDA-event timing with warmup, L2 flush before every timed call, median."""

    def __init__(self, torch, l2_bytes: int, warmup: int, iters: int, min_iters: int,
                 budget_s: float):
        self.torch = torch
        self.l2_bytes = int(l2_bytes)
        self.flush_bytes = int(max(2 * self.l2_bytes, MIN_FLUSH_BYTES))
        self.flush_buf = torch.empty(self.flush_bytes, dtype=torch.int8, device="cuda")
        self.warmup, self.iters, self.min_iters, self.budget_s = warmup, iters, min_iters, budget_s

    def flush(self) -> None:
        self.flush_buf.zero_()

    def measure(self, fn) -> dict:
        """fn(i) launches the op for iteration i (i selects a rotated weight copy)."""
        torch = self.torch
        for i in range(self.warmup):
            fn(i)
        torch.cuda.synchronize()
        # one cold timed call to size the iteration count to the time budget
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.flush()
        s.record()
        fn(0)
        e.record()
        e.synchronize()
        est_s = max(s.elapsed_time(e) * 1e-3, 1e-7)
        n = int(min(self.iters, max(self.min_iters, self.budget_s / est_s)))
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            self.flush()          # evict L2 (not timed: it is before the start event)
            starts[i].record()
            fn(i)
            ends[i].record()
        torch.cuda.synchronize()
        us = sorted(starts[i].elapsed_time(ends[i]) * 1e3 for i in range(n))  # ms -> us
        return {"us": statistics.median(us), "us_min": us[0], "us_max": us[-1], "n_iter": n}


def _record(spec, timing, impl, **extra) -> dict:
    flops, nbytes = spec_cost(spec)
    rec = {"op": spec["op"], "dtype": DTYPE, "shape": spec["shape"], "us": timing["us"],
           "flops": flops, "bytes": nbytes, "impl": impl,
           "us_min": timing["us_min"], "us_max": timing["us_max"], "n_iter": timing["n_iter"]}
    rec.update(extra)
    return rec


def _free_bytes(torch) -> int:
    try:
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return 1 << 62  # unknown: let the allocator decide (OOM is caught anyway)


def _check_fits(torch, need_bytes: float, what: str) -> None:
    free = _free_bytes(torch)
    if need_bytes > 0.85 * free:
        raise MemoryError(f"skipped {what}: needs {need_bytes / 1e9:.2f} GB, "
                          f"{free / 1e9:.2f} GB free")


# ---- gemm ------------------------------------------------------------------------
def run_gemm(torch, timer: Timer, spec: dict, w: Writer) -> None:
    s = spec["shape"]
    M, K, N = s["M"], s["K"], s["N"]
    wbytes = K * N * ELEM_BYTES
    copies = 1
    if wbytes < 4 * timer.l2_bytes:
        copies = min(MAX_WEIGHT_COPIES, math.ceil(4 * timer.l2_bytes / wbytes))
    _check_fits(torch, (copies * K * N + M * K + M * N) * ELEM_BYTES, f"gemm {s}")
    W = torch.randn(copies, K, N, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def fn(i):
        torch.matmul(x, W[i % copies], out=y)

    t = timer.measure(fn)
    w.record(_record(spec, t, "torch.matmul", weight_copies=copies))


# ---- attention -------------------------------------------------------------------
def sdpa_backends(torch):
    """[(name, context-manager factory)] restricting SDPA to ONE backend, in the
    order we try them. Uses torch.nn.attention.sdpa_kernel (torch >= 2.3) or the
    older torch.backends.cuda.sdp_kernel."""
    out = []
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
            b = getattr(SDPBackend, name, None)
            if b is not None:
                out.append((name.lower(), lambda b=b: sdpa_kernel(b)))
    except ImportError:
        ctx = torch.backends.cuda.sdp_kernel
        out = [("flash_attention", lambda: ctx(enable_flash=True, enable_math=False, enable_mem_efficient=False)),
               ("efficient_attention", lambda: ctx(enable_flash=False, enable_math=False, enable_mem_efficient=True)),
               ("math", lambda: ctx(enable_flash=False, enable_math=True, enable_mem_efficient=False))]
    return out


def run_attention(torch, timer: Timer, spec: dict, w: Writer) -> None:
    """Measure every fused SDPA backend that accepts this shape; MATH only if none does.
    For each backend: native GQA (enable_gqa=True, torch >= 2.5) first, else K/V
    expanded to Hq heads with repeat_interleave (done once, outside the timing)."""
    import torch.nn.functional as F
    s = spec["shape"]
    B, ql, kvl, hq, hkv, D, causal = (s["batch"], s["q_len"], s["kv_len"], s["n_q_heads"],
                                       s["n_kv_heads"], s["head_dim"], s["causal"])
    group = hq // hkv
    e = ELEM_BYTES
    base = (2 * B * ql * hq * D + 2 * B * kvl * hkv * D) * e          # q, out, k, v
    _check_fits(torch, base, f"{spec['op']} {s}")
    q = torch.randn(B, hq, ql, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, hkv, kvl, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, hkv, kvl, D, dtype=torch.bfloat16, device="cuda")
    expanded = {}

    def kv_expanded():
        if "kv" not in expanded:
            _check_fits(torch, 2 * B * kvl * hq * D * e, f"expand_kv {s}")
            expanded["kv"] = (k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1))
        return expanded["kv"]

    def call(gqa_mode):
        if gqa_mode == "native":
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        ke, ve = kv_expanded()
        return F.scaled_dot_product_attention(q, ke, ve, is_causal=causal)

    modes = ("native", "expand") if group > 1 else ("expand",)  # expand == plain MHA if group 1
    tried_errors, measured, results = [], 0, []
    backends = sdpa_backends(torch)
    fused = [b for b in backends if b[0] != "math"]
    math_ = [b for b in backends if b[0] == "math"]
    for phase_backends in (fused, math_):
        if measured:
            break  # MATH only as a last resort
        for name, ctx in phase_backends:
            for mode in modes:
                try:
                    with ctx():
                        out = call(mode)          # does this backend accept the shape?
                        torch.cuda.synchronize()
                        finite = bool(torch.isfinite(out).all().item())
                        del out
                        t = timer.measure(lambda i: call(mode))
                    impl = f"sdpa:{name}" + ("+native_gqa" if mode == "native" and group > 1
                                             else "+expand_kv" if group > 1 else "")
                    results.append(_record(spec, t, impl, finite=finite))
                    measured += 1
                    break  # native worked: no need to also time the expanded variant
                except Exception as ex:  # "No available kernel", TypeError (old torch), OOM ...
                    first_line = (str(ex).strip().splitlines() or [""])[0][:200]
                    tried_errors.append(f"{name}/{mode}: {type(ex).__name__}: {first_line}")
                    if isinstance(ex, getattr(torch.cuda, "OutOfMemoryError", MemoryError)):
                        torch.cuda.empty_cache()
    if not measured:
        raise RuntimeError("no SDPA backend ran this shape:\n  " + "\n  ".join(tried_errors))
    for rec in results:  # every record says which backends/modes refused this shape, and why
        rec["rejected"] = tried_errors
        w.record(rec)


# ---- rmsnorm ---------------------------------------------------------------------
def rmsnorm_impls(torch):
    """[(name, fn(x, w))]: a torch.compile'd single fused kernel (closest to the
    fused CUDA RMSNorm vLLM uses), F.rms_norm, and the eager composite (several
    kernels) as fallbacks. Math follows HF Qwen3RMSNorm: fp32 statistics."""
    import torch.nn.functional as F
    eps = 1e-6

    def ref(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return w * xf.to(x.dtype)

    impls = []
    if hasattr(torch, "compile"):
        impls.append(("torch.compile", torch.compile(ref, dynamic=False)))
    if hasattr(F, "rms_norm"):
        impls.append(("F.rms_norm", lambda x, w: F.rms_norm(x, (x.shape[-1],), w, eps)))
    impls.append(("eager_composite", ref))
    return impls


def run_rmsnorm(torch, timer: Timer, spec: dict, w: Writer, impls: list, broken: set) -> None:
    s = spec["shape"]
    x = torch.randn(s["tokens"], s["hidden"], dtype=torch.bfloat16, device="cuda")
    g = torch.randn(s["hidden"], dtype=torch.bfloat16, device="cuda")
    measured = 0
    for name, fn in impls:
        if name in broken:
            continue
        try:
            t = timer.measure(lambda i: fn(x, g))   # warmup includes torch.compile's compile
            w.record(_record(spec, t, name))
            measured += 1
        except Exception:
            broken.add(name)  # e.g. no C compiler for Triton: do not retry on every shape
            w.error(f"rmsnorm impl {name} (disabled for the rest of the run)", traceback.format_exc())
    if not measured:
        raise RuntimeError("no RMSNorm implementation ran")


# ----------------------------------------------------------------------------------
# Worker (one op family, one process) and parent
# ----------------------------------------------------------------------------------
def _cuda_healthy(torch) -> bool:
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda").add_(1).item()
        return True
    except Exception:
        return False


def _write_meta(out: Path, update: dict) -> None:
    path = out / "ops_meta.json"
    meta = {}
    if path.exists():
        try:
            meta = json.loads(path.read_text())
        except Exception:
            meta = {}
    fam = update.pop("family_info", None)
    meta.update(update)
    if fam:
        meta.setdefault("families", {}).update(fam)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def run_worker(family: str, args) -> int:
    out = Path(args.out)
    w = Writer(out)
    import torch
    if not torch.cuda.is_available():
        w.error(f"worker {family}", "torch.cuda.is_available() is False")
        return 2
    props = torch.cuda.get_device_properties(0)
    l2 = int(args.l2_bytes or getattr(props, "L2_cache_size", 0) or 0)
    l2_note = "from torch device properties" if not args.l2_bytes else "from --l2-bytes"
    if l2 <= 0:
        l2, l2_note = 128 * 2**20, "UNKNOWN (torch lacks L2_cache_size): assumed 128 MiB"
    quick = args.quick
    timer = Timer(torch, l2, warmup=3 if quick else 10, iters=args.iters or (20 if quick else 100),
                  min_iters=5 if quick else 10, budget_s=0.05 if quick else 0.3)
    def safe(f):  # metadata must never stop the benchmark
        try:
            return f()
        except Exception as ex:
            return f"unavailable ({type(ex).__name__})"

    _write_meta(out, {
        "torch": torch.__version__, "cuda": safe(lambda: torch.version.cuda),
        "cudnn": safe(lambda: torch.backends.cudnn.version()),
        "device": safe(lambda: torch.cuda.get_device_name(0)),
        "capability": safe(lambda: list(torch.cuda.get_device_capability(0))),
        "sm_count": safe(lambda: props.multi_processor_count), "l2_bytes": l2, "l2_source": l2_note,
        "flush_bytes": timer.flush_bytes, "warmup": timer.warmup, "max_iters": timer.iters,
        "min_iters": timer.min_iters, "budget_s_per_op": timer.budget_s, "quick": quick,
        "timing": "CUDA events around the op only; L2 flushed before every timed call; median reported",
    })
    specs = plan(quick, families=(family,))
    print(f"[{family}] {len(specs)} shapes, L2 {l2 / 2**20:.0f} MiB, flush {timer.flush_bytes / 2**20:.0f} MiB",
          flush=True)
    rms_impls, rms_broken = (rmsnorm_impls(torch), set()) if family == "rmsnorm" else (None, None)
    for n, spec in enumerate(specs):
        label = f"{spec['op']} {json.dumps(spec['shape'], sort_keys=True)}"
        print(f"[{family}] {n + 1}/{len(specs)} {label}", flush=True)
        try:
            if family == "gemm":
                run_gemm(torch, timer, spec, w)
            elif family in ("attn_decode", "attn_prefill"):
                run_attention(torch, timer, spec, w)
            elif family == "rmsnorm":
                run_rmsnorm(torch, timer, spec, w, rms_impls, rms_broken)
        except Exception:
            w.error(label, traceback.format_exc())
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if not _cuda_healthy(torch):
            w.error(f"worker {family}", "CUDA context is no longer usable (sticky error); "
                                        "abandoning the rest of this family")
            return 3
    _write_meta(out, {"family_info": {family: {"records": w.n_ok, "errors": w.n_err}}})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output dir, e.g. data/h100_sxm")
    ap.add_argument("--quick", action="store_true", help="few shapes, few iterations (smoke test)")
    ap.add_argument("--families", default=",".join(FAMILIES), help=f"subset of {','.join(FAMILIES)}")
    ap.add_argument("--iters", type=int, default=0, help="max timed iterations per op (default 100 / quick 20)")
    ap.add_argument("--l2-bytes", type=int, default=0, help="override the L2 size torch reports")
    ap.add_argument("--family-timeout", type=float, default=3600.0, help="seconds per family process")
    ap.add_argument("--inprocess", action="store_true", help="run all families in this process (debug)")
    ap.add_argument("--append", action="store_true",
                    help="keep existing ops.jsonl (e.g. re-running one family that crashed)")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            return run_worker(args.worker, args)
        except Exception:
            Writer(out).error(f"worker {args.worker} crashed", traceback.format_exc())
            return 1

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        ap.error(f"unknown families {bad}; choose from {FAMILIES}")
    if not args.append:  # fresh run: start empty files (old errors would be misleading)
        for name in ("ops.jsonl", "ops.jsonl.error.txt", "ops_meta.json"):
            (out / name).unlink(missing_ok=True)
    w = Writer(out)
    t_start = time.time()
    for fam in families:
        print(f"=== {fam} ===", flush=True)
        if args.inprocess:
            try:
                rc = run_worker(fam, args)
            except Exception:
                w.error(f"family {fam} crashed", traceback.format_exc())
                rc = 1
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--out", str(out), "--worker", fam,
                   "--families", fam]
            if args.quick:
                cmd.append("--quick")
            if args.iters:
                cmd += ["--iters", str(args.iters)]
            if args.l2_bytes:
                cmd += ["--l2-bytes", str(args.l2_bytes)]
            try:
                rc = subprocess.run(cmd, timeout=args.family_timeout).returncode
            except subprocess.TimeoutExpired:
                w.error(f"family {fam}", f"timed out after {args.family_timeout:.0f} s (killed)")
                rc = -1
        if rc not in (0, None):
            w.error(f"family {fam}", f"worker exited with code {rc}; continuing with the next family")
    n = 0
    if (out / "ops.jsonl").exists():
        with open(out / "ops.jsonl") as f:
            n = sum(1 for line in f if line.strip())
    err = out / "ops.jsonl.error.txt"
    print(f"done in {time.time() - t_start:.0f} s: {n} records in {out / 'ops.jsonl'}"
          + (f"; errors logged in {err}" if err.exists() else ""), flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
