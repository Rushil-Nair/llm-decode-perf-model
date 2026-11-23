"""Benchmark paged GQA decode attention: our Triton split-KV kernel vs FlashInfer vs PyTorch SDPA.

Run on the GPU box (see gpu/run_attention.sh):

    python3 bench/bench_attention.py --out data/<gpu_tag>/attention          # main matrix
    python3 bench/bench_attention.py --mode splits --out ...                  # split-count sweep
    python3 bench/bench_attention.py --mode ncu --shape llama8b,1,32768       # one shape, eager, for Nsight Compute
    python3 bench/bench_attention.py --mode tune --out ...                    # pick version-2 settings + ablation

Tuning uses its own shapes (batch 2/8/32, context 1k/4k/16k/64k), none of which is in the main matrix
(batch 1/4/16/64, context 512/2k/8k/32k/128k), so the matrix scores settings it was not tuned on.

Method (each choice is there to keep the comparison fair):
  * Every kernel is checked against the PyTorch reference / FlashInfer before it is timed.
  * Each kernel is captured in a CUDA graph and replayed; launch overhead would otherwise swamp
    microsecond decode kernels. If capture fails the row says mode=eager.
  * FlashInfer's plan() and our plan() are outside the timed region (per-step metadata shared by
    all layers). Their cost is timed separately (plan_us).
  * L2 is flushed before every replay by *reading* a 512 MB buffer (a zero_() flush leaves dirty
    lines whose write-back lands in the next kernel's time). Timing events bracket only the replay.
  * FlashInfer runs with use_tensor_cores False and True; both are recorded.
  * SDPA is recorded two ways: on K/V already gathered into contiguous tensors (no paging cost,
    favourable to SDPA) and gather + SDPA (what a paged engine would pay).
  * Kernels run in a different order for each shape; medians over --iters replays, p10/p90 kept.
  * Effective bandwidth = (K + V bytes read + q/o bytes) / time, reported as GB/s and as a fraction
    of this GPU's measured device-to-device copy bandwidth.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from attn.paged import make_paged_kv  # noqa: E402
from attn.reference import paged_decode_reference  # noqa: E402
from attn.triton_decode import (BalancedTritonPagedDecode, BalancedTritonPagedDecodeV2,  # noqa: E402
                                 TritonPagedDecode, choose_num_splits)

try:
    import flashinfer
except Exception:  # noqa: BLE001
    flashinfer = None

# (num_qo_heads, num_kv_heads, head_dim) per layer, from each model's config.json
CONFIGS = {
    "llama8b": (32, 8, 128),    # Llama-3.1-8B, Qwen3-8B: G = 4
    "qwen7b": (28, 4, 128),     # Qwen2.5-7B: G = 7 (not a power of two)
    "tp8_70b": (8, 1, 128),     # Llama-3.1-70B at tensor parallel 8: one KV head per GPU
    "mha": (32, 32, 128),       # no GQA, control
}


def gpu_env(copy_gbps):
    p = torch.cuda.get_device_properties(0)
    env = dict(gpu=p.name, sms=p.multi_processor_count, mem_gb=round(p.total_memory / 2**30, 1),
               torch=torch.__version__, cuda=torch.version.cuda, copy_gbps=copy_gbps,
               flashinfer=getattr(flashinfer, "__version__", None))
    import triton
    env["triton"] = triton.__version__
    try:
        q = "name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,power.limit,temperature.gpu"
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"],
                                           capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        env["nvidia_smi"] = f"unavailable: {e}"
    return env


def measure_copy_gbps(nbytes=2 << 30, iters=20):
    """Device-to-device copy bandwidth (read + write bytes / time): the practical memory ceiling."""
    a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); b.copy_(a); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    ts.sort()
    del a, b
    torch.cuda.empty_cache()
    return 2 * nbytes / ts[len(ts) // 2] / 1e9


class Timer:
    def __init__(self, flush_bytes=512 << 20):
        self.flush_buf = torch.ones(flush_bytes // 4, dtype=torch.float32, device="cuda")

    def flush(self):
        self.flush_buf.sum()  # read-only sweep of a buffer much larger than L2

    def time(self, fn, iters, use_graph=True):
        """Median/p10/p90 microseconds of fn(). Returns (stats, mode)."""
        mode = "eager"
        call = fn
        if use_graph:
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                torch.cuda.synchronize()
                call, mode = g.replay, "graph"
            except Exception as e:  # noqa: BLE001
                print(f"    graph capture failed ({type(e).__name__}: {e}); timing eager", flush=True)
                torch.cuda.synchronize()
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        evs = []
        for _ in range(iters):
            self.flush()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); call(); e.record()
            evs.append((s, e))
        torch.cuda.synchronize()
        us = sorted(s.elapsed_time(e) * 1e3 for s, e in evs)
        pick = lambda f: us[min(len(us) - 1, int(f * len(us)))]  # noqa: E731
        return dict(median_us=pick(0.5), p10_us=pick(0.1), p90_us=pick(0.9)), mode


def seq_lens_for(batch, ctx, skew):
    if not skew:
        return [ctx] * batch
    # one long request and many short ones: where per-request split balancing matters
    return [ctx] + [max(64, ctx // 16)] * (batch - 1)


def build_case(cfg, batch, ctx, page_size, dtype, skew, seed=0):
    hq, hkv, d = CONFIGS[cfg]
    lens = seq_lens_for(batch, ctx, skew)
    k, v, indptr, indices, last = make_paged_kv(lens, hkv, d, page_size, dtype, "cuda", seed=seed)
    q = torch.randn(batch, hq, d, device="cuda", dtype=dtype)
    elt = torch.finfo(dtype).bits // 8
    kv_bytes = 2 * sum(lens) * hkv * d * elt
    qo_bytes = 2 * batch * hq * d * elt
    return dict(q=q, k=k, v=v, indptr=indptr, indices=indices, last=last, lens=lens,
                hq=hq, hkv=hkv, d=d, bytes=kv_bytes + qo_bytes)


def make_kernels(c, page_size, dtype, splits_override=None, fi_tensor_cores=(False, True), sdpa=True,
                 v2_cfg=None, triton=True):
    """name -> (fn, plan_seconds, extra dict). Every fn returns the attention output."""
    out = {}
    if triton:
        _add_triton(out, c, page_size, dtype, splits_override, v2_cfg)
    _add_flashinfer(out, c, page_size, dtype, fi_tensor_cores)
    if sdpa:
        _add_sdpa(out, c, page_size)
    return out


def _add_triton(out, c, page_size, dtype, splits_override, v2_cfg):
    q, k, v = c["q"], c["k"], c["v"]
    t0 = time.perf_counter()
    tri = TritonPagedDecode().plan(c["indptr"], c["indices"], c["last"], c["hq"], c["hkv"], c["d"],
                                   page_size, max(c["lens"]), dtype, num_splits=splits_override)
    torch.cuda.synchronize()
    out["triton_splitkv"] = (lambda: tri.run(q, k, v)[0], time.perf_counter() - t0,
                             dict(splits=tri.num_splits))
    if splits_override is None:
        t0 = time.perf_counter()
        bal = BalancedTritonPagedDecode().plan(c["indptr"], c["indices"], c["last"], c["hq"], c["hkv"],
                                               c["d"], page_size, q_dtype=dtype)
        torch.cuda.synchronize()
        out["triton_balanced"] = (lambda: bal.run(q, k, v)[0], time.perf_counter() - t0,
                                  dict(splits=bal.num_work, chunk=bal.chunk))
        # first plan compiles the kernel for the occupancy query (cached); time the second one
        bal2 = _plan_balanced(c, page_size, dtype, BalancedTritonPagedDecodeV2(**(v2_cfg or {})))
        t0 = time.perf_counter()
        bal2 = _plan_balanced(c, page_size, dtype, BalancedTritonPagedDecodeV2(**(v2_cfg or {})))
        torch.cuda.synchronize()
        out["triton_balanced_v2"] = (lambda: bal2.run(q, k, v)[0], time.perf_counter() - t0, bal2.describe())


def _add_flashinfer(out, c, page_size, dtype, fi_tensor_cores):
    q, k, v = c["q"], c["k"], c["v"]
    if flashinfer is not None:
        for tc in fi_tensor_cores:
            ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
            try:
                # use_cuda_graph=True with fixed index buffers is FlashInfer's documented way to make
                # run() capturable; fall back to the plain wrapper if this version rejects it.
                try:
                    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        ws, "NHD", use_cuda_graph=True, use_tensor_cores=tc,
                        paged_kv_indptr_buffer=c["indptr"], paged_kv_indices_buffer=c["indices"],
                        paged_kv_last_page_len_buffer=c["last"])
                except TypeError:
                    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
                t0 = time.perf_counter()
                w.plan(c["indptr"], c["indices"], c["last"], c["hq"], c["hkv"], c["d"], page_size,
                       q_data_type=dtype, kv_data_type=dtype)
                torch.cuda.synchronize()
                out[f"flashinfer_tc{int(tc)}"] = ((lambda w=w: w.run(q, (k, v))),
                                                  time.perf_counter() - t0, {})
            except Exception as e:  # noqa: BLE001
                print(f"    flashinfer tc={tc} unavailable: {type(e).__name__}: {e}", flush=True)


def _add_sdpa(out, c, page_size):
    q, k, v = c["q"], c["k"], c["v"]
    if len(set(c["lens"])) == 1:
        import torch.nn.functional as F
        L, ps = c["lens"][0], page_size
        # token index of every cached token of every request in the flat [pages*page_size] cache
        pages = c["indices"].long().view(len(c["lens"]), -1)                          # [B, P]
        tok = (pages[:, :, None] * ps + torch.arange(ps, device="cuda")).reshape(len(c["lens"]), -1)[:, :L]
        kf = k.view(-1, c["hkv"], c["d"])
        vf = v.view(-1, c["hkv"], c["d"])
        q4 = q[:, :, None, :]                                                          # [B, Hq, 1, D]

        def gather():
            return kf[tok].transpose(1, 2), vf[tok].transpose(1, 2)                   # [B, Hkv, L, D]
        kc, vc = (t.contiguous() for t in gather())
        out["sdpa_contig"] = (lambda: F.scaled_dot_product_attention(q4, kc, vc, enable_gqa=True), 0.0,
                              dict(note="K/V pre-gathered; paging cost excluded"))

        def gather_sdpa():
            kg, vg = gather()
            return F.scaled_dot_product_attention(q4, kg, vg, enable_gqa=True)
        out["sdpa_gather"] = (gather_sdpa, 0.0, dict(note="gather pages + SDPA"))


def _plan_balanced(c, page_size, dtype, ker):
    return ker.plan(c["indptr"], c["indices"], c["last"], c["hq"], c["hkv"], c["d"], page_size, q_dtype=dtype)


def check(c, kernels):
    """Max abs error of each kernel vs the reference (on small cases) or vs FlashInfer."""
    small = sum(c["lens"]) <= 64 * 1024
    if small:
        ref, _ = paged_decode_reference(c["q"], c["k"], c["v"], c["indptr"], c["indices"], c["last"])
    else:
        name = next((n for n in kernels if n.startswith("flashinfer")), None)
        if name is None:
            return {n: None for n in kernels}
        ref = kernels[name][0]()
    errs = {}
    for n, (fn, _, _) in kernels.items():
        o = fn()
        o = o.reshape(ref.shape) if o.numel() == ref.numel() else o.squeeze(2)
        errs[n] = float((o.float() - ref.float()).abs().max())
    return errs


def run_matrix(args, timer, copy_gbps, rows):
    dtype = getattr(torch, args.dtype)
    shapes = []
    for cfg in args.configs:
        for b in args.batches:
            for ctx in args.ctx:
                shapes.append((cfg, b, ctx, False))
            if b > 1:
                shapes += [(cfg, b, ctx, True) for ctx in args.ctx if ctx >= 2048]
    for ctx in args.long_ctx:
        shapes += [(cfg, 1, ctx, False) for cfg in args.configs]
    for cfg, b, ctx, skew in shapes:
        hq, hkv, d = CONFIGS[cfg]
        est = 2 * b * ctx * hkv * d * (torch.finfo(dtype).bits // 8)
        if est > args.max_cache_gb * 2**30:
            print(f"skip {cfg} B={b} ctx={ctx}: KV cache {est / 2**30:.1f} GB > budget", flush=True)
            continue
        c = build_case(cfg, b, ctx, args.page_size, dtype, skew)
        kernels = make_kernels(c, args.page_size, dtype, v2_cfg=args.v2_cfg)
        errs = check(c, kernels)
        order = list(kernels)
        random.Random(f"{cfg}-{b}-{ctx}-{skew}").shuffle(order)
        for name in order:
            fn, plan_s, extra = kernels[name]
            stats, mode = timer.time(fn, args.iters)
            gbps = c["bytes"] / (stats["median_us"] * 1e-6) / 1e9
            row = dict(config=cfg, hq=hq, hkv=hkv, d=d, batch=b, ctx=ctx, skew=skew,
                       sum_tokens=sum(c["lens"]), page_size=args.page_size, dtype=args.dtype,
                       kernel=name, mode=mode, plan_us=plan_s * 1e6, bytes=c["bytes"],
                       gbps=gbps, frac_copy_bw=gbps / copy_gbps, max_abs_err=errs.get(name),
                       **stats, **extra)
            rows.append(row)
            print(f"{cfg:8s} B={b:<4d} ctx={ctx:<6d} skew={int(skew)} {name:16s} "
                  f"{stats['median_us']:9.1f} us  {gbps:7.0f} GB/s ({gbps / copy_gbps:5.1%})  "
                  f"err={errs.get(name)}", flush=True)
        del c, kernels
        torch.cuda.empty_cache()


def run_splits(args, timer, copy_gbps, rows):
    dtype = getattr(torch, args.dtype)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for b in (1, 4, 16):
        for ctx in (2048, 8192, 32768):
            c = build_case("llama8b", b, ctx, args.page_size, dtype, False)
            heur = choose_num_splits(b, c["hkv"], ctx, sms)
            for s in sorted({1, 2, 4, 8, 16, 32, 64, heur}):
                fn, _, extra = make_kernels(c, args.page_size, dtype, splits_override=s,
                                            fi_tensor_cores=(), sdpa=False)["triton_splitkv"]
                stats, mode = timer.time(fn, args.iters)
                gbps = c["bytes"] / (stats["median_us"] * 1e-6) / 1e9
                rows.append(dict(config="llama8b", batch=b, ctx=ctx, kernel="triton_splitkv", mode=mode,
                                 splits=s, heuristic_splits=heur, is_heuristic=s == heur, bytes=c["bytes"],
                                 gbps=gbps, frac_copy_bw=gbps / copy_gbps, **stats))
                print(f"B={b:<3d} ctx={ctx:<6d} splits={s:<3d}{'*' if s == heur else ' '} "
                      f"{stats['median_us']:9.1f} us  {gbps:7.0f} GB/s", flush=True)
            del c
            torch.cuda.empty_cache()


TUNE_SHAPES = [(1, 4096, False), (1, 16384, False), (1, 65536, False), (2, 16384, True), (8, 1024, False),
               (8, 4096, False), (8, 4096, True), (32, 1024, False), (32, 4096, True)]
TUNE_TILES = [(32, 2, 2), (32, 4, 2), (32, 4, 3), (64, 4, 2), (64, 4, 3), (64, 8, 2), (64, 8, 3),
              (128, 4, 2), (128, 8, 2), (128, 8, 3)]                      # (block_n, num_warps, num_stages)
TUNE_PLAN = [(p, o) for p in (0, 256, 512, 2048) for o in (64, 128, 256)]  # (split_penalty, program_overhead)
TUNE_TREE = [32, 64, 128, 0]                    # two-level merge above this many chunks per request (0: never)


def _geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def run_tune(args, timer, copy_gbps, rows):
    """Pick version-2 settings on shapes outside the main matrix, and measure what each change adds.

    Pass 1: FlashInfer (best of its two paths), the ablation chain without tuning
            (v1 -> +wave planner -> +blocked merge -> +direct path), and every tile config.
    Pass 2: with the best tile, every (split_penalty, program_overhead) pair.
    Pass 3: with both, the chunk count above which the two-level merge is used.
    Score of a variant = geomean over shapes of FlashInfer time / variant time (above 1 = faster).
    A variant whose output differs from FlashInfer by more than 2e-2 is dropped from scoring.
    """
    dtype = getattr(torch, args.dtype)
    shapes = [(cfg, b, ctx, skew) for cfg in args.configs for (b, ctx, skew) in TUNE_SHAPES]
    ablation = {
        "v1": BalancedTritonPagedDecode,
        "v1+wave": lambda: BalancedTritonPagedDecode(planner="wave"),
        "v1+wave+blocked": lambda: BalancedTritonPagedDecode(planner="wave", merge="blocked"),
        "v1+wave+blocked+direct": lambda: BalancedTritonPagedDecodeV2(tree_min_chunks=None),
        "v1+wave+blocked+direct+tree": lambda: BalancedTritonPagedDecodeV2(),
    }
    tiles = {f"tile_{bn}_{w}_{st}": (lambda bn=bn, w=w, st=st: BalancedTritonPagedDecodeV2(
        block_n=bn, num_warps=w, num_stages=st)) for bn, w, st in TUNE_TILES}
    score, scored = {}, []

    def one_shape(cfg, b, ctx, skew, variants, stage):
        c = build_case(cfg, b, ctx, args.page_size, dtype, skew)
        fi = make_kernels(c, args.page_size, dtype, sdpa=False, triton=False)
        fi_t = {n: timer.time(fn, args.iters)[0]["median_us"] for n, (fn, _, _) in fi.items()}
        if not fi_t:
            print("  no FlashInfer: cannot score", flush=True)
            return
        scored.append((cfg, b, ctx, skew, stage))
        fi_best = min(fi_t.values())
        ref = fi[min(fi_t, key=fi_t.get)][0]().float()
        for name, make in variants.items():
            try:
                ker = _plan_balanced(c, args.page_size, dtype, make())
                fn = (lambda ker=ker, q=c["q"], k=c["k"], v=c["v"]: ker.run(q, k, v)[0])
                err = float((fn().float().reshape(ref.shape) - ref).abs().max())
                stats, mode = timer.time(fn, args.iters)
            except Exception as e:  # noqa: BLE001  (e.g. a tile that does not fit in shared memory)
                print(f"  {name}: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
                continue
            ok = err < 2e-2
            ratio = fi_best / stats["median_us"]
            if ok:
                score.setdefault(name, []).append(ratio)
            rows.append(dict(config=cfg, batch=b, ctx=ctx, skew=skew, page_size=args.page_size,
                             dtype=args.dtype, kernel="tune", variant=name, stage=stage, mode=mode,
                             flashinfer_us=fi_best, speed_vs_flashinfer=ratio, max_abs_err=err, ok=ok,
                             **stats, **ker.describe()))
            print(f"{cfg:8s} B={b:<3d} ctx={ctx:<6d} skew={int(skew)} {name:26s} {stats['median_us']:8.1f} us "
                  f"vs FI {fi_best:8.1f}  x{ratio:5.2f}  err={err:.1e}{'' if ok else '  WRONG'}", flush=True)
        del c
        torch.cuda.empty_cache()

    for cfg, b, ctx, skew in shapes:
        one_shape(cfg, b, ctx, skew, {**ablation, **tiles}, 1)
    n1 = sum(1 for x in scored if x[4] == 1)
    tile_scores = {n: _geo(score[n]) for n in tiles if len(score.get(n, [])) == n1}
    best_tile = max(tile_scores, key=tile_scores.get)
    bn, w, st = (int(x) for x in best_tile.split("_")[1:])
    print(f"\nbest tile {best_tile}: {tile_scores[best_tile]:.3f}", flush=True)

    plans = {f"plan_{p}_{o}": (lambda p=p, o=o: BalancedTritonPagedDecodeV2(
        block_n=bn, num_warps=w, num_stages=st, split_penalty=p, program_overhead=o)) for p, o in TUNE_PLAN}
    for cfg, b, ctx, skew in shapes:
        one_shape(cfg, b, ctx, skew, plans, 2)
    n2 = sum(1 for x in scored if x[4] == 2)
    plan_scores = {n: _geo(score[n]) for n in plans if len(score.get(n, [])) == n2}
    best_plan = max(plan_scores, key=plan_scores.get)
    p, o = (int(x) for x in best_plan.split("_")[1:])

    trees = {f"tree_{t}": (lambda t=t: BalancedTritonPagedDecodeV2(
        block_n=bn, num_warps=w, num_stages=st, split_penalty=p, program_overhead=o,
        tree_min_chunks=t or None)) for t in TUNE_TREE}
    for cfg, b, ctx, skew in shapes:
        one_shape(cfg, b, ctx, skew, trees, 3)
    n3 = sum(1 for x in scored if x[4] == 3)
    tree_scores = {n: _geo(score[n]) for n in trees if len(score.get(n, [])) == n3}
    best_tree = max(tree_scores, key=tree_scores.get)
    t = int(best_tree.split("_")[1]) or None

    cfg_out = dict(block_n=bn, num_warps=w, num_stages=st, split_penalty=p, program_overhead=o,
                   tree_min_chunks=t)
    summary = dict(config=cfg_out, n_shapes=n1, tile_scores=tile_scores, plan_scores=plan_scores,
                   tree_scores=tree_scores,
                   ablation={n: _geo(score.get(n, [])) for n in ablation},
                   tuned_score=tree_scores[best_tree], note="geomean FlashInfer time / ours on tune shapes")
    with open(os.path.join(args.out, "tuned.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


def run_ncu(args):
    """Eager, few launches, for: ncu --kernel-name regex:'split_kv|merge|Decode|decode' python3 ..."""
    cfg, b, ctx = args.shape.split(",")
    dtype = getattr(torch, args.dtype)
    c = build_case(cfg, int(b), int(ctx), args.page_size, dtype, False)
    for name, (fn, _, _) in make_kernels(c, args.page_size, dtype, sdpa=False, v2_cfg=args.v2_cfg).items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        print("ran", name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["matrix", "splits", "ncu", "tune"], default="matrix")
    ap.add_argument("--tuned", default=None, help="tuned.json from --mode tune (default: <out>/tuned.json)")
    ap.add_argument("--out", default="data/attention")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64])
    ap.add_argument("--ctx", nargs="+", type=int, default=[512, 2048, 8192, 32768])
    ap.add_argument("--long-ctx", nargs="*", type=int, default=[131072])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--max-cache-gb", type=float, default=24.0)
    ap.add_argument("--shape", default="llama8b,1,32768")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "run this on the GPU box"
    tuned = args.tuned or os.path.join(args.out, "tuned.json")
    args.v2_cfg = None
    if args.mode != "tune" and os.path.exists(tuned):
        with open(tuned) as f:
            args.v2_cfg = json.load(f)["config"]
        print(f"version 2 settings from {tuned}: {args.v2_cfg}", flush=True)
    if args.mode == "ncu":
        return run_ncu(args)
    os.makedirs(args.out, exist_ok=True)
    copy_gbps = measure_copy_gbps()
    env = gpu_env(copy_gbps)
    print(json.dumps(env, indent=1), flush=True)
    with open(os.path.join(args.out, "env.json"), "w") as f:
        json.dump(env, f, indent=1)
    timer, rows = Timer(), []
    try:
        {"matrix": run_matrix, "splits": run_splits, "tune": run_tune}[args.mode](args, timer, copy_gbps, rows)
    finally:
        path = os.path.join(args.out, f"{args.mode}_ps{args.page_size}_{args.dtype}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {path}", flush=True)


if __name__ == "__main__":
    main()
