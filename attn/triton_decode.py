"""Paged-KV GQA decode attention in Triton, with split-KV (Flash-Decoding) and a log-sum-exp merge.

Two kernels:

1. split kernel, grid (batch, num_kv_heads, num_splits)
   One program owns one KV head of one request and one contiguous slice ("split") of its tokens.
   It loads the G = num_qo_heads / num_kv_heads query heads that share that KV head as one
   [G_pad, head_dim] tile, so every K/V tile read from memory is used by all G heads (that reuse
   is the whole point of GQA at decode). It walks its tokens BLOCK_N at a time through the page
   table, keeps an online softmax (running max m, running sum l, running output acc), and writes a
   normalized partial output O_s and its log-sum-exp LSE_s to a fp32 workspace.

2. merge kernel, grid (batch, num_qo_heads)
   Combines the splits:  m = max_s LSE_s,  w_s = exp(LSE_s - m),
   O = sum_s w_s O_s / sum_s w_s,  LSE = m + log(sum_s w_s).
   Splits with no tokens carry LSE = -inf and get weight 0.

Why split at all: at decode the query length is 1, so without splitting the grid is only
batch * num_kv_heads programs. At batch 1 with 8 KV heads that is 8 programs on a GPU with
132 (H100 SXM) or 170 (RTX 5090) SMs, so most of the GPU idles while one program streams a
long KV cache. Splitting the sequence fills the GPU, at the cost of the workspace round trip
and the merge.

Three entry points share the inner loop (_attend_tokens):
* TritonPagedDecode: the same number of splits for every request (the kernels above);
* BalancedTritonPagedDecode: per-request chunks of ~equal size (version 1), separate merge kernel;
* BalancedTritonPagedDecodeV2: occupancy-aware wave planner, blocked merge, a direct path that
  skips the merge when no request is split, and a two-level merge for requests with many chunks. See attn/README.md for why each change was made.
Not covered: FP8 KV, prefill, RoPE fusion.
"""
import math

import torch
import triton
import triton.language as tl

LN2 = math.log(2.0)
LOG2E = 1.0 / LN2


@triton.jit
def _attend_tokens(
    Q, K, V, KV_INDICES, b, kvh, page_start, split_start, split_end, sm_scale_log2,
    stride_qb, stride_qh, stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
    PAGE_SIZE: tl.constexpr, GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Attention of the G query heads of KV head `kvh` (request b) over tokens [split_start, split_end).
    Returns the normalized partial output [GROUP_PAD, HEAD_DIM], its natural-log LSE [GROUP_PAD],
    the query-head indices and their mask. Shared by the uniform-split and balanced kernels."""
    offs_g = tl.arange(0, GROUP_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    g_mask = offs_g < GROUP
    head = kvh * GROUP + offs_g                      # query heads served by this KV head

    q = tl.load(Q + b * stride_qb + head[:, None] * stride_qh + offs_d[None, :],
                mask=g_mask[:, None], other=0.0)

    m_i = tl.full([GROUP_PAD], float("-inf"), tl.float32)   # running max, log2 domain
    l_i = tl.zeros([GROUP_PAD], tl.float32)                 # running sum of 2^(score - m)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)       # running unnormalized output

    for start in range(split_start, split_end, BLOCK_N):
        t = start + offs_n
        t_mask = t < split_end
        # logical token t -> (page id, slot in page) through the page table
        page = tl.load(KV_INDICES + page_start + t // PAGE_SIZE, mask=t_mask, other=0).to(tl.int64)
        slot = t % PAGE_SIZE

        k = tl.load(K + page[:, None] * stride_kp + slot[:, None] * stride_kt
                    + kvh * stride_kh + offs_d[None, :], mask=t_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2          # [GROUP_PAD, BLOCK_N], log2 domain
        qk = tl.where(t_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)   # avoid (-inf) - (-inf) = nan
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)                           # rescale what we had so far
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(V + page[:, None] * stride_vp + slot[:, None] * stride_vt
                    + kvh * stride_vh + offs_d[None, :], mask=t_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has = l_i > 0.0
    l_safe = tl.where(has, l_i, 1.0)
    o = acc / l_safe[:, None]
    # natural-log LSE: sum_j e^{s_j} = 2^{m} * l  ->  ln = (m + log2 l) * ln 2
    lse = tl.where(has, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    return o, lse, head, g_mask


@triton.jit
def _split_kv_kernel(
    Q, K, V, KV_INDPTR, KV_INDICES, KV_LAST,
    O_PART, LSE_PART,
    sm_scale_log2,
    stride_qb, stride_qh,
    stride_kp, stride_kt, stride_kh,
    stride_vp, stride_vt, stride_vh,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    PAGE_SIZE: tl.constexpr, GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    """Uniform splitting: every request is cut into NUM_SPLITS equal slices."""
    b = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)
    s = tl.program_id(2)

    page_start = tl.load(KV_INDPTR + b)
    page_end = tl.load(KV_INDPTR + b + 1)
    last_len = tl.load(KV_LAST + b)
    seq_len = (page_end - page_start - 1) * PAGE_SIZE + last_len

    tokens_per_split = tl.cdiv(seq_len, NUM_SPLITS)
    split_start = s * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, seq_len)

    o, lse, head, g_mask = _attend_tokens(
        Q, K, V, KV_INDICES, b, kvh, page_start, split_start, split_end, sm_scale_log2,
        stride_qb, stride_qh, stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
        PAGE_SIZE, GROUP, GROUP_PAD, HEAD_DIM, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    tl.store(O_PART + b * stride_ob + head[:, None] * stride_oh + s * stride_os + offs_d[None, :],
             o, mask=g_mask[:, None])
    tl.store(LSE_PART + b * stride_lb + head * stride_lh + s, lse, mask=g_mask)


@triton.jit
def _balanced_kernel(
    Q, K, V, KV_INDPTR, KV_INDICES, WORK_REQ, WORK_START, WORK_END,
    O_PART, LSE_PART, O, LSE,
    sm_scale_log2,
    stride_qb, stride_qh,
    stride_kp, stride_kt, stride_kh,
    stride_vp, stride_vt, stride_vh,
    stride_ow, stride_oh, stride_lw,
    stride_xb, stride_xh, stride_yb,
    PAGE_SIZE: tl.constexpr, GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, DIRECT: tl.constexpr,
):
    """Balanced splitting: plan() cut the batch into work items of ~equal token count
    (a long request gets many chunks, a short one few), grid = (num_work_items, num_kv_heads).
    DIRECT: every request is a single work item (w == b), so the program writes the final output
    and LSE itself and no merge kernel runs."""
    w = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)
    b = tl.load(WORK_REQ + w).to(tl.int64)
    split_start = tl.load(WORK_START + w)
    split_end = tl.load(WORK_END + w)
    page_start = tl.load(KV_INDPTR + b)

    o, lse, head, g_mask = _attend_tokens(
        Q, K, V, KV_INDICES, b, kvh, page_start, split_start, split_end, sm_scale_log2,
        stride_qb, stride_qh, stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
        PAGE_SIZE, GROUP, GROUP_PAD, HEAD_DIM, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if DIRECT:
        tl.store(O + b * stride_xb + head[:, None] * stride_xh + offs_d[None, :],
                 o.to(O.dtype.element_ty), mask=g_mask[:, None])
        tl.store(LSE + b * stride_yb + head, lse, mask=g_mask)
    else:
        tl.store(O_PART + w * stride_ow + head[:, None] * stride_oh + offs_d[None, :], o,
                 mask=g_mask[:, None])
        tl.store(LSE_PART + w * stride_lw + head, lse, mask=g_mask)


@triton.jit
def _balanced_merge_kernel(
    O_PART, LSE_PART, CHUNK_INDPTR, O, LSE,
    stride_ow, stride_oh, stride_lw,
    stride_xb, stride_xh, stride_yb,
    HEAD_DIM: tl.constexpr,
):
    """Merge request b's chunks [CHUNK_INDPTR[b], CHUNK_INDPTR[b+1]) with a running (online) LSE merge.
    Every chunk is non-empty by construction, so the first chunk seeds the running state."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    c0 = tl.load(CHUNK_INDPTR + b).to(tl.int64)
    c1 = tl.load(CHUNK_INDPTR + b + 1).to(tl.int64)

    m = tl.load(LSE_PART + c0 * stride_lw + h)
    acc = tl.load(O_PART + c0 * stride_ow + h * stride_oh + offs_d)
    denom = m * 0.0 + 1.0                              # exp(LSE_c0 - m) = 1
    for c in range(c0 + 1, c1):
        lse_c = tl.load(LSE_PART + c * stride_lw + h)
        o_c = tl.load(O_PART + c * stride_ow + h * stride_oh + offs_d)
        m_new = tl.maximum(m, lse_c)
        a = tl.exp(m - m_new)
        wc = tl.exp(lse_c - m_new)
        denom = denom * a + wc
        acc = acc * a + wc * o_c
        m = m_new
    tl.store(O + b * stride_xb + h * stride_xh + offs_d, (acc / denom).to(O.dtype.element_ty))
    tl.store(LSE + b * stride_yb + h, m + tl.log(denom))


@triton.jit
def _blocked_merge_kernel(
    O_PART, LSE_PART, CHUNK_INDPTR, O, LSE,
    stride_ow, stride_oh, stride_lw,
    stride_xb, stride_xh, stride_yb,
    HEAD_DIM: tl.constexpr, CHUNK_BLOCK: tl.constexpr,
):
    """Same result as _balanced_merge_kernel, but reads CHUNK_BLOCK chunks per step instead of one,
    so a request with hundreds of chunks costs a few dependent loads instead of hundreds.
    Online merge over blocks: m_new = max(m, max_block LSE), rescale the running sums by e^(m - m_new)."""
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_c = tl.arange(0, CHUNK_BLOCK)
    c0 = tl.load(CHUNK_INDPTR + b).to(tl.int64)
    c1 = tl.load(CHUNK_INDPTR + b + 1).to(tl.int64)

    m = tl.full([1], float("-inf"), tl.float32)
    denom = tl.zeros([1], tl.float32)
    acc = tl.zeros([HEAD_DIM], tl.float32)
    for start in range(c0, c1, CHUNK_BLOCK):
        c = start + offs_c
        c_mask = c < c1
        lse_c = tl.load(LSE_PART + c * stride_lw + h, mask=c_mask, other=float("-inf"))
        o_c = tl.load(O_PART + c[:, None] * stride_ow + h * stride_oh + offs_d[None, :],
                      mask=c_mask[:, None], other=0.0)
        m_new = tl.maximum(m, tl.max(lse_c, 0))       # finite: every block has >= 1 non-empty chunk
        a = tl.exp(m - m_new)                          # 0 on the first block (m = -inf)
        w = tl.exp(lse_c - m_new)                      # masked chunks: exp(-inf) = 0
        denom = denom * a + tl.sum(w, 0)
        acc = acc * a + tl.sum(w[:, None] * o_c, 0)
        m = m_new
    tl.store(O + b * stride_xb + h * stride_xh + offs_d, (acc / denom).to(O.dtype.element_ty))
    tl.store(LSE + b * stride_yb + h + tl.arange(0, 1), m + tl.log(denom))


@triton.jit
def _merge_kernel(
    O_PART, LSE_PART, O, LSE,
    stride_pb, stride_ph, stride_ps,
    stride_lb, stride_lh,
    stride_ob, stride_oh, stride_xb,
    NUM_SPLITS: tl.constexpr, SPLIT_PAD: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs_s = tl.arange(0, SPLIT_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    s_mask = offs_s < NUM_SPLITS

    lse_s = tl.load(LSE_PART + b * stride_lb + h * stride_lh + offs_s, mask=s_mask, other=float("-inf"))
    m = tl.max(lse_s, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse_s - m_safe)                                   # empty splits: exp(-inf) = 0
    denom = tl.sum(w, 0)

    o_s = tl.load(O_PART + b * stride_pb + h * stride_ph + offs_s[:, None] * stride_ps + offs_d[None, :],
                  mask=s_mask[:, None], other=0.0)
    o = tl.sum(w[:, None] * o_s, 0) / tl.where(denom > 0.0, denom, 1.0)
    lse = tl.where(denom > 0.0, m_safe + tl.log(denom), float("-inf"))

    tl.store(O + b * stride_ob + h * stride_oh + offs_d, o.to(O.dtype.element_ty))
    tl.store(LSE + b * stride_xb + h, lse)


def choose_num_splits(batch, num_kv_heads, max_seq_len, num_sms, min_tokens_per_split=256, max_splits=64):
    """Same split count for every request: enough programs to cover ~2 waves of SMs, but no split
    shorter than min_tokens_per_split (tiny splits cost more in workspace traffic than they save)."""
    want = math.ceil(2 * num_sms / max(1, batch * num_kv_heads))
    cap = max(1, math.ceil(max_seq_len / min_tokens_per_split))
    return int(max(1, min(want, cap, max_splits)))


class TritonPagedDecode:
    """plan() once per decode step (host-side metadata, buffers), run() per layer (no host syncs,
    so run() can be captured in a CUDA graph). Mirrors FlashInfer's plan/run split."""

    def __init__(self, num_sms=None, block_n=64, num_warps=4, num_stages=2, min_tokens_per_split=256):
        self.num_sms = num_sms
        self.block_n = block_n
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.min_tokens_per_split = min_tokens_per_split

    def plan(self, kv_indptr, kv_indices, kv_last_page_len, num_qo_heads, num_kv_heads, head_dim,
             page_size, max_seq_len, q_dtype, num_splits=None):
        assert num_qo_heads % num_kv_heads == 0
        assert head_dim in (16, 32, 64, 128, 256), "head_dim must be a power of two >= 16"
        device = kv_indptr.device
        batch = kv_indptr.numel() - 1
        num_sms = self.num_sms
        if num_sms is None:
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count \
                if device.type == "cuda" else 132
        self.num_splits = num_splits or choose_num_splits(
            batch, num_kv_heads, max_seq_len, num_sms, self.min_tokens_per_split)
        self.batch, self.num_qo_heads, self.num_kv_heads = batch, num_qo_heads, num_kv_heads
        self.head_dim, self.page_size = head_dim, page_size
        self.group = num_qo_heads // num_kv_heads
        self.group_pad = max(16, triton.next_power_of_2(self.group))     # tl.dot needs M >= 16
        self.kv_indptr, self.kv_indices, self.kv_last = kv_indptr, kv_indices, kv_last_page_len
        f32 = dict(dtype=torch.float32, device=device)
        self.o_part = torch.empty(batch, num_qo_heads, self.num_splits, head_dim, **f32)
        self.lse_part = torch.empty(batch, num_qo_heads, self.num_splits, **f32)
        self.o = torch.empty(batch, num_qo_heads, head_dim, dtype=q_dtype, device=device)
        self.lse = torch.empty(batch, num_qo_heads, **f32)
        return self

    def run(self, q, k_cache, v_cache, sm_scale=None):
        assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
        scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(self.head_dim)
        grid = (self.batch, self.num_kv_heads, self.num_splits)
        _split_kv_kernel[grid](
            q, k_cache, v_cache, self.kv_indptr, self.kv_indices, self.kv_last,
            self.o_part, self.lse_part,
            scale * LOG2E,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            self.o_part.stride(0), self.o_part.stride(1), self.o_part.stride(2),
            self.lse_part.stride(0), self.lse_part.stride(1),
            PAGE_SIZE=self.page_size, GROUP=self.group, GROUP_PAD=self.group_pad,
            HEAD_DIM=self.head_dim, BLOCK_N=self.block_n, NUM_SPLITS=self.num_splits,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        _merge_kernel[(self.batch, self.num_qo_heads)](
            self.o_part, self.lse_part, self.o, self.lse,
            self.o_part.stride(0), self.o_part.stride(1), self.o_part.stride(2),
            self.lse_part.stride(0), self.lse_part.stride(1),
            self.o.stride(0), self.o.stride(1), self.lse.stride(0),
            NUM_SPLITS=self.num_splits, SPLIT_PAD=triton.next_power_of_2(self.num_splits),
            HEAD_DIM=self.head_dim, num_warps=4,
        )
        return self.o, self.lse


def plan_balanced_chunks(seq_lens, num_kv_heads, num_sms, block_n=64, min_chunk=256):
    """Cut the batch into work items of about equal size, like FlashInfer's plan().

    chunk = max(min_chunk, total_tokens / target_items), rounded up to a multiple of block_n, where
    target_items gives ~2 waves of (work item x KV head) programs over the SMs. Request b gets
    ceil(L_b / chunk) chunks, so a 32k-token request and a 2k-token request in the same batch get
    work proportional to their length instead of the same number of splits.
    Returns (work_req, work_start, work_end, chunk_indptr, chunk) as Python lists / int.
    """
    total = sum(seq_lens)
    target_items = max(1, math.ceil(2 * num_sms / num_kv_heads))
    chunk = max(min_chunk, math.ceil(total / target_items))
    chunk = math.ceil(chunk / block_n) * block_n
    work_req, work_start, work_end, chunk_indptr = [], [], [], [0]
    for b, L in enumerate(seq_lens):
        n = max(1, math.ceil(L / chunk))
        for i in range(n):
            work_req.append(b)
            work_start.append(i * chunk)
            work_end.append(min(L, (i + 1) * chunk))
        chunk_indptr.append(chunk_indptr[-1] + n)
    return work_req, work_start, work_end, chunk_indptr, chunk


def plan_wave_chunks(seq_lens, num_kv_heads, slots, block_n=64, min_chunk=256, split_penalty=0,
                     program_overhead=128):
    """Pick the chunk size for a whole wave of resident programs (version 2 planner).

    `slots` = SMs x programs that fit on one SM at once (from the compiled kernel's registers and
    shared memory). With chunk c, request b has ceil(L_b / c) work items, the grid has
    items * num_kv_heads programs, and they run in ceil(programs / slots) waves of programs that
    each stream up to c tokens. Estimated time ~ waves * (c + program_overhead), where the overhead
    (in tokens) is the fixed cost of a program (loading q, writing its partial result, pipeline
    fill), so many tiny chunks are not free; `split_penalty` (in tokens) is added
    whenever some request is split, standing for the merge kernel. Lowest estimate wins, ties go
    to the larger chunk (fewer partial results to merge).

    The version 1 planner (plan_balanced_chunks) aimed at 2 x SMs programs whatever the occupancy,
    which on H100 filled half of the 4 programs per SM that fit (Nsight: 0.48 waves), and its
    chunk did not divide the requests evenly (32768 tokens -> 15936 + 15936 + 896).
    """
    lens = [max(1, int(L)) for L in seq_lens]
    lmax = max(lens)
    items_per_wave = max(1.0, slots / num_kv_heads)
    up = lambda x: max(block_n, math.ceil(x / block_n) * block_n)  # noqa: E731
    lo = up(min_chunk)
    cands = {up(math.ceil(lmax / k)) for k in range(1, max(1, math.ceil(lmax / lo)) + 1)}
    cands = sorted(c for c in cands if c >= lo) or [up(lmax)]
    best = None
    for c in cands:
        items = sum(math.ceil(L / c) for L in lens)
        waves = math.ceil(items / items_per_wave)
        cost = waves * (c + program_overhead) + (split_penalty if items > len(lens) else 0)
        key = (cost, -c)
        if best is None or key < best[0]:
            best = (key, c)
    chunk = best[1]
    work_req, work_start, work_end, chunk_indptr = [], [], [], [0]
    for b, L in enumerate(lens):
        n = math.ceil(L / chunk)
        for i in range(n):
            work_req.append(b)
            work_start.append(i * chunk)
            work_end.append(min(L, (i + 1) * chunk))
        chunk_indptr.append(chunk_indptr[-1] + n)
    return work_req, work_start, work_end, chunk_indptr, chunk


_OCC_CACHE = {}


def programs_per_sm(device, q_dtype, page_size, group, group_pad, head_dim, block_n, num_warps, num_stages,
                    direct=False, fallback=4):
    """How many programs of _balanced_kernel fit on one SM at once, from the compiled kernel's
    register and shared-memory use (the method of Triton's fused-softmax tutorial). Falls back to
    `fallback` off-GPU (the interpreter) or if the query fails."""
    key = (str(device), q_dtype, page_size, group, head_dim, block_n, num_warps, num_stages, direct)
    if key in _OCC_CACHE:
        return _OCC_CACHE[key]
    res = dict(ctas=fallback, n_regs=None, smem=None, n_spills=None)
    try:
        if device.type != "cuda":
            raise RuntimeError("not on a GPU")
        z = lambda *shape, dt=torch.float32: torch.zeros(*shape, dtype=dt, device=device)  # noqa: E731
        i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
        q, k = z(1, group, head_dim, dt=q_dtype), z(1, page_size, 1, head_dim, dt=q_dtype)
        op, lp, o, lse = z(1, group, head_dim), z(1, group), z(1, group, head_dim, dt=q_dtype), z(1, group)
        ck = _balanced_kernel.warmup(
            q, k, k, i32(0, 1), i32(0), i32(0), i32(0), i32(1), op, lp, o, lse, 1.0,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), k.stride(0), k.stride(1),
            k.stride(2), op.stride(0), op.stride(1), lp.stride(0), o.stride(0), o.stride(1), lse.stride(0),
            PAGE_SIZE=page_size, GROUP=group, GROUP_PAD=group_pad, HEAD_DIM=head_dim, BLOCK_N=block_n,
            DIRECT=direct, num_warps=num_warps, num_stages=num_stages, grid=(1, 1))
        if hasattr(ck, "_init_handles"):
            ck._init_handles()
        n_regs, smem = int(ck.n_regs), int(ck.metadata.shared)
        props = torch.cuda.get_device_properties(device)
        regs_sm = getattr(props, "regs_per_multiprocessor", 65536)
        smem_sm = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
        threads_sm = getattr(props, "max_threads_per_multi_processor", 2048)
        regs_warp = math.ceil(n_regs * 32 / 256) * 256          # registers are allocated per warp in 256s
        by_regs = regs_sm // max(1, regs_warp * num_warps)
        by_smem = smem_sm // (smem + 1024) if smem else 32       # 1 KB per block reserved by the driver
        by_threads = threads_sm // (32 * num_warps)
        res = dict(ctas=int(max(1, min(by_regs, by_smem, by_threads, 32))), n_regs=n_regs, smem=smem,
                   n_spills=int(getattr(ck, "n_spills", 0) or 0))
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    _OCC_CACHE[key] = res
    return res


class BalancedTritonPagedDecode:
    """Same kernel body as TritonPagedDecode, but work is split per request (proportional to its
    length) instead of one split count for the whole batch. plan() reads the sequence lengths on the
    host (one device->host copy per decode step, as FlashInfer's plan does); run() has no host syncs.

    Defaults reproduce version 1 (the first H100 results). Version 2 (BalancedTritonPagedDecodeV2)
    changes three things, each switchable for the ablation:
      planner="wave"   size chunks for one full wave of resident programs (plan_wave_chunks);
      merge="blocked"  merge CHUNK_BLOCK partial results per step (_blocked_merge_kernel);
      direct=True      if no request is split, write the output directly and skip the merge.
    """

    def __init__(self, num_sms=None, block_n=64, num_warps=4, num_stages=2, min_chunk=256,
                 planner="v1", merge="serial", direct=False, split_penalty=0, ctas_per_sm=None,
                 chunk_block=16, chunk=None, program_overhead=128, tree_min_chunks=None, tree_group=32):
        assert planner in ("v1", "wave") and merge in ("serial", "blocked")
        self.num_sms = num_sms
        self.block_n = block_n
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.min_chunk = min_chunk
        self.planner, self.merge, self.direct = planner, merge, direct
        self.split_penalty = split_penalty
        self.ctas_per_sm = ctas_per_sm
        self.chunk_block = chunk_block
        # two-level merge when some request has more than tree_min_chunks chunks (None: never)
        self.tree_min_chunks, self.tree_group = tree_min_chunks, tree_group
        self.chunk_override = chunk          # tests: force a chunk size
        self.program_overhead = program_overhead

    def plan(self, kv_indptr, kv_indices, kv_last_page_len, num_qo_heads, num_kv_heads, head_dim,
             page_size, max_seq_len=None, q_dtype=torch.float16, num_splits=None):
        assert num_qo_heads % num_kv_heads == 0
        assert head_dim in (16, 32, 64, 128, 256), "head_dim must be a power of two >= 16"
        device = kv_indptr.device
        num_sms = self.num_sms
        if num_sms is None:
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count \
                if device.type == "cuda" else 132
        self.group = num_qo_heads // num_kv_heads
        self.group_pad = max(16, triton.next_power_of_2(self.group))
        pages = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
        last = kv_last_page_len.cpu().tolist()
        seq_lens = [(p - 1) * page_size + l for p, l in zip(pages, last)]
        self.occupancy = None
        if self.chunk_override:          # huge num_sms -> chunk = chunk_override (rounded to block_n)
            req, start, end, cind, self.chunk = plan_balanced_chunks(
                seq_lens, num_kv_heads, 10**9, self.block_n, self.chunk_override)
        elif self.planner == "v1":
            req, start, end, cind, self.chunk = plan_balanced_chunks(
                seq_lens, num_kv_heads, num_sms, self.block_n, self.min_chunk)
        else:
            if self.ctas_per_sm is None:
                self.occupancy = programs_per_sm(device, q_dtype, page_size, self.group, self.group_pad,
                                                 head_dim, self.block_n, self.num_warps, self.num_stages)
                ctas = self.occupancy["ctas"]
            else:
                ctas = self.ctas_per_sm
            self.slots = num_sms * ctas
            req, start, end, cind, self.chunk = plan_wave_chunks(
                seq_lens, num_kv_heads, self.slots, self.block_n, self.min_chunk, self.split_penalty,
                self.program_overhead)
        i32 = dict(dtype=torch.int32, device=device)
        self.work_req = torch.tensor(req, **i32)
        self.work_start = torch.tensor(start, **i32)
        self.work_end = torch.tensor(end, **i32)
        self.chunk_indptr = torch.tensor(cind, **i32)
        self.num_work = len(req)
        self.batch, self.num_qo_heads, self.num_kv_heads = len(seq_lens), num_qo_heads, num_kv_heads
        self.head_dim, self.page_size = head_dim, page_size
        self.is_direct = bool(self.direct and self.num_work == self.batch)
        self.kv_indptr, self.kv_indices = kv_indptr, kv_indices
        f32 = dict(dtype=torch.float32, device=device)
        nw = 1 if self.is_direct else self.num_work
        self.o_part = torch.empty(nw, num_qo_heads, head_dim, **f32)
        self.lse_part = torch.empty(nw, num_qo_heads, **f32)
        self.o = torch.empty(self.batch, num_qo_heads, head_dim, dtype=q_dtype, device=device)
        self.lse = torch.empty(self.batch, num_qo_heads, **f32)
        self.num_splits = self.num_work  # reported in benchmarks as total work items
        self.max_chunks = max(b - a for a, b in zip(cind[:-1], cind[1:]))
        self.is_tree = bool(not self.is_direct and self.merge == "blocked" and self.tree_min_chunks
                            and self.max_chunks > self.tree_min_chunks)
        if self.is_tree:
            # level 1: groups of up to tree_group consecutive chunks of one request -> one partial each;
            # level 2: each request merges its groups. Both levels are the blocked merge kernel.
            g_ind, r_ind = [0], [0]
            for b in range(self.batch):
                for c in range(cind[b], cind[b + 1], self.tree_group):
                    g_ind.append(min(c + self.tree_group, cind[b + 1]))
                r_ind.append(len(g_ind) - 1)
            self.group_indptr = torch.tensor(g_ind, **i32)
            self.group_req_indptr = torch.tensor(r_ind, **i32)
            self.num_groups = len(g_ind) - 1
            self.o_mid = torch.empty(self.num_groups, num_qo_heads, head_dim, **f32)
            self.lse_mid = torch.empty(self.num_groups, num_qo_heads, **f32)
        return self

    def describe(self):
        d = dict(splits=self.num_work, chunk=self.chunk, direct=self.is_direct, tree=self.is_tree,
                 max_chunks=self.max_chunks, planner=self.planner,
                 merge=self.merge, block_n=self.block_n, num_warps=self.num_warps,
                 num_stages=self.num_stages, split_penalty=self.split_penalty,
                 program_overhead=self.program_overhead)
        if getattr(self, "slots", None):
            d["slots"] = self.slots
        if self.occupancy:
            d.update(ctas_per_sm=self.occupancy["ctas"], n_regs=self.occupancy["n_regs"],
                     smem=self.occupancy["smem"], n_spills=self.occupancy["n_spills"])
            if "error" in self.occupancy:
                d["occupancy_error"] = self.occupancy["error"]
        return d

    def run(self, q, k_cache, v_cache, sm_scale=None):
        assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
        scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(self.head_dim)
        _balanced_kernel[(self.num_work, self.num_kv_heads)](
            q, k_cache, v_cache, self.kv_indptr, self.kv_indices,
            self.work_req, self.work_start, self.work_end,
            self.o_part, self.lse_part, self.o, self.lse,
            scale * LOG2E,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            self.o_part.stride(0), self.o_part.stride(1), self.lse_part.stride(0),
            self.o.stride(0), self.o.stride(1), self.lse.stride(0),
            PAGE_SIZE=self.page_size, GROUP=self.group, GROUP_PAD=self.group_pad,
            HEAD_DIM=self.head_dim, BLOCK_N=self.block_n, DIRECT=self.is_direct,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if self.is_direct:
            return self.o, self.lse
        if self.is_tree:
            assert self.tree_group & (self.tree_group - 1) == 0, "tree_group must be a power of two"
            _blocked_merge_kernel[(self.num_groups, self.num_qo_heads)](
                self.o_part, self.lse_part, self.group_indptr, self.o_mid, self.lse_mid,
                self.o_part.stride(0), self.o_part.stride(1), self.lse_part.stride(0),
                self.o_mid.stride(0), self.o_mid.stride(1), self.lse_mid.stride(0),
                HEAD_DIM=self.head_dim, CHUNK_BLOCK=self.tree_group, num_warps=4)
            _blocked_merge_kernel[(self.batch, self.num_qo_heads)](
                self.o_mid, self.lse_mid, self.group_req_indptr, self.o, self.lse,
                self.o_mid.stride(0), self.o_mid.stride(1), self.lse_mid.stride(0),
                self.o.stride(0), self.o.stride(1), self.lse.stride(0),
                HEAD_DIM=self.head_dim, CHUNK_BLOCK=self.chunk_block, num_warps=4)
            return self.o, self.lse
        args = (self.o_part, self.lse_part, self.chunk_indptr, self.o, self.lse,
                self.o_part.stride(0), self.o_part.stride(1), self.lse_part.stride(0),
                self.o.stride(0), self.o.stride(1), self.lse.stride(0))
        if self.merge == "blocked":
            _blocked_merge_kernel[(self.batch, self.num_qo_heads)](
                *args, HEAD_DIM=self.head_dim, CHUNK_BLOCK=self.chunk_block, num_warps=4)
        else:
            _balanced_merge_kernel[(self.batch, self.num_qo_heads)](*args, HEAD_DIM=self.head_dim, num_warps=4)
        return self.o, self.lse


# version 2 defaults; bench/bench_attention.py --mode tune picks block_n / warps / stages / split_penalty
V2_DEFAULTS = dict(planner="wave", merge="blocked", direct=True, block_n=64, num_warps=4, num_stages=2,
                   min_chunk=256, split_penalty=512, program_overhead=128, tree_min_chunks=64, tree_group=32)


class BalancedTritonPagedDecodeV2(BalancedTritonPagedDecode):
    def __init__(self, **overrides):
        super().__init__(**{**V2_DEFAULTS, **overrides})
