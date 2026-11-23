"""Correctness of the Triton paged GQA decode kernel against the PyTorch reference (and FlashInfer when
it is installed). Without a GPU these run under the Triton interpreter (TRITON_INTERPRET=1), which is
slow but checks the same kernel code."""
import math
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
pytest.importorskip("triton")

from attn.paged import make_paged_kv  # noqa: E402
from attn.reference import paged_decode_reference, gather_kv  # noqa: E402
from attn.triton_decode import TritonPagedDecode, choose_num_splits  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL = {torch.float16: dict(atol=2e-3, rtol=2e-3), torch.bfloat16: dict(atol=1e-2, rtol=1e-2)}


def _run(seq_lens, hq, hkv, d, page_size=16, dtype=torch.float16, num_splits=None, block_n=16, seed=0):
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, page_size, dtype, DEV, seed=seed,
                                                extra_pages=3)
    q = torch.randn(len(seq_lens), hq, d, generator=torch.Generator().manual_seed(seed + 1)).to(dtype).to(DEV)
    ker = TritonPagedDecode(num_sms=132, block_n=block_n).plan(
        indptr, indices, last, hq, hkv, d, page_size, max(seq_lens), dtype, num_splits=num_splits)
    o, lse = ker.run(q, k, v)
    o_ref, lse_ref = paged_decode_reference(q, k, v, indptr, indices, last)
    return o, lse, o_ref, lse_ref, ker


# ragged lengths around page boundaries: 1, page_size-1, page_size, page_size+1, a longer one
LENS = [1, 15, 16, 17, 100]


@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2), (28, 4), (8, 1), (48, 2)])  # G = 1, 4, 7, 8, 24
def test_gqa_groups(hq, hkv):
    o, lse, o_ref, lse_ref, _ = _run(LENS, hq, hkv, 64, num_splits=2)
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_splits", [1, 2, 4, 16])
def test_split_counts_agree(num_splits):
    o, lse, o_ref, lse_ref, _ = _run(LENS, 8, 2, 64, num_splits=num_splits)
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_head_dim_128_and_page_size_64():
    o, lse, o_ref, lse_ref, _ = _run([1, 63, 64, 65, 200], 8, 2, 128, page_size=64, num_splits=3)
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_empty_splits_give_no_nan():
    # 1 token split 16 ways: 15 splits are empty and must contribute nothing (no NaN from -inf - -inf)
    o, lse, o_ref, lse_ref, ker = _run([1, 2], 8, 2, 64, num_splits=16)
    assert torch.isfinite(o.float()).all() and torch.isfinite(lse).all()
    assert torch.isinf(ker.lse_part).any()          # some partial states really were empty
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_large_logits_are_stable():
    # scores ~ +-100: exp would overflow without max subtraction
    k, v, indptr, indices, last = make_paged_kv([40, 70], 2, 64, 16, torch.float16, DEV, seed=3)
    q = (torch.randn(2, 8, 64, generator=torch.Generator().manual_seed(4)) * 30).to(torch.float16).to(DEV)
    ker = TritonPagedDecode(num_sms=132, block_n=16).plan(indptr, indices, last, 8, 2, 64, 16, 70,
                                                          torch.float16, num_splits=4)
    o, lse = ker.run(q, k, v)
    o_ref, lse_ref = paged_decode_reference(q, k, v, indptr, indices, last)
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)


def test_reference_matches_sdpa_on_contiguous_kv():
    # the oracle itself, checked against torch's attention on gathered K/V
    k, v, indptr, indices, last = make_paged_kv([33, 50], 2, 64, 16, torch.float32, "cpu", seed=5)
    q = torch.randn(2, 8, 64, generator=torch.Generator().manual_seed(6))
    o_ref, _ = paged_decode_reference(q, k, v, indptr, indices, last)
    for b in range(2):
        kb = gather_kv(k, indptr, indices, last, b).repeat_interleave(4, dim=1).transpose(0, 1)  # [Hq, L, D]
        vb = gather_kv(v, indptr, indices, last, b).repeat_interleave(4, dim=1).transpose(0, 1)
        o_sdpa = torch.nn.functional.scaled_dot_product_attention(q[b][:, None], kb, vb)[:, 0]
        torch.testing.assert_close(o_ref[b], o_sdpa, atol=1e-5, rtol=1e-5)


def test_split_heuristic():
    # batch 1, 8 KV heads, 132 SMs: wants ceil(264/8) = 33 splits, long context allows it
    assert choose_num_splits(1, 8, 32768, 132) == 33
    # short context caps splits so each covers >= 256 tokens
    assert choose_num_splits(1, 8, 512, 132) == 2
    # big batch already fills the GPU
    assert choose_num_splits(64, 8, 32768, 132) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 path checked on GPU")
def test_bf16():
    o, lse, o_ref, lse_ref, _ = _run(LENS, 32, 8, 128, dtype=torch.bfloat16, block_n=64)
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.bfloat16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_matches_flashinfer():
    flashinfer = pytest.importorskip("flashinfer")
    seq_lens, hq, hkv, d, ps = [1, 17, 300, 1000], 32, 8, 128, 16
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, ps, torch.float16, "cuda", seed=7)
    q = torch.randn(len(seq_lens), hq, d, device="cuda", dtype=torch.float16)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(indptr, indices, last, hq, hkv, d, ps, q_data_type=torch.float16, kv_data_type=torch.float16)
    o_fi, lse_fi = w.run(q, (k, v), return_lse=True)
    ker = TritonPagedDecode().plan(indptr, indices, last, hq, hkv, d, ps, max(seq_lens), torch.float16)
    o, lse = ker.run(q, k, v)
    torch.testing.assert_close(o.float(), o_fi.float(), **TOL[torch.float16])
    # FlashInfer's LSE convention (natural log vs log2) has differed between kernels/versions:
    # accept whichever matches, and report which one it was.
    nat = torch.allclose(lse, lse_fi.float(), atol=1e-2)
    base2 = torch.allclose(lse, lse_fi.float() * math.log(2.0), atol=1e-2)
    print(f"flashinfer {flashinfer.__version__}: LSE is {'natural log' if nat else 'log2' if base2 else '??'}")
    assert nat or base2


# ---------------------------------------------------------------- balanced (per-request) splitting
from attn.triton_decode import BalancedTritonPagedDecode, plan_balanced_chunks  # noqa: E402


def _run_balanced(seq_lens, hq, hkv, d, page_size=16, dtype=torch.float16, min_chunk=16, block_n=16,
                  num_sms=132, seed=0):
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, page_size, dtype, DEV, seed=seed,
                                                extra_pages=3)
    q = torch.randn(len(seq_lens), hq, d, generator=torch.Generator().manual_seed(seed + 1)).to(dtype).to(DEV)
    ker = BalancedTritonPagedDecode(num_sms=num_sms, block_n=block_n, min_chunk=min_chunk).plan(
        indptr, indices, last, hq, hkv, d, page_size, q_dtype=dtype)
    o, lse = ker.run(q, k, v)
    o_ref, lse_ref = paged_decode_reference(q, k, v, indptr, indices, last)
    return o, lse, o_ref, lse_ref, ker


@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2), (28, 4), (8, 1), (48, 2)])
def test_balanced_gqa_groups(hq, hkv):
    o, lse, o_ref, lse_ref, ker = _run_balanced(LENS, hq, hkv, 64)
    assert ker.num_work > len(LENS)                  # long requests really were split into chunks
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_balanced_skewed_batch():
    # one long request and many short ones: the case uniform splitting handles badly
    lens = [700] + [20] * 7
    o, lse, o_ref, lse_ref, ker = _run_balanced(lens, 8, 2, 64, num_sms=8)
    counts = (ker.chunk_indptr[1:] - ker.chunk_indptr[:-1]).cpu().tolist()
    assert counts[0] > max(counts[1:])               # the long request got more chunks
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_balanced_head_dim_128_page_size_64():
    o, lse, o_ref, lse_ref, _ = _run_balanced([1, 63, 64, 65, 300], 8, 2, 128, page_size=64)
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[torch.float16])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def test_plan_balanced_chunks():
    req, start, end, cind, chunk = plan_balanced_chunks([32768] + [2048] * 15, 8, 132, 64, 256)
    assert chunk % 64 == 0
    assert cind[-1] == len(req) and cind[0] == 0
    # chunks tile each request exactly, in order, with no gaps or overlap
    for b, L in enumerate([32768] + [2048] * 15):
        ranges = [(start[i], end[i]) for i in range(cind[b], cind[b + 1])]
        assert ranges[0][0] == 0 and ranges[-1][1] == L
        assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    # the 32k request gets ~16x the chunks of each 2k request
    n0, n1 = cind[1] - cind[0], cind[2] - cind[1]
    assert n0 >= 8 * n1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_balanced_matches_flashinfer_skewed():
    flashinfer = pytest.importorskip("flashinfer")
    seq_lens, hq, hkv, d, ps = [8192] + [512] * 15, 32, 8, 128, 16
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, ps, torch.float16, "cuda", seed=9)
    q = torch.randn(len(seq_lens), hq, d, device="cuda", dtype=torch.float16)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
    w.plan(indptr, indices, last, hq, hkv, d, ps, q_data_type=torch.float16, kv_data_type=torch.float16)
    o_fi = w.run(q, (k, v))
    o, _ = BalancedTritonPagedDecode().plan(indptr, indices, last, hq, hkv, d, ps, q_dtype=torch.float16).run(q, k, v)
    torch.testing.assert_close(o.float(), o_fi.float(), **TOL[torch.float16])


# ---------------------------------------------------------------- version 2: wave planner, blocked merge, direct
from attn.triton_decode import BalancedTritonPagedDecodeV2, plan_wave_chunks  # noqa: E402


def _run_v2(seq_lens, hq, hkv, d, page_size=16, dtype=torch.float16, seed=0, **kw):
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, page_size, dtype, DEV, seed=seed,
                                                extra_pages=3)
    q = torch.randn(len(seq_lens), hq, d, generator=torch.Generator().manual_seed(seed + 1)).to(dtype).to(DEV)
    kw = {**dict(num_sms=4, ctas_per_sm=1, block_n=16, min_chunk=16, split_penalty=0), **kw}
    ker = BalancedTritonPagedDecodeV2(**kw).plan(indptr, indices, last, hq, hkv, d, page_size, q_dtype=dtype)
    o, lse = ker.run(q, k, v)
    o_ref, lse_ref = paged_decode_reference(q, k, v, indptr, indices, last)
    return o, lse, o_ref, lse_ref, ker


def _close(o, lse, o_ref, lse_ref, dtype=torch.float16):
    torch.testing.assert_close(o.float(), o_ref.float(), **TOL[dtype])
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2), (28, 4), (8, 1), (48, 2)])
def test_v2_gqa_groups(hq, hkv):
    o, lse, o_ref, lse_ref, ker = _run_v2(LENS, hq, hkv, 64)
    assert ker.num_work > len(LENS) and not ker.is_direct
    _close(o, lse, o_ref, lse_ref)


def test_v2_blocked_merge_many_chunks():
    # 45 chunks for one request: crosses several CHUNK_BLOCK=16 blocks, last block partly masked
    o, lse, o_ref, lse_ref, ker = _run_v2([45 * 16, 17], 8, 2, 64, chunk=16)
    assert ker.chunk_indptr[1].item() == 45 and ker.merge == "blocked"
    _close(o, lse, o_ref, lse_ref)


def test_v2_blocked_merge_large_logits():
    # chunks whose LSEs differ by a lot: the rescale e^(m - m_new) must keep this finite and exact
    seq_lens, hq, hkv, d = [40 * 16], 8, 2, 64
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, 16, torch.float16, DEV, seed=3)
    k[..., :] *= 8.0
    q = (torch.randn(1, hq, d, generator=torch.Generator().manual_seed(4)) * 8).to(torch.float16).to(DEV)
    ker = BalancedTritonPagedDecodeV2(num_sms=4, ctas_per_sm=1, block_n=16, chunk=16).plan(
        indptr, indices, last, hq, hkv, d, 16, q_dtype=torch.float16)
    o, lse = ker.run(q, k, v)
    o_ref, lse_ref = paged_decode_reference(q, k, v, indptr, indices, last)
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)


def test_v2_direct_path():
    # a big split penalty means no request is split: one program per (request, KV head), no merge
    o, lse, o_ref, lse_ref, ker = _run_v2(LENS, 8, 2, 64, split_penalty=10**9)
    assert ker.is_direct and ker.num_work == len(LENS)
    _close(o, lse, o_ref, lse_ref)


def test_v2_skewed_batch():
    o, lse, o_ref, lse_ref, ker = _run_v2([700] + [20] * 7, 8, 2, 64, num_sms=8)
    counts = (ker.chunk_indptr[1:] - ker.chunk_indptr[:-1]).cpu().tolist()
    assert counts[0] > max(counts[1:])
    _close(o, lse, o_ref, lse_ref)


def test_v2_head_dim_128_page_size_64():
    o, lse, o_ref, lse_ref, _ = _run_v2([1, 63, 64, 65, 300], 8, 2, 128, page_size=64)
    _close(o, lse, o_ref, lse_ref)


def test_plan_wave_chunks_divides_evenly_and_fills_a_wave():
    # 16 x 32k, 8 KV heads, 132 SMs x 4 programs: version 1 cut each request into 15936+15936+896;
    # the wave planner picks 8192 = four even chunks, 64 items x 8 heads = 512 programs <= 528 slots
    req, start, end, cind, chunk = plan_wave_chunks([32768] * 16, 8, 132 * 4, 64, 256)
    assert chunk == 8192 and len(req) == 64
    _, _, _, _, chunk1 = plan_balanced_chunks([32768] * 16, 8, 132, 64, 256)
    assert 32768 % chunk1 != 0                       # the version 1 chunk leaves a short tail
    # tiling is exact for skewed batches too
    lens = [32768] + [2048] * 15
    req, start, end, cind, chunk = plan_wave_chunks(lens, 8, 132 * 4, 64, 256)
    for b, L in enumerate(lens):
        ranges = [(start[i], end[i]) for i in range(cind[b], cind[b + 1])]
        assert ranges[0][0] == 0 and ranges[-1][1] == L
        assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert chunk % 64 == 0 and chunk >= 256


def test_plan_wave_split_penalty_prefers_no_split_for_short_requests():
    _, _, _, cind, _ = plan_wave_chunks([512], 1, 132 * 4, 64, 256, split_penalty=0)
    assert cind[-1] == 2                              # without a penalty: split in two
    _, _, _, cind, _ = plan_wave_chunks([512], 1, 132 * 4, 64, 256, split_penalty=512)
    assert cind[-1] == 1                              # merge cost counted: no split, direct path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_v2_occupancy_query_on_gpu():
    o, lse, o_ref, lse_ref, ker = _run_v2([4096, 300], 32, 8, 128, num_sms=None, ctas_per_sm=None,
                                          block_n=64, min_chunk=256)
    info = ker.describe()
    print("\noccupancy:", {k: info.get(k) for k in ("ctas_per_sm", "n_regs", "smem", "n_spills",
                                                     "occupancy_error")})
    # if this Triton version's warmup API differs, the planner falls back to 4 programs per SM (the value
    # Nsight measured on H100); that is reported, not a correctness failure, so it does not stop the run
    if "occupancy_error" in info:
        print("occupancy query failed, using fallback:", info["occupancy_error"])
    assert 1 <= info["ctas_per_sm"] <= 32
    _close(o, lse, o_ref, lse_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v2_matches_flashinfer_skewed(dtype):
    flashinfer = pytest.importorskip("flashinfer")
    seq_lens, hq, hkv, d, ps = [8192] + [512] * 15, 32, 8, 128, 16
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, ps, dtype, "cuda", seed=9)
    q = torch.randn(len(seq_lens), hq, d, device="cuda", dtype=dtype)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
    w.plan(indptr, indices, last, hq, hkv, d, ps, q_data_type=dtype, kv_data_type=dtype)
    o_fi = w.run(q, (k, v))
    o, _ = BalancedTritonPagedDecodeV2().plan(indptr, indices, last, hq, hkv, d, ps, q_dtype=dtype).run(q, k, v)
    torch.testing.assert_close(o.float(), o_fi.float(), **TOL[dtype])


# ---------------------------------------------------------------- v2: two-level (tree) merge
@pytest.mark.parametrize("hq,hkv", [(8, 1), (28, 4)])
def test_v2_tree_merge(hq, hkv):
    # 70 chunks in request 0 -> groups of 16: 16+16+16+16+6 (last group partial); request 1 has 2
    # chunks (one partial group); request 2 is a single chunk (its group is one chunk)
    o, lse, o_ref, lse_ref, ker = _run_v2([70 * 16, 20, 9], hq, hkv, 64, chunk=16, tree_min_chunks=8,
                                          tree_group=16)
    assert ker.is_tree and ker.num_groups == 5 + 1 + 1
    assert ker.group_req_indptr.tolist() == [0, 5, 6, 7]
    _close(o, lse, o_ref, lse_ref)


def test_v2_tree_off_below_threshold():
    _, _, _, _, ker = _run_v2([20 * 16], 8, 2, 64, chunk=16, tree_min_chunks=64)
    assert not ker.is_tree


def test_v2_tree_large_logits():
    seq_lens, hq, hkv, d = [100 * 16], 8, 1, 64
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, 16, torch.float16, DEV, seed=5)
    k[..., :] *= 8.0
    q = (torch.randn(1, hq, d, generator=torch.Generator().manual_seed(6)) * 8).to(torch.float16).to(DEV)
    ker = BalancedTritonPagedDecodeV2(num_sms=4, ctas_per_sm=1, block_n=16, chunk=16, tree_min_chunks=8,
                                      tree_group=16).plan(indptr, indices, last, hq, hkv, d, 16,
                                                          q_dtype=torch.float16)
    assert ker.is_tree
    o, lse = ker.run(q, k, v)
    o_ref, lse_ref = paged_decode_reference(q, k, v, indptr, indices, last)
    assert torch.isfinite(o.float()).all()
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_v2_tree_matches_flashinfer_long_single_head():
    # the shape that motivated the tree: one KV head, one long request, hundreds of chunks
    flashinfer = pytest.importorskip("flashinfer")
    seq_lens, hq, hkv, d, ps = [65536], 8, 1, 128, 16
    k, v, indptr, indices, last = make_paged_kv(seq_lens, hkv, d, ps, torch.float16, "cuda", seed=11)
    q = torch.randn(1, hq, d, device="cuda", dtype=torch.float16)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
    w.plan(indptr, indices, last, hq, hkv, d, ps, q_data_type=torch.float16, kv_data_type=torch.float16)
    o_fi = w.run(q, (k, v))
    ker = BalancedTritonPagedDecodeV2().plan(indptr, indices, last, hq, hkv, d, ps, q_dtype=torch.float16)
    assert ker.is_tree
    o, _ = ker.run(q, k, v)
    torch.testing.assert_close(o.float(), o_fi.float(), **TOL[torch.float16])
