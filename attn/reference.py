"""Plain PyTorch reference for paged-KV GQA decode attention.

This is the oracle the Triton kernel is tested against. It is written for clarity, not speed:
gather each request's pages into a contiguous K/V, then do attention in fp32.

Layout (same as FlashInfer's NHD paged layout, K and V as separate tensors):
    q          [batch, num_qo_heads, head_dim]
    k_cache    [num_pages, page_size, num_kv_heads, head_dim]
    v_cache    [num_pages, page_size, num_kv_heads, head_dim]
    kv_indptr  [batch + 1]  int32, request b owns kv_indices[kv_indptr[b]:kv_indptr[b+1]]
    kv_indices [total_pages] int32, page ids (any order)
    kv_last_page_len [batch] int32, number of valid tokens in each request's last page (1..page_size)

Returns
    o   [batch, num_qo_heads, head_dim]  in q.dtype
    lse [batch, num_qo_heads]            fp32, natural-log log-sum-exp of the scaled scores
"""
import math

import torch


def seq_lens(kv_indptr, kv_last_page_len, page_size):
    """Tokens per request: (pages - 1) * page_size + last_page_len."""
    pages = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int64)
    return (pages - 1) * page_size + kv_last_page_len.to(torch.int64)


def gather_kv(cache, kv_indptr, kv_indices, kv_last_page_len, b):
    """Contiguous [seq_len, num_kv_heads, head_dim] view of request b's tokens."""
    page_size = cache.shape[1]
    start, end = int(kv_indptr[b]), int(kv_indptr[b + 1])
    pages = kv_indices[start:end].long()
    tokens = cache[pages].reshape(-1, cache.shape[2], cache.shape[3])  # [pages*page_size, Hkv, D]
    n = (end - start - 1) * page_size + int(kv_last_page_len[b])
    return tokens[:n]


def paged_decode_reference(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale=None):
    batch, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    assert num_qo_heads % num_kv_heads == 0, "GQA needs num_qo_heads to be a multiple of num_kv_heads"
    group = num_qo_heads // num_kv_heads
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    lse = torch.empty(batch, num_qo_heads, dtype=torch.float32, device=q.device)
    for b in range(batch):
        k = gather_kv(k_cache, kv_indptr, kv_indices, kv_last_page_len, b).float()  # [L, Hkv, D]
        v = gather_kv(v_cache, kv_indptr, kv_indices, kv_last_page_len, b).float()
        # query head h reads kv head h // group
        k = k.repeat_interleave(group, dim=1)                                        # [L, Hq, D]
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,lhd->hl", q[b].float(), k) * scale                 # [Hq, L]
        lse[b] = torch.logsumexp(scores, dim=-1)
        p = torch.softmax(scores, dim=-1)
        o[b] = torch.einsum("hl,lhd->hd", p, v).to(q.dtype)
    return o, lse
