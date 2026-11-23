"""Build random paged KV caches for tests and benchmarks (FlashInfer NHD layout, K and V separate)."""
import torch


def make_paged_kv(seq_lens, num_kv_heads, head_dim, page_size, dtype, device, shuffle=True, seed=0,
                  extra_pages=0):
    """Random K/V for requests of the given lengths, scattered over pages in random order.

    Returns k_cache, v_cache [num_pages, page_size, Hkv, D], kv_indptr [B+1], kv_indices, kv_last_page_len.
    extra_pages adds unused pages so request pages are not a dense prefix of the cache.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_req = [(L + page_size - 1) // page_size for L in seq_lens]
    assert all(L >= 1 for L in seq_lens), "decode needs at least one cached token"
    total = sum(pages_per_req) + extra_pages
    order = torch.randperm(total, generator=g) if shuffle else torch.arange(total)
    kv_indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.tensor(pages_per_req, dtype=torch.int32).cumsum(0)
    kv_indices = order[: sum(pages_per_req)].to(torch.int32)
    kv_last = torch.tensor([L - (p - 1) * page_size for L, p in zip(seq_lens, pages_per_req)],
                           dtype=torch.int32)
    shape = (total, page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, generator=g).to(dtype)
    v_cache = torch.randn(shape, generator=g).to(dtype)
    to = dict(device=device)
    return (k_cache.to(**to), v_cache.to(**to), kv_indptr.to(**to), kv_indices.to(**to), kv_last.to(**to))
