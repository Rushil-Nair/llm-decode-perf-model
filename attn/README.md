# Paged GQA decode attention (Triton split-KV) and the attention term of the decode model

The GEMV levels of this repo model the weight-streaming part of decode. At long context the other big
cost is attention: every layer streams the whole KV cache once per generated token. This part adds

* `attn/triton_decode.py`: a paged-KV, GQA decode-attention kernel in Triton with split-KV
  (Flash-Decoding) and a log-sum-exp merge, using FlashInfer's NHD page-table format
  (`kv_indptr`, `kv_indices`, `kv_last_page_len`) so both kernels read identical inputs;
* `attn/reference.py`: a plain PyTorch reference (gather pages, fp32 softmax) used as the oracle;
* `tests/test_paged_decode.py`: correctness tests (GQA group sizes 1, 4, 7, 8, 24; ragged lengths
  around page boundaries; shuffled pages; 1-16 splits including empty splits; large logits; bf16 and
  agreement with FlashInfer on GPU);
* `bench/bench_attention.py`: CUDA-graph timing against FlashInfer `BatchDecodeWithPagedKVCacheWrapper`
  (tensor cores off and on) and PyTorch SDPA (pre-gathered and gather + SDPA), with L2 flushed by a
  read-only sweep before every replay;
* `perfsim/attn_model.py` + `bench/attn_report.py`: the attention term `t = t0 + bytes / BW_eff`,
  scored in-sample, on unseen head configurations, and leave-one-GPU-out.

## How the kernel works

Grid `(batch, num_kv_heads, num_splits)`. One program owns one KV head of one request and one slice of
its tokens. It loads the G query heads that share that KV head as one tile (padded to 16 rows for
`tl.dot`), so each K/V tile fetched from memory is used by all G heads. It walks its tokens through the
page table, keeps an online softmax in the log2 domain (running max `m`, running sum `l`, rescaling the
accumulator by `2^(m_old - m_new)` whenever the max grows), and writes a normalized partial output and
its natural-log LSE. A second kernel merges the splits:

    m = max_s LSE_s,  w_s = exp(LSE_s - m),  O = sum_s w_s O_s / sum_s w_s,  LSE = m + log sum_s w_s

Splits exist because at decode the query length is 1: without them the grid is only
`batch x num_kv_heads` programs (8 at batch 1 for Llama-8B heads), far fewer than the SMs.
`choose_num_splits` targets about two waves of programs, with no split shorter than 256 tokens.

## Two ways of splitting the work

* `TritonPagedDecode` (uniform): every request is cut into the same number of splits. The first H100
  run showed this is fine when all requests have the same length but slow on skewed batches (one long
  request plus many short ones): the long request becomes a few long-running programs while the GPU
  waits for them.
* `BalancedTritonPagedDecode` (per request): `plan()` cuts the whole batch into chunks of about equal
  token count (`plan_balanced_chunks`), so a request gets work in proportion to its length, the idea
  behind FlashInfer's `plan()`. Grid = (work items, KV heads); a second kernel merges each request's
  chunks with a running log-sum-exp merge. Both versions share the same inner attention loop
  (`_attend_tokens`), so differences between them come from scheduling only.

## Version 2 (`BalancedTritonPagedDecodeV2`): changes driven by the version 1 profile

* **Wave planner** (`plan_wave_chunks`). Version 1 aimed at 2 x SMs programs whatever fits on an SM,
  and its chunk did not divide requests evenly (a 32k request became 15936 + 15936 + 896 tokens).
  Version 2 asks the compiled kernel for its registers and shared memory, computes how many programs
  fit per SM (`programs_per_sm`, the method of Triton's fused-softmax tutorial), and picks the chunk
  that minimises waves x (chunk + per-program overhead), with a penalty for splitting at all.
* **Blocked merge** (`_blocked_merge_kernel`). The version 1 merge read one chunk per step; a request
  with 256 chunks (Llama-70B TP8 heads at 128k) is 256 dependent loads. Version 2 reads 16 chunks per
  step with the same online log-sum-exp rescale.
* **Direct path.** When no request is split, each program writes the final output and LSE itself and
  the merge kernel is not launched. Targets the small, latency-bound shapes.
* **Two-level merge** (added after the second H100 run). At Llama-70B TP8 heads, batch 1, 131k,
  Nsight showed the merge (50 us) costing more than the attention kernel (31 us): 8 programs each
  combined 512 partials. Now groups of 32 chunks are merged by many programs in parallel, then each
  request merges its groups. Used only above a tuned number of chunks per request.
* **Tuned settings.** `bench_attention.py --mode tune` picks the tile (tokens per step, warps,
  pipeline stages) and the planner's two constants on 36 shapes that are not in the benchmark
  matrix, and measures each change above on its own (the ablation in `results/attention.md`).

## Results (H100 80GB SXM, 132 SMs; torch 2.12 / CUDA 13.0, Triton 3.7.0, FlashInfer 0.7.5.post1)

Full tables: `results/attention.md`, regenerated from `data/h100_sxm/attention/` by
`python3 bench/attn_report.py`. All kernels below were measured in the same run. Speed is
FlashInfer time / our time, geomean (min-max) over the shape matrix (4 head configs, batch 1-64,
context 512-131072). FlashInfer is the faster of its tensor-core and CUDA-core decode paths for
each shape. Version 2 settings were tuned on 36 other shapes, none of them in this matrix.

| setting | kernel | uniform batches | skewed batches |
|---|---|---|---|
| page 16, fp16 | uniform splitting | 0.98x (0.56-1.11, n=67) | 0.48x (0.19-0.99, n=35) |
| page 16, fp16 | balanced v1 | 0.96x (0.76-1.09, n=67) | 0.97x (0.82-1.75, n=35) |
| page 16, fp16 | **balanced v2** | **1.05x (0.90-1.26, n=67)** | **1.10x (0.88-1.84, n=35)** |
| page 64, fp16 | balanced v2 | 1.06x (0.96-1.23, n=32) | 1.13x (0.88-1.76, n=18) |
| page 16, bf16 (Llama-8B heads) | balanced v2 | 1.06x (0.96-1.25, n=16) | 1.11x (0.94-1.34, n=9) |

* Per-request balancing fixes skewed batches: v1 vs uniform splitting is 2.05x on skewed batches.
* v2 vs v1 (same inner loop): 1.09x on uniform and 1.13x on skewed batches (min 0.91x).
* Excluding Qwen-7B heads (see caveat), v2 is 1.04x (0.90-1.20) on uniform and 1.06x (0.88-1.25)
  on skewed batches. On the 26 memory-bound shapes (FlashInfer above 150 us) it is 1.03x.
  v2 is at least as fast as FlashInfer on 73 of the 102 shapes.
* v1's worst shape (Llama-70B TP8 heads, batch 1, 131k) went from 0.76x to 1.08x (37.7 vs 40.6 us).
* Max abs error: 4.9e-4 (fp16), 3.9e-3 (bf16), the same as FlashInfer's on those shapes.
* 45 tests pass on the H100 (bf16, agreement with FlashInfer, occupancy query, two-level merge).

**What each change added** (geomean vs FlashInfer on the 36 tuning shapes):

| version | speed vs FlashInfer |
|---|---|
| v1 | 0.953x |
| + wave planner | 0.977x |
| + blocked merge | 0.987x |
| + direct path | 0.988x |
| + two-level merge | 0.994x |
| + tuned tile (128 tokens per step, 4 warps, 2 stages) and planner constants | 1.086x |

The tile is the largest single gain: 128 tokens per step (1.085x) vs 64 (0.995x) vs 32 (0.924x),
i.e. more bytes in flight per program. It uses 166 registers and 72 KB of shared memory, so 3
programs fit per SM instead of 4, and the planner adapts to that automatically. The two-level merge
matters only on few shapes (tuning picked "above 64 chunks per request"), but there it is large.

**Nsight Compute.** Version 1 at Llama-8B heads, batch 1, 32k: 256 programs, 0.48 waves, 12%
achieved occupancy, 70% of DRAM peak. Tuned v2: 0.87 waves, 16% occupancy, 76-79% of DRAM peak
(52-54 us for the attention kernel vs 57 us for FlashInfer's). At batch 16, 8k: 87-89% of DRAM peak
(181-184 us) vs FlashInfer's 75-76% (212-215 us). At Llama-70B TP8 heads, batch 1, 131k the merge
went from 50 us (8 programs combining 512 partials each) to 7 + 4 us for the two levels.

**Caveat on the best numbers.** The 1.6-1.8x wins are Qwen-7B heads (group size 7) on skewed
batches, where FlashInfer's CUDA-core path does not run (a limitation already known upstream), so
the comparison there is against its tensor-core path only.

## Remaining gaps versus FlashInfer

* Small, latency-bound shapes: MHA and TP8 heads at batch 1, 512 tokens are 0.90-0.93x (about
  1-1.5 us slower on 14 us) even with the direct path.
* Skewed batch 64 at 2k context: 0.88x of FlashInfer and 0.91-0.93x of v1; its chunks are 256
  tokens, the planner's minimum.
* The merge is still a separate launch (8 us at Llama-8B heads, batch 1, 32k).
* No FP8 KV, no persistent scheduling, tuned and measured on one GPU only.

**Attention term of the decode model** (`t = t0 + bytes / BW_eff`, fitted on uniform batches):

| kernel | t0 (us) | BW_eff | in-sample MAPE | unseen head configs | skewed batches |
|---|---|---|---|---|---|
| uniform splitting | 14.8 | 3114 GB/s | 4.8% | 7.4% | 49.7% |
| balanced v1 | 15.4 | 3087 GB/s | 4.6% | 7.3% | 6.9% |
| balanced v2 | 12.9 | 3149 GB/s | 3.5% | 5.1% | 5.2% |
| FlashInfer | 14.4 | 3133 GB/s | 3.3% | 5.7% | 9.2% |

A bytes-only model cannot predict uniform splitting on skewed batches (50% error); after balancing,
the same model fitted on uniform batches predicts skewed ones to 5-7%. BW_eff is about 103% of the
measured device-to-device copy bandwidth: the copy counts read and write traffic and a read-only
stream can go slightly faster, so this means "at the memory limit", not faster than memory.

## How this was built

Written with AI coding assistance (Claude Code). The tests were checked by deliberately breaking the
kernel (removing the accumulator rescale; reading the wrong KV head; dropping the rescale in the
balanced merge; ignoring chunk start offsets; in v2, dropping the blocked-merge rescale of the output
or of the denominator, padding masked chunks with LSE 0 instead of -inf, writing the direct path to
the wrong request, wrong group boundaries or indices in the two-level merge): each break fails tests that cover it.
