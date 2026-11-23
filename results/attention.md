## Paged GQA decode attention vs FlashInfer

Speed = FlashInfer time / our time (above 1: ours is faster). FlashInfer = the faster of its tensor-core and CUDA-core decode paths per shape. Uniform: every request has the same length; skewed: one request at the full context, the rest at 1/16 of it.

| GPU | setting | kernel | uniform batches | skewed batches |
|---|---|---|---|---|
| h100_sxm | page 16, float16 | triton_splitkv | 0.98x (0.56-1.11, n=67) | 0.48x (0.19-0.99, n=35) |
| h100_sxm | page 16, float16 | triton_balanced | 0.96x (0.76-1.09, n=67) | 0.97x (0.82-1.75, n=35) |
| h100_sxm | page 16, float16 | triton_balanced_v2 | 1.05x (0.90-1.26, n=67) | 1.10x (0.88-1.84, n=35) |
| h100_sxm | page 16, bfloat16 | triton_splitkv | 1.02x (0.95-1.15, n=16) | 0.39x (0.24-0.95, n=9) |
| h100_sxm | page 16, bfloat16 | triton_balanced | 1.00x (0.94-1.13, n=16) | 1.02x (0.95-1.13, n=9) |
| h100_sxm | page 16, bfloat16 | triton_balanced_v2 | 1.06x (0.96-1.25, n=16) | 1.11x (0.94-1.34, n=9) |
| h100_sxm | page 64, float16 | triton_splitkv | 1.01x (0.93-1.11, n=32) | 0.45x (0.24-1.01, n=18) |
| h100_sxm | page 64, float16 | triton_balanced | 0.99x (0.89-1.09, n=32) | 1.06x (0.95-1.72, n=18) |
| h100_sxm | page 64, float16 | triton_balanced_v2 | 1.06x (0.96-1.23, n=32) | 1.13x (0.88-1.76, n=18) |

## Effect of each scheduling change (same kernel body, page 16 fp16)

| GPU | comparison | uniform batches | skewed batches |
|---|---|---|---|
| h100_sxm | balanced v1 vs uniform splitting | 0.98x (0.90-1.37, n=67) | 2.05x (0.91-4.91, n=35) |
| h100_sxm | balanced v2 vs balanced v1 | 1.09x (0.95-1.41, n=67) | 1.13x (0.91-1.36, n=35) |

## h100_sxm: the 10 shapes where balanced v1 was slowest vs FlashInfer, before and after

| head config | batch | context | skewed | v1 vs FlashInfer | v2 vs FlashInfer | FlashInfer (us) | v2 (us) | v2 work items / chunk / merge |
|---|---|---|---|---|---|---|---|---|
| tp8_70b | 1 | 131072 | no | 0.76x | 1.08x | 40.6 | 37.7 | 342 / 384 / tree |
| mha | 1 | 512 | no | 0.82x | 0.90x | 13.8 | 15.4 | 1 / 512 / direct |
| mha | 64 | 2048 | yes | 0.82x | 1.09x | 79.1 | 72.4 | 71 / 256 / no |
| tp8_70b | 4 | 8192 | yes | 0.83x | 1.07x | 15.7 | 14.7 | 38 / 256 / no |
| tp8_70b | 1 | 512 | no | 0.83x | 0.93x | 13.7 | 14.7 | 1 / 512 / direct |
| tp8_70b | 4 | 2048 | yes | 0.84x | 1.06x | 14.1 | 13.3 | 11 / 256 / no |
| mha | 16 | 32768 | yes | 0.84x | 1.00x | 347.5 | 346.9 | 31 / 2048 / no |
| mha | 16 | 8192 | yes | 0.85x | 1.00x | 95.6 | 95.7 | 31 / 512 / no |
| tp8_70b | 1 | 8192 | no | 0.85x | 1.06x | 15.8 | 14.9 | 32 / 256 / no |
| tp8_70b | 1 | 2048 | no | 0.85x | 1.08x | 13.9 | 12.8 | 8 / 256 / no |

v2 occupancy query: 3 programs per SM (166 registers/thread, 73728 B shared memory, 0 spills).

## h100_sxm: tuning and ablation on separate shapes (n=36, none in the matrix)

Geomean FlashInfer time / ours over the tuning shapes (above 1: ours is faster).

| version | speed vs FlashInfer |
|---|---|
| v1 (first H100 run) | 0.953x |
| + wave planner (occupancy-aware chunk size) | 0.977x |
| + blocked merge | 0.987x |
| + direct path (no merge when unsplit) | 0.988x |
| + two-level merge (requests with many chunks) | 0.994x |
| + tuned tile and planner (128 tokens/step, 4 warps, 2 stages, split penalty 512, overhead 256, two-level merge above 64 chunks) | 1.086x |

Tile configs (block_n_warps_stages): 128_4_2 1.085x, 128_8_3 1.049x, 128_8_2 1.027x, 64_4_3 1.027x, 64_8_3 1.000x, 64_4_2 0.995x, 64_8_2 0.989x, 32_4_3 0.941x, 32_4_2 0.924x, 32_2_2 0.874x

## Attention term: t = t0 + bytes / BW_eff (fitted on uniform batches, page 16, fp16)

Copy BW is a device-to-device copy (read + write bytes / time). Read-only streams can exceed it slightly, so values just above 100% mean 'at the memory limit', not faster than memory.

| GPU | kernel | t0 (us) | BW_eff (GB/s) | % of copy BW | in-sample MAPE | holdout MAPE (unseen head configs) | skewed batches MAPE | zero-fit roofline MAPE |
|---|---|---|---|---|---|---|---|---|
| h100_sxm | triton_splitkv | 14.8 | 3114 | 103% | 4.8% | 7.4% (max 52%, n=34) | 49.7% (n=35) | 39.3% |
| h100_sxm | triton_balanced | 15.4 | 3087 | 102% | 4.6% | 7.3% (max 32%, n=34) | 6.9% (n=35) | 40.0% |
| h100_sxm | triton_balanced_v2 | 12.9 | 3149 | 104% | 3.5% | 5.1% (max 17%, n=34) | 5.2% (n=35) | 36.8% |
| h100_sxm | flashinfer | 14.4 | 3133 | 103% | 3.3% | 5.7% (max 23%, n=34) | 9.2% (n=35) | 38.7% |
