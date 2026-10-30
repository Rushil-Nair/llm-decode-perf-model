# Correlation: model vs measured

Generated 2026-10-29T23:01:46+00:00 by `correlate/correlate.py` from `data` (backend: perfsim, model: qwen3-8b). GPUs: h100_sxm, rtx5090.

Error % = 100 * (predicted - measured) / measured. For times, negative = model too optimistic; for GB/s and tok/s, positive = model too optimistic.

## Headline

| GPU | quantity | model | n | MAPE % | max abs err % | Pearson r |
|---|---|---|---|---|---|---|
| h100_sxm | L1 GEMV achieved BW (plain Little's Law) | measured probes | 168 | 336.5 | 732.4 | 0.147 |
| h100_sxm | L1 GEMV achieved BW (+ loaded latency) | alpha from stream probe (GEMV out-of-sample) | 168 | 261.1 | 666.4 | 0.184 |
| rtx5090 | L1 GEMV achieved BW (plain Little's Law) | measured probes | 146 | 79.3 | 224.6 | -0.000 |
| rtx5090 | L1 GEMV achieved BW (+ loaded latency) | alpha from stream probe (GEMV out-of-sample) | 146 | 69.5 | 193.7 | 0.190 |
| h100_sxm | vLLM TPOT (ms) | calibrated (own GPU) | 7 | 23.4 | 28.7 | 0.989 |
| h100_sxm | vLLM TTFT (ms) | calibrated (own GPU) | 10 | 24.6 | 58.8 | 1.000 |
| h100_sxm | llama.cpp slot tok/s | calibrated (own GPU) | 8 | 208.8 | 376.4 | 0.989 |
| h100_sxm | llama.cpp fix/master speedup | calibrated (own GPU) | 4 | 35.6 | 76.9 | 0.987 |
| rtx5090 | vLLM TPOT (ms) | calibrated (own GPU) | 7 | 11.3 | 32.4 | 0.799 |
| rtx5090 | vLLM TTFT (ms) | calibrated (own GPU) | 10 | 3.5 | 6.5 | 1.000 |
| rtx5090 | llama.cpp slot tok/s | calibrated (own GPU) | 8 | 77.3 | 154.3 | 0.993 |
| rtx5090 | llama.cpp fix/master speedup | calibrated (own GPU) | 4 | 20.1 | 46.1 | 0.991 |

Leave-one-GPU-out **projections** (calibrated on another GPU; target from its spec sheet only):

| target GPU | calibrated on | quantity | n | MAPE % | max abs err % | Pearson r |
|---|---|---|---|---|---|---|
| h100_sxm | rtx5090 | vLLM TPOT (ms) | 7 | 8.3 | 21.7 | 0.869 |
| h100_sxm | rtx5090 | vLLM TTFT (ms) | 10 | 43.5 | 68.0 | 1.000 |
| h100_sxm | rtx5090 | llama.cpp slot tok/s | 8 | 259.1 | 450.7 | 0.989 |
| h100_sxm | rtx5090 | llama.cpp fix/master speedup | 4 | 34.3 | 75.4 | 0.984 |
| rtx5090 | h100_sxm | vLLM TPOT (ms) | 7 | 23.6 | 32.7 | 0.983 |
| rtx5090 | h100_sxm | vLLM TTFT (ms) | 10 | 29.4 | 43.0 | 1.000 |
| rtx5090 | h100_sxm | llama.cpp slot tok/s | 8 | 56.8 | 126.3 | 0.993 |
| rtx5090 | h100_sxm | llama.cpp fix/master speedup | 4 | 21.4 | 47.4 | 0.994 |

## Level 1: GEMV achieved bandwidth (Little's Law)

- h100_sxm: SMs=132, DRAM latency=275.3 ns (pointer chase), peak=3141 GB/s (stream max), alpha=0.636 (fit on stream probe (GEMV out-of-sample))
- rtx5090: SMs=170, DRAM latency=128.8 ns (pointer chase), peak=1700 GB/s (stream max), alpha=0.717 (fit on stream probe (GEMV out-of-sample))

| GPU | kernel | n | MAPE plain % | MAPE loaded % | max abs plain % | max abs loaded % | r plain | r loaded |
|---|---|---|---|---|---|---|---|---|
| h100_sxm | v2 | 66 | 558.9 | 452.7 | 732.4 | 666.4 | 0.332 | 0.421 |
| h100_sxm | v3 | 51 | 202.3 | 145.0 | 443.7 | 305.0 | 0.341 | 0.532 |
| h100_sxm | v4 | 51 | 182.8 | 129.4 | 384.0 | 268.5 | 0.375 | 0.563 |
| h100_sxm | all | 168 | 336.5 | 261.1 | 732.4 | 666.4 | 0.147 | 0.184 |
| rtx5090 | v2 | 54 | 135.0 | 122.7 | 224.6 | 193.7 | -0.000 | 0.441 |
| rtx5090 | v3 | 46 | 51.4 | 42.9 | 113.7 | 93.0 | -0.000 | 0.377 |
| rtx5090 | v4 | 46 | 41.7 | 33.8 | 119.3 | 81.0 | 0.000 | 0.350 |
| rtx5090 | all | 146 | 79.3 | 69.5 | 224.6 | 193.7 | -0.000 | 0.190 |

Per-configuration rows: `level1_gemv.csv`.

## Level 2: % of speed-of-light per op

SOL = max(FLOPs / datasheet peak, bytes / datasheet bandwidth); % SOL = SOL time / measured time.

| GPU | op | dtype | n | median % SOL | min | max | mostly | Level 2 model MAPE % (own calib, in-sample) | max abs % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | attn_decode | bf16 | 51 | 29.1 | 0.0 | 95.4 | memory | 29.4 | 93.9 |
| h100_sxm | attn_prefill | bf16 | 12 | 15.0 | 0.1 | 53.2 | compute | 26.1 | 68.5 |
| h100_sxm | gemm | bf16 | 131 | 62.3 | 0.2 | 91.3 | memory | 0.6 | 12.8 |
| h100_sxm | rmsnorm | bf16 | 12 | 0.5 | 0.0 | 69.3 | memory | 35.1 | 89.7 |
| rtx5090 | attn_decode | bf16 | 50 | 52.8 | 0.1 | 92.1 | memory | 31.2 | 95.0 |
| rtx5090 | attn_prefill | bf16 | 12 | 33.5 | 0.2 | 99.8 | compute | 15.1 | 57.0 |
| rtx5090 | gemm | bf16 | 131 | 78.1 | 0.5 | 107.0 | memory | 1.0 | 14.2 |
| rtx5090 | rmsnorm | bf16 | 12 | 1.4 | 0.0 | 97.3 | memory | 34.8 | 83.5 |

Per-shape rows: `level2_ops_sol.csv`.

Level 2 leave-one-GPU-out **projection** (efficiency curves from another GPU on this GPU's spec sheet):

| target | calibrated on | op | dtype | n | MAPE % | max abs % |
|---|---|---|---|---|---|---|
| rtx5090 | h100_sxm | attn_decode | bf16 | 50 | 38.3 | 94.1 |
| rtx5090 | h100_sxm | attn_prefill | bf16 | 12 | 36.2 | 86.4 |
| rtx5090 | h100_sxm | gemm | bf16 | 131 | 21.7 | 66.0 |
| rtx5090 | h100_sxm | rmsnorm | bf16 | 12 | 44.8 | 80.4 |
| h100_sxm | rtx5090 | attn_decode | bf16 | 51 | 42.8 | 94.9 |
| h100_sxm | rtx5090 | attn_prefill | bf16 | 12 | 46.1 | 83.0 |
| h100_sxm | rtx5090 | gemm | bf16 | 131 | 18.1 | 41.1 |
| h100_sxm | rtx5090 | rmsnorm | bf16 | 12 | 49.8 | 91.4 |

## Level 3: vLLM TPOT and TTFT, with per-op attribution

Shares = fraction of the *predicted* time in weights GEMV/GEMM / attention / norms+elementwise / overhead (engine time not explained by per-op costs).

| GPU | model | metric | batch | input_len | out | measured ms | predicted ms | err % | shares W/A/N/O % |
|---|---|---|---|---|---|---|---|---|---|
| h100_sxm | calibrated (own GPU) | TPOT | 1 | 512 | 128 | 6.492 | 8.204 | +26.4 | 84 / 5 / 10 / 0 |
| h100_sxm | calibrated (own GPU) | TPOT | 2 | 512 | 128 | 6.344 | 8.165 | +28.7 | 85 / 6 / 10 / 0 |
| h100_sxm | calibrated (own GPU) | TPOT | 4 | 512 | 128 | 6.626 | 8.137 | +22.8 | 84 / 6 / 10 / 0 |
| h100_sxm | calibrated (own GPU) | TPOT | 8 | 512 | 128 | 6.740 | 8.375 | +24.3 | 82 / 8 / 9 / 0 |
| h100_sxm | calibrated (own GPU) | TPOT | 16 | 512 | 128 | 6.995 | 8.551 | +22.3 | 80 / 10 / 9 / 0 |
| h100_sxm | calibrated (own GPU) | TPOT | 32 | 512 | 128 | 7.409 | 9.138 | +23.3 | 75 / 16 / 9 / 0 |
| h100_sxm | calibrated (own GPU) | TPOT | 64 | 512 | 128 | 8.698 | 10.117 | +16.3 | 68 / 23 / 9 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 1 | 128 | 1 | 12.826 | 8.497 | -33.7 | 83 / 5 / 12 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 1 | 512 | 1 | 20.116 | 13.441 | -33.2 | 83 / 5 / 12 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 1 | 2048 | 1 | 58.867 | 47.244 | -19.7 | 85 / 7 / 9 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 1 | 8192 | 1 | 251.150 | 216.264 | -13.9 | 76 / 18 / 6 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 2 | 512 | 1 | 57.661 | 23.782 | -58.8 | 85 / 4 / 10 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 4 | 512 | 1 | 59.625 | 45.744 | -23.3 | 87 / 4 / 9 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 8 | 512 | 1 | 114.545 | 93.110 | -18.7 | 89 / 3 / 8 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 16 | 512 | 1 | 219.524 | 183.516 | -16.4 | 89 / 3 / 8 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 32 | 512 | 1 | 427.240 | 364.080 | -14.8 | 90 / 3 / 8 / 0 |
| h100_sxm | calibrated (own GPU) | TTFT | 64 | 512 | 1 | 838.660 | 724.804 | -13.6 | 90 / 2 / 7 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 1 | 512 | 128 | 10.276 | 13.609 | +32.4 | 93 / 3 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 2 | 512 | 128 | 10.578 | 11.875 | +12.3 | 92 / 3 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 4 | 512 | 128 | 10.995 | 12.108 | +10.1 | 91 / 4 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 8 | 512 | 128 | 11.244 | 12.130 | +7.9 | 89 / 6 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 16 | 512 | 128 | 11.827 | 12.295 | +4.0 | 86 / 9 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 32 | 512 | 128 | 12.839 | 13.696 | +6.7 | 80 / 15 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TPOT | 64 | 512 | 128 | 14.772 | 15.646 | +5.9 | 71 / 24 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 1 | 128 | 1 | 14.858 | 14.908 | +0.3 | 91 / 3 / 6 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 1 | 512 | 1 | 41.640 | 39.499 | -5.1 | 92 / 3 / 5 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 1 | 2048 | 1 | 158.710 | 151.682 | -4.4 | 91 / 5 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 1 | 8192 | 1 | 681.514 | 661.901 | -2.9 | 82 / 14 / 3 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 2 | 512 | 1 | 80.558 | 75.283 | -6.5 | 93 / 2 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 4 | 512 | 1 | 152.626 | 146.334 | -4.1 | 94 / 2 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 8 | 512 | 1 | 294.190 | 288.817 | -1.8 | 95 / 2 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 16 | 512 | 1 | 580.351 | 574.395 | -1.0 | 95 / 1 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 32 | 512 | 1 | 1195.623 | 1145.360 | -4.2 | 95 / 1 / 4 / 0 |
| rtx5090 | calibrated (own GPU) | TTFT | 64 | 512 | 1 | 2392.161 | 2286.438 | -4.4 | 95 / 1 / 4 / 0 |

### Per-op error attribution

Mean share of predicted time per op group, next to the error. A large error where one group dominates points at that group's cost model.

| GPU | metric | mean err % | weights % | attention % | norms % | overhead % | worst row (dominant group) |
|---|---|---|---|---|---|---|---|
| h100_sxm | TPOT | +23.4 | 80 | 11 | 9 | 0 | b=2 in=512: +28.7% (weights) |
| h100_sxm | TTFT | -24.6 | 86 | 5 | 9 | 0 | b=2 in=512: -58.8% (weights) |
| rtx5090 | TPOT | +11.3 | 86 | 9 | 5 | 0 | b=1 in=512: +32.4% (weights) |
| rtx5090 | TTFT | -3.4 | 92 | 4 | 4 | 0 | b=2 in=512: -6.5% (weights) |

## Level 3: llama.cpp slot test (master vs slot-order fix)

Predicted tok/s = users / decode_step_time(engine='llamacpp', passes_per_step=k) at an assumed context of 512 tokens; k from the DIAG/CPU counters.

| GPU | model | build | users | k (passes/step) | k source | measured tok/s | predicted tok/s | err % |
|---|---|---|---|---|---|---|---|---|
| h100_sxm | calibrated (own GPU) | fix | 1 | 1.00 | DIAG counter | 204.2 | 373.1 | +82.7 |
| h100_sxm | calibrated (own GPU) | fix | 4 | 1.03 | DIAG counter | 466.7 | 1464.6 | +213.9 |
| h100_sxm | calibrated (own GPU) | fix | 8 | 1.12 | DIAG counter | 575.6 | 2547.2 | +342.5 |
| h100_sxm | calibrated (own GPU) | fix | 16 | 1.76 | DIAG counter | 769.7 | 3167.2 | +311.5 |
| h100_sxm | calibrated (own GPU) | master | 1 | 1.00 | DIAG counter | 201.6 | 373.1 | +85.0 |
| h100_sxm | calibrated (own GPU) | master | 4 | 2.79 | DIAG counter | 258.1 | 538.0 | +108.5 |
| h100_sxm | calibrated (own GPU) | master | 8 | 3.81 | DIAG counter | 316.9 | 792.9 | +150.2 |
| h100_sxm | calibrated (own GPU) | master | 16 | 1.49 | DIAG counter | 775.2 | 3693.3 | +376.4 |
| rtx5090 | calibrated (own GPU) | fix | 1 | 1.00 | DIAG counter | 237.3 | 279.5 | +17.8 |
| rtx5090 | calibrated (own GPU) | fix | 4 | 1.01 | DIAG counter | 629.4 | 1081.3 | +71.8 |
| rtx5090 | calibrated (own GPU) | fix | 8 | 1.08 | DIAG counter | 842.3 | 1899.7 | +125.5 |
| rtx5090 | calibrated (own GPU) | fix | 16 | 1.58 | DIAG counter | 1026.3 | 2515.1 | +145.1 |
| rtx5090 | calibrated (own GPU) | master | 1 | 1.00 | DIAG counter | 237.2 | 279.5 | +17.8 |
| rtx5090 | calibrated (own GPU) | master | 4 | 2.93 | DIAG counter | 290.2 | 381.2 | +31.3 |
| rtx5090 | calibrated (own GPU) | master | 8 | 3.53 | DIAG counter | 408.0 | 629.7 | +54.3 |
| rtx5090 | calibrated (own GPU) | master | 16 | 1.49 | DIAG counter | 1042.5 | 2650.7 | +154.3 |

Fix speedup (fix tok/s / master tok/s):

| GPU | users | master tok/s | fix tok/s | measured speedup | predicted speedup | err % |
|---|---|---|---|---|---|---|
| h100_sxm | 1 | 201.6 | 204.2 | 1.01x | 1.00x | -1.3 |
| h100_sxm | 4 | 258.1 | 466.7 | 1.81x | 2.72x | +50.5 |
| h100_sxm | 8 | 316.9 | 575.6 | 1.82x | 3.21x | +76.9 |
| h100_sxm | 16 | 775.2 | 769.7 | 0.99x | 0.86x | -13.6 |
| rtx5090 | 1 | 237.2 | 237.3 | 1.00x | 1.00x | -0.0 |
| rtx5090 | 4 | 290.2 | 629.4 | 2.17x | 2.84x | +30.8 |
| rtx5090 | 8 | 408.0 | 842.3 | 2.06x | 3.02x | +46.1 |
| rtx5090 | 16 | 1042.5 | 1026.3 | 0.98x | 0.95x | -3.6 |

## Leave-one-GPU-out: Projection (not measured on the calibration GPU)

Efficiency curves from GPU A's calib.json; GPU B described only by its hw/hardware.yaml spec sheet. B's measurements are used only to score the projection.

| target | calibrated on | metric | batch | input_len | measured ms | projected ms | err % | shares W/A/N/O % |
|---|---|---|---|---|---|---|---|---|
| rtx5090 | h100_sxm | TPOT | 1 | 512 | 10.276 | 13.639 | +32.7 | 89 / 4 / 7 / 0 |
| rtx5090 | h100_sxm | TPOT | 2 | 512 | 10.578 | 13.566 | +28.2 | 89 / 5 / 6 / 0 |
| rtx5090 | h100_sxm | TPOT | 4 | 512 | 10.995 | 13.514 | +22.9 | 89 / 5 / 6 / 0 |
| rtx5090 | h100_sxm | TPOT | 8 | 512 | 11.244 | 13.959 | +24.1 | 87 / 8 / 6 / 0 |
| rtx5090 | h100_sxm | TPOT | 16 | 512 | 11.827 | 14.288 | +20.8 | 84 / 10 / 6 / 0 |
| rtx5090 | h100_sxm | TPOT | 32 | 512 | 12.839 | 15.385 | +19.8 | 78 / 16 / 6 / 0 |
| rtx5090 | h100_sxm | TPOT | 64 | 512 | 14.772 | 17.215 | +16.5 | 70 / 25 / 6 / 0 |
| rtx5090 | h100_sxm | TTFT | 1 | 128 | 14.858 | 14.635 | -1.5 | 88 / 4 / 8 / 0 |
| rtx5090 | h100_sxm | TTFT | 1 | 512 | 41.640 | 51.541 | +23.8 | 93 / 3 / 5 / 0 |
| rtx5090 | h100_sxm | TTFT | 1 | 2048 | 158.710 | 204.842 | +29.1 | 90 / 7 / 3 / 0 |
| rtx5090 | h100_sxm | TTFT | 1 | 8192 | 681.514 | 974.746 | +43.0 | 79 / 18 / 3 / 0 |
| rtx5090 | h100_sxm | TTFT | 2 | 512 | 80.558 | 97.356 | +20.9 | 94 / 2 / 4 / 0 |
| rtx5090 | h100_sxm | TTFT | 4 | 512 | 152.626 | 195.088 | +27.8 | 94 / 2 / 4 / 0 |
| rtx5090 | h100_sxm | TTFT | 8 | 512 | 294.190 | 406.973 | +38.3 | 95 / 2 / 3 / 0 |
| rtx5090 | h100_sxm | TTFT | 16 | 512 | 580.351 | 810.450 | +39.6 | 95 / 2 / 3 / 0 |
| rtx5090 | h100_sxm | TTFT | 32 | 512 | 1195.623 | 1616.719 | +35.2 | 95 / 2 / 3 / 0 |
| rtx5090 | h100_sxm | TTFT | 64 | 512 | 2392.161 | 3228.146 | +34.9 | 95 / 2 / 3 / 0 |
| h100_sxm | rtx5090 | TPOT | 1 | 512 | 6.492 | 7.901 | +21.7 | 89 / 3 / 8 / 0 |
| h100_sxm | rtx5090 | TPOT | 2 | 512 | 6.344 | 6.973 | +9.9 | 87 / 4 / 9 / 0 |
| h100_sxm | rtx5090 | TPOT | 4 | 512 | 6.626 | 7.098 | +7.1 | 87 / 4 / 9 / 0 |
| h100_sxm | rtx5090 | TPOT | 8 | 512 | 6.740 | 7.110 | +5.5 | 85 / 7 / 9 / 0 |
| h100_sxm | rtx5090 | TPOT | 16 | 512 | 6.995 | 7.198 | +2.9 | 82 / 9 / 9 / 0 |
| h100_sxm | rtx5090 | TPOT | 32 | 512 | 7.409 | 7.947 | +7.3 | 77 / 14 / 8 / 0 |
| h100_sxm | rtx5090 | TPOT | 64 | 512 | 8.698 | 8.991 | +3.4 | 69 / 23 / 8 / 0 |
| h100_sxm | rtx5090 | TTFT | 1 | 128 | 12.826 | 8.350 | -34.9 | 86 / 4 / 9 / 0 |
| h100_sxm | rtx5090 | TTFT | 1 | 512 | 20.116 | 10.338 | -48.6 | 81 / 6 / 13 / 0 |
| h100_sxm | rtx5090 | TTFT | 1 | 2048 | 58.867 | 35.232 | -40.2 | 85 / 6 / 10 / 0 |
| h100_sxm | rtx5090 | TTFT | 1 | 8192 | 251.150 | 148.359 | -40.9 | 78 / 14 / 8 / 0 |
| h100_sxm | rtx5090 | TTFT | 2 | 512 | 57.661 | 18.429 | -68.0 | 85 / 4 / 11 / 0 |
| h100_sxm | rtx5090 | TTFT | 4 | 512 | 59.625 | 34.481 | -42.2 | 87 / 4 / 10 / 0 |
| h100_sxm | rtx5090 | TTFT | 8 | 512 | 114.545 | 66.635 | -41.8 | 88 / 3 / 9 / 0 |
| h100_sxm | rtx5090 | TTFT | 16 | 512 | 219.524 | 131.020 | -40.3 | 89 / 2 / 9 / 0 |
| h100_sxm | rtx5090 | TTFT | 32 | 512 | 427.240 | 259.721 | -39.2 | 89 / 2 / 9 / 0 |
| h100_sxm | rtx5090 | TTFT | 64 | 512 | 838.660 | 516.814 | -38.4 | 89 / 2 / 9 / 0 |

| target | calibrated on | build | users | k | measured tok/s | projected tok/s | err % |
|---|---|---|---|---|---|---|---|
| rtx5090 | h100_sxm | fix | 1 | 1.00 | 237.3 | 242.3 | +2.1 |
| rtx5090 | h100_sxm | fix | 4 | 1.01 | 629.4 | 972.2 | +54.5 |
| rtx5090 | h100_sxm | fix | 8 | 1.08 | 842.3 | 1690.3 | +100.7 |
| rtx5090 | h100_sxm | fix | 16 | 1.58 | 1026.3 | 2237.6 | +118.0 |
| rtx5090 | h100_sxm | master | 1 | 1.00 | 237.2 | 242.3 | +2.1 |
| rtx5090 | h100_sxm | master | 4 | 2.93 | 290.2 | 332.8 | +14.7 |
| rtx5090 | h100_sxm | master | 8 | 3.53 | 408.0 | 555.5 | +36.2 |
| rtx5090 | h100_sxm | master | 16 | 1.49 | 1042.5 | 2359.3 | +126.3 |
| h100_sxm | rtx5090 | fix | 1 | 1.00 | 204.2 | 442.1 | +116.5 |
| h100_sxm | rtx5090 | fix | 4 | 1.03 | 466.7 | 1684.3 | +260.9 |
| h100_sxm | rtx5090 | fix | 8 | 1.12 | 575.6 | 2946.4 | +411.9 |
| h100_sxm | rtx5090 | fix | 16 | 1.76 | 769.7 | 3666.4 | +376.3 |
| h100_sxm | rtx5090 | master | 1 | 1.00 | 201.6 | 442.1 | +119.3 |
| h100_sxm | rtx5090 | master | 4 | 2.79 | 258.1 | 633.2 | +145.4 |
| h100_sxm | rtx5090 | master | 8 | 3.81 | 316.9 | 925.1 | +191.9 |
| h100_sxm | rtx5090 | master | 16 | 1.49 | 775.2 | 4269.3 | +450.7 |

## Notes (skipped inputs, assumptions)

- llama.cpp context assumed 512 tokens (prompt lengths not logged).
- TPOT = (lat(out=OL) - lat(out=1)) / (OL - 1); TTFT ~= lat(out=1); predictions use the same formula on predict_vllm_latency, so both sides include the same terms.
- Attribution: engine per-op breakdown (or OpCostModel on bf16 perfsim.models ops) at the mean decode context input_len + (output_len-2)/2; overhead = per-pass constant / unexplained engine time.
