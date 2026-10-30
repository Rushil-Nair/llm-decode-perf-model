# What-if projections: h100_sxm

Generated 2026-10-29T22:59:23+00:00 by `whatif/sweeps.py`. **Every number below is a model projection, not a measurement.** Cost model mode: empirical (empirical = calibrated on this GPU's ops.jsonl; sol = uncalibrated). Decode at context 1024 tokens (vLLM engine, bf16 weights); TTFT = batch 1, prompt 1024.

### Projection (not measured): HBM BW (ideal)

| BW x | tok/s b=1 | tok/s b=8 | tok/s b=32 | tok/s b=64 | speedup b=64 | TTFT ms | TTFT speedup |
|---|---|---|---|---|---|---|---|
| 0.5 | 69 | 529 | 1804 | 3031 | 0.55x | 26.5 | 0.90x |
| 0.75 | 97 | 746 | 2565 | 4345 | 0.78x | 24.7 | 0.97x |
| 1.0 | 122 | 938 | 3251 | 5548 | 1.00x | 23.9 | 1.00x |
| 1.5 | 163 | 1262 | 4436 | 7673 | 1.38x | 23.2 | 1.03x |
| 2.0 | 196 | 1526 | 5425 | 9490 | 1.71x | 22.8 | 1.05x |

### Projection (not measured): HBM BW (Little's-Law cap)

| BW x | tok/s b=1 | tok/s b=8 | tok/s b=32 | tok/s b=64 | speedup b=64 | TTFT ms | TTFT speedup |
|---|---|---|---|---|---|---|---|
| 0.5 | 71 | 547 | 1866 | 3136 | 0.57x | 26.3 | 0.91x |
| 0.75 | 98 | 758 | 2606 | 4417 | 0.80x | 24.6 | 0.97x |
| 1.0 | 122 | 938 | 3251 | 5548 | 1.00x | 23.9 | 1.00x |
| 1.5 | 159 | 1230 | 4315 | 7454 | 1.34x | 23.2 | 1.03x |
| 2.0 | 187 | 1455 | 5156 | 8991 | 1.62x | 22.9 | 1.04x |

### Projection (not measured): SM count

| SMs x | tok/s b=1 | tok/s b=8 | tok/s b=32 | tok/s b=64 | speedup b=64 | TTFT ms | TTFT speedup |
|---|---|---|---|---|---|---|---|
| 0.75 | 119 | 920 | 3186 | 5433 | 0.98x | 30.5 | 0.78x |
| 1.25 | 123 | 949 | 3290 | 5618 | 1.01x | 19.9 | 1.20x |

### Projection (not measured): FP8 weights

| weight bytes x | tok/s b=1 | tok/s b=8 | tok/s b=32 | tok/s b=64 | speedup b=64 | TTFT ms | TTFT speedup |
|---|---|---|---|---|---|---|---|
| 0.5 | 179 | 1370 | 4448 | 7098 | 1.28x | 23.7 | 1.01x |

### Projection (not measured): TP=2 (NVLink)

| GPUs | tok/s b=1 | tok/s b=8 | tok/s b=32 | tok/s b=64 | speedup b=64 | TTFT ms | TTFT speedup |
|---|---|---|---|---|---|---|---|
| 2 | 171 | 1356 | 4817 | 8506 | 1.53x | 14.0 | 1.70x |

Baseline (model, not measured): b=1: 122 tok/s, b=8: 938 tok/s, b=32: 3251 tok/s, b=64: 5548 tok/s

Little's-Law achieved-BW ratios at the best measured GEMV config (kernel v4, 48.0 warps/SM, ILP 2): BW x0.5: 0.52, BW x0.75: 0.76, BW x1.0: 1.00, BW x1.5: 1.44, BW x2.0: 1.86, SMs x0.75: 0.98, SMs x1.25: 1.02

## Notes and assumptions

- HBM (ideal): efficiency curves unchanged when the datasheet BW changes.
- SM count: compute peak x s; memory BW x predict_bw(sm*s)/predict_bw(sm) at fixed warps/SM and ILP.
- FP8 weights: weight bytes halved; compute precision and KV cache unchanged; no dequant cost added.
- TP=2: ops split evenly, KV heads divisible; 2 all-reduces/layer, alpha-beta model (perfsim.roofline.allreduce_time) with hw/hardware.yaml interconnect_bandwidth and hop latency; per-step overhead not split.
