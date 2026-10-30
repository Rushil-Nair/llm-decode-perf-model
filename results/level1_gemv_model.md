# Level 1: three-ceiling GEMV model (memory / SM throughput / per-warp) vs measured

| GPU | busy SM clock | stream peak | bytes per SM-cycle | DRAM latency | alpha |
|---|---|---|---|---|---|
| h100_sxm | 1980 MHz | 3141 GB/s | 12.0 | 275 ns | 0.64 |
| rtx5090 | 2887 MHz | 1700 GB/s | 3.5 | 129 ns | 0.72 |

Fitted on each GPU (cpg = SM-cycles per 18-byte group, cpw = per-warp cycles per group):

| fitted on | kernel | ILP | cpg | cpw | t0 (us) |
|---|---|---|---|---|---|
| h100_sxm | v2 | 1 | 8.48 | 82.0 | 4.32 |
| h100_sxm | v2 | 2 | 8.45 | 78.9 | 4.32 |
| h100_sxm | v2 | 4 | 8.46 | 78.2 | 4.32 |
| h100_sxm | v3 | 1 | 2.61 | 41.8 | 4.32 |
| h100_sxm | v3 | 2 | 2.58 | 47.5 | 4.32 |
| h100_sxm | v3 | 4 | 2.64 | 43.2 | 4.32 |
| h100_sxm | v4 | 1 | 2.45 | 40.1 | 4.32 |
| h100_sxm | v4 | 2 | 2.43 | 42.4 | 4.32 |
| h100_sxm | v4 | 4 | 2.47 | 39.7 | 4.32 |
| rtx5090 | v2 | 1 | 8.86 | 92.3 | 4.10 |
| rtx5090 | v2 | 2 | 8.49 | 92.1 | 4.10 |
| rtx5090 | v2 | 4 | 8.32 | 86.9 | 4.10 |
| rtx5090 | v3 | 1 | 5.12 | 51.2 | 4.10 |
| rtx5090 | v3 | 2 | 5.16 | 51.9 | 4.10 |
| rtx5090 | v3 | 4 | 5.12 | 51.1 | 4.10 |
| rtx5090 | v4 | 1 | 4.48 | 51.2 | 4.10 |
| rtx5090 | v4 | 2 | 4.50 | 41.1 | 4.10 |
| rtx5090 | v4 | 4 | 4.48 | 49.0 | 4.10 |

Predicted vs measured achieved bandwidth (all shapes and configurations):

| fitted on | predicted | n | MAPE % | max abs % | Pearson r | bound: mem / sm / warp |
|---|---|---|---|---|---|---|
| h100_sxm | h100_sxm | 168 | 6.2 | 37.6 | 0.978 | 0 / 123 / 45 |
| h100_sxm | rtx5090 (projection) | 146 | 10.7 | 32.9 | 0.884 | 89 / 45 / 12 |
| rtx5090 | h100_sxm (projection) | 168 | 18.9 | 44.5 | 0.949 | 0 / 141 / 27 |
| rtx5090 | rtx5090 | 146 | 9.1 | 26.4 | 0.907 | 77 / 45 / 24 |

Plain Little's Law (memory ceiling only) on the same data: see correlation.md Level 1.

## What-if (projection, not measured): best measured v4 config, one change at a time

| GPU | change | predicted GB/s | % of (scaled) peak | bound | speedup |
|---|---|---|---|---|---|
| h100_sxm | measured GPU | 1498 | 48 | sm | 1.00x |
| h100_sxm | HBM bandwidth x2 | 1498 | 24 | sm | 1.00x |
| h100_sxm | SM count x1.25 | 1771 | 56 | sm | 1.18x |
| h100_sxm | SM clock x1.25 | 1771 | 56 | sm | 1.18x |
| h100_sxm | dequant cost / 2 (cpg, cpw halved) | 2017 | 64 | mem | 1.35x |
| rtx5090 | measured GPU | 1338 | 79 | mem | 1.00x |
| rtx5090 | HBM bandwidth x2 | 1536 | 45 | sm | 1.15x |
| rtx5090 | SM count x1.25 | 1343 | 79 | mem | 1.00x |
| rtx5090 | SM clock x1.25 | 1338 | 79 | mem | 1.00x |
| rtx5090 | dequant cost / 2 (cpg, cpw halved) | 1338 | 79 | mem | 1.00x |

