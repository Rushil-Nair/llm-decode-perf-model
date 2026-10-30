# v5 GEMV kernel: prediction made before it ran on any GPU

Written 2026-10-29T23:57:43+00:00 by `perfsim/v5_prediction.py`. cpg(v5) = cpg(v4) x SASS instructions per group (v5 / v4); all other model inputs unchanged.

## h100_sxm

| ILP | SASS instr/group v4 | v5 | ratio | predicted cpg v5 (SM cycles/group) |
|---|---|---|---|---|
| 1 | 208.0 | 93.0 | 0.45 | 1.10 |
| 2 | 205.0 | 86.0 | 0.42 | 1.02 |
| 4 | 202.5 | 85.8 | 0.42 | 1.05 |

Speedup vs measured v4 compares the prediction with the real v4; speedup within the model compares v5 and v4 both as predicted (cancels the model's own error on v4).

| shape | best v4 measured GB/s | best v5 predicted GB/s | vs measured v4 | within the model | bound | % of stream peak |
|---|---|---|---|---|---|---|
| down_proj | 1362 | 2041 | 1.50x | 1.36x | mem | 65% |
| gate_proj | 1525 | 2060 | 1.35x | 1.38x | mem | 66% |
| q_proj | 1066 | 1259 | 1.18x | 1.22x | mem | 40% |

Overall: best v4 measured 1525 GB/s; best v5 predicted 2060 GB/s (1.35x), mem-bound at gate_proj ILP 4, 5 blocks/SM.

## rtx5090

| ILP | SASS instr/group v4 | v5 | ratio | predicted cpg v5 (SM cycles/group) |
|---|---|---|---|---|
| 1 | 210.0 | 87.0 | 0.41 | 1.86 |
| 2 | 208.5 | 85.0 | 0.41 | 1.84 |
| 4 | 206.5 | 84.8 | 0.41 | 1.84 |

Speedup vs measured v4 compares the prediction with the real v4; speedup within the model compares v5 and v4 both as predicted (cancels the model's own error on v4).

| shape | best v4 measured GB/s | best v5 predicted GB/s | vs measured v4 | within the model | bound | % of stream peak |
|---|---|---|---|---|---|---|
| down_proj | 1385 | 1347 | 0.97x | 1.00x | mem | 79% |
| gate_proj | 1538 | 1354 | 0.88x | 1.00x | mem | 80% |
| q_proj | 1155 | 970 | 0.84x | 1.00x | mem | 57% |

Overall: best v4 measured 1538 GB/s; best v5 predicted 1354 GB/s (0.88x), mem-bound at gate_proj ILP 4, 5 blocks/SM.

