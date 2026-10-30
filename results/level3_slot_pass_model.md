# Level 3: llama.cpp slot throughput from measured per-pass cost

t_step = k * t_pass(users / k) + s; t_pass from llama-batched-bench; s fitted on the 1-user master run only; k = measured passes per step.

## h100_sxm

t_pass(B) ms: B=1: 4.22, B=2: 4.55, B=4: 5.94, B=8: 8.56, B=16: 8.18, B=32: 10.07; s = 0.74 ms

| build | users | k | measured tok/s | predicted tok/s | err % | role |
|---|---|---|---|---|---|---|
| fix | 1 | 1.00 | 204.2 | 201.6 | -1.3 | predicted |
| master | 1 | 1.00 | 201.6 | 201.6 | +0.0 | fit (s) |
| fix | 4 | 1.03 | 466.7 | 590.3 | +26.5 | predicted |
| master | 4 | 2.79 | 258.1 | 309.9 | +20.1 | predicted |
| fix | 8 | 1.12 | 575.6 | 824.8 | +43.3 | predicted |
| master | 8 | 3.81 | 316.9 | 436.5 | +37.7 | predicted |
| fix | 16 | 1.76 | 769.7 | 1018.1 | +32.3 | predicted |
| master | 16 | 1.49 | 775.2 | 1202.7 | +55.1 | predicted |

| users | measured fix/master | predicted | err % |
|---|---|---|---|
| 4 | 1.81x | 1.90x | +5.3 |
| 8 | 1.82x | 1.89x | +4.0 |
| 16 | 0.99x | 0.85x | -14.7 |

## rtx5090

t_pass(B) ms: B=1: 3.88, B=2: 4.35, B=4: 4.85, B=8: 5.74, B=16: 6.72, B=32: 8.20; s = 0.33 ms

| build | users | k | measured tok/s | predicted tok/s | err % | role |
|---|---|---|---|---|---|---|
| fix | 1 | 1.00 | 237.3 | 237.2 | -0.0 | predicted |
| master | 1 | 1.00 | 237.2 | 237.2 | -0.0 | fit (s) |
| fix | 4 | 1.01 | 629.4 | 766.2 | +21.7 | predicted |
| master | 4 | 2.93 | 290.2 | 327.7 | +12.9 | predicted |
| fix | 8 | 1.08 | 842.3 | 1252.6 | +48.7 | predicted |
| master | 8 | 3.53 | 408.0 | 502.6 | +23.2 | predicted |
| fix | 16 | 1.58 | 1026.3 | 1630.7 | +58.9 | predicted |
| master | 16 | 1.49 | 1042.5 | 1705.0 | +63.5 | predicted |

| users | measured fix/master | predicted | err % |
|---|---|---|---|
| 4 | 2.17x | 2.34x | +7.8 |
| 8 | 2.06x | 2.49x | +20.7 |
| 16 | 0.98x | 0.96x | -2.8 |

