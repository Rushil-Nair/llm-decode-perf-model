# Repeatability and cuBLAS baseline: rtx5090_lium

Stream peak 1682 GB/s; 3 sweep repeats.

## Best config per kernel, across repeats (GB/s)

| shape | kernel | mean | min | max | spread % |
|---|---|---|---|---|---|
| q_proj | v2 | 622 | 622 | 622 | 0.1 |
| q_proj | v3 | 1171 | 1170 | 1173 | 0.3 |
| q_proj | v4 | 1224 | 1222 | 1224 | 0.2 |
| q_proj | v5 | 1249 | 1249 | 1249 | 0.0 |
| q_proj | v6 | 1196 | 1194 | 1197 | 0.3 |
| q_proj | v7 | 1269 | 1268 | 1269 | 0.1 |
| gate_proj | v2 | 810 | 810 | 811 | 0.2 |
| gate_proj | v3 | 1434 | 1431 | 1436 | 0.3 |
| gate_proj | v4 | 1562 | 1561 | 1563 | 0.2 |
| gate_proj | v5 | 1578 | 1574 | 1581 | 0.5 |
| gate_proj | v6 | 1545 | 1543 | 1548 | 0.3 |
| gate_proj | v7 | 1601 | 1600 | 1602 | 0.1 |
| down_proj | v2 | 670 | 669 | 670 | 0.2 |
| down_proj | v3 | 1318 | 1315 | 1320 | 0.4 |
| down_proj | v4 | 1447 | 1445 | 1447 | 0.2 |
| down_proj | v5 | 1483 | 1478 | 1486 | 0.5 |
| down_proj | v6 | 1409 | 1409 | 1410 | 0.1 |
| down_proj | v7 | 1495 | 1491 | 1500 | 0.6 |

## cuBLAS fp16 dense GEMV vs the best 4-bit kernel

| shape | cuBLAS fp16 GB/s | % of peak | cuBLAS us | best 4-bit kernel | GB/s | % of peak | us | time vs cuBLAS |
|---|---|---|---|---|---|---|---|---|
| q_proj | 1455 | 86% | 23.1 | v7 | 1269 | 75% | 7.5 | 3.10x faster |
| gate_proj | 1726 | 103% | 58.3 | v7 | 1601 | 95% | 17.7 | 3.29x faster |
| down_proj | 1673 | 99% | 60.2 | v7 | 1495 | 89% | 19.0 | 3.17x faster |
