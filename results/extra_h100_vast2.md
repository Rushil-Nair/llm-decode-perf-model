# Repeatability and cuBLAS baseline: h100_vast2

Stream peak 3142 GB/s; 3 sweep repeats.

## Best config per kernel, across repeats (GB/s)

| shape | kernel | mean | min | max | spread % |
|---|---|---|---|---|---|
| q_proj | v2 | 462 | 461 | 464 | 0.6 |
| q_proj | v3 | 1013 | 1011 | 1016 | 0.5 |
| q_proj | v4 | 1094 | 1092 | 1096 | 0.4 |
| q_proj | v5 | 1141 | 1134 | 1148 | 1.3 |
| q_proj | v6 | 1028 | 1024 | 1032 | 0.8 |
| q_proj | v7 | 1263 | 1261 | 1265 | 0.3 |
| gate_proj | v2 | 521 | 520 | 524 | 0.8 |
| gate_proj | v3 | 1474 | 1464 | 1481 | 1.1 |
| gate_proj | v4 | 1534 | 1508 | 1553 | 3.0 |
| gate_proj | v5 | 1609 | 1576 | 1628 | 3.2 |
| gate_proj | v6 | 1448 | 1430 | 1458 | 2.0 |
| gate_proj | v7 | 2049 | 2010 | 2069 | 2.9 |
| down_proj | v2 | 514 | 514 | 514 | 0.1 |
| down_proj | v3 | 1297 | 1291 | 1302 | 0.9 |
| down_proj | v4 | 1370 | 1365 | 1375 | 0.7 |
| down_proj | v5 | 1390 | 1384 | 1394 | 0.7 |
| down_proj | v6 | 1281 | 1278 | 1285 | 0.5 |
| down_proj | v7 | 1713 | 1705 | 1724 | 1.1 |

## cuBLAS fp16 dense GEMV vs the best 4-bit kernel

| shape | cuBLAS fp16 GB/s | % of peak | cuBLAS us | best 4-bit kernel | GB/s | % of peak | us | time vs cuBLAS |
|---|---|---|---|---|---|---|---|---|
| q_proj | 2038 | 65% | 16.5 | v7 | 1263 | 40% | 7.5 | 2.20x faster |
| gate_proj | 2761 | 88% | 36.5 | v7 | 2049 | 65% | 13.8 | 2.63x faster |
| down_proj | 2627 | 84% | 38.3 | v7 | 1713 | 55% | 16.6 | 2.32x faster |
