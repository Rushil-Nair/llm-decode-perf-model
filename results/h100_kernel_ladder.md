# H100 GEMV kernel ladder (gate_proj, data_kernels/h100_lium2)

Stream peak 3144 GB/s; DRAM latency 274 ns; busy SM clock 1980 MHz; launch cost t0 = 4.14 us. Best timed config per kernel; counters from Nsight Compute --set full on that config. Memory ceiling = bytes / (t0 + bytes / BW_mem), BW_mem = loaded-latency Little's Law at that config's bytes in flight.

| kernel | change | best config | GB/s | % of peak | vs v4 | memory ceiling | warp instr / group | issue busy % | shared bank conflicts | shared wavefronts | stall: shared mem | stall: DRAM | occupancy % | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | x read through L1 | ILP 2, 6 blocks/SM | 522 | 17% | 0.34x | 2045 | 7.33 | 22 | 0 | 0 | 3.01 | 18.55 | 64 | 36 |
| v3 | x in shared memory, persistent grid | ILP 2, 6 blocks/SM | 1464 | 47% | 0.96x | 2045 | 7.23 | 64 | 2,359,296 | 3,145,728 | 7.13 | 5.93 | 70 | 40 |
| v4 | v3 + streaming weight loads | ILP 2, 6 blocks/SM | 1528 | 49% | 1.00x | 2045 | 7.23 | 64 | 2,359,296 | 3,145,728 | 7.19 | 5.78 | 70 | 40 |
| v5 | v4, int4->fp16 bit trick (half the instructions) | ILP 4, 4 blocks/SM | 1614 | 51% | 1.06x | 2072 | 3.62 | 32 | 2,359,786 | 3,146,218 | 11.42 | 8.58 | 48 | 42 |
| v6 | v4, swizzled x (no bank conflicts) | ILP 4, 3 blocks/SM | 1457 | 46% | 0.95x | 2045 | 7.67 | 64 | 0 | 786,432 | 0.29 | 3.61 | 34 | 58 |
| v7 | v5 + v6: both fixes | ILP 4, 4 blocks/SM | 2048 | 65% | 1.34x | 2072 | 4.31 | 47 | 516 | 786,948 | 0.41 | 9.58 | 46 | 58 |

