# Nsight Compute (--set full): rtx5090_lium

| metric | v2_gate_proj_ilp4_b6 | v3_gate_proj_ilp1_b6 | v4_gate_proj_ilp4_b4 | v4_q_proj_ilp1_b4 | v5_gate_proj_ilp2_b2 | v5_q_proj_ilp1_b4 | v6_gate_proj_ilp1_b6 | v6_q_proj_ilp4_b5 | v7_gate_proj_ilp4_b2 | v7_q_proj_ilp4_b2 |
|---|---|---|---|---|---|---|---|---|---|---|
| kernel time (us) | 37.5 | 21.6 | 21.6 | 9.9 | 21.0 | 9.3 | 21.4 | 10.0 | 20.5 | 9.9 |
| DRAM read (GB/s) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| DRAM throughput (% of peak) | 42.9 | 74.6 | 74.5 | 54.4 | 76.5 | 58.0 | 75.0 | 54.0 | 78.3 | 54.7 |
| SM throughput (% of peak) | 19.3 | 35.9 | 33.9 | 26.4 | 15.2 | 15.4 | 45.2 | 32.0 | 19.2 | 15.9 |
| issue slots busy (%) | 21.5 | 37.8 | 35.8 | 34.0 | 17.1 | 18.9 | 42.9 | 36.9 | 20.1 | 20.6 |
| IPC (per SM) | 0.86 | 1.49 | 1.42 | 1.27 | 0.69 | 0.76 | 1.74 | 1.52 | 0.81 | 0.76 |
| achieved occupancy (%) | 81.3 | 82.2 | 58.4 | 47.3 | 30.8 | 49.1 | 82.0 | 48.5 | 31.4 | 27.4 |
| SM clock (MHz) | 2413 | 2380 | 2393 | 2397 | 2507 | 2396 | 2470 | 2378 | 2505 | 2400 |
| warp instructions per 32-weight group | 7.32 | 7.53 | 7.15 | 7.82 | 3.46 | 4.46 | 8.94 | 8.84 | 4.02 | 4.69 |
| busiest pipes (% of peak) | lsu 14<br>alu 11<br>fma 8<br>fma_type_fp16 6<br>adu 1<br>cbu 0 | alu 21<br>fma 14<br>lsu 13<br>fma_type_fp16 10<br>adu 3<br>xu 1 | alu 19<br>fma 14<br>lsu 12<br>fma_type_fp16 10<br>adu 2<br>xu 0 | alu 17<br>lsu 12<br>fma 12<br>fma_type_fp16 8<br>adu 3<br>xu 1 | fma 12<br>lsu 11<br>fma_type_fp16 11<br>alu 8<br>adu 2<br>cbu 0 | lsu 15<br>fma 11<br>fma_type_fp16 9<br>alu 9<br>adu 3<br>xu 1 | alu 26<br>lsu 14<br>fma 14<br>fma_type_fp16 10<br>adu 3<br>xu 1 | alu 22<br>lsu 15<br>fma 13<br>fma_type_fp16 9<br>adu 3<br>xu 1 | fma 12<br>lsu 11<br>alu 11<br>fma_type_fp16 11<br>adu 1<br>cbu 0 | lsu 12<br>alu 10<br>fma 10<br>fma_type_fp16 9<br>adu 2<br>xu 0 |
| top stall reasons (warps per issue) | long_scoreboard 18.50<br>lg_throttle 17.87<br>mio_throttle 3.76<br>short_scoreboard 1.44<br>wait 1.43 | long_scoreboard 19.54<br>short_scoreboard 2.60<br>selected 1.00<br>not_selected 0.91<br>wait 0.67 | long_scoreboard 15.22<br>short_scoreboard 1.02<br>wait 1.02<br>selected 1.00<br>not_selected 0.53 | long_scoreboard 12.20<br>selected 1.00<br>short_scoreboard 0.97<br>wait 0.74<br>not_selected 0.69 | long_scoreboard 17.28<br>short_scoreboard 1.52<br>selected 1.00<br>wait 0.77<br>barrier 0.19 | long_scoreboard 24.22<br>short_scoreboard 1.90<br>wait 1.22<br>selected 1.00<br>no_instruction 0.57 | long_scoreboard 17.37<br>wait 1.15<br>not_selected 1.07<br>selected 1.00<br>math_pipe_throttle 0.71 | long_scoreboard 10.55<br>wait 1.07<br>selected 1.00<br>not_selected 0.88<br>math_pipe_throttle 0.62 | long_scoreboard 14.90<br>selected 1.00<br>wait 0.84<br>short_scoreboard 0.50<br>not_selected 0.27 | long_scoreboard 12.47<br>selected 1.00<br>wait 0.97<br>short_scoreboard 0.58<br>barrier 0.39 |
