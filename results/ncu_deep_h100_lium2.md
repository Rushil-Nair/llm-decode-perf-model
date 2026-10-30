# Nsight Compute (--set full): h100_lium2

| metric | v2_gate_proj_ilp2_b6 | v3_gate_proj_ilp2_b6 | v4_gate_proj_ilp2_b6 | v4_q_proj_ilp4_b4 | v5_gate_proj_ilp4_b4 | v5_q_proj_ilp4_b5 | v6_gate_proj_ilp4_b3 | v6_q_proj_ilp4_b4 | v7_gate_proj_ilp4_b4 | v7_q_proj_ilp4_b4 |
|---|---|---|---|---|---|---|---|---|---|---|
| kernel time (us) | 54.8 | 20.5 | 20.7 | 10.7 | 19.6 | 10.2 | 22.1 | 11.4 | 16.5 | 9.8 |
| DRAM read (GB/s) | 517 | 1379 | 1369 | 883 | 1449 | 924 | 1284 | 833 | 1716 | 966 |
| DRAM throughput (% of peak) | 15.8 | 42.2 | 41.9 | 26.4 | 44.5 | 27.7 | 39.3 | 24.9 | 52.4 | 28.9 |
| SM throughput (% of peak) | 20.1 | 53.0 | 53.0 | 35.3 | 28.2 | 21.5 | 54.1 | 39.1 | 39.0 | 28.0 |
| issue slots busy (%) | 21.8 | 64.3 | 64.3 | 45.5 | 32.4 | 28.4 | 63.7 | 52.7 | 47.3 | 36.8 |
| IPC (per SM) | 0.87 | 2.50 | 2.51 | 1.80 | 1.28 | 1.09 | 2.53 | 2.04 | 1.88 | 1.41 |
| achieved occupancy (%) | 63.9 | 70.1 | 69.8 | 46.0 | 48.4 | 47.5 | 33.9 | 40.6 | 45.8 | 45.0 |
| SM clock (MHz) | 1982 | 1984 | 1968 | 1983 | 1958 | 1980 | 1915 | 1968 | 2004 | 1958 |
| warp instructions per 32-weight group | 7.33 | 7.23 | 7.23 | 7.52 | 3.62 | 4.36 | 7.67 | 8.77 | 4.31 | 5.37 |
| busiest pipes (% of peak) | alu 19<br>fma 10<br>lsu 7<br>fma_type_fp16 6<br>adu 1<br>uniform 0 | alu 57<br>fma 29<br>fma_type_fp16 18<br>lsu 10<br>adu 3<br>uniform 1 | alu 57<br>fma 30<br>fma_type_fp16 18<br>lsu 11<br>adu 3<br>uniform 1 | alu 40<br>fma 21<br>fma_type_fp16 12<br>lsu 9<br>adu 4<br>uniform 1 | alu 29<br>fma 18<br>fma_type_fp16 15<br>lsu 11<br>tensor_op_hmma 4<br>shared 4 | alu 23<br>fma 14<br>lsu 11<br>fma_type_fp16 11<br>adu 4<br>tensor_op_hmma 3 | alu 59<br>fma 30<br>fma_type_fp16 17<br>lsu 9<br>adu 2<br>uniform 1 | alu 48<br>fma 23<br>fma_type_fp16 12<br>lsu 10<br>adu 4<br>uniform 1 | alu 44<br>fma 25<br>fma_type_fp16 18<br>lsu 13<br>tensor_op_hmma 5<br>shared 5 | alu 33<br>fma 17<br>lsu 11<br>fma_type_fp16 11<br>adu 4<br>tensor_op_hmma 3 |
| top stall reasons (warps per issue) | long_scoreboard 18.55<br>lg_throttle 13.76<br>mio_throttle 7.29<br>short_scoreboard 3.01<br>wait 1.42 | short_scoreboard 7.13<br>long_scoreboard 5.93<br>not_selected 1.41<br>selected 1.00<br>wait 0.98 | short_scoreboard 7.19<br>long_scoreboard 5.78<br>not_selected 1.41<br>selected 1.00<br>wait 0.97 | long_scoreboard 7.33<br>short_scoreboard 4.34<br>wait 1.22<br>not_selected 1.06<br>selected 1.00 | short_scoreboard 11.42<br>long_scoreboard 8.58<br>wait 1.39<br>selected 1.00<br>mio_throttle 0.50 | long_scoreboard 12.72<br>short_scoreboard 8.68<br>wait 1.63<br>selected 1.00<br>mio_throttle 0.86 | long_scoreboard 3.61<br>not_selected 1.32<br>wait 1.10<br>selected 1.00<br>math_pipe_throttle 0.78 | long_scoreboard 6.63<br>not_selected 1.87<br>wait 1.29<br>math_pipe_throttle 1.14<br>selected 1.00 | long_scoreboard 9.58<br>wait 1.55<br>not_selected 1.07<br>selected 1.00<br>math_pipe_throttle 0.74 | long_scoreboard 13.07<br>wait 1.65<br>not_selected 1.10<br>selected 1.00<br>math_pipe_throttle 0.69 |
