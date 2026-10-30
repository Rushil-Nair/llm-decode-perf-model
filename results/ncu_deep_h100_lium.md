# Nsight Compute (--set full): h100_lium

| metric | v2_gate_proj_ilp2_b6 | v3_gate_proj_ilp2_b6 | v4_gate_proj_ilp2_b6 | v4_q_proj_ilp2_b4 | v5_gate_proj_ilp4_b4 | v5_q_proj_ilp4_b5 |
|---|---|---|---|---|---|---|
| kernel time (us) | 54.6 | 21.0 | 20.8 | 11.0 | 19.2 | 10.4 |
| DRAM read (GB/s) | 519 | 1348 | 1360 | 859 | 1473 | 912 |
| DRAM throughput (% of peak) | 15.8 | 41.2 | 41.5 | 25.8 | 45.0 | 27.4 |
| SM throughput (% of peak) | 20.1 | 53.1 | 52.5 | 34.5 | 27.4 | 21.6 |
| issue slots busy (%) | 21.8 | 64.5 | 64.1 | 45.7 | 31.8 | 27.0 |
| IPC (per SM) | 0.87 | 2.48 | 2.52 | 1.80 | 1.27 | 1.07 |
| achieved occupancy (%) | 63.8 | 70.0 | 70.0 | 45.3 | 48.4 | 47.5 |
| SM clock (MHz) | 1992 | 1934 | 1975 | 1996 | 2052 | 1955 |
| warp instructions per 32-weight group | 7.33 | 7.23 | 7.23 | 7.60 | 3.62 | 4.36 |
| busiest pipes (% of peak) | alu 19<br>fma 10<br>lsu 7<br>fma_type_fp16 6<br>adu 1<br>uniform 0 | alu 56<br>fma 29<br>fma_type_fp16 18<br>lsu 10<br>adu 3<br>uniform 1 | alu 57<br>fma 30<br>fma_type_fp16 18<br>lsu 11<br>adu 3<br>uniform 1 | alu 40<br>fma 21<br>fma_type_fp16 12<br>lsu 9<br>adu 4<br>uniform 1 | alu 29<br>fma 18<br>fma_type_fp16 15<br>lsu 11<br>tensor_op_hmma 4<br>shared 4 | alu 23<br>fma 14<br>lsu 11<br>fma_type_fp16 10<br>adu 4<br>tensor_op_hmma 3 |
| top stall reasons (warps per issue) | long_scoreboard 19.17<br>lg_throttle 13.73<br>mio_throttle 7.34<br>short_scoreboard 2.99<br>wait 1.42 | short_scoreboard 6.97<br>long_scoreboard 5.87<br>not_selected 1.44<br>selected 1.00<br>wait 0.99 | short_scoreboard 7.22<br>long_scoreboard 5.75<br>not_selected 1.41<br>selected 1.00<br>wait 0.96 | long_scoreboard 8.15<br>short_scoreboard 3.25<br>not_selected 1.19<br>wait 1.05<br>selected 1.00 | short_scoreboard 11.40<br>long_scoreboard 8.41<br>wait 1.39<br>selected 1.00<br>mio_throttle 0.52 | long_scoreboard 13.71<br>short_scoreboard 6.01<br>wait 1.63<br>selected 1.00<br>mio_throttle 0.84 |
