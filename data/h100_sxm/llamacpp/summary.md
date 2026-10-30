# NVIDIA H100 80GB HBM3 results: llama.cpp slot-order fix + roofline check

- GPU: NVIDIA H100 80GB HBM3  ()
- llama.cpp: 7182ef71e6d5dd0c8ea76486fac134fcd25025d4 + diag_counter.patch (build-master) / + slot_order.patch (build-fix)
- model: Qwen3-8B-Q4_K_M.gguf  sha256 d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
- workload: October 8 five-kind mix (code, tool_calls, chat, chinese, reasoning), first 100 of the 200 in the October order; closed loop, greedy, max 192 tokens (384 with thinking), cache_prompt off, 1 repeat
- server: `llama-server -c 65536 -np 16 -ngl 99 --seed 0 --metrics` (one KV stream per slot unless noted)

## llama-bench (build-fix)

| test | t/s | stdev |
|---|---|---|
| pp512 | 9693.0 | 898.8 |
| tg128 | 243.1 | 0.7 |
| pp512+tg128 | 1088.7 | 4.0 |

## llama-batched-bench (build-fix, pp128 tg128, per-sequence KV)

| B | S_PP t/s | S_TG t/s (decode, all seqs) | per-seq decode t/s |
|---|---|---|---|
| 1 | 3041 | 237.1 | 237.1 |
| 2 | 6902 | 439.9 | 220.0 |
| 4 | 9531 | 673.4 | 168.4 |
| 8 | 9750 | 934.6 | 116.8 |
| 16 | 9633 | 1955.3 | 122.2 |
| 32 | 10420 | 3176.8 | 99.3 |

## Slot test: aggregate decode tokens/s and forward passes per generation step

| users | master tok/s | master passes/step | fix tok/s | fix passes/step | fix / master | RTX 5090 Oct 8 (master -> fix) |
|---|---|---|---|---|---|---|
| 1 | 201.6 | 1.00 | 204.2 | 1.00 | 1.01x | 228.3 -> 228.1 (1.00x) |
| 4 | 258.1 | 2.79 | 466.7 | 1.03 | 1.81x | 262.5 -> 487.8 (1.86x) |
| 8 | 316.9 | 3.81 | 575.6 | 1.12 | 1.82x | 333.6 -> 878.9 (2.63x) |
| 16 | 775.2 | 1.49 | 769.7 | 1.76 | 0.99x | 1143.6 -> 1136.2 (0.99x) |
| 8, master --kv-unified | 419.6 | 1.00 | | | | 639.0 |

passes/step = last DIAG line of the server log (cumulative over the run, generation-only llama_decode calls).
Failed requests: master_plain_u1 0/100, master_plain_u4 0/100, master_plain_u8 0/100, master_plain_u16 0/100, fix_plain_u1 0/100, fix_plain_u4 0/100, fix_plain_u8 0/100, fix_plain_u16 0/100

(ncu csv unreadable: )

## nsys (build-master, 8 users, ~20 s)

```
     45.8       6281152150    2435489      2579.0     2278.0       474    3719756       3604.0  cudaLaunchKernelExC_v11060    
     44.9       6157937579     107729     57161.4     1994.0      1512   15131950     415844.4  cudaStreamSynchronize         
      2.2        302169028      31374      9631.2     2366.0      1981  212978216    1202388.1  cudaLaunchKernel              
      0.1         18056028        106    170339.9   183106.5    104937     563281      60434.6  cudaGraphLaunch_v10000        
```

## nsys (build-fix, 8 users, ~20 s)

```
     83.7      13113091791     117518    111583.7     2038.0      1596   18600517     921931.0  cudaStreamSynchronize         
      6.3        988433517     416682      2372.2     2427.0       465     279805       1337.8  cudaLaunchKernelExC_v11060    
      1.1        176121434      62001      2840.6     2522.0      2157      59579        846.0  cudaLaunchKernel              
      1.1        172509565       1349    127879.6   116798.0    103826     568740      28632.1  cudaGraphLaunch_v10000        
```

## Warnings

```
ncu run failed (see results_h100/ncu/fix_tg_b1.stderr.log; ERR_NVGPUCTRPERM = needs sudo)
```

