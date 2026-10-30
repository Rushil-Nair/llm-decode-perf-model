# NVIDIA GeForce RTX 5090 results: llama.cpp slot-order fix + roofline check

- GPU: NVIDIA GeForce RTX 5090  (CUDA Version: 13.0)
- llama.cpp: 7182ef71e6d5dd0c8ea76486fac134fcd25025d4 + diag_counter.patch (build-master) / + slot_order.patch (build-fix)
- model: Qwen3-8B-Q4_K_M.gguf  sha256 d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785
- workload: October 8 five-kind mix (code, tool_calls, chat, chinese, reasoning), first 100 of the 200 in the October order; closed loop, greedy, max 192 tokens (384 with thinking), cache_prompt off, 1 repeat
- server: `llama-server -c 65536 -np 16 -ngl 99 --seed 0 --metrics` (one KV stream per slot unless noted)

## llama-bench (build-fix)

| test | t/s | stdev |
|---|---|---|
| pp512 | 13809.3 | 1233.2 |
| tg128 | 261.4 | 0.3 |
| pp512+tg128 | 1187.7 | 1.7 |

## llama-batched-bench (build-fix, pp128 tg128, per-sequence KV)

| B | S_PP t/s | S_TG t/s (decode, all seqs) | per-seq decode t/s |
|---|---|---|---|
| 1 | 4833 | 257.6 | 257.6 |
| 2 | 11158 | 460.0 | 230.0 |
| 4 | 13522 | 825.0 | 206.3 |
| 8 | 13676 | 1394.6 | 174.3 |
| 16 | 14065 | 2379.7 | 148.7 |
| 32 | 14466 | 3904.6 | 122.0 |

## Slot test: aggregate decode tokens/s and forward passes per generation step

| users | master tok/s | master passes/step | fix tok/s | fix passes/step | fix / master | RTX 5090 Oct 8 (master -> fix) |
|---|---|---|---|---|---|---|
| 1 | 237.2 | 1.00 | 237.3 | 1.00 | 1.00x | 228.3 -> 228.1 (1.00x) |
| 4 | 290.2 | 2.93 | 629.4 | 1.01 | 2.17x | 262.5 -> 487.8 (1.86x) |
| 8 | 408.0 | 3.53 | 842.3 | 1.08 | 2.06x | 333.6 -> 878.9 (2.63x) |
| 16 | 1042.5 | 1.49 | 1026.3 | 1.58 | 0.98x | 1143.6 -> 1136.2 (0.99x) |
| 8, master --kv-unified | 657.5 | 1.00 | | | | 639.0 |

passes/step = last DIAG line of the server log (cumulative over the run, generation-only llama_decode calls).
Failed requests: master_plain_u1 0/100, master_plain_u4 0/100, master_plain_u8 0/100, master_plain_u16 0/100, fix_plain_u1 0/100, fix_plain_u4 0/100, fix_plain_u8 0/100, fix_plain_u16 0/100

(ncu csv unreadable: )

## nsys (build-master, 8 users, ~20 s)

```
     51.2       7413819959     133298    55618.4      531.0       310  11574279     338886.3  cudaStreamSynchronize         
     42.3       6117380420    3041676     2011.2     1953.0       461    319877        802.1  cudaLaunchKernelExC_v11060    
      2.3        328601999     163137     2014.3     1983.0       481    230841        700.1  cudaLaunchKernel              
      0.1         11692692        108   108265.7    97917.5     95739    368036      41817.9  cudaGraphLaunch_v10000        
```

## nsys (build-fix, 8 users, ~20 s)

```
     85.4      13141495805     172166    76330.4      501.0       310  11586547     628638.4  cudaStreamSynchronize         
      4.8        732095455     395994     1848.8     1923.0       441    433711       1072.9  cudaLaunchKernelExC_v11060    
      3.5        542497075     325179     1668.3     1933.0       460    101170        764.0  cudaLaunchKernel              
      1.3        202038150       2011   100466.5    98324.0     94287    269443      10659.4  cudaGraphLaunch_v10000        
```

## Warnings

```
ncu run failed (see results_h100/ncu/fix_tg_b1.stderr.log; ERR_NVGPUCTRPERM = needs sudo)
```

