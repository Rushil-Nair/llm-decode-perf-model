# Level-2 op calibration: h100_sxm

Measured: `data/h100_sxm/ops.jsonl` (206 records; 156 after keeping the fastest implementation per shape). Spec sheet: `hw/hardware.yaml` `h100_sxm` (NVIDIA H100 SXM (80 GB)): peak bf16 989.5 TFLOP/s, DRAM 3.350 TB/s.

* SOL us = max(flops / peak, bytes / bandwidth); **% of SOL = SOL us / measured us**.
* model us = t0 + SOL / eff(feature), the empirical-mode prediction (in-sample).
* Probe rows (tiny shapes) only set the fixed overhead t0; they are not on the curves.

## Fitted curves

| op | dtype | feature | t0 us (from) | shapes | eff min / median / max | clipped at eff_max | faster than SOL |
|---|---|---|---|---|---|---|---|
| attn_decode | bf16 | bytes | 6.96 (probe) | 16 | 0.122 / 0.760 / 0.956 | 0 | 0 |
| attn_prefill | bf16 | flops | 9.68 (probe) | 3 | 0.355 / 0.442 / 0.536 | 0 | 0 |
| gemm | bf16 | ai | 6.45 (probe) | 130 | 0.599 / 0.681 / 0.759 | 0 | 0 |
| rmsnorm | bf16 | bytes | 5.25 (probe) | 3 | 0.010 / 0.788 / 0.847 | 0 | 0 |

## Per op

| op | shape | impl | measured us | SOL us | % of SOL | bound | model us | model err % |
|---|---|---|---|---|---|---|---|---|
| attn_decode (probe) | batch=1 q_len=1 kv_len=16 n_q_heads=4 n_kv_heads=1 head_dim=128 causal=False probe=True | sdpa:cudnn_attention+native_gqa | 6.96 | 0.00 | 0.0% | memory | 6.98 | +0.3 |
| attn_decode | batch=1 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 12.13 | 0.63 | 5.2% | memory | 12.13 | -0.0 |
| attn_decode | batch=1 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 15.14 | 2.51 | 16.6% | memory | 15.14 | -0.0 |
| attn_decode | batch=1 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 27.26 | 10.02 | 36.8% | memory | 27.26 | +0.0 |
| attn_decode | batch=1 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 64.48 | 40.07 | 62.1% | memory | 64.48 | -0.0 |
| attn_decode | batch=4 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 12.96 | 2.52 | 19.5% | memory | 12.96 | +0.0 |
| attn_decode | batch=4 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 26.67 | 10.04 | 37.6% | memory | 26.67 | -0.0 |
| attn_decode | batch=4 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 64.34 | 40.08 | 62.3% | memory | 64.34 | +0.0 |
| attn_decode | batch=4 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 189.62 | 160.28 | 84.5% | memory | 189.62 | -0.0 |
| attn_decode | batch=16 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 22.34 | 10.09 | 45.2% | memory | 22.34 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 59.74 | 40.14 | 67.2% | memory | 59.74 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 185.38 | 160.34 | 86.5% | memory | 185.38 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 684.64 | 641.12 | 93.6% | memory | 684.64 | +0.0 |
| attn_decode | batch=64 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 60.11 | 40.38 | 67.2% | memory | 60.11 | +0.0 |
| attn_decode | batch=64 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 186.54 | 160.57 | 86.1% | memory | 186.54 | +0.0 |
| attn_decode | batch=64 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 689.41 | 641.35 | 93.0% | memory | 689.41 | +0.0 |
| attn_decode | batch=64 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 2689.22 | 2564.47 | 95.4% | memory | 2689.22 | +0.0 |
| attn_prefill (probe) | batch=1 q_len=16 kv_len=16 n_q_heads=4 n_kv_heads=1 head_dim=128 causal=True probe=True | sdpa:flash_attention+native_gqa | 9.70 | 0.01 | 0.1% | memory | 9.72 | +0.2 |
| attn_prefill | batch=1 q_len=512 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=True | sdpa:cudnn_attention+native_gqa | 18.50 | 3.13 | 16.9% | memory | 18.50 | +0.0 |
| attn_prefill | batch=1 q_len=2048 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=True | sdpa:cudnn_attention+native_gqa | 89.10 | 35.08 | 39.4% | compute | 89.10 | +0.0 |
| attn_prefill | batch=1 q_len=8192 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=True | sdpa:cudnn_attention+native_gqa | 1055.52 | 561.08 | 53.2% | compute | 1055.52 | -0.0 |
| gemm | proj=down M=1 K=12288 N=4096 | torch.matmul | 49.92 | 30.06 | 60.2% | memory | 49.30 | -1.3 |
| gemm | proj=down M=2 K=12288 N=4096 | torch.matmul | 49.68 | 30.07 | 60.5% | memory | 48.34 | -2.7 |
| gemm | proj=down M=4 K=12288 N=4096 | torch.matmul | 49.52 | 30.09 | 60.8% | memory | 48.27 | -2.5 |
| gemm | proj=down M=8 K=12288 N=4096 | torch.matmul | 49.54 | 30.13 | 60.8% | memory | 47.86 | -3.4 |
| gemm | proj=down M=16 K=12288 N=4096 | torch.matmul | 49.82 | 30.21 | 60.6% | memory | 47.95 | -3.8 |
| gemm | proj=down M=32 K=12288 N=4096 | torch.matmul | 50.05 | 30.36 | 60.7% | memory | 48.24 | -3.6 |
| gemm | proj=down M=64 K=12288 N=4096 | torch.matmul | 50.88 | 30.67 | 60.3% | memory | 47.31 | -7.0 |
| gemm | proj=down M=128 K=12288 N=4096 | torch.matmul | 50.19 | 31.30 | 62.4% | memory | 48.75 | -2.9 |
| gemm | proj=down M=256 K=12288 N=4096 | torch.matmul | 57.31 | 32.55 | 56.8% | memory | 49.98 | -12.8 |
| gemm | proj=down M=512 K=12288 N=4096 | torch.matmul | 74.13 | 52.09 | 70.3% | compute | 76.78 | +3.6 |
| gemm | proj=down M=1024 K=12288 N=4096 | torch.matmul | 135.95 | 104.17 | 76.6% | compute | 143.97 | +5.9 |
| gemm | proj=down M=2048 K=12288 N=4096 | torch.matmul | 268.80 | 208.35 | 77.5% | compute | 280.32 | +4.3 |
| gemm | proj=down M=4096 K=12288 N=4096 | torch.matmul | 579.10 | 416.69 | 72.0% | compute | 571.31 | -1.3 |
| gemm | proj=gate M=1 K=4096 N=12288 | torch.matmul | 48.08 | 30.06 | 62.5% | memory | 49.30 | +2.5 |
| gemm | proj=gate M=2 K=4096 N=12288 | torch.matmul | 48.34 | 30.07 | 62.2% | memory | 48.34 | +0.0 |
| gemm | proj=gate M=4 K=4096 N=12288 | torch.matmul | 47.74 | 30.09 | 63.0% | memory | 48.27 | +1.1 |
| gemm | proj=gate M=8 K=4096 N=12288 | torch.matmul | 47.86 | 30.13 | 63.0% | memory | 47.86 | +0.0 |
| gemm | proj=gate M=16 K=4096 N=12288 | torch.matmul | 47.55 | 30.21 | 63.5% | memory | 47.95 | +0.8 |
| gemm | proj=gate M=32 K=4096 N=12288 | torch.matmul | 48.24 | 30.36 | 62.9% | memory | 48.24 | +0.0 |
| gemm | proj=gate M=64 K=4096 N=12288 | torch.matmul | 47.22 | 30.67 | 65.0% | memory | 47.31 | +0.2 |
| gemm | proj=gate M=128 K=4096 N=12288 | torch.matmul | 48.75 | 31.30 | 64.2% | memory | 48.75 | +0.0 |
| gemm | proj=gate M=256 K=4096 N=12288 | torch.matmul | 49.76 | 32.55 | 65.4% | memory | 49.98 | +0.5 |
| gemm | proj=gate M=512 K=4096 N=12288 | torch.matmul | 77.12 | 52.09 | 67.5% | compute | 76.78 | -0.4 |
| gemm | proj=gate M=1024 K=4096 N=12288 | torch.matmul | 143.97 | 104.17 | 72.4% | compute | 143.97 | +0.0 |
| gemm | proj=gate M=2048 K=4096 N=12288 | torch.matmul | 280.32 | 208.35 | 74.3% | compute | 280.32 | +0.0 |
| gemm | proj=gate M=4096 K=4096 N=12288 | torch.matmul | 571.31 | 416.69 | 72.9% | compute | 571.31 | +0.0 |
| gemm | proj=gate_up M=1 K=4096 N=24576 | torch.matmul | 78.80 | 60.11 | 76.3% | memory | 78.80 | +0.0 |
| gemm | proj=gate_up M=2 K=4096 N=24576 | torch.matmul | 78.96 | 60.13 | 76.2% | memory | 78.96 | +0.0 |
| gemm | proj=gate_up M=4 K=4096 N=24576 | torch.matmul | 78.75 | 60.17 | 76.4% | memory | 78.75 | +0.0 |
| gemm | proj=gate_up M=8 K=4096 N=24576 | torch.matmul | 81.66 | 60.23 | 73.8% | memory | 81.66 | +0.0 |
| gemm | proj=gate_up M=16 K=4096 N=24576 | torch.matmul | 78.94 | 60.37 | 76.5% | memory | 78.94 | +0.0 |
| gemm | proj=gate_up M=32 K=4096 N=24576 | torch.matmul | 79.50 | 60.65 | 76.3% | memory | 79.50 | +0.0 |
| gemm | proj=gate_up M=64 K=4096 N=24576 | torch.matmul | 79.71 | 61.19 | 76.8% | memory | 79.71 | +0.0 |
| gemm | proj=gate_up M=128 K=4096 N=24576 | torch.matmul | 81.25 | 62.29 | 76.7% | memory | 81.25 | -0.0 |
| gemm | proj=gate_up M=256 K=4096 N=24576 | torch.matmul | 88.22 | 64.48 | 73.1% | memory | 88.22 | +0.0 |
| gemm | proj=gate_up M=512 K=4096 N=24576 | torch.matmul | 146.53 | 104.17 | 71.1% | compute | 146.53 | +0.0 |
| gemm | proj=gate_up M=1024 K=4096 N=24576 | torch.matmul | 285.86 | 208.35 | 72.9% | compute | 285.86 | -0.0 |
| gemm | proj=gate_up M=2048 K=4096 N=24576 | torch.matmul | 581.50 | 416.69 | 71.7% | compute | 581.50 | +0.0 |
| gemm | proj=gate_up M=4096 K=4096 N=24576 | torch.matmul | 1252.00 | 833.38 | 66.6% | compute | 1252.00 | +0.0 |
| gemm | proj=k M=1 K=4096 N=1024 | torch.matmul | 12.67 | 2.51 | 19.8% | memory | 12.67 | +0.0 |
| gemm | proj=k M=2 K=4096 N=1024 | torch.matmul | 12.82 | 2.51 | 19.6% | memory | 12.78 | -0.3 |
| gemm | proj=k M=4 K=4096 N=1024 | torch.matmul | 12.83 | 2.52 | 19.6% | memory | 12.82 | -0.1 |
| gemm | proj=k M=8 K=4096 N=1024 | torch.matmul | 12.77 | 2.53 | 19.8% | memory | 12.82 | +0.4 |
| gemm | proj=k M=16 K=4096 N=1024 | torch.matmul | 12.94 | 2.55 | 19.7% | memory | 12.95 | +0.1 |
| gemm | proj=k M=32 K=4096 N=1024 | torch.matmul | 13.09 | 2.60 | 19.9% | memory | 13.09 | +0.0 |
| gemm | proj=k M=64 K=4096 N=1024 | torch.matmul | 13.22 | 2.70 | 20.4% | memory | 13.23 | +0.1 |
| gemm | proj=k M=128 K=4096 N=1024 | torch.matmul | 13.18 | 2.90 | 22.0% | memory | 13.18 | +0.0 |
| gemm | proj=k M=256 K=4096 N=1024 | torch.matmul | 13.89 | 3.29 | 23.7% | memory | 13.86 | -0.2 |
| gemm | proj=k M=512 K=4096 N=1024 | torch.matmul | 14.98 | 4.34 | 29.0% | compute | 15.45 | +3.2 |
| gemm | proj=k M=1024 K=4096 N=1024 | torch.matmul | 20.10 | 8.68 | 43.2% | compute | 20.08 | -0.1 |
| gemm | proj=k M=2048 K=4096 N=1024 | torch.matmul | 29.74 | 17.36 | 58.4% | compute | 29.69 | -0.2 |
| gemm | proj=k M=4096 K=4096 N=1024 | torch.matmul | 48.72 | 34.72 | 71.3% | compute | 48.60 | -0.2 |
| gemm | proj=lm_head M=1 K=4096 N=151936 | torch.matmul | 407.10 | 371.63 | 91.3% | memory | 407.10 | +0.0 |
| gemm | proj=lm_head M=2 K=4096 N=151936 | torch.matmul | 407.04 | 371.73 | 91.3% | memory | 407.04 | +0.0 |
| gemm | proj=lm_head M=4 K=4096 N=151936 | torch.matmul | 408.24 | 371.91 | 91.1% | memory | 408.24 | -0.0 |
| gemm | proj=lm_head M=8 K=4096 N=151936 | torch.matmul | 409.06 | 372.29 | 91.0% | memory | 409.06 | +0.0 |
| gemm | proj=lm_head M=16 K=4096 N=151936 | torch.matmul | 411.65 | 373.03 | 90.6% | memory | 411.65 | +0.0 |
| gemm | proj=lm_head M=32 K=4096 N=151936 | torch.matmul | 414.27 | 374.52 | 90.4% | memory | 414.27 | +0.0 |
| gemm | proj=lm_head M=64 K=4096 N=151936 | torch.matmul | 418.34 | 377.50 | 90.2% | memory | 418.34 | +0.0 |
| gemm | proj=lm_head M=128 K=4096 N=151936 | torch.matmul | 437.36 | 383.46 | 87.7% | memory | 437.36 | +0.0 |
| gemm | proj=lm_head M=256 K=4096 N=151936 | torch.matmul | 500.78 | 395.39 | 79.0% | memory | 500.78 | +0.0 |
| gemm | proj=lm_head M=512 K=4096 N=151936 | torch.matmul | 981.36 | 644.03 | 65.6% | compute | 981.36 | -0.0 |
| gemm | proj=lm_head M=1024 K=4096 N=151936 | torch.matmul | 2043.33 | 1288.06 | 63.0% | compute | 2043.33 | +0.0 |
| gemm | proj=lm_head M=2048 K=4096 N=151936 | torch.matmul | 3893.34 | 2576.11 | 66.2% | compute | 3893.34 | -0.0 |
| gemm | proj=lm_head M=4096 K=4096 N=151936 | torch.matmul | 8202.26 | 5152.22 | 62.8% | compute | 8202.26 | +0.0 |
| gemm | proj=o M=1 K=4096 N=4096 | torch.matmul | 21.41 | 10.02 | 46.8% | memory | 21.42 | +0.1 |
| gemm | proj=o M=2 K=4096 N=4096 | torch.matmul | 21.57 | 10.03 | 46.5% | memory | 21.55 | -0.1 |
| gemm | proj=o M=4 K=4096 N=4096 | torch.matmul | 21.22 | 10.04 | 47.3% | memory | 21.22 | -0.0 |
| gemm | proj=o M=8 K=4096 N=4096 | torch.matmul | 21.38 | 10.06 | 47.0% | memory | 21.41 | +0.1 |
| gemm | proj=o M=16 K=4096 N=4096 | torch.matmul | 23.39 | 10.09 | 43.2% | memory | 23.33 | -0.3 |
| gemm | proj=o M=32 K=4096 N=4096 | torch.matmul | 21.76 | 10.17 | 46.7% | memory | 21.74 | -0.1 |
| gemm | proj=o M=64 K=4096 N=4096 | torch.matmul | 22.14 | 10.33 | 46.6% | memory | 22.17 | +0.1 |
| gemm | proj=o M=128 K=4096 N=4096 | torch.matmul | 23.17 | 10.64 | 45.9% | memory | 23.23 | +0.3 |
| gemm | proj=o M=256 K=4096 N=4096 | torch.matmul | 28.27 | 11.27 | 39.9% | memory | 28.33 | +0.2 |
| gemm | proj=o M=512 K=4096 N=4096 | torch.matmul | 31.55 | 17.36 | 55.0% | compute | 31.57 | +0.1 |
| gemm | proj=o M=1024 K=4096 N=4096 | torch.matmul | 48.32 | 34.72 | 71.9% | compute | 48.41 | +0.2 |
| gemm | proj=o M=2048 K=4096 N=4096 | torch.matmul | 94.02 | 69.45 | 73.9% | compute | 93.83 | -0.2 |
| gemm | proj=o M=4096 K=4096 N=4096 | torch.matmul | 186.43 | 138.90 | 74.5% | compute | 186.00 | -0.2 |
| gemm (probe) | proj=probe M=1 K=128 N=128 probe=True | torch.matmul | 6.46 | 0.01 | 0.2% | memory | 6.47 | +0.1 |
| gemm | proj=q M=1 K=4096 N=4096 | torch.matmul | 21.44 | 10.02 | 46.7% | memory | 21.42 | -0.1 |
| gemm | proj=q M=2 K=4096 N=4096 | torch.matmul | 21.54 | 10.03 | 46.6% | memory | 21.55 | +0.1 |
| gemm | proj=q M=4 K=4096 N=4096 | torch.matmul | 21.22 | 10.04 | 47.3% | memory | 21.22 | -0.0 |
| gemm | proj=q M=8 K=4096 N=4096 | torch.matmul | 21.44 | 10.06 | 46.9% | memory | 21.41 | -0.1 |
| gemm | proj=q M=16 K=4096 N=4096 | torch.matmul | 23.26 | 10.09 | 43.4% | memory | 23.33 | +0.3 |
| gemm | proj=q M=32 K=4096 N=4096 | torch.matmul | 21.73 | 10.17 | 46.8% | memory | 21.74 | +0.1 |
| gemm | proj=q M=64 K=4096 N=4096 | torch.matmul | 22.19 | 10.33 | 46.5% | memory | 22.17 | -0.1 |
| gemm | proj=q M=128 K=4096 N=4096 | torch.matmul | 23.30 | 10.64 | 45.7% | memory | 23.23 | -0.3 |
| gemm | proj=q M=256 K=4096 N=4096 | torch.matmul | 28.38 | 11.27 | 39.7% | memory | 28.33 | -0.2 |
| gemm | proj=q M=512 K=4096 N=4096 | torch.matmul | 31.58 | 17.36 | 55.0% | compute | 31.57 | -0.1 |
| gemm | proj=q M=1024 K=4096 N=4096 | torch.matmul | 48.50 | 34.72 | 71.6% | compute | 48.41 | -0.2 |
| gemm | proj=q M=2048 K=4096 N=4096 | torch.matmul | 93.65 | 69.45 | 74.2% | compute | 93.83 | +0.2 |
| gemm | proj=q M=4096 K=4096 N=4096 | torch.matmul | 185.57 | 138.90 | 74.8% | compute | 186.00 | +0.2 |
| gemm | proj=qkv M=1 K=4096 N=6144 | torch.matmul | 31.42 | 15.03 | 47.8% | memory | 31.42 | -0.0 |
| gemm | proj=qkv M=2 K=4096 N=6144 | torch.matmul | 31.81 | 15.04 | 47.3% | memory | 31.81 | +0.0 |
| gemm | proj=qkv M=4 K=4096 N=6144 | torch.matmul | 31.10 | 15.05 | 48.4% | memory | 31.10 | -0.0 |
| gemm | proj=qkv M=8 K=4096 N=6144 | torch.matmul | 29.23 | 15.07 | 51.6% | memory | 29.23 | +0.0 |
| gemm | proj=qkv M=16 K=4096 N=6144 | torch.matmul | 29.46 | 15.12 | 51.3% | memory | 29.46 | +0.0 |
| gemm | proj=qkv M=32 K=4096 N=6144 | torch.matmul | 29.50 | 15.22 | 51.6% | memory | 29.50 | +0.0 |
| gemm | proj=qkv M=64 K=4096 N=6144 | torch.matmul | 29.86 | 15.42 | 51.6% | memory | 29.86 | -0.0 |
| gemm | proj=qkv M=128 K=4096 N=6144 | torch.matmul | 31.86 | 15.81 | 49.6% | memory | 31.86 | -0.0 |
| gemm | proj=qkv M=256 K=4096 N=6144 | torch.matmul | 36.18 | 16.59 | 45.9% | memory | 36.18 | +0.0 |
| gemm | proj=qkv M=512 K=4096 N=6144 | torch.matmul | 42.10 | 26.04 | 61.9% | compute | 42.10 | +0.0 |
| gemm | proj=qkv M=1024 K=4096 N=6144 | torch.matmul | 73.60 | 52.09 | 70.8% | compute | 73.60 | +0.0 |
| gemm | proj=qkv M=2048 K=4096 N=6144 | torch.matmul | 142.30 | 104.17 | 73.2% | compute | 142.30 | -0.0 |
| gemm | proj=qkv M=4096 K=4096 N=6144 | torch.matmul | 278.64 | 208.35 | 74.8% | compute | 278.64 | +0.0 |
| gemm | proj=up M=1 K=4096 N=12288 | torch.matmul | 49.30 | 30.06 | 61.0% | memory | 49.30 | +0.0 |
| gemm | proj=up M=2 K=4096 N=12288 | torch.matmul | 48.11 | 30.07 | 62.5% | memory | 48.34 | +0.5 |
| gemm | proj=up M=4 K=4096 N=12288 | torch.matmul | 48.27 | 30.09 | 62.3% | memory | 48.27 | +0.0 |
| gemm | proj=up M=8 K=4096 N=12288 | torch.matmul | 47.62 | 30.13 | 63.3% | memory | 47.86 | +0.5 |
| gemm | proj=up M=16 K=4096 N=12288 | torch.matmul | 47.95 | 30.21 | 63.0% | memory | 47.95 | +0.0 |
| gemm | proj=up M=32 K=4096 N=12288 | torch.matmul | 48.18 | 30.36 | 63.0% | memory | 48.24 | +0.1 |
| gemm | proj=up M=64 K=4096 N=12288 | torch.matmul | 47.31 | 30.67 | 64.8% | memory | 47.31 | -0.0 |
| gemm | proj=up M=128 K=4096 N=12288 | torch.matmul | 48.22 | 31.30 | 64.9% | memory | 48.75 | +1.1 |
| gemm | proj=up M=256 K=4096 N=12288 | torch.matmul | 49.98 | 32.55 | 65.1% | memory | 49.98 | +0.0 |
| gemm | proj=up M=512 K=4096 N=12288 | torch.matmul | 76.78 | 52.09 | 67.8% | compute | 76.78 | +0.0 |
| gemm | proj=up M=1024 K=4096 N=12288 | torch.matmul | 145.04 | 104.17 | 71.8% | compute | 143.97 | -0.7 |
| gemm | proj=up M=2048 K=4096 N=12288 | torch.matmul | 281.92 | 208.35 | 73.9% | compute | 280.32 | -0.6 |
| gemm | proj=up M=4096 K=4096 N=12288 | torch.matmul | 566.37 | 416.69 | 73.6% | compute | 571.31 | +0.9 |
| gemm | proj=v M=1 K=4096 N=1024 | torch.matmul | 12.67 | 2.51 | 19.8% | memory | 12.67 | +0.0 |
| gemm | proj=v M=2 K=4096 N=1024 | torch.matmul | 12.75 | 2.51 | 19.7% | memory | 12.78 | +0.2 |
| gemm | proj=v M=4 K=4096 N=1024 | torch.matmul | 12.82 | 2.52 | 19.6% | memory | 12.82 | +0.1 |
| gemm | proj=v M=8 K=4096 N=1024 | torch.matmul | 12.88 | 2.53 | 19.6% | memory | 12.82 | -0.4 |
| gemm | proj=v M=16 K=4096 N=1024 | torch.matmul | 12.96 | 2.55 | 19.7% | memory | 12.95 | -0.1 |
| gemm | proj=v M=32 K=4096 N=1024 | torch.matmul | 13.09 | 2.60 | 19.9% | memory | 13.09 | +0.0 |
| gemm | proj=v M=64 K=4096 N=1024 | torch.matmul | 13.25 | 2.70 | 20.4% | memory | 13.23 | -0.1 |
| gemm | proj=v M=128 K=4096 N=1024 | torch.matmul | 13.18 | 2.90 | 22.0% | memory | 13.18 | +0.0 |
| gemm | proj=v M=256 K=4096 N=1024 | torch.matmul | 13.82 | 3.29 | 23.8% | memory | 13.86 | +0.2 |
| gemm | proj=v M=512 K=4096 N=1024 | torch.matmul | 15.98 | 4.34 | 27.2% | compute | 15.45 | -3.3 |
| gemm | proj=v M=1024 K=4096 N=1024 | torch.matmul | 20.06 | 8.68 | 43.3% | compute | 20.08 | +0.1 |
| gemm | proj=v M=2048 K=4096 N=1024 | torch.matmul | 29.63 | 17.36 | 58.6% | compute | 29.69 | +0.2 |
| gemm | proj=v M=4096 K=4096 N=1024 | torch.matmul | 48.48 | 34.72 | 71.6% | compute | 48.60 | +0.2 |
| rmsnorm (probe) | tokens=1 hidden=128 probe=True | torch.compile | 5.25 | 0.00 | 0.0% | memory | 5.27 | +0.4 |
| rmsnorm | tokens=1 hidden=4096 | torch.compile | 5.98 | 0.01 | 0.1% | memory | 5.98 | +0.0 |
| rmsnorm | tokens=64 hidden=4096 | torch.compile | 5.65 | 0.32 | 5.6% | memory | 5.65 | +0.0 |
| rmsnorm | tokens=4096 hidden=4096 | torch.compile | 28.90 | 20.03 | 69.3% | memory | 28.90 | -0.0 |

In-sample model error over 152 non-probe shapes: mean |err| 0.52%, max |err| 12.79%.
