# Level-2 op calibration: rtx5090

Measured: `data/rtx5090/ops.jsonl` (205 records; 156 after keeping the fastest implementation per shape). Spec sheet: `hw/hardware.yaml` `rtx5090` (NVIDIA GeForce RTX 5090): peak bf16 209.5 TFLOP/s, DRAM 1.792 TB/s.

* SOL us = max(flops / peak, bytes / bandwidth); **% of SOL = SOL us / measured us**.
* model us = t0 + SOL / eff(feature), the empirical-mode prediction (in-sample).
* Probe rows (tiny shapes) only set the fixed overhead t0; they are not on the curves.

## Fitted curves

| op | dtype | feature | t0 us (from) | shapes | eff min / median / max | clipped at eff_max | faster than SOL |
|---|---|---|---|---|---|---|---|
| attn_decode | bf16 | bytes | 4.19 (probe) | 16 | 0.199 / 0.841 / 0.922 | 0 | 0 |
| attn_prefill | bf16 | flops | 8.15 (probe) | 3 | 0.421 / 0.757 / 1.000 | 1 | 0 |
| gemm | bf16 | ai | 4.05 (probe) | 130 | 0.615 / 0.788 / 1.000 | 21 | 19 |
| rmsnorm | bf16 | bytes | 4.10 (probe) | 3 | 1.000 / 1.000 / 1.000 | 3 | 0 |

## Per op

| op | shape | impl | measured us | SOL us | % of SOL | bound | model us | model err % |
|---|---|---|---|---|---|---|---|---|
| attn_decode (probe) | batch=1 q_len=1 kv_len=16 n_q_heads=4 n_kv_heads=1 head_dim=128 causal=False probe=True | sdpa:cudnn_attention+native_gqa | 4.19 | 0.01 | 0.1% | memory | 4.21 | +0.5 |
| attn_decode | batch=1 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 10.11 | 1.18 | 11.7% | memory | 10.11 | +0.0 |
| attn_decode | batch=1 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 12.29 | 4.69 | 38.2% | memory | 12.29 | +0.0 |
| attn_decode | batch=1 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 32.38 | 18.73 | 57.8% | memory | 32.38 | +0.0 |
| attn_decode | batch=1 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 96.22 | 74.91 | 77.8% | memory | 96.22 | -0.0 |
| attn_decode | batch=4 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 12.03 | 4.72 | 39.2% | memory | 12.03 | -0.0 |
| attn_decode | batch=4 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 30.51 | 18.76 | 61.5% | memory | 30.51 | +0.0 |
| attn_decode | batch=4 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 94.18 | 74.93 | 79.6% | memory | 94.18 | +0.0 |
| attn_decode | batch=4 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 342.02 | 299.63 | 87.6% | memory | 342.02 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 26.88 | 18.87 | 70.2% | memory | 26.88 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 90.03 | 75.04 | 83.4% | memory | 90.03 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 335.58 | 299.74 | 89.3% | memory | 335.58 | +0.0 |
| attn_decode | batch=16 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 1305.82 | 1198.52 | 91.8% | memory | 1305.82 | +0.0 |
| attn_decode | batch=64 q_len=1 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 93.95 | 75.48 | 80.3% | memory | 93.95 | +0.0 |
| attn_decode | batch=64 q_len=1 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 345.98 | 300.18 | 86.8% | memory | 345.98 | -0.0 |
| attn_decode | batch=64 q_len=1 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 1335.55 | 1198.96 | 89.8% | memory | 1335.55 | -0.0 |
| attn_decode | batch=64 q_len=1 kv_len=32768 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=False | sdpa:cudnn_attention+native_gqa | 5204.42 | 4794.08 | 92.1% | memory | 5204.42 | +0.0 |
| attn_prefill (probe) | batch=1 q_len=16 kv_len=16 n_q_heads=4 n_kv_heads=1 head_dim=128 causal=True probe=True | sdpa:cudnn_attention+native_gqa | 8.18 | 0.02 | 0.3% | memory | 8.21 | +0.4 |
| attn_prefill | batch=1 q_len=512 kv_len=512 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=True | sdpa:flash_attention+native_gqa | 32.77 | 10.37 | 31.6% | compute | 32.77 | -0.0 |
| attn_prefill | batch=1 q_len=2048 kv_len=2048 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=True | sdpa:cudnn_attention+native_gqa | 227.17 | 165.69 | 72.9% | compute | 227.17 | +0.0 |
| attn_prefill | batch=1 q_len=8192 kv_len=8192 n_q_heads=32 n_kv_heads=8 head_dim=128 causal=True | sdpa:cudnn_attention+native_gqa | 2655.09 | 2650.08 | 99.8% | compute | 2658.24 | +0.1 |
| gemm | proj=down M=1 K=12288 N=4096 | torch.matmul | 71.94 | 56.19 | 78.1% | memory | 82.18 | +14.2 |
| gemm | proj=down M=2 K=12288 N=4096 | torch.matmul | 73.73 | 56.21 | 76.2% | memory | 73.82 | +0.1 |
| gemm | proj=down M=4 K=12288 N=4096 | torch.matmul | 73.73 | 56.25 | 76.3% | memory | 75.74 | +2.7 |
| gemm | proj=down M=8 K=12288 N=4096 | torch.matmul | 73.82 | 56.32 | 76.3% | memory | 75.49 | +2.3 |
| gemm | proj=down M=16 K=12288 N=4096 | torch.matmul | 73.73 | 56.47 | 76.6% | memory | 69.86 | -5.3 |
| gemm | proj=down M=32 K=12288 N=4096 | torch.matmul | 73.38 | 56.76 | 77.4% | memory | 71.39 | -2.7 |
| gemm | proj=down M=64 K=12288 N=4096 | torch.matmul | 75.81 | 57.34 | 75.6% | memory | 71.97 | -5.1 |
| gemm | proj=down M=128 K=12288 N=4096 | torch.matmul | 81.95 | 61.50 | 75.0% | compute | 86.02 | +5.0 |
| gemm | proj=down M=256 K=12288 N=4096 | torch.matmul | 151.55 | 123.01 | 81.2% | compute | 145.79 | -3.8 |
| gemm | proj=down M=512 K=12288 N=4096 | torch.matmul | 260.13 | 246.01 | 94.6% | compute | 255.98 | -1.6 |
| gemm | proj=down M=1024 K=12288 N=4096 | torch.matmul | 487.42 | 492.02 | 100.9% | compute | 496.07 | +1.8 |
| gemm | proj=down M=2048 K=12288 N=4096 | torch.matmul | 1023.98 | 984.05 | 96.1% | compute | 988.10 | -3.5 |
| gemm | proj=down M=4096 K=12288 N=4096 | torch.matmul | 1934.06 | 1968.10 | 101.8% | compute | 1972.15 | +2.0 |
| gemm | proj=gate M=1 K=4096 N=12288 | torch.matmul | 82.21 | 56.19 | 68.4% | memory | 82.18 | -0.0 |
| gemm | proj=gate M=2 K=4096 N=12288 | torch.matmul | 73.84 | 56.21 | 76.1% | memory | 73.82 | -0.0 |
| gemm | proj=gate M=4 K=4096 N=12288 | torch.matmul | 75.78 | 56.25 | 74.2% | memory | 75.74 | -0.0 |
| gemm | proj=gate M=8 K=4096 N=12288 | torch.matmul | 75.84 | 56.32 | 74.3% | memory | 75.49 | -0.5 |
| gemm | proj=gate M=16 K=4096 N=12288 | torch.matmul | 69.86 | 56.47 | 80.8% | memory | 69.86 | -0.0 |
| gemm | proj=gate M=32 K=4096 N=12288 | torch.matmul | 71.39 | 56.76 | 79.5% | memory | 71.39 | +0.0 |
| gemm | proj=gate M=64 K=4096 N=12288 | torch.matmul | 71.94 | 57.34 | 79.7% | memory | 71.97 | +0.0 |
| gemm | proj=gate M=128 K=4096 N=12288 | torch.matmul | 86.02 | 61.50 | 71.5% | compute | 86.02 | +0.0 |
| gemm | proj=gate M=256 K=4096 N=12288 | torch.matmul | 145.79 | 123.01 | 84.4% | compute | 145.79 | +0.0 |
| gemm | proj=gate M=512 K=4096 N=12288 | torch.matmul | 255.90 | 246.01 | 96.1% | compute | 255.98 | +0.0 |
| gemm | proj=gate M=1024 K=4096 N=12288 | torch.matmul | 489.47 | 492.02 | 100.5% | compute | 496.07 | +1.3 |
| gemm | proj=gate M=2048 K=4096 N=12288 | torch.matmul | 942.38 | 984.05 | 104.4% | compute | 988.10 | +4.9 |
| gemm | proj=gate M=4096 K=4096 N=12288 | torch.matmul | 1839.07 | 1968.10 | 107.0% | compute | 1972.15 | +7.2 |
| gemm | proj=gate_up M=1 K=4096 N=24576 | torch.matmul | 135.42 | 112.38 | 83.0% | memory | 135.42 | +0.0 |
| gemm | proj=gate_up M=2 K=4096 N=24576 | torch.matmul | 131.17 | 112.41 | 85.7% | memory | 131.17 | +0.0 |
| gemm | proj=gate_up M=4 K=4096 N=24576 | torch.matmul | 131.17 | 112.48 | 85.7% | memory | 131.17 | +0.0 |
| gemm | proj=gate_up M=8 K=4096 N=24576 | torch.matmul | 131.17 | 112.60 | 85.8% | memory | 131.17 | +0.0 |
| gemm | proj=gate_up M=16 K=4096 N=24576 | torch.matmul | 131.79 | 112.86 | 85.6% | memory | 131.79 | +0.0 |
| gemm | proj=gate_up M=32 K=4096 N=24576 | torch.matmul | 139.26 | 113.37 | 81.4% | memory | 139.26 | +0.0 |
| gemm | proj=gate_up M=64 K=4096 N=24576 | torch.matmul | 141.30 | 114.40 | 81.0% | memory | 141.30 | +0.0 |
| gemm | proj=gate_up M=128 K=4096 N=24576 | torch.matmul | 178.53 | 123.01 | 68.9% | compute | 178.53 | +0.0 |
| gemm | proj=gate_up M=256 K=4096 N=24576 | torch.matmul | 264.22 | 246.01 | 93.1% | compute | 264.22 | +0.0 |
| gemm | proj=gate_up M=512 K=4096 N=24576 | torch.matmul | 495.55 | 492.02 | 99.3% | compute | 496.07 | +0.1 |
| gemm | proj=gate_up M=1024 K=4096 N=24576 | torch.matmul | 952.18 | 984.05 | 103.3% | compute | 988.10 | +3.8 |
| gemm | proj=gate_up M=2048 K=4096 N=24576 | torch.matmul | 1887.44 | 1968.10 | 104.3% | compute | 1972.15 | +4.5 |
| gemm | proj=gate_up M=4096 K=4096 N=24576 | torch.matmul | 3813.30 | 3936.20 | 103.2% | compute | 3940.24 | +3.3 |
| gemm | proj=k M=1 K=4096 N=1024 | torch.matmul | 20.22 | 4.69 | 23.2% | memory | 20.21 | -0.1 |
| gemm | proj=k M=2 K=4096 N=1024 | torch.matmul | 14.46 | 4.69 | 32.4% | memory | 14.48 | +0.1 |
| gemm | proj=k M=4 K=4096 N=1024 | torch.matmul | 16.38 | 4.70 | 28.7% | memory | 16.38 | -0.0 |
| gemm | proj=k M=8 K=4096 N=1024 | torch.matmul | 16.43 | 4.73 | 28.8% | memory | 16.46 | +0.1 |
| gemm | proj=k M=16 K=4096 N=1024 | torch.matmul | 18.43 | 4.77 | 25.9% | memory | 18.43 | +0.0 |
| gemm | proj=k M=32 K=4096 N=1024 | torch.matmul | 14.40 | 4.86 | 33.8% | memory | 14.37 | -0.2 |
| gemm | proj=k M=64 K=4096 N=1024 | torch.matmul | 13.92 | 5.05 | 36.3% | memory | 13.97 | +0.3 |
| gemm | proj=k M=128 K=4096 N=1024 | torch.matmul | 16.00 | 5.41 | 33.8% | memory | 16.18 | +1.1 |
| gemm | proj=k M=256 K=4096 N=1024 | torch.matmul | 22.18 | 10.25 | 46.2% | compute | 22.31 | +0.6 |
| gemm | proj=k M=512 K=4096 N=1024 | torch.matmul | 30.72 | 20.50 | 66.7% | compute | 30.72 | +0.0 |
| gemm | proj=k M=1024 K=4096 N=1024 | torch.matmul | 51.20 | 41.00 | 80.1% | compute | 51.18 | -0.0 |
| gemm | proj=k M=2048 K=4096 N=1024 | torch.matmul | 102.40 | 82.00 | 80.1% | compute | 102.40 | -0.0 |
| gemm | proj=k M=4096 K=4096 N=1024 | torch.matmul | 184.70 | 164.01 | 88.8% | compute | 185.34 | +0.3 |
| gemm | proj=lm_head M=1 K=4096 N=151936 | torch.matmul | 782.05 | 694.74 | 88.8% | memory | 782.05 | +0.0 |
| gemm | proj=lm_head M=2 K=4096 N=151936 | torch.matmul | 782.29 | 694.91 | 88.8% | memory | 782.29 | +0.0 |
| gemm | proj=lm_head M=4 K=4096 N=151936 | torch.matmul | 783.39 | 695.26 | 88.8% | memory | 783.39 | -0.0 |
| gemm | proj=lm_head M=8 K=4096 N=151936 | torch.matmul | 782.45 | 695.96 | 88.9% | memory | 782.45 | +0.0 |
| gemm | proj=lm_head M=16 K=4096 N=151936 | torch.matmul | 784.18 | 697.35 | 88.9% | memory | 784.18 | +0.0 |
| gemm | proj=lm_head M=32 K=4096 N=151936 | torch.matmul | 788.19 | 700.14 | 88.8% | memory | 788.19 | +0.0 |
| gemm | proj=lm_head M=64 K=4096 N=151936 | torch.matmul | 794.35 | 705.71 | 88.8% | memory | 794.35 | +0.0 |
| gemm | proj=lm_head M=128 K=4096 N=151936 | torch.matmul | 864.59 | 760.46 | 88.0% | compute | 864.59 | +0.0 |
| gemm | proj=lm_head M=256 K=4096 N=151936 | torch.matmul | 1437.87 | 1520.92 | 105.8% | compute | 1524.97 | +6.1 |
| gemm | proj=lm_head M=512 K=4096 N=151936 | torch.matmul | 3037.73 | 3041.84 | 100.1% | compute | 3045.89 | +0.3 |
| gemm | proj=lm_head M=1024 K=4096 N=151936 | torch.matmul | 5951.52 | 6083.68 | 102.2% | compute | 6087.73 | +2.3 |
| gemm | proj=lm_head M=2048 K=4096 N=151936 | torch.matmul | 11762.69 | 12167.37 | 103.4% | compute | 12171.41 | +3.5 |
| gemm | proj=lm_head M=4096 K=4096 N=151936 | torch.matmul | 23323.65 | 24334.73 | 104.3% | compute | 24338.78 | +4.4 |
| gemm | proj=o M=1 K=4096 N=4096 | torch.matmul | 52.96 | 18.73 | 35.4% | memory | 52.96 | -0.0 |
| gemm | proj=o M=2 K=4096 N=4096 | torch.matmul | 35.01 | 18.74 | 53.5% | memory | 34.98 | -0.1 |
| gemm | proj=o M=4 K=4096 N=4096 | torch.matmul | 36.86 | 18.76 | 50.9% | memory | 36.86 | +0.0 |
| gemm | proj=o M=8 K=4096 N=4096 | torch.matmul | 29.10 | 18.80 | 64.6% | memory | 29.54 | +1.5 |
| gemm | proj=o M=16 K=4096 N=4096 | torch.matmul | 30.56 | 18.87 | 61.8% | memory | 29.60 | -3.1 |
| gemm | proj=o M=32 K=4096 N=4096 | torch.matmul | 30.37 | 19.02 | 62.6% | memory | 30.37 | +0.0 |
| gemm | proj=o M=64 K=4096 N=4096 | torch.matmul | 32.42 | 19.31 | 59.6% | memory | 32.42 | +0.0 |
| gemm | proj=o M=128 K=4096 N=4096 | torch.matmul | 34.46 | 20.50 | 59.5% | compute | 34.46 | -0.0 |
| gemm | proj=o M=256 K=4096 N=4096 | torch.matmul | 57.34 | 41.00 | 71.5% | compute | 57.34 | -0.0 |
| gemm | proj=o M=512 K=4096 N=4096 | torch.matmul | 98.29 | 82.00 | 83.4% | compute | 98.24 | -0.0 |
| gemm | proj=o M=1024 K=4096 N=4096 | torch.matmul | 194.56 | 164.01 | 84.3% | compute | 194.54 | -0.0 |
| gemm | proj=o M=2048 K=4096 N=4096 | torch.matmul | 345.81 | 328.02 | 94.9% | compute | 345.40 | -0.1 |
| gemm | proj=o M=4096 K=4096 N=4096 | torch.matmul | 641.76 | 656.03 | 102.2% | compute | 660.08 | +2.9 |
| gemm (probe) | proj=probe M=1 K=128 N=128 probe=True | torch.matmul | 4.06 | 0.02 | 0.5% | memory | 4.08 | +0.3 |
| gemm | proj=q M=1 K=4096 N=4096 | torch.matmul | 52.96 | 18.73 | 35.4% | memory | 52.96 | -0.0 |
| gemm | proj=q M=2 K=4096 N=4096 | torch.matmul | 34.94 | 18.74 | 53.6% | memory | 34.98 | +0.1 |
| gemm | proj=q M=4 K=4096 N=4096 | torch.matmul | 36.86 | 18.76 | 50.9% | memory | 36.86 | +0.0 |
| gemm | proj=q M=8 K=4096 N=4096 | torch.matmul | 29.98 | 18.80 | 62.7% | memory | 29.54 | -1.5 |
| gemm | proj=q M=16 K=4096 N=4096 | torch.matmul | 28.70 | 18.87 | 65.7% | memory | 29.60 | +3.1 |
| gemm | proj=q M=32 K=4096 N=4096 | torch.matmul | 30.37 | 19.02 | 62.6% | memory | 30.37 | +0.0 |
| gemm | proj=q M=64 K=4096 N=4096 | torch.matmul | 32.42 | 19.31 | 59.6% | memory | 32.42 | +0.0 |
| gemm | proj=q M=128 K=4096 N=4096 | torch.matmul | 34.46 | 20.50 | 59.5% | compute | 34.46 | -0.0 |
| gemm | proj=q M=256 K=4096 N=4096 | torch.matmul | 57.34 | 41.00 | 71.5% | compute | 57.34 | -0.0 |
| gemm | proj=q M=512 K=4096 N=4096 | torch.matmul | 98.19 | 82.00 | 83.5% | compute | 98.24 | +0.0 |
| gemm | proj=q M=1024 K=4096 N=4096 | torch.matmul | 194.51 | 164.01 | 84.3% | compute | 194.54 | +0.0 |
| gemm | proj=q M=2048 K=4096 N=4096 | torch.matmul | 344.99 | 328.02 | 95.1% | compute | 345.40 | +0.1 |
| gemm | proj=q M=4096 K=4096 N=4096 | torch.matmul | 641.49 | 656.03 | 102.3% | compute | 660.08 | +2.9 |
| gemm | proj=qkv M=1 K=4096 N=6144 | torch.matmul | 59.10 | 28.10 | 47.5% | memory | 59.10 | -0.0 |
| gemm | proj=qkv M=2 K=4096 N=6144 | torch.matmul | 40.93 | 28.11 | 68.7% | memory | 40.93 | +0.0 |
| gemm | proj=qkv M=4 K=4096 N=6144 | torch.matmul | 40.96 | 28.13 | 68.7% | memory | 40.96 | +0.0 |
| gemm | proj=qkv M=8 K=4096 N=6144 | torch.matmul | 41.01 | 28.18 | 68.7% | memory | 41.01 | +0.0 |
| gemm | proj=qkv M=16 K=4096 N=6144 | torch.matmul | 40.96 | 28.27 | 69.0% | memory | 40.96 | +0.0 |
| gemm | proj=qkv M=32 K=4096 N=6144 | torch.matmul | 42.69 | 28.45 | 66.7% | memory | 42.69 | +0.0 |
| gemm | proj=qkv M=64 K=4096 N=6144 | torch.matmul | 40.91 | 28.82 | 70.4% | memory | 40.91 | +0.0 |
| gemm | proj=qkv M=128 K=4096 N=6144 | torch.matmul | 55.30 | 30.75 | 55.6% | compute | 55.30 | +0.0 |
| gemm | proj=qkv M=256 K=4096 N=6144 | torch.matmul | 77.82 | 61.50 | 79.0% | compute | 77.82 | -0.0 |
| gemm | proj=qkv M=512 K=4096 N=6144 | torch.matmul | 139.26 | 123.01 | 88.3% | compute | 139.26 | +0.0 |
| gemm | proj=qkv M=1024 K=4096 N=6144 | torch.matmul | 251.86 | 246.01 | 97.7% | compute | 251.86 | +0.0 |
| gemm | proj=qkv M=2048 K=4096 N=6144 | torch.matmul | 495.01 | 492.02 | 99.4% | compute | 496.07 | +0.2 |
| gemm | proj=qkv M=4096 K=4096 N=6144 | torch.matmul | 941.07 | 984.05 | 104.6% | compute | 988.10 | +5.0 |
| gemm | proj=up M=1 K=4096 N=12288 | torch.matmul | 82.18 | 56.19 | 68.4% | memory | 82.18 | -0.0 |
| gemm | proj=up M=2 K=4096 N=12288 | torch.matmul | 73.82 | 56.21 | 76.1% | memory | 73.82 | +0.0 |
| gemm | proj=up M=4 K=4096 N=12288 | torch.matmul | 75.74 | 56.25 | 74.3% | memory | 75.74 | +0.0 |
| gemm | proj=up M=8 K=4096 N=12288 | torch.matmul | 75.49 | 56.32 | 74.6% | memory | 75.49 | +0.0 |
| gemm | proj=up M=16 K=4096 N=12288 | torch.matmul | 69.70 | 56.47 | 81.0% | memory | 69.86 | +0.2 |
| gemm | proj=up M=32 K=4096 N=12288 | torch.matmul | 71.10 | 56.76 | 79.8% | memory | 71.39 | +0.4 |
| gemm | proj=up M=64 K=4096 N=12288 | torch.matmul | 71.97 | 57.34 | 79.7% | memory | 71.97 | +0.0 |
| gemm | proj=up M=128 K=4096 N=12288 | torch.matmul | 86.37 | 61.50 | 71.2% | compute | 86.02 | -0.4 |
| gemm | proj=up M=256 K=4096 N=12288 | torch.matmul | 145.79 | 123.01 | 84.4% | compute | 145.79 | +0.0 |
| gemm | proj=up M=512 K=4096 N=12288 | torch.matmul | 255.98 | 246.01 | 96.1% | compute | 255.98 | -0.0 |
| gemm | proj=up M=1024 K=4096 N=12288 | torch.matmul | 489.49 | 492.02 | 100.5% | compute | 496.07 | +1.3 |
| gemm | proj=up M=2048 K=4096 N=12288 | torch.matmul | 942.08 | 984.05 | 104.5% | compute | 988.10 | +4.9 |
| gemm | proj=up M=4096 K=4096 N=12288 | torch.matmul | 1863.86 | 1968.10 | 105.6% | compute | 1972.15 | +5.8 |
| gemm | proj=v M=1 K=4096 N=1024 | torch.matmul | 20.19 | 4.69 | 23.2% | memory | 20.21 | +0.1 |
| gemm | proj=v M=2 K=4096 N=1024 | torch.matmul | 14.50 | 4.69 | 32.4% | memory | 14.48 | -0.1 |
| gemm | proj=v M=4 K=4096 N=1024 | torch.matmul | 16.38 | 4.70 | 28.7% | memory | 16.38 | -0.0 |
| gemm | proj=v M=8 K=4096 N=1024 | torch.matmul | 16.48 | 4.73 | 28.7% | memory | 16.46 | -0.1 |
| gemm | proj=v M=16 K=4096 N=1024 | torch.matmul | 18.43 | 4.77 | 25.9% | memory | 18.43 | +0.0 |
| gemm | proj=v M=32 K=4096 N=1024 | torch.matmul | 14.34 | 4.86 | 33.9% | memory | 14.37 | +0.2 |
| gemm | proj=v M=64 K=4096 N=1024 | torch.matmul | 14.02 | 5.05 | 36.0% | memory | 13.97 | -0.3 |
| gemm | proj=v M=128 K=4096 N=1024 | torch.matmul | 16.37 | 5.41 | 33.1% | memory | 16.18 | -1.1 |
| gemm | proj=v M=256 K=4096 N=1024 | torch.matmul | 22.45 | 10.25 | 45.7% | compute | 22.31 | -0.6 |
| gemm | proj=v M=512 K=4096 N=1024 | torch.matmul | 30.72 | 20.50 | 66.7% | compute | 30.72 | +0.0 |
| gemm | proj=v M=1024 K=4096 N=1024 | torch.matmul | 51.17 | 41.00 | 80.1% | compute | 51.18 | +0.0 |
| gemm | proj=v M=2048 K=4096 N=1024 | torch.matmul | 102.40 | 82.00 | 80.1% | compute | 102.40 | -0.0 |
| gemm | proj=v M=4096 K=4096 N=1024 | torch.matmul | 185.98 | 164.01 | 88.2% | compute | 185.34 | -0.3 |
| rmsnorm (probe) | tokens=1 hidden=128 probe=True | torch.compile | 4.10 | 0.00 | 0.0% | memory | 4.10 | -0.0 |
| rmsnorm | tokens=1 hidden=4096 | torch.compile | 3.81 | 0.01 | 0.4% | memory | 4.11 | +7.9 |
| rmsnorm | tokens=64 hidden=4096 | torch.compile | 3.78 | 0.59 | 15.6% | memory | 4.69 | +24.1 |
| rmsnorm | tokens=4096 hidden=4096 | F.rms_norm | 38.51 | 37.45 | 97.3% | memory | 41.55 | +7.9 |

In-sample model error over 152 non-probe shapes: mean |err| 1.13%, max |err| 24.08%.

## Warnings

* gemm|bf16: 19 shapes ran FASTER than the spec-sheet SOL (clocks above the rated boost clock, an L2 hit, or a wrong spec value); their efficiency is clipped to eff_max, so the model over-predicts them.
