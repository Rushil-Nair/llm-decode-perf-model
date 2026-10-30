# llm-perfsim: from warps to tokens/s

A three-level performance model of LLM inference on NVIDIA GPUs. Each level is checked against
measurements on real hardware (H100 and RTX 5090), including a leave-one-GPU-out projection: the model is
calibrated on one GPU and used to predict the other before that GPU's measurements are looked at.

> **Status:** code complete; measurement runs on H100 and RTX 5090 in progress. Result tables below are
> filled only from `data/` produced by `gpu/run_all.sh`.

## The three levels

| Level | Question | Model | Checked against |
|---|---|---|---|
| 1. Microarchitecture | How much memory bandwidth does a decode GEMV reach, and why? | Little's Law latency hiding: bytes in flight = warps/SM × loads in flight × bytes/load × SMs; achieved BW = min(peak, bytes in flight / latency), plus a loaded-latency correction (Volkov 2016) | Custom 4-bit GEMV CUDA kernels swept over occupancy × ILP; pointer-chase latency and stream-bandwidth probes; Nsight Compute counters |
| 2. Operators | How long does each op of a Qwen3-8B layer take? | Speed-of-light roofline (max of FLOPs/peak and bytes/BW) divided by an efficiency curve fitted from microbenchmarks, plus a fixed per-kernel cost | PyTorch microbenchmarks of every GEMM, attention and norm shape, cold L2 |
| 3. Serving | Prefill time, time per output token, and throughput vs. batch | Step-level decode/prefill model built from Level 2 op costs; llama.cpp slot model (forward passes per decode step) | `vllm bench latency` grid (bf16) and llama.cpp batched-bench / 16-slot server test (Q4_K_M) |

The structure follows the measure-ops-on-silicon, fall-back-to-roofline approach used by NVIDIA's
AIConfigurator, and the latency-hiding analysis in Vasily Volkov's thesis *Understanding Latency Hiding on
GPUs* (UC Berkeley, 2016).

## Case study: a llama.cpp scheduling bug

In `llama-server` with one KV stream per slot, decode batches only take consecutive slot ids, and the
least-recently-used slot order scatters busy slots, so one decode step becomes several forward passes,
each re-reading all weights. A 12-line slot-selection change raised throughput 2.6x at 8 users on an
RTX 5090 (334 to 879 tokens/s). The Level 3 model predicts the speedup from the measured passes per step
(see [SpecEdge](https://github.com/Rushil-Nair/SpecEdge)).

## Repository

```
kernels/        4-bit GEMV CUDA kernels (q4gemv.cu: v0-v4 + cuBLAS reference; gemv_sweep.cu: occupancy x ILP)
microarch/      pointer-chase and stream-bandwidth probes, Nsight Compute merge
perfsim/        models: microarch.py (L1), ops_model.py (L2), engine.py (L3), roofline, model configs
bench/          GPU-side benchmarks: run_microarch.sh, ops_bench.py, run_vllm.sh, run_llamacpp.sh
correlate/      predicted vs measured, MAPE / max error / Pearson r, leave-one-GPU-out, per-op attribution
whatif/         projections: HBM bandwidth, SM count, FP8 weights, TP=2 over NVLink (labeled as projections)
agents/         bench -> fit -> correlate -> flag ops with >15% error -> AGENT_LOG.md
figures/        all plots from results/
gpu/run_all.sh  single entry point on a rented GPU
hw/             hardware specs, each number with a source and a verified flag
```

## Reproduce

On a GPU machine (Ubuntu, NVIDIA driver installed):

```
git clone https://github.com/Rushil-Nair/llm-perfsim && cd llm-perfsim
bash gpu/run_all.sh            # ~2-2.5 h; writes data/<gpu>/ and results_<gpu>.tar.gz
```

Anywhere (no GPU):

```
pip install -r requirements.txt
python -m pytest -q
python -m correlate.correlate --data data --out results
python -m whatif.sweeps --gpu h100_sxm
python -m figures.make_figures --results results --out figures
```

## Limitations

- Post-silicon correlation on two GPUs, not a cycle-level simulator. The SM-count and bandwidth what-ifs are
  first-order projections.
- Clocks are logged at 1 Hz but not locked on rented machines.
- Nsight Compute counters are only available on hosts that allow GPU profiling; without them, Level 1 is
  checked against timings and analytically known bytes.

## License

MIT
