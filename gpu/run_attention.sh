#!/usr/bin/env bash
# Paged decode-attention benchmark on a rented GPU. From the repo root:
#
#     bash gpu/run_attention.sh                       # tests + tune + matrix + Nsight Compute, ~20-30 min on H100
#     ONLY="test matrix" bash gpu/run_attention.sh    # subset of: test tune matrix splits ncu
#
# Writes data/<gpu_tag>/attention/ and attention_<gpu_tag>.tar.gz in the repo root: send that back.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=bench/common.sh
. "$ROOT/bench/common.sh"
TAG="${GPU_TAG:-$(detect_gpu_tag)}"
D="$ROOT/data/$TAG/attention"; mkdir -p "$D"
ONLY="${ONLY:-test tune matrix ncu}"
want() { case " $ONLY " in *" $1 "*) return 0 ;; esac; return 1; }
say() { echo "== $(date +%H:%M:%S) $*" | tee -a "$D/run.log"; }

say "gpu_tag=$TAG"
nvidia-smi | tee "$D/nvidia_smi.txt"
python3 -c "import torch" 2>/dev/null || pip_install torch
pip_install triton numpy pytest matplotlib pyyaml scipy pandas >/dev/null 2>&1 || true
# FlashInfer wheels are named flashinfer-python; the JIT build needs a CUDA toolkit for some kernels
python3 -c "import flashinfer" 2>/dev/null || pip_install flashinfer-python || say "flashinfer install FAILED"
python3 - <<'EOF' | tee "$D/versions.txt"
import torch, triton
try:
    import flashinfer; fv = flashinfer.__version__
except Exception as e:
    fv = f"missing ({e})"
print("torch", torch.__version__, "cuda", torch.version.cuda, "| triton", triton.__version__, "| flashinfer", fv)
print("gpu", torch.cuda.get_device_name(0), "| SMs", torch.cuda.get_device_properties(0).multi_processor_count)
EOF

# memory budget for the KV cache: leave headroom on 32 GB consumer cards
MEM_GB=$(python3 -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**30))")
BUDGET=$(( MEM_GB > 60 ? 24 : 10 ))

if want test; then
    say "correctness tests (GPU: includes bf16 and FlashInfer agreement)"
    python3 -m pytest tests/test_paged_decode.py -v -s 2>&1 | tee "$D/tests.txt"
    if ! grep -q " passed" "$D/tests.txt" || grep -q " failed" "$D/tests.txt"; then
        say "TESTS FAILED: not benchmarking a kernel that is not correct"; exit 1
    fi
fi
# version-2 settings are tuned on shapes outside the matrix, then the matrix scores them
want tune && { say "tune version 2 (tile config, split penalty) + ablation"; python3 bench/bench_attention.py \
    --mode tune --out "$D" --iters 50 2>&1 | tee "$D/tune.log"; }
want matrix && { say "matrix (fp16)"; python3 bench/bench_attention.py --out "$D" --max-cache-gb "$BUDGET" 2>&1 | tee "$D/matrix.log"; }
want matrix && { say "matrix (bf16, llama8b only)"; python3 bench/bench_attention.py --out "$D" --dtype bfloat16 \
    --configs llama8b --long-ctx --max-cache-gb "$BUDGET" 2>&1 | tee "$D/matrix_bf16.log"; }
want matrix && { say "matrix (page size 64)"; python3 bench/bench_attention.py --out "$D" --page-size 64 \
    --configs llama8b qwen7b --long-ctx --max-cache-gb "$BUDGET" 2>&1 | tee "$D/matrix_ps64.log"; }
want splits && { say "split sweep"; python3 bench/bench_attention.py --mode splits --out "$D" 2>&1 | tee "$D/splits.log"; }

if want ncu; then
    find_ncu() { command -v ncu 2>/dev/null || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -n1; }
    NCU="$(find_ncu)"
    if [ -z "$NCU" ]; then
        # CUDA devel images ship NVIDIA's apt repo; install the Nsight Compute that matches nvcc
        CV="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')"
        say "installing Nsight Compute (cuda $CV)"
        SUDO_CMD=""; [ "$(id -u)" != 0 ] && SUDO_CMD="sudo"
        { $SUDO_CMD apt-get update -qq && $SUDO_CMD apt-get install -y -qq "cuda-nsight-compute-$CV"; } >> "$D/ncu_install.log" 2>&1 \
            || $SUDO_CMD apt-get install -y -qq nsight-compute >> "$D/ncu_install.log" 2>&1 || true
        NCU="$(find_ncu)"
    fi
    if [ -n "$NCU" ]; then
        for SHAPE in llama8b,1,32768 llama8b,16,8192 tp8_70b,1,131072; do  # last: version 1's worst shape
            say "ncu $SHAPE"
            $NCU --set full -k regex:'split_kv|balanced|blocked|merge|[Dd]ecode' -c 40 -f -o "$D/ncu_${SHAPE//,/_}" \
                python3 bench/bench_attention.py --mode ncu --out "$D" --shape "$SHAPE" > "$D/ncu_${SHAPE//,/_}.log" 2>&1 \
                || say "ncu $SHAPE failed (counters may need admin rights on this host; see log)"
            $NCU --import "$D/ncu_${SHAPE//,/_}.ncu-rep" --page details --csv > "$D/ncu_${SHAPE//,/_}.csv" 2>/dev/null || true
        done
    else
        say "ncu not found: skipping Nsight Compute"
    fi
fi

python3 bench/attn_report.py | tee "$D/report.md" || true
tar czf "attention_${TAG}.tar.gz" "data/$TAG/attention" results/attention.md figures/attention_*.png 2>/dev/null
say "done: send attention_${TAG}.tar.gz"
