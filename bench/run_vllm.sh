#!/usr/bin/env bash
# Level 3, vLLM side: `vllm bench latency` grid for Qwen/Qwen3-8B (bf16) on this GPU.
#
#     bash bench/run_vllm.sh [gpu_tag]          # gpu_tag defaults to one detected from nvidia-smi
#     DRY_RUN=1 bash bench/run_vllm.sh rtx5090  # print the commands only
#
# Writes data/<tag>/vllm_latency.jsonl (one record per grid point, see bench/vllm_grid.py),
# data/<tag>/vllm/<point>.{json,log[,json.error.txt]} and data/<tag>/versions.txt.
# Never aborts the caller: every failure is written to a *.error.txt and the script exits 0.
#
# Env knobs: VLLM_POINT_TIMEOUT=1800 (s per point), MAX_NUM_BATCHED_TOKENS=16384 ('' = vLLM
# default), GPU_MEM_UTIL ('' = vLLM default 0.9), VLLM_PIP_SPEC=vllm (e.g. "vllm==0.32.0"),
# HF_TOKEN (only if the HF download is rate limited).
#
# CUDA graphs: ON (vLLM default; we never pass --enforce-eager). See vllm_grid.py.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=bench/common.sh
. "$ROOT/bench/common.sh"
TAG="${1:-${GPU_TAG:-$(detect_gpu_tag)}}"
OUT="$ROOT/data/$TAG"
DRY_RUN="${DRY_RUN:-0}"
echo "== vLLM latency grid, gpu_tag=$TAG -> $OUT"

if [ "$DRY_RUN" = 1 ]; then
    python3 "$ROOT/bench/vllm_grid.py" --tag "$TAG" --dry-run
    exit 0
fi
mkdir -p "$OUT/vllm"

# 1. vLLM (pinned to nothing: whatever pip resolves today; versions are recorded)
if ! python3 -c "import vllm" >/dev/null 2>&1; then
    echo "== installing ${VLLM_PIP_SPEC:-vllm} (pulls a matching torch; several minutes)"
    pip_install "${VLLM_PIP_SPEC:-vllm}" > "$OUT/vllm/pip_install.log" 2>&1 \
        || { tail -n 40 "$OUT/vllm/pip_install.log" > "$OUT/vllm/pip_install.log.error.txt"
             echo "!! pip install vllm failed (see $OUT/vllm/pip_install.log.error.txt); vLLM grid skipped"
             exit 0; }
fi
{
    echo "# $(date -u +%FT%TZ)  python: $(python3 --version 2>&1)"
    python3 -m pip show vllm torch 2>&1
    echo; echo "# torch build"
    python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'arch_list', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'cuda unavailable', \
'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', \
'capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')" 2>&1
    echo; echo "# driver"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv 2>&1
} > "$OUT/versions.txt"
cat "$OUT/versions.txt"
# Blackwell consumer (sm_120) needs a torch built with sm_120 kernels (CUDA >= 12.8 wheels)
if ! python3 -c "import torch,sys; c=torch.cuda.get_device_capability(0); \
sys.exit(0 if f'sm_{c[0]}{c[1]}' in torch.cuda.get_arch_list() or c[0] < 10 else 1)" 2>/dev/null; then
    echo "!! WARNING: this torch build has no kernels for this GPU's compute capability (see versions.txt);"
    echo "   vLLM will probably fail. Try VLLM_PIP_SPEC with a CUDA 12.8+/13 build." | tee "$OUT/vllm/torch_arch_warning.txt"
fi

# 2. weights once, outside the per-point timeout (16 GB in bf16)
echo "== downloading Qwen/Qwen3-8B (skipped if cached)"
python3 - > "$OUT/vllm/download.log" 2>&1 <<'EOF' || { cp "$OUT/vllm/download.log" "$OUT/vllm/download.log.error.txt"; echo "!! model download failed; the grid will try again per point"; }
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
print(p)
EOF

# 3. the grid: one process per point, timeout + *.error.txt per failed point
python3 "$ROOT/bench/vllm_grid.py" --tag "$TAG" 2>&1 | tee "$OUT/vllm/grid.log"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || echo "vllm_grid.py exited $rc" > "$OUT/vllm/grid.log.error.txt"
echo "== done: $(wc -l < "$OUT/vllm_latency.jsonl" 2>/dev/null || echo 0) records in $OUT/vllm_latency.jsonl"
exit 0
