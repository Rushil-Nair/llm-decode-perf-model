#!/usr/bin/env bash
# One entry point on a rented GPU (H100 VM or RTX 5090 container). From the repo root:
#
#     bash gpu/run_all.sh                 # everything, ~2-2.5 h
#     ONLY="micro ops" bash gpu/run_all.sh   # subset: micro ops vllm llamacpp
#
# Writes data/<gpu_tag>/... and, at the end, results_<gpu_tag>.tar.gz in the repo root: send that back.
# Every stage is independent: a failed stage writes data/<tag>/<stage>.error.txt and the run continues.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=bench/common.sh
. "$ROOT/bench/common.sh"
TAG="${GPU_TAG:-$(detect_gpu_tag)}"
export GPU_TAG="$TAG"
D="$ROOT/data/$TAG"; mkdir -p "$D"
ONLY="${ONLY:-micro ops vllm llamacpp}"
T0=$(date +%s)
say() { echo "== [$(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a "$D/run_all.log"; }
want() { case " $ONLY " in *" $1 "*) return 0 ;; esac; return 1; }
stage() {  # name command...
    local name="$1"; shift
    say "stage $name: start"
    if "$@" >> "$D/$name.log" 2>&1; then say "stage $name: done"
    else echo "exit $? (see $name.log)" > "$D/$name.error.txt"; say "stage $name: FAILED (continuing)"; fi
}

say "gpu_tag=$TAG"
# root in containers often has no sudo binary
SUDO_CMD=""; [ "$(id -u)" != 0 ] && SUDO_CMD="sudo"
# --- environment ----------------------------------------------------------------------------
if [ "${PIP_DONE:-0}" != 1 ]; then
    command -v pip >/dev/null 2>&1 || { $SUDO_CMD apt-get update -qq && $SUDO_CMD apt-get install -y -qq python3-pip; } >/dev/null 2>&1
    pip_install numpy scipy pandas matplotlib pyyaml pytest >/dev/null 2>&1 || true
fi
# CUDA toolkit (nvcc + Nsight Compute): VM images often have only the driver. Install the toolkit
# that matches the driver's CUDA version from NVIDIA's apt repo.
find_nvcc() { command -v nvcc 2>/dev/null || ls /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc 2>/dev/null | head -n1; }
if [ -z "$(find_nvcc)" ]; then
    CV="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"; MAJ="${CV%%.*}"; MIN="${CV#*.}"
    . /etc/os-release; REPO="ubuntu${VERSION_ID//./}"
    say "nvcc missing: installing cuda-toolkit-$MAJ-$MIN from NVIDIA's $REPO repo (~5-10 min)"
    { curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb" \
      && $SUDO_CMD dpkg -i /tmp/cuda-keyring.deb && $SUDO_CMD apt-get update -qq \
      && $SUDO_CMD apt-get install -y -qq "cuda-toolkit-$MAJ-$MIN"; } > "$D/cuda_install.log" 2>&1 \
      || echo "cuda toolkit install failed" > "$D/cuda_install.error.txt"
fi
NVCC_PATH="$(find_nvcc)"; [ -n "$NVCC_PATH" ] && export PATH="$(dirname "$NVCC_PATH"):$PATH"
say "nvcc: ${NVCC_PATH:-NOT FOUND}"
NCU="$(command -v ncu 2>/dev/null || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -n1 || true)"
python3 - "$D" "$NCU" <<'EOF'
import json, subprocess, sys, os
d, ncu = sys.argv[1], sys.argv[2]
def q(f):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return out[0].strip() if out else None
env = {}
for k, f in [("gpu_name", "name"), ("driver", "driver_version"), ("compute_cap", "compute_cap"),
             ("sm_clock_max_mhz", "clocks.max.sm"), ("mem_clock_max_mhz", "clocks.max.memory"),
             ("power_limit_w", "power.limit"), ("memory_total_mib", "memory.total")]:
    try: env[k] = q(f)
    except Exception: env[k] = None
env["gpu_tag"] = os.environ.get("GPU_TAG")
env["provider"] = os.environ.get("PROVIDER", "unknown")
env["ncu_path"] = ncu or None
env["clocks_locked"] = False
smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
env["cuda"] = smi.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in smi else None
json.dump(env, open(os.path.join(d, "env.json"), "w"), indent=1)
print(json.dumps(env))
EOF
# counters probe: does Nsight Compute get hardware counters here? (grep, don't trust the exit code)
COUNTERS_OK=false
if [ -n "$NCU" ]; then
    SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -E"
    cat > /tmp/probe.cu <<'EOF'
__global__ void k(float* a){ a[threadIdx.x] += 1.f; }
int main(){ float* a; cudaMalloc(&a, 1024); k<<<1,256>>>(a); cudaDeviceSynchronize(); return 0; }
EOF
    NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -n1)"
    if [ -n "$NVCC" ] && "$NVCC" -o /tmp/probe /tmp/probe.cu >/dev/null 2>&1; then
        $SUDO "$NCU" --metrics dram__bytes_read.sum /tmp/probe > "$D/ncu_probe.txt" 2>&1 || true
        if ! grep -q "==ERROR==" "$D/ncu_probe.txt" && grep -q "dram__bytes_read.sum" "$D/ncu_probe.txt"; then COUNTERS_OK=true; fi
    fi
fi
python3 -c "import json,sys; p='$D/env.json'; e=json.load(open(p)); e['counters_ok']=$( [ $COUNTERS_OK = true ] && echo True || echo False ); json.dump(e,open(p,'w'),indent=1)"
say "nsight compute counters available: $COUNTERS_OK"
# 1 Hz clock/power/temperature log for the whole run
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 > "$D/clocks.csv" 2>/dev/null &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

# --- stages ---------------------------------------------------------------------------------
want micro    && stage micro    bash bench/run_microarch.sh "$D"
want vllm     && stage vllm     bash bench/run_vllm.sh "$TAG"          # also installs torch via vllm
if want ops; then
    python3 -c "import torch" >/dev/null 2>&1 || pip_install torch >/dev/null 2>&1
    stage ops     python3 bench/ops_bench.py --out "$D"
    stage calib   python3 bench/fit_calib.py --data "$D"
fi
want llamacpp && stage llamacpp bash bench/run_llamacpp.sh "$TAG"

# --- package ----------------------------------------------------------------------------------
kill $SMI_PID 2>/dev/null
say "all stages finished"
tar -czf "$ROOT/results_$TAG.tar.gz" --exclude="*.nsys-rep" --exclude="*.ncu-rep" --exclude="*.sqlite" -C "$ROOT/data" "$TAG"
say "SEND BACK: $ROOT/results_$TAG.tar.gz ($(du -h "$ROOT/results_$TAG.tar.gz" | cut -f1))"
