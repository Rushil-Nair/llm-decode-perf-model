#!/usr/bin/env bash
# H100 kit, step 1 of 2. Idempotent: safe to run again after a failure; finished steps are skipped.
#
#     bash setup.sh            (from inside the h100kit folder; uses sudo for apt only)
#
# Does: apt packages -> Python packages -> llama.cpp at a pinned commit (two source trees:
# diag counter only, and diag counter + slot-order fix) -> two CUDA builds -> Qwen3-8B Q4_K_M
# -> prompt set. Everything goes under ./work (override with WORK=/path).
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${WORK:-$KIT/work}"
COMMIT="7182ef71e6d5dd0c8ea76486fac134fcd25025d4"     # llama.cpp master, Oct 16 2026
REPO="https://github.com/ggml-org/llama.cpp"
MODEL_URL1="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
MODEL_URL2="https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
MODEL="$W/models/Qwen3-8B-Q4_K_M.gguf"
LOGS="$W/logs"
mkdir -p "$W" "$W/models" "$LOGS"

T0=$(date +%s); TS=$T0
declare -a TIMES=()
stage() {   # print elapsed time for the previous stage, then the new stage name
    local now; now=$(date +%s)
    if [ -n "${STAGE:-}" ]; then TIMES+=("$(printf '%-28s %5ss' "$STAGE" $((now - TS)))"); fi
    STAGE="$1"; TS=$now
    echo; echo "== [$((now - T0))s] $1"
}
die() { echo; echo "!! SETUP FAILED in stage '${STAGE:-?}': $*" >&2; exit 1; }
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "!! setup.sh exited with code $rc in stage ${STAGE:-?} (logs: $LOGS)" >&2; fi' EXIT

SUDO=""; if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

# ---------------------------------------------------------------------------------------------
stage "0. checks"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this must run on the GPU VM (Lambda Stack image)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader \
    || die "nvidia-smi cannot talk to the GPU (driver not loaded? try: sudo reboot)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
case "$GPU_NAME" in *H100*|*"RTX 5090"*) ;; *) echo "WARNING: GPU is '$GPU_NAME', not an H100 or RTX 5090. Continuing anyway." ;; esac
# CUDA architecture for CMAKE_CUDA_ARCHITECTURES. Override with CUDA_ARCH=<value>, e.g.
# CUDA_ARCH=native, CUDA_ARCH=120 (RTX 5090, needs CUDA >= 12.8), CUDA_ARCH="90;120".
# Default: this GPU's compute capability from nvidia-smi (H100 -> 90, RTX 5090 -> 120).
ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')}"
if ! [[ "$ARCH" =~ ^(native|all|all-major|[0-9]+[af]?(;[0-9]+[af]?)*)$ ]]; then
    echo "WARNING: could not read compute capability ('$ARCH'); using 90 (H100). Set CUDA_ARCH to override."
    ARCH=90
fi
echo "CUDA arch: $ARCH"
FREE_GB=$(df -Pk "$W" | awk 'NR==2 {print int($4/1024/1024)}')
echo "free disk at $W: ${FREE_GB} GB"
[ "$FREE_GB" -ge 15 ] || die "need >= 15 GB free under $W (have $FREE_GB GB)"

# ---------------------------------------------------------------------------------------------
stage "1. apt packages"
need_apt=()
for cmd in cmake git curl make g++; do command -v "$cmd" >/dev/null || need_apt+=("$cmd"); done
pkgs=()
for p in "${need_apt[@]}"; do case "$p" in g++|make) pkgs+=(build-essential) ;; *) pkgs+=("$p") ;; esac; done
dpkg -s python3-pip >/dev/null 2>&1 || pkgs+=(python3-pip)
dpkg -s libssl-dev  >/dev/null 2>&1 || pkgs+=(libssl-dev)
if [ "${#pkgs[@]}" -gt 0 ]; then
    echo "installing: ${pkgs[*]}"
    # Lambda VMs often run unattended-upgrades right after boot: wait for the dpkg lock
    $SUDO apt-get -o DPkg::Lock::Timeout=600 update -qq > "$LOGS/apt.log" 2>&1 || die "apt-get update failed (see $LOGS/apt.log)"
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq "${pkgs[@]}" >> "$LOGS/apt.log" 2>&1 \
        || die "apt-get install ${pkgs[*]} failed (see $LOGS/apt.log)"
else
    echo "all present"
fi
cmake --version | head -n1
CMV="$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
python3 - "$CMV" <<'EOF' || die "cmake >= 3.18 needed (have $CMV). Try: sudo pip install cmake"
import sys; a, b = map(int, sys.argv[1].split(".")); sys.exit(0 if (a, b) >= (3, 18) else 1)
EOF

# nvcc: Lambda Stack puts it in /usr/bin; NVIDIA installers use /usr/local/cuda/bin
if ! command -v nvcc >/dev/null; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break; done
fi
command -v nvcc >/dev/null || die "nvcc (CUDA toolkit) not found. On Lambda Stack: sudo apt-get install -y nvidia-cuda-toolkit, or install CUDA 12.x from NVIDIA"
nvcc --version | tail -n1
# Blackwell (sm_100 / sm_120) needs nvcc >= 12.8; older nvcc fails late with "unsupported gpu architecture"
NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1 || true)"
if [[ "$ARCH" =~ (^|;)1[0-9][0-9] ]]; then
    python3 -c 'import sys; a, b = map(int, sys.argv[1].split(".")); sys.exit(0 if (a, b) >= (12, 8) else 1)' "${NVCC_VER:-0.0}" \
        || die "CUDA arch $ARCH (Blackwell) needs nvcc >= 12.8, have ${NVCC_VER:-?}. Install CUDA 12.8+ (its bin/ first on PATH) or set CUDA_ARCH."
fi
if [ "$ARCH" = native ]; then
    python3 -c 'import sys; a, b = map(int, sys.argv[1].split(".")); sys.exit(0 if (a, b) >= (3, 24) else 1)' "$CMV" \
        || die "CUDA_ARCH=native needs cmake >= 3.24 (have $CMV); give a number instead, e.g. CUDA_ARCH=120"
fi
echo "$(command -v nvcc)" > "$W/nvcc_path.txt"

# ---------------------------------------------------------------------------------------------
stage "2. python packages"
PYPKGS=(requests numpy pandas yaml matplotlib)
missing=()
for m in "${PYPKGS[@]}"; do python3 -c "import $m" 2>/dev/null || missing+=("$m"); done
if [ "${#missing[@]}" -gt 0 ]; then
    pipnames=(); for m in "${missing[@]}"; do [ "$m" = yaml ] && pipnames+=(pyyaml) || pipnames+=("$m"); done
    echo "pip installing: ${pipnames[*]}"
    python3 -m pip install -q --user "${pipnames[@]}" > "$LOGS/pip.log" 2>&1 \
        || python3 -m pip install -q --user --break-system-packages "${pipnames[@]}" >> "$LOGS/pip.log" 2>&1 \
        || python3 -m pip install -q --break-system-packages "${pipnames[@]}" >> "$LOGS/pip.log" 2>&1 \
        || die "pip install ${pipnames[*]} failed (see $LOGS/pip.log)"
else
    echo "all present"
fi
# 'datasets' is only for rebuilding the October prompt set; optional
python3 -c "import datasets" 2>/dev/null || {
    python3 -m pip install -q --user datasets >> "$LOGS/pip.log" 2>&1 \
        || python3 -m pip install -q --user --break-system-packages datasets >> "$LOGS/pip.log" 2>&1 \
        || echo "WARNING: could not install 'datasets'; run.sh will fall back to the tool_calls prompts only"
}

# ---------------------------------------------------------------------------------------------
stage "3. llama.cpp source @ ${COMMIT:0:8}"
SRC_M="$W/llama.cpp"        # diag counter only          -> build-master
SRC_F="$W/llama.cpp-fix"    # diag counter + slot fix    -> build-fix
if [ ! -d "$SRC_M/.git" ]; then
    rm -rf "$SRC_M"; mkdir -p "$SRC_M"
    git -C "$SRC_M" init -q
    git -C "$SRC_M" remote add origin "$REPO"
    if ! git -C "$SRC_M" fetch -q --depth 1 origin "$COMMIT" > "$LOGS/git.log" 2>&1; then
        echo "shallow fetch by sha failed; doing a full clone (slower)"
        rm -rf "$SRC_M"; git clone -q "$REPO" "$SRC_M" >> "$LOGS/git.log" 2>&1 || die "git clone $REPO failed (see $LOGS/git.log)"
    fi
    git -C "$SRC_M" -c advice.detachedHead=false checkout -q "$COMMIT" || git -C "$SRC_M" checkout -q FETCH_HEAD \
        || die "cannot check out $COMMIT"
fi
[ "$(git -C "$SRC_M" rev-parse HEAD)" = "$COMMIT" ] || die "$SRC_M is at $(git -C "$SRC_M" rev-parse HEAD), expected $COMMIT. Delete $SRC_M and rerun."
if [ ! -e "$SRC_F/.git" ]; then
    rm -rf "$SRC_F"
    git -C "$SRC_M" worktree add -q --detach "$SRC_F" "$COMMIT" || die "git worktree add failed"
fi
[ "$(git -C "$SRC_F" rev-parse HEAD)" = "$COMMIT" ] || die "$SRC_F is not at $COMMIT. Delete it and rerun."

apply_patch() {   # src patch
    local src="$1" patch="$2" name; name="$(basename "$2")"
    if git -C "$src" apply --reverse --check "$patch" 2>/dev/null; then
        echo "$(basename "$src"): $name already applied"
    elif git -C "$src" apply --check "$patch" 2>/dev/null; then
        git -C "$src" apply "$patch"; echo "$(basename "$src"): applied $name"
    else
        die "$name does not apply to $src. Reset with: git -C $src checkout -- . && rerun"
    fi
}
apply_patch "$SRC_M" "$KIT/patches/diag_counter.patch"
apply_patch "$SRC_F" "$KIT/patches/diag_counter.patch"
apply_patch "$SRC_F" "$KIT/patches/slot_order.patch"
git -C "$SRC_M" diff --stat; git -C "$SRC_F" diff --stat
git -C "$SRC_M" diff > "$W/master.diff"; git -C "$SRC_F" diff > "$W/fix.diff"

# ---------------------------------------------------------------------------------------------
TARGETS=(llama-server llama-bench)
if [ -f "$SRC_M/tools/batched-bench/CMakeLists.txt" ] && grep -q "llama-batched-bench" "$SRC_M/tools/batched-bench/CMakeLists.txt"; then
    TARGETS+=(llama-batched-bench)
fi
JOBS="${JOBS:-$(nproc)}"
build() {   # name src
    local name="$1" src="$2" dir="$W/$1"
    stage "4. build $name (${TARGETS[*]}, -j$JOBS)"
    # (re)configure when there is no build yet or it was configured for another CUDA arch
    # (e.g. a work/ dir reused from an H100 on an RTX 5090). The arch we asked for is kept
    # in our own stamp file: ggml's CMake may rewrite CMAKE_CUDA_ARCHITECTURES (120 -> 120a).
    if [ ! -f "$dir/CMakeCache.txt" ] || [ "$(cat "$dir/.kit_cuda_arch" 2>/dev/null)" != "$ARCH" ]; then
        cmake -S "$src" -B "$dir" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
            > "$LOGS/${name}_cmake.log" 2>&1 || { tail -n 30 "$LOGS/${name}_cmake.log"; die "cmake configure failed for $name (see $LOGS/${name}_cmake.log)"; }
        echo "$ARCH" > "$dir/.kit_cuda_arch"
    fi
    cmake --build "$dir" -j "$JOBS" --target "${TARGETS[@]}" > "$LOGS/${name}_build.log" 2>&1 \
        || { tail -n 40 "$LOGS/${name}_build.log"; die "build failed for $name (see $LOGS/${name}_build.log)"; }
    for t in "${TARGETS[@]}"; do [ -x "$dir/bin/$t" ] || die "$dir/bin/$t missing after build"; done
    "$dir/bin/llama-server" --version 2>&1 | tail -n 3 || true
    echo "$name: built"
}
build build-master "$SRC_M"
build build-fix "$SRC_F"
grep -q "params_base.kv_unified" "$SRC_F/tools/server/server-context.cpp" || die "fix patch not present in $SRC_F"
grep -q "DIAG decode_calls" "$SRC_M/src/llama-context.cpp" || die "diag patch not present in $SRC_M"

# ---------------------------------------------------------------------------------------------
stage "5. model Qwen3-8B Q4_K_M"
model_ok() { [ -f "$MODEL" ] && [ "$(stat -c %s "$MODEL")" -gt 4500000000 ] && [ "$(head -c 4 "$MODEL")" = "GGUF" ]; }
if model_ok; then
    echo "already downloaded"
else
    got=0
    for url in "$MODEL_URL1" "$MODEL_URL2"; do
        echo "downloading $url"
        if curl -L --fail --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 -C - \
                -o "$MODEL.part" "$url" 2> "$LOGS/curl.log" \
           || curl -L --fail --retry 8 --retry-delay 5 --connect-timeout 30 -C - -o "$MODEL.part" "$url" 2>> "$LOGS/curl.log"; then
            mv -f "$MODEL.part" "$MODEL"
            if model_ok; then got=1; break; fi
            echo "downloaded file is not a valid GGUF; trying next source"; rm -f "$MODEL"
        else
            echo "download failed from $url (see $LOGS/curl.log)"; rm -f "$MODEL.part"
        fi
    done
    [ "$got" = 1 ] || die "could not download Qwen3-8B-Q4_K_M.gguf from Qwen/ or unsloth/ on Hugging Face"
fi
ls -l "$MODEL"
[ -f "$W/models/model_sha256.txt" ] || sha256sum "$MODEL" > "$W/models/model_sha256.txt"
cat "$W/models/model_sha256.txt"

# ---------------------------------------------------------------------------------------------
stage "6. prompt set (October 8 mix)"
PD="$KIT/loadgen/data/first_hour"
if [ -f "$PD/code.jsonl" ] && [ -f "$PD/chat.jsonl" ] && [ -f "$PD/reasoning.jsonl" ] && [ -f "$PD/chinese.jsonl" ]; then
    echo "already built"
elif python3 -c "import datasets" 2>/dev/null; then
    timeout 900 python3 "$KIT/loadgen/data/build_first_hour_public.py" "$PD" > "$LOGS/prompts.log" 2>&1 \
        && echo "built: $(ls "$PD" | tr '\n' ' ')" \
        || echo "WARNING: prompt set build failed (see $LOGS/prompts.log); run.sh will use tool_calls prompts only"
else
    echo "WARNING: no 'datasets' package; run.sh will use tool_calls prompts only"
fi

stage "done"
echo
echo "Setup timings:"; printf '  %s\n' "${TIMES[@]}"
echo "  total                        $(( $(date +%s) - T0 ))s"
{ echo "commit=$COMMIT"; echo "arch=$ARCH"; echo "gpu=$GPU_NAME"; echo "date=$(date -u +%FT%TZ)"; } > "$W/setup_ok"
echo
echo "SETUP OK. Next: bash run.sh"
