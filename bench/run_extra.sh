#!/usr/bin/env bash
# run_extra.sh: baseline + repeatability runs, after run_microarch.sh has built gemv_sweep.
#
#   bench/run_extra.sh data/<gpu_tag>
#
#   1. kernels/q4gemv.cu (v0-v4 + cuBLAS fp16 dense GEMV on the dequantized weights), run 3 times
#      -> q4gemv_rep{1,2,3}.jsonl. cuBLAS is the vendor reference for "% of peak bandwidth" and for time.
#   2. gemv_sweep (v2-v7, all configs) run 2 more times -> gemv_sweep_rep{2,3}.jsonl
#      (gemv_sweep.jsonl from run_microarch.sh is repeat 1), for run-to-run spread.
# Each step may fail without stopping the others; failures leave <step>.error.txt.
set -u
OUT=${1:?usage: $0 data/<gpu_tag>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$(cd "$OUT" && pwd)
BUILD="$REPO/build/microarch"; mkdir -p "$BUILD"
LOG="$OUT/extra.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }
err() { local n=$1; shift; printf '%s\n' "$@" > "$OUT/$n.error.txt"; log "ERROR $n: $1"; }

NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V -r | head -n 1)
[ -z "$NVCC" ] && { err extra "nvcc not found"; exit 1; }
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')

# 1. q4gemv + cuBLAS baseline
built=0
for arch in native ${CC:+sm_$CC} sm_90 sm_120; do
    if "$NVCC" -O3 -std=c++17 -arch="$arch" -o "$BUILD/q4gemv" "$REPO/kernels/q4gemv.cu" -lcublas \
           > "$BUILD/q4gemv.build.txt" 2>&1 && ! grep -q "Cannot find valid GPU" "$BUILD/q4gemv.build.txt"; then
        log "built q4gemv (-arch=$arch)"; built=1; break
    fi
done
if [ $built -eq 1 ]; then
    for r in 1 2 3; do
        log "q4gemv repeat $r"
        timeout --kill-after=30 900 "$BUILD/q4gemv" > "$OUT/q4gemv_rep$r.jsonl" 2> "$OUT/q4gemv_rep$r.stderr.txt" \
            || err "q4gemv_rep$r" "q4gemv exited non-zero" "$(tail -n 20 "$OUT/q4gemv_rep$r.stderr.txt")"
        log "  $(grep -c '^{' "$OUT/q4gemv_rep$r.jsonl") records"
    done
else
    err q4gemv_build "q4gemv build failed" "$(tail -n 30 "$BUILD/q4gemv.build.txt")"
fi

# 2. repeat the full sweep twice
if [ -x "$BUILD/gemv_sweep" ]; then
    for r in 2 3; do
        log "gemv_sweep repeat $r"
        timeout --kill-after=30 1800 "$BUILD/gemv_sweep" > "$OUT/gemv_sweep_rep$r.jsonl" 2> "$OUT/gemv_sweep_rep$r.stderr.txt" \
            || err "gemv_sweep_rep$r" "gemv_sweep exited non-zero" "$(tail -n 20 "$OUT/gemv_sweep_rep$r.stderr.txt")"
        log "  $(grep -c '^{' "$OUT/gemv_sweep_rep$r.jsonl") records"
    done
else
    err gemv_sweep_rep "gemv_sweep not built (run the micro stage first)"
fi
log "done"
