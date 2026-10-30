#!/usr/bin/env bash
# run_microarch.sh: Level 1 (microarchitecture) measurements on a GPU box.
#
#   bench/run_microarch.sh data/<gpu_tag>
#
# Steps (each one may fail without stopping the others; failures leave <step>.error.txt in the data dir):
#   1. find nvcc; build pointer_chase, stream_bw, gemv_sweep (-O3, -arch=native, fallback sm_<cc>, sm_90, sm_120)
#   2. pointer_chase + stream_bw  -> probes.jsonl
#   3. gemv_sweep (timed)         -> gemv_sweep.jsonl
#   4. if ncu exists and GPU performance counters are accessible (sudo if not root):
#        ncu on `gemv_sweep --ncu` -> ncu_gemv_sweep.csv, gemv_sweep_ncu_configs.jsonl
#        microarch/merge_ncu.py    -> ncu_* fields merged into gemv_sweep.jsonl (+ gemv_sweep_ncu.jsonl)
#   5. python -m perfsim.microarch -> microarch_summary.txt (model vs measurement, quick look)
#
# Intentionally no `set -e`: a failing step must not abort the rest of the run.
set -u

OUT=${1:?usage: $0 data/<gpu_tag>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
BUILD="$REPO/build/microarch"
mkdir -p "$BUILD"
LOG="$OUT/microarch_run.log"
PY=${PYTHON:-python3}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

# error <name> <message...>: write data/<tag>/<name>.error.txt and keep going
error() { local name=$1; shift; printf '%s\n' "$@" > "$OUT/$name.error.txt"; log "ERROR $name: $1"; }

# ---------------------------------------------------------------- 1. toolchain + build
find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return 0; fi
    local d
    for d in /usr/local/cuda/bin $(ls -d /usr/local/cuda-*/bin 2>/dev/null | sort -V -r) /usr/bin; do
        if [ -x "$d/nvcc" ]; then echo "$d/nvcc"; return 0; fi
    done
    return 1
}

NVCC=$(find_nvcc) || NVCC=""
if [ -z "$NVCC" ]; then
    error build "nvcc not found (looked in PATH, /usr/local/cuda*/bin, /usr/bin)"
    exit 1
fi
log "nvcc: $NVCC ($("$NVCC" --version | tail -n 1))"

# Compute capability from the driver (e.g. "9.0" -> 90); empty if nvidia-smi is too old to report it.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')
log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1) cc=${CC:-?}"

# build <binary name> <source>: try each -arch in turn; keep the ptxas register report in the run log.
build() {
    local name=$1 src=$2 arch
    for arch in native ${CC:+sm_$CC} sm_90 sm_120; do
        if "$NVCC" -O3 -std=c++17 -arch="$arch" -Xptxas -v -o "$BUILD/$name" "$src" \
               > "$BUILD/$name.build.txt" 2>&1; then
            # nvcc "succeeds" with -arch=native when it cannot see a GPU, silently using a default arch.
            if [ "$arch" = native ] && grep -q "Cannot find valid GPU" "$BUILD/$name.build.txt"; then
                log "-arch=native found no GPU, trying next"; continue
            fi
            log "built $name (-arch=$arch)"
            grep -E "Compiling entry|registers" "$BUILD/$name.build.txt" | sed 's/^ptxas info *: //' >> "$LOG"
            return 0
        fi
        log "build of $name with -arch=$arch failed, trying next"
    done
    error "build_$name" "all -arch options failed for $src; last compiler output:" "$(tail -n 40 "$BUILD/$name.build.txt")"
    return 1
}

build pointer_chase "$REPO/microarch/probes/pointer_chase.cu"; HAVE_CHASE=$?
build stream_bw     "$REPO/microarch/probes/stream_bw.cu";     HAVE_STREAM=$?
build gemv_sweep    "$REPO/kernels/gemv_sweep.cu";             HAVE_SWEEP=$?

# run <name> <timeout s> <stdout file> <command...>: stderr -> <name>.stderr.txt; non-zero exit -> error file.
# The programs print + flush one JSON line per config, so a crash/timeout still leaves every finished line.
run() {
    local name=$1 t=$2 out=$3; shift 3
    log "running $name (timeout ${t}s)"
    timeout --kill-after=30 "$t" "$@" > "$out" 2> "$OUT/$name.stderr.txt"
    local rc=$?
    if [ $rc -ne 0 ]; then
        error "$name" "$name exited with code $rc (124 = timeout); command: $*" \
              "last stderr lines:" "$(tail -n 30 "$OUT/$name.stderr.txt")"
    fi
    local n; n=$(grep -c '^{' "$out" 2>/dev/null); log "$name: ${n:-0} JSON records"
    return $rc
}

# ---------------------------------------------------------------- 2. probes
: > "$OUT/probes.jsonl"
if [ $HAVE_CHASE -eq 0 ]; then
    run pointer_chase 1200 "$BUILD/pointer_chase.out" "$BUILD/pointer_chase"
    grep '^{' "$BUILD/pointer_chase.out" >> "$OUT/probes.jsonl"
fi
if [ $HAVE_STREAM -eq 0 ]; then
    run stream_bw 1200 "$BUILD/stream_bw.out" "$BUILD/stream_bw"
    grep '^{' "$BUILD/stream_bw.out" >> "$OUT/probes.jsonl"
fi

# ---------------------------------------------------------------- 3. GEMV sweep (timed)
if [ $HAVE_SWEEP -eq 0 ]; then
    run gemv_sweep 1800 "$BUILD/gemv_sweep.out" "$BUILD/gemv_sweep"
    grep '^{' "$BUILD/gemv_sweep.out" > "$OUT/gemv_sweep.jsonl"
fi

# ---------------------------------------------------------------- 4. Nsight Compute (optional)
NCU=$(command -v ncu 2>/dev/null || ls /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu 2>/dev/null | sort -V -r | head -n 1)
COUNTERS_OK=false
SUDO=""
if [ $HAVE_SWEEP -ne 0 ]; then
    log "skipping ncu: gemv_sweep was not built"
elif [ -z "$NCU" ]; then
    log "skipping ncu: ncu not found"
else
    # Reading GPU performance counters needs root unless the driver allows all users
    # (NVreg_RestrictProfilingToAdminUsers=0). Use passwordless sudo if we are not root.
    if [ "$(id -u)" != 0 ] && sudo -n true 2>/dev/null; then SUDO="sudo -n"; fi
    # Probe: profile one launch; ncu prints ==ERROR== (e.g. ERR_NVGPUCTRPERM) when counters are blocked.
    # shellcheck disable=SC2086  # $SUDO is intentionally unquoted: empty -> no word
    timeout --kill-after=30 600 $SUDO "$NCU" --metrics gpu__time_duration.sum --launch-count 1 --kernel-name regex:gemv_ \
        "$BUILD/gemv_sweep" --ncu > "$OUT/ncu_probe.txt" 2>&1
    if ! grep -q '==ERROR==' "$OUT/ncu_probe.txt" && grep -q 'gpu__time_duration' "$OUT/ncu_probe.txt"; then
        COUNTERS_OK=true
    fi
    log "ncu: $NCU  sudo: ${SUDO:-no}  counters_ok: $COUNTERS_OK"
    if [ "$COUNTERS_OK" = true ]; then
        METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"
        METRICS="$METRICS,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_shared_mem"
        METRICS="$METRICS,launch__registers_per_thread"
        # --clock-control none: profile at the same (unlocked) clocks as the timed run.
        # --log-file: ncu's report goes there; the app's stdout (one JSON line per config) to the configs file.
        log "running ncu on gemv_sweep --ncu (timeout 3600s)"
        # shellcheck disable=SC2086
        timeout --kill-after=30 3600 $SUDO "$NCU" --metrics "$METRICS" --clock-control none \
            --csv --page raw --print-units base --kernel-name regex:gemv_ \
            --log-file "$OUT/ncu_gemv_sweep.csv" \
            "$BUILD/gemv_sweep" --ncu > "$OUT/gemv_sweep_ncu_configs.jsonl" 2> "$OUT/ncu_gemv_sweep.stderr.txt"
        rc=$?
        [ $rc -ne 0 ] && error ncu_gemv_sweep "ncu exited with code $rc (124 = timeout)" \
                                "$(tail -n 30 "$OUT/ncu_gemv_sweep.stderr.txt")"
        # Merge whatever was profiled (also after a partial run). The configs file is passed as a 2nd
        # CSV candidate in case this ncu version printed the table to stdout instead of the log file.
        PYTHONPATH="$REPO" "$PY" "$REPO/microarch/merge_ncu.py" \
            --sweep "$OUT/gemv_sweep.jsonl" --configs "$OUT/gemv_sweep_ncu_configs.jsonl" \
            --csv "$OUT/ncu_gemv_sweep.csv" "$OUT/gemv_sweep_ncu_configs.jsonl" \
            --out-ncu "$OUT/gemv_sweep_ncu.jsonl" > "$OUT/merge_ncu.stderr.txt" 2>&1 \
            || error merge_ncu "merge_ncu.py failed:" "$(tail -n 30 "$OUT/merge_ncu.stderr.txt")"
        cat "$OUT/merge_ncu.stderr.txt" >> "$LOG"
    else
        error ncu_counters "GPU performance counters not accessible to ncu (see ncu_probe.txt); skipped ncu"
    fi
fi
# Small status file so the top-level script can fill env.json's counters_ok.
printf '{"ncu":"%s","sudo":%s,"counters_ok":%s}\n' "${NCU:-}" "$([ -n "$SUDO" ] && echo true || echo false)" \
    "$COUNTERS_OK" > "$OUT/microarch_ncu_status.json"

# ---------------------------------------------------------------- 5. quick model-vs-measurement summary
if [ -s "$OUT/probes.jsonl" ]; then
    PYTHONPATH="$REPO" "$PY" -m perfsim.microarch "$OUT" > "$OUT/microarch_summary.txt" 2> "$OUT/microarch_summary.stderr.txt" \
        || error microarch_summary "python -m perfsim.microarch failed:" "$(tail -n 30 "$OUT/microarch_summary.stderr.txt")"
fi
log "done; results in $OUT"
exit 0
