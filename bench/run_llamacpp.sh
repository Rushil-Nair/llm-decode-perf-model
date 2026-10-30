#!/usr/bin/env bash
# Level 3, llama.cpp side: run the existing kit in bench/llamacpp (setup.sh, then run.sh) and
# copy its results into data/<tag>/llamacpp/ as-is (layout in CONTRACTS.md).
#
#     bash bench/run_llamacpp.sh [gpu_tag]        # gpu_tag defaults to one detected from nvidia-smi
#     SKIP_NSYS=1 bash bench/run_llamacpp.sh      # any run.sh knob passes through (SKIP_NCU, USERS, N_REQ ...)
#     CUDA_ARCH=120 bash bench/run_llamacpp.sh    # force the CUDA arch of the llama.cpp builds
#     DRY_RUN=1 bash bench/run_llamacpp.sh        # run.sh prints its commands; nothing is built or copied
#
# Notes
#  * The kit always writes ./results_h100/ (the name is historical; it is used on any GPU).
#    run.sh skips steps whose result files already exist, so results left by a run on a
#    DIFFERENT GPU would be silently reused: we move such a folder aside first
#    (results_h100.<old gpu>.<time>/), then copy the fresh one to data/<tag>/llamacpp/.
#  * CUDA arch: setup.sh builds for this GPU's compute capability (nvidia-smi compute_cap:
#    90 on H100, 120 on RTX 5090; CUDA_ARCH overrides; sm_120 needs nvcc >= 12.8).
#  * The kit's CUDA GEMV step (2d) is skipped by default here (SKIP_GEMV=1): Level 1 has its
#    own GEMV sweep. With SKIP_GEMV=0 it compiles kernels/q4gemv.cu from this repo.
#  * Never aborts the caller: failures are written to data/<tag>/llamacpp/*.error.txt, exit 0.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIT="$ROOT/bench/llamacpp"
# shellcheck source=bench/common.sh
. "$ROOT/bench/common.sh"
TAG="${1:-${GPU_TAG:-$(detect_gpu_tag)}}"
DEST="$ROOT/data/$TAG/llamacpp"
W="${WORK:-$KIT/work}"
DRY_RUN="${DRY_RUN:-0}"
export SKIP_GEMV="${SKIP_GEMV:-1}"
export GEMV_SRC="${GEMV_SRC:-$ROOT/kernels/q4gemv.cu}"
echo "== llama.cpp kit, gpu_tag=$TAG -> $DEST"

if [ "$DRY_RUN" = 1 ]; then
    echo "+ bash $KIT/setup.sh   (skipped in dry run: it needs a GPU)"
    DRY_RUN=1 bash "$KIT/run.sh"
    echo "+ cp -a $KIT/results_h100/. $DEST/"
    exit 0
fi
mkdir -p "$DEST"

fail() {   # name message -> data/<tag>/llamacpp/<name>.error.txt
    { echo "$2"; echo "date: $(date -u +%FT%TZ)"; } > "$DEST/$1.error.txt"
    for f in "$W"/logs/*.log; do [ -f "$f" ] && { echo "--- $f (last 40 lines)"; tail -n 40 "$f"; } >> "$DEST/$1.error.txt"; done
    echo "!! $2 (see $DEST/$1.error.txt)"
}

# 1. stale results from another GPU? move them aside so run.sh does not skip work
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
OLD="$KIT/results_h100"
if [ -d "$OLD" ]; then
    old_gpu="$(cat "$OLD/meta/gpu_name.txt" 2>/dev/null || echo unknown)"
    if [ "$old_gpu" != "$GPU_NAME" ]; then
        aside="$OLD.$(echo "$old_gpu" | tr -c 'A-Za-z0-9\n' '_').$(date +%Y%m%d%H%M%S)"
        echo "== results_h100/ is from '$old_gpu', this GPU is '$GPU_NAME': moving it to $aside"
        mv "$OLD" "$aside"
    else
        echo "== results_h100/ is from this GPU: run.sh resumes it (finished steps are skipped)"
    fi
fi

# 2. setup (idempotent; rebuilds if the CUDA arch changed)
if ! bash "$KIT/setup.sh" 2>&1 | tee "$DEST/setup.log"; then
    fail setup "llama.cpp kit setup.sh failed; run.sh skipped"
    exit 0
fi
rm -f "$DEST/setup.error.txt"

# 3. run (SKIP_NSYS / SKIP_NCU / USERS / N_REQ ... pass through the environment)
if ! bash "$KIT/run.sh" 2>&1 | tee "$DEST/run.log"; then
    fail run "llama.cpp kit run.sh exited non-zero; copying whatever it produced"
else
    rm -f "$DEST/run.error.txt"
fi

# 4. copy results as-is (results_h100/* -> data/<tag>/llamacpp/*)
if [ -d "$OLD" ]; then
    cp -a "$OLD/." "$DEST/" || fail copy "copying $OLD to $DEST failed"
    cp -a "$W/setup_ok" "$DEST/meta/setup_ok.txt" 2>/dev/null || true
    echo "== copied $(find "$DEST" -type f | wc -l) files to $DEST"
    ls "$DEST/slot" 2>/dev/null | head -n 20
else
    fail copy "run.sh produced no $OLD folder"
fi
exit 0
