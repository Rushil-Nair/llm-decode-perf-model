#!/usr/bin/env bash
# run_ncu_deep.sh: full Nsight Compute profiles (--set full) of the best GEMV configs, after run_microarch.sh.
#
#   bench/run_ncu_deep.sh data/<gpu_tag>
#
# Needs data/<tag>/gemv_sweep.jsonl (timed sweep) and build/microarch/gemv_sweep. For each selected config
# it profiles exactly one launch of `gemv_sweep --ncu` and writes, in data/<tag>/ncu_deep/:
#   <name>.ncu-rep     the full report (open in the Nsight Compute GUI)
#   <name>.raw.csv     every collected metric (ncu --page raw)
#   <name>.details.csv the section tables (SOL, compute workload, memory, scheduler, warp state, occupancy)
# Selected: best timed config of v2-v7 on gate_proj, and of v4-v7 on q_proj.
set -u
OUT=${1:?usage: $0 data/<gpu_tag>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$(cd "$OUT" && pwd)
BIN="$REPO/build/microarch/gemv_sweep"
DEEP="$OUT/ncu_deep"; mkdir -p "$DEEP"
LOG="$OUT/ncu_deep.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

NCU=$(command -v ncu 2>/dev/null || ls /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V -r | head -n 1)
[ -z "$NCU" ] && { log "ncu not found"; echo "ncu not found" > "$OUT/ncu_deep.error.txt"; exit 1; }
[ -x "$BIN" ] || { log "gemv_sweep not built"; echo "gemv_sweep not built" > "$OUT/ncu_deep.error.txt"; exit 1; }
SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -E"

# Launch order of `gemv_sweep --ncu` (one record per launch, with ncu_launch). Reuse it if the microarch
# step already wrote it, otherwise run the app once without the profiler.
CFG="$OUT/gemv_sweep_ncu_configs.jsonl"
if ! grep -q '"ncu_launch"' "$CFG" 2>/dev/null; then
    log "listing --ncu launch order"
    "$BIN" --ncu > "$CFG" 2> "$DEEP/list.stderr.txt"
fi

PICKS=$(python3 - "$OUT/gemv_sweep.jsonl" "$CFG" <<'EOF'
import json, sys
timed = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
timed = [r for r in timed if r.get("ok") and r.get("GBps", 0) > 0 and r.get("mode") != "ncu"]
cfg = [json.loads(l) for l in open(sys.argv[2]) if l.startswith("{")]
idx = {(r["kernel"], r["ilp"], r["shape"], r["blocks_per_sm"]): r["ncu_launch"] for r in cfg}
want = [(k, "gate_proj") for k in ("v2", "v3", "v4", "v5", "v6", "v7")] + [(k, "q_proj") for k in ("v4", "v5", "v6", "v7")]
for k, shape in want:
    rs = [r for r in timed if r["kernel"] == k and r["shape"] == shape]
    if not rs:
        continue
    b = max(rs, key=lambda r: r["GBps"])
    key = (b["kernel"], b["ilp"], b["shape"], b["blocks_per_sm"])
    if key in idx:
        print(f"{k}_{shape}_ilp{b['ilp']}_b{b['blocks_per_sm']} {idx[key]} {b['GBps']}")
EOF
)
[ -z "$PICKS" ] && { log "no configs selected"; echo "no configs selected" > "$OUT/ncu_deep.error.txt"; exit 1; }
echo "$PICKS" > "$DEEP/picks.txt"
log "ncu: $NCU; profiling:"; echo "$PICKS" | tee -a "$LOG" >&2

while read -r NAME IDX GBPS; do
    log "profiling $NAME (launch $IDX, timed $GBPS GB/s)"
    # shellcheck disable=SC2086
    timeout --kill-after=30 900 $SUDO "$NCU" --set full --import-source no --clock-control none \
        --kernel-name regex:gemv_ --launch-skip "$IDX" --launch-count 1 \
        -f -o "$DEEP/$NAME" "$BIN" --ncu > "$DEEP/$NAME.app.txt" 2> "$DEEP/$NAME.stderr.txt"
    if [ -s "$DEEP/$NAME.ncu-rep" ]; then
        "$NCU" --import "$DEEP/$NAME.ncu-rep" --csv --page raw --print-units base > "$DEEP/$NAME.raw.csv" 2>>"$LOG"
        "$NCU" --import "$DEEP/$NAME.ncu-rep" --csv --page details --print-units base > "$DEEP/$NAME.details.csv" 2>>"$LOG"
        log "  ok: $(wc -c < "$DEEP/$NAME.ncu-rep") bytes"
    else
        log "  FAILED (see $DEEP/$NAME.stderr.txt)"; tail -n 5 "$DEEP/$NAME.stderr.txt" | tee -a "$LOG" >&2
    fi
done <<< "$PICKS"
log "done"
