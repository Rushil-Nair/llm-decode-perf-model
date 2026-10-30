#!/usr/bin/env bash
# H100 kit, step 2 of 2. Run after setup.sh, from inside the h100kit folder:
#
#     bash run.sh                 # everything, ~25-40 min on 1x H100
#     DRY_RUN=1 bash run.sh       # print the commands only (no GPU needed)
#
# Results: ./results_h100/ and ./results_h100.tar.gz (send the .tar.gz back).
# A stopped run can be started again: finished slot-test results are skipped.
#
# Knobs (env): N_REQ=100 requests per user count, USERS="1 4 8 16", PORT=8091,
#              SKIP_BENCH=1, SKIP_SLOTS=1, SKIP_NSYS=1, SKIP_NCU=1, SKIP_GEMV=1, NSYS_SECONDS=20
set -uo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${WORK:-$KIT/work}"
R="$KIT/results_h100"; [ "${DRY_RUN:-0}" = 1 ] && R="$KIT/results_h100_dryrun"
COMMIT="7182ef71e6d5dd0c8ea76486fac134fcd25025d4"
MODEL="${MODEL:-$W/models/Qwen3-8B-Q4_K_M.gguf}"
DRY_RUN="${DRY_RUN:-0}"
N_REQ="${N_REQ:-100}"
USERS="${USERS:-1 4 8 16}"
PORT="${PORT:-8091}"
NSYS_SECONDS="${NSYS_SECONDS:-20}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
LOADGEN_TIMEOUT="${LOADGEN_TIMEOUT:-900}"    # per (build, users) run
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
SERVER_ARGS_COMMON="-c 65536 -np 16 -ngl 99 --seed 0 --metrics"   # = October 8 (no --kv-unified)

T0=$(date +%s)
say()  { echo "== [$(( $(date +%s) - T0 ))s] $*"; }
warn() { echo "!! WARNING: $*" >&2; echo "$*" >> "$R/warnings.txt" 2>/dev/null || true; }
die()  { echo "!! RUN FAILED: $*" >&2; exit 1; }
run()  {   # print and run (or only print in dry run)
    echo "+ $*"
    [ "$DRY_RUN" = 1 ] && return 0
    "$@"
}

SERVER_PID=""; NSYS_PID=""; BG_PID=""
cleanup() {
    for p in $BG_PID $SERVER_PID $NSYS_PID; do kill "$p" 2>/dev/null || true; done
    sleep 1
    for p in $BG_PID $SERVER_PID $NSYS_PID; do kill -9 "$p" 2>/dev/null || true; done
    # anything still bound to our port
    if [ "$DRY_RUN" != 1 ]; then pkill -f -- "^$W/build-[a-z]*/bin/llama-server .*--port $PORT" 2>/dev/null || true; fi
    SERVER_PID=""; NSYS_PID=""; BG_PID=""
}
trap 'echo; echo "!! interrupted"; cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

mkdir -p "$R/meta" "$R/bench" "$R/slot" "$R/nsys"

# ---------------------------------------------------------------------------------------------
# preflight
BM="$W/build-master/bin"; BF="$W/build-fix/bin"
if [ "$DRY_RUN" != 1 ]; then
    [ -f "$W/setup_ok" ] || die "run setup.sh first (no $W/setup_ok)"
    for b in "$BM/llama-server" "$BF/llama-server" "$BF/llama-bench"; do [ -x "$b" ] || die "missing $b: run setup.sh"; done
    [ -f "$MODEL" ] || die "missing model $MODEL: run setup.sh"
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
    if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then die "something already listens on port $PORT; stop it or set PORT="; fi
    others="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"
    [ -n "$others" ] && warn "other GPU processes are running (numbers may be off): $others"
fi

PD="$KIT/loadgen/data/first_hour"
if [ -f "$PD/code.jsonl" ] && [ -f "$PD/chat.jsonl" ] && [ -f "$PD/reasoning.jsonl" ] && [ -f "$PD/chinese.jsonl" ]; then
    # October 8 workload: 40 rows of each of 5 kinds from row 30, shuffled with seed 11 (200 requests);
    # the first N_REQ of that order are sent
    WL_ARGS=(--kind-dir "$PD" --start 30 --count 40 --seed 11 --limit "$N_REQ")
    WL_DESC="October 8 five-kind mix (code, tool_calls, chat, chinese, reasoning), first $N_REQ of the 200 in the October order"
else
    WL_ARGS=(--kind-dir "$PD" --kinds tool_calls --start 0 --count 100 --seed 11 --limit "$N_REQ")
    WL_DESC="tool_calls prompts only (public prompt set was not built), $N_REQ requests"
    warn "public prompt set missing; using tool_calls only. Numbers are not the October 8 mix."
fi
echo "$WL_DESC" > "$R/meta/workload.txt"
say "workload: $WL_DESC"

# ---------------------------------------------------------------------------------------------
say "1. machine info"
if [ "$DRY_RUN" != 1 ]; then
    nvidia-smi -q > "$R/meta/nvidia-smi-q.txt" 2>&1
    nvidia-smi > "$R/meta/nvidia-smi.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.max.sm,clocks.max.memory,power.limit,pcie.link.gen.max,pcie.link.width.max \
        --format=csv > "$R/meta/gpu.csv" 2>&1
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 > "$R/meta/gpu_name.txt"
    nvidia-smi | grep -o "CUDA Version: [0-9.]*" > "$R/meta/cuda_driver_version.txt" || true
    { command -v nvcc >/dev/null && nvcc --version || cat "$W/nvcc_path.txt"; } > "$R/meta/nvcc.txt" 2>&1
    git -C "$W/llama.cpp" rev-parse HEAD > "$R/meta/llama_commit.txt" 2>&1
    cp "$W/master.diff" "$R/meta/build-master.diff" 2>/dev/null; cp "$W/fix.diff" "$R/meta/build-fix.diff" 2>/dev/null
    "$BF/llama-server" --version > "$R/meta/llama_server_version.txt" 2>&1 || true
    cp "$W/models/model_sha256.txt" "$R/meta/" 2>/dev/null || true
    { uname -a; echo; lscpu; echo; free -g; echo; df -h "$W"; cat /etc/os-release; } > "$R/meta/host.txt" 2>&1
    cp "$W/setup_ok" "$R/meta/setup_ok.txt" 2>/dev/null || true
    cat "$R/meta/gpu.csv"
fi
GPU_NAME="$(cat "$R/meta/gpu_name.txt" 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------------------------
if [ "${SKIP_BENCH:-0}" != 1 ]; then
    say "2a. llama-bench (build-fix): pp512, tg128, pg512+128"
    if [ "$DRY_RUN" = 1 ] || [ ! -s "$R/bench/llama_bench.json" ]; then
        echo "+ $BF/llama-bench -m $MODEL -ngl 99 -p 512 -n 128 -pg 512,128 -b 2048 -ub 512 -r 3 -o json > $R/bench/llama_bench.json"
        if [ "$DRY_RUN" != 1 ]; then
            timeout "$BENCH_TIMEOUT" "$BF/llama-bench" -m "$MODEL" -ngl 99 -p 512 -n 128 -pg 512,128 -b 2048 -ub 512 -r 3 \
                -o json > "$R/bench/llama_bench.json" 2> "$R/bench/llama_bench.log" \
                || warn "llama-bench failed (see results_h100/bench/llama_bench.log)"
        fi
    fi
    if [ -x "$BF/llama-batched-bench" ] || [ "$DRY_RUN" = 1 ]; then
        say "2b. llama-batched-bench (build-fix): pp128 tg128, npl 1..32"
        if [ "$DRY_RUN" = 1 ] || [ ! -s "$R/bench/batched_bench.jsonl" ]; then
            # no -kvu: one KV stream per sequence like the server; -c 32768 -> 1024 per stream >= 256
            echo "+ $BF/llama-batched-bench -m $MODEL -ngl 99 -c 32768 -b 2048 -ub 512 -npp 128 -ntg 128 -npl 1,2,4,8,16,32 --output-format jsonl"
            if [ "$DRY_RUN" != 1 ]; then
                timeout "$BENCH_TIMEOUT" "$BF/llama-batched-bench" -m "$MODEL" -ngl 99 -c 32768 -b 2048 -ub 512 \
                    -npp 128 -ntg 128 -npl 1,2,4,8,16,32 --output-format jsonl > "$R/bench/batched_bench.log" 2>&1 \
                    || warn "llama-batched-bench failed (see results_h100/bench/batched_bench.log)"
                grep -h '^{"n_kv_max"' "$R/bench/batched_bench.log" > "$R/bench/batched_bench.jsonl" || true
                [ -s "$R/bench/batched_bench.jsonl" ] && cat "$R/bench/batched_bench.jsonl"
            fi
        fi
    else
        warn "llama-batched-bench not built; skipped"
    fi
fi

# ---------------------------------------------------------------------------------------------
# 2c. Nsight Compute: measured DRAM traffic and bandwidth of 1-user decode (build-fix)
# Turns the "~61% of peak bandwidth" roofline ESTIMATE into a MEASURED number, and measures how many
# bytes one forward pass actually reads (the "~4.7 GB of weights per pass" claim).
mkdir -p "$R/ncu"
SUDO_CMD=""; [ "$(id -u)" != 0 ] && SUDO_CMD="sudo"
find_ncu() {
    local c
    c="$(command -v ncu 2>/dev/null || true)"; [ -n "$c" ] && { echo "$c"; return; }
    for c in /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /usr/local/cuda/nsight-compute-*/ncu \
             /usr/local/cuda-*/nsight-compute-*/ncu /opt/nvidia/nsight-compute/*/ncu; do
        [ -x "$c" ] && { echo "$c"; return; }
    done
}
NCU="$(find_ncu)"
if [ "${SKIP_NCU:-0}" = 1 ]; then
    say "2c. ncu: skipped (SKIP_NCU=1)"
else
    if [ -z "$NCU" ] && [ "$DRY_RUN" != 1 ] && [ "${AUTO_INSTALL_NCU:-1}" = 1 ]; then
        say "2c. ncu not found; trying NVIDIA's CUDA apt repo (cuda-nsight-compute-12-*)"
        . /etc/os-release
        repo="ubuntu${VERSION_ID//./}"
        if ! apt-cache policy 2>/dev/null | grep -q developer.download.nvidia.com; then
            curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" \
                && ${SUDO_CMD:-} dpkg -i /tmp/cuda-keyring.deb >> "$R/ncu/install.log" 2>&1
        fi
        ${SUDO_CMD:-} apt-get update -qq >> "$R/ncu/install.log" 2>&1 || true
        pkg="$(apt-cache search --names-only '^cuda-nsight-compute-12-[0-9]+$' 2>/dev/null | awk '{print $1}' | sort -V | tail -n1)"
        if [ -n "$pkg" ]; then
            ${SUDO_CMD:-} apt-get install -y -qq "$pkg" >> "$R/ncu/install.log" 2>&1 || warn "ncu: apt install $pkg failed (see results_h100/ncu/install.log)"
        else
            warn "ncu: no cuda-nsight-compute-12-* package found in apt"
        fi
        NCU="$(find_ncu)"
    fi
    if [ -z "$NCU" ] && [ "$DRY_RUN" != 1 ]; then
        say "2c. ncu: not available, skipped"
        echo "ncu not found" > "$R/ncu/SKIPPED.txt"
    elif [ "$DRY_RUN" = 1 ] || [ ! -s "$R/ncu/fix_tg_b1.csv" ]; then
        NCU="${NCU:-ncu}"
        say "2c. ncu: build-fix llama-bench decode, batch 1, 16 tokens (CUDA graphs off, real clocks)"
        SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -E env PATH=$PATH GGML_CUDA_DISABLE_GRAPHS=1"
        NCU_METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"
        echo "+ GGML_CUDA_DISABLE_GRAPHS=1 $NCU --clock-control none --metrics $NCU_METRICS --csv --page raw --print-units base --log-file $R/ncu/fix_tg_b1.csv $BF/llama-bench -m $MODEL -ngl 99 -p 0 -n 16 -r 1"
        if [ "$DRY_RUN" != 1 ]; then
            # shellcheck disable=SC2086
            GGML_CUDA_DISABLE_GRAPHS=1 timeout 1800 $SUDO "$NCU" --clock-control none --metrics "$NCU_METRICS" \
                --csv --page raw --print-units base --log-file "$R/ncu/fix_tg_b1.csv" \
                "$BF/llama-bench" -m "$MODEL" -ngl 99 -p 0 -n 16 -r 1 -o json \
                > "$R/ncu/fix_tg_b1.bench.json" 2> "$R/ncu/fix_tg_b1.stderr.log" \
                || warn "ncu run failed (see results_h100/ncu/fix_tg_b1.stderr.log; ERR_NVGPUCTRPERM = needs sudo)"
            [ -n "$SUDO" ] && sudo chown "$(id -u):$(id -g)" "$R/ncu/"* 2>/dev/null || true
            head -c 300 "$R/ncu/fix_tg_b1.csv" 2>/dev/null; echo
        fi
    fi
fi

# ---------------------------------------------------------------------------------------------
# 2d. CUDA: 4-bit-weight decode GEMV kernels (gemv/q4gemv.cu), v0 -> v4 + cuBLAS fp16 reference,
#     on the 8 Qwen3-8B projection shapes; then Nsight Compute on each kernel (gate_proj shape)
mkdir -p "$R/gemv"
if [ "${SKIP_GEMV:-0}" = 1 ]; then
    say "2d. gemv: skipped (SKIP_GEMV=1)"
else
    say "2d. gemv: build + correctness + timing (DRAM-resident weights)"
    NVCC="$(command -v nvcc 2>/dev/null || true)"
    if [ -z "$NVCC" ]; then for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /usr/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done; fi
    GB="$W/gemv"; mkdir -p "$GB"
    GEMV_SRC="${GEMV_SRC:-$KIT/gemv/q4gemv.cu}"   # the llm-perfsim repo keeps it in kernels/q4gemv.cu
    # fallback when -arch=native fails: the numeric arch setup.sh built for (90 = H100, 120 = RTX 5090)
    GEMV_ARCH="$(sed -n 's/^arch=\([0-9][0-9]*\)[af]\{0,1\}$/\1/p' "$W/setup_ok" 2>/dev/null | head -n1)"
    GEMV_ARCH="${GEMV_ARCH:-90}"
    echo "+ $NVCC -O3 -std=c++17 -arch=native -o $GB/q4gemv $GEMV_SRC -lcublas"
    if [ "$DRY_RUN" != 1 ]; then
        if [ ! -x "$GB/q4gemv" ]; then
            "$NVCC" -O3 -std=c++17 -arch=native -Xptxas -v -o "$GB/q4gemv" "$GEMV_SRC" -lcublas > "$R/gemv/build.log" 2>&1 \
            || "$NVCC" -O3 -std=c++17 -arch="sm_$GEMV_ARCH" -Xptxas -v -o "$GB/q4gemv" "$GEMV_SRC" -lcublas >> "$R/gemv/build.log" 2>&1 \
            || "$NVCC" -O3 -std=c++17 -arch="sm_$GEMV_ARCH" -DNO_CUBLAS -Xptxas -v -o "$GB/q4gemv" "$GEMV_SRC" >> "$R/gemv/build.log" 2>&1 \
            || warn "gemv build failed (see results_h100/gemv/build.log)"
        fi
        if [ -x "$GB/q4gemv" ]; then
            [ -s "$R/gemv/gemv.jsonl" ] || timeout 900 "$GB/q4gemv" > "$R/gemv/gemv.jsonl" 2> "$R/gemv/gemv.stderr.log" \
                || warn "gemv run failed (see results_h100/gemv/gemv.stderr.log)"
            cat "$R/gemv/gemv.jsonl" 2>/dev/null | python3 -c "import sys,json
for l in sys.stdin:
    r=json.loads(l); print(f\"   {r['shape']:10s} {r['kernel']:28s} {r['us']:9.1f} us {r['GBps']:7.0f} GB/s ok={r['ok']}\")" || true
            if [ -n "${NCU:-}" ] && [ "${SKIP_NCU:-0}" != 1 ] && [ ! -s "$R/gemv/ncu_gemv.csv" ]; then
                say "2d. gemv: Nsight Compute per kernel (gate_proj 12288x4096)"
                SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -E env PATH=$PATH"
                # shellcheck disable=SC2086
                timeout 900 $SUDO "$NCU" --clock-control none --kernel-name regex:"gemv|gemm|gemv2T|Kernel" \
                    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,lts__t_sector_hit_rate.pct \
                    --csv --page raw --print-units base --log-file "$R/gemv/ncu_gemv.csv" "$GB/q4gemv" --ncu \
                    > "$R/gemv/ncu_gemv.stdout.jsonl" 2> "$R/gemv/ncu_gemv.stderr.log" || warn "ncu on gemv failed (see results_h100/gemv/ncu_gemv.stderr.log)"
                [ -n "$SUDO" ] && sudo chown "$(id -u):$(id -g)" "$R/gemv/"* 2>/dev/null || true
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------------------------
start_server() {   # bin log extra-args...  -> sets SERVER_PID
    local bin="$1" log="$2"; shift 2
    # shellcheck disable=SC2086
    echo "+ $bin -m $MODEL --host 127.0.0.1 --port $PORT $SERVER_ARGS_COMMON $* > $log 2>&1 &"
    [ "$DRY_RUN" = 1 ] && return 0
    # shellcheck disable=SC2086
    "$bin" -m "$MODEL" --host 127.0.0.1 --port "$PORT" $SERVER_ARGS_COMMON "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    wait_health "$log"
}
wait_health() {
    local log="$1" i
    for i in $(seq 1 "$HEALTH_TIMEOUT"); do
        if curl -fs "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
            echo "   server up after ${i}s"; return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "   server exited during startup; last log lines:"; tail -n 20 "$log"; SERVER_PID=""; return 1
        fi
        sleep 1
    done
    echo "   server not healthy after ${HEALTH_TIMEOUT}s"; tail -n 20 "$log"; return 1
}
stop_server() {
    [ "$DRY_RUN" = 1 ] && { echo "+ kill <server>"; return 0; }
    [ -z "$SERVER_PID" ] && return 0
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

slot_run() {   # build(master|fix) policy users extra-server-args...
    local build="$1" policy="$2" users="$3"; shift 3
    local bin="$W/build-$build/bin/llama-server" name="${policy}_u${users}"
    local log="$R/slot/${name}.server.log" res="$R/slot/${name}.json"
    say "3. slot test: build-$build, $users users, server args: $SERVER_ARGS_COMMON $*"
    if [ "$DRY_RUN" != 1 ] && [ -s "$res" ]; then echo "   already done, skipped"; return 0; fi
    start_server "$bin" "$log" "$@" || { warn "server failed to start for $name"; stop_server; return 0; }
    if [ "$DRY_RUN" != 1 ]; then
        grep -o "kv_unified = '[a-z]*'" "$log" | head -n1 | sed "s/^/   /" || true
    fi
    local label="$COMMIT+diag_counter"; [ "$build" = fix ] && label="$label+slot_order"
    run timeout "$LOADGEN_TIMEOUT" python3 "$KIT/loadgen/loadgen.py" --url "http://127.0.0.1:$PORT" \
        --policy "$policy" --server-args "-m $(basename "$MODEL") $SERVER_ARGS_COMMON $*" --out "$R/slot" \
        --users "$users" --repeats 1 --max-tokens 192 "${WL_ARGS[@]}" \
        --hardware "$GPU_NAME" --llama-commit "$label" \
        || warn "loadgen failed or timed out for $name"
    if [ "$DRY_RUN" != 1 ]; then
        curl -fs "http://127.0.0.1:$PORT/metrics" > "$R/slot/${name}.metrics.txt" 2>/dev/null || true
        curl -fs "http://127.0.0.1:$PORT/slots" > "$R/slot/${name}.slots.json" 2>/dev/null || true
    fi
    stop_server
    if [ "$DRY_RUN" != 1 ]; then
        local diag; diag="$(grep "^DIAG" "$log" | tail -n1)"
        echo "   ${diag:-no DIAG line (fewer than 100 generation steps?)}"
        echo "${diag}" > "$R/slot/${name}.diag.txt"
        python3 "$KIT/loadgen/step_stats.py" "$log" --users "$users" --repeats 1 --requests "$N_REQ" \
            > "$R/slot/${name}.step_stats.md" 2>&1 || true
    fi
}

if [ "${SKIP_SLOTS:-0}" != 1 ]; then
    for build in master fix; do
        for u in $USERS; do slot_run "$build" "${build}_plain" "$u"; done
    done
    slot_run master master_kvu 8 --kv-unified
fi

# ---------------------------------------------------------------------------------------------
NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "$NSYS" ]; then
    for c in /usr/local/cuda/bin/nsys /usr/local/cuda-*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys /usr/lib/nsight-systems/bin/nsys; do
        [ -x "$c" ] && NSYS="$c" && break
    done
fi
if [ "${SKIP_NSYS:-0}" = 1 ]; then
    say "4. nsys: skipped (SKIP_NSYS=1)"
elif [ -z "$NSYS" ] && [ "$DRY_RUN" != 1 ]; then
    say "4. nsys: not installed, skipped"
    echo "nsys not found" > "$R/nsys/SKIPPED.txt"
else
    NSYS="${NSYS:-nsys}"
    for build in master fix; do
        say "4. nsys: build-$build, 8 users, ~${NSYS_SECONDS}s of load"
        out="$R/nsys/${build}_u8"
        if [ "$DRY_RUN" != 1 ] && [ -s "$out.stats.txt" ]; then echo "   already done, skipped"; continue; fi
        echo "+ $NSYS profile -t cuda --stats=true --force-overwrite=true -o $out $W/build-$build/bin/llama-server -m $MODEL --host 127.0.0.1 --port $PORT $SERVER_ARGS_COMMON &"
        echo "+ python3 loadgen/synthetic_load.py http://127.0.0.1:$PORT 8 100000 &  (stopped after ${NSYS_SECONDS}s)"
        [ "$DRY_RUN" = 1 ] && continue
        # shellcheck disable=SC2086
        timeout 900 "$NSYS" profile -t cuda --stats=true --force-overwrite=true -o "$out" \
            "$W/build-$build/bin/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" $SERVER_ARGS_COMMON \
            > "$out.stats.txt" 2> "$out.log" &
        NSYS_PID=$!
        ok=0
        for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
            if curl -fs "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
            kill -0 "$NSYS_PID" 2>/dev/null || break
            sleep 1
        done
        if [ "$ok" = 1 ]; then
            python3 "$KIT/loadgen/synthetic_load.py" "http://127.0.0.1:$PORT" 8 100000 > "$out.load.txt" 2>&1 &
            BG_PID=$!
            sleep "$NSYS_SECONDS"
            curl -fs "http://127.0.0.1:$PORT/metrics" > "$out.metrics.txt" 2>/dev/null || true
            kill "$BG_PID" 2>/dev/null || true; wait "$BG_PID" 2>/dev/null || true; BG_PID=""
        else
            warn "nsys: server did not come up for $build (see $out.log)"
        fi
        # stop the profiled server gracefully so nsys writes the report and the stats
        pkill -TERM -f -- "^$W/build-$build/bin/llama-server .*--port $PORT" 2>/dev/null || true   # anchored: not nsys itself
        for _ in $(seq 1 600); do kill -0 "$NSYS_PID" 2>/dev/null || break; sleep 1; done
        kill "$NSYS_PID" 2>/dev/null || true; wait "$NSYS_PID" 2>/dev/null || true; NSYS_PID=""
        if [ -f "$out.nsys-rep" ]; then
            for rep in cuda_gpu_kern_sum cuda_api_sum; do
                timeout 300 "$NSYS" stats -q --report "$rep" --format csv --force-export=true -o "$out" "$out.nsys-rep" \
                    >> "$out.log" 2>&1 || true
            done
            # the .nsys-rep can be big; keep it but outside the tarball if > 200 MB
            sz=$(stat -c %s "$out.nsys-rep" 2>/dev/null || echo 0)
            if [ "$sz" -gt 200000000 ]; then mkdir -p "$KIT/nsys_big"; mv "$out.nsys-rep" "$KIT/nsys_big/"; fi
            rm -f "$out.sqlite"
            grep -iE "cudaGraphLaunch|cudaLaunchKernel" "$out.stats.txt" | head -n 4 | sed 's/^/   /' || true
        else
            warn "nsys: no report for $build (see $out.log)"
        fi
    done
fi

# ---------------------------------------------------------------------------------------------
say "5. summary"
if [ "$DRY_RUN" = 1 ]; then
    echo "+ python3 (inline) -> $R/summary.md"
    echo "+ tar -czf $KIT/results_h100.tar.gz -C $KIT results_h100"
    say "dry run done"; exit 0
fi
python3 - "$R" "$USERS" <<'EOF'
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1]); users = [int(u) for u in sys.argv[2].split()]
def read(p, default=""):
    try: return Path(p).read_text().strip()
    except Exception: return default
out = []
out.append("# H100 results: llama.cpp slot-order fix + roofline check\n")
out.append(f"- GPU: {read(R/'meta/gpu_name.txt','?')}  ({read(R/'meta/cuda_driver_version.txt')})")
out.append(f"- llama.cpp: {read(R/'meta/llama_commit.txt','?')} + diag_counter.patch (build-master) / + slot_order.patch (build-fix)")
out.append(f"- model: Qwen3-8B-Q4_K_M.gguf  sha256 {read(R/'meta/model_sha256.txt','?').split(' ')[0]}")
out.append(f"- workload: {read(R/'meta/workload.txt')}; closed loop, greedy, max 192 tokens (384 with thinking), cache_prompt off, 1 repeat")
out.append("- server: `llama-server -c 65536 -np 16 -ngl 99 --seed 0 --metrics` (one KV stream per slot unless noted)\n")

# llama-bench
try:
    rows = json.loads(read(R/"bench/llama_bench.json", "[]") or "[]")
    if rows:
        out.append("## llama-bench (build-fix)\n\n| test | t/s | stdev |\n|---|---|---|")
        for r in rows:
            p, g = r.get("n_prompt", 0), r.get("n_gen", 0)
            name = f"pp{p}" if g == 0 else (f"tg{g}" if p == 0 else f"pp{p}+tg{g}")
            out.append(f"| {name} | {r.get('avg_ts', 0):.1f} | {r.get('stddev_ts', 0):.1f} |")
        out.append("")
except Exception as e:
    out.append(f"(llama-bench json unreadable: {e})\n")

# batched-bench
bb = [json.loads(l) for l in read(R/"bench/batched_bench.jsonl").splitlines() if l.startswith("{")]
if bb:
    out.append("## llama-batched-bench (build-fix, pp128 tg128, per-sequence KV)\n\n| B | S_PP t/s | S_TG t/s (decode, all seqs) | per-seq decode t/s |\n|---|---|---|---|")
    for r in bb:
        out.append(f"| {r['pl']} | {r['speed_pp']:.0f} | {r['speed_tg']:.1f} | {r['speed_tg']/r['pl']:.1f} |")
    out.append("")

# slot test
def slot(policy, u):
    f = R/"slot"/f"{policy}_u{u}.json"
    tok = p50 = fail = None
    if f.exists():
        try:
            p = json.loads(f.read_text())["passes"][0]
            tok, p50, fail = p["throughput_tok_s"], p["latency_p50_s"], f"{p['failed']}/{p['requests']}"
        except Exception: pass
    d = read(R/"slot"/f"{policy}_u{u}.diag.txt")
    m = re.search(r"passes_per_step=([\d.]+)", d); k = float(m.group(1)) if m else None
    return tok, k, p50, fail
fmt = lambda v, s="{:.1f}": "-" if v is None else s.format(v)
oct8 = {1: (228.3, 228.1), 4: (262.5, 487.8), 8: (333.6, 878.9), 16: (1143.6, 1136.2)}
out.append("## Slot test: aggregate decode tokens/s and forward passes per generation step\n")
out.append("| users | master tok/s | master passes/step | fix tok/s | fix passes/step | fix / master | RTX 5090 Oct 8 (master -> fix) |")
out.append("|---|---|---|---|---|---|---|")
for u in users:
    tm, km, _, fm = slot("master_plain", u); tf, kf, _, ff = slot("fix_plain", u)
    ratio = (tf / tm) if (tm and tf) else None
    o = oct8.get(u); ostr = f"{o[0]} -> {o[1]} ({o[1]/o[0]:.2f}x)" if o else "-"
    out.append(f"| {u} | {fmt(tm)} | {fmt(km, '{:.2f}')} | {fmt(tf)} | {fmt(kf, '{:.2f}')} | {fmt(ratio, '{:.2f}x')} | {ostr} |")
tk, kk, _, _ = slot("master_kvu", 8)
out.append(f"| 8, master --kv-unified | {fmt(tk)} | {fmt(kk, '{:.2f}')} | | | | 639.0 |")
out.append("\npasses/step = last DIAG line of the server log (cumulative over the run, generation-only llama_decode calls).")
out.append("Failed requests: " + ", ".join(f"{p}_u{u} {slot(p,u)[3]}" for p in ("master_plain","fix_plain") for u in users if slot(p,u)[3]) + "\n")

# ncu
import csv
PEAK = [("H100 NVL", 3900), ("H100 PCIe", 2000), ("H100", 3350), ("A100-SXM4-80GB", 2039), ("A100 80GB PCIe", 1935),
        ("A100", 1555), ("RTX 5090", 1792), ("L40S", 864)]
gpu = read(R/"meta/gpu_name.txt", "?")
peak = next((v for k, v in PEAK if k in gpu), None)
f = R/"ncu"/"fix_tg_b1.csv"
if f.exists():
    try:
        lines = f.read_text(errors="replace").splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith('"ID"') or '"Kernel Name"' in l)
        rows = list(csv.reader(lines[start:]))
        hdr = rows[0]; data = [r for r in rows[2:] if len(r) == len(hdr)]   # rows[1] = units
        col = lambda n: hdr.index(n)
        def num(x):
            try: return float(x.replace(",", ""))
            except Exception: return 0.0
        kn, t, rd, wr = col("Kernel Name"), col("gpu__time_duration.sum"), col("dram__bytes_read.sum"), col("dram__bytes_write.sum")
        T = sum(num(r[t]) for r in data) * 1e-9; RD = sum(num(r[rd]) for r in data); WR = sum(num(r[wr]) for r in data)
        passes = sum(1 for r in data if "get_rows" in r[kn])
        bw = (RD + WR) / T / 1e9 if T else 0
        out.append("## Nsight Compute: 1-user decode, build-fix (llama-bench tg16, CUDA graphs off, real clocks)\n")
        out.append(f"- kernels profiled: {len(data)}; forward passes (get_rows launches): {passes}")
        out.append(f"- DRAM read {RD/1e9:.2f} GB, write {WR/1e9:.3f} GB over {T*1e3:.1f} ms of kernel time")
        if passes:
            out.append(f"- **measured DRAM read per forward pass: {RD/passes/1e9:.2f} GB** (model: 4.68 GB of weights read per pass)")
            out.append(f"- kernel time per pass: {T/passes*1e3:.2f} ms")
        out.append(f"- **achieved DRAM bandwidth over kernel time: {bw:.0f} GB/s" + (f" = {100*bw/peak:.0f}% of {peak} GB/s spec peak**" if peak else "**"))
        try:
            b = json.loads(read(R/"ncu/fix_tg_b1.bench.json", "[]") or "[]")
            tg = next((r for r in b if r.get("n_prompt", 1) == 0), None)
            if tg and passes:
                ms_tok = 1000.0 / tg["avg_ts"]
                out.append(f"- end-to-end under ncu: {tg['avg_ts']:.1f} t/s ({ms_tok:.2f} ms/token) -- slowed by profiling, use llama-bench above for speed")
        except Exception: pass
        agg = {}
        for r in data:
            k = r[kn].split("(")[0].split("<")[0].strip()[:60]
            a = agg.setdefault(k, [0.0, 0.0, 0]); a[0] += num(r[t]); a[1] += num(r[rd]) + num(r[wr]); a[2] += 1
        out.append("\n| kernel | launches | time share | GB/s while running |\n|---|---|---|---|")
        for k, (kt, kb, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:8]:
            out.append(f"| `{k}` | {c} | {100*kt/1e9/T:.1f}% | {kb/(kt*1e-9)/1e9:.0f} |")
        out.append("")
    except Exception as e:
        out.append(f"(ncu csv unreadable: {e})\n")
if (R/"ncu/SKIPPED.txt").exists():
    out.append("ncu: not available on the VM, skipped.\n")

# gemv kernels
g = [json.loads(l) for l in read(R/"gemv/gemv.jsonl").splitlines() if l.startswith("{")]
if g:
    out.append("## CUDA 4-bit decode GEMV kernels (gemv/q4gemv.cu): time and achieved DRAM bandwidth\n")
    out.append("Weights rotated through copies > 4x L2, so each launch reads from DRAM. Bytes = weights + scales + x + y.\n")
    kernels = []
    for r in g:
        if r["kernel"] not in kernels: kernels.append(r["kernel"])
    shapes = []
    for r in g:
        if r["shape"] not in shapes: shapes.append(r["shape"])
    out.append("| shape (N x K) | " + " | ".join(kernels) + " |")
    out.append("|---|" + "---|" * len(kernels))
    for sh in shapes:
        rr = {r["kernel"]: r for r in g if r["shape"] == sh}
        any_r = next(iter(rr.values()))
        cells = []
        for k in kernels:
            r = rr.get(k)
            if not r: cells.append("-"); continue
            pct = f" ({100*r['GBps']/peak:.0f}%)" if peak else ""
            cells.append(f"{r['us']:.1f} us, {r['GBps']:.0f} GB/s{pct}" + ("" if r["ok"] else " WRONG"))
        out.append(f"| {sh} ({any_r['N']}x{any_r['K']}) | " + " | ".join(cells) + " |")
    bad = [f"{r['shape']}/{r['kernel']}" for r in g if not r["ok"]]
    out.append("\ncorrectness vs CPU double reference: " + ("all passed" if not bad else "FAILED: " + ", ".join(bad)))
    # per-token projection: 36 layers x (q,k,v,o,gate,up,down) + lm_head, best 4-bit kernel per shape
    layer = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    best = {}
    for sh in shapes:
        cand = [r for r in g if r["shape"] == sh and r["ok"] and r["kernel"].startswith("v")]
        if cand: best[sh] = min(cand, key=lambda r: r["us"])
    if all(s in best for s in layer + ["lm_head"]):
        tok_us = 36 * sum(best[s]["us"] for s in layer) + best["lm_head"]["us"]
        out.append(f"\nweight GEMVs for one Qwen3-8B token with the best kernel per shape: {tok_us/1000:.2f} ms "
                   f"(= {1e6/tok_us:.0f} tok/s if GEMVs were the only work; format is 4.5 bits/weight, not llama.cpp's Q4_K_M mix)")
        try:
            rows = json.loads(read(R/"bench/llama_bench.json", "[]") or "[]")
            tg = next((r for r in rows if r.get("n_prompt", 1) == 0), None)
            if tg: out.append(f"llama.cpp tg128 on this GPU: {1000/tg['avg_ts']:.2f} ms/token ({tg['avg_ts']:.1f} tok/s)")
        except Exception: pass
    out.append("")
f = R/"gemv"/"ncu_gemv.csv"
if f.exists():
    try:
        lines = f.read_text(errors="replace").splitlines()
        start = next(i for i, l in enumerate(lines) if '"Kernel Name"' in l)
        rows = list(csv.reader(lines[start:])); hdr = rows[0]; data = [r for r in rows[2:] if len(r) == len(hdr)]
        c = {n: hdr.index(n) for n in hdr}
        def v(r, n):
            try: return float(r[c[n]].replace(",", ""))
            except Exception: return float("nan")
        out.append("## Nsight Compute per GEMV kernel (gate_proj 12288x4096, one launch each)\n")
        out.append("| kernel | time us | DRAM read MB | DRAM % of peak | warps active % | regs | L2 hit % |\n|---|---|---|---|---|---|---|")
        for r in data:
            name = r[c["Kernel Name"]].split("(")[0][:40]
            out.append(f"| `{name}` | {v(r,'gpu__time_duration.sum')/1e3:.1f} | {v(r,'dram__bytes_read.sum')/1e6:.1f} | "
                       f"{v(r,'dram__throughput.avg.pct_of_peak_sustained_elapsed'):.0f} | {v(r,'sm__warps_active.avg.pct_of_peak_sustained_active'):.0f} | "
                       f"{v(r,'launch__registers_per_thread'):.0f} | {v(r,'lts__t_sector_hit_rate.pct'):.0f} |")
        out.append("")
    except Exception as e:
        out.append(f"(ncu gemv csv unreadable: {e})\n")

# nsys
for b in ("master", "fix"):
    t = read(R/"nsys"/f"{b}_u8.stats.txt")
    lines = [l for l in t.splitlines() if re.search(r"cudaGraphLaunch|cudaLaunchKernel|cudaStreamSynchronize", l)]
    if lines:
        out.append(f"## nsys (build-{b}, 8 users, ~20 s)\n\n```\n" + "\n".join(lines[:6]) + "\n```\n")
if (R/"nsys/SKIPPED.txt").exists():
    out.append("nsys: not installed on the VM, skipped.\n")
w = read(R/"warnings.txt")
if w: out.append("## Warnings\n\n```\n" + w + "\n```\n")
(R/"summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
EOF
tar -czf "$KIT/results_h100.tar.gz" -C "$KIT" results_h100 || die "tar failed"
say "done. Send back: $KIT/results_h100.tar.gz ($(du -h "$KIT/results_h100.tar.gz" | cut -f1))"
