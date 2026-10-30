# shellcheck shell=bash
# Shared helpers for bench/run_*.sh (sourced, not run).

# gpu_tag from nvidia-smi's GPU name (CONTRACTS: h100_sxm, h100_nvl, h100_pcie, rtx5090).
# Unknown GPUs get a lower-case, underscore-joined version of the name.
detect_gpu_tag() {
    local name
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    case "$name" in
        *"H100"*"NVL"*)            echo h100_nvl ;;
        *"H100"*"PCIe"*)           echo h100_pcie ;;
        *"H100"*)                  echo h100_sxm ;;   # "NVIDIA H100 80GB HBM3" is the SXM part
        *"RTX 5090"*)              echo rtx5090 ;;
        "")                        echo unknown_gpu ;;
        *) echo "$name" | tr '[:upper:]' '[:lower:]' | sed -e 's/nvidia//' -e 's/geforce//' \
                 -e 's/[^a-z0-9]\+/_/g' -e 's/^_//' -e 's/_$//' ;;
    esac
}

# pip install into the current python3, trying the variants that work on Lambda/Ubuntu images.
pip_install() {
    python3 -m pip install -q "$@" \
        || python3 -m pip install -q --user "$@" \
        || python3 -m pip install -q --break-system-packages "$@" \
        || python3 -m pip install -q --user --break-system-packages "$@"
}
