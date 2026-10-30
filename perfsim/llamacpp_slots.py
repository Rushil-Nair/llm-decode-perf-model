"""Model of the llama.cpp slot-ordering bug vs the owner's RTX 5090 measurements.

Mechanism (from the owner's investigation): with per-slot KV, split_equal only
batches consecutive sequence ids, and LRU slot selection scattered the busy
slot ids, so one decode step of B sequences became k > 1 forward passes. Every
pass re-reads all weights; the per-sequence work (KV reads, per-token compute)
is the same however the sequences are grouped. So, per decode step:

    t_step(k, B) = k * a + B * c

    a = per-pass cost that does not depend on batch (dominated by weight reads)
    c = per-sequence cost per step (KV-cache reads, per-token kernel work, and
        anything else that scales with users)

Two ways to get a and c:

1. "ideal" -- pure bandwidth roofline (roofline.step with eff=1): a = weight
   bytes / BW, c = KV bytes per sequence / BW (+ activations). No fitting.
2. "calibrated" -- fit a and c to two *post-patch* measurements where k ~= 1
   (1 user and 16 users; with 16 clients on 16 slots every slot is busy, so ids
   are contiguous). Then predict the held-out points (4 and 8 users, before
   and after) using the forward-pass counts from the counters.

Model inputs that are assumptions (not measured) are listed in ASSUMPTIONS.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import yaml

from .hardware import load_hardware
from .models import REPO_ROOT, load_model
from .roofline import step

DATA = REPO_ROOT / "data" / "llamacpp_rtx5090_qwen3_8b.yaml"

GGUF_Q4_K_M_BYTES = 5_027_783_488  # Qwen/Qwen3-8B-GGUF file listing on HF
Q4_K_BITS = 4.5  # ggml-common.h: block_q4_K "Effectively 4.5 bits per weight"

ASSUMPTIONS = {
    "token_embd_type": "Q4_K (4.5 bpw); the embedding table is stored but only one row per token "
                       "is read, so it is subtracted from per-pass weight bytes. Not verified "
                       "from the GGUF header; the full-file bound is reported alongside.",
    "kv_bits": "16 (llama.cpp default f16 K and V cache)",
    "avg_context_tokens": "512 (prompt lengths were not logged; sensitivity at 0 and 2048 is reported)",
    "passes_after_patch_1_4_16_users": "1.0 (one sequence, or all slots busy, or patched ordering)",
}


def load_measurements(path: Path = DATA) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def weight_bytes_per_pass(include_embedding: bool = False) -> float:
    m = load_model("qwen3-8b")
    if include_embedding:
        return float(GGUF_Q4_K_M_BYTES)
    return GGUF_Q4_K_M_BYTES - m.params()["embedding"] * Q4_K_BITS / 8


def ideal_coeffs(L: int = 512, include_embedding: bool = False, hw_key: str = "rtx5090"):
    """(a, c) in seconds from the pure bandwidth roofline (no fitting)."""
    m = load_model("qwen3-8b")
    hw = load_hardware()[hw_key]
    W = weight_bytes_per_pass(include_embedding)
    kw = dict(w_bits=Q4_K_BITS, kv_bits=16, precision="fp16", weight_bytes_override=W)
    t1 = step(m, hw, 1, L, 1, **kw)["time"]
    t2 = step(m, hw, 2, L, 1, **kw)["time"]
    c = t2 - t1
    return t1 - c, c


def calibrated_coeffs(meas: Dict):
    """Fit t_step = a + B*c through post-patch 1-user and 16-user points (k=1)."""
    thr = meas["throughput_tok_s"]
    t1 = 1 / thr[1][1]
    t16 = 16 / thr[16][1]
    c = (t16 - t1) / 15
    return t1 - c, c


def predict_throughput(a: float, c: float, B: int, k: float) -> float:
    return B / (k * a + B * c)


def implied_passes(a: float, c: float, B: int, measured_tok_s: float) -> float:
    """k that would reproduce a measured throughput under t = k*a + B*c."""
    return (B / measured_tok_s - B * c) / a


def analyse(L: int = 512) -> Dict:
    meas = load_measurements()
    hw = load_hardware()["rtx5090"]
    thr = meas["throughput_tok_s"]
    cnt = meas["counters"]
    k_before_cpu = cnt["cpu_check_master"]["forward_passes_per_generation_step_before"]
    k_after_cpu = cnt["cpu_check_master"]["forward_passes_per_generation_step_after"]
    g = cnt["gpu_run_before"]
    tok_per_ub_before = g["tokens"] / g["ubatches"]
    # 8 decode tokens per step / mean tokens per ubatch. Prompt ubatches raise the
    # mean, so this is a LOWER bound on decode passes per step.
    k_before_gpu_lb = 8 / tok_per_ub_before

    out: Dict = {"L": L, "assumptions": ASSUMPTIONS}

    # ---- 1-user roofline fraction ------------------------------------------
    roof = {}
    for label, inc in (("weights_excl_embedding", False), ("full_gguf_file", True)):
        a, c = ideal_coeffs(L, include_embedding=inc)
        bound = 1 / (a + c)
        roof[label] = {
            "bytes_per_token": (a + c) * hw.memory_bandwidth,
            "bound_tok_s": bound,
            "measured_tok_s": thr[1][0],
            "fraction": thr[1][0] / bound,
            "achieved_bw_GBs": thr[1][0] * (a + c) * hw.memory_bandwidth / 1e9,
        }
    out["one_user_roofline"] = roof
    out["one_user_roofline_sensitivity"] = {
        Lx: thr[1][0] / (1 / sum(ideal_coeffs(Lx))) for Lx in (0, 512, 2048)
    }

    # ---- 8-user before/after ------------------------------------------------
    rows = []
    a_i, c_i = ideal_coeffs(L)
    a_c, c_c = calibrated_coeffs(meas)
    out["coeffs"] = {"ideal": (a_i, c_i), "calibrated": (a_c, c_c),
                     "calibrated_effective_weight_bw_GBs": weight_bytes_per_pass() / a_c / 1e9}
    cases = [
        (1, "after", 1.0), (4, "after", 1.0), (8, "after", k_after_cpu), (16, "after", 1.0),
        (8, "before", k_before_cpu), (8, "before (GPU-counter lower bound on k)", k_before_gpu_lb),
    ]
    for B, phase, k in cases:
        meas_v = thr[B][1] if phase == "after" else thr[B][0]
        p_i = predict_throughput(a_i, c_i, B, k)
        p_c = predict_throughput(a_c, c_c, B, k)
        rows.append({
            "users": B, "phase": phase, "passes_per_step_k": round(k, 3),
            "measured_tok_s": meas_v,
            "ideal_roofline_tok_s": round(p_i, 1),
            "calibrated_tok_s": round(p_c, 1),
            "calibrated_err_pct": round(100 * (p_c - meas_v) / meas_v, 1),
            "used_for_fit": B in (1, 16) and phase == "after",
        })
    out["rows"] = rows

    meas_ratio = thr[8][1] / thr[8][0]
    out["speedup_8_users"] = {
        "measured": meas_ratio,
        "pass_ratio_cpu_counter": k_before_cpu / k_after_cpu,
        "ideal_roofline": predict_throughput(a_i, c_i, 8, k_after_cpu)
        / predict_throughput(a_i, c_i, 8, k_before_cpu),
        "calibrated": predict_throughput(a_c, c_c, 8, k_after_cpu)
        / predict_throughput(a_c, c_c, 8, k_before_cpu),
        "calibrated_with_gpu_counter_k": predict_throughput(a_c, c_c, 8, k_after_cpu)
        / predict_throughput(a_c, c_c, 8, k_before_gpu_lb),
    }
    out["implied_passes"] = {
        "4 users before": implied_passes(a_c, c_c, 4, thr[4][0]),
        "8 users before": implied_passes(a_c, c_c, 8, thr[8][0]),
        "8 users --kv-unified": implied_passes(a_c, c_c, 8, meas["kv_unified_8_users_before_patch_tok_s"]),
        "4 users after": implied_passes(a_c, c_c, 4, thr[4][1]),
        "8 users after": implied_passes(a_c, c_c, 8, thr[8][1]),
    }
    out["counters"] = {
        "gpu_before_tokens_per_ubatch": tok_per_ub_before,
        "gpu_before_single_seq_fraction": g["single_sequence_ubatches"] / g["ubatches"],
        "gpu_after_all8_fraction": cnt["gpu_run_after"]["ubatches_with_all_8_sequences"]
        / cnt["gpu_run_after"]["ubatches"],
        "k_before_gpu_lower_bound": k_before_gpu_lb,
    }
    return out


def to_markdown(r: Dict) -> str:
    L = r["L"]
    s = []
    s.append("### 1-user decode vs RTX 5090 bandwidth roofline\n")
    s.append("Bound = 1792 GB/s / (weight bytes read per token + KV bytes at "
             f"context {L}). Measured = 228.3 tok/s (before patch).\n")
    s.append("| weight bytes counted | bytes/token (model) | bound tok/s (model) | measured tok/s | % of bound | achieved GB/s |")
    s.append("|---|---|---|---|---|---|")
    for k, v in r["one_user_roofline"].items():
        s.append(f"| {k} | {v['bytes_per_token']/1e9:.3f} GB | {v['bound_tok_s']:.1f} | "
                 f"{v['measured_tok_s']:.1f} | {100*v['fraction']:.1f}% | {v['achieved_bw_GBs']:.0f} |")
    sens = ", ".join(f"L={k}: {100*v:.1f}%" for k, v in r["one_user_roofline_sensitivity"].items())
    s.append(f"\nSensitivity to the assumed context (weights excl. embedding): {sens}.\n")

    a_i, c_i = r["coeffs"]["ideal"]
    a_c, c_c = r["coeffs"]["calibrated"]
    s.append("### Slot bug: t_step = k*a + B*c\n")
    s.append(f"- ideal roofline: a = {a_i*1e3:.3f} ms/pass, c = {c_i*1e3:.4f} ms/sequence (no fitting)")
    s.append(f"- calibrated on post-patch 1-user and 16-user points: a = {a_c*1e3:.3f} ms/pass, "
             f"c = {c_c*1e3:.3f} ms/sequence (a implies {r['coeffs']['calibrated_effective_weight_bw_GBs']:.0f} GB/s effective weight streaming)\n")
    s.append("| users | phase | k (passes/step) | measured tok/s | ideal-roofline prediction | calibrated prediction | calibrated error |")
    s.append("|---|---|---|---|---|---|---|")
    for row in r["rows"]:
        tag = " (fit point)" if row["used_for_fit"] else ""
        s.append(f"| {row['users']} | {row['phase']}{tag} | {row['passes_per_step_k']} | {row['measured_tok_s']} | "
                 f"{row['ideal_roofline_tok_s']} | {row['calibrated_tok_s']} | {row['calibrated_err_pct']:+.1f}% |")
    sp = r["speedup_8_users"]
    s.append("\n8-user speedup (after/before):\n")
    s.append(f"- measured: {sp['measured']:.2f}x")
    s.append(f"- forward-pass ratio from CPU counter (4.34/1.05): {sp['pass_ratio_cpu_counter']:.2f}x")
    s.append(f"- ideal roofline model: {sp['ideal_roofline']:.2f}x")
    s.append(f"- calibrated model (CPU-counter k): {sp['calibrated']:.2f}x")
    s.append(f"- calibrated model (GPU-counter lower-bound k = {r['counters']['k_before_gpu_lower_bound']:.2f}): "
             f"{sp['calibrated_with_gpu_counter_k']:.2f}x\n")
    s.append("Passes per step implied by the calibrated model (inverting it on measured throughput):\n")
    for k, v in r["implied_passes"].items():
        s.append(f"- {k}: {v:.2f}")
    return "\n".join(s) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = ap.parse_args(argv)
    r = analyse(args.context)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "llamacpp_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r["rows"][0]))
        w.writeheader()
        w.writerows(r["rows"])
    md = to_markdown(r)
    (out / "llamacpp_validation.md").write_text(md)
    print(md)
    return r


if __name__ == "__main__":
    main()
