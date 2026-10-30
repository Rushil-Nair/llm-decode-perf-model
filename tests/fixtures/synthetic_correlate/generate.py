"""Regenerate the SYNTHETIC fixture tree tests/fixtures/synthetic_correlate/data/.

    cd <repo> && python3 tests/fixtures/synthetic_correlate/generate.py

Every "measured" number here is the stub model's prediction times a designed error
factor, so the tests know the right answers. NOTHING here is a measurement; these files
must never be copied into data/. Each file is marked synthetic (env.json "synthetic":
true, gpu_name "SYNTHETIC ...").

Designed errors (error % = predicted vs "measured"):
  h100_sxm  vLLM TPOT batch=64        -> -20 % (must be flagged at 15 %)
            gemv v2 ilp=1             -> about -20 % (must be flagged)
            everything else           -> within +-6 %
  rtx5090   all within +-6 %; one malformed JSON line in vllm_latency.jsonl;
            no ops.jsonl and no llamacpp/ dir (exercise the skip notes)
  h100_pcie env.json only (no measurements at all)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from perfsim.hardware import load_hardware  # noqa: E402
from perfsim.models import load_model  # noqa: E402

import stubs  # noqa: E402
from correlate.loaders import GEMV_BYTES_PER_LOAD  # noqa: E402

OUT = HERE / "data"
GPUS = {
    "h100_sxm": dict(sm=132, lat=650.0, peak=3000e9, l2=50 * 2**20, calib={"mem_eff": 0.85, "compute_eff": 0.6}),
    "rtx5090": dict(sm=170, lat=560.0, peak=1650e9, l2=96 * 2**20, calib={"mem_eff": 0.92, "compute_eff": 0.5}),
}
WIGGLE = [0.03, -0.02, 0.01, -0.04, 0.02, -0.01]


def wjsonl(path: Path, rows, extra_lines=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n")


def gen_gpu(tag: str, g: dict):
    d = OUT / tag
    hw = load_hardware()[tag]
    model = load_model("qwen3-8b")
    params = stubs.StubGPUParams(g["sm"], g["lat"], g["peak"], g["l2"])
    (d).mkdir(parents=True, exist_ok=True)
    (d / "env.json").write_text(json.dumps({
        "gpu_name": f"SYNTHETIC {hw.name}", "gpu_tag": tag, "driver": "synthetic", "cuda": "synthetic",
        "sm_count": g["sm"], "l2_bytes": g["l2"], "provider": "synthetic-fixture", "counters_ok": False,
        "clocks_locked": False, "sm_clock_mhz": None, "mem_clock_mhz": None, "synthetic": True}, indent=1))

    probes = []
    for fp, lat in ((4 * 2**20, 250.0), (2 * 2**30, g["lat"])):
        for stride, f in ((64, 0.8), (128, 1.0), (256, 1.0)):
            probes.append({"probe": "pointer_chase", "footprint_bytes": fp, "stride_bytes": stride,
                           "latency_cycles": lat * f * 1.9, "latency_ns": lat * f})
    for bps in (1, 2, 4, 8):
        bif = bps * 256 * 64 * g["sm"]
        probes.append({"probe": "stream", "blocks_per_sm": bps, "threads_per_block": 256,
                       "bytes_per_thread_in_flight": 64, "bytes_in_flight_total": bif,
                       "GBps": stubs.predict_bw(params, bif, True, 0.2) / 1e9})
    wjsonl(d / "probes.jsonl", probes)

    sweep, i = [], 0
    for kern, ilps in (("v2", (1,)), ("v3", (1, 2)), ("v4", (2, 4))):
        for ilp in ilps:
            for warps in (4, 8, 16, 32, 48):
                bif = stubs.bytes_in_flight(warps, ilp, GEMV_BYTES_PER_LOAD, g["sm"])
                bw = stubs.predict_bw(params, bif, True, 0.2) * (1 + WIGGLE[i % len(WIGGLE)])
                if tag == "h100_sxm" and kern == "v2":
                    bw *= 1.25   # designed: model under-predicts v2 ilp=1 by ~20 %
                i += 1
                N, K = 12288, 4096
                nbytes = N * K // 2 + N * K // 32 * 2 + K * 2 + N * 4
                sweep.append({"kernel": kern, "ilp": ilp, "blocks_per_sm": warps // 4, "warps_per_sm": warps,
                              "N": N, "K": K, "us": nbytes / bw * 1e6, "bytes": nbytes, "GBps": bw / 1e9, "ok": True})
    sweep.append({"kernel": "v4", "ilp": 8, "blocks_per_sm": 1, "warps_per_sm": 4, "N": 1, "K": 1,
                  "us": 0, "bytes": 0, "GBps": 0, "ok": False})  # failed run: must be dropped
    wjsonl(d / "gemv_sweep.jsonl", sweep)

    (d / "calib.json").write_text(json.dumps({"synthetic": True, **g["calib"]}, indent=1))
    cost = stubs.StubCost(hw, g["calib"], "empirical")

    if tag == "h100_sxm":
        ops = []
        for M, eff in ((1, 0.75), (16, 0.7), (256, 0.55), (4096, 0.7)):
            N = K = 4096
            fl, by = 2.0 * M * N * K, 2.0 * (M * K + K * N + M * N)
            sol = max(fl / hw.peak("bf16"), by / hw.memory_bandwidth)
            ops.append({"op": "gemm", "dtype": "bf16", "shape": {"M": M, "N": N, "K": K}, "us": sol / eff * 1e6,
                        "flops": fl, "bytes": by})
        for n_tok in (1, 64, 4096):
            by = 2.0 * 2 * n_tok * 4096
            ops.append({"op": "rmsnorm", "dtype": "bf16", "shape": {"tokens": n_tok, "hidden": 4096},
                        "us": max(by / hw.memory_bandwidth / 0.5, 3e-6) * 1e6, "flops": 4.0 * n_tok * 4096,
                        "bytes": by})
        for kv in (1024, 8192):
            by = 8 * kv * 2 * 8 * 128 * 2.0
            ops.append({"op": "attn_decode", "dtype": "bf16", "shape": {"batch": 8, "kv_len": kv},
                        "us": by / hw.memory_bandwidth / 0.6 * 1e6, "flops": 8 * 32 * kv * 4.0 * 128, "bytes": by})
        wjsonl(d / "ops.jsonl", ops)

    # vLLM bench latency: (batch, input_len, out) with out in {1, 128}
    vl, j = [], 0
    for b in (1, 8, 64):
        for il in (512, 2048):
            for ol in (1, 128):
                t = stubs.predict_vllm_latency(model, cost, b, il, ol)
                if ol > 1:
                    t1 = stubs.predict_vllm_latency(model, cost, b, il, 1)
                    tp = (t - t1) / (ol - 1)
                    f = 0.80 if (tag == "h100_sxm" and b == 64) else 1 + WIGGLE[j % len(WIGGLE)]
                    t = t1_meas + (ol - 1) * tp / f   # measured TPOT = pred / f -> error = f - 1
                else:
                    f = 1 + WIGGLE[(j + 2) % len(WIGGLE)]
                    t = t / f
                    t1_meas = t
                j += 1
                vl.append({"engine": "vllm", "batch": b, "input_len": il, "output_len": ol,
                           "avg_latency_s": t, "p50_latency_s": t, "synthetic": True})
    extra = ['{"engine": "vllm", "batch": 1, BROKEN LINE'] if tag == "rtx5090" else []
    wjsonl(d / "vllm_latency.jsonl", vl, extra)

    if tag == "h100_sxm":
        sd = d / "llamacpp" / "slot"
        sd.mkdir(parents=True, exist_ok=True)
        for (build, users, k, e) in (("master", 1, 1.0, 0.02), ("fix", 1, 1.0, -0.04),
                                     ("master", 8, 4.3, 0.03), ("fix", 8, 1.05, 0.01)):
            pred = users / stubs.decode_step_time(model, cost, users, 512, engine="llamacpp", passes_per_step=k)
            meas = pred / (1 + e)
            name = f"{build}_plain_u{users}"
            (sd / f"{name}.json").write_text(json.dumps({
                "part": 7, "purpose": "SYNTHETIC fixture", "policy": f"{build}_plain", "users": users,
                "passes": [{"throughput_tok_s": meas, "latency_p50_s": 1.0, "failed": 0, "requests": 10}]}))
            (sd / f"{name}.diag.txt").write_text(f"DIAG steps=1000 passes_per_step={k}\n")
            (sd / f"{name}.slots.json").write_text("[]")  # must not be mistaken for a result file


def main():
    for tag, g in GPUS.items():
        gen_gpu(tag, g)
    pc = OUT / "h100_pcie"
    pc.mkdir(parents=True, exist_ok=True)
    (pc / "env.json").write_text(json.dumps({"gpu_name": "SYNTHETIC H100 PCIe", "gpu_tag": "h100_pcie",
                                             "sm_count": 114, "l2_bytes": 50 * 2**20, "synthetic": True}))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
