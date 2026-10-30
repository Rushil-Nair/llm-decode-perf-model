"""Tests for perfsim.gemv_model (three-ceiling GEMV model) and perfsim.slot_pass_model.

SYNTHETIC data only: records are generated from the model itself with known cpg / cpw / t0, so the fit must
recover them and the prediction error must be ~0.
"""
import json
import math

import numpy as np
import pytest

from perfsim import gemv_model as G
from perfsim import slot_pass_model as S
from perfsim.microarch import GPUParams

TRUE_CPG = {("v3", 1): 3.0, ("v3", 4): 2.8}
TRUE_CPW = {("v3", 1): 48.0, ("v3", 4): 40.0}
TRUE_T0 = 3e-6


def _gpu(sm=132, f=1.98e9, peak=3.1e12, lat=275.0):
    g = object.__new__(G.GPU)
    g.tag, g.params, g.alpha, g.f_sm = "syn", GPUParams(sm, lat, peak, 50 << 20, "syn"), 0.6, f
    g.records = []
    for N, K in [(4096, 4096), (12288, 4096), (4096, 12288), (8192, 8192)]:
        for k in TRUE_CPG:
            for w in (4, 8, 16, 32, 64):
                r = {"kernel": k[0], "ilp": k[1], "N": N, "K": K, "bytes": N * K // 32 * 18 + 2 * (N + K),
                     "blocks_per_sm": w // 8 or 1, "warps_per_sm": w, "shape": f"{N}x{K}"}
                c = G.ceilings(g, r, TRUE_CPG, TRUE_CPW)
                t = TRUE_T0 + r["bytes"] / min(c.values())
                r.update(us=t * 1e6, GBps=r["bytes"] / t / 1e9)
                g.records.append(r)
    return g


def test_bytes_per_sm_cycle():
    assert _gpu().bytes_per_sm_cycle == pytest.approx(3.1e12 / (132 * 1.98e9))


def test_ceilings_scale_with_clock_and_warps():
    g = _gpu()
    r = dict(g.records[0])
    c1 = G.ceilings(g, r, TRUE_CPG, TRUE_CPW)
    r2 = dict(r, warps_per_sm=2 * r["warps_per_sm"])
    c2 = G.ceilings(g, r2, TRUE_CPG, TRUE_CPW)
    assert c2["warp"] == pytest.approx(2 * c1["warp"])
    assert c2["sm"] == pytest.approx(c1["sm"])


def test_fit_recovers_sm_cost_and_predicts_own_data():
    g = _gpu()
    cpg, cpw, t0 = G.fit(g)
    for k in TRUE_CPG:  # best large-shape config is SM-bound with launch cost included: slightly above truth
        assert TRUE_CPG[k] <= cpg[k] < TRUE_CPG[k] * 1.25
    rows = G.predict(g, (cpg, cpw, t0))
    assert G.summary(rows)["mape"] < 12


def test_projection_to_memory_bound_gpu():
    a = _gpu()
    b = _gpu(sm=170, f=2.89e9, peak=1.7e12, lat=129.0)  # less bandwidth per SM-cycle: memory ceiling binds more
    rows = G.predict(b, G.fit(a))
    s = G.summary(rows)
    assert s["bound"]["mem"] > 0
    assert all(math.isfinite(r["predicted_GBps"]) for r in rows)


def test_busy_clock(tmp_path):
    (tmp_path / "clocks.csv").write_text(
        "ts, sm, mem, temp, power, util\n"
        "t, 345 MHz, 1593 MHz, 30, 70 W, 0 %\n"
        "t, 1980 MHz, 2619 MHz, 50, 500 W, 100 %\n"
        "t, 1970 MHz, 2619 MHz, 50, 500 W, 90 %\n"
        "t, 1990 MHz, 2619 MHz, 50, 500 W, 99 %\n")
    assert G.busy_sm_clock_hz(tmp_path) == pytest.approx(1.98e9)


# ------------------------------------------------------------------- slot pass model
def _write_llamacpp(d, bb, slots):
    (d / "llamacpp" / "bench").mkdir(parents=True)
    (d / "llamacpp" / "slot").mkdir(parents=True)
    with open(d / "llamacpp" / "bench" / "batched_bench.jsonl", "w") as f:
        for B, tok in bb.items():
            f.write(json.dumps({"pl": B, "speed_tg": tok}) + "\n")
    for (policy, u), (tok, k) in slots.items():
        (d / "llamacpp" / "slot" / f"{policy}_u{u}.json").write_text(
            json.dumps({"passes": [{"throughput_tok_s": tok}]}))
        (d / "llamacpp" / "slot" / f"{policy}_u{u}.diag.txt").write_text(
            f"DIAG gen_decode_calls=100 gen_steps=100 passes_per_step={k:.2f}\n")


def test_slot_pass_model_exact_on_synthetic(tmp_path):
    a, c, s = 4e-3, 0.25e-3, 0.5e-3                      # t_pass(B) = a + c*B, server cost s
    bb = {B: B / (a + c * B) for B in (1, 2, 4, 8, 16)}
    tok = lambda u, k: u / (k * (a + c * u / k) + s)
    slots = {("master_plain", 1): (tok(1, 1), 1.0), ("fix_plain", 1): (tok(1, 1), 1.0),
             ("master_plain", 8): (tok(8, 4), 4.0), ("fix_plain", 8): (tok(8, 1), 1.0)}
    _write_llamacpp(tmp_path, bb, slots)
    r = S.analyse(tmp_path)
    assert r["s_ms"] == pytest.approx(s * 1e3, rel=1e-6)
    for x in r["rows"]:
        assert x["err_pct"] == pytest.approx(0, abs=1e-6)
    (sp,) = r["speedup"]
    assert sp["predicted"] == pytest.approx(sp["measured"], rel=1e-6)


# ------------------------------------------------------------------- v6/v7 shared-memory swizzle
def _swz(c):  # same formula as swz() in kernels/gemv_sweep.cu
    return (c & ~7) | ((c & 7) ^ ((c >> 3) & 3))


def test_swizzle_is_a_permutation_within_each_128_byte_line():
    for line in range(64):
        cs = range(8 * line, 8 * line + 8)
        assert sorted(_swz(c) for c in cs) == list(cs)


def test_swizzle_removes_quarter_warp_bank_conflicts():
    # lane l reads chunk i of group g = l + 32 j; a quarter-warp (8 lanes x 16 B) must hit 8 distinct
    # 16-byte slots of a 128-byte bank line. Unswizzled it hits only 2 (a 4-way conflict).
    for j in range(8):
        for i in range(4):
            for q in range(4):
                lanes = range(8 * q, 8 * q + 8)
                plain = {((l + 32 * j) * 4 + i) & 7 for l in lanes}
                swz = {_swz((l + 32 * j) * 4 + i) & 7 for l in lanes}
                assert len(plain) == 2 and len(swz) == 8
