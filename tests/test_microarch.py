"""Tests for perfsim.microarch (Little's Law + loaded-latency model) and microarch/merge_ncu.py.

All data here is SYNTHETIC (tests/fixtures/synthetic_*), generated from known parameters by
tests/fixtures/synthetic_generate.py, so we can check the model recovers those parameters.
"""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from perfsim.microarch import (GEMV_BYTES_PER_GROUP, GPUParams, bytes_in_flight, demand, dram_latency_from_chase,
                               fit_loaded_latency, load_jsonl, loaded_latency_ns, loaded_utilization,
                               loaded_utilization_iterative, mape, params_from_probes, predict_bw,
                               prediction_table, summarize)

FIX = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_module("synthetic_generate", FIX / "synthetic_generate.py")
merge_ncu = _load_module("merge_ncu", REPO / "microarch" / "merge_ncu.py")

P = gen.true_params()
ALPHA = gen.TRUE["alpha"]


# ---------------------------------------------------------------- bytes in flight / Little's Law

def test_bytes_in_flight_arithmetic():
    # 64 warps x 32 lanes x 2 loads x 16 B = 64 KiB per SM; x 132 SMs
    assert bytes_in_flight(64, 2, 16, 1) == 64 * 32 * 2 * 16
    assert bytes_in_flight(64, 2, 16, 132) == 64 * 32 * 2 * 16 * 132
    assert np.allclose(bytes_in_flight(np.array([8, 16]), 1, 16, 2), [8 * 512 * 2, 16 * 512 * 2])


def test_unloaded_model_is_min_of_peak_and_littles_law():
    knee = P.latency_bandwidth_product                 # bytes in flight where BIF / L0 == peak
    L0 = P.dram_latency_ns * 1e-9
    # below the knee: linear, BIF / L0
    assert predict_bw(P, 0.25 * knee) == pytest.approx(0.25 * knee / L0)
    assert predict_bw(P, 0.25 * knee) == pytest.approx(0.25 * P.peak_bw_Bps)
    # at and above the knee: saturated at peak
    assert predict_bw(P, knee) == pytest.approx(P.peak_bw_Bps)
    assert predict_bw(P, 10 * knee) == pytest.approx(P.peak_bw_Bps)
    # vectorised
    bw = predict_bw(P, np.array([0.5, 2.0]) * knee)
    assert np.allclose(bw, [0.5 * P.peak_bw_Bps, P.peak_bw_Bps])


def test_loaded_model_with_alpha_zero_equals_unloaded():
    bif = np.geomspace(1e4, 1e9, 50)
    assert np.allclose(predict_bw(P, bif, loaded_latency=True, alpha=0.0), predict_bw(P, bif))


def test_loaded_model_rounds_the_corner_and_saturates():
    bif = np.geomspace(1e4, 1e10, 200)
    bw = predict_bw(P, bif, loaded_latency=True, alpha=ALPHA)
    unl = predict_bw(P, bif)
    assert np.all(np.diff(bw) > 0)                     # more bytes in flight -> more bandwidth
    assert np.all(bw < P.peak_bw_Bps)                  # never reaches peak (latency -> inf as u -> 1)
    assert np.all(bw <= unl + 1e-6)                    # queueing can only cost bandwidth
    assert bw[-1] / P.peak_bw_Bps > 0.99               # ... but approaches peak with huge BIF
    # biggest gap is at the knee of the unloaded model
    knee_bw = predict_bw(P, P.latency_bandwidth_product, loaded_latency=True, alpha=ALPHA)
    assert knee_bw < 0.8 * P.peak_bw_Bps


def test_loaded_prediction_satisfies_littles_law_with_loaded_latency():
    # self-consistency: predicted BW == BIF / L(u) with u = BW / peak
    for bif in (1e5, 1e6, P.latency_bandwidth_product, 1e8):
        bw = predict_bw(P, bif, loaded_latency=True, alpha=ALPHA)
        L = loaded_latency_ns(P, bw / P.peak_bw_Bps, ALPHA) * 1e-9
        assert bw == pytest.approx(bif / L, rel=1e-9)


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.4, 1.0, 3.0])
def test_closed_form_matches_fixed_point_iteration(alpha):
    for D in (0.0, 0.01, 0.3, 0.9, 1.0, 1.5, 5.0, 100.0):
        assert float(loaded_utilization(D, alpha)) == pytest.approx(loaded_utilization_iterative(D, alpha), abs=1e-9)


def test_loaded_needs_alpha():
    with pytest.raises(ValueError):
        predict_bw(P, 1e6, loaded_latency=True)


# ---------------------------------------------------------------- fitting alpha

def test_fit_recovers_alpha_exactly_from_noiseless_stream():
    recs = gen.stream_records(noise=0.0)
    assert fit_loaded_latency(P, recs) == pytest.approx(ALPHA, rel=1e-3)


def test_fit_recovers_alpha_exactly_from_noiseless_gemv():
    recs = gen.gemv_records(noise=0.0)
    assert fit_loaded_latency(P, recs) == pytest.approx(ALPHA, rel=1e-3)


@pytest.mark.parametrize("alpha", [0.05, 1.5])
def test_fit_recovers_other_alphas(alpha, monkeypatch):
    monkeypatch.setitem(gen.TRUE, "alpha", alpha)
    assert fit_loaded_latency(P, gen.stream_records(noise=0.0)) == pytest.approx(alpha, rel=1e-3)


def test_fit_on_noisy_fixture_and_mape_improves():
    recs = [r for r in load_jsonl(FIX / "synthetic_probes.jsonl") if r["probe"] == "stream"]
    alpha = fit_loaded_latency(P, recs)
    assert alpha == pytest.approx(ALPHA, rel=0.2)
    m = mape(prediction_table(P, recs, alpha))
    assert m["n"] == len(recs)
    assert m["mape_loaded_pct"] < 5.0                  # ~ the 3% noise
    assert m["mape_loaded_pct"] < m["mape_littles_pct"]


def test_fit_rejects_too_few_records():
    with pytest.raises(ValueError):
        fit_loaded_latency(P, [])


def test_failed_or_untimed_gemv_records_are_ignored():
    recs = gen.gemv_records(noise=0.0)
    bad = [dict(recs[0], ok=False, GBps=1.0), dict(recs[1], us=0.0, GBps=0.0, mode="ncu")]
    assert fit_loaded_latency(P, recs + bad) == pytest.approx(ALPHA, rel=1e-3)
    assert len(prediction_table(P, bad, ALPHA)) == 0


# ---------------------------------------------------------------- loading data/<tag>/ and GPUParams

def test_fixtures_are_reproducible(tmp_path):
    gen.write_all(tmp_path)
    for name in ("synthetic_probes.jsonl", "synthetic_gemv_sweep.jsonl"):
        assert (tmp_path / name).read_text() == (FIX / name).read_text(), f"regenerate {name}"


def test_fixtures_are_marked_synthetic():
    for name in ("synthetic_probes.jsonl", "synthetic_gemv_sweep.jsonl"):
        recs = load_jsonl(FIX / name)
        assert recs and all(r.get("synthetic") is True and r.get("gpu") == "SYNTHETIC" for r in recs)


def test_dram_latency_uses_first_plateau_beyond_l2():
    chase = gen.pointer_chase_records()
    L0 = dram_latency_from_chase(chase, P.l2_bytes)
    # residual L2 hits (prob L2/footprint) pull it slightly below the true 650 ns, but not to the L2 plateau
    assert gen.L2_NS < L0 < P.dram_latency_ns
    assert L0 == pytest.approx(P.dram_latency_ns, rel=0.1)
    # fallback: when no footprint reaches 8x L2, the largest one is used
    assert dram_latency_from_chase(chase, 10 ** 12) == chase[-1]["latency_ns"]


def test_params_and_summary_from_data_dir(tmp_path):
    shutil.copy(FIX / "synthetic_probes.jsonl", tmp_path / "probes.jsonl")
    shutil.copy(FIX / "synthetic_gemv_sweep.jsonl", tmp_path / "gemv_sweep.jsonl")
    with open(tmp_path / "probes.jsonl", "a") as f:
        f.write('{"probe": "stream", "blocks_per_')         # a truncated last line must be skipped
    p = params_from_probes(load_jsonl(tmp_path / "probes.jsonl"))
    assert p.sm_count == 132 and p.l2_bytes == 50_000_000
    assert p.peak_bw_Bps == pytest.approx(max(r["GBps"] for r in load_jsonl(FIX / "synthetic_probes.jsonl")
                                              if r["probe"] == "stream") * 1e9)
    s = summarize(tmp_path)
    assert s["alpha_stream"] > 0 and s["alpha_gemv"] > 0
    # out-of-sample: alpha fitted on the stream probe predicts the GEMV sweep
    m = mape(s["gemv_table_stream_alpha"])
    assert m["n"] == len(load_jsonl(FIX / "synthetic_gemv_sweep.jsonl"))
    assert m["mape_loaded_pct"] < m["mape_littles_pct"]


def test_gemv_bytes_per_group_includes_scale():
    assert GEMV_BYTES_PER_GROUP == 16 + 2


# ---------------------------------------------------------------- merge_ncu

NCU_CSV = '''==PROF== Connected to process 1234 (/x/gemv_sweep)
{"kernel":"v2","ilp":1}
==PROF== Disconnected from process 1234
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","dram__bytes_read.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed","gpu__time_duration.sum","launch__occupancy_limit_shared_mem","launch__registers_per_thread","sm__warps_active.avg.pct_of_peak_sustained_active"
"","","","","","","","","","","","byte","%","nsecond","block","register/thread","%"
"0","1234","gemv_sweep","h","void gemv_v2<1>(const unsigned char *, const __half *, const __half *, float *, int, int)","1","7","(256, 1, 1)","(512, 1, 1)","0","9.0","9,437,184","61.5","12,345","1","32","12.4"
"1","1234","gemv_sweep","h","void gemv_v3<4>(const unsigned char *, const __half *, const __half *, float *, int, int)","1","7","(256, 1, 1)","(132, 1, 1)","0","9.0","9437184","88.1","8000","2","42","24.9"
'''


def _cfg(kernel, ilp, launch, mode="ncu", bps=1):
    return {"kernel": kernel, "ilp": ilp, "N": 4096, "K": 4096, "blocks_per_sm": bps, "requested_blocks_per_sm": bps,
            "threads_per_block": 256, "mode": mode, "ncu_launch": launch, "us": 0.0, "GBps": 0.0, "ok": True}


def test_parse_ncu_csv_units_and_separators():
    rows = merge_ncu.parse_ncu_csv(NCU_CSV)
    assert [r["id"] for r in rows] == [0, 1]
    assert rows[0]["ncu_time_us"] == pytest.approx(12.345)        # nsecond -> us, "12,345" separators
    assert rows[0]["ncu_dram_bytes"] == 9437184
    assert rows[1]["ncu_dram_pct"] == pytest.approx(88.1)
    assert rows[1]["ncu_warps_active_pct"] == pytest.approx(24.9)
    assert rows[1]["ncu_registers_per_thread"] == 42
    assert rows[1]["ncu_occupancy_limit_shared_mem"] == 2


def test_merge_by_launch_order_then_config_key(tmp_path):
    configs = [_cfg("v2", 1, 0), _cfg("v3", 4, 1)]
    timed = [dict(_cfg("v3", 4, -1, "timed"), us=10.0, GBps=900.0),
             dict(_cfg("v2", 1, -1, "timed"), us=12.0, GBps=800.0),
             dict(_cfg("v2", 1, -1, "timed", bps=2), us=11.0, GBps=850.0)]   # no ncu row for this one
    (tmp_path / "sweep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in timed))
    (tmp_path / "cfg.jsonl").write_text("".join(json.dumps(r) + "\n" for r in configs))
    (tmp_path / "ncu.csv").write_text(NCU_CSV)
    rc = merge_ncu.main(["--sweep", str(tmp_path / "sweep.jsonl"), "--configs", str(tmp_path / "cfg.jsonl"),
                         "--csv", str(tmp_path / "missing.csv"), str(tmp_path / "ncu.csv"),
                         "--out-ncu", str(tmp_path / "ncu.jsonl")])
    assert rc == 0
    out = load_jsonl(tmp_path / "sweep.jsonl")
    assert out[0]["kernel"] == "v3" and out[0]["ncu_dram_pct"] == pytest.approx(88.1)
    assert out[1]["kernel"] == "v2" and out[1]["ncu_time_us"] == pytest.approx(12.345)
    assert "ncu_time_us" not in out[2]
    assert out[0]["GBps"] == 900.0                                 # timed fields untouched
    assert len(load_jsonl(tmp_path / "ncu.jsonl")) == 2


def test_merge_refuses_misaligned_order():
    rows = merge_ncu.parse_ncu_csv(NCU_CSV)
    with pytest.raises(RuntimeError):
        merge_ncu.merge([], [_cfg("v3", 4, 0), _cfg("v2", 1, 1)], rows)
