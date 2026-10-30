"""Tests for correlate/, whatif/, agents/ and figures/ using SYNTHETIC fixtures and stub
predictors (tests/fixtures/synthetic_correlate/), so they pass without the real
perfsim.microarch / ops_model / engine modules."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parent / "fixtures" / "synthetic_correlate"
DATA = FIX / "data"
sys.path.insert(0, str(FIX))

import stubs  # noqa: E402

from agents import pipeline  # noqa: E402
from correlate import correlate as C  # noqa: E402
from correlate import loaders as L  # noqa: E402
from correlate.backend import Backend, attribution, default_backend  # noqa: E402
from correlate.metrics import flag_rows, mape, max_abs_pct, pct_error, pearson_r, summarize  # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    res = C.run(DATA, out, backend=stubs.make_backend())
    return res, out


# ---------------------------------------------------------------- metrics math
def test_pct_error_sign_and_zero():
    assert pct_error(110, 100) == pytest.approx(10.0)
    assert pct_error(90, 100) == pytest.approx(-10.0)
    assert math.isnan(pct_error(1, 0))
    assert math.isnan(pct_error(None, 1))


def test_mape_and_max_abs_known_values():
    pred, meas = [110, 90, 100, 150], [100, 100, 100, 100]
    assert mape(pred, meas) == pytest.approx((10 + 10 + 0 + 50) / 4)
    assert max_abs_pct(pred, meas) == pytest.approx(50.0)


def test_metrics_drop_nan_and_zero_measured():
    assert mape([1.1, float("nan"), 5, None], [1.0, 2.0, 0.0, 3.0]) == pytest.approx(10.0)


def test_pearson_matches_numpy_and_edge_cases():
    rng = np.random.default_rng(0)
    m = rng.uniform(1, 10, 20)
    p = 1.3 * m + rng.normal(0, 0.5, 20)
    assert pearson_r(p, m) == pytest.approx(np.corrcoef(p, m)[0, 1])
    assert pearson_r([2, 4, 6], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson_r([3, 2, 1], [1, 2, 3]) == pytest.approx(-1.0)
    assert math.isnan(pearson_r([1, 2], [1, 2]))          # too few points
    assert math.isnan(pearson_r([5, 5, 5], [1, 2, 3]))    # zero variance


def test_summarize_fields():
    s = summarize([2.2, 4.4, 6.6], [2, 4, 6])
    assert s["n"] == 3 and s["mape"] == pytest.approx(10.0) and s["mean_signed_pct"] == pytest.approx(10.0)
    assert s["pearson_r"] == pytest.approx(1.0)


def test_flag_threshold_is_strict_and_skips_nan():
    rows = [{"err_pct": 15.0}, {"err_pct": -15.01}, {"err_pct": 20}, {"err_pct": float("nan")}, {"err_pct": None}]
    assert [r["err_pct"] for r in flag_rows(rows, 15.0)] == [-15.01, 20]
    assert flag_rows(rows, 25.0) == []


# ---------------------------------------------------------------- loaders
def test_vllm_tpot_ttft_contract_formula():
    recs = [
        {"batch": 8, "input_len": 512, "output_len": 1, "avg_latency_s": 0.10},
        {"batch": 8, "input_len": 512, "output_len": 128, "avg_latency_s": 0.10 + 127 * 0.02},
        {"batch": 8, "input_len": 512, "output_len": 128, "avg_latency_s": 0.10 + 127 * 0.04},  # repeat -> mean
        {"batch": 4, "input_len": 512, "output_len": 64, "avg_latency_s": 1.0},  # no out=1 -> dropped
    ]
    tpot, ttft = L.derive_vllm(recs)
    assert len(tpot) == 1 and tpot[0]["tpot_s"] == pytest.approx(0.03)
    assert ttft == [{"batch": 8, "input_len": 512, "ttft_s": 0.10}]


def test_slot_loader_reads_policy_users_and_diag():
    rows = L.load_slots(DATA / "h100_sxm")
    got = {(r["build"], r["users"]): r["passes_per_step"] for r in rows}
    assert got == {("master", 1): 1.0, ("fix", 1): 1.0, ("master", 8): 4.3, ("fix", 8): 1.05}


def test_october_yaml_rows():
    rows = L.load_rtx5090_october_slots(Path(__file__).resolve().parents[1] / "data" / "llamacpp_rtx5090_qwen3_8b.yaml")
    r8 = {r["build"]: r for r in rows if r["users"] == 8}
    assert r8["master"]["tok_s"] == 333.6 and r8["master"]["passes_per_step"] == 4.34
    assert r8["fix"]["passes_per_step"] == 1.05


# ---------------------------------------------------------------- correlation plumbing
def test_outputs_written(result):
    res, out = result
    for f in ("correlation.md", "level1_gemv.csv", "level2_ops_sol.csv", "level3_vllm.csv",
              "level3_llamacpp_slots.csv", "logo_projection.csv", "correlation_metrics.csv", "errors.csv"):
        assert (out / f).exists(), f
    md = (out / "correlation.md").read_text()
    assert md.index("## Headline") < md.index("## Level 1")   # results first
    assert "Projection" in md


def test_designed_errors_recovered(result):
    res, _ = result
    own = [r for r in res["vllm"] if r["gpu"] == "h100_sxm" and r["kind"] == C.KIND_CAL and r["metric"] == "TPOT"]
    by_b = {}
    for r in own:
        by_b.setdefault(r["batch"], []).append(r["err_pct"])
    assert all(e == pytest.approx(-20.0, abs=1e-6) for e in by_b[64])
    assert all(abs(e) <= 6.0 + 1e-6 for b in (1, 8) for e in by_b[b])
    m = next(x for x in res["metrics"] if x["gpu"] == "h100_sxm" and x["quantity"] == "vLLM TPOT (ms)"
             and x["kind"] == C.KIND_CAL)
    assert m["mape"] == pytest.approx(mape([r["predicted_ms"] for r in own], [r["measured_ms"] for r in own]))


def test_level1_loaded_latency_improves_well_modelled_kernels(result):
    res, _ = result
    s = {(r["gpu"], r["kernel"]): r for r in res["level1_summary"]}
    assert s[("rtx5090", "all")]["mape_loaded"] < s[("rtx5090", "all")]["mape_plain"]
    assert s[("rtx5090", "all")]["n"] == 25            # the ok=false record was dropped
    assert s[("rtx5090", "v4")]["mape_loaded"] < 5


def test_logo_plumbing(result):
    res, out = result
    logo = [r for r in res["vllm"] if r["kind"] == C.KIND_LOGO]
    pairs = {(r["calib_from"], r["gpu"]) for r in logo}
    assert pairs == {("h100_sxm", "rtx5090"), ("rtx5090", "h100_sxm")}
    assert all(r["calib_from"] != r["gpu"] for r in logo)
    # LOGO rows never leak into the own-GPU tables
    own_csv = (out / "level3_vllm.csv").read_text()
    assert C.KIND_LOGO not in own_csv
    assert C.KIND_LOGO in (out / "logo_projection.csv").read_text()
    # the projection really used the other GPU's calibration: different numbers than own
    own = {(r["gpu"], r["metric"], r["batch"], r["input_len"]): r["predicted_ms"]
           for r in res["vllm"] if r["kind"] == C.KIND_CAL}
    for r in logo:
        assert r["predicted_ms"] != pytest.approx(own[(r["gpu"], r["metric"], r["batch"], r["input_len"])])


def test_attribution_shares_sum_to_one(result):
    res, _ = result
    for r in res["vllm"]:
        tot = sum(r[f"share_{g}"] for g in ("weights", "attention", "norms", "overhead"))
        assert tot == pytest.approx(1.0, abs=1e-6)


def test_attribution_normalises_when_engine_faster_than_ops():
    class Cost:
        def time(self, op, batch):
            return 1.0
    from perfsim.models import Op
    ops = [Op("a", 1, kind="gemm", count=2), Op("b", 1, kind="attention")]
    notes = []
    sh = attribution(Backend(), Cost(), ops, 1, total_s=1.0, notes=notes)
    assert sh["weights"] == pytest.approx(2 / 3) and sh["overhead"] == 0.0 and notes
    sh = attribution(Backend(), Cost(), ops, 1, total_s=6.0)
    assert sh["overhead"] == pytest.approx(0.5)


def test_robust_to_missing_files_and_bad_lines(result):
    res, _ = result
    assert "h100_pcie" in res["gpus"]
    notes = " ".join(res["gpu_notes"]["rtx5090"])
    assert "bad JSON line" in notes and "ops.jsonl" in notes


def test_runs_without_model_modules(tmp_path):
    """An empty backend (no microarch/ops_model/engine) must skip levels, not crash."""
    res = C.run(DATA, tmp_path, backend=Backend(name="none"))
    assert res["vllm"] == [] and res["level1"] == []
    assert res["level2"]  # Level 2 needs only hardware.yaml + ops.jsonl
    assert "skipped" in (tmp_path / "correlation.md").read_text()


def test_default_backend_never_raises():
    b = default_backend()
    assert isinstance(b, Backend)


def test_empty_data_root(tmp_path):
    res = C.run(tmp_path / "nope", tmp_path / "out", backend=stubs.make_backend())
    assert res["gpus"] == [] and (tmp_path / "out" / "correlation.md").exists()


# ---------------------------------------------------------------- agent pipeline
def test_pipeline_flags_designed_errors_and_logs(tmp_path):
    out = tmp_path / "results"
    s1 = pipeline.run_pipeline(DATA, out, backend=stubs.make_backend(), fit=False, now="T1")
    flags = s1["flags"]
    own = [f for f in flags if f["kind"] == C.KIND_CAL]
    items = {(f["gpu"], f["item"], f["config"].split(" output_len")[0]) for f in own}
    assert ("h100_sxm", "vLLM TPOT", "batch=64 input_len=512") in items
    assert ("h100_sxm", "vLLM TPOT", "batch=64 input_len=2048") in items
    assert any(f["level"] == "L1" and f["gpu"] == "h100_sxm" and "v2" in f["item"] for f in flags)
    # nothing within threshold is flagged, every flag has a hypothesis
    assert all(abs(f["err_pct"]) > 15 for f in flags)
    assert all(f["hypothesis"] for f in flags)
    tpot64 = next(f for f in own if f["item"] == "vLLM TPOT" and "batch=64 input_len=2048" in f["config"])
    assert "attention" in tpot64["hypothesis"] and "kv_len" in tpot64["hypothesis"]
    assert any("LOGO" in f["hypothesis"] for f in flags if f["kind"] == C.KIND_LOGO)
    # higher threshold -> fewer flags; log is appended, not overwritten
    s2 = pipeline.run_pipeline(DATA, out, backend=stubs.make_backend(), fit=False, threshold=25, now="T2")
    assert len(s2["flags"]) < len(flags)
    log = (out / "AGENT_LOG.md").read_text()
    assert "## T1 pipeline run" in log and "## T2 pipeline run" in log
    assert (out / "flags.csv").exists()


def test_pipeline_fit_step_calls_fit_calib_main(tmp_path):
    import shutil
    data = tmp_path / "data"
    shutil.copytree(DATA / "h100_sxm", data / "h100_sxm")
    (data / "h100_sxm" / "calib.json").unlink()
    script = tmp_path / "fit_calib.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "def main(argv):\n"
        "    d = Path(argv[argv.index('--data') + 1])\n"
        "    (d / 'calib.json').write_text(json.dumps({'mem_eff': 0.85, 'compute_eff': 0.6}))\n"
        "    return 0\n")
    st = pipeline.step_fit(data, script)
    assert st["status"] == "ok" and (data / "h100_sxm" / "calib.json").exists()
    assert pipeline.step_fit(data, tmp_path / "missing.py", Backend())["status"] == "skipped"
    assert pipeline.step_bench(False)["status"] == "skipped"


def _real_modules_present():
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("perfsim.microarch", "perfsim.ops_model", "perfsim.engine"))


@pytest.mark.skipif(not _real_modules_present(), reason="perfsim microarch/ops_model/engine not present yet")
def test_integration_with_real_modules(tmp_path):
    """End to end with the real Level 1-3 modules on a COPY of the synthetic fixtures: the
    fit step replaces the stub calib.json with a real one, correlation and what-if run."""
    import shutil
    from whatif import sweeps
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    for d in data.iterdir():
        (d / "calib.json").unlink(missing_ok=True)
    out = tmp_path / "results"
    s = pipeline.run_pipeline(data, out, now="INTEG")
    fit = next(st for st in s["steps"] if st["step"] == "fit")
    assert fit["status"] == "ok", fit
    assert (data / "h100_sxm" / "calib.json").exists()
    assert next(st for st in s["steps"] if st["step"] == "correlate")["status"] == "ok"
    md = (out / "correlation.md").read_text()
    import csv
    assert "calibrated (own GPU)" in md and "Leave-one-GPU-out **projections**" in md
    logo = list(csv.DictReader(open(out / "logo_projection.csv")))
    assert logo and {r["calib_from"] for r in logo} == {"h100_sxm"} and {r["gpu"] for r in logo} == {"rtx5090"}
    rows = list(csv.DictReader(open(out / "level3_vllm.csv")))
    assert rows and all(float(r["predicted_ms"]) > 0 for r in rows)
    l1 = list(csv.DictReader(open(out / "level1_summary.csv")))
    assert any(r["alpha_source"].startswith("stream") for r in l1)
    w = sweeps.run_whatif("h100_sxm", data, out)
    assert w["mode"] == "empirical" and w["rows"]


# ---------------------------------------------------------------- what-if
def test_whatif_projection_labels_and_directions(tmp_path):
    from whatif import sweeps
    res = sweeps.run_whatif("h100_sxm", DATA, tmp_path, backend=stubs.make_backend(), batches=(1, 8))
    md = (tmp_path / "whatif.md").read_text()
    titles = [l for l in md.splitlines() if l.startswith("### ")]
    assert titles and all(t.startswith("### Projection (not measured)") for t in titles)
    assert all(r["label"] == "Projection (not measured)" for r in res["rows"])
    get = lambda sc, v, b, m="decode tok/s": next(r for r in res["rows"] if r["scenario"] == sc and r["value"] == v
                                                  and r["batch"] == b and r["metric"] == m)
    assert get("HBM BW (ideal)", 2.0, 1)["ratio"] > 1.5          # memory-bound decode scales with BW
    assert get("HBM BW (ideal)", 0.5, 1)["ratio"] < 0.6
    assert get("HBM BW (Little's-Law cap)", 2.0, 1)["ratio"] < get("HBM BW (ideal)", 2.0, 1)["ratio"]
    assert get("FP8 weights", 0.5, 1)["ratio"] > 1.5              # weight bytes dominate batch-1 decode
    assert 1.0 < get("TP=2 (NVLink)", 2, 8)["ratio"] < 2.0        # split work, plus all-reduce cost
    assert {r["scenario"] for r in res["rows"]} >= {"baseline", "SM count"}


def test_whatif_variant_time_identity():
    from whatif.sweeps import variant_time
    from perfsim.models import Op

    class Cost:
        def time(self, op, batch):
            return op.bytes * 1e-9
    ops = [Op("w", 0, weight_bytes=1e6, count=2), Op("k", 0, kv_read_bytes=1e6)]
    t = variant_time(Backend(), Cost(), ops, 1, engine_time=0.01)
    assert t == pytest.approx(0.01)                                  # no change -> engine time
    t = variant_time(Backend(), Cost(), ops, 1, engine_time=0.01, w_scale=0.5)
    assert t == pytest.approx(0.01 - 1e-3)                           # half of 2e-3 s of weights saved


# ---------------------------------------------------------------- figures
def test_figures_headless(tmp_path, result):
    import matplotlib
    from figures import make_figures
    from whatif import sweeps
    _, out = result
    sweeps.run_whatif("h100_sxm", DATA, out, backend=stubs.make_backend())
    made = make_figures.make_all(out, tmp_path / "figs")
    assert matplotlib.get_backend().lower() == "agg"
    names = {p.name for p in made}
    assert names == {"littles_law.png", "ops_sol.png", "correlation_scatter.png", "slot_fix.png", "whatif_bw.png"}
    assert all(p.stat().st_size > 5000 for p in made)


def test_figures_skip_missing_inputs(tmp_path):
    from figures import make_figures
    assert make_figures.make_all(tmp_path / "empty", tmp_path / "figs") == []
