"""Level 3 step model (perfsim.engine) and serving-data loaders (perfsim.serving_data).

No GPU and no perfsim.ops_model needed: the cost model here is a local SOL stub
(per-op roofline from perfsim.roofline) with the OpCostModel interface. All data
come from tests/fixtures/synthetic_* (made-up numbers, not measurements).
"""
import importlib.util
import json
from pathlib import Path

import pytest

from perfsim import engine, llamacpp_slots
from perfsim import serving_data as sd
from perfsim.hardware import load_hardware
from perfsim.models import decode_ops, layer_ops, load_model, totals
from perfsim.roofline import op_time

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent
QWEN = load_model("qwen3-8b")
H100 = load_hardware()["h100_sxm"]


class SolStub:
    """Stand-in for perfsim.ops_model.OpCostModel(hw, mode="sol"): time(op) -> seconds
    for ONE instance of the op (roofline max(compute, memory))."""

    def __init__(self, hw, pass_overhead_s=0.0, precision="bf16"):
        self.hw, self.pass_overhead_s, self.precision = hw, pass_overhead_s, precision

    def time(self, op):
        return op_time(op, self.hw, self.precision).time


class BatchArgStub(SolStub):
    """Same costs, but the CONTRACTS.md signature time(op, batch)."""

    def time(self, op, batch):
        assert batch >= 1
        return op_time(op, self.hw, self.precision).time


class WeightsOnlyStub:
    """Only GEMM weight bytes cost time: isolates the 'k passes = k weight reads' effect."""
    pass_overhead_s = 0.0

    def __init__(self, bw):
        self.bw = bw

    def time(self, op):
        return op.weight_bytes / self.bw if op.kind == "gemm" else 0.0


# ---------------------------------------------------------------------------------------
# decode step
# ---------------------------------------------------------------------------------------
def test_batch1_decode_is_bandwidth_bound_bytes_over_bw_plus_overhead():
    oh = 25e-6
    t = engine.decode_step_time(QWEN, SolStub(H100, oh), 1, 512)
    nbytes = totals(decode_ops(QWEN, 1, 512, w_bits=16, kv_bits=16))["bytes"]
    assert t == pytest.approx(nbytes / H100.memory_bandwidth + oh, rel=1e-9)
    # ~15.1 GB of bf16 weights (embedding table not streamed) at 3.35 TB/s ~= 4.5 ms
    assert 4.3e-3 < t < 4.8e-3


def test_decode_step_grows_sublinearly_with_batch():
    cost = SolStub(H100, 20e-6)
    ts = {b: engine.decode_step_time(QWEN, cost, b, 512) for b in (1, 2, 4, 8, 16, 32, 64)}
    bs = sorted(ts)
    for a, b in zip(bs, bs[1:]):
        assert ts[b] > ts[a]                     # more sequences = more KV reads and FLOPs
        assert ts[b] / ts[a] < b / a             # but far less than proportional
        assert ts[b] / b < ts[a] / a             # per-token time falls
    assert ts[64] < 2 * ts[1]                    # weights dominate even at 64 on H100 (bf16)


def test_context_adds_kv_reads():
    cost = SolStub(H100)
    t0 = engine.decode_step_time(QWEN, cost, 8, 0)
    t1 = engine.decode_step_time(QWEN, cost, 8, 4096)
    kv = 8 * 4096 * QWEN.kv_bytes_per_token(16)  # extra K/V bytes read
    assert t1 - t0 == pytest.approx(kv / H100.memory_bandwidth, rel=0.02)


def test_cost_signature_with_batch_and_dict_overhead():
    a = engine.decode_step_time(QWEN, SolStub(H100, 0.0), 4, 256, overhead_s=1e-4)
    stub = BatchArgStub(H100)
    stub.pass_overhead_s = {"vllm": 1e-4, "llamacpp": 0.0}
    assert engine.decode_step_time(QWEN, stub, 4, 256) == pytest.approx(a)


def test_time_includes_count_flag():
    class Totals(SolStub):
        time_includes_count = True

        def time(self, op):
            return op_time(op, self.hw, self.precision).total_time

    assert engine.decode_step_time(QWEN, Totals(H100), 2, 100) == \
        pytest.approx(engine.decode_step_time(QWEN, SolStub(H100), 2, 100))


def test_breakdown_sums_to_total():
    br = engine.decode_step_breakdown(QWEN, SolStub(H100, 3e-5), 8, 1024, passes_per_step=2)
    assert sum(br["by_op"].values()) + br["overhead_s"] == pytest.approx(br["total_s"])
    assert br["overhead_s"] == pytest.approx(2 * 3e-5)
    assert max(br["by_op"], key=br["by_op"].get) == "mlp_gate_up"  # largest weight matrix


# ---------------------------------------------------------------------------------------
# llama.cpp passes per step (slot bug)
# ---------------------------------------------------------------------------------------
def test_integer_passes_equal_k_times_a_plus_B_times_c():
    cost = SolStub(load_hardware()["rtx5090"], 40e-6, precision="fp16")
    a, c = engine.slot_coeffs(QWEN, cost, 512)
    for k in (1, 2, 4, 8):
        t = engine.decode_step_time(QWEN, cost, 8, 512, "llamacpp", k)
        assert t == pytest.approx(k * a + 8 * c, rel=1e-6)


def test_fractional_passes_interpolate():
    cost = SolStub(H100, 10e-6)
    t2 = engine.decode_step_time(QWEN, cost, 8, 512, "llamacpp", 2)
    t3 = engine.decode_step_time(QWEN, cost, 8, 512, "llamacpp", 3)
    t25 = engine.decode_step_time(QWEN, cost, 8, 512, "llamacpp", 2.5)
    assert t25 == pytest.approx((t2 + t3) / 2)
    # k is clamped to [1, batch]
    assert engine.decode_step_time(QWEN, cost, 2, 512, "llamacpp", 9) == \
        pytest.approx(engine.decode_step_time(QWEN, cost, 2, 512, "llamacpp", 2))


def test_master_vs_fix_speedup_weights_only_equals_pass_ratio():
    r = engine.predict_llamacpp_master_vs_fix(QWEN, WeightsOnlyStub(1.792e12), users=8,
                                              k_master=4.0, k_fix=1.0)
    assert r["speedup"] == pytest.approx(4.0)


def test_master_vs_fix_speedup_from_measured_pass_counts():
    cost = SolStub(load_hardware()["rtx5090"], 0.0, precision="fp16")
    r = engine.predict_llamacpp_master_vs_fix(QWEN, cost, users=8)  # k = 4.34 vs 1.05 (yaml counters)
    ratio = 4.34 / 1.05
    # per-sequence costs (KV, activations) dilute the pure pass ratio, but only a little at 8 users
    assert 0.8 * ratio < r["speedup"] < ratio
    # same answer as llamacpp_slots' t = k*a + B*c with (a, c) from this cost model
    assert r["slots_formula_speedup"] == pytest.approx(r["speedup"], rel=1e-6)
    # and the per-pass weight time is the GGUF bytes at spec bandwidth
    assert r["a_s"] == pytest.approx(llamacpp_slots.weight_bytes_per_pass() / 1.792e12, rel=0.02)
    # a fixed per-pass overhead makes extra passes even more expensive -> larger speedup
    r_oh = engine.predict_llamacpp_master_vs_fix(QWEN, SolStub(load_hardware()["rtx5090"], 2e-4, "fp16"), users=8)
    assert r_oh["speedup"] > r["speedup"]


def test_slot_tps_matches_llamacpp_slots_formula():
    cost = SolStub(load_hardware()["rtx5090"], 5e-5, precision="fp16")
    a, c = engine.slot_coeffs(QWEN, cost)
    for users, k in ((1, 1.0), (4, 2.0), (8, 4.34), (16, 1.0)):
        assert engine.predict_llamacpp_slot_tps(QWEN, cost, users, k) == \
            pytest.approx(llamacpp_slots.predict_throughput(a, c, users, k), rel=1e-6)


# ---------------------------------------------------------------------------------------
# prefill and vLLM latency
# ---------------------------------------------------------------------------------------
def test_prefill_passes_chunking():
    assert engine.prefill_passes(64, 512) == [(64, 0, 512)]
    assert engine.prefill_passes(64, 512, 16384) == [(32, 0, 512), (32, 0, 512)]
    assert engine.prefill_passes(1, 8192, 16384) == [(1, 0, 8192)]
    assert engine.prefill_passes(1, 10000, 4096) == [(1, 0, 4096), (1, 4096, 4096), (1, 8192, 1808)]


def test_prefill_compute_bound_and_chunks_cost_weight_rereads():
    cost = SolStub(H100)
    t1 = engine.prefill_time(QWEN, cost, 1, 8192)
    flops = totals(layer_ops(QWEN, 1, 0, 8192))["flops"]
    assert t1 >= flops / H100.peak("bf16")                       # cannot beat peak FLOP/s
    t_chunk = engine.prefill_time(QWEN, cost, 64, 512, chunk_tokens=16384)
    t_one = engine.prefill_time(QWEN, cost, 64, 512)
    assert t_chunk >= t_one


def test_vllm_latency_is_prefill_plus_growing_decode_steps():
    cost = SolStub(H100, 30e-6)
    B, S = 4, 512
    assert engine.predict_vllm_latency(QWEN, cost, B, S, 1) == pytest.approx(engine.prefill_time(QWEN, cost, B, S))
    lat = engine.predict_vllm_latency(QWEN, cost, B, S, 4)
    steps = sum(engine.decode_step_time(QWEN, cost, B, S + i) for i in range(3))
    assert lat == pytest.approx(engine.prefill_time(QWEN, cost, B, S) + steps)
    # model TPOT = mean decode step over the generated contexts; between first and last step
    tpot = engine.predict_vllm_tpot(QWEN, cost, B, S, 128)
    assert engine.decode_step_time(QWEN, cost, B, S) < tpot < engine.decode_step_time(QWEN, cost, B, S + 126)


def test_bad_engine_rejected():
    with pytest.raises(ValueError):
        engine.decode_step_time(QWEN, SolStub(H100), 1, 0, engine="tgi")


# ---------------------------------------------------------------------------------------
# loaders (synthetic fixtures)
# ---------------------------------------------------------------------------------------
def test_vllm_loader_and_tpot():
    recs = sd.load_vllm_latency(FIX / "synthetic_vllm_latency.jsonl")
    assert len(recs) == 6                          # garbage + incomplete lines skipped
    tp = {r["batch"]: r for r in sd.tpot_records(recs)}
    assert set(tp) == {1, 8}                       # batch 16 has no out=1 record
    assert tp[1]["tpot_s"] == pytest.approx(0.010)
    assert tp[8]["tpot_s"] == pytest.approx(0.012)
    assert tp[8]["decode_tok_s"] == pytest.approx(8 / 0.012)
    assert tp[1]["output_len"] == 128
    p50 = {r["batch"]: r for r in sd.tpot_records(recs, stat="p50")}
    assert p50[1]["tpot_s"] == pytest.approx(0.011)


def test_ttft_records():
    ttft = sd.ttft_records(sd.load_vllm_latency(FIX / "synthetic_vllm_latency.jsonl"))
    got = {(r["batch"], r["input_len"]): r["ttft_s"] for r in ttft}
    assert got == {(1, 512): 0.05, (1, 2048): 0.2, (8, 512): 0.08}


def test_vllm_duplicate_records_last_wins(tmp_path):
    p = tmp_path / "synthetic_dup.jsonl"
    rows = [{"batch": 1, "input_len": 512, "output_len": 1, "avg_latency_s": 1.0},
            {"batch": 1, "input_len": 512, "output_len": 1, "avg_latency_s": 2.0}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert sd.load_vllm_latency(p)[0]["avg_latency_s"] == 2.0


def test_llamacpp_slot_loader():
    rows = sd.load_llamacpp_slots(FIX / "synthetic_llamacpp")
    by = {(r["build"], r["kv_unified"], r["users"]): r for r in rows}
    assert len(rows) == 4                          # *.slots.json ignored
    m8 = by[("master", False, 8)]
    assert m8["tok_s"] == 310.0                    # median of 3 passes
    assert m8["passes_per_step"] == 4.30           # LAST DIAG line (cumulative)
    assert m8["decode_calls"] == 200
    assert by[("fix", False, 8)]["passes_per_step"] == 1.05
    assert by[("fix", False, 1)]["passes_per_step"] is None   # no diag file -> unknown, not 1
    assert by[("master", True, 8)]["policy"] == "master_kvu"


def test_llamacpp_bench_loaders():
    bb = sd.load_llamacpp_batched(FIX / "synthetic_llamacpp")
    assert [r["users"] for r in bb] == [1, 8]
    assert bb[1]["step_s"] == pytest.approx(8 / 1000.0)
    lb = {r["test"]: r["tok_s"] for r in sd.load_llama_bench(FIX / "synthetic_llamacpp")}
    assert lb == {"pp512": 9000.0, "tg128": 200.0}
    assert sd.load_llamacpp_slots(FIX / "does_not_exist") == []


# ---------------------------------------------------------------------------------------
# bench/vllm_grid.py (no vLLM needed: parser and grid only)
# ---------------------------------------------------------------------------------------
def _grid_module():
    spec = importlib.util.spec_from_file_location("vllm_grid", REPO / "bench" / "vllm_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vllm_json_parser_current_format():
    g = _grid_module()
    # key layout of vllm/benchmarks/latency.py (v0.32.0); values synthetic
    obj = {"model_id": "x", "avg_latency": 1.5, "latencies": [1.0, 2.0],
           "percentiles": {"10": 1.1, "25": 1.25, "50": 1.5, "75": 1.75, "90": 1.9, "99": 1.99}}
    r = g.parse_latency_json(obj)
    assert r["avg_latency_s"] == 1.5 and r["p50_latency_s"] == 1.5 and r["p99_latency_s"] == 1.99


def test_vllm_json_parser_fallbacks():
    g = _grid_module()
    r = g.parse_latency_json({"latencies": [1.0, 2.0, 3.0, 4.0]})
    assert r["avg_latency_s"] == 2.5
    assert r["p50_latency_s"] == pytest.approx(2.5)        # numpy-style linear interpolation
    assert r["p90_latency_s"] == pytest.approx(3.7)
    assert g.parse_latency_json({})["avg_latency_s"] is None


def test_vllm_grid_points():
    pts = {(p["batch"], p["input_len"], p["output_len"]) for p in _grid_module().build_grid()}
    assert len(pts) == 7 * 2 + 4 - 1               # (1, 512, 1) is in both parts
    assert (64, 512, 128) in pts and (1, 8192, 1) in pts
