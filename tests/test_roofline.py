import math

import pytest

from perfsim.hardware import Hardware, load_hardware
from perfsim.models import ModelConfig, Op
from perfsim.roofline import (allreduce_time, attainable_flops, comm_time, decode, devices_needed,
                                op_time, shard_plan, step)

HW = Hardware(key="toy", name="toy", kind="gpu", memory_capacity=1e9, memory_bandwidth=1e12,
              peak_flops={"fp16": 100e12}, interconnect_bandwidth=1e11, hop_latency=1e-6)
TOY = ModelConfig(name="toy", n_layers=2, hidden=8, n_heads=2, n_kv_heads=1, head_dim=4,
                  intermediate=16, vocab=32)


def test_attainable_is_min_of_peak_and_bandwidth_slope():
    assert attainable_flops(10, 100e12, 1e12) == 10e12  # below ridge (100)
    assert attainable_flops(1000, 100e12, 1e12) == 100e12  # above ridge


def test_memory_bound_op_time():
    op = Op("x", flops=1e9, weight_bytes=1e9)  # AI = 1 < ridge 100
    t = op_time(op, HW, "fp16")
    assert t.bound == "memory"
    assert t.time == pytest.approx(1e9 / 1e12)


def test_compute_bound_op_time():
    op = Op("x", flops=1e12, weight_bytes=1e6)  # AI = 1e6 > ridge
    t = op_time(op, HW, "fp16")
    assert t.bound == "compute"
    assert t.time == pytest.approx(1e12 / 100e12)


def test_efficiency_factor_scales_time():
    op = Op("x", flops=1e9, weight_bytes=1e9)
    assert op_time(op, HW, "fp16", eff_memory=0.5).time == pytest.approx(2 * op_time(op, HW, "fp16").time)


def test_decode_throughput_increases_then_saturates():
    r1 = decode(TOY, HW, 1, 16)
    r8 = decode(TOY, HW, 8, 16)
    assert r8["throughput"] > r1["throughput"]
    assert r1["throughput"] == pytest.approx(1 / r1["time"])


def test_no_comm_single_device_and_comm_grows_with_devices():
    assert comm_time(TOY, HW, 1, 1, "tp") == 0
    assert comm_time(TOY, HW, 1, 8, "tp") > comm_time(TOY, HW, 1, 2, "tp")


def test_allreduce_alpha_beta():
    # P=4: 2*log2(4)=4 latency steps + 2*(3/4)*msg/bw
    assert allreduce_time(1e6, 4, 1e9, 1e-6) == pytest.approx(4e-6 + 1.5e-3)


def test_shard_plan():
    assert shard_plan(TOY, 8, "tp") == (1, 8)
    assert shard_plan(TOY, 8, "pp") == (2, 4)  # 2 layers -> 2 stages x TP4


def test_devices_needed_ceil():
    big = ModelConfig(name="b", n_layers=1, hidden=1000, n_heads=10, n_kv_heads=10, head_dim=100,
                      intermediate=1000, vocab=10)
    need = big.weight_bytes(16)
    hw = Hardware(key="h", name="h", kind="gpu", memory_capacity=need / 2.5, memory_bandwidth=1,
                  peak_flops={"fp16": 1})
    assert devices_needed(big, hw, 1, 0, 16, 16) == 3


def test_weight_override():
    r = step(TOY, HW, 1, 0, 1, weight_bytes_override=12345.0)
    # override covers GEMM weights; plus 1 embedding row (8*2 B) and 5 RMSNorm weight vectors (8*2 B each)
    assert r["weight_bytes_read"] == pytest.approx(12345.0 + 8 * 2 + 5 * 8 * 2)


def test_hardware_yaml_loads_and_is_numeric():
    hw = load_hardware()
    assert hw["rtx5090"].memory_bandwidth == 1792e9
    assert hw["h100_sxm"].peak("fp16") == pytest.approx(989.5e12)
    assert isinstance(hw["groq_lpu_v1"].interconnect_bandwidth, float)
    for h in hw.values():
        assert all(isinstance(v, float) for v in h.peak_flops.values())
