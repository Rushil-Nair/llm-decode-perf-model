import pytest

from perfsim import llamacpp_slots as s


def test_more_passes_is_slower():
    a, c = 4e-3, 0.5e-3
    assert s.predict_throughput(a, c, 8, 4.0) < s.predict_throughput(a, c, 8, 1.0)


def test_pure_weight_bound_ratio_equals_pass_ratio():
    a = 4e-3
    r = s.predict_throughput(a, 0.0, 8, 1.0) / s.predict_throughput(a, 0.0, 8, 4.0)
    assert r == pytest.approx(4.0)


def test_implied_passes_inverts_prediction():
    a, c = 3e-3, 0.4e-3
    thr = s.predict_throughput(a, c, 8, 2.7)
    assert s.implied_passes(a, c, 8, thr) == pytest.approx(2.7)


def test_calibration_reproduces_fit_points():
    meas = s.load_measurements()
    a, c = s.calibrated_coeffs(meas)
    assert a > 0 and c > 0
    assert s.predict_throughput(a, c, 1, 1.0) == pytest.approx(meas["throughput_tok_s"][1][1])
    assert s.predict_throughput(a, c, 16, 1.0) == pytest.approx(meas["throughput_tok_s"][16][1])


def test_measured_never_exceeds_ideal_bound():
    r = s.analyse()
    for row in r["rows"]:
        assert row["measured_tok_s"] < row["ideal_roofline_tok_s"]
    assert 0 < r["one_user_roofline"]["weights_excl_embedding"]["fraction"] < 1


def test_measured_speedup_value():
    r = s.analyse()
    assert r["speedup_8_users"]["measured"] == pytest.approx(878.9 / 333.6)
