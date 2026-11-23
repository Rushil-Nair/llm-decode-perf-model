"""The attention-term fit recovers known parameters from synthetic rows (test data only, never reported)."""
from perfsim import attn_model as am


def _rows(kernel, t0_us, gbps, configs=("llama8b", "mha", "qwen7b", "tp8_70b")):
    rows = []
    for cfg in configs:
        for b in (1, 4, 16):
            for ctx in (512, 2048, 8192):
                nbytes = 2 * b * ctx * 8 * 128 * 2
                rows.append(dict(config=cfg, batch=b, ctx=ctx, skew=False, kernel=kernel, bytes=nbytes,
                                 median_us=t0_us + nbytes / (gbps * 1e9) * 1e6))
    return rows


def test_fit_recovers_parameters():
    f = am.fit(_rows("flashinfer", 6.0, 2500.0), "flashinfer", copy_gbps=3000.0)
    assert abs(f.t0_us - 6.0) < 1e-6 and abs(f.bw_gbps - 2500.0) < 1e-3
    assert abs(f.bw_frac - 2500 / 3000) < 1e-6


def test_projection_scales_bandwidth_with_copy_bw():
    f = am.fit(_rows("flashinfer", 5.0, 2000.0), "flashinfer", copy_gbps=3000.0)
    nb = 1e9
    # same fraction of a GPU with half the copy bandwidth -> byte term doubles
    assert abs(f.predict_us(nb, copy_gbps=1500.0) - (5.0 + 2 * nb / 2000e9 * 1e6)) < 1e-6


def test_best_flashinfer_and_ratio():
    a = _rows("flashinfer_tc0", 8.0, 2000.0)
    b = _rows("flashinfer_tc1", 6.0, 2500.0)
    t = _rows("triton_splitkv", 10.0, 1800.0)
    rows = am.best_flashinfer(a + b + t)
    assert all(r.get("flashinfer_variant") == "flashinfer_tc1" for r in rows if r["kernel"] == "flashinfer")
    s = am.ratio_summary(rows)["uniform"]
    assert s["n"] == 36 and s["max"] < 1.0          # triton slower everywhere in this synthetic case


def test_holdout_on_unseen_configs():
    rows = _rows("flashinfer", 6.0, 2500.0)
    res = am.holdout(rows, "flashinfer", 3000.0)
    assert res["score"]["n"] == 18 and res["score"]["mape"] < 1e-6


def test_per_token_attention():
    f = am.AttnFit("flashinfer", t0_us=5.0, bw_gbps=2000.0, copy_gbps=3000.0, n=10)
    # Qwen3-8B: 36 layers, 8 KV heads, head_dim 128, batch 1, 8k context, bf16
    us = am.attention_us_per_token(f, 36, 8, 128, 1, 8192)
    nbytes = 2 * 8192 * 8 * 128 * 2
    assert abs(us - 36 * (5.0 + nbytes / 2000e9 * 1e6)) < 1e-9


def test_best_flashinfer_keeps_settings_apart():
    a = [dict(r, page_size=16, dtype="float16") for r in _rows("flashinfer_tc0", 8.0, 2000.0)]
    b = [dict(r, page_size=64, dtype="float16") for r in _rows("flashinfer_tc0", 1.0, 9000.0)]
    rows = am.best_flashinfer(a + b)
    assert sum(r["page_size"] == 16 for r in rows) == len(a)   # page-16 rows not replaced by faster page-64 rows
