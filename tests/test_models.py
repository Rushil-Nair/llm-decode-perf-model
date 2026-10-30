"""FLOP / byte counts against hand-derived numbers for a toy config."""
import pytest

from perfsim.models import ModelConfig, decode_ops, layer_ops, load_model, prefill_ops, totals

TOY = ModelConfig(name="toy", n_layers=2, hidden=8, n_heads=2, n_kv_heads=1, head_dim=4,
                  intermediate=16, vocab=32, tie_embeddings=False)


def by_name(ops):
    return {o.name: o for o in ops}


def test_param_counts():
    p = TOY.params_per_layer()
    assert p["qkv_proj"] == 8 * (8 + 2 * 4) == 128
    assert p["o_proj"] == 64
    assert p["mlp_gate_up"] == 2 * 8 * 16 == 256
    assert p["mlp_down"] == 128
    assert p["norms"] == 16
    # 2 layers * 592 + embedding 256 + lm_head 256 + final norm 8
    assert TOY.total_params() == 2 * 592 + 256 + 256 + 8 == 1704


def test_decode_flops_hand_derived():
    B, L = 2, 5
    ops = by_name(decode_ops(TOY, B, L, w_bits=16, kv_bits=16))
    N = B
    assert ops["qkv_proj"].flops == 2 * N * 8 * 16 == 512
    assert ops["o_proj"].flops == 2 * N * 8 * 8
    assert ops["mlp_gate_up"].flops == 2 * N * 8 * 32
    assert ops["mlp_down"].flops == 2 * N * 16 * 8
    # each of the 2 sequences: 1 query x (5 cached + 1 new) keys = 6 pairs
    pairs = 12
    assert ops["attention"].flops == pairs * 2 * (4 * 4 + 5) == 504
    assert ops["lm_head"].flops == 2 * N * 8 * 32
    assert ops["qkv_proj"].count == 2 and ops["lm_head"].count == 1


def test_decode_bytes_hand_derived():
    B, L = 2, 5
    ops = by_name(decode_ops(TOY, B, L, w_bits=16, kv_bits=16, act_bits=16))
    # K and V, kv_dim=4, 2 bytes, (L+1) positions per sequence
    assert ops["attention"].kv_read_bytes == B * (L + 1) * 2 * 4 * 2 == 192
    assert ops["attention"].kv_write_bytes == B * 1 * 2 * 4 * 2 == 32
    assert ops["qkv_proj"].weight_bytes == 128 * 2
    assert ops["qkv_proj"].act_bytes == (B * 8 + B * 16) * 2
    t = totals(list(ops.values()))
    # GEMM weights read once per pass: 2 layers * 576 params + lm_head 256, at 2 bytes
    gemm_w = sum(o.weight_bytes * o.count for o in ops.values() if o.kind == "gemm")
    assert gemm_w == (2 * 576 + 256) * 2 == 2816
    assert t["kv_read_bytes"] == 2 * 192


def test_weight_bytes_do_not_scale_with_batch():
    a = totals(decode_ops(TOY, 1, 100))
    b = totals(decode_ops(TOY, 64, 100))
    gemm = lambda B: sum(o.weight_bytes * o.count for o in decode_ops(TOY, B, 100) if o.kind == "gemm")
    assert gemm(1) == gemm(64)
    assert b["kv_read_bytes"] == 64 * a["kv_read_bytes"]


def test_prefill_causal_pairs_and_last_token_logits():
    T = 4
    ops = by_name(prefill_ops(TOY, 1, T))
    pairs = T * (T + 1) / 2  # causal: 1+2+3+4
    assert ops["attention"].flops == pairs * 2 * (4 * 4 + 5)
    assert ops["lm_head"].flops == 2 * 1 * 8 * 32  # last token only
    # prefill of T tokens = same GEMM FLOPs as T decode tokens for the projections
    assert ops["qkv_proj"].flops == T * by_name(decode_ops(TOY, 1, 0))["qkv_proj"].flops


def test_quantized_weights_scale_bytes():
    w16 = by_name(decode_ops(TOY, 1, 0, w_bits=16))["mlp_down"].weight_bytes
    w4 = by_name(decode_ops(TOY, 1, 0, w_bits=4))["mlp_down"].weight_bytes
    assert w4 == w16 / 4


def test_kv_bytes_per_token():
    assert TOY.kv_bytes_per_token(16) == 2 * 2 * 4 * 2  # K+V, layers, kv_dim, bytes


def test_real_configs_match_published_param_counts():
    # Qwen3-8B model card: 8.2B total, 6.95B non-embedding
    q = load_model("qwen3-8b")
    assert q.total_params() / 1e9 == pytest.approx(8.2, abs=0.05)
    assert q.non_embedding_params() / 1e9 == pytest.approx(6.95, abs=0.01)
    assert load_model("llama-3.1-70b").head_dim == 128  # derived, key absent in config
    assert load_model("llama-3.1-8b").total_params() / 1e9 == pytest.approx(8.03, abs=0.01)


def test_invalid_args():
    with pytest.raises(ValueError):
        layer_ops(TOY, 0, 0, 1)
