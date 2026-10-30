"""Decoder-only transformer cost model (GQA attention + SwiGLU MLP).

Everything here is analytic: FLOPs and bytes moved per operator, as functions of

    B  batch size (number of sequences processed together)
    L  context already in the KV cache *before* this step (past tokens per sequence)
    T  new tokens per sequence in this step (prefill: T = prompt length, L = 0;
       decode: T = 1, L = current context length)
    w_bits   bits per weight (16 for BF16, 8 for FP8/INT8, ~4.9 effective for Q4_K_M)
    kv_bits  bits per KV-cache element
    act_bits bits per activation element

Conventions / assumptions (documented, deliberately simple):
  * A multiply-accumulate is 2 FLOPs. A GEMM [N x K] @ [K x M] costs 2*N*K*M.
  * Each weight matrix is read from memory exactly once per forward pass
    (no re-reads, perfect reuse across the batch). This is the property the
    llama.cpp slot bug violated -- see llamacpp_slots.py.
  * Attention is "flash-style": the B x heads x T x (L+T) score matrix is never
    written to memory; each sequence's K and V for a layer are read once per
    forward pass. Softmax is counted as 5 FLOPs per score element
    (max, subtract, exp, sum, divide) -- a convention, not a measurement.
  * RMSNorm is 4 FLOPs/element; SwiGLU elementwise (silu + mul) is 5 FLOPs/element.
    These are small and only matter for the roofline plot placement.
  * Activations are read and written once per op at act_bits (no fusion credit).
  * Logits: decode computes the LM head for every new token; prefill computes it
    only for the last token of each sequence (llama.cpp / vLLM default).
  * Rotary embedding, residual adds, sampling and bias terms are ignored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# short name -> config file
KNOWN_MODELS = {
    "qwen3-8b": "qwen3-8b.json",
    "llama-3.2-3b": "llama-3.2-3b.json",
    "llama-3.1-8b": "llama-3.1-8b.json",
    "llama-3.1-70b": "llama-3.1-70b.json",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tie_embeddings: bool = False
    qk_norm: bool = False  # Qwen3 has per-head RMSNorm on q and k

    @classmethod
    def from_hf_config(cls, cfg: Dict, name: str | None = None) -> "ModelConfig":
        n_heads = int(cfg["num_attention_heads"])
        hidden = int(cfg["hidden_size"])
        head_dim = int(cfg.get("head_dim") or hidden // n_heads)
        return cls(
            name=name or cfg.get("_name_or_path") or cfg.get("model_type", "model"),
            n_layers=int(cfg["num_hidden_layers"]),
            hidden=hidden,
            n_heads=n_heads,
            n_kv_heads=int(cfg.get("num_key_value_heads", n_heads)),
            head_dim=head_dim,
            intermediate=int(cfg["intermediate_size"]),
            vocab=int(cfg["vocab_size"]),
            tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            qk_norm=cfg.get("model_type") == "qwen3",
        )

    # ---- parameter counts -------------------------------------------------
    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def params_per_layer(self) -> Dict[str, int]:
        h, I = self.hidden, self.intermediate
        p = {
            "qkv_proj": h * (self.q_dim + 2 * self.kv_dim),
            "o_proj": self.q_dim * h,
            "mlp_gate_up": 2 * h * I,
            "mlp_down": I * h,
            "norms": 2 * h + (2 * self.head_dim if self.qk_norm else 0),
        }
        return p

    def params(self) -> Dict[str, int]:
        per_layer = sum(self.params_per_layer().values())
        emb = self.vocab * self.hidden
        return {
            "layers": per_layer * self.n_layers,
            "embedding": emb,
            "lm_head": 0 if self.tie_embeddings else emb,
            "final_norm": self.hidden,
        }

    def total_params(self) -> int:
        return sum(self.params().values())

    def non_embedding_params(self) -> int:
        p = self.params()
        return p["layers"] + p["final_norm"]

    def kv_bytes_per_token(self, kv_bits: float = 16) -> float:
        """KV-cache bytes per token per sequence, all layers (K and V)."""
        return 2 * self.n_layers * self.kv_dim * kv_bits / 8

    def weight_bytes(self, w_bits: float = 16, include_embedding_table: bool = True) -> float:
        """Bytes to store the weights. The embedding table is stored but only
        B*T rows of it are read per step, so it is excluded from per-step reads."""
        p = self.params()
        n = p["layers"] + p["lm_head"] + p["final_norm"]
        if include_embedding_table or self.tie_embeddings:
            # tied: the table *is* the LM head and is read in full every step
            n += p["embedding"]
        return n * w_bits / 8


def load_model(name: str) -> ModelConfig:
    """Load one of KNOWN_MODELS (or a path to an HF config.json)."""
    path = Path(name)
    if name in KNOWN_MODELS:
        path = CONFIG_DIR / KNOWN_MODELS[name]
        short = name
    else:
        short = path.stem
    with open(path) as f:
        return ModelConfig.from_hf_config(json.load(f), name=short)


@dataclass
class Op:
    """One operator instance. `count` = how many times it runs per forward pass
    (n_layers for per-layer ops, 1 for embedding/LM head)."""

    name: str
    flops: float
    weight_bytes: float = 0.0
    kv_read_bytes: float = 0.0
    kv_write_bytes: float = 0.0
    act_bytes: float = 0.0
    count: int = 1
    kind: str = "gemm"  # gemm | attention | elementwise | lookup

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes + self.act_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else float("inf")

    @property
    def total_flops(self) -> float:
        return self.flops * self.count

    @property
    def total_bytes(self) -> float:
        return self.bytes * self.count


def _gemm(name, n_tok, k, m, w_bytes_per_el, a_bytes, count) -> Op:
    return Op(
        name=name,
        flops=2.0 * n_tok * k * m,
        weight_bytes=k * m * w_bytes_per_el,
        act_bytes=(n_tok * k + n_tok * m) * a_bytes,
        count=count,
        kind="gemm",
    )


def layer_ops(
    m: ModelConfig,
    B: int,
    L: int,
    T: int,
    w_bits: float = 16,
    kv_bits: float = 16,
    act_bits: float = 16,
) -> List[Op]:
    """Ops for one forward pass over B sequences x T new tokens with L cached tokens."""
    if B < 1 or T < 1 or L < 0:
        raise ValueError("need B>=1, T>=1, L>=0")
    wb, kb, ab = w_bits / 8, kv_bits / 8, act_bits / 8
    N = B * T  # tokens in the pass
    h, nl = m.hidden, m.n_layers
    ops: List[Op] = []

    # embedding lookup: reads N rows of the table (not the whole table)
    ops.append(Op("embedding", 0.0, weight_bytes=N * h * wb, act_bytes=N * h * ab, kind="lookup"))

    def norm(name, n_el, count):
        return Op(name, 4.0 * n_el, weight_bytes=h * ab, act_bytes=2 * n_el * ab,
                  count=count, kind="elementwise")

    ops.append(norm("attn_norm", N * h, nl))
    ops.append(_gemm("qkv_proj", N, h, m.q_dim + 2 * m.kv_dim, wb, ab, nl))
    if m.qk_norm:
        n_el = N * (m.q_dim + m.kv_dim)
        ops.append(Op("qk_norm", 4.0 * n_el, weight_bytes=2 * m.head_dim * ab,
                      act_bytes=2 * n_el * ab, count=nl, kind="elementwise"))

    # attention: query token i (0-based within the T new ones) attends to L+i+1 keys
    pairs = B * (T * L + T * (T + 1) / 2)  # (query, key) pairs summed over sequences
    attn_flops = pairs * m.n_heads * (4.0 * m.head_dim + 5.0)  # QK^T + AV + softmax
    kv_read = B * (L + T) * 2 * m.kv_dim * kb  # each seq's K,V read once per layer
    kv_write = N * 2 * m.kv_dim * kb  # new K,V appended to the cache
    attn_act = 2 * N * m.q_dim * ab  # read Q, write context
    ops.append(Op("attention", attn_flops, kv_read_bytes=kv_read, kv_write_bytes=kv_write,
                  act_bytes=attn_act, count=nl, kind="attention"))

    ops.append(_gemm("o_proj", N, m.q_dim, h, wb, ab, nl))
    ops.append(norm("mlp_norm", N * h, nl))
    ops.append(_gemm("mlp_gate_up", N, h, 2 * m.intermediate, wb, ab, nl))
    n_el = N * m.intermediate
    ops.append(Op("swiglu", 5.0 * n_el, act_bytes=3 * n_el * ab, count=nl, kind="elementwise"))
    ops.append(_gemm("mlp_down", N, m.intermediate, h, wb, ab, nl))

    ops.append(norm("final_norm", N * h, 1))
    n_logits = N if T == 1 else B  # prefill: last token only
    ops.append(_gemm("lm_head", n_logits, h, m.vocab, wb, ab, 1))
    return ops


def forward_ops(m: ModelConfig, B: int, L: int, T: int, **kw) -> List[Op]:
    return layer_ops(m, B, L, T, **kw)


def decode_ops(m: ModelConfig, B: int, L: int, **kw) -> List[Op]:
    """One decode step: every sequence has L tokens cached and generates 1."""
    return layer_ops(m, B, L, 1, **kw)


def prefill_ops(m: ModelConfig, B: int, prompt_len: int, **kw) -> List[Op]:
    return layer_ops(m, B, 0, prompt_len, **kw)


def totals(ops: List[Op]) -> Dict[str, float]:
    return {
        "flops": sum(o.total_flops for o in ops),
        "bytes": sum(o.total_bytes for o in ops),
        "weight_bytes": sum(o.weight_bytes * o.count for o in ops),
        "kv_read_bytes": sum(o.kv_read_bytes * o.count for o in ops),
        "kv_write_bytes": sum(o.kv_write_bytes * o.count for o in ops),
        "act_bytes": sum(o.act_bytes * o.count for o in ops),
    }


def kv_cache_bytes(m: ModelConfig, B: int, L: int, kv_bits: float = 16) -> float:
    """Resident KV-cache size for B sequences of L tokens."""
    return B * L * m.kv_bytes_per_token(kv_bits)
