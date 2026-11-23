"""Decode-attention term for the decode model, fitted to the paged-attention benchmark.

At decode every layer streams the whole KV cache once, so attention time is modelled like the GEMV
level: a fixed cost plus bytes over an effective bandwidth,

    t_attn(layer) = t0 + bytes / BW_eff,     bytes = 2 * sum_b(L_b) * Hkv * D * elt  (+ q, o)

BW_eff is stored as a fraction of the GPU's measured copy bandwidth, so a fit on one GPU can be
projected to another (leave-one-GPU-out), the same way the GEMV level is scored.

Inputs: rows written by bench/bench_attention.py (one JSON object per kernel x shape).
"""
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


def load_rows(data_dir: str, mode: str = "matrix") -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"{mode}_*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def load_env(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "env.json")) as f:
        return json.load(f)


def best_flashinfer(rows: List[dict]) -> List[dict]:
    """Collapse flashinfer_tc0 / flashinfer_tc1 into one 'flashinfer' row per shape (the faster one),
    i.e. compare against the best configuration of the baseline."""
    key = lambda r: (r["config"], r["batch"], r["ctx"], r.get("skew", False),  # noqa: E731
                     r.get("page_size"), r.get("dtype"))
    best: Dict[tuple, dict] = {}
    out = []
    for r in rows:
        if r["kernel"].startswith("flashinfer"):
            k = key(r)
            if k not in best or r["median_us"] < best[k]["median_us"]:
                best[k] = r
        else:
            out.append(r)
    for r in best.values():
        out.append(dict(r, kernel="flashinfer", flashinfer_variant=r["kernel"]))
    return out


@dataclass
class AttnFit:
    kernel: str
    t0_us: float          # fixed cost per call
    bw_gbps: float        # effective bandwidth of the byte-proportional part
    copy_gbps: float      # this GPU's measured copy bandwidth
    n: int

    @property
    def bw_frac(self) -> float:
        return self.bw_gbps / self.copy_gbps

    def predict_us(self, nbytes: float, copy_gbps: Optional[float] = None) -> float:
        bw = self.bw_frac * copy_gbps if copy_gbps else self.bw_gbps
        return self.t0_us + nbytes / (bw * 1e9) * 1e6


def fit(rows: Iterable[dict], kernel: str, copy_gbps: float) -> AttnFit:
    """Least squares t = t0 + bytes * c with t0 >= 0 (c = 1 / BW_eff)."""
    rs = [r for r in rows if r["kernel"] == kernel]
    if len(rs) < 2:
        raise ValueError(f"need >= 2 rows for {kernel}, have {len(rs)}")
    x = np.array([r["bytes"] for r in rs], dtype=float)
    y = np.array([r["median_us"] for r in rs], dtype=float) * 1e-6
    # relative-error weighting: small and large shapes count equally
    w = 1.0 / y
    A = np.stack([np.ones_like(x), x], 1) * w[:, None]
    t0, c = np.linalg.lstsq(A, y * w, rcond=None)[0]
    if t0 < 0:  # refit through the origin
        t0, c = 0.0, float(np.sum(w * w * x * y) / np.sum(w * w * x * x))
    return AttnFit(kernel, t0 * 1e6, 1.0 / c / 1e9, copy_gbps, len(rs))


def mape(rows: Iterable[dict], pred) -> Dict[str, float]:
    errs = [abs(pred(r) - r["median_us"]) / r["median_us"] for r in rows]
    if not errs:
        return dict(n=0, mape=float("nan"), max=float("nan"))
    return dict(n=len(errs), mape=100 * float(np.mean(errs)), max=100 * float(np.max(errs)))


def ratio_summary(rows: List[dict], num: str = "triton_splitkv", den: str = "flashinfer") -> Dict[str, dict]:
    """time(den) / time(num): > 1 means `num` is faster. Separate for uniform and skewed batches."""
    by = {}
    for r in rows:
        by.setdefault((r["config"], r["batch"], r["ctx"], bool(r.get("skew")), r.get("page_size"),
                       r.get("dtype")), {})[r["kernel"]] = r
    out = {}
    for skew in (False, True):
        rat = [v[den]["median_us"] / v[num]["median_us"] for k, v in by.items()
               if k[3] == skew and num in v and den in v]
        if rat:
            out["skewed" if skew else "uniform"] = dict(
                n=len(rat), geomean=float(math.exp(np.mean(np.log(rat)))),
                min=float(min(rat)), max=float(max(rat)))
    return out


def attention_us_per_token(fit_: AttnFit, num_layers: int, num_kv_heads: int, head_dim: int,
                           batch: int, context_len: int, elt_bytes: int = 2,
                           copy_gbps: Optional[float] = None) -> float:
    """Decode attention time for one generated token step of the whole model (all layers)."""
    nbytes = 2 * batch * context_len * num_kv_heads * head_dim * elt_bytes
    return num_layers * fit_.predict_us(nbytes, copy_gbps)


def holdout(rows: List[dict], kernel: str, copy_gbps: float,
            train_configs=("llama8b", "mha"), test_configs=("qwen7b", "tp8_70b")) -> dict:
    """Fit on some head configurations, score on the others (unseen GQA ratios)."""
    tr = [r for r in rows if r["config"] in train_configs]
    te = [r for r in rows if r["config"] in test_configs and r["kernel"] == kernel]
    f = fit(tr, kernel, copy_gbps)
    return dict(fit=f, score=mape(te, lambda r: f.predict_us(r["bytes"])))
