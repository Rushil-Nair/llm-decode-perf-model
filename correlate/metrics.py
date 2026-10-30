"""Error metrics used everywhere in the correlation report.

Sign convention: error % = 100 * (predicted - measured) / measured.
  * For a time (TPOT, TTFT), a negative error means the model is too optimistic
    (predicts faster than measured) -- "under-predicted time".
  * For a rate (GB/s, tok/s), a positive error means the model is too optimistic.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def pct_error(pred: float, meas: float) -> float:
    """Signed percent error of one prediction. NaN if measured is 0/missing."""
    if pred is None or meas is None:
        return float("nan")
    if meas == 0 or not math.isfinite(meas) or not math.isfinite(pred):
        return float("nan")
    return 100.0 * (pred - meas) / meas


def _pairs(pred: Iterable, meas: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only pairs where both values are finite and measured != 0."""
    p = np.asarray([np.nan if v is None else v for v in pred], dtype=float)
    m = np.asarray([np.nan if v is None else v for v in meas], dtype=float)
    if p.shape != m.shape:
        raise ValueError(f"length mismatch: {p.shape} vs {m.shape}")
    ok = np.isfinite(p) & np.isfinite(m) & (m != 0)
    return p[ok], m[ok]


def mape(pred: Sequence[float], meas: Sequence[float]) -> float:
    """Mean absolute percentage error, in percent: mean(|p - m| / |m|) * 100."""
    p, m = _pairs(pred, meas)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - m) / np.abs(m)) * 100.0)


def max_abs_pct(pred: Sequence[float], meas: Sequence[float]) -> float:
    """Largest |% error| over the set."""
    p, m = _pairs(pred, meas)
    if len(p) == 0:
        return float("nan")
    return float(np.max(np.abs(p - m) / np.abs(m)) * 100.0)


def pearson_r(pred: Sequence[float], meas: Sequence[float], min_n: int = 3) -> float:
    """Pearson correlation of predicted vs measured. NaN when fewer than `min_n`
    points (two points always give |r| = 1, which says nothing) or zero variance."""
    p, m = _pairs(pred, meas)
    if len(p) < min_n:
        return float("nan")
    sp, sm = np.std(p), np.std(m)
    if sp == 0 or sm == 0:
        return float("nan")
    return float(np.mean((p - p.mean()) * (m - m.mean())) / (sp * sm))


def summarize(pred: Sequence[float], meas: Sequence[float]) -> Dict[str, float]:
    """{n, mape, max_abs_pct, pearson_r, mean_signed_pct} for one set of pairs."""
    p, m = _pairs(pred, meas)
    signed = float(np.mean((p - m) / m) * 100.0) if len(p) else float("nan")
    return {
        "n": int(len(p)),
        "mape": mape(pred, meas),
        "max_abs_pct": max_abs_pct(pred, meas),
        "pearson_r": pearson_r(pred, meas),
        "mean_signed_pct": signed,
    }


def fmt(v, spec: str = "{:.1f}", missing: str = "-") -> str:
    """Format a number for a markdown table; None/NaN -> '-'."""
    if v is None:
        return missing
    try:
        if isinstance(v, float) and not math.isfinite(v):
            return missing
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def flag_rows(rows: List[Dict], threshold_pct: float = 15.0, key: str = "err_pct") -> List[Dict]:
    """Rows whose |error| strictly exceeds the threshold (NaN never flags)."""
    out = []
    for r in rows:
        e = r.get(key)
        if e is None:
            continue
        try:
            e = float(e)
        except (TypeError, ValueError):
            continue
        if math.isfinite(e) and abs(e) > threshold_pct:
            out.append(r)
    return out
