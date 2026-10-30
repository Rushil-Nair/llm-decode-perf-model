"""Level 2: per-op calibrated roofline (speed-of-light x measured efficiency).

Same idea as NVIDIA AIConfigurator: measure each operator in isolation on the real
GPU (bench/ops_bench.py), express the measurement as an efficiency against the
spec-sheet speed of light (SOL), and use that efficiency for any op of that kind.

Per op instance (FLOPs F, bytes moved Q, from perfsim.models or the bench record):

    SOL(op)  = max(F / peak_flops[dtype], Q / peak_bw)          spec sheet, hw/hardware.yaml
    time(op) = t0[family] + SOL(op) / eff[family](feature(op))  "empirical" mode
    time(op) = SOL(op)                                           "sol" mode

  * family: gemm | attn_decode | attn_prefill | rmsnorm (| gemv_q4 if measured).
  * feature: the size variable the efficiency is interpolated over
        gemm          arithmetic intensity F/Q  (~ M, the token count, for M << K, N)
        attn_decode   bytes  (memory bound: KV cache streaming)
        attn_prefill  FLOPs  (compute bound)
        rmsnorm       bytes  (memory bound)
        anything else bytes
  * t0: fixed per-kernel cost (launch, prologue, tail), in seconds. Fitted from the
    "probe" records (a deliberately tiny instance of the same op, so its time is
    ~all overhead): t0 = median(t_probe - SOL_probe). Without probes:
    t0 = max(0, min_i(t_i - SOL_i)), the largest constant that never makes any
    measurement beat SOL. The bench measures GPU time with the CPU running ahead,
    so t0 is the GPU-side cost of a kernel, not the CPU launch latency.
  * eff_i = SOL_i / (t_i - t0) for each measured (non-probe) shape, i.e. the
    fraction of speed of light achieved by the part of the time that scales with
    work. Clipped to [1e-3, eff_max] (eff_max = 1 by default). Clipped points and
    points measured faster than SOL are counted and reported (a spec sheet below
    the real clocks, e.g. RTX 5090 boosting past its rated 2.41 GHz, shows up here).
  * Curve: piecewise-linear interpolation of eff against log(feature) through the
    measured points, flat beyond the measured range. Linear interpolation is
    shape-preserving: between two neighbouring measurements it is monotone and
    never overshoots them (unlike a spline), and it reproduces every measured
    point exactly. Several measurements at the same feature value (e.g. decode
    attention with equal KV bytes from different batch x length splits) are
    replaced by their median.
  * GEMM shapes: one curve per weight-matrix size (bytes of K x N, which identifies
    the projection: q/o, k/v, qkv, gate/up/down, gate_up, lm_head) built from that
    shape's own points, plus a pooled curve (per token count M: median AI and
    median efficiency over all shapes) used for shapes that were not measured.
    An Op is matched to a measured shape by its weight bytes (0.5% tolerance).
  * Duplicate measurements of one shape with different implementations (SDPA
    backends, compiled vs eager RMSNorm): the fastest is kept, i.e. we model an
    engine that uses the best available kernel.

Cross-GPU use (leave-one-GPU-out projection): efficiency is a dimensionless ratio
against a spec sheet, so a calibration from GPU A can be applied to GPU B's spec
sheet:  time_B(op) = t0_A + SOL_B(op) / eff_A(feature(op)).  Use
`OpCostModel.projected(hw_B, calib_A)` (or `OpCostModel(hw_B, calib_A, "empirical")`;
the model notices the mismatch, sets `is_projection` and says so in describe()).
Pass cross_gpu=False to forbid it. Known limitation: the GEMM feature is the raw
arithmetic intensity, so the efficiency dip near A's ridge point (peak/BW) is
carried to the same AI on B even though B's ridge point is elsewhere.

How a perfsim.models.Op maps to a family:
  kind "gemm"        -> gemm (dtype from op.dtype if set, else the model default).
                        Weight-only quantized dtypes (q4_k, ...) without their own
                        gemm curve use the gemv_q4 curve if present (proxy).
  kind "attention"   -> attn_decode / attn_prefill: from the `phase` argument, else
                        op.phase, else inferred (infer_attention_phase).
  "norm" in the name -> rmsnorm (attn_norm, mlp_norm, final_norm, qk_norm)
  other elementwise / lookup (swiglu, embedding) -> rmsnorm curve as a PROXY for
                        a memory-bound elementwise kernel (flagged in explain()).
  anything without a curve -> SOL with eff = 1, t0 = 0 (counted in .fallbacks).
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .hardware import Hardware
from .models import Op

FEATURE = {"gemm": "ai", "attn_prefill": "flops", "attn_decode": "bytes",
           "rmsnorm": "bytes", "gemv_q4": "bytes"}
DEFAULT_FEATURE = "bytes"
FAMILIES = tuple(FEATURE)

DTYPE_ALIASES = {"bfloat16": "bf16", "f16": "fp16", "half": "fp16", "float16": "fp16",
                 "float32": "fp32", "f32": "fp32"}
# fp16 and bf16 have the same size and the same tensor-core rate: curves are shared
EQUIVALENT_DTYPES = {"bf16": ("bf16", "fp16"), "fp16": ("fp16", "bf16")}
ELEM_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "int8": 1}

EFF_MIN = 1e-3
SHAPE_MATCH_RTOL = 0.005
CALIB_VERSION = 1


# ----------------------------------------------------------------------------------
# Speed of light
# ----------------------------------------------------------------------------------
def norm_dtype(dtype) -> str:
    d = str(dtype).lower()
    return DTYPE_ALIASES.get(d, d)


def is_weight_only_quant(dtype) -> bool:
    """q4_k, q4_0, q8_0, int4, nf4 ...: weights dequantized, math done in fp16."""
    d = norm_dtype(dtype)
    return d.startswith("q") or d in ("int4", "nf4")


def precision_for(dtype, hw: Hardware) -> str:
    """hardware.yaml peak_flops key used for this dtype's compute. Weight-only
    quantized formats compute in fp16 after dequantization."""
    d = norm_dtype(dtype)
    for p in (d,) + EQUIVALENT_DTYPES.get(d, ()):
        if p in hw.peak_flops:
            return p
    for p in ("fp16", "bf16"):
        if p in hw.peak_flops:
            return p
    raise KeyError(f"{hw.key}: no peak for dtype {dtype!r} (have {sorted(hw.peak_flops)})")


def sol_time(flops: float, nbytes: float, hw: Hardware, dtype="bf16") -> float:
    """Speed-of-light seconds: max(compute time at peak, memory time at peak BW)."""
    return max(flops / hw.peak(precision_for(dtype, hw)), nbytes / hw.memory_bandwidth)


def feature_value(feature: str, flops: float, nbytes: float) -> float:
    if feature == "ai":
        return flops / nbytes
    if feature == "bytes":
        return nbytes
    if feature == "flops":
        return flops
    raise ValueError(f"unknown feature {feature!r}")


# ----------------------------------------------------------------------------------
# Efficiency curve
# ----------------------------------------------------------------------------------
def _interp_log(xs: List[float], ys: List[float], x: float) -> float:
    """Piecewise-linear in log(x), flat outside [xs[0], xs[-1]] (np.interp clamps)."""
    if len(xs) == 1:
        return float(ys[0])
    return float(np.interp(math.log(x), np.log(np.asarray(xs, float)), np.asarray(ys, float)))


def _aggregate(points: Iterable[Tuple[object, float, float]]) -> Tuple[List[float], List[float]]:
    """points: (group key, x, eff). Median x and median eff per group, sorted by x;
    groups that end up at the same x are merged (median again)."""
    groups: Dict[object, List[Tuple[float, float]]] = {}
    for key, x, e in points:
        groups.setdefault(key, []).append((x, e))
    knots = sorted((statistics.median(x for x, _ in g), statistics.median(e for _, e in g))
                   for g in groups.values())
    xs: List[float] = []
    es: List[List[float]] = []
    for x, e in knots:
        if xs and math.isclose(x, xs[-1], rel_tol=1e-9):
            es[-1].append(e)
        else:
            xs.append(x)
            es.append([e])
    return xs, [statistics.median(e) for e in es]


def _record_weight_bytes(rec: dict) -> Optional[float]:
    s = rec.get("shape") or {}
    if "weight_bytes" in s:
        return float(s["weight_bytes"])
    eb = ELEM_BYTES.get(norm_dtype(rec.get("dtype", "")))
    if eb and "K" in s and "N" in s:
        return float(s["K"] * s["N"] * eb)
    return None


def _is_probe(rec: dict) -> bool:
    return bool((rec.get("shape") or {}).get("probe", False))


@dataclass
class EfficiencyCurve:
    """eff(size) in (0, 1] for one (op family, dtype), plus its fixed overhead t0."""

    op: str
    dtype: str
    feature: str
    xs: List[float]                      # knots of the pooled curve (ascending)
    effs: List[float]
    t0_s: float = 0.0
    t0_source: str = "none"              # "probe" | "min_excess" | "none"
    # gemm only: str(int(weight_bytes)) -> {"xs": [...], "effs": [...], "shapes": [...]}
    shape_curves: Dict[str, Dict] = field(default_factory=dict)
    n_records: int = 0
    n_probes: int = 0
    n_clipped: int = 0                   # eff hit eff_max (t - t0 <= SOL)
    n_faster_than_sol: int = 0           # measured time below the spec-sheet SOL
    eff_max: float = 1.0

    # -- evaluation -------------------------------------------------------------
    def lookup(self, size: float, weight_bytes: Optional[float] = None) -> Tuple[float, str]:
        """(efficiency, "shape"|"pooled"). The per-shape curve is used when the
        op's weight bytes match a measured GEMM shape."""
        if size <= 0 or not math.isfinite(size):
            raise ValueError(f"{self.op}: feature value must be > 0, got {size}")
        if weight_bytes and self.shape_curves:
            best, err = None, SHAPE_MATCH_RTOL
            for k, c in self.shape_curves.items():
                rel = abs(weight_bytes - float(k)) / float(k)
                if rel <= err:
                    best, err = c, rel
            if best is not None:
                return _interp_log(best["xs"], best["effs"], size), "shape"
        return _interp_log(self.xs, self.effs, size), "pooled"

    def eff(self, size: float, weight_bytes: Optional[float] = None) -> float:
        return self.lookup(size, weight_bytes)[0]

    __call__ = eff

    def predict(self, sol_s: float, size: float, weight_bytes: Optional[float] = None) -> float:
        """t0 + SOL / eff(size)  (seconds)."""
        return self.t0_s + sol_s / self.eff(size, weight_bytes)

    # -- fitting ----------------------------------------------------------------
    @classmethod
    def fit(cls, op: str, dtype: str, records: Iterable[dict], hw: Hardware,
            eff_max: float = 1.0) -> "EfficiencyCurve":
        """Fit from ops.jsonl records of ONE (op, dtype), already de-duplicated
        (see keep_fastest). `hw` is the spec sheet of the GPU that was measured."""
        feature = FEATURE.get(op, DEFAULT_FEATURE)
        rows = []
        for r in records:
            t = float(r["us"]) * 1e-6
            fl, nb = float(r["flops"]), float(r["bytes"])
            if not (t > 0 and nb > 0 and fl >= 0):
                continue
            x = feature_value(feature, fl, nb)
            if not x > 0:
                continue
            rows.append({"t": t, "sol": sol_time(fl, nb, hw, dtype), "x": x, "rec": r,
                         "probe": _is_probe(r)})
        main = [w for w in rows if not w["probe"]]
        probes = [w for w in rows if w["probe"]]
        if not main:
            raise ValueError(f"no usable (non-probe) records for {op}/{dtype}")

        # fixed per-kernel overhead
        if probes:
            t0 = max(0.0, statistics.median(w["t"] - w["sol"] for w in probes))
            t0_source = "probe"
        else:
            t0 = max(0.0, min(w["t"] - w["sol"] for w in main))
            t0_source = "min_excess"

        n_clipped = n_fast = 0
        pooled_pts, shape_pts, shape_names = [], {}, {}
        for w in main:
            net = w["t"] - t0
            raw = w["sol"] / net if net > 0 else math.inf
            if raw > eff_max:
                n_clipped += 1
            if w["t"] < w["sol"]:
                n_fast += 1
            e = min(eff_max, max(EFF_MIN, raw))
            s = w["rec"].get("shape") or {}
            xkey = ("x", round(math.log(w["x"]), 9))
            # pooled GEMM curve: group by token count M across all shapes
            key = ("M", s["M"]) if (op == "gemm" and "M" in s) else xkey
            pooled_pts.append((key, w["x"], e))
            if op == "gemm":
                wb = _record_weight_bytes(w["rec"])
                if wb:
                    k = str(int(round(wb)))
                    shape_pts.setdefault(k, []).append((xkey, w["x"], e))
                    shape_names.setdefault(k, set()).add(
                        s.get("proj") or f"{s.get('K')}x{s.get('N')}")
        xs, effs = _aggregate(pooled_pts)
        shape_curves = {}
        for k, pts in shape_pts.items():
            sx, se = _aggregate(pts)
            shape_curves[k] = {"xs": sx, "effs": se, "shapes": sorted(map(str, shape_names[k]))}
        return cls(op=op, dtype=norm_dtype(dtype), feature=feature, xs=xs, effs=effs, t0_s=t0,
                   t0_source=t0_source, shape_curves=shape_curves, n_records=len(main),
                   n_probes=len(probes), n_clipped=n_clipped, n_faster_than_sol=n_fast,
                   eff_max=eff_max)

    # -- (de)serialisation -------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"op": self.op, "dtype": self.dtype, "feature": self.feature,
                "xs": list(self.xs), "effs": list(self.effs), "t0_s": self.t0_s,
                "t0_source": self.t0_source, "shape_curves": self.shape_curves,
                "n_records": self.n_records, "n_probes": self.n_probes,
                "n_clipped": self.n_clipped, "n_faster_than_sol": self.n_faster_than_sol,
                "eff_max": self.eff_max}

    @classmethod
    def from_dict(cls, d: Dict) -> "EfficiencyCurve":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ----------------------------------------------------------------------------------
# Calibration = all curves of one GPU
# ----------------------------------------------------------------------------------
@dataclass
class Calibration:
    gpu_tag: str                 # data/<gpu_tag>/
    hw_key: str                  # hardware.yaml device whose spec sheet the efficiencies are relative to
    curves: Dict[str, EfficiencyCurve]   # "op|dtype" -> curve
    meta: Dict = field(default_factory=dict)

    @staticmethod
    def key(op: str, dtype: str) -> str:
        return f"{op}|{norm_dtype(dtype)}"

    def get(self, op: str, dtype) -> Optional[EfficiencyCurve]:
        d = norm_dtype(dtype)
        for dd in (d,) + EQUIVALENT_DTYPES.get(d, ()):
            c = self.curves.get(self.key(op, dd))
            if c is not None:
                return c
        return None

    def any_dtype(self, op: str) -> Optional[EfficiencyCurve]:
        for k in sorted(self.curves):
            if k.split("|", 1)[0] == op:
                return self.curves[k]
        return None

    def to_dict(self) -> Dict:
        return {"version": CALIB_VERSION, "gpu_tag": self.gpu_tag, "hw_key": self.hw_key,
                "meta": self.meta, "curves": {k: c.to_dict() for k, c in sorted(self.curves.items())}}

    @classmethod
    def from_dict(cls, d: Dict) -> "Calibration":
        return cls(gpu_tag=d["gpu_tag"], hw_key=d["hw_key"], meta=d.get("meta", {}),
                   curves={k: EfficiencyCurve.from_dict(c) for k, c in d["curves"].items()})

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_calib(path) -> Calibration:
    """data/<tag>/calib.json -> Calibration (the loader correlate/backend.py looks for)."""
    return Calibration.load(path)


def read_ops_jsonl(path) -> Tuple[List[dict], List[str]]:
    """(records, problems). Malformed or incomplete lines are skipped and reported."""
    recs, problems = [], []
    need = ("op", "dtype", "us", "flops", "bytes")
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {i}: bad JSON ({e})")
                continue
            missing = [k for k in need if k not in r]
            if missing:
                problems.append(f"line {i}: missing {missing}")
                continue
            try:
                ok = float(r["us"]) > 0 and float(r["bytes"]) > 0 and float(r["flops"]) >= 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                problems.append(f"line {i}: non-positive or non-numeric us/bytes/flops")
                continue
            r.setdefault("shape", {})
            recs.append(r)
    return recs, problems


def shape_key(rec: dict) -> Tuple[str, str, str]:
    return (rec["op"], norm_dtype(rec["dtype"]), json.dumps(rec.get("shape", {}), sort_keys=True))


def keep_fastest(records: Iterable[dict]) -> List[dict]:
    """One record per (op, dtype, shape): the fastest implementation."""
    best: Dict[Tuple, dict] = {}
    for r in records:
        k = shape_key(r)
        if k not in best or float(r["us"]) < float(best[k]["us"]):
            best[k] = r
    return list(best.values())


def fit_calibration(records: Iterable[dict], hw: Hardware, gpu_tag: Optional[str] = None,
                    eff_max: float = 1.0) -> Calibration:
    """Fit one EfficiencyCurve per (op, dtype) from raw ops.jsonl records."""
    raw = list(records)
    used = keep_fastest(raw)
    by: Dict[Tuple[str, str], List[dict]] = {}
    for r in used:
        by.setdefault((r["op"], norm_dtype(r["dtype"])), []).append(r)
    curves, skipped = {}, {}
    for (op, dt), rs in sorted(by.items()):
        try:
            curves[Calibration.key(op, dt)] = EfficiencyCurve.fit(op, dt, rs, hw, eff_max)
        except ValueError as e:
            skipped[Calibration.key(op, dt)] = str(e)
    meta = {
        "n_records_raw": len(raw), "n_records_used": len(used), "eff_max": eff_max,
        "spec_sheet": {"hw_key": hw.key, "name": hw.name, "memory_bandwidth": hw.memory_bandwidth,
                       "peak_flops": dict(hw.peak_flops)},
        "model": "time = t0 + max(F/peak, Q/BW) / eff(feature)",
    }
    if skipped:
        meta["skipped"] = skipped
    return Calibration(gpu_tag=gpu_tag or hw.key, hw_key=hw.key, curves=curves, meta=meta)


# ----------------------------------------------------------------------------------
# Cost model used by Level 3 (perfsim/engine.py) and correlate/
# ----------------------------------------------------------------------------------
def infer_attention_phase(op: Op, act_elem_bytes: float = 2.0) -> str:
    """"decode" (one new token per sequence) or "prefill", from the Op's numbers alone.

    For perfsim.models attention ops (P query-key pairs, K = KV tokens read,
    N = new tokens, Hq/Hkv heads, head dim D, activation element size a):
        flops = P*Hq*(4D+5), kv_read = K*2*Hkv*D*kb, kv_write = N*2*Hkv*D*kb, act = 2*N*Hq*D*a
    so   flops*kv_write / (kv_read*act) = (P/K) * (4D+5)/(2*D*a)
    P/K is the mean number of queries per KV token: exactly 1 for decode, (T+1)/2 for
    a T-token prefill. With a = 2 bytes (bf16/fp16 activations) the factor is ~1.01,
    so the ratio is ~1.0 for decode and >= ~1.5 for any prefill. Threshold 1.25.
    """
    if op.kv_write_bytes > 0 and op.kv_read_bytes > 0 and op.act_bytes > 0 and op.flops > 0:
        r = op.flops * op.kv_write_bytes / (op.kv_read_bytes * op.act_bytes) * (act_elem_bytes / 2.0)
        return "decode" if r < 1.25 else "prefill"
    # no KV write recorded (e.g. built from a bench record): fall back on intensity
    return "decode" if op.arithmetic_intensity < 16 else "prefill"


class OpCostModel:
    """time(op) in seconds for ONE instance of a perfsim.models.Op (multiply by
    op.count, as with roofline.op_time, or use total_time / pass_time).

    OpCostModel(hw, calib=None, mode="sol"|"empirical", dtype="bf16")
      hw      the spec sheet to predict FOR (the target GPU)
      calib   Calibration / path to calib.json / its dict; required for "empirical"
      mode    "sol": pure speed of light; "empirical": t0 + SOL / eff
      dtype   default dtype when an Op has no `dtype` attribute
      family_map  optional {op name: family or (family, dtype)} overrides
      cross_gpu   None: allow a calibration from another GPU and flag it
                  (is_projection); False: refuse it; True: state the intent.
    """

    MODES = ("sol", "empirical")

    def __init__(self, hw: Hardware, calib: Union[Calibration, Dict, str, Path, None] = None,
                 mode: str = "sol", dtype: str = "bf16",
                 family_map: Optional[Dict[str, Union[str, Tuple[str, str]]]] = None,
                 cross_gpu: Optional[bool] = None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if isinstance(calib, (str, Path)):
            calib = load_calib(calib)
        elif isinstance(calib, dict):
            calib = Calibration.from_dict(calib)
        if mode == "empirical" and calib is None:
            raise ValueError("empirical mode needs a calibration (data/<tag>/calib.json)")
        self.hw, self.calib, self.mode, self.dtype = hw, calib, mode, norm_dtype(dtype)
        self.is_projection = bool(mode == "empirical" and calib.hw_key != hw.key)
        if self.is_projection and cross_gpu is False:
            raise ValueError(f"calibration from {calib.hw_key} applied to {hw.key}: "
                             "pass cross_gpu=True (or use OpCostModel.projected) for a LOGO projection")
        self.family_map = dict(family_map or {})
        self.fallbacks: Dict[str, int] = {}   # op name -> times it fell back to plain SOL

    @classmethod
    def projected(cls, target_hw: Hardware, calib, **kw) -> "OpCostModel":
        """Leave-one-GPU-out projection: GPU A's efficiency curves on GPU B's spec sheet."""
        return cls(target_hw, calib, mode="empirical", cross_gpu=True, **kw)

    def describe(self) -> str:
        if self.mode == "sol":
            return f"SOL roofline on the {self.hw.key} spec sheet (no efficiency, no overhead)"
        s = (f"empirical: {self.hw.key} spec sheet x efficiency curves calibrated on "
             f"{self.calib.gpu_tag} ({self.calib.hw_key} spec sheet)")
        return s + (" -- LOGO PROJECTION" if self.is_projection else "")

    # -- op -> family ------------------------------------------------------------
    def family(self, op: Op, phase: Optional[str] = None) -> Tuple[str, str, bool]:
        """(family, dtype, is_proxy)."""
        dtype = norm_dtype(getattr(op, "dtype", None) or self.dtype)
        if op.name in self.family_map:
            m = self.family_map[op.name]
            fam, dtype = (m, dtype) if isinstance(m, str) else (m[0], norm_dtype(m[1]))
            return fam, dtype, False
        fam = getattr(op, "family", None)
        if fam:
            return fam, dtype, False
        if op.kind == "gemm":
            return "gemm", dtype, False
        if op.kind == "attention":
            ph = phase or getattr(op, "phase", None) or infer_attention_phase(op)
            if ph not in ("decode", "prefill"):
                raise ValueError(f"phase must be 'decode' or 'prefill', got {ph!r}")
            return f"attn_{ph}", dtype, False
        if "norm" in op.name:
            return "rmsnorm", dtype, False
        return "rmsnorm", dtype, True  # swiglu, embedding: memory-bound elementwise proxy

    def _curve(self, fam: str, dtype: str) -> Tuple[Optional[EfficiencyCurve], str]:
        c = self.calib.get(fam, dtype)
        if c is not None:
            return c, "measured"
        if fam == "gemm" and is_weight_only_quant(dtype):
            c = self.calib.get("gemv_q4", dtype) or self.calib.any_dtype("gemv_q4")
            if c is not None:
                return c, "proxy:gemv_q4"
        return None, "sol_fallback"

    # -- timing ------------------------------------------------------------------
    def sol_time(self, op: Op) -> float:
        dtype = norm_dtype(getattr(op, "dtype", None) or self.dtype)
        return sol_time(op.flops, op.bytes, self.hw, dtype)

    def explain(self, op: Op, batch=None, phase: Optional[str] = None) -> Dict:
        """Everything behind time(op): family, curve source, SOL, eff, t0."""
        fam, dtype, proxy = self.family(op, phase)
        sol = sol_time(op.flops, op.bytes, self.hw, dtype)
        out = {"name": op.name, "family": fam, "dtype": dtype, "sol_s": sol, "eff": 1.0,
               "t0_s": 0.0, "time_s": sol, "feature": None, "x": None, "source": "sol"}
        if self.mode == "sol" or op.bytes <= 0:
            return out
        curve, src = self._curve(fam, dtype)
        if curve is None:
            self.fallbacks[op.name] = self.fallbacks.get(op.name, 0) + 1
            out["source"] = "sol_fallback"
            return out
        x = feature_value(curve.feature, op.flops, op.bytes) if (op.flops > 0 or curve.feature == "bytes") else None
        if not x or x <= 0:  # e.g. a zero-FLOP lookup on an "ai"/"flops" curve
            self.fallbacks[op.name] = self.fallbacks.get(op.name, 0) + 1
            out["source"] = "sol_fallback"
            return out
        wb = op.weight_bytes if (curve.op == "gemm" and op.kind == "gemm") else None
        eff, where = curve.lookup(x, wb)
        if proxy:
            src = f"proxy:{fam}"
        out.update(eff=eff, t0_s=curve.t0_s, time_s=curve.t0_s + sol / eff, feature=curve.feature,
                   x=x, source=f"{src}/{where}" + ("/projection" if self.is_projection else ""))
        return out

    def time(self, op: Op, batch=None, phase: Optional[str] = None) -> float:
        """Seconds for ONE instance of `op`. `batch` is accepted for the CONTRACTS
        signature but unused: the Op's FLOPs and bytes already include the batch."""
        return self.explain(op, batch, phase)["time_s"]

    def total_time(self, op: Op, phase: Optional[str] = None) -> float:
        return self.time(op, phase=phase) * op.count

    def pass_time(self, ops: Iterable[Op], phase: Optional[str] = None) -> float:
        """Sum over ops of time x count (ops run back to back)."""
        return sum(self.total_time(o, phase) for o in ops)


# ----------------------------------------------------------------------------------
# Bench record <-> Op (for validation: predict a measured record through the model)
# ----------------------------------------------------------------------------------
def op_from_record(rec: dict) -> Tuple[Op, Optional[str]]:
    """(Op, phase) with the record's FLOPs and bytes (Op.bytes == rec["bytes"]) and
    op.family / op.dtype set, so OpCostModel uses the record's own curve."""
    s, name = rec.get("shape") or {}, rec["op"]
    fl, nb = float(rec["flops"]), float(rec["bytes"])
    phase = None
    if name == "gemm" and "M" in s:
        wb = _record_weight_bytes(rec) or 0.0
        op = Op(f"gemm_{s.get('proj', 'x')}", fl, weight_bytes=wb, act_bytes=nb - wb, kind="gemm")
    elif name in ("attn_decode", "attn_prefill"):
        phase = name.split("_", 1)[1]
        op = Op(name, fl, kv_read_bytes=nb, kind="attention")
    else:
        op = Op(name, fl, act_bytes=nb, kind="elementwise")
    op.family = name
    op.dtype = norm_dtype(rec["dtype"])
    return op, phase


def predict_record(cost: OpCostModel, rec: dict) -> float:
    """Model seconds for a bench record (in-sample check, or LOGO on another GPU)."""
    op, phase = op_from_record(rec)
    return cost.time(op, phase=phase)
