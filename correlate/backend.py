"""Lazy binding to the model modules written in parallel (CONTRACTS.md interfaces).

correlate / whatif / agents never import perfsim.microarch, perfsim.ops_model or
perfsim.engine at module import time. `default_backend()` imports them when called and
leaves any missing function as None (with a note), so the report says
"Level N skipped: <module> not available" instead of crashing. Tests build a Backend
from stub functions instead.

Interfaces used (exactly the contract signatures):
  microarch.GPUParams(sm_count=, dram_latency_ns=, peak_bw_Bps=, l2_bytes=)
  microarch.bytes_in_flight(warps_per_sm, ilp, bytes_per_load, sm_count) -> bytes
  microarch.predict_bw(params, bytes_in_flight, loaded_latency=False, alpha=None) -> B/s
  microarch.fit_loaded_latency(params, sweep_records) -> alpha
  ops_model.OpCostModel(hw, calib=None, mode="sol"|"empirical").time(op, batch) -> s
  engine.decode_step_time(model, cost, batch, context_len, engine=, passes_per_step=) -> s
  engine.prefill_time(model, cost, batch, input_len) -> s
  engine.predict_vllm_latency(model, cost, batch, input_len, output_len) -> s
Optional extras, used when the module provides them (else a local equivalent is used):
  microarch.params_from_probes / load_probes, microarch.record_bytes_in_flight
  engine.op_seconds (cost-model calling convention incl. op.count),
  engine.decode_step_breakdown / prefill_breakdown (by_op + overhead_s, for attribution),
  engine.predict_vllm_tpot
  ops_model.load_calib, predict_record (Level 2 model vs measured), sol_time, fit_calibration
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Op kind (perfsim.models.Op.kind) -> attribution group
OP_GROUPS = {
    "gemm": "weights",        # qkv/o/gate_up/down/lm_head: GEMV at small batch, GEMM otherwise
    "attention": "attention",
    "elementwise": "norms",   # RMSNorms, qk_norm, SwiGLU
    "lookup": "norms",        # embedding row gather (tiny); lumped with the elementwise ops
}
GROUPS = ("weights", "attention", "norms", "overhead")


@dataclass
class Backend:
    name: str = "perfsim"
    # Level 1 (microarch)
    make_gpu_params: Optional[Callable[[Dict], Any]] = None
    bytes_in_flight: Optional[Callable] = None
    predict_bw: Optional[Callable] = None
    fit_loaded_latency: Optional[Callable] = None
    params_from_dir: Optional[Callable[[Path], Any]] = None        # tag_dir -> GPUParams
    record_bytes_in_flight: Optional[Callable] = None              # (record, params) -> bytes
    # Level 2/3 (ops_model + engine)
    make_cost: Optional[Callable[[Any, Any, str], Any]] = None   # (hw, calib, mode) -> cost
    load_calib: Optional[Callable[[Path], Any]] = None
    decode_step_time: Optional[Callable] = None
    prefill_time: Optional[Callable] = None
    predict_vllm_latency: Optional[Callable] = None
    predict_vllm_tpot: Optional[Callable] = None
    decode_breakdown: Optional[Callable] = None    # -> {"by_op": {name: s}, "overhead_s": s, "total_s": s}
    prefill_breakdown: Optional[Callable] = None
    op_seconds_fn: Optional[Callable] = None        # (cost, op, batch) -> s for all op.count instances
    predict_record: Optional[Callable] = None       # (cost, ops.jsonl record) -> s  (Level 2 check)
    sol_time: Optional[Callable] = None             # (flops, bytes, hw, dtype) -> s  (Level 2 SOL)
    fit_calibration: Optional[Callable] = None      # (records, hw, gpu_tag) -> object with .save(path)
    notes: List[str] = field(default_factory=list)

    def op_seconds(self, cost, op, batch) -> float:
        """Seconds for all op.count instances of a perfsim.models.Op in one pass. Default
        convention (same as roofline.op_time and engine.op_seconds): cost.time is per instance."""
        if self.op_seconds_fn is not None:
            return float(self.op_seconds_fn(cost, op, batch))
        return float(cost.time(op, batch)) * op.count

    def tpot(self, model, cost, batch, input_len, output_len) -> float:
        """Predicted TPOT with the same formula as the measurement."""
        if self.predict_vllm_tpot is not None:
            return float(self.predict_vllm_tpot(model, cost, batch, input_len, output_len))
        P = self.predict_vllm_latency
        return (P(model, cost, batch, input_len, output_len) - P(model, cost, batch, input_len, 1)) / (output_len - 1)

    @property
    def has_level1(self) -> bool:
        return all((self.make_gpu_params, self.bytes_in_flight, self.predict_bw))

    @property
    def has_engine(self) -> bool:
        return all((self.make_cost, self.decode_step_time, self.prefill_time, self.predict_vllm_latency))


def _try_import(name: str, notes: List[str]):
    try:
        return importlib.import_module(name)
    except Exception as e:  # ImportError, or an error inside a half-written module
        notes.append(f"{name} not available ({type(e).__name__}: {e}); dependent levels skipped")
        return None


def default_backend() -> Backend:
    """Bind to the real perfsim modules, whatever subset of them exists."""
    notes: List[str] = []
    b = Backend(name="perfsim", notes=notes)
    ma = _try_import("perfsim.microarch", notes)
    if ma is not None:
        GP = getattr(ma, "GPUParams", None)
        if GP is not None:
            def make_gpu_params(fields: Dict, _GP=GP):
                return _GP(**fields)
            b.make_gpu_params = make_gpu_params
        b.bytes_in_flight = getattr(ma, "bytes_in_flight", None)
        b.predict_bw = getattr(ma, "predict_bw", None)
        b.fit_loaded_latency = getattr(ma, "fit_loaded_latency", None)
        b.record_bytes_in_flight = getattr(ma, "record_bytes_in_flight", None)
        if hasattr(ma, "params_from_probes") and hasattr(ma, "load_probes"):
            def params_from_dir(tag_dir: Path, _ma=ma):
                env = {}
                try:
                    env = json.loads((Path(tag_dir) / "env.json").read_text())
                except (OSError, ValueError):
                    pass
                return _ma.params_from_probes(_ma.load_probes(tag_dir), sm_count=env.get("sm_count"),
                                              l2_bytes=env.get("l2_bytes"), name=env.get("gpu_name", ""))
            b.params_from_dir = params_from_dir
    om = _try_import("perfsim.ops_model", notes)
    if om is not None and hasattr(om, "OpCostModel"):
        def make_cost(hw, calib=None, mode="sol", _C=om.OpCostModel):
            return _C(hw, calib=calib, mode=mode)
        b.make_cost = make_cost

        def load_calib(path: Path, _om=om):
            # Prefer the module's own loader; otherwise hand OpCostModel the parsed JSON.
            for fn in ("load_calib", "load_calibration"):
                f = getattr(_om, fn, None)
                if callable(f):
                    return f(path)
            return json.loads(Path(path).read_text())
        b.load_calib = load_calib
        b.predict_record = getattr(om, "predict_record", None)
        b.sol_time = getattr(om, "sol_time", None)
        b.fit_calibration = getattr(om, "fit_calibration", None)
    en = _try_import("perfsim.engine", notes)
    if en is not None:
        b.decode_step_time = getattr(en, "decode_step_time", None)
        b.prefill_time = getattr(en, "prefill_time", None)
        b.predict_vllm_latency = getattr(en, "predict_vllm_latency", None)
        b.predict_vllm_tpot = getattr(en, "predict_vllm_tpot", None)
        b.decode_breakdown = getattr(en, "decode_step_breakdown", None)
        b.prefill_breakdown = getattr(en, "prefill_breakdown", None)
        b.op_seconds_fn = getattr(en, "op_seconds", None)
    return b


def gpu_params(backend: Backend, tag_dir: Path, notes: List[str]):
    """GPUParams for Level 1 from this GPU's own probes: the microarch loader if present,
    else env.json + probes.jsonl read here (loaders.measured_gpu_fields). None if unusable."""
    from . import loaders as L
    if backend.params_from_dir is not None:
        try:
            return backend.params_from_dir(Path(tag_dir))
        except Exception as e:
            notes.append(f"params_from_probes failed ({type(e).__name__}: {e}); using local probe reader")
    fields = L.measured_gpu_fields(tag_dir, notes)
    if any(fields[k] is None for k in ("sm_count", "dram_latency_ns", "peak_bw_Bps")) or backend.make_gpu_params is None:
        notes.append(f"Level 1 params unavailable (need env.json sm_count + pointer_chase + stream probes; got {fields})")
        return None
    return backend.make_gpu_params(fields)


def params_fields(params) -> Dict:
    return {k: getattr(params, k, None) for k in ("sm_count", "dram_latency_ns", "peak_bw_Bps", "l2_bytes")}


def replace_params(params, **changes):
    """Copy of a GPUParams with some fields changed (dataclass or plain object)."""
    if dataclasses.is_dataclass(params) and not isinstance(params, type):
        return dataclasses.replace(params, **changes)
    p = copy.copy(params)
    for k, v in changes.items():
        setattr(p, k, v)
    return p


def calib_for(backend: Backend, tag_dir: Path, notes: List[str]):
    """(calib, mode): ("empirical" with the loaded calib.json) or (None, "sol")."""
    path = Path(tag_dir) / "calib.json"
    if not path.exists():
        notes.append(f"{path} missing: Level 3 uses SOL mode (uncalibrated); run bench/fit_calib.py")
        return None, "sol"
    if backend.load_calib is None:
        notes.append("no calib loader available: SOL mode")
        return None, "sol"
    try:
        return backend.load_calib(path), "empirical"
    except Exception as e:
        notes.append(f"{path}: failed to load ({type(e).__name__}: {e}); SOL mode")
        return None, "sol"


# ---------------------------------------------------------------------------------
# Per-op attribution: share of the predicted time per op group
# ---------------------------------------------------------------------------------
def ops_time(backend: Backend, cost, ops, batch) -> Dict[str, float]:
    """Seconds per op group summed over one forward pass: time(op) * op.count."""
    out = {g: 0.0 for g in GROUPS if g != "overhead"}
    for op in ops:
        out[OP_GROUPS.get(op.kind, "norms")] += backend.op_seconds(cost, op, batch)
    return out


# perfsim.models.layer_ops op names -> group (for engine breakdowns keyed by op name)
NAME_GROUPS = {"qkv_proj": "weights", "o_proj": "weights", "mlp_gate_up": "weights", "mlp_down": "weights",
               "lm_head": "weights", "attention": "attention"}


def breakdown_shares(bd: Dict) -> Dict[str, float]:
    """Group shares from an engine breakdown {"by_op": {name: s}, "overhead_s": s}."""
    g = {k: 0.0 for k in GROUPS}
    for name, t in bd["by_op"].items():
        g[NAME_GROUPS.get(name, "norms")] += t
    g["overhead"] += bd.get("overhead_s", 0.0)
    tot = sum(g.values())
    return {k: (v / tot if tot > 0 else float("nan")) for k, v in g.items()}


def attribution(backend: Backend, cost, ops, batch, total_s: float,
                notes: Optional[List[str]] = None) -> Dict[str, float]:
    """Shares (0..1) of the predicted engine time per group; overhead = the part of the
    engine's time not explained by the per-op costs (launch, scheduling, sampling...).

    If the per-op sum exceeds the engine time (the engine may fuse or overlap ops),
    overhead is 0 and the op shares are normalised to sum to 1.
    """
    g = ops_time(backend, cost, ops, batch)
    s = sum(g.values())
    if total_s is None or total_s <= 0 or s <= 0:
        return {k: float("nan") for k in GROUPS}
    if s > total_s * 1.05:
        if notes is not None:
            notes.append("per-op times exceed engine step time (fusion/overlap in engine); overhead share set to 0")
        shares = {k: v / s for k, v in g.items()}
        shares["overhead"] = 0.0
        return shares
    shares = {k: v / total_s for k, v in g.items()}
    shares["overhead"] = max(0.0, 1.0 - sum(shares.values()))
    return shares
