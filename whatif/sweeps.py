"""What-if projections for one GPU with the calibrated model. Nothing here is measured.

    python -m whatif.sweeps --gpu h100_sxm --data data --out results

Scenarios (decode TPOT / tok/s per batch at a fixed context, and batch-1 TTFT):
  1. HBM bandwidth x {0.5, 0.75, 1, 1.5, 2}
       "ideal":            datasheet BW scaled, efficiency curves unchanged
       "Little's-Law cap": the BW the GEMV can actually pull at the measured best
                           bytes-in-flight, predict_bw(peak * f) / predict_bw(peak) --
                           more BW only helps if enough bytes are in flight to use it.
  2. SM count -25 % / +25 %: compute peak scaled by s, and memory BW scaled by
       predict_bw(sm*s) / predict_bw(sm) (bytes in flight scale with the SM count).
  3. FP8 weights: every op's weight bytes halved (W8A16-style: compute precision unchanged).
  4. TP=2 over NVLink: every op's FLOPs / weight / KV / activation bytes split over 2 GPUs,
       plus 2 all-reduces per layer of an N x hidden bf16 message (roofline.allreduce_time);
       fixed per-step overhead is not split.

Scenarios 3-4 change op shapes, which the engine API does not expose, so they use
    t' = overhead + sum_ops time'(op),  overhead = engine_time - sum_ops time(op)
with OpCostModel.time on perfsim.models ops (the overhead is kept as is).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional

from correlate import loaders as L
from correlate.backend import Backend, calib_for, default_backend, gpu_params, params_fields, replace_params
from correlate.correlate import write_csv
from correlate.metrics import fmt

HBM_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
SM_SCALES = (0.75, 1.25)
DEFAULT_BATCHES = (1, 8, 32, 64)
DEFAULT_CONTEXT = 1024
DEFAULT_PROMPT = 1024
# Operating point when there is no GEMV sweep to take the best config from (assumption).
DEFAULT_OP_POINT = {"warps_per_sm": 32, "ilp": 2, "bytes_per_load": L.GEMV_BYTES_PER_LOAD}
PROJ = "Projection (not measured)"


def _sum_ops(backend: Backend, cost, ops, batch) -> float:
    return sum(backend.op_seconds(cost, o, batch) for o in ops)


def _hint(ops, dtype: str = "bf16"):
    """Same optional dtype hint perfsim.engine puts on ops (picks the efficiency curve)."""
    for o in ops:
        o.dtype = dtype
    return ops


def _scaled_ops(ops, w_scale: float = 1.0, shard: int = 1):
    """Copies of perfsim.models.Op with weight bytes x w_scale and everything / shard."""
    return _hint([dataclasses.replace(o, flops=o.flops / shard, weight_bytes=o.weight_bytes * w_scale / shard,
                                      kv_read_bytes=o.kv_read_bytes / shard, kv_write_bytes=o.kv_write_bytes / shard,
                                      act_bytes=o.act_bytes / shard) for o in ops])


def variant_time(backend: Backend, cost, ops, batch, engine_time: float,
                 w_scale: float = 1.0, shard: int = 1, extra_s: float = 0.0) -> float:
    """Engine time with op shapes changed; non-op overhead kept. If the engine's time is
    below the per-op sum (fusion/overlap), scale it by the per-op ratio instead."""
    base = _sum_ops(backend, cost, ops, batch)
    var = _sum_ops(backend, cost, _scaled_ops(ops, w_scale, shard), batch)
    if base <= 0:
        return engine_time + extra_s
    if base > engine_time:
        return engine_time * var / base + extra_s
    return (engine_time - base) + var + extra_s


def tp_comm_time(model, hw, n_tokens: int, P: int = 2, act_bytes: float = 2.0) -> float:
    """2 all-reduces per layer (after o_proj and mlp_down) of n_tokens x hidden activations."""
    from perfsim.roofline import allreduce_time
    if not hw.interconnect_bandwidth:
        raise ValueError(f"{hw.key}: no interconnect bandwidth in hw/hardware.yaml")
    msg = n_tokens * model.hidden * act_bytes
    return 2 * model.n_layers * allreduce_time(msg, P, hw.interconnect_bandwidth, hw.hop_latency)


class LittleLaw:
    """Achieved-BW ratios from microarch.predict_bw at this GPU's best measured GEMV config."""

    def __init__(self, backend: Backend, tag_dir: Path, notes: List[str]):
        self.ok = False
        if not (backend.bytes_in_flight and backend.predict_bw):
            notes.append("microarch not available: BW/SM scenarios use ideal scaling only")
            return
        params = gpu_params(backend, tag_dir, [])
        if params is None:
            notes.append("no env.json/probes for this GPU: BW/SM scenarios use ideal scaling only")
            return
        sweep = L.load_gemv_sweep(tag_dir)
        self.b, self.params, self.fields = backend, params, params_fields(params)
        self.alpha = None
        stream = [p for p in L.read_jsonl(Path(tag_dir) / "probes.jsonl")
                  if p.get("probe") == "stream" and (p.get("GBps") or 0) > 0]
        fit_on = stream if len(stream) >= 2 else sweep   # same choice as correlate.level1
        if len(fit_on) >= 2 and backend.fit_loaded_latency is not None:
            try:
                self.alpha = float(backend.fit_loaded_latency(self.params, fit_on))
            except Exception as e:
                notes.append(f"fit_loaded_latency failed ({e}); plain Little's Law used")
        if sweep:
            best = max(sweep, key=lambda r: r["GBps"])
            self.op = {"warps_per_sm": best.get("warps_per_sm_eff", best["warps_per_sm"]), "ilp": best["ilp"],
                       "bytes_per_load": best.get("bytes_per_load", L.GEMV_BYTES_PER_LOAD),
                       "kernel": best.get("kernel")}
        else:
            self.op = dict(DEFAULT_OP_POINT, kernel="assumed")
            notes.append(f"no gemv_sweep: Little's-Law operating point assumed {DEFAULT_OP_POINT}")
        self.ok = True
        self.base = self.bw(1.0, 1.0)

    def bw(self, peak_scale: float, sm_scale: float) -> float:
        sm = self.fields["sm_count"] * sm_scale
        p = replace_params(self.params, peak_bw_Bps=self.fields["peak_bw_Bps"] * peak_scale, sm_count=sm)
        bif = self.b.bytes_in_flight(self.op["warps_per_sm"], self.op["ilp"], self.op["bytes_per_load"], sm)
        if self.alpha is not None:
            return float(self.b.predict_bw(p, bif, loaded_latency=True, alpha=self.alpha))
        return float(self.b.predict_bw(p, bif))

    def ratio(self, peak_scale: float = 1.0, sm_scale: float = 1.0) -> float:
        return self.bw(peak_scale, sm_scale) / self.base


def run_whatif(gpu: str, data_root="data", out_dir="results", backend: Optional[Backend] = None,
               model_name: str = "qwen3-8b", batches=DEFAULT_BATCHES, context: int = DEFAULT_CONTEXT,
               prompt_len: int = DEFAULT_PROMPT, write: bool = True) -> Dict:
    from perfsim.hardware import load_hardware
    from perfsim.models import decode_ops, load_model, prefill_ops

    backend = backend or default_backend()
    notes: List[str] = list(backend.notes)
    res = {"gpu": gpu, "rows": [], "notes": notes, "mode": None, "context": context, "prompt_len": prompt_len,
           "batches": list(batches), "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    hw = load_hardware().get(gpu)
    if hw is None or not backend.has_engine:
        notes.append(f"what-if skipped: {'no hardware entry for ' + gpu if hw is None else 'engine/ops_model not available'}")
        if write:
            _write(res, Path(out_dir))
        return res
    model = load_model(model_name)
    tag_dir = Path(data_root) / gpu
    calib, mode = calib_for(backend, tag_dir, notes)
    res["mode"] = mode
    ll = LittleLaw(backend, tag_dir, notes)
    res["op_point"] = getattr(ll, "op", None)
    cost0 = backend.make_cost(hw, calib, mode)

    def measure(cost, label, knob, value, w_scale=1.0, shard=1, tp_hw=None):
        """Rows for one scenario point: tok/s + TPOT per batch, batch-1 TTFT."""
        for b in batches:
            t = backend.decode_step_time(model, cost, b, context, engine="vllm")
            if w_scale != 1.0 or shard != 1:
                ops = _hint(decode_ops(model, b, context, w_bits=16))
                extra = tp_comm_time(model, tp_hw, b, shard) if shard > 1 else 0.0
                t = variant_time(backend, cost, ops, b, t, w_scale, shard, extra)
            add(label, knob, value, b, "decode tok/s", b / t, "tok/s")
            add(label, knob, value, b, "decode TPOT", t * 1e3, "ms")
        t = backend.prefill_time(model, cost, 1, prompt_len)
        if w_scale != 1.0 or shard != 1:
            ops = _hint(prefill_ops(model, 1, prompt_len, w_bits=16))
            extra = tp_comm_time(model, tp_hw, prompt_len, shard) if shard > 1 else 0.0
            t = variant_time(backend, cost, ops, 1, t, w_scale, shard, extra)
        add(label, knob, value, 1, f"TTFT (prompt {prompt_len})", t * 1e3, "ms")

    baseline: Dict = {}

    def add(label, knob, value, b, metric, v, unit):
        key = (b, metric)
        if label == "baseline":
            baseline[key] = v
        base = baseline.get(key)
        res["rows"].append({"gpu": gpu, "scenario": label, "knob": knob, "value": value, "batch": b,
                            "metric": metric, "unit": unit, "baseline": base, "projected": v,
                            "ratio": (v / base) if base else None, "label": PROJ, "calib_mode": mode})

    measure(cost0, "baseline", "-", "-")
    for f in HBM_SCALES:
        measure(backend.make_cost(hw.scaled(bw=f), calib, mode), "HBM BW (ideal)", "BW x", f)
    if ll.ok:
        for f in HBM_SCALES:
            r = ll.ratio(peak_scale=f)
            measure(backend.make_cost(hw.scaled(bw=r), calib, mode), "HBM BW (Little's-Law cap)", "BW x", f)
    for s in SM_SCALES:
        r = ll.ratio(sm_scale=s) if ll.ok else 1.0
        measure(backend.make_cost(hw.scaled(flops=s, bw=r), calib, mode), "SM count", "SMs x", s)
    measure(cost0, "FP8 weights", "weight bytes x", 0.5, w_scale=0.5)
    try:
        measure(cost0, "TP=2 (NVLink)", "GPUs", 2, shard=2, tp_hw=hw)
        if "NVLink" not in str(hw.provenance.get("interconnect_bandwidth", {}).get("note", "")):
            notes.append(f"{gpu} has no NVLink entry: TP=2 uses its interconnect_bandwidth "
                         f"({hw.interconnect_bandwidth / 1e9:.0f} GB/s) from hw/hardware.yaml")
    except ValueError as e:
        notes.append(f"TP=2 skipped: {e}")
    if ll.ok:
        res["ll_ratios"] = {f"BW x{f}": ll.ratio(peak_scale=f) for f in HBM_SCALES}
        res["ll_ratios"].update({f"SMs x{s}": ll.ratio(sm_scale=s) for s in SM_SCALES})
    if write:
        _write(res, Path(out_dir))
    return res


def _write(res: Dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "whatif.csv", res["rows"])
    (out / "whatif.md").write_text(to_markdown(res))


def to_markdown(res: Dict) -> str:
    gpu = res["gpu"]
    s = [f"# What-if projections: {gpu}", "",
         f"Generated {res['generated']} by `whatif/sweeps.py`. **Every number below is a model projection, "
         f"not a measurement.** Cost model mode: {res['mode'] or '-'} (empirical = calibrated on this GPU's "
         f"ops.jsonl; sol = uncalibrated). Decode at context {res['context']} tokens (vLLM engine, bf16 weights); "
         f"TTFT = batch 1, prompt {res['prompt_len']}.", ""]
    rows = res["rows"]
    if not rows:
        s += ["_No projections (see Notes)._", ""]
    scen = list(dict.fromkeys(r["scenario"] for r in rows if r["scenario"] != "baseline"))
    batches = res["batches"]
    for sc in scen:
        rs = [r for r in rows if r["scenario"] == sc]
        values = list(dict.fromkeys(r["value"] for r in rs))
        knob = rs[0]["knob"]
        s += [f"### {PROJ}: {sc}", ""]
        head = [knob] + [f"tok/s b={b}" for b in batches] + [f"speedup b={batches[-1]}", "TTFT ms", "TTFT speedup"]
        tab = []
        for v in values:
            get = lambda b, m: next((r for r in rs if r["value"] == v and r["batch"] == b and r["metric"].startswith(m)), None)
            ttft = next((r for r in rs if r["value"] == v and r["metric"].startswith("TTFT")), None)
            last = get(batches[-1], "decode tok/s")
            tab.append([fmt(v, "{}")] + [fmt(get(b, "decode tok/s")["projected"], "{:.0f}") for b in batches]
                       + [fmt(last["ratio"], "{:.2f}x") if last else "-",
                          fmt(ttft["projected"], "{:.1f}") if ttft else "-",
                          fmt(1 / ttft["ratio"] if ttft and ttft["ratio"] else None, "{:.2f}x")])
        s += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        s += ["| " + " | ".join(t) + " |" for t in tab]
        s.append("")
    base = [r for r in rows if r["scenario"] == "baseline" and r["metric"] == "decode tok/s"]
    if base:
        s += [f"Baseline (model, not measured): " + ", ".join(f"b={r['batch']}: {r['projected']:.0f} tok/s" for r in base), ""]
    if res.get("ll_ratios"):
        op = res.get("op_point") or {}
        s += [f"Little's-Law achieved-BW ratios at the best measured GEMV config "
              f"(kernel {op.get('kernel')}, {op.get('warps_per_sm')} warps/SM, ILP {op.get('ilp')}): "
              + ", ".join(f"{k}: {v:.2f}" for k, v in res["ll_ratios"].items()), ""]
    s += ["## Notes and assumptions", "",
          "- HBM (ideal): efficiency curves unchanged when the datasheet BW changes.",
          "- SM count: compute peak x s; memory BW x predict_bw(sm*s)/predict_bw(sm) at fixed warps/SM and ILP.",
          "- FP8 weights: weight bytes halved; compute precision and KV cache unchanged; no dequant cost added.",
          "- TP=2: ops split evenly, KV heads divisible; 2 all-reduces/layer, alpha-beta model "
          "(perfsim.roofline.allreduce_time) with hw/hardware.yaml interconnect_bandwidth and hop latency; "
          "per-step overhead not split."]
    for n in dict.fromkeys(res["notes"]):
        s.append(f"- {n}")
    return "\n".join(s) + "\n"


def _default_gpu(data_root: Path) -> str:
    for d in L.gpu_dirs(data_root):
        if (d / "calib.json").exists():
            return d.name
    return "h100_sxm"


def main(argv=None):
    ap = argparse.ArgumentParser(description="What-if projections (not measured).")
    ap.add_argument("--gpu", default=None, help="gpu_tag (default: first data/<tag> with calib.json, else h100_sxm)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    ap.add_argument("--prompt", type=int, default=DEFAULT_PROMPT)
    ap.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    args = ap.parse_args(argv)
    gpu = args.gpu or _default_gpu(Path(args.data))
    res = run_whatif(gpu, args.data, args.out, batches=tuple(int(b) for b in args.batches.split(",")),
                     context=args.context, prompt_len=args.prompt)
    print((Path(args.out) / "whatif.md").read_text())
    return res


if __name__ == "__main__":
    main()
