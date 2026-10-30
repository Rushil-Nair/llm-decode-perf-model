"""Correlate the three-level model against measured data, per GPU.

    python -m correlate.correlate --data data --out results

For every gpu_tag directory in data/:
  (a) Level 1  GEMV sweep: predicted vs measured achieved bandwidth, plain Little's Law
               and with the loaded-latency (queueing) correction; MAPE before/after.
  (b) Level 2  % of speed-of-light (SOL) per op from ops.jsonl.
  (c) Level 3  vLLM TPOT per batch / TTFT per input_len, llama.cpp slot tok/s
               master vs fix and the fix speedup; MAPE, max |% err|, Pearson r.
  (d) LOGO     calibrate on GPU A (A's calib.json), predict GPU B from B's spec
               sheet (hw/hardware.yaml) only. Reported separately, labelled projection.
  (e) Attribution: share of the predicted time per op group (weights GEMV/GEMM,
               attention, norms, overhead) next to each Level 3 error.

Writes results/correlation.md (tables first) and results/*.csv. Missing files or
missing model modules skip that part with a note; the run never crashes on data.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import statistics
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from . import loaders as L
from .backend import (GROUPS, Backend, attribution, breakdown_shares, calib_for, default_backend,
                      gpu_params, params_fields)
from .metrics import fmt, pct_error, summarize

DEFAULT_MODEL = "qwen3-8b"
LLAMACPP_CONTEXT = 512        # assumed mean context in the slot test (prompt lengths not logged)
OCT8_YAML_NAME = "llamacpp_rtx5090_qwen3_8b.yaml"

KIND_CAL = "calibrated (own GPU)"
KIND_SOL = "SOL (uncalibrated)"
KIND_LOGO = "projection (LOGO)"


def _kind(mode: str) -> str:
    return KIND_CAL if mode == "empirical" else KIND_SOL


# =================================================================================
# (a) Level 1: Little's Law on the GEMV sweep
# =================================================================================
def level1(tag: str, tag_dir: Path, backend: Backend, notes: List[str]) -> Dict:
    """Predicted vs measured GEMV bandwidth. alpha (loaded-latency) is fitted on the stream
    probe when it has >= 2 records, so the GEMV prediction is out-of-sample; otherwise on the
    GEMV sweep itself (in-sample, labelled as such)."""
    out = {"rows": [], "summary": [], "params": None, "alpha": None, "alpha_source": None}
    recs = L.load_gemv_sweep(tag_dir, notes)
    if not recs:
        return out
    if not (backend.bytes_in_flight and backend.predict_bw):
        notes.append(f"{tag}: Level 1 skipped (perfsim.microarch functions not available)")
        return out
    params = gpu_params(backend, tag_dir, notes)
    if params is None:
        notes.append(f"{tag}: Level 1 skipped (no GPU parameters)")
        return out
    fields = params_fields(params)
    out["params"] = fields
    alpha, src = None, None
    if backend.fit_loaded_latency is not None:
        stream = [p for p in L.read_jsonl(Path(tag_dir) / "probes.jsonl")
                  if p.get("probe") == "stream" and (p.get("GBps") or 0) > 0]
        for cand, label in ((stream, "stream probe (GEMV out-of-sample)"), (recs, "GEMV sweep (in-sample)")):
            if len(cand) < 2:
                continue
            try:
                alpha, src = float(backend.fit_loaded_latency(params, cand)), label
                break
            except Exception as e:
                notes.append(f"{tag}: fit_loaded_latency on {label} failed ({type(e).__name__}: {e})")
    out["alpha"], out["alpha_source"] = alpha, src
    rows = []
    for r in recs:
        try:
            if backend.record_bytes_in_flight is not None:
                bif = float(backend.record_bytes_in_flight(r, params))
            else:
                warps = r.get("warps_per_sm_eff", r["warps_per_sm"])
                bif = float(backend.bytes_in_flight(warps, r["ilp"], r.get("bytes_per_load", L.GEMV_BYTES_PER_LOAD),
                                                    fields["sm_count"]))
            p0 = float(backend.predict_bw(params, bif)) / 1e9
            p1 = (float(backend.predict_bw(params, bif, loaded_latency=True, alpha=alpha)) / 1e9
                  if alpha is not None else None)
        except Exception as e:
            notes.append(f"{tag}: Level 1 record skipped ({type(e).__name__}: {e})")
            continue
        meas = float(r["GBps"])
        rows.append({
            "gpu": tag, "kernel": r.get("kernel"), "ilp": r.get("ilp"),
            "blocks_per_sm": r.get("blocks_per_sm"), "warps_per_sm": r.get("warps_per_sm"),
            "N": r.get("N"), "K": r.get("K"), "bytes_in_flight": bif,
            "measured_GBps": meas, "pred_plain_GBps": p0, "pred_loaded_GBps": p1,
            "err_plain_pct": pct_error(p0, meas),
            "err_loaded_pct": pct_error(p1, meas) if p1 is not None else None,
            "peak_stream_GBps": fields["peak_bw_Bps"] / 1e9 if fields["peak_bw_Bps"] else None,
        })
    out["rows"] = rows
    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append(r)
    for k, rs in sorted(by_kernel.items(), key=lambda kv: str(kv[0])) + [("all", rows)]:
        s0 = summarize([r["pred_plain_GBps"] for r in rs], [r["measured_GBps"] for r in rs])
        s1 = summarize([r["pred_loaded_GBps"] for r in rs], [r["measured_GBps"] for r in rs])
        out["summary"].append({
            "gpu": tag, "kernel": k, "n": s0["n"],
            "mape_plain": s0["mape"], "mape_loaded": s1["mape"],
            "max_abs_plain": s0["max_abs_pct"], "max_abs_loaded": s1["max_abs_pct"],
            "r_plain": s0["pearson_r"], "r_loaded": s1["pearson_r"], "alpha": alpha, "alpha_source": src,
        })
    return out


def level1_error_groups(rows: List[Dict]) -> List[Dict]:
    """One error record per (gpu, kernel, ilp), for flagging (median BW, mean signed error).
    Uses the loaded-latency prediction when present, else plain Little's Law."""
    grp = defaultdict(list)
    for r in rows:
        grp[(r["gpu"], r["kernel"], r["ilp"])].append(r)
    out = []
    for (gpu, kern, ilp), rs in sorted(grp.items(), key=lambda kv: tuple(map(str, kv[0]))):
        loaded = all(r["pred_loaded_GBps"] is not None for r in rs)
        pk, ek = ("pred_loaded_GBps", "err_loaded_pct") if loaded else ("pred_plain_GBps", "err_plain_pct")
        errs = [r[ek] for r in rs if r[ek] is not None and math.isfinite(r[ek])]
        if not errs:
            continue
        out.append({
            "level": "L1", "gpu": gpu, "kind": "Little's Law" + (" + loaded latency" if loaded else ""),
            "item": f"gemv {kern} ilp={ilp}", "config": f"kernel={kern} ilp={ilp} n={len(rs)}",
            "kernel": kern, "ilp": ilp,
            "bif_min": min(r["bytes_in_flight"] for r in rs), "bif_max": max(r["bytes_in_flight"] for r in rs),
            "measured": statistics.median(r["measured_GBps"] for r in rs),
            "predicted": statistics.median(r[pk] for r in rs),
            "unit": "GB/s", "err_pct": sum(errs) / len(errs),
        })
    return out


# =================================================================================
# (b) Level 2: % of SOL per op
# =================================================================================
def dtype_precision(dtype: str, hw) -> str:
    """Peak-FLOP/s key in hardware.yaml for an ops.jsonl dtype string."""
    d = (dtype or "").lower()
    if "fp8" in d or "e4m3" in d or "e5m2" in d:
        p = "fp8"
    elif "bf16" in d:
        p = "bf16"
    elif "fp32" in d or "f32" in d or "tf32" in d:
        p = "fp32"
    elif "int8" in d or d == "i8":
        p = "int8"
    else:  # fp16, half, and 4-bit weight formats (dequantized to fp16 for the math)
        p = "fp16"
    return p if p in hw.peak_flops else "fp16"


def level2(tag: str, tag_dir: Path, hw, notes: List[str], backend: Optional[Backend] = None,
           cost=None, kind: str = "", calib_from: str = "") -> Dict:
    """% of SOL per measured op; plus the Level 2 model (OpCostModel) prediction of each
    record when a cost model is given (in-sample with own calib, projection with another's)."""
    out = {"rows": [], "summary": []}
    recs = L.load_ops(tag_dir, notes if kind != KIND_LOGO else None)
    if not recs:
        return out
    if hw is None:
        notes.append(f"{tag}: Level 2 skipped (no hw/hardware.yaml entry for '{tag}')")
        return out
    stream = L.measured_gpu_fields(Path(tag_dir)).get("peak_bw_Bps")
    rows = []
    for r in recs:
        prec = dtype_precision(r.get("dtype"), hw)
        flops, nbytes, t = float(r.get("flops") or 0), float(r.get("bytes") or 0), float(r["us"]) * 1e-6
        tc, tm = flops / hw.peak(prec), nbytes / hw.memory_bandwidth
        sol = max(tc, tm)
        if backend is not None and backend.sol_time is not None:
            try:  # the Level 2 module's own SOL (same dtype -> peak mapping as the model)
                sol = float(backend.sol_time(flops, nbytes, hw, r.get("dtype")))
            except Exception:
                pass
        model_us = None
        if cost is not None and backend is not None and backend.predict_record is not None:
            try:
                model_us = float(backend.predict_record(cost, r)) * 1e6
            except Exception as e:
                notes.append(f"{tag}: Level 2 model failed on {r.get('op')} ({type(e).__name__}: {e})")
        rows.append({
            "gpu": tag, "kind": kind, "calib_from": calib_from, "op": r.get("op"), "dtype": r.get("dtype"),
            "shape": json.dumps(r.get("shape", {}), sort_keys=True, separators=(",", ":")),
            "measured_us": float(r["us"]), "sol_us": sol * 1e6, "bound": "compute" if tc >= tm else "memory",
            "pct_sol": 100.0 * sol / t if t > 0 else None,
            # memory-bound ops against the measured stream max instead of the datasheet BW
            "pct_stream_bw": (100.0 * (nbytes / stream) / t) if (stream and t > 0 and tm >= tc) else None,
            "achieved_TFLOPs": flops / t / 1e12 if t > 0 else None,
            "achieved_GBps": nbytes / t / 1e9 if t > 0 else None,
            "precision_peak": prec, "model_us": model_us,
            "model_err_pct": pct_error(model_us, float(r["us"])) if model_us is not None else None,
        })
    out["rows"] = rows
    grp = defaultdict(list)
    for r in rows:
        grp[(r["op"], r["dtype"])].append(r)
    for (op, dt), rs in sorted(grp.items(), key=lambda kv: tuple(map(str, kv[0]))):
        v = [r["pct_sol"] for r in rs if r["pct_sol"] is not None]
        bounds = [r["bound"] for r in rs]
        m = summarize([r["model_us"] for r in rs], [r["measured_us"] for r in rs])
        out["summary"].append({
            "gpu": tag, "kind": kind, "calib_from": calib_from, "op": op, "dtype": dt, "n": len(rs),
            "median_pct_sol": statistics.median(v) if v else None,
            "min_pct_sol": min(v) if v else None, "max_pct_sol": max(v) if v else None,
            "mostly": max(set(bounds), key=bounds.count),
            "model_mape": m["mape"] if m["n"] else None, "model_max_abs": m["max_abs_pct"] if m["n"] else None,
        })
    return out


# =================================================================================
# (c)/(d)/(e) Level 3: vLLM TPOT/TTFT and llama.cpp slots
# =================================================================================
def predict_vllm(tag: str, tpot: List[Dict], ttft: List[Dict], backend: Backend, model, cost,
                 kind: str, calib_from: str, notes: List[str]) -> List[Dict]:
    """Predicted vs measured TPOT/TTFT rows with per-group attribution of the prediction."""
    from perfsim.models import decode_ops, prefill_ops

    rows = []
    P = backend.predict_vllm_latency
    for t in tpot:
        b, il, ol = t["batch"], t["input_len"], t["output_len"]
        try:
            pred = backend.tpot(model, cost, b, il, ol)
            ctx = il + (ol - 2) // 2   # mean context over decode steps at contexts il .. il+ol-2
            if backend.decode_breakdown is not None:
                sh = breakdown_shares(backend.decode_breakdown(model, cost, b, ctx, "vllm"))
            else:
                step = backend.decode_step_time(model, cost, b, ctx, engine="vllm")
                sh = attribution(backend, cost, decode_ops(model, b, ctx, w_bits=16), b, step, notes)
        except Exception as e:
            notes.append(f"{tag}: TPOT prediction failed b={b} in={il} ({type(e).__name__}: {e})")
            continue
        rows.append(_l3_row(tag, kind, calib_from, "TPOT", b, il, ol, ctx, t["tpot_s"], pred, sh))
    for t in ttft:
        b, il = t["batch"], t["input_len"]
        try:
            pred = P(model, cost, b, il, 1)
            if backend.prefill_breakdown is not None:
                sh = breakdown_shares(backend.prefill_breakdown(model, cost, b, il))
            else:
                pf = backend.prefill_time(model, cost, b, il)
                sh = attribution(backend, cost, prefill_ops(model, b, il, w_bits=16), b, pf, notes)
        except Exception as e:
            notes.append(f"{tag}: TTFT prediction failed b={b} in={il} ({type(e).__name__}: {e})")
            continue
        rows.append(_l3_row(tag, kind, calib_from, "TTFT", b, il, 1, il, t["ttft_s"], pred, sh))
    return rows


def _l3_row(gpu, kind, calib_from, metric, b, il, ol, ctx, meas_s, pred_s, shares) -> Dict:
    r = {"gpu": gpu, "kind": kind, "calib_from": calib_from, "metric": metric,
         "batch": b, "input_len": il, "output_len": ol, "context_len": ctx,
         "measured_ms": meas_s * 1e3, "predicted_ms": pred_s * 1e3,
         "err_pct": pct_error(pred_s, meas_s)}
    for g in GROUPS:
        r[f"share_{g}"] = shares.get(g)
    return r


def predict_slots(tag: str, slots: List[Dict], backend: Backend, model, cost, kind: str,
                  calib_from: str, notes: List[str], context: int = LLAMACPP_CONTEXT):
    """Slot rows (tok/s) and fix/master speedup rows per user count."""
    rows = []
    for s in slots:
        if s["build"] not in ("master", "fix") or s["policy"].endswith("kvu"):
            continue  # --kv-unified run is a reference point, not part of master vs fix
        k, pred = s["passes_per_step"], None
        if k is not None:
            try:
                t = backend.decode_step_time(model, cost, s["users"], context, engine="llamacpp",
                                             passes_per_step=k)
                pred = s["users"] / t
            except Exception as e:
                notes.append(f"{tag}: slot prediction failed {s['policy']} u={s['users']} ({type(e).__name__}: {e})")
        rows.append({"gpu": tag, "kind": kind, "calib_from": calib_from, "build": s["build"],
                     "users": s["users"], "passes_per_step": k, "k_source": s["k_source"],
                     "context_len": context, "measured_tok_s": s["tok_s"], "predicted_tok_s": pred,
                     "err_pct": pct_error(pred, s["tok_s"]) if pred else None, "source": s["source"]})
    sp = []
    by_u = defaultdict(dict)
    for r in rows:
        by_u[r["users"]][r["build"]] = r
    for u, d in sorted(by_u.items()):
        if "master" not in d or "fix" not in d:
            continue
        m, f = d["master"], d["fix"]
        ms = f["measured_tok_s"] / m["measured_tok_s"]
        ps = (f["predicted_tok_s"] / m["predicted_tok_s"]) if (f["predicted_tok_s"] and m["predicted_tok_s"]) else None
        sp.append({"gpu": tag, "kind": kind, "calib_from": calib_from, "users": u,
                   "master_tok_s": m["measured_tok_s"], "fix_tok_s": f["measured_tok_s"],
                   "k_master": m["passes_per_step"], "k_fix": f["passes_per_step"],
                   "measured_speedup": ms, "predicted_speedup": ps,
                   "err_pct": pct_error(ps, ms) if ps else None})
    return rows, sp


def _metric_rows(gpu, kind, calib_from, quantity, pred, meas) -> Dict:
    s = summarize(pred, meas)
    return {"gpu": gpu, "kind": kind, "calib_from": calib_from, "quantity": quantity, **s}


def _l3_metrics(vrows, srows, sprows) -> List[Dict]:
    out = []
    keys = sorted({(r["gpu"], r["kind"], r["calib_from"]) for r in vrows + srows + sprows})
    for gpu, kind, cf in keys:
        sel = lambda rows: [r for r in rows if (r["gpu"], r["kind"], r["calib_from"]) == (gpu, kind, cf)]
        for metric in ("TPOT", "TTFT"):
            rs = [r for r in sel(vrows) if r["metric"] == metric]
            if rs:
                out.append(_metric_rows(gpu, kind, cf, f"vLLM {metric} (ms)",
                                        [r["predicted_ms"] for r in rs], [r["measured_ms"] for r in rs]))
        rs = [r for r in sel(srows) if r["predicted_tok_s"] is not None]
        if rs:
            out.append(_metric_rows(gpu, kind, cf, "llama.cpp slot tok/s",
                                    [r["predicted_tok_s"] for r in rs], [r["measured_tok_s"] for r in rs]))
        rs = [r for r in sel(sprows) if r["predicted_speedup"] is not None]
        if rs:
            out.append(_metric_rows(gpu, kind, cf, "llama.cpp fix/master speedup",
                                    [r["predicted_speedup"] for r in rs], [r["measured_speedup"] for r in rs]))
    return out


# =================================================================================
# driver
# =================================================================================
def _hw_for(tag: str, notes: List[str]):
    from perfsim.hardware import load_hardware
    hw = load_hardware().get(tag)
    if hw is None:
        notes.append(f"{tag}: no hw/hardware.yaml entry; Levels 2/3 need one (skipped)")
    return hw


def run(data_root, out_dir=None, backend: Optional[Backend] = None, model_name: str = DEFAULT_MODEL,
        write: bool = True, oct8_yaml: Optional[Path] = None) -> Dict:
    """Run the whole correlation. Returns the result dict (also written to out_dir)."""
    from perfsim.models import load_model

    data_root = Path(data_root)
    backend = backend or default_backend()
    oct8_yaml = Path(oct8_yaml) if oct8_yaml else data_root / OCT8_YAML_NAME
    res: Dict = {"generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "data_root": str(data_root), "backend": backend.name, "model": model_name,
                 "notes": list(backend.notes), "gpu_notes": {}, "gpus": [],
                 "level1": [], "level1_summary": [], "level1_params": {}, "level2": [], "level2_summary": [], "level2_logo": [],
                 "vllm": [], "slots": [], "slot_speedup": [], "metrics": [], "errors": []}
    model = load_model(model_name)
    dirs = L.gpu_dirs(data_root)
    if oct8_yaml.exists() and "rtx5090" not in [d.name for d in dirs]:
        # the owner's Oct-8 RTX 5090 slot log is measured data even without a data/rtx5090/ run
        dirs.append(data_root / "rtx5090")
    if not dirs:
        res["notes"].append(f"no gpu_tag directories with data in {data_root}; nothing to correlate")
    measured: Dict[str, Dict] = {}   # tag -> {tpot, ttft, slots, hw, calib, mode}

    for d in dirs:
        tag = d.name
        notes: List[str] = []
        res["gpus"].append(tag)
        res["gpu_notes"][tag] = notes
        try:
            l1 = level1(tag, d, backend, notes)
            res["level1"] += l1["rows"]
            res["level1_summary"] += l1["summary"]
            if l1["params"]:
                res["level1_params"][tag] = {**l1["params"], "alpha": l1["alpha"], "alpha_source": l1["alpha_source"]}
        except Exception as e:
            notes.append(f"Level 1 crashed: {type(e).__name__}: {e}")
            notes.append(traceback.format_exc(limit=2))
        hw = None
        try:
            hw = _hw_for(tag, notes)
        except Exception as e:
            notes.append(f"hardware lookup failed: {type(e).__name__}: {e}")
        # Level 3 inputs
        tpot, ttft, slots = L.load_level3(d, notes)
        if not slots and tag == "rtx5090" and oct8_yaml.exists():
            slots = L.load_rtx5090_october_slots(oct8_yaml)
            notes.append(f"llama.cpp slot data for rtx5090 taken from {oct8_yaml} (Oct 8 measured log)")
        has_ops = (d / "ops.jsonl").exists()
        calib, mode = calib_for(backend, d, notes) if (tpot or ttft or slots or has_ops) else (None, "sol")
        measured[tag] = {"tpot": tpot, "ttft": ttft, "slots": slots, "hw": hw, "calib": calib, "mode": mode,
                         "dir": d, "has_ops": has_ops}
        try:
            cost2 = None
            if mode == "empirical" and hw is not None and backend.make_cost is not None:
                cost2 = backend.make_cost(hw, calib, mode)
            l2 = level2(tag, d, hw, notes, backend, cost2, _kind(mode), tag)
            res["level2"] += l2["rows"]
            res["level2_summary"] += l2["summary"]
        except Exception as e:
            notes.append(f"Level 2 crashed: {type(e).__name__}: {e}")
        if not (tpot or ttft or slots):
            continue
        if hw is None or not backend.has_engine:
            if not backend.has_engine:
                notes.append(f"{tag}: Level 3 skipped (perfsim.ops_model/engine not available)")
            continue
        try:
            cost = backend.make_cost(hw, calib, mode)
            res["vllm"] += predict_vllm(tag, tpot, ttft, backend, model, cost, _kind(mode), tag, notes)
            s, sp = predict_slots(tag, slots, backend, model, cost, _kind(mode), tag, notes)
            res["slots"] += s
            res["slot_speedup"] += sp
        except Exception as e:
            notes.append(f"Level 3 crashed: {type(e).__name__}: {e}")

    # (d) leave-one-GPU-out: calibrate on A, predict B from B's spec sheet only
    if backend.has_engine and len(measured) >= 2:
        for a, ma in measured.items():
            if ma["mode"] != "empirical":
                continue
            for b, mb in measured.items():
                if a == b or mb["hw"] is None or not (mb["tpot"] or mb["ttft"] or mb["slots"] or mb["has_ops"]):
                    continue
                notes = res["gpu_notes"][b]
                try:
                    cost = backend.make_cost(mb["hw"], ma["calib"], "empirical")
                    l2 = level2(b, mb["dir"], mb["hw"], notes, backend, cost, KIND_LOGO, a)
                    if any(r["model_us"] is not None for r in l2["rows"]):
                        res["level2_logo"] += l2["summary"]
                    res["vllm"] += predict_vllm(b, mb["tpot"], mb["ttft"], backend, model, cost, KIND_LOGO, a, notes)
                    s, sp = predict_slots(b, mb["slots"], backend, model, cost, KIND_LOGO, a, notes)
                    res["slots"] += s
                    res["slot_speedup"] += sp
                except Exception as e:
                    notes.append(f"LOGO {a}->{b} crashed: {type(e).__name__}: {e}")
    elif len(measured) < 2:
        res["notes"].append("LOGO skipped: needs data from at least two GPUs")

    res["metrics"] = _l3_metrics(res["vllm"], res["slots"], res["slot_speedup"])
    res["errors"] = collect_errors(res)
    if write and out_dir is not None:
        write_outputs(res, Path(out_dir))
    return res


def collect_errors(res: Dict) -> List[Dict]:
    """Unified error records (one per op/config) for the agent pipeline's flagging."""
    errs = level1_error_groups(res["level1"])
    for r in res["vllm"]:
        errs.append({**r, "level": "L3", "item": f"vLLM {r['metric']}",
                     "config": f"batch={r['batch']} input_len={r['input_len']}"
                               + (f" output_len={r['output_len']}" if r["metric"] == "TPOT" else ""),
                     "measured": r["measured_ms"], "predicted": r["predicted_ms"], "unit": "ms"})
    for r in res["slots"]:
        if r["err_pct"] is None:
            continue
        errs.append({**r, "level": "L3", "item": "llama.cpp slot tok/s",
                     "config": f"build={r['build']} users={r['users']} k={r['passes_per_step']}",
                     "measured": r["measured_tok_s"], "predicted": r["predicted_tok_s"], "unit": "tok/s"})
    for r in res["slot_speedup"]:
        if r["err_pct"] is None:
            continue
        errs.append({**r, "level": "L3", "item": "llama.cpp fix/master speedup",
                     "config": f"users={r['users']} k_master={r['k_master']} k_fix={r['k_fix']}",
                     "measured": r["measured_speedup"], "predicted": r["predicted_speedup"], "unit": "x"})
    return errs


# =================================================================================
# output
# =================================================================================
def write_csv(path: Path, rows: List[Dict]) -> None:
    """CSV with the union of keys (first-seen order). Writes a header-only file if empty."""
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def write_outputs(res: Dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "level1_gemv.csv", res["level1"])
    write_csv(out / "level1_summary.csv", res["level1_summary"])
    write_csv(out / "level2_ops_sol.csv", res["level2"])
    write_csv(out / "level2_summary.csv", res["level2_summary"])
    write_csv(out / "level2_logo_projection.csv", res["level2_logo"])
    write_csv(out / "level3_vllm.csv", [r for r in res["vllm"] if r["kind"] != KIND_LOGO])
    write_csv(out / "level3_llamacpp_slots.csv", [r for r in res["slots"] if r["kind"] != KIND_LOGO])
    write_csv(out / "level3_slot_speedup.csv", [r for r in res["slot_speedup"] if r["kind"] != KIND_LOGO])
    write_csv(out / "logo_projection.csv",
              [r for r in res["vllm"] + res["slots"] + res["slot_speedup"] if r["kind"] == KIND_LOGO])
    write_csv(out / "correlation_metrics.csv", res["metrics"])
    write_csv(out / "errors.csv", [{k: v for k, v in e.items() if not isinstance(v, dict)} for e in res["errors"]])
    (out / "correlation.md").write_text(to_markdown(res))


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    s = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    s += ["| " + " | ".join(r) + " |" for r in rows]
    return s


def _shares(r: Dict) -> str:
    return " / ".join(fmt(100 * r.get(f"share_{g}") if r.get(f"share_{g}") is not None else None, "{:.0f}")
                      for g in GROUPS)


def to_markdown(res: Dict) -> str:
    s = ["# Correlation: model vs measured", "",
         f"Generated {res['generated']} by `correlate/correlate.py` from `{res['data_root']}` "
         f"(backend: {res['backend']}, model: {res['model']}). GPUs: {', '.join(res['gpus']) or 'none'}.",
         "", "Error % = 100 * (predicted - measured) / measured. For times, negative = model too "
         "optimistic; for GB/s and tok/s, positive = model too optimistic.", ""]

    # ---- headline -----------------------------------------------------------
    s += ["## Headline", ""]
    head = []
    for r in res["level1_summary"]:
        if r["kernel"] == "all":
            head.append([r["gpu"], "L1 GEMV achieved BW (plain Little's Law)", "measured probes", str(r["n"]),
                         fmt(r["mape_plain"]), fmt(r["max_abs_plain"]), fmt(r["r_plain"], "{:.3f}")])
            head.append([r["gpu"], "L1 GEMV achieved BW (+ loaded latency)", f"alpha from {r['alpha_source']}", str(r["n"]),
                         fmt(r["mape_loaded"]), fmt(r["max_abs_loaded"]), fmt(r["r_loaded"], "{:.3f}")])
    for r in res["metrics"]:
        if r["kind"] == KIND_LOGO:
            continue
        head.append([r["gpu"], r["quantity"], r["kind"], str(r["n"]), fmt(r["mape"]),
                     fmt(r["max_abs_pct"]), fmt(r["pearson_r"], "{:.3f}")])
    if head:
        s += _table(["GPU", "quantity", "model", "n", "MAPE %", "max abs err %", "Pearson r"], head)
    else:
        s.append("_No predicted-vs-measured pairs: no data in data/<gpu_tag>/ or model modules missing (see Notes)._")
    logo = [r for r in res["metrics"] if r["kind"] == KIND_LOGO]
    if logo:
        s += ["", "Leave-one-GPU-out **projections** (calibrated on another GPU; target from its spec sheet only):", ""]
        s += _table(["target GPU", "calibrated on", "quantity", "n", "MAPE %", "max abs err %", "Pearson r"],
                    [[r["gpu"], r["calib_from"], r["quantity"], str(r["n"]), fmt(r["mape"]),
                      fmt(r["max_abs_pct"]), fmt(r["pearson_r"], "{:.3f}")] for r in logo])
    s.append("")

    # ---- Level 1 ------------------------------------------------------------
    s += ["## Level 1: GEMV achieved bandwidth (Little's Law)", ""]
    if res["level1_summary"]:
        for tag, p in res["level1_params"].items():
            s.append(f"- {tag}: SMs={p['sm_count']}, DRAM latency={fmt(p['dram_latency_ns'])} ns (pointer chase), "
                     f"peak={fmt(p['peak_bw_Bps'] / 1e9, '{:.0f}')} GB/s (stream max), "
                     f"alpha={fmt(p['alpha'], '{:.3f}')} (fit on {p['alpha_source']})")
        s.append("")
        s += _table(["GPU", "kernel", "n", "MAPE plain %", "MAPE loaded %", "max abs plain %",
                     "max abs loaded %", "r plain", "r loaded"],
                    [[r["gpu"], str(r["kernel"]), str(r["n"]), fmt(r["mape_plain"]), fmt(r["mape_loaded"]),
                      fmt(r["max_abs_plain"]), fmt(r["max_abs_loaded"]), fmt(r["r_plain"], "{:.3f}"),
                      fmt(r["r_loaded"], "{:.3f}")] for r in res["level1_summary"]])
        s.append("\nPer-configuration rows: `level1_gemv.csv`.")
    else:
        s.append("_Skipped (see Notes)._")
    s.append("")

    # ---- Level 2 ------------------------------------------------------------
    s += ["## Level 2: % of speed-of-light per op", "",
          "SOL = max(FLOPs / datasheet peak, bytes / datasheet bandwidth); % SOL = SOL time / measured time.", ""]
    if res["level2_summary"]:
        s += _table(["GPU", "op", "dtype", "n", "median % SOL", "min", "max", "mostly",
                     "Level 2 model MAPE % (own calib, in-sample)", "max abs %"],
                    [[r["gpu"], str(r["op"]), str(r["dtype"]), str(r["n"]), fmt(r["median_pct_sol"]),
                      fmt(r["min_pct_sol"]), fmt(r["max_pct_sol"]), r["mostly"], fmt(r["model_mape"]),
                      fmt(r["model_max_abs"])] for r in res["level2_summary"]])
        s.append("\nPer-shape rows: `level2_ops_sol.csv`.")
        if res["level2_logo"]:
            s += ["", "Level 2 leave-one-GPU-out **projection** (efficiency curves from another GPU on this GPU's "
                  "spec sheet):", ""]
            s += _table(["target", "calibrated on", "op", "dtype", "n", "MAPE %", "max abs %"],
                        [[r["gpu"], r["calib_from"], str(r["op"]), str(r["dtype"]), str(r["n"]), fmt(r["model_mape"]),
                          fmt(r["model_max_abs"])] for r in res["level2_logo"]])
    else:
        s.append("_Skipped (see Notes)._")
    s.append("")

    # ---- Level 3 ------------------------------------------------------------
    own = [r for r in res["vllm"] if r["kind"] != KIND_LOGO]
    s += ["## Level 3: vLLM TPOT and TTFT, with per-op attribution", "",
          "Shares = fraction of the *predicted* time in weights GEMV/GEMM / attention / norms+elementwise / "
          "overhead (engine time not explained by per-op costs).", ""]
    if own:
        s += _table(["GPU", "model", "metric", "batch", "input_len", "out", "measured ms", "predicted ms", "err %",
                     "shares W/A/N/O %"],
                    [[r["gpu"], r["kind"], r["metric"], str(r["batch"]), str(r["input_len"]), str(r["output_len"]),
                      fmt(r["measured_ms"], "{:.3f}"), fmt(r["predicted_ms"], "{:.3f}"), fmt(r["err_pct"], "{:+.1f}"),
                      _shares(r)] for r in own])
    else:
        s.append("_No vLLM rows (see Notes)._")
    s.append("")
    s += _attribution_section(own)

    own_s = [r for r in res["slots"] if r["kind"] != KIND_LOGO]
    s += ["## Level 3: llama.cpp slot test (master vs slot-order fix)", "",
          f"Predicted tok/s = users / decode_step_time(engine='llamacpp', passes_per_step=k) at an assumed "
          f"context of {LLAMACPP_CONTEXT} tokens; k from the DIAG/CPU counters.", ""]
    if own_s:
        s += _table(["GPU", "model", "build", "users", "k (passes/step)", "k source", "measured tok/s",
                     "predicted tok/s", "err %"],
                    [[r["gpu"], r["kind"], r["build"], str(r["users"]), fmt(r["passes_per_step"], "{:.2f}"),
                      r["k_source"], fmt(r["measured_tok_s"]), fmt(r["predicted_tok_s"]),
                      fmt(r["err_pct"], "{:+.1f}")] for r in own_s])
        sp = [r for r in res["slot_speedup"] if r["kind"] != KIND_LOGO]
        if sp:
            s += ["", "Fix speedup (fix tok/s / master tok/s):", ""]
            s += _table(["GPU", "users", "master tok/s", "fix tok/s", "measured speedup", "predicted speedup", "err %"],
                        [[r["gpu"], str(r["users"]), fmt(r["master_tok_s"]), fmt(r["fix_tok_s"]),
                          fmt(r["measured_speedup"], "{:.2f}x"), fmt(r["predicted_speedup"], "{:.2f}x"),
                          fmt(r["err_pct"], "{:+.1f}")] for r in sp])
    else:
        s.append("_No slot rows (see Notes)._")
    s.append("")

    # ---- LOGO ---------------------------------------------------------------
    lv = [r for r in res["vllm"] if r["kind"] == KIND_LOGO]
    ls = [r for r in res["slots"] if r["kind"] == KIND_LOGO]
    s += ["## Leave-one-GPU-out: Projection (not measured on the calibration GPU)", "",
          "Efficiency curves from GPU A's calib.json; GPU B described only by its hw/hardware.yaml spec sheet. "
          "B's measurements are used only to score the projection.", ""]
    if lv or ls:
        if lv:
            s += _table(["target", "calibrated on", "metric", "batch", "input_len", "measured ms", "projected ms",
                         "err %", "shares W/A/N/O %"],
                        [[r["gpu"], r["calib_from"], r["metric"], str(r["batch"]), str(r["input_len"]),
                          fmt(r["measured_ms"], "{:.3f}"), fmt(r["predicted_ms"], "{:.3f}"),
                          fmt(r["err_pct"], "{:+.1f}"), _shares(r)] for r in lv])
            s.append("")
        if ls:
            s += _table(["target", "calibrated on", "build", "users", "k", "measured tok/s", "projected tok/s", "err %"],
                        [[r["gpu"], r["calib_from"], r["build"], str(r["users"]), fmt(r["passes_per_step"], "{:.2f}"),
                          fmt(r["measured_tok_s"]), fmt(r["predicted_tok_s"]), fmt(r["err_pct"], "{:+.1f}")]
                         for r in ls])
    else:
        s.append("_No LOGO projection (needs >= 2 GPUs with data, one with calib.json; see Notes)._")
    s.append("")

    # ---- notes --------------------------------------------------------------
    s += ["## Notes (skipped inputs, assumptions)", ""]
    for n in res["notes"]:
        s.append(f"- {n}")
    for tag, ns in res["gpu_notes"].items():
        for n in dict.fromkeys(ns):  # de-duplicate, keep order
            s.append(f"- {tag}: {n.strip()}")
    s += [f"- llama.cpp context assumed {LLAMACPP_CONTEXT} tokens (prompt lengths not logged).",
          "- TPOT = (lat(out=OL) - lat(out=1)) / (OL - 1); TTFT ~= lat(out=1); predictions use the same formula "
          "on predict_vllm_latency, so both sides include the same terms.",
          "- Attribution: engine per-op breakdown (or OpCostModel on bf16 perfsim.models ops) at the mean "
          "decode context input_len + (output_len-2)/2; overhead = per-pass constant / unexplained engine time."]
    return "\n".join(s) + "\n"


def _attribution_section(rows: List[Dict]) -> List[str]:
    """(e) Per-op error attribution summary: mean shares of predicted time per GPU/metric
    next to the mean signed error, and the dominant group of the worst row."""
    if not rows:
        return []
    s = ["### Per-op error attribution", "",
         "Mean share of predicted time per op group, next to the error. A large error where one group "
         "dominates points at that group's cost model.", ""]
    grp = defaultdict(list)
    for r in rows:
        grp[(r["gpu"], r["metric"])].append(r)
    tab = []
    for (gpu, metric), rs in sorted(grp.items()):
        mean = lambda k: (sum(r[k] for r in rs if r.get(k) is not None) /
                          max(1, sum(1 for r in rs if r.get(k) is not None)))
        worst = max(rs, key=lambda r: abs(r["err_pct"]) if r["err_pct"] is not None and math.isfinite(r["err_pct"]) else -1)
        dom = max(("weights", "attention", "norms", "overhead"), key=lambda g: worst.get(f"share_{g}") or 0)
        tab.append([gpu, metric, fmt(mean("err_pct"), "{:+.1f}")]
                   + [fmt(100 * mean(f"share_{g}"), "{:.0f}") for g in GROUPS]
                   + [f"b={worst['batch']} in={worst['input_len']}: {fmt(worst['err_pct'], '{:+.1f}')}% ({dom})"])
    s += _table(["GPU", "metric", "mean err %", "weights %", "attention %", "norms %", "overhead %",
                 "worst row (dominant group)"], tab)
    s.append("")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Correlate the perfsim model with measured data.")
    ap.add_argument("--data", default="data", help="data root with one dir per gpu_tag")
    ap.add_argument("--out", default="results", help="output dir for correlation.md and CSVs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    res = run(args.data, args.out, model_name=args.model)
    print((Path(args.out) / "correlation.md").read_text())
    return res


if __name__ == "__main__":
    main()
