#!/usr/bin/env python3
"""Fit Level-2 efficiency curves from measured op timings.

    python bench/fit_calib.py --data data/<gpu_tag> [--hw <hardware.yaml key>] [--results results]

Reads   data/<tag>/ops.jsonl            (bench/ops_bench.py)
Writes  data/<tag>/calib.json           (perfsim.ops_model.Calibration)
        results/<tag>_ops_sol.md        per-op measured us, SOL us, % of SOL, model us, error

Definitions (see perfsim/ops_model.py):
    SOL us      = max(flops / peak_flops[dtype], bytes / peak_bw)   from hw/hardware.yaml
    % of SOL    = SOL us / measured us                              (100% = speed of light)
    model us    = t0 + SOL / eff(feature)  (in-sample: reproduces every measured point
                  up to clipping and the median used when shapes share a feature value)
Runs anywhere (no GPU, no torch).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from perfsim.hardware import load_hardware  # noqa: E402
from perfsim.ops_model import (OpCostModel, fit_calibration, keep_fastest, precision_for,  # noqa: E402
                               predict_record, read_ops_jsonl, sol_time)


def _shape_str(shape: dict) -> str:
    keys = [k for k in shape if k != "weight_bytes"]  # proj/K/N already identify the weight
    return " ".join(f"{k}={shape[k]}" for k in keys)


def _bound(rec, hw) -> str:
    tc = float(rec["flops"]) / hw.peak(precision_for(rec["dtype"], hw))
    tm = float(rec["bytes"]) / hw.memory_bandwidth
    return "compute" if tc >= tm else "memory"


def _f(x, fmt="{:.2f}"):
    return fmt.format(x) if x is not None else "-"


def report_md(tag, hw, calib, raw_recs, problems, data_path) -> str:
    used = keep_fastest(raw_recs)
    cost = OpCostModel(hw, calib, mode="empirical")
    lines = [f"# Level-2 op calibration: {tag}", "",
             f"Measured: `{data_path}` ({len(raw_recs)} records; {len(used)} after keeping the fastest "
             f"implementation per shape). Spec sheet: `hw/hardware.yaml` `{hw.key}` ({hw.name}): "
             f"peak bf16 {hw.peak_flops.get('bf16', float('nan')) / 1e12:.1f} TFLOP/s, "
             f"DRAM {hw.memory_bandwidth / 1e12:.3f} TB/s.", "",
             "* SOL us = max(flops / peak, bytes / bandwidth); **% of SOL = SOL us / measured us**.",
             "* model us = t0 + SOL / eff(feature), the empirical-mode prediction (in-sample).",
             "* Probe rows (tiny shapes) only set the fixed overhead t0; they are not on the curves.", ""]

    lines += ["## Fitted curves", "",
              "| op | dtype | feature | t0 us (from) | shapes | eff min / median / max | clipped at eff_max | faster than SOL |",
              "|---|---|---|---|---|---|---|---|"]
    warn = []
    for key, c in sorted(calib.curves.items()):
        effs = sorted(c.effs)
        med = effs[len(effs) // 2]
        lines.append(f"| {c.op} | {c.dtype} | {c.feature} | {c.t0_s * 1e6:.2f} ({c.t0_source}) | "
                     f"{c.n_records} | {effs[0]:.3f} / {med:.3f} / {effs[-1]:.3f} | {c.n_clipped} | "
                     f"{c.n_faster_than_sol} |")
        if c.n_faster_than_sol:
            warn.append(f"{key}: {c.n_faster_than_sol} shapes ran FASTER than the spec-sheet SOL "
                        "(clocks above the rated boost clock, an L2 hit, or a wrong spec value); "
                        "their efficiency is clipped to eff_max, so the model over-predicts them.")
        if c.t0_source != "probe":
            warn.append(f"{key}: no probe record; t0 = min(measured - SOL) (an upper bound).")
    for k, msg in (calib.meta.get("skipped") or {}).items():
        warn.append(f"{k}: not fitted ({msg})")

    lines += ["", "## Per op", "",
              "| op | shape | impl | measured us | SOL us | % of SOL | bound | model us | model err % |",
              "|---|---|---|---|---|---|---|---|---|"]

    def order(r):  # by op, then projection and token count (gemm), then the rest of the shape
        s = r.get("shape", {})
        rest = tuple((k, s[k] if isinstance(s[k], (int, float)) else 0, str(s[k])) for k in sorted(s))
        return (r["op"], str(s.get("proj", "")), s.get("M", 0), rest)

    errs = []
    for r in sorted(used, key=order):
        sol_us = sol_time(float(r["flops"]), float(r["bytes"]), hw, r["dtype"]) * 1e6
        meas = float(r["us"])
        probe = bool(r.get("shape", {}).get("probe"))
        try:
            model = predict_record(cost, r) * 1e6
            err = 100.0 * (model / meas - 1.0)
            if not probe:
                errs.append(abs(err))
        except Exception as e:  # never let one odd record kill the report
            model, err = None, None
            warn.append(f"{r['op']} {_shape_str(r.get('shape', {}))}: model failed ({e})")
        lines.append(f"| {r['op']}{' (probe)' if probe else ''} | {_shape_str(r.get('shape', {}))} | "
                     f"{r.get('impl', '-')} | {meas:.2f} | {sol_us:.2f} | {100 * sol_us / meas:.1f}% | "
                     f"{_bound(r, hw)} | {_f(model)} | {_f(err, '{:+.1f}')} |")
    if errs:
        lines += ["", f"In-sample model error over {len(errs)} non-probe shapes: "
                      f"mean |err| {sum(errs) / len(errs):.2f}%, max |err| {max(errs):.2f}%."]
    if problems or warn:
        lines += ["", "## Warnings", ""] + [f"* {p}" for p in problems + warn]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit Level-2 efficiency curves (calib.json + SOL table)")
    ap.add_argument("--data", required=True, help="data/<gpu_tag> directory containing ops.jsonl")
    ap.add_argument("--hw", default=None, help="hardware.yaml device key (default: the directory name)")
    ap.add_argument("--results", default=str(REPO / "results"), help="where <tag>_ops_sol.md goes")
    ap.add_argument("--ops", default=None, help="ops file (default: <data>/ops.jsonl)")
    ap.add_argument("--calib-out", default=None, help="calib file (default: <data>/calib.json)")
    ap.add_argument("--eff-max", type=float, default=1.0, help="upper clip of the efficiency (default 1)")
    args = ap.parse_args(argv)

    data = Path(args.data)
    tag = data.resolve().name
    ops_path = Path(args.ops) if args.ops else data / "ops.jsonl"
    hw_all = load_hardware()
    key = args.hw or tag
    if key not in hw_all:
        print(f"error: no device '{key}' in hw/hardware.yaml (have {sorted(hw_all)}); pass --hw",
              file=sys.stderr)
        return 2
    hw = hw_all[key]
    if not ops_path.exists():
        print(f"error: {ops_path} not found (run bench/ops_bench.py on the GPU first)", file=sys.stderr)
        return 2
    recs, problems = read_ops_jsonl(ops_path)
    if not recs:
        print(f"error: no usable records in {ops_path}: {problems[:5]}", file=sys.stderr)
        return 2
    calib = fit_calibration(recs, hw, gpu_tag=tag, eff_max=args.eff_max)
    calib.meta["source"] = str(ops_path)
    calib_path = Path(args.calib_out) if args.calib_out else data / "calib.json"
    calib.save(calib_path)
    md_path = Path(args.results) / f"{tag}_ops_sol.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_md(tag, hw, calib, recs, problems, ops_path))
    print(f"{len(calib.curves)} curves -> {calib_path}; table -> {md_path}")
    for p in problems:
        print("warning:", p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
