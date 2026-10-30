"""Predict the v5 GEMV kernel from the three-ceiling model before running it, then score the prediction.

v5 (kernels/gemv_sweep.cu) is v4 with the int4 -> float conversions replaced by an fp16 bit trick.
Its SM cost per group is predicted from v4's fitted cost and the static SASS instruction counts:

    cpg(v5, ILP) = cpg(v4, ILP) * instr_per_group(v5, ILP) / instr_per_group(v4, ILP)     (same for cpw)

i.e. "the SM spends the same time per instruction as in v4". Everything else (memory ceiling, t0,
clock, SM count) is unchanged, and v5 runs the same shapes and occupancy configs as v4.

    python -m perfsim.v5_prediction --data data --out results            # write the prediction
    python -m perfsim.v5_prediction --out results --score data_kernels/h100_lium --gpu h100_sxm
                                                                         # compare with a v5 run
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from perfsim import gemv_model as G
from perfsim import microarch as M

SASS = {"h100_sxm": "sass_instr_sm90.json", "rtx5090": "sass_instr_sm120.json"}
CFG = ("shape", "N", "K", "ilp", "blocks_per_sm")


def predict_v5(gpu: G.GPU, sass: Dict, base: str = "v4", new: str = "v5") -> List[dict]:
    cpg, cpw, t0 = G.fit(gpu)
    ratio = {ilp: sass[f"{new}_ilp{ilp}"]["instr_per_group"] / sass[f"{base}_ilp{ilp}"]["instr_per_group"]
             for ilp in (1, 2, 4)}
    cpg2 = dict(cpg); cpw2 = dict(cpw)
    for ilp, r in ratio.items():
        cpg2[(new, ilp)] = cpg[(base, ilp)] * r
        cpw2[(new, ilp)] = cpw[(base, ilp)] * r
    recs = [dict(r, kernel=new) for r in gpu.records if r["kernel"] == base]
    g = G.GPU.__new__(G.GPU)
    g.__dict__.update(gpu.__dict__); g.records = recs
    rows = G.predict(g, (cpg2, cpw2, t0))
    base_meas = {tuple(r[k] for k in CFG): r["GBps"] for r in gpu.records if r["kernel"] == base}
    base_pred = {tuple(r[k] for k in CFG): r["predicted_GBps"] for r in G.predict(gpu, (cpg, cpw, t0))
                 if r["kernel"] == base}
    for x in rows:
        x["base_predicted_GBps"] = base_pred[tuple(x[k] for k in CFG)]
        x["model_speedup"] = x["predicted_GBps"] / x["base_predicted_GBps"]
        x["base_measured_GBps"] = base_meas[tuple(x[k] for k in CFG)]
        x["predicted_speedup"] = x["predicted_GBps"] / x["base_measured_GBps"]
        x["instr_ratio"] = ratio[x["ilp"]]
        x["cpg_pred"] = cpg2[(new, x["ilp"])]
    return rows


def write_prediction(data: Path, out: Path) -> Dict[str, List[dict]]:
    allrows, lines = {}, []
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines += ["# v5 GEMV kernel: prediction made before it ran on any GPU", "",
              f"Written {now} by `perfsim/v5_prediction.py`. cpg(v5) = cpg(v4) x SASS instructions per group "
              "(v5 / v4); all other model inputs unchanged.", ""]
    for tag, fn in SASS.items():
        d = data / tag
        if not (d / "gemv_sweep.jsonl").exists() or not (out / fn).exists():
            continue
        sass = json.load(open(out / fn))
        gpu = G.GPU(d)
        rows = predict_v5(gpu, sass)
        allrows[tag] = rows
        best_v4 = max(r["GBps"] for r in gpu.records if r["kernel"] == "v4")
        best_v5 = max(rows, key=lambda r: r["predicted_GBps"])
        lines += [f"## {tag}", "",
                  "| ILP | SASS instr/group v4 | v5 | ratio | predicted cpg v5 (SM cycles/group) |", "|---|---|---|---|---|"]
        for ilp in (1, 2, 4):
            r = next(x for x in rows if x["ilp"] == ilp)
            lines.append(f"| {ilp} | {sass[f'v4_ilp{ilp}']['instr_per_group']:.1f} | "
                         f"{sass[f'v5_ilp{ilp}']['instr_per_group']:.1f} | {r['instr_ratio']:.2f} | {r['cpg_pred']:.2f} |")
        lines += ["", "Speedup vs measured v4 compares the prediction with the real v4; speedup within the model "
                      "compares v5 and v4 both as predicted (cancels the model's own error on v4).", "",
                  "| shape | best v4 measured GB/s | best v5 predicted GB/s | vs measured v4 | within the model | bound | "
                      "% of stream peak |", "|---|---|---|---|---|---|---|"]
        for shape in sorted({r["shape"] for r in rows}):
            m = max(r["GBps"] for r in gpu.records if r["kernel"] == "v4" and r["shape"] == shape)
            p = max((r for r in rows if r["shape"] == shape), key=lambda r: r["predicted_GBps"])
            pv4 = max(r["base_predicted_GBps"] for r in rows if r["shape"] == shape)
            lines.append(f"| {shape} | {m:.0f} | {p['predicted_GBps']:.0f} | {p['predicted_GBps'] / m:.2f}x | "
                         f"{p['predicted_GBps'] / pv4:.2f}x | {p['bound']} | "
                         f"{100 * p['predicted_GBps'] * 1e9 / gpu.params.peak_bw_Bps:.0f}% |")
        lines += ["", f"Overall: best v4 measured {best_v4:.0f} GB/s; best v5 predicted "
                      f"{best_v5['predicted_GBps']:.0f} GB/s ({best_v5['predicted_GBps'] / best_v4:.2f}x), "
                      f"{best_v5['bound']}-bound at {best_v5['shape']} ILP {best_v5['ilp']}, "
                      f"{best_v5['blocks_per_sm']} blocks/SM.", ""]
    (out / "v5_prediction.md").write_text("\n".join(lines) + "\n")
    flat = [dict(gpu=t, **r) for t, rs in allrows.items() for r in rs]
    if flat:
        with open(out / "v5_prediction.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print("\n".join(lines))
    return allrows


def score(pred_csv: Path, run_dir: Path, gpu_tag: str, out: Path) -> None:
    """Compare predicted v5 rows with a measured gemv_sweep.jsonl that contains v5 records."""
    pred = [r for r in csv.DictReader(open(pred_csv)) if r["gpu"] == gpu_tag]
    meas = [r for r in M.usable_records(M.load_gemv_sweep(run_dir)) if r.get("kernel") == "v5"]
    mm = {(r["shape"], int(r["ilp"]), int(r["blocks_per_sm"])): r for r in meas}
    rows = []
    for p in pred:
        k = (p["shape"], int(p["ilp"]), int(p["blocks_per_sm"]))
        if k in mm:
            m = mm[k]["GBps"]; pg = float(p["predicted_GBps"])
            rows.append({"shape": k[0], "ilp": k[1], "blocks_per_sm": k[2], "measured_GBps": m,
                         "predicted_GBps": pg, "err_pct": (pg / m - 1) * 100,
                         "v4_measured_GBps": float(p["base_measured_GBps"]),
                         "measured_speedup": m / float(p["base_measured_GBps"]),
                         "predicted_speedup": float(p["predicted_speedup"])})
    if not rows:
        print("no matching v5 configs"); return
    e = np.abs([r["err_pct"] for r in rows])
    best_m = max(meas, key=lambda r: r["GBps"])
    lines = [f"# v5 prediction vs measurement ({gpu_tag}, run {run_dir})", "",
             f"Matched configs: {len(rows)}; MAPE {e.mean():.1f}%, max {e.max():.1f}%.",
             f"Best v5 measured: {best_m['GBps']:.0f} GB/s ({best_m['shape']}, ILP {best_m['ilp']}, "
             f"{best_m['blocks_per_sm']} blocks/SM).", "",
             "| shape | ILP | blocks/SM | v4 measured | v5 predicted | v5 measured | err % | speedup predicted | measured |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["shape"], r["ilp"], r["blocks_per_sm"])):
        lines.append(f"| {r['shape']} | {r['ilp']} | {r['blocks_per_sm']} | {r['v4_measured_GBps']:.0f} | "
                     f"{r['predicted_GBps']:.0f} | {r['measured_GBps']:.0f} | {r['err_pct']:+.1f} | "
                     f"{r['predicted_speedup']:.2f}x | {r['measured_speedup']:.2f}x |")
    (out / f"v5_score_{gpu_tag}.md").write_text("\n".join(lines) + "\n")
    with open(out / f"v5_score_{gpu_tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--score", help="directory with a gemv_sweep.jsonl that includes v5 records")
    ap.add_argument("--gpu", default="h100_sxm", help="which GPU's prediction to score")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.score:
        score(out / "v5_prediction.csv", Path(a.score), a.gpu, out)
    else:
        write_prediction(Path(a.data), out)


if __name__ == "__main__":
    main()
