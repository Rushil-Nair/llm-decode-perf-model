"""Level 3, refined: llama.cpp slot throughput from measured per-pass cost.

The first Level 3 slot model (correlate.py) builds the decode step from Level 2 op costs. For the Q4_K_M
GGUF model those op costs are not calibrated (the microbenchmarks are bf16 PyTorch), so the absolute tok/s
is far off and the fix/master speedup is over-predicted by 30-77%.

This model uses llama.cpp's own measured cost of one forward pass instead:

    t_pass(B)  = decode time of one pass over B sequences, interpolated from llama-batched-bench
                 (S_TG at B = 1, 2, 4, 8, 16, 32; t_pass(B) = B / S_TG(B))
    t_step     = k * t_pass(users / k) + s
    tok/s      = users / t_step

k is the measured number of forward passes per generation step (DIAG counter in the server log); the
passes are assumed to split the users evenly. s is a per-step server cost (HTTP, sampling, scheduling)
fitted from one point only: the 1-user master run. Every other run (4, 8, 16 users, master and fix) is
predicted.

What it leaves out: in the closed-loop server test new requests are prefilled (prompt cache off) between
decode steps and not every user is decoding at every step. Both lower measured tok/s more as users grow, so
absolute tok/s is expected to be over-predicted; the fix/master ratio cancels most of it.

    python -m perfsim.slot_pass_model --data data --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from correlate.loaders import load_slots


def load_batched_bench(tag_dir) -> Tuple[np.ndarray, np.ndarray]:
    """(B, t_pass seconds) from llamacpp/bench/batched_bench.jsonl, sorted by B."""
    rows = [json.loads(l) for l in open(Path(tag_dir) / "llamacpp" / "bench" / "batched_bench.jsonl") if l.strip()]
    pts = sorted({int(r["pl"]): int(r["pl"]) / float(r["speed_tg"]) for r in rows}.items())
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def t_pass(B: np.ndarray, T: np.ndarray, b: float) -> float:
    return float(np.interp(b, B, T))


def step_time(B, T, users: int, k: float, s: float) -> float:
    return k * t_pass(B, T, users / k) + s


def fit_overhead(B, T, tok_s_1user: float) -> float:
    return 1.0 / tok_s_1user - t_pass(B, T, 1.0)


def analyse(tag_dir) -> Dict:
    B, T = load_batched_bench(tag_dir)
    slots = [r for r in load_slots(tag_dir) if r["policy"] in ("master_plain", "fix_plain") and r["passes_per_step"]]
    by = {(r["build"], r["users"]): r for r in slots}
    s = fit_overhead(B, T, by[("master", 1)]["tok_s"])
    rows = []
    for (build, u), r in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
        pred = u / step_time(B, T, u, r["passes_per_step"], s)
        rows.append({"build": build, "users": u, "k": r["passes_per_step"], "measured_tok_s": r["tok_s"],
                     "predicted_tok_s": pred, "err_pct": (pred / r["tok_s"] - 1) * 100,
                     "role": "fit (s)" if (build, u) == ("master", 1) else "predicted"})
    pr = {(x["build"], x["users"]): x for x in rows}
    speed = []
    for u in sorted({x["users"] for x in rows}):
        if ("fix", u) in pr and ("master", u) in pr and u > 1:
            m = pr[("fix", u)]["measured_tok_s"] / pr[("master", u)]["measured_tok_s"]
            p = pr[("fix", u)]["predicted_tok_s"] / pr[("master", u)]["predicted_tok_s"]
            speed.append({"users": u, "measured": m, "predicted": p, "err_pct": (p / m - 1) * 100})
    return {"B": B.tolist(), "t_pass_ms": (T * 1e3).tolist(), "s_ms": s * 1e3, "rows": rows, "speedup": speed}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = ["# Level 3: llama.cpp slot throughput from measured per-pass cost", "",
             "t_step = k * t_pass(users / k) + s; t_pass from llama-batched-bench; s fitted on the 1-user master run "
             "only; k = measured passes per step.", ""]
    csv_rows: List[dict] = []
    for d in sorted(Path(a.data).iterdir()):
        if not (d / "llamacpp" / "bench" / "batched_bench.jsonl").exists():
            continue
        r = analyse(d)
        lines += [f"## {d.name}", "",
                  "t_pass(B) ms: " + ", ".join(f"B={int(b)}: {t:.2f}" for b, t in zip(r["B"], r["t_pass_ms"]))
                  + f"; s = {r['s_ms']:.2f} ms", "",
                  "| build | users | k | measured tok/s | predicted tok/s | err % | role |", "|---|---|---|---|---|---|---|"]
        for x in r["rows"]:
            lines.append(f"| {x['build']} | {x['users']} | {x['k']:.2f} | {x['measured_tok_s']:.1f} | "
                         f"{x['predicted_tok_s']:.1f} | {x['err_pct']:+.1f} | {x['role']} |")
            csv_rows.append({"gpu": d.name, **x})
        lines += ["", "| users | measured fix/master | predicted | err % |", "|---|---|---|---|"]
        for x in r["speedup"]:
            lines.append(f"| {x['users']} | {x['measured']:.2f}x | {x['predicted']:.2f}x | {x['err_pct']:+.1f} |")
        lines.append("")
    (out / "level3_slot_pass_model.md").write_text("\n".join(lines) + "\n")
    if csv_rows:
        with open(out / "level3_slot_pass_model.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
