"""All plots from results/*.csv (written by correlate/ and whatif/), as PNGs.

    python -m figures.make_figures --results results --out figures

  littles_law.png          predicted vs measured GEMV BW vs bytes in flight, per kernel/ILP, per GPU
  ops_sol.png              median % of speed-of-light per op (Level 2)
  correlation_scatter.png  predicted vs measured TPOT and TTFT, y = x, both GPUs, LOGO marked
  slot_fix.png             llama.cpp slot tok/s, master vs fix, measured bars + predicted markers
  whatif_bw.png            projected decode tok/s vs HBM bandwidth scale (projection)

Headless (Agg backend). A missing input CSV skips that figure with a message.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = {"v1": "s", "v2": "o", "v3": "^", "v4": "D"}
LOGO_KIND = "projection (LOGO)"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        print(f"skip: {path} missing")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"skip: {path} has no rows")
        return None
    return df


def _gpu_colors(gpus) -> Dict[str, str]:
    return {g: SERIES[i % len(SERIES)] for i, g in enumerate(sorted(gpus))}


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    return p


def littles_law(results: Path, out: Path) -> Optional[Path]:
    df = _read(results / "level1_gemv.csv")
    if df is None:
        return None
    gpus = sorted(df["gpu"].unique())
    fig, axes = plt.subplots(1, len(gpus), figsize=(5.2 * len(gpus), 4), squeeze=False, sharey=False)
    ilps = sorted(df["ilp"].unique())
    ilp_color = {v: SERIES[i % len(SERIES)] for i, v in enumerate(ilps)}
    for ax, g in zip(axes[0], gpus):
        d = df[df["gpu"] == g].sort_values("bytes_in_flight")
        x = d["bytes_in_flight"] / 2**20
        for (kern, ilp), dd in d.groupby(["kernel", "ilp"]):
            ax.scatter(dd["bytes_in_flight"] / 2**20, dd["measured_GBps"], s=30, marker=MARKERS.get(kern, "o"),
                       color=ilp_color[ilp], edgecolor="white", linewidth=0.8, zorder=3,
                       label=f"measured {kern} ILP {ilp}")
        ax.plot(x, d["pred_plain_GBps"], color=INK2, linestyle="--", linewidth=1.5,
                label="Little's Law: min(peak, BIF / latency)")
        if "pred_loaded_GBps" in d and d["pred_loaded_GBps"].notna().any():
            ax.plot(x, d["pred_loaded_GBps"], color=INK, linewidth=1.5, label="+ loaded-latency correction")
        if "peak_stream_GBps" in d and d["peak_stream_GBps"].notna().any():
            ax.axhline(d["peak_stream_GBps"].iloc[0], color=INK2, linewidth=0.8, linestyle=":")
            ax.annotate("stream max", (x.min(), d["peak_stream_GBps"].iloc[0]), xytext=(2, 3),
                        textcoords="offset points", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"{g}: GEMV achieved bandwidth")
        ax.set_xlabel("bytes in flight, whole GPU (MiB, log)")
        ax.set_ylabel("achieved bandwidth (GB/s)")
        ax.set_ylim(bottom=0)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(4, len(lab)), bbox_to_anchor=(0.5, -0.12), fontsize=8)
    return _save(fig, out, "littles_law.png")


def ops_sol(results: Path, out: Path) -> Optional[Path]:
    df = _read(results / "level2_summary.csv")
    if df is None:
        return None
    df["label"] = df["op"].astype(str) + " (" + df["dtype"].astype(str) + ")"
    labels = sorted(df["label"].unique())
    gpus = sorted(df["gpu"].unique())
    col = _gpu_colors(gpus)
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(labels) * len(gpus) + 2), 3.8))
    w = 0.8 / len(gpus)
    for i, g in enumerate(gpus):
        d = df[df["gpu"] == g].set_index("label")
        xs = [j + (i - (len(gpus) - 1) / 2) * w for j in range(len(labels))]
        ys = [d["median_pct_sol"].get(l, float("nan")) for l in labels]
        ax.bar(xs, ys, width=w * 0.92, color=col[g], label=g, zorder=3)
        for xx, yy in zip(xs, ys):
            if isinstance(yy, float) and math.isfinite(yy):
                ax.annotate(f"{yy:.0f}", (xx, yy), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=INK2)
    ax.grid(axis="x", visible=False)
    ax.axhline(100, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("median % of speed-of-light")
    ax.set_title("Level 2: measured op time vs roofline SOL (100% = datasheet bound)")
    ax.set_ylim(0, max(110, (df["median_pct_sol"].max() or 0) * 1.15))
    if len(gpus) > 1:
        ax.legend()
    return _save(fig, out, "ops_sol.png")


def correlation_scatter(results: Path, out: Path) -> Optional[Path]:
    frames = [f for f in (_read(results / "level3_vllm.csv"), _read(results / "logo_projection.csv"))
              if f is not None and "metric" in f]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df["metric"].isin(["TPOT", "TTFT"])]
    if df.empty:
        print("skip: no TPOT/TTFT rows")
        return None
    col = _gpu_colors(df["gpu"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, metric in zip(axes, ("TPOT", "TTFT")):
        d = df[df["metric"] == metric]
        if d.empty:
            ax.set_visible(False)
            continue
        for g, dg in d.groupby("gpu"):
            own = dg[dg["kind"] != LOGO_KIND]
            logo = dg[dg["kind"] == LOGO_KIND]
            if not own.empty:
                kinds = "/".join(sorted(own["kind"].astype(str).unique()))  # calibrated or SOL
                ax.scatter(own["measured_ms"], own["predicted_ms"], s=36, color=col[g], edgecolor="white",
                           linewidth=0.8, zorder=3, label=f"{g}: {kinds}")
            if not logo.empty:
                src = ", ".join(sorted(logo["calib_from"].astype(str).unique()))
                ax.scatter(logo["measured_ms"], logo["predicted_ms"], s=40, facecolor="none", edgecolor=col[g],
                           linewidth=1.5, marker="D", zorder=3, label=f"{g} LOGO projection (calib {src})")
        lo = min(d["measured_ms"].min(), d["predicted_ms"].min()) * 0.8
        hi = max(d["measured_ms"].max(), d["predicted_ms"].max()) * 1.25
        xs = [lo, hi]
        ax.plot(xs, xs, color=INK, linewidth=1, label="y = x")
        ax.fill_between(xs, [v * 0.85 for v in xs], [v * 1.15 for v in xs], color=GRID, alpha=0.6,
                        zorder=0, label="+-15%")
        ax.set_xscale("log")
        ax.set_yscale("log")
        cand = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if hi / lo > 20 else (1, 2, 3, 5, 7, 10, 15, 20, 30, 50)
        ticks = [t for t in cand if lo <= t <= hi]
        for axis in (ax.xaxis, ax.yaxis):   # plain numbers instead of 2x10^1
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"measured {metric} (ms, log)")
        ax.set_ylabel(f"predicted {metric} (ms, log)")
        ax.set_title(f"vLLM {metric}: predicted vs measured")
    h, lab = axes[0].get_legend_handles_labels() if axes[0].get_visible() else axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    return _save(fig, out, "correlation_scatter.png")


def slot_fix(results: Path, out: Path) -> Optional[Path]:
    """Measured bars; markers = pass-level model (perfsim.slot_pass_model) when its CSV exists, else the
    op-level model from correlate.py."""
    pm = results / "level3_slot_pass_model.csv"
    if pm.exists():
        df = _read(pm).rename(columns={"gpu": "gpu"})
        pred_label = "pass-level model (s fitted on master, 1 user)"
    else:
        df = _read(results / "level3_llamacpp_slots.csv")
        pred_label = "model prediction"
    if df is None:
        return None
    gpus = sorted(df["gpu"].unique())
    fig, axes = plt.subplots(1, len(gpus), figsize=(5 * len(gpus), 3.8), squeeze=False)
    bcol = {"master": SERIES[1], "fix": SERIES[0]}
    for ax, g in zip(axes[0], gpus):
        d = df[df["gpu"] == g]
        users = sorted(d["users"].unique())
        for i, build in enumerate(("master", "fix")):
            db = d[d["build"] == build].set_index("users")
            xs = [j + (i - 0.5) * 0.4 for j in range(len(users))]
            meas = [db["measured_tok_s"].get(u, float("nan")) for u in users]
            pred = [db["predicted_tok_s"].get(u, float("nan")) for u in users]
            ax.bar(xs, meas, width=0.37, color=bcol[build], label=f"{build} measured", zorder=3)
            ax.scatter(xs, pred, marker="_", s=260, linewidths=2.5, color=INK, zorder=4,
                       label=pred_label if i == 0 else None)
        # measured speedup labels above each user group
        for j, u in enumerate(users):
            m = d[(d["build"] == "master") & (d["users"] == u)]["measured_tok_s"]
            f = d[(d["build"] == "fix") & (d["users"] == u)]["measured_tok_s"]
            if len(m) and len(f):
                top = max(m.iloc[0], f.iloc[0], d[d["users"] == u]["predicted_tok_s"].max())
                ax.annotate(f"{f.iloc[0] / m.iloc[0]:.2f}x", (j, top), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(range(len(users)))
        ax.set_xticklabels([str(u) for u in users])
        ax.set_xlabel("concurrent users (slots busy)")
        ax.set_ylabel("aggregate decode throughput (tok/s)")
        ax.set_title(f"{g}: llama.cpp slot-order fix")
        ax.margins(y=0.12)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    return _save(fig, out, "slot_fix.png")


def whatif_bw(results: Path, out: Path) -> Optional[Path]:
    """Projected decode throughput relative to today's bandwidth (ratio column of whatif.csv):
    dashed = ideal BW scaling, solid = capped by Little's Law at the measured bytes in flight."""
    df = _read(results / "whatif.csv")
    if df is None:
        return None
    d = df[(df["metric"] == "decode tok/s") & df["scenario"].astype(str).str.startswith("HBM BW")]
    if d.empty:
        print("skip: no HBM rows in whatif.csv")
        return None
    d = d.assign(value=d["value"].astype(float))
    batches = sorted(d["batch"].unique())
    col = {b: SERIES[i % len(SERIES)] for i, b in enumerate(batches)}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = sorted(d["value"].unique())
    ax.plot(xs, xs, color=GRID, linewidth=4, zorder=1, label="proportional to BW")
    for b in batches:
        for sc, ls in (("HBM BW (ideal)", "--"), ("HBM BW (Little's-Law cap)", "-")):
            s = d[(d["batch"] == b) & (d["scenario"] == sc)].sort_values("value")
            if s.empty:
                continue
            ax.plot(s["value"], s["ratio"], linestyle=ls, marker="o", markersize=5, color=col[b], zorder=3,
                    label=f"batch {b}, {sc.split('(')[1].rstrip(')')}")
    ax.set_xlabel("HBM bandwidth (x datasheet value)")
    ax.set_ylabel("projected decode tok/s (x today's model value)")
    gpu = d["gpu"].iloc[0]
    ax.set_title(f"Projection (not measured): {gpu} decode throughput vs HBM bandwidth")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, out, "whatif_bw.png")


def three_ceiling(results: Path, out: Path) -> Optional[Path]:
    """Level 1: plain Little's Law vs the three-ceiling model, predicted vs measured GEMV bandwidth."""
    ll = _read(results / "level1_gemv.csv")
    tc = _read(results / "level1_gemv_model.csv")
    if ll is None or tc is None:
        return None
    tc = tc[tc["fitted_on"] == tc["predicted_on"]]
    gpus = sorted(ll["gpu"].unique())
    col = _gpu_colors(gpus)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)
    panels = [(axes[0], "Plain Little's Law (memory latency only)",
               {g: (ll[ll.gpu == g].measured_GBps, ll[ll.gpu == g].pred_plain_GBps) for g in gpus}),
              (axes[1], "Three ceilings: memory / SM throughput / per-warp",
               {g: (tc[tc.predicted_on == g].measured_GBps, tc[tc.predicted_on == g].predicted_GBps) for g in gpus})]
    lo, hi = 100, 4000
    for ax, title, d in panels:
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle="--", zorder=1)
        for g in gpus:
            m, p = d[g]
            err = (abs(p / m - 1)).mean() * 100
            ax.scatter(m, p, s=22, color=col[g], edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"{g} (MAPE {err:.0f}%)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("measured bandwidth (GB/s, log)")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("predicted bandwidth (GB/s, log)")
    fig.suptitle("4-bit decode GEMV, every kernel x occupancy x ILP x shape (fitted and scored on the same GPU)",
                 fontsize=9, color=INK2, y=1.0)
    return _save(fig, out, "level1_three_ceiling.png")



def kernel_ladder(results: Path, out: Path) -> Optional[Path]:
    """H100 GEMV kernel ladder: best timed GB/s per kernel vs the memory ceiling and the stream peak."""
    df = _read(results / "h100_kernel_ladder.csv")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    xs = range(len(df))
    ax.bar(xs, df["GBps"], width=0.62, color=SERIES[0], zorder=3)
    for x, (g, v) in enumerate(zip(df["GBps"], df["vs_v4"])):
        ax.annotate(f"{g:.0f}\n{v:.2f}x v4", (x, g), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ceil = df["mem_ceiling_GBps"].max()
    ax.axhline(ceil, color=INK, linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate(f"model memory ceiling ~{ceil:.0f} GB/s", (-0.4, ceil), xytext=(0, 4),
                textcoords="offset points", ha="left", fontsize=8, color=INK)
    ax.axhspan(1900, 2100, color=GRID, alpha=0.7, zorder=1)
    ax.annotate("v7 prediction, written before the run: 1.9-2.1 TB/s", (-0.4, 1905), fontsize=7.5, color=INK2,
                va="bottom")
    ax.set_xticks(list(xs))
    short = {"v2": "x via L1", "v3": "x in shared mem", "v4": "+ streaming loads", "v5": "fewer instructions",
             "v6": "no bank conflicts", "v7": "both fixes"}
    ax.set_xticklabels([f"{k}\n{short.get(k, '')}" for k in df["kernel"]], fontsize=8)
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_ylim(0, ceil * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_title("H100 4-bit GEMV, gate_proj (12288x4096): each kernel version changes one thing")
    return _save(fig, out, "kernel_ladder.png")


ALL = (littles_law, three_ceiling, kernel_ladder, ops_sol, correlation_scatter, slot_fix, whatif_bw)


def make_all(results="results", out="figures") -> List[Path]:
    results, out = Path(results), Path(out)
    made = []
    for fn in ALL:
        try:
            p = fn(results, out)
        except Exception as e:  # one broken figure must not stop the others
            print(f"skip: {fn.__name__} failed ({type(e).__name__}: {e})")
            p = None
        if p:
            made.append(p)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make all figures from results/*.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--data", default="data", help="(unused: every figure reads results/ CSVs, "
                                                   "which correlate.py derives from data/)")
    args = ap.parse_args(argv)
    return make_all(args.results, args.out)


if __name__ == "__main__":
    main()
