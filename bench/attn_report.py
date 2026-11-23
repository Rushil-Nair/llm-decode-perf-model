"""Report for the paged decode-attention benchmark. Reads data/<tag>/attention/ for every GPU that has it.

    python3 bench/attn_report.py                 # prints markdown, writes results/attention.md + figures

Everything printed is computed from the measured rows; nothing is filled in by hand.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from perfsim import attn_model as am  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRITON = ("triton_splitkv", "triton_balanced", "triton_balanced_v2")
MAIN = (16, "float16")       # headline setting: page size 16, fp16


def _fmt(s):
    return f"{s['geomean']:.2f}x ({s['min']:.2f}-{s['max']:.2f}, n={s['n']})" if s else "-"


def _setting(rows, ps, dt):
    return [r for r in rows if r.get("page_size") == ps and r.get("dtype") == dt]


def main():
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "data", "*", "attention")) if os.path.isdir(d))
    if not dirs:
        print("no data/<gpu_tag>/attention directories yet: run gpu/run_attention.sh on a GPU first")
        return
    gpus = {}
    for d in dirs:
        tag = d.split(os.sep)[-2]
        env = am.load_env(d)
        rows = [r for r in am.best_flashinfer(am.load_rows(d)) if r.get("mode") == "graph"]
        gpus[tag] = (env, rows)

    L = []
    L.append("## Paged GQA decode attention vs FlashInfer\n")
    L.append("Speed = FlashInfer time / our time (above 1: ours is faster). FlashInfer = the faster of its "
             "tensor-core and CUDA-core decode paths per shape. Uniform: every request has the same length; "
             "skewed: one request at the full context, the rest at 1/16 of it.\n")
    L.append("| GPU | setting | kernel | uniform batches | skewed batches |")
    L.append("|---|---|---|---|---|")
    for tag, (env, rows) in gpus.items():
        settings = sorted({(r.get("page_size"), r.get("dtype")) for r in rows}, key=lambda s: s != MAIN)
        for ps, dt in settings:
            rs = _setting(rows, ps, dt)
            for k in TRITON:
                if not any(r["kernel"] == k for r in rs):
                    continue
                s = am.ratio_summary(rs, num=k, den="flashinfer")
                L.append(f"| {tag} | page {ps}, {dt} | {k} | {_fmt(s.get('uniform'))} | {_fmt(s.get('skewed'))} |")

    L.append("\n## Effect of each scheduling change (same kernel body, page 16 fp16)\n")
    L.append("| GPU | comparison | uniform batches | skewed batches |")
    L.append("|---|---|---|---|")
    for tag, (env, rows) in gpus.items():
        rs = _setting(rows, *MAIN)
        for num, den, label in (("triton_balanced", "triton_splitkv", "balanced v1 vs uniform splitting"),
                                ("triton_balanced_v2", "triton_balanced", "balanced v2 vs balanced v1")):
            if any(r["kernel"] == num for r in rs):
                s = am.ratio_summary(rs, num=num, den=den)
                L.append(f"| {tag} | {label} | {_fmt(s.get('uniform'))} | {_fmt(s.get('skewed'))} |")

    for tag, (env, rows) in gpus.items():
        rs = _setting(rows, *MAIN)
        if not any(r["kernel"] == "triton_balanced_v2" for r in rs):
            continue
        by = {}
        for r in rs:
            by.setdefault((r["config"], r["batch"], r["ctx"], bool(r.get("skew"))), {})[r["kernel"]] = r
        cmp_ = [(v["flashinfer"]["median_us"] / v["triton_balanced"]["median_us"], k, v) for k, v in by.items()
                if {"flashinfer", "triton_balanced", "triton_balanced_v2"} <= set(v)]
        cmp_.sort(key=lambda x: x[0])
        L.append(f"\n## {tag}: the 10 shapes where balanced v1 was slowest vs FlashInfer, before and after\n")
        L.append("| head config | batch | context | skewed | v1 vs FlashInfer | v2 vs FlashInfer | FlashInfer (us) "
                 "| v2 (us) | v2 work items / chunk / merge |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r1, (cfg, b, ctx, skew), v in cmp_[:10]:
            f, two = v["flashinfer"]["median_us"], v["triton_balanced_v2"]
            L.append(f"| {cfg} | {b} | {ctx} | {'yes' if skew else 'no'} | {r1:.2f}x | {f / two['median_us']:.2f}x | "
                     f"{f:.1f} | {two['median_us']:.1f} | {two.get('splits')} / {two.get('chunk')} / "
                     f"{'direct' if two.get('direct') else ('tree' if two.get('tree') else 'no')} |")
        occ = next((r for r in rs if r["kernel"] == "triton_balanced_v2" and r.get("ctas_per_sm")), None)
        if occ:
            L.append(f"\nv2 occupancy query: {occ['ctas_per_sm']} programs per SM ({occ.get('n_regs')} registers/thread, "
                     f"{occ.get('smem')} B shared memory, {occ.get('n_spills')} spills).")

    for d in dirs:
        tj = os.path.join(d, "tuned.json")
        if not os.path.exists(tj):
            continue
        with open(tj) as fh:
            t = json.load(fh)
        tag = d.split(os.sep)[-2]
        L.append(f"\n## {tag}: tuning and ablation on separate shapes (n={t['n_shapes']}, none in the matrix)\n")
        L.append("Geomean FlashInfer time / ours over the tuning shapes (above 1: ours is faster).\n")
        L.append("| version | speed vs FlashInfer |")
        L.append("|---|---|")
        names = {"v1": "v1 (first H100 run)", "v1+wave": "+ wave planner (occupancy-aware chunk size)",
                 "v1+wave+blocked": "+ blocked merge", "v1+wave+blocked+direct": "+ direct path (no merge when unsplit)",
                 "v1+wave+blocked+direct+tree": "+ two-level merge (requests with many chunks)"}
        for k, v in t["ablation"].items():
            L.append(f"| {names.get(k, k)} | {v:.3f}x |")
        c = t["config"]
        L.append(f"| + tuned tile and planner ({c['block_n']} tokens/step, {c['num_warps']} warps, {c['num_stages']} "
                 f"stages, split penalty {c['split_penalty']}, overhead {c['program_overhead']}, two-level merge "
                 f"above {c.get('tree_min_chunks') or 'never'} chunks) | {t['tuned_score']:.3f}x |")
        top = sorted(t["tile_scores"].items(), key=lambda x: -x[1])
        L.append("\nTile configs (block_n_warps_stages): " + ", ".join(f"{k[5:]} {v:.3f}x" for k, v in top))

    L.append("\n## Attention term: t = t0 + bytes / BW_eff (fitted on uniform batches, page 16, fp16)\n")
    L.append("Copy BW is a device-to-device copy (read + write bytes / time). Read-only streams can exceed "
             "it slightly, so values just above 100% mean 'at the memory limit', not faster than memory.\n")
    L.append("| GPU | kernel | t0 (us) | BW_eff (GB/s) | % of copy BW | in-sample MAPE | holdout MAPE "
             "(unseen head configs) | skewed batches MAPE | zero-fit roofline MAPE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    fits = {}
    for tag, (env, rows) in gpus.items():
        rs = _setting(rows, *MAIN)
        uni = [r for r in rs if not r.get("skew")]
        skw = [r for r in rs if r.get("skew")]
        for k in TRITON + ("flashinfer",):
            try:
                f = am.fit(uni, k, env["copy_gbps"])
            except ValueError:
                continue
            fits[(tag, k)] = f
            pred = lambda r, f=f: f.predict_us(r["bytes"])  # noqa: E731
            ins = am.mape([r for r in uni if r["kernel"] == k], pred)
            sk = am.mape([r for r in skw if r["kernel"] == k], pred)
            try:
                ho = am.holdout(uni, k, env["copy_gbps"])["score"]
                ho_s = f"{ho['mape']:.1f}% (max {ho['max']:.0f}%, n={ho['n']})"
            except ValueError:
                ho_s = "-"
            roof = am.mape([r for r in uni if r["kernel"] == k],
                           lambda r: r["bytes"] / (env["copy_gbps"] * 1e9) * 1e6)
            L.append(f"| {tag} | {k} | {f.t0_us:.1f} | {f.bw_gbps:.0f} | {100 * f.bw_frac:.0f}% | "
                     f"{ins['mape']:.1f}% | {ho_s} | {sk['mape']:.1f}% (n={sk['n']}) | {roof['mape']:.1f}% |")

    tags = list(gpus)
    if len(tags) == 2:
        L.append("\n## Leave-one-GPU-out (BW_eff as a fraction of copy BW, t0 carried over)\n")
        L.append("| fitted on | predicted on | kernel | MAPE | max |")
        L.append("|---|---|---|---|---|")
        for a, b in (tags, tags[::-1]):
            env_b, rows_b = gpus[b]
            uni_b = [r for r in _setting(rows_b, *MAIN) if not r.get("skew")]
            for k in TRITON + ("flashinfer",):
                if (a, k) not in fits:
                    continue
                f = fits[(a, k)]
                s = am.mape([r for r in uni_b if r["kernel"] == k],
                            lambda r, f=f: f.predict_us(r["bytes"], env_b["copy_gbps"]))
                L.append(f"| {a} | {b} | {k} | {s['mape']:.1f}% | {s['max']:.0f}% |")

    out = "\n".join(L) + "\n"
    print(out)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "attention.md"), "w") as fh:
        fh.write(out)
    try:
        _figures(gpus)
    except Exception as e:  # noqa: BLE001
        print(f"(figures skipped: {e})")


def _figures(gpus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kernels = ("triton_splitkv", "triton_balanced", "triton_balanced_v2", "flashinfer", "sdpa_gather", "sdpa_contig")
    fig, axes = plt.subplots(len(gpus), 2, figsize=(12, 4 * len(gpus)), squeeze=False)
    for row, (tag, (env, rows)) in zip(axes, gpus.items()):
        rs = _setting(rows, *MAIN)
        for ax, (b, skew, title) in zip(row, ((1, False, "batch 1"), (16, True, "batch 16, skewed"))):
            for k in kernels:
                pts = sorted((r["ctx"], r["median_us"]) for r in rs
                             if r["kernel"] == k and r["config"] == "llama8b" and r["batch"] == b
                             and bool(r.get("skew")) == skew)
                if pts:
                    ax.plot(*zip(*pts), marker="o", label=k)
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("context length (tokens)"); ax.set_ylabel("latency (us)")
            ax.set_title(f"{tag}: Llama-8B heads (32/8, d=128), {title}"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "attention_latency.png"), dpi=150)

    split_rows = {}
    for d in glob.glob(os.path.join(ROOT, "data", "*", "attention")):
        rs = am.load_rows(d, mode="splits")
        if rs:
            split_rows[d.split(os.sep)[-2]] = rs
    if split_rows:
        fig, axes = plt.subplots(1, len(split_rows), figsize=(6 * len(split_rows), 4), squeeze=False)
        for ax, (tag, rs) in zip(axes[0], split_rows.items()):
            for (b, ctx) in sorted({(r["batch"], r["ctx"]) for r in rs}):
                pts = sorted((r["splits"], r["median_us"]) for r in rs if r["batch"] == b and r["ctx"] == ctx)
                h = next(r for r in rs if r["batch"] == b and r["ctx"] == ctx)["heuristic_splits"]
                line, = ax.plot(*zip(*pts), marker=".", label=f"B={b} ctx={ctx}")
                ax.plot([h], [dict(pts)[h]], "*", ms=12, color=line.get_color())
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("num splits (star = heuristic)"); ax.set_ylabel("latency (us)")
            ax.set_title(f"{tag}: split sweep (uniform splitting)"); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "figures", "attention_splits.png"), dpi=150)


if __name__ == "__main__":
    main()
