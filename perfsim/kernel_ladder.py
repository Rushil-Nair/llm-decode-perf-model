"""H100 GEMV kernel ladder (v2-v7): timed bandwidth, Nsight Compute counters, and the model's memory ceiling.

    python -m perfsim.kernel_ladder --run data_kernels/h100_lium2 --out results

Reads <run>/gemv_sweep.jsonl (timed sweep, every kernel x ILP x blocks/SM x shape) and
<run>/ncu_deep/*.raw.csv (Nsight Compute --set full on the best config of each kernel), and writes
results/h100_kernel_ladder.md + .csv. The memory ceiling is the time a purely memory-bound kernel would take:
t = t0 + bytes / BW_mem, with BW_mem the loaded-latency Little's Law bound (perfsim.microarch) at that config's
bytes in flight (latency, peak and alpha from <run>/probes.jsonl) and t0 the per-launch cost fitted by
perfsim.gemv_model on --t0-from (default: the first Lium run, made before v6/v7 existed; this is the ceiling
used in results/v6v7_prediction.md).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from perfsim import gemv_model as G
from perfsim import microarch as M

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "microarch"))
from ncu_deep_report import read_raw  # noqa: E402

KERNELS = ["v2", "v3", "v4", "v5", "v6", "v7"]
WHAT = {"v2": "x read through L1", "v3": "x in shared memory, persistent grid", "v4": "v3 + streaming weight loads",
        "v5": "v4, int4->fp16 bit trick (half the instructions)", "v6": "v4, swizzled x (no bank conflicts)",
        "v7": "v5 + v6: both fixes"}
NCU = {"bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
       "shared_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
       "issue_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
       "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
       "stall_shared": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
       "stall_dram": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
       "regs": "launch__registers_per_thread", "warp_inst": "smsp__inst_executed.sum"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="data_kernels/h100_lium2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--shape", default="gate_proj")
    ap.add_argument("--t0-from", default="data_kernels/h100_lium")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    gpu = G.GPU(run)
    peak = gpu.params.peak_bw_Bps / 1e9
    t0 = G.fit(G.GPU(Path(a.t0_from)))[2]
    raws = {p.name.split("_")[0]: read_raw(p) for p in sorted((run / "ncu_deep").glob(f"v*_{a.shape}_*.raw.csv"))}
    rows = []
    for k in KERNELS:
        rs = [r for r in gpu.records if r["kernel"] == k and r["shape"] == a.shape]
        if not rs:
            continue
        b = max(rs, key=lambda r: r["GBps"])
        bw = float(M.predict_bw(gpu.params, M.record_bytes_in_flight(b, gpu.params), loaded_latency=True,
                                alpha=gpu.alpha))
        mem = b["bytes"] / (t0 + b["bytes"] / bw) / 1e9
        d = raws.get(k, {})
        row = {"kernel": k, "change": WHAT[k], "ilp": b["ilp"], "blocks_per_sm": b["blocks_per_sm"],
               "GBps": b["GBps"], "pct_of_peak": 100 * b["GBps"] / peak, "mem_ceiling_GBps": mem,
               "max_rel_err": max(r["max_rel_err"] for r in rs)}
        for name, metric in NCU.items():
            v = d.get(metric)
            row[name] = v if isinstance(v, float) else None
        NK = b["N"] * b["K"]
        row["warp_inst_per_group"] = row["warp_inst"] / (NK / 32) if row["warp_inst"] else None
        rows.append(row)
    base = next(r["GBps"] for r in rows if r["kernel"] == "v4")
    for r in rows:
        r["vs_v4"] = r["GBps"] / base

    def f(v, fmt):
        return "n/a" if v is None else fmt.format(v)

    L = [f"# H100 GEMV kernel ladder ({a.shape}, {run})", "",
         f"Stream peak {peak:.0f} GB/s; DRAM latency {gpu.params.dram_latency_ns:.0f} ns; busy SM clock "
         f"{gpu.f_sm / 1e6:.0f} MHz; launch cost t0 = {t0 * 1e6:.2f} us. Best timed config per kernel; counters from "
         "Nsight Compute --set full on that config. Memory ceiling = bytes / (t0 + bytes / BW_mem), BW_mem = "
         "loaded-latency Little's Law at that config's bytes in flight.", "",
         "| kernel | change | best config | GB/s | % of peak | vs v4 | memory ceiling | warp instr / group | "
         "issue busy % | shared bank conflicts | shared wavefronts | stall: shared mem | stall: DRAM | "
         "occupancy % | regs |", "|---" * 15 + "|"]
    for r in rows:
        L.append(f"| {r['kernel']} | {r['change']} | ILP {r['ilp']}, {r['blocks_per_sm']} blocks/SM | {r['GBps']:.0f} | "
                 f"{r['pct_of_peak']:.0f}% | {r['vs_v4']:.2f}x | {r['mem_ceiling_GBps']:.0f} | "
                 f"{f(r['warp_inst_per_group'], '{:.2f}')} | {f(r['issue_pct'], '{:.0f}')} | "
                 f"{f(r['bank_conflicts'], '{:,.0f}')} | {f(r['shared_wavefronts'], '{:,.0f}')} | "
                 f"{f(r['stall_shared'], '{:.2f}')} | {f(r['stall_dram'], '{:.2f}')} | "
                 f"{f(r['occupancy_pct'], '{:.0f}')} | {f(r['regs'], '{:.0f}')} |")
    L.append("")
    (out / f"h100_kernel_ladder.md").write_text("\n".join(L) + "\n")
    with open(out / "h100_kernel_ladder.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
