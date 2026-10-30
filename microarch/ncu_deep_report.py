"""Summarize the --set full Nsight Compute profiles written by bench/run_ncu_deep.sh.

    python microarch/ncu_deep_report.py data/<gpu_tag> results/ncu_deep_<gpu_tag>.md

Reads data/<tag>/ncu_deep/*.raw.csv (ncu --page raw --csv --print-units base) and prints one column per
profiled kernel: time, DRAM and SM throughput, issue activity, the busiest execution pipes, the top warp
stall reasons, and executed instructions per 32-weight group (to compare with microarch/sass_count.py).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _num(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def read_raw(path: Path) -> Dict[str, object]:
    """First data row of an ncu raw CSV as {metric: value}; strings kept for non-numeric columns."""
    rows = list(csv.reader(open(path, newline="")))
    rows = [r for r in rows if r and not r[0].startswith("==")]
    if len(rows) < 3:
        return {}
    head, data = rows[0], rows[2]          # rows[1] holds units
    out: Dict[str, object] = {}
    for k, v in zip(head, data):
        n = _num(v)
        out[k] = n if n is not None else v
    return out


def pick(d: Dict[str, object], *names: str) -> Optional[float]:
    for n in names:
        v = d.get(n)
        if isinstance(v, float):
            return v
    return None


PIPE_RE = re.compile(r"^sm__inst_executed_pipe_(\w+?)\.avg\.pct_of_peak_sustained_active$")
PIPE_RE2 = re.compile(r"^sm__pipe_(\w+?)_cycles_active\.avg\.pct_of_peak_sustained_active$")
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")


def summarize(name: str, d: Dict[str, object]) -> Dict[str, object]:
    m = re.match(r"(v\d)_(\w+?)_ilp(\d)_b(\d+)", name)
    shape = m.group(2) if m else ""
    NK = {"q_proj": 4096 * 4096, "gate_proj": 12288 * 4096, "down_proj": 4096 * 12288}.get(shape)
    t = pick(d, "gpu__time_duration.sum")
    dram_b = pick(d, "dram__bytes_read.sum", "dram__bytes.sum")
    inst = pick(d, "smsp__inst_executed.sum", "sm__inst_executed.sum")
    pipes = {}
    for k, v in d.items():
        mm = PIPE_RE.match(k) or PIPE_RE2.match(k)
        if mm and isinstance(v, float):
            key = mm.group(1)
            pipes[key] = max(v, pipes.get(key, 0.0))
    stalls = {mm.group(1): v for k, v in d.items() if (mm := STALL_RE.match(k)) and isinstance(v, float)}
    return {
        "name": name,
        "time_us": t * 1e-3 if t else None,                        # base unit ns
        "dram_read_GBps": dram_b / t if (dram_b and t) else None,  # bytes/ns = GB/s
        "dram_pct": pick(d, "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_pct": pick(d, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "issue_active_pct": pick(d, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
        "ipc": pick(d, "sm__inst_executed.avg.per_cycle_active"),
        "occupancy_pct": pick(d, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "sm_clock_MHz": (pick(d, "smsp__cycles_elapsed.avg.per_second", "sm__cycles_elapsed.avg.per_second") or 0) / 1e6
                        or None,
        "warp_instr_per_group": inst / (NK / 32) if (inst and NK) else None,
        "pipes": dict(sorted(pipes.items(), key=lambda x: -x[1])[:6]),
        "stalls": dict(sorted(stalls.items(), key=lambda x: -x[1])[:5]),
    }


def fmt(v, f="{:.1f}"):
    return "n/a" if v is None else f.format(v)


def main(argv: List[str]) -> None:
    data_dir = Path(argv[1]); out = Path(argv[2]) if len(argv) > 2 else None
    files = sorted((data_dir / "ncu_deep").glob("*.raw.csv"))
    if not files:
        print(f"no {data_dir}/ncu_deep/*.raw.csv"); return
    S = [summarize(f.name.replace(".raw.csv", ""), read_raw(f)) for f in files]
    L = [f"# Nsight Compute (--set full): {data_dir.name}", "",
         "| metric | " + " | ".join(s["name"] for s in S) + " |", "|---" * (len(S) + 1) + "|"]
    for label, key, f in [("kernel time (us)", "time_us", "{:.1f}"), ("DRAM read (GB/s)", "dram_read_GBps", "{:.0f}"),
                          ("DRAM throughput (% of peak)", "dram_pct", "{:.1f}"),
                          ("SM throughput (% of peak)", "sm_pct", "{:.1f}"),
                          ("issue slots busy (%)", "issue_active_pct", "{:.1f}"), ("IPC (per SM)", "ipc", "{:.2f}"),
                          ("achieved occupancy (%)", "occupancy_pct", "{:.1f}"),
                          ("SM clock (MHz)", "sm_clock_MHz", "{:.0f}"),
                          ("warp instructions per 32-weight group", "warp_instr_per_group", "{:.2f}")]:
        L.append(f"| {label} | " + " | ".join(fmt(s[key], f) for s in S) + " |")
    L.append("| busiest pipes (% of peak) | " + " | ".join(
        "<br>".join(f"{k} {v:.0f}" for k, v in s["pipes"].items()) or "n/a" for s in S) + " |")
    L.append("| top stall reasons (warps per issue) | " + " | ".join(
        "<br>".join(f"{k} {v:.2f}" for k, v in s["stalls"].items()) or "n/a" for s in S) + " |")
    text = "\n".join(L) + "\n"
    if out:
        out.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
