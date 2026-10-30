"""Merge Nsight Compute metrics into gemv_sweep.jsonl.

How the pieces line up (see bench/run_microarch.sh):
  1. `gemv_sweep` (timed)          -> gemv_sweep.jsonl            one record per config, in sweep order
  2. `ncu ... gemv_sweep --ncu`    -> gemv_sweep_ncu_configs.jsonl one record per config, each launched
                                      exactly once, with "ncu_launch" = its launch index
                                   -> ncu CSV (--csv --page raw --print-units base), one row per profiled
                                      kernel launch, in launch order (ncu's "ID" column: 0, 1, 2, ...)
  3. this script: ncu row i  <->  config with ncu_launch == i      (match by launch order)
                  config     <->  timed record with the same (kernel, ilp, N, K, requested blocks/SM, block)
     and writes the ncu fields into the timed records (and all configs + ncu fields to --out-ncu).

As a safety check, the kernel name of every ncu row must contain "gemv_<kernel>" of its config (and the
template argument <ilp> when the name shows it); a mismatch means the order is broken and nothing is written.

Usage:
  python3 microarch/merge_ncu.py --sweep data/h100_sxm/gemv_sweep.jsonl \
      --configs data/h100_sxm/gemv_sweep_ncu_configs.jsonl --csv data/h100_sxm/ncu_gemv_sweep.csv \
      [--out-ncu data/h100_sxm/gemv_sweep_ncu.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ncu metric name -> (field written into the JSONL record, kind used for unit conversion)
METRICS = {
    "gpu__time_duration.sum": ("ncu_time_us", "time"),
    "dram__bytes_read.sum": ("ncu_dram_bytes", "bytes"),
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("ncu_dram_pct", "pct"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("ncu_warps_active_pct", "pct"),
    "launch__occupancy_limit_shared_mem": ("ncu_occupancy_limit_shared_mem", "count"),
    "launch__registers_per_thread": ("ncu_registers_per_thread", "count"),
}

# Unit -> factor to our unit (time in microseconds, bytes in bytes). --print-units base should give
# nsecond / byte, but convert anything else ncu might print instead of trusting it.
TIME_TO_US = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3,
              "second": 1e6, "s": 1e6}
BYTES_TO_B = {"byte": 1.0, "b": 1.0, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12,
              "kibyte": 1024.0, "mibyte": 1024.0 ** 2, "gibyte": 1024.0 ** 3}

NAME_COLUMNS = ("Kernel Name", "Function Name", "Name")


def _num(s: str) -> Optional[float]:
    """Parse an ncu CSV value: strip thousands separators; return None for n/a or empty."""
    s = (s or "").strip().replace(",", "")
    if s in ("", "n/a", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ncu_csv(text: str) -> List[Dict]:
    """Rows of an `ncu --csv --page raw` report as dicts {"id", "name", <our field>: value}, in ID order.

    Lines starting with '==' (==PROF==, ==WARNING==, ...) and anything before the header row (e.g. the
    application's own stdout) are ignored. A units row (empty ID) directly after the header is used for
    unit conversion.
    """
    lines = [ln for ln in text.splitlines() if not ln.startswith("==")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header_i = next((i for i, r in enumerate(rows)
                     if r and r[0].strip() == "ID" and any(c in r for c in NAME_COLUMNS)), None)
    if header_i is None:
        return []
    header = rows[header_i]
    name_col = next(c for c in NAME_COLUMNS if c in header)
    col = {h: j for j, h in enumerate(header)}
    units: Dict[str, str] = {}
    out = []
    for r in rows[header_i + 1:]:
        if len(r) < len(header):
            continue
        rid = _num(r[col["ID"]])
        if rid is None:                       # units row: ID empty, metric columns hold unit names
            if not out and not units:
                units = {h: r[j].strip() for h, j in col.items()}
            continue
        rec = {"id": int(rid), "name": r[col[name_col]]}
        for metric, (field, kind) in METRICS.items():
            if metric not in col:
                continue
            v = _num(r[col[metric]])
            if v is None:
                continue
            unit = units.get(metric, "").lower()
            if kind == "time":
                v *= TIME_TO_US.get(unit, 1e-3)    # default: --print-units base -> nanoseconds
            elif kind == "bytes":
                v *= BYTES_TO_B.get(unit, 1.0)
            rec[field] = v
        out.append(rec)
    return sorted(out, key=lambda d: d["id"])


def read_jsonl(path) -> List[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def config_key(r: dict) -> Tuple:
    """Identity of a sweep config, the same in the timed and the --ncu runs."""
    return (r["kernel"], r["ilp"], r["N"], r["K"], r.get("requested_blocks_per_sm"), r.get("threads_per_block"))


def name_matches(ncu_name: str, cfg: dict) -> bool:
    """Does the profiled kernel's name agree with the config it is matched to?"""
    m = re.search(r"gemv_(v\d)", ncu_name)
    if not m or m.group(1) != cfg["kernel"]:
        return False
    t = re.search(r"gemv_v\d\s*<\s*(\d+)", ncu_name)        # template argument, if the name shows it
    return t is None or int(t.group(1)) == int(cfg["ilp"])


def merge(sweep: List[dict], configs: List[dict], ncu_rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Return (sweep records with ncu fields added, configs with ncu fields added)."""
    configs = sorted((c for c in configs if c.get("ncu_launch", -1) >= 0), key=lambda c: c["ncu_launch"])
    if len(ncu_rows) != len(configs):
        print(f"merge_ncu: warning: {len(ncu_rows)} ncu rows vs {len(configs)} configs; "
              f"matching the first {min(len(ncu_rows), len(configs))} by launch order", file=sys.stderr)
    by_launch = {c["ncu_launch"]: c for c in configs}
    merged_cfgs = []
    for i, row in enumerate(ncu_rows):
        cfg = by_launch.get(i)
        if cfg is None:
            continue
        if not name_matches(row["name"], cfg):
            raise RuntimeError(f"launch order mismatch at ncu row {i}: kernel '{row['name']}' "
                               f"vs config {cfg['kernel']} ilp={cfg['ilp']}")
        merged_cfgs.append({**cfg, **{k: v for k, v in row.items() if k.startswith("ncu_")}})

    ncu_by_key = {config_key(c): c for c in merged_cfgs}
    out = []
    for r in sweep:
        c = ncu_by_key.get(config_key(r)) if "kernel" in r else None
        if c is not None:
            r = {**r, **{k: v for k, v in c.items() if k.startswith("ncu_") and k != "ncu_launch"}}
        out.append(r)
    return out, merged_cfgs


def write_jsonl(path, records: List[dict]) -> None:
    """Write atomically (temp file + rename) so a crash never leaves a half-written gemv_sweep.jsonl."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True, help="timed gemv_sweep.jsonl (updated in place)")
    ap.add_argument("--configs", required=True, help="stdout of `gemv_sweep --ncu` (JSONL)")
    ap.add_argument("--csv", required=True, nargs="+",
                    help="ncu CSV output file(s); the first one that contains an ncu table is used")
    ap.add_argument("--out-ncu", help="also write every --ncu config with its ncu fields here")
    args = ap.parse_args(argv)

    ncu_rows: List[dict] = []
    for f in args.csv:
        if Path(f).exists():
            ncu_rows = parse_ncu_csv(Path(f).read_text(errors="replace"))
            if ncu_rows:
                break
    if not ncu_rows:
        raise SystemExit(f"merge_ncu: no ncu table found in {args.csv}")
    sweep = read_jsonl(args.sweep)
    configs = read_jsonl(args.configs)
    merged, merged_cfgs = merge(sweep, configs, ncu_rows)
    n_hit = sum(1 for r in merged if "ncu_time_us" in r)
    if sweep:
        write_jsonl(args.sweep, merged)
    if args.out_ncu:
        write_jsonl(args.out_ncu, merged_cfgs)
    print(f"merge_ncu: {len(ncu_rows)} ncu rows, {len(merged_cfgs)} configs matched, "
          f"{n_hit}/{len(sweep)} timed records enriched", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
