"""Load device specs from hardware.yaml (each number carries a source + verified flag)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

import yaml

from .models import REPO_ROOT

HARDWARE_YAML = REPO_ROOT / "hw" / "hardware.yaml"


@dataclass(frozen=True)
class Hardware:
    key: str
    name: str
    kind: str
    memory_capacity: float  # bytes
    memory_bandwidth: float  # bytes/s
    peak_flops: Dict[str, float]  # precision -> FLOP/s (dense)
    onchip_sram: Optional[float] = None
    interconnect_bandwidth: Optional[float] = None  # bytes/s per device
    hop_latency: float = 2e-6  # s per collective step (assumption)
    provenance: Dict[str, dict] = field(default_factory=dict, compare=False, repr=False)

    def peak(self, precision: str) -> float:
        if precision not in self.peak_flops:
            raise KeyError(f"{self.key}: no peak for '{precision}' (have {sorted(self.peak_flops)})")
        return self.peak_flops[precision]

    def ridge_point(self, precision: str) -> float:
        """Arithmetic intensity (FLOP/byte) where the roofline turns flat."""
        return self.peak(precision) / self.memory_bandwidth

    def scaled(self, bw: float = 1.0, flops: float = 1.0, capacity: float = 1.0,
               suffix: str = "") -> "Hardware":
        """What-if variant for design-space sweeps."""
        return replace(
            self,
            key=self.key + suffix,
            name=self.name + suffix,
            memory_bandwidth=self.memory_bandwidth * bw,
            memory_capacity=self.memory_capacity * capacity,
            peak_flops={k: v * flops for k, v in self.peak_flops.items()},
        )


def _val(entry):
    """Numeric value of a spec entry. PyYAML (YAML 1.1) reads '1792.0e9' as a
    string because the exponent has no sign, so always coerce with float()."""
    if entry is None or entry.get("value") is None:
        return None
    return float(entry["value"])


def load_hardware(path: Path | str = HARDWARE_YAML) -> Dict[str, Hardware]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    default_lat = _val(doc.get("default_hop_latency")) or 2e-6
    out: Dict[str, Hardware] = {}
    for key, d in doc["devices"].items():
        prov = {k: d[k] for k in ("memory_capacity", "memory_bandwidth", "onchip_sram",
                                  "interconnect_bandwidth", "interconnect_hop_latency") if k in d}
        prov.update({f"peak_{p}": v for p, v in d["peak_flops"].items()})
        out[key] = Hardware(
            key=key,
            name=d["name"],
            kind=d["kind"],
            memory_capacity=float(_val(d["memory_capacity"])),
            memory_bandwidth=float(_val(d["memory_bandwidth"])),
            peak_flops={p: float(v["value"]) for p, v in d["peak_flops"].items()},
            onchip_sram=_val(d.get("onchip_sram")),
            interconnect_bandwidth=_val(d.get("interconnect_bandwidth")),
            hop_latency=float(_val(d.get("interconnect_hop_latency")) or default_lat),
            provenance=prov,
        )
    return out


def provenance_table(hw: Dict[str, Hardware]):
    """Rows of (device, field, value, verified, source) for the README/results."""
    rows = []
    for key, h in hw.items():
        for fld, entry in h.provenance.items():
            rows.append({
                "device": key,
                "field": fld,
                "value": entry.get("value"),
                "verified": entry.get("verified"),
                "source": entry.get("source"),
                "note": entry.get("note"),
            })
    return rows
