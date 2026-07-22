#!/usr/bin/env python3
"""
utils.paths — every path in the project, derived from one name.

The layout is fixed by the generation side, so it is a constant here and not a
flag. Only the two names that actually vary are inputs:

    <ROOT>/output/astra_logs/<workload>/<sweep>/<tag>/stats_sys*.csv
    <ROOT>/output/ns3/<workload>/<sweep>/<tag>/{fct,pfc,qlen}.txt
    <ROOT>/configs/astra_sim/ns3/<sweep>/<tag>/{physical_topology.txt,config.txt}
    <ROOT>/results/sweep_analysis/<kind>/<sweep>/<workload>/        <- default out

`--sweep` is the one required argument. There is no template string, no {tag}
substitution and no directory search: a path either exists at the derived
location or the run is not there, and being told which of the three roots is
missing is more useful than a fallback that finds the wrong file.

SweepAxis lives here too. Reading `4.0` out of `T1_bx200_dcqcn_buf4` is a
question about a directory name, which is what this module is about; it was the
last survivor of utils/sweep.py, whose other six functions were the directory
search this replaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/home/andre/tesi/trace_evaluator")
WORKLOAD = "llama2_13b_p-tp2pp2_d-tp2pp2_stream_16reqs_512prompt"


@dataclass(frozen=True)
class SweepPaths:
    sweep: str
    workload: str = WORKLOAD
    root: Path = ROOT

    @property
    def astra_root(self) -> Path:
        return self.root / "output" / "astra_logs" / self.workload / self.sweep

    @property
    def ns3_root(self) -> Path:
        # keyed by workload (model), mirroring astra_root, so runs of different
        # models on the same fabric no longer overwrite each other's ns-3 output.
        return self.root / "output" / "ns3" / self.workload / self.sweep

    @property
    def config_root(self) -> Path:
        return self.root / "configs" / "astra_sim" / "ns3" / self.sweep

    def astra_run(self, tag: str) -> Path:
        return self.astra_root / tag

    def ns3_run(self, tag: str) -> Path:
        return self.ns3_root / tag

    def topology(self, tag: str) -> Path:
        return self.config_root / tag / "physical_topology.txt"

    def config(self, tag: str) -> Path:
        return self.config_root / tag / "config.txt"

    def missing_roots(self) -> list[str]:
        return [f"{name}: {p}" for name, p in
                (("astra_root", self.astra_root), ("ns3_root", self.ns3_root),
                 ("config_root", self.config_root)) if not p.is_dir()]

    def tags(self, source: str = "ns3") -> list[str]:
        """Run tags. `source` says which root defines the run set: 'ns3' for a
        fabric question, 'astra' for a compute one. They should agree; if they
        do not, that is a missing run and the caller must say so."""
        root = {"ns3": self.ns3_root, "astra": self.astra_root,
                "config": self.config_root}[source]
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    def describe(self) -> str:
        return (f"  sweep    {self.sweep}\n"
                f"  workload {self.workload}\n"
                f"  astra    {self.astra_root}\n"
                f"  ns3      {self.ns3_root}\n"
                f"  configs  {self.config_root}")


class SweepAxis:
    """The swept knob. `value` reads it out of a run-dir name; `variant` blanks it
    out, so that a SECOND moving knob becomes its own series instead of silently
    averaging into the first."""

    def __init__(self, token: str):
        self.token = token
        self._re = re.compile(rf"{token}(\d+(?:\.\d+)?)", re.IGNORECASE)

    def value(self, tag: str) -> float | None:
        m = self._re.search(tag)
        return float(m.group(1)) if m else None

    def variant(self, tag: str) -> str:
        return self._re.sub(f"{self.token}*", tag)


BANDWIDTH_AXIS = SweepAxis("bx")
BUFFER_AXIS = SweepAxis("buf")

# The generation side writes bx straight into physical_topology.txt as Gbps
# (e.g. bx400 -> "400Gbps" on the leaf links), but the ASTRA-sim CSVs report
# bw_bytes_per_ns, i.e. GB/s decimal (bytes/ns). The two are off by a factor of
# 8 (bits vs bytes), not by anything about the run: comparing a CSV bandwidth
# column to BANDWIDTH_AXIS.value(tag) 1:1 -- an "ideal y=x" line, a delivered/
# nominal ratio -- silently compares Gbps to GB/s unless divided by this first.
BANDWIDTH_GBPS_TO_BYTES_PER_NS = 1 / 8

# The generation side writes each domain's runs under a fixed output root.
_SOURCE_ROOT = {"astra": "astra_logs", "ns3": "ns3"}


def discover_workloads(root: Path, sweep: str, source: str = "astra") -> list[str]:
    """Every workload that ran `sweep`: the sub-directories of output/<source>
    that contain a `<workload>/<sweep>` directory. `source` is 'astra'
    (output/astra_logs, the compute domain) or 'ns3' (output/ns3, the fabric
    domain). The cross-model companions use this to find who to overlay."""
    base = Path(root) / "output" / _SOURCE_ROOT[source]
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and (p / sweep).is_dir())


def add_arguments(ap, kind: str) -> None:
    """The four path flags every sweep analyzer shares. `kind` only picks the
    default output sub-directory."""
    ap.add_argument("--sweep", required=True,
                    help="sweep sub-directory name, e.g. 'buffer_sweep_T1'. "
                         "Every other path is derived from it.")
    ap.add_argument("--workload", default=WORKLOAD,
                    help=f"workload dir under output/astra_logs (default: {WORKLOAD})")
    ap.add_argument("--root", default=str(ROOT), type=Path,
                    help=f"project root (default: {ROOT})")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help=f"output dir (default: results/sweep_analysis/{kind}/"
                         f"<sweep>/<workload>)")