#!/usr/bin/env python3
"""
utils.incast — the incast sweep's own path/axis parsing, everything that the
generic utils.paths.SweepPaths cannot express.

Why this exists at all
--------------------------------------------------------------------------------
utils.paths.SweepPaths assumes ONE topology per sweep and a
`output/<domain>/<workload>/<sweep>/<tag>` layout in which the config sub-dir
name and the output sub-dir name are the SAME string (the sweep). The incast
sweep breaks both assumptions, so it gets its own parser here rather than
bending SweepPaths into a shape that would lie for the buffer sweep:

  1. TWO different names for the same sweep. The generator wrote the fabric
     configs under `configs/astra_sim/ns3/incast_sweep/<tag>` but the ns-3 /
     ASTRA output under `output/<domain>/llama2_13b_16reqs_512prompt_incast_sweep/
     <tag>` (a FLAT layout: the tags sit directly in that one directory, with no
     extra <workload>/<sweep> nesting). config_sweep != out_workload, and
     SweepPaths has a single `sweep` field for both.

  2. MORE THAN ONE topology in one sweep. incast_sweep holds T2.1/T3/T4 — three
     topologies with three DIFFERENT placements (prefill TP2/TP4/TP8), each with
     its own buffer sub-sweep. buffer_sweep aborts the moment it sees more than
     one variant, by design. So `IncastPaths` is per-LEVEL: `level="T3"` narrows
     the same on-disk sweep down to exactly one topology's buffer runs, at which
     point buffer_sweep.analyse_sweep can score it unchanged (one definition of
     the metrics, reused — see buffer_compare for the same idea across models).

The swept knob here is the INCAST DEGREE: how many prefill ranks send into each
decode rank at once. That is NOT the prefill TP width -- it is the resharding
ratio tp_prefill / tp_decode, because a decode rank owns 1/tp_decode of each
layer's KV and therefore needs only that fraction of the prefill shards:

    T2.1  prefill TP2 -> decode TP2   fan-in 1   (no incast: the control)
    T3    prefill TP4 -> decode TP2   fan-in 2
    T4    prefill TP8 -> decode TP2   fan-in 4

Confirmed against the traces: each decode rank receives 20, 40 and 80 KV shards
over the 20 layers of its pipeline stage, i.e. 1, 2 and 4 senders per layer.
Neither number is in the directory name (which carries the topology id and the
buffer), so both are read from the placement per level: `fan_in(placement)` for
the degree, `prefill_tp(placement)` for the TP width that names the topology.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .paths import ROOT
from .roles import Placement

# The incast sweep's fixed on-disk names (see module docstring). Both are
# overridable from the CLI, but these are what the generator actually wrote.
CONFIG_SWEEP = "incast_sweep"
OUT_WORKLOAD = "llama2_13b_16reqs_512prompt_incast_sweep"

# A tag is "<level>_bx<rate>_<cc>_buf<buffer>", e.g. "T3_bx100_dcqcn_buf8". The
# level is everything before the first "_bx" token — the topology id (T2.1/T3/T4),
# which is what distinguishes one incast degree from another within the sweep.
_LEVEL = re.compile(r"^(.*?)_bx", re.IGNORECASE)


def level_of(tag: str) -> str | None:
    """The topology/incast level a run tag belongs to ('T3' from
    'T3_bx100_dcqcn_buf8'). None if the tag has no '_bx' token, i.e. it is not a
    tag this sweep produced."""
    m = _LEVEL.match(tag)
    return m.group(1) if m else None


def prefill_tp(placement: Placement) -> int:
    """The tensor-parallel width of the prefill stage -- the number that names
    the topology (T3 = prefill TP4). NOT the incast degree: see fan_in. All
    prefill stages share one TP width, so stage 0's is the number."""
    return len(placement.prefill[0]) if placement.prefill else 0


def decode_tp(placement: Placement) -> int:
    """The tensor-parallel width of a decode stage."""
    return len(placement.decode[0]) if placement.decode else 0


def fan_in(placement: Placement) -> int:
    """THE INCAST DEGREE: how many prefill ranks converge on ONE decode rank for
    a given layer = tp_prefill / tp_decode.

    A decode rank owns 1/tp_decode of every layer's KV heads, so it needs that
    same fraction of the layer's prefill shards and no more: at prefill TP8 into
    decode TP2, each decode rank is fed by 4 of the 8 prefill shards, not 8.
    Reading the prefill TP width as the degree overstates it by tp_decode and
    loses the fact that T2.1 (TP2 -> TP2) has NO incast at all -- it is the
    fan-in-1 control this sweep is measured against.

    0 when either pool is empty; the ratio is exact for the placements this
    sweep uses (utils.config rejects a prefill/decode TP pair that is not a
    multiple), and floor division only guards a malformed placement."""
    tp_p, tp_d = prefill_tp(placement), decode_tp(placement)
    return tp_p // tp_d if tp_p and tp_d else 0


@dataclass(frozen=True)
class IncastPaths:
    """SweepPaths-compatible view of ONE incast level's buffer sub-sweep.

    Implements the same surface buffer_sweep.analyse_sweep and .analyse call on a
    SweepPaths (astra_run/ns3_run/topology/config/tags/missing_roots/describe plus
    the *_root properties and a `sweep` label), so the whole buffer-sweep
    measurement pipeline runs on an incast level with no change — the ONLY
    differences from SweepPaths are the split config/output names and the
    per-level tag filter, both isolated here.
    """
    level: str
    out_workload: str = OUT_WORKLOAD
    config_sweep: str = CONFIG_SWEEP
    root: Path = ROOT

    @property
    def sweep(self) -> str:
        # what analyse_sweep prints when a derived root is missing; carry both
        # names so the message points at the right on-disk place.
        return f"{self.config_sweep}[{self.level}]"

    @property
    def astra_root(self) -> Path:
        return self.root / "output" / "astra_logs" / self.out_workload

    @property
    def ns3_root(self) -> Path:
        return self.root / "output" / "ns3" / self.out_workload

    @property
    def config_root(self) -> Path:
        return self.root / "configs" / "astra_sim" / "ns3" / self.config_sweep

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
        """This level's run tags only. The roots are shared by every level, so
        the level is a prefix filter on their contents. `startswith(level + '_')`
        keeps 'T3' from swallowing a hypothetical 'T31' and keeps 'T2.1' distinct
        from 'T2'."""
        root = {"ns3": self.ns3_root, "astra": self.astra_root,
                "config": self.config_root}[source]
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir()
                      if p.is_dir() and level_of(p.name) == self.level)

    def missing_configs(self) -> list[str]:
        """Config INPUTS absent for tags that have ns-3 output, one line each.
        Configs are what the analysis reasons FROM (topology, buffer, payload):
        with one missing, no number about that run means anything, so the caller
        must Abort on a non-empty return -- unlike a missing OUTPUT, which only
        means the run has not finished and can be skipped (usable_tags)."""
        out = []
        for t in self.tags("ns3"):
            for name, f in (("config.txt", self.config(t)),
                            ("physical_topology.txt", self.topology(t))):
                if not f.is_file():
                    out.append(f"{t}: missing {f}")
        return out

    def usable_tags(self) -> tuple[list[str], list[str]]:
        """(analysable, skipped) by OUTPUT presence alone. A run is analysable
        when every simulation output it needs is on disk: its ns-3 fct/pfc/qlen
        and at least one ASTRA stats_sys*.csv. The generator writes these in
        separate passes, so a half-written run is normal life, not an error:
        it is skipped with a one-line reason for the caller to warn with.
        Config inputs are NOT checked here -- a missing config is a reason to
        Abort, and missing_configs() reports those separately."""
        analysable, skipped = [], []
        for t in self.tags("ns3"):
            missing = [name for name, ok in (
                ("fct.txt", (self.ns3_run(t) / "fct.txt").is_file()),
                ("pfc.txt", (self.ns3_run(t) / "pfc.txt").is_file()),
                ("qlen.txt", (self.ns3_run(t) / "qlen.txt").is_file()),
                ("stats_sys*.csv",
                 any(self.astra_run(t).glob("stats_sys*.csv")))) if not ok]
            if missing:
                skipped.append(f"{t}: missing {', '.join(missing)}")
            else:
                analysable.append(t)
        return analysable, skipped

    def describe(self) -> str:
        return (f"  level    {self.level}\n"
                f"  astra    {self.astra_root}\n"
                f"  ns3      {self.ns3_root}\n"
                f"  configs  {self.config_root}")


def discover_levels(out_workload: str = OUT_WORKLOAD,
                    root: Path = ROOT, source: str = "ns3") -> list[str]:
    """Every incast level present under the sweep's output root, e.g.
    ['T2.1', 'T3', 'T4']. Sorted by name, which for Tk ids is also the incast
    order (T2.1 < T3 < T4); the caller annotates the exact degree from each
    level's recovered placement (fan_in), since the name is not the number."""
    base = root / "output" / _SOURCE_DIR[source] / out_workload
    if not base.is_dir():
        return []
    levels = {level_of(p.name) for p in base.iterdir() if p.is_dir()}
    return sorted(l for l in levels if l is not None)


_SOURCE_DIR = {"astra": "astra_logs", "ns3": "ns3"}
