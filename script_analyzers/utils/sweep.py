#!/usr/bin/env python3
"""
utils.sweep — the mechanics every sweep shares, and nothing else.

A sweep is a directory of run sub-directories whose names encode one moving knob:

    T2_bx25_dcqcn_buf32/     bandwidth sweep: 'bx' is the axis, the rest is the variant
    T1_bx100_dcqcn_buf8/     buffer sweep:    'buf' is the axis, the rest is the variant

Only the token differs, so the axis is a parameter, not a copy of the code. Anything
that interprets the runs belongs in the analyzer, not here: this module knows how to
FIND things, never what they mean.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class SweepAxis:
    """The swept knob. `value` reads it out of a run-dir name; `variant` blanks it
    out so that a second moving knob becomes its own series rather than silently
    averaging into the first."""

    def __init__(self, token: str, column: str, unit: str = ""):
        self.token = token
        self.column = column                      # column name in summary.csv
        self.unit = unit                          # for axis labels
        self._re = re.compile(rf"{token}(\d+(?:\.\d+)?)", re.IGNORECASE)

    def value(self, tag: str) -> float | None:
        m = self._re.search(tag)
        return float(m.group(1)) if m else None

    def variant(self, tag: str) -> str:
        return self._re.sub(f"{self.token}*", tag)


BANDWIDTH_AXIS = SweepAxis("bx", "bandwidth", "bx")
BUFFER_AXIS = SweepAxis("buf", "buffer_mb", "MiB")


def discover_runs(root: Path, outdir: Path, skip_names: tuple[str, ...] = ()) -> list[Path]:
    """Run sub-directories of `root`, never including the output directory itself
    (which is often written inside the sweep root and would otherwise be scanned
    as if it were a run)."""
    resolved = outdir.resolve()
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name not in skip_names
                  and p.resolve() != resolved)


def resolve_outdir(explicit: str | None, default: str | None,
                   root: Path, fallback_name: str) -> Path:
    """--out flag > module default constant > <root>/<fallback_name>."""
    if explicit:
        return Path(explicit)
    if default:
        return Path(default)
    return root / fallback_name


def resolve_ns3_root(astra_root: Path, explicit: str | None,
                     default: str | None = None) -> Path | None:
    """Where the ns-3 outputs live. --ns3-root flag > module default constant >
    the ASTRA root itself, if fct.txt files are already sitting under it."""
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    if default and Path(default).is_dir():
        return Path(default)
    return astra_root if any(astra_root.rglob("fct.txt")) else None


def find_ns3_run(root: Path | None, tag: str) -> Path | None:
    """The ns-3 output dir matching an ASTRA run dir, by tag."""
    if root is None:
        return None
    for cand in [root / tag, *root.rglob(tag)]:
        if cand.is_dir() and ((cand / "fct.txt").is_file() or (cand / "pfc.txt").is_file()):
            return cand
    return None


def project_roots(*starts: Path | str, depth: int = 8) -> list[Path]:
    """Every ancestor of each start directory, nearest first, de-duplicated.

    A run lives at <root>/output/ns3/<tag> while its config lives at
    <root>/configs/astra_sim/ns3/... -- they only share the project root, which is
    several levels up. Walking ancestors is what lets the config be found without
    a flag; not walking them silently drops the analyzer into degraded mode, which
    is far worse than not finding the file at all."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for start in starts:
        if start is None:
            continue
        d = Path(start).resolve()
        for _ in range(depth):
            if d not in seen:
                seen.add(d)
                roots.append(d)
            if d.parent == d:
                break
            d = d.parent
    return roots


def find_under_roots(roots: list[Path], relative: str) -> Path | None:
    """First existing <root>/<relative> across the ancestor chain."""
    for r in roots:
        cand = r / relative
        if cand.is_file():
            return cand
    return None


def find_aux(spec: str | None, tag: str, filename: str,
             search: list[Path | None]) -> Path | None:
    """Locate a per-run auxiliary file. `spec` may be a path, a template containing
    {tag}, or a directory (searched as <dir>/<tag>/<file> then <dir>/<file>).
    Falls back to the run dirs and sweep roots in `search`."""
    if spec:
        p = Path(spec.replace("{tag}", tag))
        if p.is_file():
            return p
        if p.is_dir():
            for cand in (p / tag / filename, p / filename):
                if cand.is_file():
                    return cand
    for base in search:
        if base is None:
            continue
        for cand in (base / filename, base / tag / filename, base.parent / filename):
            if cand.is_file():
                return cand
    return None


def order_columns(df: pd.DataFrame, front: list[str]) -> pd.DataFrame:
    """Identifiers and headline metrics first, everything else after, without
    dropping anything the analyzer bothered to compute."""
    return df[[c for c in front if c in df.columns]
              + [c for c in df.columns if c not in front]]


def write_table(df: pd.DataFrame, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    df.to_csv(path, index=False)
    return path