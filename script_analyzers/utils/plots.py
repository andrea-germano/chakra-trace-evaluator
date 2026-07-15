#!/usr/bin/env python3
"""
utils.plots — the plotting mechanics both sweeps repeat, and nothing about
what is being plotted.

`plot_series` is the one that matters: the sweeps drew the same "one line per
variant, x = the swept axis, drop NaNs" loop, differing only in the name of the
x column. Here that name is an argument.

`relative_range` used to live here. It normalised two series onto 0-1 so they
could share an axis, which made a 7.4% effect look the same size as a 26% one.
The plot that used it is gone and so is it: if two series need one axis, they
need two axes.

Decoration (titles, reference lines, twin axes, regime bands) stays in the
analyzers: it is where the two sweeps actually differ, and merging it would be
merging the questions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt


def plot_series(ax, data: pd.DataFrame, xcol: str, ycol: str, label: str,
                marker: str = "o", scale: float = 1.0, linestyle: str = "-",
                color: str | None = None) -> bool:
    """One line per variant, NaNs dropped. Returns False when the column is absent
    or entirely NaN, so callers can skip a panel instead of emitting an empty one."""
    if ycol not in data.columns or not data[ycol].notna().any():
        return False
    drawn = False
    for variant, grp in data.groupby("variant"):
        grp = grp.dropna(subset=[ycol]).sort_values(xcol)
        if grp.empty:
            continue
        lbl = label if data["variant"].nunique() == 1 else f"{label} [{variant}]"
        ax.plot(grp[xcol], grp[ycol] * scale, marker=marker, linestyle=linestyle,
                label=lbl, color=color)
        drawn = True
    return drawn


def logx_pow2(ax, data: pd.DataFrame, xcol: str, xlabel: str) -> None:
    """Log-2 x axis ticked at exactly the swept values. Buffers are 2/4/8/16/32, so
    a linear axis crushes the small end -- which is the end where the regime lives."""
    ax.set_xscale("log", base=2)
    xs = sorted(data[xcol].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel(xlabel)


def save_fig(fig, outdir: Path, name: str, written: list[Path] | None = None) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if written is not None:
        written.append(path)
    return path