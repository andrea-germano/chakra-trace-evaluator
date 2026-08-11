#!/usr/bin/env python3
"""
utils.plots — the plotting mechanics both sweeps repeat, and nothing about
what is being plotted.

`plot_series` is the one that matters: the sweeps drew the same "one line per
variant, x = the swept axis, drop NaNs" loop, differing only in the name of the
x column. Here that name is an argument. `lines_by_group` is its cross-model
sibling (one line per workload); `zoom_y`, `mark_lossy`/`loss_proxies`, the MS
scale and the shared palette live here for the same reason -- every analyzer
was importing them from another analyzer.

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
from matplotlib.lines import Line2D

MS = 1e-6                      # ns -> ms, the scale every figure converts with

# The shared palette of the sweep figures, so one quantity keeps one colour
# across analyzers. LOSS_RED is reserved for "this run dropped packets" overlays
# and never used for a data series.
BLUE, CORAL, GREEN, VIOLET, MUTED = \
    "#1f77b4", "#d1495b", "#2b8a3e", "#6a4c93", "#9aa0a6"
LOSS_RED = "#e8000b"


def downsample_max(ts, ys, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Max-per-bucket downsample of a queue time series to at most `n` points.

    A qlen.txt series is millions of 100 ns samples -- more points than the
    figure has pixels. The bucket keeps the MAX, not the mean, because the peak
    is the quantity being compared to KMAX/the buffer, and averaging would erase
    exactly the excursion that matters. Series already under `n` points, or with
    no time span, are returned unchanged."""
    ts = np.asarray(ts, float)
    ys = np.asarray(ys, float)
    if len(ts) <= n or ts[-1] <= ts[0]:
        return ts, ys
    edges = np.linspace(ts[0], ts[-1], n + 1)
    idx = np.clip(np.searchsorted(edges, ts) - 1, 0, n - 1)
    hi = np.full(n, -1.0)
    np.maximum.at(hi, idx, ys)
    keep = hi >= 0
    centres = (edges[:-1] + edges[1:]) / 2
    return centres[keep], hi[keep]


def buf_colour(b: float, bufs) -> object:
    """One colour per swept value, ordered light->dark along the SWEPT axis.

    Spaced in log2, like the axis itself, so 2 and 4 MiB are as far apart on the
    colour ramp as 32 and 64 are. The ramp stops short of viridis' bright yellow
    end, which is unreadable as a thin line on white."""
    uniq = sorted(set(bufs))
    if len(uniq) < 2:
        return BLUE
    lo, hi = np.log2(min(uniq)), np.log2(max(uniq))
    f = (np.log2(b) - lo) / (hi - lo)
    return plt.get_cmap("viridis")(0.05 + 0.78 * f)


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


# The three knees of utils.measures.knee_scalars, as vertical rules. Drawn on
# every figure whose curve they explain, with one style each, so the same rule
# means the same thing in every analyzer. The STYLE is here (drawing mechanics);
# the knees themselves are a measurement (utils.measures).
KNEE_STYLE = {
    "knee_pfc_mb": (CORAL, "PFC knee"),
    "knee_stall_mb": (VIOLET, "stall onset"),
    "knee_saturation_mb": (MUTED, "saturation"),
}


def mark_knees(ax, s: pd.DataFrame, which=tuple(KNEE_STYLE),
               label: bool = True) -> None:
    """Vertical rules at the knees carried in `s` (constant columns). Silent for
    a knee the sweep never reached (NaN) -- an absent rule means 'did not
    happen', which is itself the reading. `s` must already be one sweep's rows
    (one buffer axis): a multi-topology frame carries several knee values per
    column and only the first would be drawn."""
    for col in which:
        if col not in s.columns or not s[col].notna().any():
            continue
        v = float(s[col].dropna().iloc[0])
        colour, name = KNEE_STYLE[col]
        ax.axvline(v, color=colour, linestyle="--", lw=1.1, alpha=0.55,
                   label=(name if label else None), zorder=0)


def zoom_y(ax, series, pad: float = 0.15) -> None:
    """Autoscale one panel's y-axis to its own data, with a small margin --
    for quantities whose variation would be invisible against a zero baseline
    (a makespan that moves by a percent)."""
    v = series.dropna()
    if v.empty:
        return
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        band = max(abs(hi) * 0.02, 0.5)
        ax.set_ylim(hi - band, hi + band)
    else:
        ax.set_ylim(lo - pad * span, hi + pad * span)


def mark_lossy(ax, g: pd.DataFrame, xcol: str, ycol: str) -> tuple[bool, bool]:
    """Overlay, on top of a plotted series, the runs that were NOT lossless: a red
    ring on runs that dropped packets, a grey ring on runs whose loss is UNKNOWN
    (no drops.txt captured). Returns (marked_lossy, marked_unknown) so the caller
    can add matching legend proxies (loss_proxies)."""
    ml = mu = False
    if "lossy" in g.columns:
        gl = g[g["lossy"] == True].dropna(subset=[xcol, ycol])       # noqa: E712
        if not gl.empty:
            ax.scatter(gl[xcol], gl[ycol], s=170, facecolors="none",
                       edgecolors=LOSS_RED, linewidths=2.6, zorder=6)
            ml = True
    if "loss_captured" in g.columns:
        gu = g[g["loss_captured"] == False].dropna(subset=[xcol, ycol])  # noqa: E712
        if not gu.empty:
            ax.scatter(gu[xcol], gu[ycol], s=140, facecolors="none",
                       edgecolors=MUTED, linewidths=1.8, zorder=5)
            mu = True
    return ml, mu


def loss_proxies(marked_lossy: bool, marked_unknown: bool) -> list[Line2D]:
    """Legend proxies matching mark_lossy's rings."""
    h: list[Line2D] = []
    if marked_lossy:
        h.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=LOSS_RED,
                        mew=2.6, ms=12, label="dropped packets (not lossless)"))
    if marked_unknown:
        h.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=MUTED,
                        mew=1.8, ms=11, label="loss unknown (no drops.txt)"))
    return h


def lines_by_group(df: pd.DataFrame, group_col: str, xcol: str, ycol: str,
                   xlabel: str, ylabel: str, title: str, fname: str,
                   outdir: Path, written: list[Path],
                   hline: float | None = None, logx2: bool = False) -> None:
    """One line per value of `group_col`, x = `xcol` -- the figure shape both
    cross-model companions repeat. Skipped when the column is absent or entirely
    NaN, so a metric a group's runs never produced just drops out instead of an
    empty line. `logx2` draws the log-2 swept-value axis (logx_pow2); otherwise
    the x axis is linear with `xlabel`."""
    if ycol not in df.columns or not df[ycol].notna().any():
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g, grp in df.groupby(group_col):
        grp = grp.dropna(subset=[ycol]).sort_values(xcol)
        if grp.empty:
            continue
        ax.plot(grp[xcol], grp[ycol], marker="o", label=g)
    if hline is not None:
        ax.axhline(hline, color="k", linestyle=":", alpha=0.4)
    if logx2:
        logx_pow2(ax, df, xcol, xlabel)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both" if logx2 else "major")
    ax.legend(fontsize=8)
    save_fig(fig, outdir, fname, written)