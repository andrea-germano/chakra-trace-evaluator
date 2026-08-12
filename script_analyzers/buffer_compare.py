#!/usr/bin/env python3
"""
buffer_compare — same buffer sweep, different MODELS: buffer_sweep's own figure
set, with one line (or one waterfall block) per model.

Auto-discovers every workload directory under output/ns3 that ran the given
sweep (buffer_sweep_T1 by default), scores each with buffer_sweep.analyse_sweep
-- one definition of the metrics, two tools -- and re-draws THE SAME FIGURES
buffer_sweep draws, same numbers, same quantities, with the model as the line
family. Reading a single-model analysis and this one is the same experience;
only the number of lines changes.

The figure numbering mirrors buffer_sweep 1:1:

    01  CAUSAL CHAIN TO TTFT      same stacked panels (PP skew -> gated
                                  all-reduce bw -> steady bw -> TTFT), one line
                                  per group. TTFT is x this group's
                                  largest-buffer run: raw ns would compare
                                  model size, not fabric effect.
    03  KV TP-SHARD SKEW          same quantity as the boxplots (per-layer
                                  |shard 1 - shard 0| within one PP stage),
                                  reduced to the mean | p99 panels -- the layer
                                  POPULATIONS are per-run material the
                                  cross-group summary does not carry.
    04  FIRST TOKEN TO SECOND     the same waterfall, one block per group:
                                  handoff in flight | decode awake, KV still
                                  arriving | first pass completing, PAUSE count
                                  beside each bar.
    05  DECODE KV STALL           same two panels; the first pass is drawn in
                                  units of its own steady ITL (the dimensionless
                                  twin), and the right panel overlays stall
                                  (solid) with its cause, the KV tail (dashed),
                                  per group -- the pairs hugging each other is
                                  the same finding as in buffer_sweep.
    06  DECODE ALL-REDUCE         same panel: first (KV-gated) effective bw
                                  (solid) vs steady mean (dashed), worst stage.
    09  BOTTLENECK CONGESTION     buffer_sweep 09 restricted to the measured
                                  bottleneck: delivered %, PAUSE frames
                                  (symlog), peak queue. The full per-link view
                                  cannot cross groups -- link labels are
                                  topology-local.
    10  BUFFER BLOAT              same two panels: peak & mean occupancy (log)
                                  against the buffer itself | mean / peak.
    11  KNEES                     buffer_sweep's three knee rules, cross-group
                                  form: one row per group, a dot per knee; an
                                  open marker at the right edge = never reached.

    02 / 07 / 08 are time-domain figures (cumulative KV arrival, occupancy(t),
    per-switch queues(t)); they cannot be rebuilt from sweep scalars and are
    NOT duplicated here -- they live in each group's own buffer_sweep output
    (results/sweep_analysis/buffer/<workload>/<sweep>/, or by_oversub/os<N>/
    for the oversubscription plane).

One figure set, two grouping dimensions: story_plots is defined once here and
drawn by two callers -- this script, grouping by MODEL, and oversub2d_sweep,
grouping by OVERSUBSCRIPTION LEVEL. What is worth comparing across a buffer
sweep does not depend on which knob the second dimension is.

Raw vs normalised (why TTFT and the first pass are ratios): fabric-domain
magnitudes (a skew, a queue depth, a PAUSE count, an effective bandwidth) are
plotted RAW -- they do not carry the model's compute scale. Compute-scaled
durations (TTFT, the first decode pass) are normalised WITHIN each group:
ttft_slowdown to the group's largest-buffer run, tok2_over_itl to the group's
steady inter-token gap. Flat at 1 = the fabric effect does not reach the user.

PP=1 models are first-class citizens: they have no PP wave, so pp_skew is NaN
and their line drops from that panel only. Bottleneck consistency is checked
WITHIN each model's sweep, never ACROSS models (switch numbering is
topology-local); pass --bottleneck to buffer_sweep.py directly if one model's
auto-detected bottleneck needs overriding.

Usage
-----
    python3 buffer_compare.py
    python3 buffer_compare.py --sweep buffer_sweep_T2 --workloads 'llama2_13b_*'
    python3 buffer_compare.py --list
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import paths, roles
from utils.cli import Abort, need
from utils.paths import fresh_dir
from utils.plots import (BLUE, CORAL, GREEN, MS, MUTED, VIOLET,
                         logx_pow2, save_fig)
from utils.roles import Placement
from buffer_sweep import analyse_sweep

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Shared machinery (used by this script and by oversub2d_sweep)
# --------------------------------------------------------------------------- #
def group_colours(groups) -> dict:
    """One colour per group, stable across every figure of a run.

    Numeric groups (oversubscription ratios) are an ORDERED axis: they get the
    viridis ramp, dark->light with the value, like buffer_sweep's per-buffer
    colouring. Categorical groups (workload names) have no order to encode:
    they get the tab10 cycle."""
    gs = list(groups)
    if all(isinstance(g, (int, float, np.integer, np.floating)) for g in gs):
        order = sorted(gs)
        n = max(len(order) - 1, 1)
        return {g: plt.get_cmap("viridis")(0.05 + 0.78 * i / n)
                for i, g in enumerate(order)}
    cmap = plt.get_cmap("tab10")
    return {g: cmap(i % 10) for i, g in enumerate(gs)}


def short_labels(names: list[str]) -> dict[str, str]:
    """Legend labels with the longest common '_'-prefix and '_'-suffix removed:
    with one sweep's workloads sharing 'llama2_13b_p-tp2pp2_..._stream_', only
    the part that actually differs ('64reqs_512prompt') earns legend space."""
    if len(names) < 2:
        return {n: n for n in names}
    parts = [n.split("_") for n in names]
    limit = min(len(p) for p in parts) - 1          # keep at least one token
    pre = 0
    while pre < limit and len({p[pre] for p in parts}) == 1:
        pre += 1
    suf = 0
    while suf < limit - pre and len({p[-1 - suf] for p in parts}) == 1:
        suf += 1
    return {n: ("_".join(p[pre:len(p) - suf]) or n)
            for n, p in zip(names, parts)}


def add_group_norms(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """The two dimensionless 'does it reach the user?' columns, normalised
    WITHIN each group (each model / each os level is its own reference):

        tok2_over_itl   first decode pass over this group's steady inter-token
                        gap -- decode_kv_stall made dimensionless.
        ttft_slowdown   TTFT over this group's largest-buffer TTFT.
    """
    df = df.copy()
    df["tok2_over_itl"] = df["tok2_latency_ns"] / df["itl_steady_ns"]
    df["ttft_slowdown"] = NAN
    for _, idx in df.groupby(group_col).groups.items():
        sub = df.loc[idx].dropna(subset=["ttft_ns"]).sort_values("buffer_mb")
        ref = float(sub["ttft_ns"].iloc[-1]) if len(sub) else NAN
        if pd.notna(ref) and ref > 0:
            df.loc[idx, "ttft_slowdown"] = df.loc[idx, "ttft_ns"] / ref
    return df


# --------------------------------------------------------------------------- #
# The shared figure set: buffer_sweep's figures, one line family per group
# --------------------------------------------------------------------------- #
def _ordered(df: pd.DataFrame, group_col: str) -> list:
    return sorted(df[group_col].unique(), key=lambda g: (str(type(g)), g))


def _lines(ax, df, group_col, groups, colours, label, col, scale=1.0,
           ls="-", marker="o", labelled=True, alpha=1.0) -> bool:
    """One line per group for `col`; returns False if nothing was drawn."""
    if col not in df.columns or not df[col].notna().any():
        return False
    drawn = False
    for g in groups:
        sub = df[df[group_col] == g].dropna(subset=[col]).sort_values("buffer_mb")
        if sub.empty:
            continue
        ax.plot(sub["buffer_mb"], sub[col] * scale, marker=marker, ms=4.5,
                ls=ls, color=colours[g], alpha=alpha,
                label=(label(g) if labelled else None))
        drawn = True
    return drawn


def _fig01_causal_chain(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 01: the same stacked panels, one line per group. The TTFT
    panel is x the group's largest-buffer run (see module docstring)."""
    panels = [
        ("PP arrival skew (ms)", [("pp_skew_ns", MS, "-", True)]),
        ("Gated all-reduce\neff. bw (GB/s, n=1)", [("rs_ar_first_bw", 1.0, "-", False)]),
        # solid = receiving stage steady, dashed = first prefill stage (ungated
        # reference) -- same colour per group, a style legend disambiguates.
        ("Steady all-reduce\neff. bw (GB/s)", [("rs_ar_rest_bw", 1.0, "-", False),
                                               ("rs_ar_first_stage_bw", 1.0, "--", False)]),
        ("TTFT\n(× largest buffer)", [("ttft_slowdown", 1.0, "-", False)]),
    ]
    panels = [(yl, sp) for yl, sp in panels
              if any(c in df.columns and df[c].notna().any() for c, *_ in sp)]
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8.5, 2.0 * n + 1.0))
    axes = np.atleast_1d(axes)
    for ax, (ylabel, specs) in zip(axes, panels):
        for col, scale, ls, labelled in specs:
            _lines(ax, df, group_col, groups, colours, label, col, scale,
                   ls=ls, labelled=labelled, alpha=0.55 if ls == "--" else 1.0)
        if col == "ttft_slowdown":
            ax.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.5)
        logx_pow2(ax, df, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    if any(yl.startswith("Steady") for yl, _ in panels):
        i = next(i for i, (yl, _) in enumerate(panels) if yl.startswith("Steady"))
        axes[i].legend(handles=[
            Line2D([], [], color="k", ls="-", label="receiving stage (steady)"),
            Line2D([], [], color="k", ls="--", alpha=0.55, label="first prefill stage")],
            fontsize=7, loc="best")
    fig.suptitle("PP arrival skew, the gated all-reduce and TTFT vs buffer",
                 y=0.99)
    save_fig(fig, outdir, "01_causal_chain_to_ttft.png", written)


def _fig03_shard_skew(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 03's quantity (per-layer |shard 1 - shard 0| within one PP
    stage) reduced to mean | p99 -- the layer populations are per-run material
    the cross-group summary does not carry."""
    cols = [("kv_tp_skew_mean_ns", "Per-layer shard skew, mean (ms)"),
            ("kv_tp_skew_p99_ns", "Per-layer shard skew, p99 (ms)")]
    cols = [(c, yl) for c, yl in cols if c in df.columns and df[c].notna().any()]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5.8 * len(cols), 4.8),
                             squeeze=False)
    for ax, (col, ylabel) in zip(axes[0], cols):
        _lines(ax, df, group_col, groups, colours, label, col, MS)
        logx_pow2(ax, df, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("KV skew between the TP shards of a PP stage", y=1.02)
    save_fig(fig, outdir, "03_kv_tp_shard_skew.png", written)


def _fig04_waterfall(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 04, one block per group: the TTFT -> token-2 interval as
    three consecutive segments per buffer, PAUSE count beside each bar. Each
    block keeps its own ms axis -- the absolute scale is the group's own."""
    needed = ["ttft_ns", "dec_start_ns", "kv_gate_ns", "tok2_ns"]
    if not all(c in df.columns for c in needed):
        return
    blocks = []
    for g in groups:
        sub = df[df[group_col] == g].dropna(subset=needed).sort_values("buffer_mb")
        if len(sub):
            blocks.append((g, sub))
    if not blocks:
        return
    total_bars = sum(len(sub) for _, sub in blocks)
    fig, axes = plt.subplots(len(blocks), 1,
                             figsize=(12.5, 0.42 * total_bars + 1.1 * len(blocks) + 1.2),
                             squeeze=False)
    for ax, (g, sub) in zip(axes[:, 0], blocks):
        end = 0.0
        for i, r in enumerate(sub.itertuples()):
            t0 = r.ttft_ns
            ds = (r.dec_start_ns - t0) * MS
            kg = (r.kv_gate_ns - t0) * MS
            t2 = (r.tok2_ns - t0) * MS
            end = max(end, t2)
            ax.barh(i, ds, height=0.62, color=BLUE)
            ax.barh(i, max(kg - ds, 0.0), left=ds, height=0.62, color=CORAL)
            ax.barh(i, max(t2 - kg, 0.0), left=max(kg, ds), height=0.62, color=GREEN)
            pf = getattr(r, "link0_pause_frames", NAN)
            if pd.notna(pf):
                ax.text(t2 * 1.012, i, f"{pf:,.0f} PAUSE", va="center",
                        fontsize=8, color=MUTED)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"{b:g} MiB" for b in sub["buffer_mb"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, end * 1.22)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(label(g), loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("ms after the first token")
    axes[0, 0].legend(handles=[Patch(color=BLUE, label="handoff in flight"),
                               Patch(color=CORAL, label="decode awake, KV still arriving"),
                               Patch(color=GREEN, label="first pass completing")],
                      fontsize=9, ncol=3, loc="lower center",
                      bbox_to_anchor=(0.5, 1.25), frameon=False)
    save_fig(fig, outdir, "04_first_token_to_second.png", written)


def _fig05_decode_stall(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 05: LEFT the first pass in units of its own steady ITL (the
    dimensionless twin of the raw-ms panel -- raw ns would compare model size);
    RIGHT stall (solid) overlaid with its cause, the KV tail (dashed), per
    group. The pairs hugging each other is buffer_sweep's 'the stall IS the
    tail' finding, now visible per group."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    okL = _lines(axL, df, group_col, groups, colours, label, "tok2_over_itl")
    axL.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.5)
    axL.set_ylabel("First decode pass (× steady ITL)")
    axL.set_title("First decode pass vs steady state", fontsize=11)

    okR = _lines(axR, df, group_col, groups, colours, label,
                 "decode_kv_stall_ns", MS, ls="-", labelled=False)
    _lines(axR, df, group_col, groups, colours, label,
           "kv_tail_after_dec_start_ns", MS, ls="--", marker="s",
           labelled=False, alpha=0.6)
    axR.axhline(0.0, color="k", ls=":", lw=1.0, alpha=0.5)
    axR.set_ylabel("ms")
    axR.set_title("First-pass stall (solid) and the KV tail (dashed)",
                  fontsize=11)
    axR.legend(handles=[Line2D([], [], color="k", ls="-", label="first-pass stall"),
                        Line2D([], [], color="k", ls="--", alpha=0.6,
                               label="KV tail past decode start")], fontsize=8)
    if not (okL or okR):
        plt.close(fig)
        return
    for a in (axL, axR):
        logx_pow2(a, df, "buffer_mb", "Per-switch buffer (MiB)")
        a.grid(True, alpha=0.3, which="both")
    axL.legend(fontsize=8)
    fig.suptitle("First decode pass, steady inter-token gap and KV stall",
                 y=1.02)
    save_fig(fig, outdir, "05_decode_kv_stall.png", written)


def _fig06_decode_ar(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 06: first (KV-gated) effective bw solid vs steady mean
    dashed, per group -- decode_worst_stage's reduction (worst stage)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ok = _lines(ax, df, group_col, groups, colours, label, "dec_ar_first_bw")
    _lines(ax, df, group_col, groups, colours, label, "dec_ar_rest_bw",
           ls="--", marker="v", labelled=False, alpha=0.55)
    if not ok:
        plt.close(fig)
        return
    logx_pow2(ax, df, "buffer_mb", "Per-switch buffer (MiB)")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title("Decode first TP all-reduce (solid) vs steady state (dashed), "
                 "worst stage", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save_fig(fig, outdir, "06_decode_allreduce.png", written)


def _fig09_congestion(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 09 restricted to the measured bottleneck (link0): delivered
    efficiency, PAUSE frames (symlog), peak queue. The full per-link view stays
    per group -- link labels are topology-local and cannot cross groups."""
    panels = [("link0_eff_pct", 1.0, "KV bandwidth (% of nominal)", None),
              ("link0_pause_frames", 1.0, "PAUSE frames (symlog)", "symlog"),
              ("link0_qpeak_bytes", 1 / 2**20, "Peak occupancy (MiB)", None)]
    panels = [p for p in panels if p[0] in df.columns and df[p[0]].notna().any()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.8),
                             squeeze=False)
    for ax, (col, scale, ylabel, yscale) in zip(axes[0], panels):
        _lines(ax, df, group_col, groups, colours, label, col, scale)
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=1)
        logx_pow2(ax, df, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Congestion at the measured bottleneck", y=1.02)
    save_fig(fig, outdir, "09_bottleneck_congestion.png", written)


def _fig10_bloat(df, group_col, groups, colours, label, outdir, written):
    """buffer_sweep 10: peak (solid) & mean (dashed) occupancy on a log axis
    against the buffer itself, and the mean/peak ratio."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    ok = _lines(axL, df, group_col, groups, colours, label,
                "link0_qpeak_bytes", 1 / 2**20)
    _lines(axL, df, group_col, groups, colours, label,
           "link0_qmean_bytes", 1 / 2**20, ls="--", marker="v",
           labelled=False, alpha=0.55)
    if not ok:
        plt.close(fig)
        return
    bufs = sorted(df["buffer_mb"].unique())
    axL.plot(bufs, bufs, ":", color=MUTED, lw=1.0, label="the buffer itself")
    axL.set_yscale("log")
    axL.set_ylabel("Occupancy at the bottleneck (MiB, log)")
    axL.set_title("Peak (solid) and mean (dashed) occupancy", fontsize=11)
    axL.legend(fontsize=8)

    _lines(axR, df, group_col, groups, colours, label, "q_bloat_ratio",
           labelled=False)
    axR.set_ylabel("mean ÷ peak occupancy")
    axR.set_title("Mean ÷ peak occupancy", fontsize=11)
    for a in (axL, axR):
        logx_pow2(a, df, "buffer_mb", "Per-switch buffer (MiB)")
        a.grid(True, alpha=0.3, which="both")
    fig.suptitle("Peak and mean queue occupancy at the bottleneck", y=1.02)
    save_fig(fig, outdir, "10_buffer_bloat.png", written)


# buffer_sweep's knee rules, cross-group form. Same colours as its KNEE_STYLE.
KNEE_COLS = [
    ("knee_pfc_mb", CORAL, "o", "PFC knee (backpressure gone)"),
    ("knee_stall_mb", VIOLET, "s", "stall onset (first pass > steady)"),
    ("knee_saturation_mb", MUTED, "D", "saturation (nothing changes)"),
]


def _fig11_knees(df, group_col, groups, colours, label, outdir, written):
    """The three regime changes buffer_sweep draws as vertical rules, one row
    per group, one dot per knee; an OPEN marker parked at the right edge is a
    knee the sweep never reached -- absence drawn, not omitted."""
    if not any(c in df.columns and df[c].notna().any() for c, *_ in KNEE_COLS):
        return
    bufs = sorted(df["buffer_mb"].unique())
    edge = bufs[-1] * 2                              # the "never reached" slot
    fig, ax = plt.subplots(figsize=(9, 0.85 * len(groups) + 2.2))
    for yi, g in enumerate(groups):
        row = df[df[group_col] == g].iloc[0]
        ax.axhline(yi, color=colours[g], lw=0.8, alpha=0.25, zorder=0)
        for k, (col, c, m, _lab) in enumerate(KNEE_COLS):
            v = row.get(col, NAN)
            dy = (k - 1) * 0.18                      # coinciding knees stay visible
            if pd.notna(v):
                ax.plot(v, yi + dy, m, color=c, ms=10, zorder=3)
            else:
                ax.plot(edge, yi + dy, m, mfc="none", mec=c, mew=1.8, ms=10,
                        zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bufs + [edge])
    ax.set_xticklabels([f"{b:g}" for b in bufs] + ["never"])
    ax.set_xlabel("Per-switch buffer (MiB)")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([label(g) for g in groups])
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3, which="major")
    ax.set_title("Knee buffer size per workload (open = never reached)",
                 fontsize=11)
    ax.legend(handles=[Line2D([], [], marker=m, color=c, ls="none", ms=9,
                              label=lab) for _col, c, m, lab in KNEE_COLS],
              fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_fig(fig, outdir, "11_knees.png", written)


def story_plots(df: pd.DataFrame, group_col: str, outdir: Path,
                label: Callable = str) -> list[Path]:
    """buffer_sweep's figure set with one line family per value of `group_col`.
    Figure numbers match buffer_sweep's; 02/07/08 (time-domain) are per-group
    material and deliberately absent -- see the module docstring."""
    written: list[Path] = []
    groups = _ordered(df, group_col)
    colours = group_colours(groups)
    for fn in (_fig01_causal_chain, _fig03_shard_skew, _fig04_waterfall,
               _fig05_decode_stall, _fig06_decode_ar, _fig09_congestion,
               _fig10_bloat, _fig11_knees):
        fn(df, group_col, groups, colours, label, outdir, written)
    return written


# --------------------------------------------------------------------------- #
# Cross-model scoring
# --------------------------------------------------------------------------- #
def load_workload(root: Path, workload: str, sweep: str,
                  placement: Placement, top_links: int) -> pd.DataFrame:
    """One row per (workload, buffer) run -- mirrors buffer_sweep.main's call to
    analyse_sweep so a model is scored identically whether analysed alone or
    here, then adds only unit conversions for summary.csv readability. The
    normalised columns (ttft_slowdown, tok2_over_itl) are added over the
    COMBINED frame by add_group_norms, one definition for both compare tools."""
    p = paths.SweepPaths(sweep=sweep, workload=workload, root=root)
    need(not p.missing_roots(),
         f"{workload}: derived root(s) do not exist:\n    "
         + "\n    ".join(p.missing_roots()))
    # want_series=False: this cross-model compare plots only scalars, so the
    # per-tag queue timelines are never built -- the big saving on qlen.txt reads.
    _, s, _ = analyse_sweep(p, placement, top_links=top_links,
                            bn_force=None, verbose=False, want_series=False)
    s = s.copy()
    s["pp_skew_ms"] = s["pp_skew_ns"] / 1e6
    s["kv_tp_skew_mean_ms"] = s["kv_tp_skew_mean_ns"] / 1e6
    s["kv_tp_skew_p99_ms"] = s["kv_tp_skew_p99_ns"] / 1e6
    s["decode_kv_stall_ms"] = s["decode_kv_stall_ns"] / 1e6
    s["dec_kv_lateness_ms"] = s["dec_kv_lateness_ns"] / 1e6
    s["kv_tail_ms"] = s["kv_tail_after_dec_start_ns"] / 1e6
    s["pause_frames"] = s.get("link0_pause_frames")
    win = s.get("link0_window_ns")
    s["pause_rate"] = (s["pause_frames"] / (win / 1e6)      # frames per ms of
                       if win is not None else NAN)         # the KV window
    s["line_rate_pct"] = s.get("link0_eff_pct")
    qb, qm = s.get("link0_qpeak_bytes"), s.get("link0_qmean_bytes")
    s["qpeak_mb"] = qb / 2**20 if qb is not None else NAN
    s["qmean_mb"] = qm / 2**20 if qm is not None else NAN
    s.insert(0, "workload", workload)
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_compare_arguments(ap, "buffer_compare",
                                default_sweep="buffer_sweep_T1")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links analyse_sweep scores "
                         "(only link0 is compared here; default: 6)")
    roles.add_argument(ap)
    a = ap.parse_args(argv)

    root = Path(a.root)
    workloads = paths.discover_workloads(root, a.sweep, "ns3")
    if a.workloads:
        workloads = [w for w in workloads
                     if any(fnmatch.fnmatch(w, pat) for pat in a.workloads)]
    if a.exclude:
        workloads = [w for w in workloads
                     if not any(fnmatch.fnmatch(w, pat) for pat in a.exclude)]

    print(f"sweep    {a.sweep}")
    print(f"root     {root}")
    print(f"found    {len(workloads)} workload(s):")
    for w in workloads:
        print(f"  - {w}")
    if a.list:
        return 0

    try:
        need(workloads,
             f"no workload under {root / 'output' / 'ns3'} has a "
             f"{a.sweep!r} sub-directory (or --workloads/--exclude filtered "
             f"all of them out)")

        placement = Placement.parse(a.placement)
        outdir = (Path(a.out) if a.out else
                  root / "results" / "sweep_analysis" / "buffer_compare" / a.sweep)

        frames = []
        print(f"\nScanning {len(workloads)} workload(s):")
        for w in workloads:
            summ = load_workload(root, w, a.sweep, placement, a.top_links)
            frames.append(summ)
            sk = summ.sort_values("buffer_mb")["pp_skew_ms"].iloc[0]
            print(f"  + {w:<55} buf={summ['buffer_mb'].min():g}.."
                  f"{summ['buffer_mb'].max():g}  "
                  f"bn={summ['bottleneck'].iloc[0]}  "
                  f"skew@min_buf="
                  f"{f'{sk:.2f}ms' if pd.notna(sk) else 'n/a (PP=1)'}")

        combined = add_group_norms(pd.concat(frames, ignore_index=True),
                                   "workload")
        front = ["workload", "tag", "bottleneck", "buffer_mb",
                 "knee_pfc_mb", "knee_stall_mb", "knee_saturation_mb",
                 "pp_skew_ms", "kv_tp_skew_mean_ms", "kv_tp_skew_p99_ms",
                 "decode_kv_stall_ms", "kv_tail_ms", "dec_kv_lateness_ms",
                 "qpeak_mb", "qmean_mb", "q_bloat_ratio",
                 "pause_frames", "pause_rate", "line_rate_pct",
                 "rs_ar_first_bw", "rs_ar_rest_bw", "dec_ar_first_bw",
                 "ttft_slowdown", "tok2_over_itl", "kv_shard_bias_ns"]
        combined = combined[[c for c in front if c in combined.columns]
                            + [c for c in combined.columns if c not in front]]
        fresh_dir(outdir)                 # stale figures from a previous figure
        combined.to_csv(outdir / "summary.csv", index=False)   # set never linger

        labels = short_labels(list(combined["workload"].unique()))
        written = story_plots(combined, "workload", outdir,
                              label=lambda g: labels.get(g, str(g)))

        print(f"\nWrote {outdir}:")
        print("  summary.csv")
        for pth in written:
            print(f"  {pth.name}")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
