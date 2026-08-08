#!/usr/bin/env python3
"""
oversub2d_sweep — the oversubscription x buffer plane on T1.

The 1D question ("what does oversubscription cost") is confounded by the buffer:
at a large buffer DCQCN absorbs the rate mismatch and every congestion signal
reads zero (the T1 buffer sweep showed buf32 already non-binding at the native
10.24:1). This sweep instead asks the 2D question:

    how much buffer is needed to HIDE a given oversubscription ratio,
    and what does the fabric cost once the buffer can no longer hide it?

Grid: oversubscription os in {1,2,4,8} (uplink fixed 200 Gbps, PCIe downlink
scaled: 2*downlink/200) x buffer in {2,4,8,16} MiB. buf32/64 are dropped -- they
sit in the infinite-memory regime even at 10.24:1.

What it produces
----------------
1. by_oversub/os<N>/ -- the COMPLETE buffer_sweep figure set (01..11: causal
   chain to TTFT, KV cumulative arrival, per-link bandwidth/concurrency, buffer
   occupancy(t) with PFC pauses, per-switch queue timelines, occupancy-vs-buffer,
   PFC count, KV TP-group skew, KV time per rank, decode first all-reduce, decode
   KV stall), one full set per oversubscription level. Same layout you already
   read on the buffer sweep, so every per-run/time-series figure stays legible
   (4 buffers per panel, not 16).
2. top level -- the 2D synthesis across all OS levels, one line per level,
   x = buffer, in two families:

   * the buffer-sweep STORY, re-drawn in buffer_sweep's own figure layouts so
     the two analyses read side by side -- the causal chain to TTFT (01: PP
     skew -> gated all-reduce bw -> steady bw -> TTFT, stacked panels), the KV
     TP-group skew (08), the decode first all-reduce (09) and the decode KV
     stall (10). These are the same questions the buffer sweep asks, now with
     "and how does the answer move with the oversubscription ratio" on top.
     Being tail-sensitive, several respond non-monotonically to the buffer: a
     LINE shows that honestly (every point is visible), which is why they get
     line families here and stay OUT of the heatmaps, where the interpolated
     gradient would read as a trend.
   * SEVERITY / OUTCOME / MECHANISM (02-07): congestion severity (paused
     fraction of the window, NOT the pause-frame count, which inverts as os
     rises), the accurate outcomes (delivered KV bandwidth, KV completion,
     TTFT), and the mechanism (concurrency, queue peak). The three that read
     as a clean field also get heatmaps (11-13) -- this is where the "knee
     moves right as os grows" story is read directly.

Reuse, not reinvention
----------------------
The per-run metrics are exactly what buffer_sweep measures from the ns-3 outputs.
Its engine, buffer_sweep.analyse_sweep, is called ONCE PER OVERSUBSCRIPTION LEVEL
(each call sees a single buffer axis, so its single-variant contract holds), and
its make_plots draws the per-OS sets. The bottleneck is forced to the
oversubscribed uplink (--bottleneck 8->12 by default), the link whose
2*downlink : 200 ratio IS the oversubscription.

Usage
-----
    python3 oversub2d_sweep.py --sweep oversub_sweep_T1
    python3 oversub2d_sweep.py --sweep oversub_sweep_T1 --workload <wl>
    python3 oversub2d_sweep.py --sweep oversub_sweep_T1 --no-per-os   # 2D synthesis only (fast)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import buffer_sweep
from buffer_sweep import analyse_sweep

from utils import paths, roles
from utils.cli import Abort, drain_warnings, need
from utils.paths import OVERSUB_AXIS, fresh_dir
from utils.plots import MS, logx_pow2, save_fig
from utils.roles import Placement

KIND = "oversub2d"


# --------------------------------------------------------------------------- #
# Line-family: one metric vs buffer, one line per oversubscription level.
# --------------------------------------------------------------------------- #
def _os_color(i: int, n: int):
    return plt.get_cmap("viridis")(i / max(n - 1, 1))


def line_family(s: pd.DataFrame, col: str, ylabel: str, title: str, name: str,
                outdir: Path, written: list, scale: float = 1.0,
                yscale: str | None = None) -> None:
    if col not in s.columns or not s[col].notna().any():
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = sorted(s["oversub"].unique())
    for i, os_ratio in enumerate(levels):
        sub = s[s["oversub"] == os_ratio].sort_values("buffer_mb")
        if not sub[col].notna().any():
            continue
        ax.plot(sub["buffer_mb"], sub[col] * scale, marker="o",
                color=_os_color(i, len(levels)), label=f"{os_ratio:g}:1")
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    if yscale:
        ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="oversubscription", fontsize=8)
    save_fig(fig, outdir, name, written)


def multi_panel(s: pd.DataFrame, panels: list, suptitle: str, name: str,
                outdir: Path, written: list) -> None:
    """Stacked panels sharing the buffer x-axis -- buffer_sweep's causal-chain
    layout -- with one line per oversubscription level in every panel. A panel
    whose column is absent or all-NaN is dropped; the whole figure is skipped
    when nothing survives. `panels` = [(col, ylabel, scale, hline), ...]."""
    panels = [p for p in panels if p[0] in s.columns and s[p[0]].notna().any()]
    if not panels:
        return
    levels = sorted(s["oversub"].unique())
    n = len(panels)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8.5, 2.2 * n + 1.0))
    axes = np.atleast_1d(axes)
    for ax, (col, ylabel, scale, hline) in zip(axes, panels):
        for i, os_ratio in enumerate(levels):
            sub = s[s["oversub"] == os_ratio].dropna(subset=[col]) \
                   .sort_values("buffer_mb")
            if sub.empty:
                continue
            ax.plot(sub["buffer_mb"], sub[col] * scale, marker="o", ms=4,
                    color=_os_color(i, len(levels)), label=f"{os_ratio:g}:1")
        if hline is not None:
            ax.axhline(hline, color="k", ls=":", lw=1.0, alpha=0.5)
        logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[0].legend(title="oversub", fontsize=7,
                   ncol=min(len(levels), 4), loc="best")
    fig.suptitle(suptitle, y=1.0)
    save_fig(fig, outdir, name, written)


# --------------------------------------------------------------------------- #
# Heatmap: oversubscription (rows) x buffer (cols) for one metric.
# --------------------------------------------------------------------------- #
def heatmap(s: pd.DataFrame, col: str, title: str, name: str, outdir: Path,
            written: list, fmt: str = "{:.0f}", cmap: str = "magma",
            scale: float = 1.0) -> None:
    if col not in s.columns or not s[col].notna().any():
        return
    piv = (s.pivot_table(index="oversub", columns="buffer_mb", values=col)
             .sort_index(ascending=False))          # high oversub at top
    piv = piv[sorted(piv.columns)] * scale
    data = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.5 * data.shape[1] + 2.6,
                                    1.0 * data.shape[0] + 2.2))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([f"{r:g}:1" for r in piv.index])
    ax.set_xlabel("Per-switch buffer (MiB)")
    ax.set_ylabel("Oversubscription")
    ax.set_title(title)
    lo, hi = np.nanmin(data), np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                norm = (v - lo) / (hi - lo + 1e-12)
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="black" if norm > 0.55 else "white", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    save_fig(fig, outdir, name, written)


# --------------------------------------------------------------------------- #
# Top-level 2D synthesis: the buffer-sweep STORY (01, 08-10) + SEVERITY /
# OUTCOME / MECHANISM (02-07) + heatmaps (11-13), across os levels.
# --------------------------------------------------------------------------- #
# The story figures re-draw buffer_sweep's own layouts (its figs 01/08/10/11)
# with the oversubscription level as the line family, so the buffer analysis
# and this one read side by side. Several of these are tail-sensitive (set by a
# single straggler) and respond non-monotonically to the buffer: a line shows
# that honestly -- every point is visible -- which is why they are drawn as
# line families here and deliberately kept OUT of the heatmaps, whose
# interpolated gradient would read deterministic sensitivity as a trend.
# The full time-series versions stay in by_oversub/os<N>/ (buffer_sweep figs).
#
# The severity metric is the paused FRACTION of the window, not the pause-frame
# count: the count INVERTS with oversubscription (as os rises the bottleneck
# stays paused in longer continuous stretches, so pause/resume events DROP,
# 132k -> 67k -> 51k at buf2, even though congestion worsens) while the paused
# fraction rises monotonically (51% -> 75% -> 85%). The outcome metrics
# (delivered bw, KV completion) come from fct bytes and are not affected by
# the pfc qIndex caveat.

# Buffer-story stacked figures: (filename, suptitle, [(col, ylabel, scale,
# hline), ...]). Worst-stage reductions (dec_*) come from decode_worst_stage,
# so they are stage-count independent; tok2_over_itl is derived in main().
STORY_SPECS = [
    ("01_causal_chain_to_ttft.png",
     "Does PP skew propagate into TTFT?  (one line per os level)", [
        ("pp_skew_ns", "PP arrival skew (µs)", 1e-3, None),
        ("rs_ar_first_bw", "Gated all-reduce\neff. bw (GB/s)", 1.0, None),
        ("rs_ar_rest_bw", "Steady all-reduce\neff. bw (GB/s)", 1.0, None),
        ("ttft_ns", "TTFT (ms)", MS, None),
     ]),
    ("08_kv_tp_group_skew_vs_buffer.png",
     "KV arrival skew within each TP group  (one line per os level)", [
        ("kv_tp_skew_mean_ns", "Cross-shard skew,\nmean (ms)", MS, None),
        ("kv_tp_skew_p99_ns", "Cross-shard skew,\np99 (ms)", MS, None),
     ]),
    ("09_decode_first_allreduce_vs_buffer.png",
     "Skew inherited by the decode first TP all-reduce (worst stage)", [
        ("dec_ar_first_skew_ns", "First AR entry\nskew (ms)", MS, None),
        ("dec_ar_first_over_rest", "First AR duration\n(× steady-state)", 1.0, 1.0),
     ]),
    ("10_decode_kv_stall_vs_buffer.png",
     "How much the decode is stalled by its KV transfer", [
        ("decode_kv_stall_ns", "First-pass KV\nstall (ms)", MS, 0.0),
        ("tok2_over_itl", "First decode pass\n(× steady ITL)", 1.0, 1.0),
        ("dec_kv_lateness_ns", "KV ready − first\ninput (ms)", MS, 0.0),
     ]),
]

# (col, ylabel, scale, yscale, title, filename) -- one line per os level.
#   SEVERITY   pause_pct_of_window   how long the bottleneck is backpressured
#   OUTCOME    eff_pct, kv_gate, ttft   what the KV/prefill actually cost
#   MECHANISM  conc_mean, qpeak      why: bufferbloat admits more concurrency,
#                                    and whether the extra buffer is even used
LINE_SPECS = [
    ("link0_pause_pct_of_window", "Bottleneck paused (% of window)", 1.0, None,
     "Congestion severity vs buffer  (per os level)",
     "02_pause_pct_vs_buffer.png"),
    ("link0_eff_pct", "Delivered KV bw (% of 200G uplink)", 1.0, None,
     "Delivered KV bandwidth vs buffer  (outcome, per os level)",
     "03_delivered_bw_vs_buffer.png"),
    ("kv_gate_ns", "KV completion / decode gate (ms)", MS, None,
     "KV completion time vs buffer  (outcome, per os level)",
     "04_kv_completion_vs_buffer.png"),
    ("ttft_ns", "TTFT (ms)", MS, None,
     "TTFT vs buffer  (outcome, per os level)",
     "05_ttft_vs_buffer.png"),
    ("link0_conc_mean", "Mean concurrent KV flows", 1.0, None,
     "Bottleneck concurrency vs buffer  (mechanism, per os level)",
     "06_concurrency_vs_buffer.png"),
    ("link0_qpeak_bytes", "Peak queue occupancy (kB)", 1e-3, None,
     "Bottleneck queue peak vs buffer  (mechanism, per os level)",
     "07_queue_peak_vs_buffer.png"),
]

# The three that read as a clean os x buffer field: one severity, two outcomes.
HEATMAP_SPECS = [
    ("link0_pause_pct_of_window", "Bottleneck paused (% of window)", "{:.0f}",
     "magma", 1.0, "11_heatmap_pause_pct.png"),
    ("kv_gate_ns", "KV completion (ms)", "{:.0f}", "magma", MS,
     "12_heatmap_kv_completion_ms.png"),
    ("link0_eff_pct", "Delivered KV bw (% of 200G uplink)", "{:.0f}", "viridis", 1.0,
     "13_heatmap_delivered_bw.png"),
]


def make_2d_plots(s: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    name, title, panels = STORY_SPECS[0]                     # 01 causal chain
    multi_panel(s, panels, title, name, outdir, written)
    for col, ylabel, scale, yscale, title, name in LINE_SPECS:   # 02-07
        line_family(s, col, ylabel, title, name, outdir, written,
                    scale=scale, yscale=yscale)
    for name, title, panels in STORY_SPECS[1:]:              # 08-10
        multi_panel(s, panels, title, name, outdir, written)
    for col, title, fmt, cmap, scale, name in HEATMAP_SPECS:     # 11-13
        heatmap(s, col, f"{title} — oversubscription × buffer", name, outdir,
                written, fmt=fmt, cmap=cmap, scale=scale)
    return written


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    for act in ap._actions:
        if act.dest == "out":
            act.help = f"output dir (default: results/sweep_analysis/{KIND}/<workload>/<sweep>)"
    roles.add_argument(ap)
    ap.add_argument("--bottleneck", default="8->12",
                    help="'sw->peer' forced as the bottleneck on every run "
                         "(default: 8->12, the oversubscribed PCIe->ToR uplink).")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links summary_2d.csv carries (default: 6)")
    ap.add_argument("--no-per-os", action="store_true",
                    help="skip the full per-oversubscription buffer_sweep figure sets "
                         "(by_oversub/os<N>/); emit only the 2D synthesis. Faster: "
                         "skips the per-run queue-timeline (qlen.txt) reads.")
    a = ap.parse_args(argv)

    try:
        p = paths.SweepPaths(sweep=a.sweep, workload=a.workload, root=Path(a.root))
        outdir = (Path(a.out) if a.out else
                  p.root / "results" / "sweep_analysis" / KIND / p.workload / p.sweep)
        placement = Placement.parse(a.placement)
        want_series = not a.no_per_os

        need(not p.missing_roots(),
             "derived root(s) do not exist:\n    " + "\n    ".join(p.missing_roots())
             + f"\n  --sweep {a.sweep!r} is probably wrong.")
        all_tags = p.tags("ns3")
        need(all_tags, f"no run sub-directory under {p.ns3_root}")

        # group the flat run set by oversubscription level; each group is a plain
        # buffer sweep (one buffer axis) that buffer_sweep.analyse_sweep can score.
        by_os: dict[float, list[str]] = defaultdict(list)
        for t in all_tags:
            os_ratio = OVERSUB_AXIS.value(t)
            need(os_ratio is not None,
                 f"{t}: no 'os<num>' token; the oversubscription axis is unreadable.")
            by_os[os_ratio].append(t)

        print(p.describe())
        print(f"  out      {outdir}")
        print(f"  bottleneck (forced): {a.bottleneck}   per-os sets: {not a.no_per_os}")
        print(f"\n{len(by_os)} oversubscription levels:")

        fresh_dir(outdir)

        frames, per_os_plots = [], 0
        for os_ratio in sorted(by_os):
            os_tags = by_os[os_ratio]
            rows, s_os, labels = analyse_sweep(
                p, placement, top_links=a.top_links, bn_force=a.bottleneck,
                verbose=False, want_series=want_series, tags=os_tags)
            # full buffer_sweep figure set for THIS oversubscription level
            if want_series:
                os_dir = outdir / "by_oversub" / f"os{os_ratio:g}"
                per_os_plots += len(buffer_sweep.make_plots(rows, s_os, os_dir, labels))
            s_os.insert(0, "oversub", os_ratio)
            frames.append(s_os)
            row0 = s_os.sort_values("buffer_mb").iloc[0]
            print(f"  + os {os_ratio:g}:1  ({len(os_tags)} buffers)  "
                  f"bn={row0.get('bottleneck', '?')}  "
                  f"paused@buf{row0['buffer_mb']:g}="
                  f"{row0.get('link0_pause_pct_of_window', float('nan')):.0f}%")

        s = pd.concat(frames, ignore_index=True).sort_values(["oversub", "buffer_mb"])
        # first decode pass in units of this level's own steady inter-token gap:
        # the dimensionless twin of decode_kv_stall_ns (same as buffer_compare).
        s["tok2_over_itl"] = s["tok2_latency_ns"] / s["itl_steady_ns"]
        s.to_csv(outdir / "summary_2d.csv", index=False)
        plots = make_2d_plots(s, outdir)

        pd.set_option("display.width", 200)
        report = [c for c in ["oversub", "buffer_mb",
                              "link0_pause_pct_of_window", "link0_conc_mean",
                              "link0_eff_pct", "kv_gate_ns", "ttft_ns",
                              "pp_skew_ns", "decode_kv_stall_ns"]
                  if c in s.columns]
        print("\n================ OVERSUBSCRIPTION × BUFFER (2D) ================")
        print(s[report].to_string(index=False))
        print(f"\nWrote {outdir}:")
        print(f"  summary_2d.csv")
        for q in plots:
            print(f"  {q.name}")
        if want_series:
            print(f"  by_oversub/os*/ — {per_os_plots} per-level buffer_sweep figures")
        return drain_warnings(" from the per-level buffer analysis — the "
                              "numbers are conditional on them")
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
