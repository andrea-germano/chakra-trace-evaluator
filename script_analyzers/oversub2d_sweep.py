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
2. top level -- the 2D synthesis across all OS levels: line families (one line
   per oversubscription level, x = buffer) and heatmaps (os x buffer) for the
   signals that read as a CLEAN FIELD on this plane -- congestion severity
   (paused fraction of the window, NOT the pause-frame count, which inverts as
   os rises), the two accurate outcomes (delivered KV bandwidth and KV
   completion time), TTFT, and the mechanism (concurrency, queue peak). This is
   where the "knee moves right as os grows" story is read directly. The
   tail-dominated skew/stall scalars are deliberately NOT synthesised here --
   they oscillate non-monotonically with buffer and are read per os level in the
   by_oversub sets instead.

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
def line_family(s: pd.DataFrame, col: str, ylabel: str, title: str, name: str,
                outdir: Path, written: list, scale: float = 1.0,
                yscale: str | None = None) -> None:
    if col not in s.columns or not s[col].notna().any():
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    levels = sorted(s["oversub"].unique())
    denom = max(len(levels) - 1, 1)
    for i, os_ratio in enumerate(levels):
        sub = s[s["oversub"] == os_ratio].sort_values("buffer_mb")
        if not sub[col].notna().any():
            continue
        ax.plot(sub["buffer_mb"], sub[col] * scale, marker="o",
                color=cmap(i / denom), label=f"{os_ratio:g}:1")
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    if yscale:
        ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="oversubscription", fontsize=8)
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
# Top-level 2D synthesis: SEVERITY + OUTCOME + MECHANISM across os levels.
# --------------------------------------------------------------------------- #
# On the os x buffer plane, only metrics that read as a clean field survive here.
# The old set carried every buffer_sweep scalar, including the tail-dominated
# ones (pp_skew, kv_tp_skew, dec_ar_first_*, decode_kv_stall, dec_kv_lateness):
# on this plane those are set by a single straggler and oscillate non-monotonically
# with buffer, so a heatmap of them paints a gradient that is deterministic
# sensitivity, not a trend. They are all still drawn PER OS LEVEL in
# by_oversub/os<N>/ (buffer_sweep figs 08/10/11) where each is read on its own
# time series; they are just no longer synthesised on the 2D plane.
#
# Worse, the old headline metric here -- link0_pause_frames -- INVERTS with
# oversubscription: as os rises the bottleneck stays paused in longer continuous
# stretches, so the pause/resume COUNT drops (132k -> 67k -> 51k at buf2) even
# though congestion worsens. The severity that actually rises monotonically is
# the paused FRACTION of the window (51% -> 75% -> 85%), so that is the severity
# signal now, backed by the two accurate outcome metrics (delivered bw and KV
# completion, from fct bytes -- not affected by the pfc qIndex caveat).
#
# (col, ylabel, scale, yscale, title, filename) -- one line per os level.
#   SEVERITY   pause_pct_of_window   how long the bottleneck is backpressured
#   OUTCOME    eff_pct, kv_gate, ttft   what the KV/prefill actually cost
#   MECHANISM  conc_mean, qpeak      why: bufferbloat admits more concurrency,
#                                    and whether the extra buffer is even used
LINE_SPECS = [
    ("link0_pause_pct_of_window", "Bottleneck paused (% of window)", 1.0, None,
     "Congestion severity vs buffer  (per os level)",
     "01_pause_pct_vs_buffer.png"),
    ("link0_eff_pct", "Delivered KV bw (% of 200G uplink)", 1.0, None,
     "Delivered KV bandwidth vs buffer  (outcome, per os level)",
     "02_delivered_bw_vs_buffer.png"),
    ("kv_gate_ns", "KV completion / decode gate (ms)", MS, None,
     "KV completion time vs buffer  (outcome, per os level)",
     "03_kv_completion_vs_buffer.png"),
    ("ttft_ns", "TTFT (ms)", MS, None,
     "TTFT vs buffer  (outcome, per os level)",
     "04_ttft_vs_buffer.png"),
    ("link0_conc_mean", "Mean concurrent KV flows", 1.0, None,
     "Bottleneck concurrency vs buffer  (mechanism, per os level)",
     "05_concurrency_vs_buffer.png"),
    ("link0_qpeak_bytes", "Peak queue occupancy (kB)", 1e-3, None,
     "Bottleneck queue peak vs buffer  (mechanism, per os level)",
     "06_queue_peak_vs_buffer.png"),
]

# The three that read as a clean os x buffer field: one severity, two outcomes.
HEATMAP_SPECS = [
    ("link0_pause_pct_of_window", "Bottleneck paused (% of window)", "{:.0f}",
     "magma", 1.0, "07_heatmap_pause_pct.png"),
    ("kv_gate_ns", "KV completion (ms)", "{:.0f}", "magma", MS,
     "08_heatmap_kv_completion_ms.png"),
    ("link0_eff_pct", "Delivered KV bw (% of 200G uplink)", "{:.0f}", "viridis", 1.0,
     "09_heatmap_delivered_bw.png"),
]


def make_2d_plots(s: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for col, ylabel, scale, yscale, title, name in LINE_SPECS:
        line_family(s, col, ylabel, title, name, outdir, written,
                    scale=scale, yscale=yscale)
    for col, title, fmt, cmap, scale, name in HEATMAP_SPECS:
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
        s.to_csv(outdir / "summary_2d.csv", index=False)
        plots = make_2d_plots(s, outdir)

        pd.set_option("display.width", 200)
        report = [c for c in ["oversub", "buffer_mb",
                              "link0_pause_pct_of_window", "link0_conc_mean",
                              "link0_eff_pct", "kv_gate_ns", "ttft_ns"]
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
