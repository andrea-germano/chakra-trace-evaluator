#!/usr/bin/env python3
"""
incast_sweep — the SAME PFC/backpressure phenomenon the buffer sweep studies,
now driven by the INCAST DEGREE instead of by the buffer.

The buffer sweep held one topology and moved the per-switch buffer. This sweep
holds the buffer axis too but stacks it under a second, coarser knob: the incast
degree = the prefill tensor-parallel width (T2.1->tp2, T3->tp4, T4->tp8), i.e.
how many prefill ranks shard-and-send the KV cache into each decode rank. More
senders converging on one receiver is textbook incast, and the guess this script
tests is that it reproduces the buffer sweep's chain -- deep queue on the
oversubscribed ToR/core link -> PFC PAUSE -> KV delivery stalls and skews across
decode ranks -> decode start slips relative to TTFT -- with the incast degree,
not the buffer, as the thing that turns it on.

Because the mechanism is identical, the MEASUREMENT is identical: this reuses
buffer_sweep.analyse_sweep and buffer_sweep.make_plots verbatim, once per incast
level, so a level is scored by exactly the same code (and produces the same seven
figures) as a standalone buffer sweep -- one definition of the metrics, reused,
the same discipline buffer_compare follows across models. Everything incast-
specific is confined to two places: utils.incast (the split config/output path
names and the per-level tag filter that SweepPaths cannot express) and the
cross-incast comparison below.

What differs from buffer_sweep, and why it has to
--------------------------------------------------------------------------------
  * One sweep dir, THREE topologies. incast_sweep holds T2.1/T3/T4, each its own
    buffer sub-sweep. buffer_sweep aborts on >1 variant by design, so this
    iterates the levels (utils.incast.IncastPaths is per-level) and runs the
    buffer machinery on each.
  * Three placements, not one. Prefill is TP2/TP4/TP8 across the levels, so the
    rank->role map differs per level and is NOT a single --placement flag.
    It is recovered per level from that level's ASTRA trace (roles.from_astra),
    the same source buffer_sweep only cross-checks against.

Output
--------------------------------------------------------------------------------
    <out>/<level>/                 the full buffer-sweep figure set + summary.csv
                                   for that incast degree (buffer on x)
    <out>/_cross_incast/           the comparison: every level overlaid, buffer on
                                   x (one line per incast degree), plus the
                                   headline metrics with the incast degree ITSELF
                                   on x. summary.csv is every level's rows, with
                                   `level` and `incast_degree` columns.

Usage
-----
    python3 incast_sweep.py
    python3 incast_sweep.py --levels T3 T4
    python3 incast_sweep.py --top-links 4 -o /tmp/incast
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import incast, roles
from utils.cli import Abort, need
from utils.plots import logx_pow2, save_fig
from utils.roles import Placement

# The whole per-run + per-sweep measurement pipeline and its seven figures are
# buffer_sweep's; incast only re-orchestrates them. Importing rather than copying
# is the point -- a level is scored identically to a standalone buffer sweep.
import buffer_sweep as bs
from buffer_sweep import MS

NAN = float("nan")
BLUE, CORAL, GREEN, VIOLET, MUTED = bs.BLUE, bs.CORAL, bs.GREEN, bs.VIOLET, bs.MUTED


# --------------------------------------------------------------------------- #
# Per-level placement, recovered not declared
# --------------------------------------------------------------------------- #
def recover_placement(p: incast.IncastPaths, tags: list[str]) -> Placement:
    """The level's rank->role map, read from its ASTRA trace. buffer_sweep takes
    this on the CLI because a buffer sweep is one placement; an incast sweep is
    three, one per prefill TP width, so it is recovered here (MLSynth writes
    pl=/ss=/sh= into every op name -- roles.from_astra). Tries each run until one
    has a readable trace, so a single missing/half-written ASTRA dir does not
    sink the level."""
    last_err = None
    for tag in tags:
        adir = p.astra_run(tag)
        if not adir.is_dir():
            continue
        try:
            return roles.from_astra(adir)
        except Exception as e:                                  # noqa: BLE001
            last_err = e
    raise Abort(f"level {p.level}: no run has a readable ASTRA trace to recover "
                f"the placement from (needed because prefill TP width, hence the "
                f"rank->role map, differs per level). Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Comparable columns for the cross-incast overlay
# --------------------------------------------------------------------------- #
def derive_compare_cols(s: pd.DataFrame) -> pd.DataFrame:
    """The handful of derived quantities the cross-incast figures plot, named the
    same way buffer_compare names them so the two comparison tools stay legible
    together. The bottleneck-link stats come from link0 (the measured bottleneck)
    exactly as buffer_compare reads them."""
    s = s.copy()
    # -- the incast signal itself: KV arrival misalignment across decode ranks.
    #    On a buffer sweep this barely moved; here it is what the incast degree is
    #    expected to drive. A delta (idle time), already comparable in absolute ms.
    s["kv_skew_ms"] = s["cross_rank_skew_ns"] / 1e6
    s["pp_skew_us"] = s["pp_skew_ns"] / 1e3
    s["ttft_ms"] = s["ttft_ns"] * MS
    s["decode_start_ms"] = s["kv_gate_ns"] * MS
    # -- fabric-domain magnitudes at the measured bottleneck (link0), raw units --
    s["line_rate_pct"] = s.get("link0_eff_pct")
    s["conc_peak"] = s.get("link0_conc_peak")           # peak concurrent KV flows:
                                                        # the incast fan-in as seen
                                                        # on the wire
    s["pause_frames"] = s.get("link0_pause_frames")
    win = s.get("link0_window_ns")
    s["pause_rate"] = (s["pause_frames"] / (win / 1e6)
                       if win is not None else NAN)     # PFC PAUSE frames per ms
    qb = s.get("link0_qpeak_bytes")
    s["qpeak_mb"] = qb / 2**20 if qb is not None else NAN
    # -- does it reach the user? decode start relative to TTFT is already
    #    dimensionless; ttft_slowdown is normalised to THIS level's largest buffer.
    tt = s.dropna(subset=["ttft_ns"]).sort_values("buffer_mb")
    ref = float(tt["ttft_ns"].iloc[-1]) if len(tt) else NAN
    s["ttft_slowdown"] = (s["ttft_ns"] / ref
                          if pd.notna(ref) and ref > 0 else NAN)
    s["ar_first_over_rest"] = s["rs_ar_first_ns"] / s["rs_ar_rest_mean_ns"]
    return s


# --------------------------------------------------------------------------- #
# One incast level
# --------------------------------------------------------------------------- #
def analyse_level(level: str, root: Path, out_workload: str, config_sweep: str,
                  top_links: int, outdir: Path) -> pd.DataFrame | None:
    """Score one incast level's buffer sub-sweep with the buffer-sweep pipeline,
    write its figure set under <outdir>/<level>, and return the level's summary
    rows (with level/incast_degree columns) for the cross-incast comparison.
    Returns None if the level has no analysable run."""
    p = incast.IncastPaths(level=level, out_workload=out_workload,
                           config_sweep=config_sweep, root=root)
    if p.missing_roots():
        bs.warn(f"{level}: skipped, derived root(s) missing:\n    "
                + "\n    ".join(p.missing_roots()))
        return None
    tags = p.tags("ns3")
    if not tags:
        bs.warn(f"{level}: no run under {p.ns3_root} for this level; skipped.")
        return None

    placement = recover_placement(p, tags)
    degree = incast.prefill_tp(placement)
    print(f"\n===== level {level}  (prefill TP{degree}, incast degree {degree}) =====")
    print(f"  placement {roles.spec_of(placement)}")

    # One corrupt config or half-written run in ONE level must not sink the whole
    # multi-level comparison: analyse_sweep aborts on the first bad run, so catch
    # it, warn, and drop just this level. The levels that parsed still get scored.
    try:
        rows, s, chosen_labels = bs.analyse_sweep(
            p, placement, top_links=top_links, bn_force=None, verbose=True)
    except Abort as e:
        bs.warn(f"{level}: dropped from the analysis — {e}")
        return None

    ldir = outdir / level
    if ldir.exists():
        shutil.rmtree(ldir)
    ldir.mkdir(parents=True, exist_ok=True)
    s.to_csv(ldir / "summary.csv", index=False)
    bs.make_plots(rows, s, ldir, chosen_labels)

    s = derive_compare_cols(s)
    s.insert(0, "incast_degree", degree)
    s.insert(0, "level", level)
    return s


# --------------------------------------------------------------------------- #
# Cross-incast comparison figures
# --------------------------------------------------------------------------- #
def _degree_label(level: str, degree: int) -> str:
    return f"{level} (tp{degree})"


def make_compare_plots(combined: pd.DataFrame, outdir: Path) -> list[Path]:
    """Every incast level overlaid. Two readings of the same table:
      A. buffer on x, one line per incast degree -- the per-level buffer response,
         side by side, so a metric flat on a buffer sweep but climbing with the
         degree is visible as a fan of near-flat lines at rising heights.
      B. the incast degree on x, one line per buffer -- the direct "does the
         phenomenon scale with incast?" view, for the headline metrics only.
    """
    written: list[Path] = []
    outdir.mkdir(parents=True, exist_ok=True)
    # stable degree order for legends/colours
    order = (combined[["level", "incast_degree"]].drop_duplicates()
             .sort_values("incast_degree"))
    levels = list(order["level"])
    deg_of = dict(zip(order["level"], order["incast_degree"]))
    cmap = plt.get_cmap("viridis")
    lvl_color = {lv: cmap(i / max(len(levels) - 1, 1))
                 for i, lv in enumerate(levels)}

    # ---- A: buffer on x, one line per incast level -------------------------- #
    def overlay_by_level(ycol: str, ylabel: str, title: str, fname: str,
                         scale: float = 1.0, hline: float | None = None,
                         logy: bool = False) -> None:
        if ycol not in combined.columns or not combined[ycol].notna().any():
            return
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for lv in levels:
            grp = (combined[combined["level"] == lv]
                   .dropna(subset=[ycol]).sort_values("buffer_mb"))
            if grp.empty:
                continue
            ax.plot(grp["buffer_mb"], grp[ycol] * scale, marker="o",
                    color=lvl_color[lv], label=_degree_label(lv, deg_of[lv]))
        if hline is not None:
            ax.axhline(hline, color="k", linestyle=":", alpha=0.4)
        if logy:
            ax.set_yscale("log")
        logx_pow2(ax, combined, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, title="incast level")
        save_fig(fig, outdir, fname, written)

    overlay_by_level("kv_skew_ms", "Cross-rank KV skew (ms)",
                     "KV arrival skew across decode ranks vs buffer",
                     "A01_kv_skew_ms_by_level.png")
    overlay_by_level("kv_gate_over_ttft", "Decode start (×TTFT)",
                     "Decode start relative to TTFT vs buffer",
                     "A02_decode_start_over_ttft_by_level.png", hline=1.0)
    overlay_by_level("line_rate_pct", "KV bandwidth (% of line-rate)",
                     "Delivered KV bandwidth at the bottleneck vs buffer",
                     "A03_line_rate_pct_by_level.png")
    overlay_by_level("conc_peak", "Peak concurrent KV flows",
                     "Incast fan-in on the wire vs buffer",
                     "A04_concurrency_by_level.png")
    overlay_by_level("pause_rate", "PFC PAUSE (frames/ms)",
                     "Backpressure intensity at the bottleneck vs buffer",
                     "A05_pause_rate_by_level.png")
    overlay_by_level("qpeak_mb", "Peak queue occupancy (MB)",
                     "Bottleneck buffer occupancy vs buffer",
                     "A06_qpeak_mb_by_level.png")
    overlay_by_level("ttft_ms", "TTFT (ms)",
                     "TTFT vs buffer (note: compute scale differs per level)",
                     "A07_ttft_ms_by_level.png")

    # ---- B: incast degree on x, one line per buffer ------------------------- #
    # The direct view of the sweep's own knob. Degree is 2/4/8 -> a log2 x-axis,
    # the same as the buffer one, keeps the small end readable.
    def overlay_by_buffer(ycol: str, ylabel: str, title: str, fname: str,
                          hline: float | None = None) -> None:
        if ycol not in combined.columns or not combined[ycol].notna().any():
            return
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        bufs = sorted(combined["buffer_mb"].dropna().unique())
        bcmap = plt.get_cmap("tab10")
        for i, bufv in enumerate(bufs):
            grp = (combined[combined["buffer_mb"] == bufv]
                   .dropna(subset=[ycol]).sort_values("incast_degree"))
            if grp.empty:
                continue
            ax.plot(grp["incast_degree"], grp[ycol], marker="s",
                    color=bcmap(i % 10), label=f"buf {bufv:g} MiB")
        if hline is not None:
            ax.axhline(hline, color="k", linestyle=":", alpha=0.4)
        logx_pow2(ax, combined, "incast_degree", "Incast degree (prefill TP width)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, title="buffer")
        save_fig(fig, outdir, fname, written)

    overlay_by_buffer("kv_skew_ms", "Cross-rank KV skew (ms)",
                      "KV arrival skew vs incast degree",
                      "B01_kv_skew_ms_vs_incast.png")
    overlay_by_buffer("conc_peak", "Peak concurrent KV flows",
                      "Incast fan-in vs incast degree",
                      "B02_concurrency_vs_incast.png")
    overlay_by_buffer("line_rate_pct", "KV bandwidth (% of line-rate)",
                      "Delivered KV bandwidth vs incast degree",
                      "B03_line_rate_pct_vs_incast.png")
    overlay_by_buffer("pause_rate", "PFC PAUSE (frames/ms)",
                      "Backpressure vs incast degree",
                      "B04_pause_rate_vs_incast.png")
    overlay_by_buffer("kv_gate_over_ttft", "Decode start (×TTFT)",
                      "Decode start relative to TTFT vs incast degree",
                      "B05_decode_start_over_ttft_vs_incast.png", hline=1.0)

    return written


# --------------------------------------------------------------------------- #
REPORT = ["level", "incast_degree", "buffer_mb", "bottleneck", "ttft_ms",
          "decode_start_ms", "kv_gate_over_ttft", "kv_skew_ms", "line_rate_pct",
          "conc_peak", "pause_rate", "qpeak_mb"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(incast.ROOT), type=Path,
                    help=f"project root (default: {incast.ROOT})")
    ap.add_argument("--out-workload", default=incast.OUT_WORKLOAD,
                    help=f"output dir name under output/<domain> holding the "
                         f"incast runs (default: {incast.OUT_WORKLOAD})")
    ap.add_argument("--config-sweep", default=incast.CONFIG_SWEEP,
                    help=f"config sub-dir name under configs/astra_sim/ns3 "
                         f"(default: {incast.CONFIG_SWEEP})")
    ap.add_argument("--levels", nargs="+", default=None,
                    help="incast levels to analyse, e.g. --levels T3 T4 "
                         "(default: every level found)")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links each level's figures carry "
                         "(default: 6)")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/incast/"
                         "<out-workload>)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    outdir = (Path(a.out) if a.out else
              root / "results" / "sweep_analysis" / "incast" / a.out_workload)

    try:
        levels = incast.discover_levels(a.out_workload, root, "ns3")
        need(levels,
             f"no incast level found under "
             f"{root / 'output' / 'ns3' / a.out_workload}. Is --out-workload "
             f"right? (expected sub-dirs like 'T3_bx100_dcqcn_buf8')")
        if a.levels:
            want = set(a.levels)
            missing = want - set(levels)
            need(not missing,
                 f"--levels {sorted(missing)} not present; found {levels}")
            levels = [l for l in levels if l in want]

        print(f"  root      {root}")
        print(f"  workload  {a.out_workload}")
        print(f"  configs   {root / 'configs' / 'astra_sim' / 'ns3' / a.config_sweep}")
        print(f"  out       {outdir}")
        print(f"  levels    {levels}")

        frames = []
        for level in levels:
            s = analyse_level(level, root, a.out_workload, a.config_sweep,
                              a.top_links, outdir)
            if s is not None:
                frames.append(s)
        need(frames, "no incast level produced any analysable run.")

        combined = pd.concat(frames, ignore_index=True)
        cdir = outdir / "_cross_incast"
        cdir.mkdir(parents=True, exist_ok=True)
        front = [c for c in REPORT if c in combined.columns]
        combined = combined[front + [c for c in combined.columns if c not in front]]
        combined.to_csv(cdir / "summary.csv", index=False)
        plots = make_compare_plots(combined, cdir)

        pd.set_option("display.width", 240)
        print("\n================ INCAST SWEEP (cross-level) ================")
        print(combined[[c for c in REPORT if c in combined.columns]]
              .to_string(index=False))
        print(f"\nWrote {outdir}:")
        for level in levels:
            if (outdir / level).is_dir():
                print(f"  {level}/  (buffer-sweep figure set + summary.csv)")
        print(f"  _cross_incast/summary.csv")
        for fpath in plots:
            print(f"  _cross_incast/{fpath.name}")

        if bs.WARNINGS:
            print(f"\n{len(bs.WARNINGS)} WARNING(S) — the numbers above are "
                  f"conditional on them:")
            for w in bs.WARNINGS:
                print(f"  ! {w}")
            return 1
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
