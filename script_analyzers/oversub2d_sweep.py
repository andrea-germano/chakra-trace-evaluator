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
1. by_oversub/os<N>/ -- the COMPLETE buffer_sweep figure set (01..10: causal
   chain to TTFT, KV cumulative arrival per stage, KV TP-shard skew boxplots,
   first-token-to-second waterfall, decode KV stall, decode all-reduce, buffer
   occupancy(t) with PFC pauses, per-switch queue timelines, per-link
   congestion, buffer bloat), one full set per oversubscription level. Same
   layout you already read on the buffer sweep, so every per-run/time-series
   figure stays legible (4 buffers per panel, not 16).
2. top level -- the 2D synthesis: buffer_sweep's OWN figure set with one line
   family per oversubscription level (buffer_compare.story_plots -- the same
   figures, same numbers, same quantities the cross-model compare draws, so
   "across models" and "across oversubscription" read identically), plus
   figure 14, the per-link view, which is this script's own: story_plots
   collapses the per-link figure to the measured bottleneck because link labels
   cannot cross MODELS, a restriction that does not apply when the second
   dimension is a link rate on one fixed topology. See _fig14_per_link_2d. The
   time-domain figures (02 cumulative arrival, 07/08 occupancy/queues over
   time) are per-level material and are not duplicated at the top level: they
   live in by_oversub/os<N>/. See buffer_compare's module docstring for the
   figure-by-figure mapping and for what is raw vs normalised. One caveat
   specific to this plane: the PAUSE-frame count (fig 09, middle panel)
   INVERTS as os rises -- longer continuous pauses mean FEWER pause/resume
   events (132k -> 67k -> 51k at buf2 even though congestion worsens) -- so
   read it per line, along the buffer axis, not across lines. The monotone
   severity metric, link0_pause_pct_of_window (51% -> 75% -> 85% on those same
   three points), used to live only in summary_2d.csv; it is now drawn beside
   the count in figure 12, so the inversion is visible rather than described.

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

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import buffer_sweep
from buffer_sweep import analyse_sweep
from buffer_compare import add_group_norms, story_plots

from utils import paths, roles
from utils.cli import Abort, drain_warnings, need
from utils.paths import OVERSUB_AXIS, fresh_dir
from utils.plots import logx_pow2, save_fig
from utils.roles import Placement

KIND = "oversub2d"
NAN = float("nan")


# --------------------------------------------------------------------------- #
# Top-level 2D synthesis: buffer_sweep's own figure set, one line family per
# oversubscription level -- buffer_compare.story_plots, the same figures the
# cross-model compare draws (same numbers, same quantities). The time-domain
# figures (02/07/08) are per-level material and live in by_oversub/os<N>/.
# --------------------------------------------------------------------------- #
def make_2d_plots(s: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = story_plots(s, "oversub", outdir, label=lambda v: f"{v:g}:1")
    _fig14_per_link_2d(s, outdir, written)
    return written


# --------------------------------------------------------------------------- #
# 14 PER-LINK CONGESTION ACROSS THE WHOLE PLANE -- oversub2d-local on purpose.
# --------------------------------------------------------------------------- #
# buffer_sweep's figure 09 is per-link; story_plots' version of it (figure 09
# here) collapses to link0, the measured bottleneck, because buffer_compare's
# grouping dimension is the MODEL and link labels are topology-local -- 'sw12'
# means a different switch in two different topologies, so the lines cannot be
# put on one axis.
#
# That reason does not hold on this plane. Every run of an oversubscription
# sweep is the SAME topology with the same node numbering (only link RATES
# change), and the sweep additionally forces one bottleneck (--bottleneck
# 8->12), so '8->12' denotes the same link in all 16 runs. Restricting to link0
# here therefore throws away exactly what buffer_sweep 09 exists to show, and
# what an oversubscribed tree is most often wrong about:
#
#   * a SECOND uplink (9->12) carries comparable backpressure -- the bottleneck
#     ranking picks one, the fabric congests on both;
#   * the spine DOWNLINKS (12->10, 12->11) never queue and never pause, which
#     localises the congestion to the ingress side of the core switch and says
#     there is no incast on the KV fan-in -- an absence that only a per-link
#     figure can state.
#
# Keyed by LABEL, never by index: analyse_sweep is called once per
# oversubscription level and ranks each level's links independently, so
# 'link0_*' is 8->12 at one level and something else at another. The label
# columns are the join key.
def _fig14_per_link_2d(s: pd.DataFrame, outdir: Path, written: list[Path]) -> None:
    idx = [int(c[4:-6]) for c in s.columns
           if c.startswith("link") and c.endswith("_label")]
    if not idx:
        return
    # The forced bottleneck first (it carries the emphasis weight), then the
    # rest ALPHABETICALLY rather than in discovery order. Discovery order is
    # each level's own congestion ranking, so it differs between two sweeps that
    # cross the identical link set -- the T1 and T7 planes ranked 9->12 and
    # 12->10 the other way round and the same link came out a different colour
    # in the two figures, which is exactly the comparison this figure exists to
    # support. Sorting makes the colour of a link a property of its name.
    seen: set[str] = set()
    for i in sorted(idx):
        seen.update(str(v) for v in s[f"link{i}_label"].dropna().unique() if v)
    if not seen:
        return
    bn = s["bottleneck"].dropna()
    first = str(bn.iloc[0]) if len(bn) and str(bn.iloc[0]) in seen else None
    labels = ([first] if first else []) + sorted(seen - {first})
    levels = sorted(s["oversub"].unique())
    if not labels or not levels:
        return

    def series(sub: pd.DataFrame, label: str, field: str):
        """(buffer, value) for one link on one oversubscription level, gathered
        across whichever link index carried that label in each run."""
        out = []
        for r in sub.sort_values("buffer_mb").itertuples():
            val = NAN
            for i in sorted(idx):
                if getattr(r, f"link{i}_label", None) == label:
                    val = getattr(r, f"link{i}_{field}", NAN)
                    break
            out.append((r.buffer_mb, val))
        xs = [b for b, v in out if pd.notna(v)]
        ys = [v for _b, v in out if pd.notna(v)]
        return xs, ys

    cols = [("eff_pct", 1.0, "Link busy (% of KV window)", None),
            ("pause_pct_of_window", 1.0, "Worst victim held (%)", None),
            ("qpeak_bytes", 1 / 2**20, "Peak occupancy (MiB)", None)]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(levels), len(cols), squeeze=False,
                             sharex=True, sharey="col",
                             figsize=(5.2 * len(cols), 2.9 * len(levels) + 1.0))
    for r_i, lvl in enumerate(levels):
        sub = s[s["oversub"] == lvl]
        rate = sub["bn_rate_gbps"].dropna()
        rate_txt = f"pinch {rate.iloc[0]:g} Gb/s" if len(rate) else ""
        for c_i, (field, scale, ylabel, _yscale) in enumerate(cols):
            ax = axes[r_i][c_i]
            for l_i, label in enumerate(labels):
                xs, ys = series(sub, label, field)
                if not xs:
                    continue
                ax.plot(xs, [y * scale for y in ys], marker="o", ms=4,
                        lw=2.4 if l_i == 0 else 1.1, color=cmap(l_i % 10),
                        label=label if (r_i == 0 and c_i == 0) else None)
            logx_pow2(ax, sub, "buffer_mb", "Per-switch buffer (MiB)")
            ax.grid(True, alpha=0.3, which="both")
            if c_i == 0:
                ax.set_ylabel(f"{lvl:g}:1\n{rate_txt}", fontsize=9)
            if r_i == 0:
                ax.set_title(ylabel, fontsize=10)
            if r_i != len(levels) - 1:
                ax.set_xlabel("")
    axes[0][0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Congestion per KV-crossed link — rows = oversubscription "
                 "level", y=1.0)
    save_fig(fig, outdir, "14_per_link_congestion_2d.png", written)


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
        # tok2_over_itl / ttft_slowdown, normalised per os level -- the same
        # definition the cross-model compare uses (buffer_compare.add_group_norms).
        s = add_group_norms(s, "oversub")
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
