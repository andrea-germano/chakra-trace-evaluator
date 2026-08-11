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
   "across models" and "across oversubscription" read identically). The
   time-domain figures (02 cumulative arrival, 07/08 occupancy/queues over
   time) are per-level material and are not duplicated at the top level: they
   live in by_oversub/os<N>/. See buffer_compare's module docstring for the
   figure-by-figure mapping and for what is raw vs normalised. One caveat
   specific to this plane: the PAUSE-frame count (fig 09, middle panel)
   INVERTS as os rises -- longer continuous pauses mean FEWER pause/resume
   events (132k -> 67k -> 51k at buf2 even though congestion worsens) -- so
   read it per line, along the buffer axis, not across lines; the monotone
   severity metric, link0_pause_pct_of_window, stays in summary_2d.csv.

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

import buffer_sweep
from buffer_sweep import analyse_sweep
from buffer_compare import add_group_norms, story_plots

from utils import paths, roles
from utils.cli import Abort, drain_warnings, need
from utils.paths import OVERSUB_AXIS, fresh_dir
from utils.roles import Placement

KIND = "oversub2d"


# --------------------------------------------------------------------------- #
# Top-level 2D synthesis: buffer_sweep's own figure set, one line family per
# oversubscription level -- buffer_compare.story_plots, the same figures the
# cross-model compare draws (same numbers, same quantities). The time-domain
# figures (02/07/08) are per-level material and live in by_oversub/os<N>/.
# --------------------------------------------------------------------------- #
def make_2d_plots(s: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    return story_plots(s, "oversub", outdir, label=lambda v: f"{v:g}:1")


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
