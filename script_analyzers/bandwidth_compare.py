#!/usr/bin/env python3
"""
bandwidth_compare — same sweep, different workloads: whose makespan scales
better with link bandwidth?

Auto-discovers every workload directory under output/astra_logs that ran the
given sweep (bandwidth_sweep by default) and puts them all on one figure.

Absolute makespan is not the right axis here: a 1-request/32768-prompt run and
a 64-request/512-prompt run finish on completely different wall-clock scales,
so a shared ms axis would just show "which workload does more work" and hide
how each one responds to bandwidth. The comparable quantity is speedup,
makespan[bx_min] / makespan[bx], normalised per workload to its OWN lowest-
bandwidth run -- same normalisation bandwidth_sweep.py's speedup plot uses for
a single workload, just overlaid instead of drawn once.

Discovery
---------
    <ROOT>/output/astra_logs/<workload>/<sweep>/<tag>/stats_sys*.csv

Every sub-directory of output/astra_logs that contains a `<sweep>` directory is
a workload; nothing is hard-coded. Run with --list to see what would be picked
up without analysing anything.

Reuses bandwidth_sweep.load_run / summarise_run so a workload is scored
exactly the same way here as in the single-workload analysis -- one
definition of "makespan", two tools.

Usage
-----
    python3 bandwidth_compare.py
    python3 bandwidth_compare.py --sweep bandwidth_sweep --workloads 'llama2_13b_*'
    python3 bandwidth_compare.py --list
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt

from utils import paths
from utils.paths import BANDWIDTH_AXIS, BANDWIDTH_GBPS_TO_BYTES_PER_NS
from utils.plots import save_fig
from bandwidth_sweep import load_run, summarise_run, Abort, need


def discover_workloads(root: Path, sweep: str) -> list[str]:
    astra_logs = root / "output" / "astra_logs"
    if not astra_logs.is_dir():
        return []
    return sorted(p.name for p in astra_logs.iterdir()
                  if p.is_dir() and (p / sweep).is_dir())


def load_workload(root: Path, workload: str, sweep: str, pattern: str) -> pd.DataFrame:
    """One row per (workload, bandwidth) run -- mirrors bandwidth_sweep.main's
    loop so a workload is scored identically whether analysed alone or here."""
    p = paths.SweepPaths(sweep=sweep, workload=workload, root=root)
    tags = p.tags("astra")
    need(tags, f"{workload}: no run sub-directory under {p.astra_root}")
    rows = []
    for tag in tags:
        bw = BANDWIDTH_AXIS.value(tag)
        need(bw is not None,
             f"{workload}/{tag}: no 'bx<num>' token in the directory name")
        df = load_run(p.astra_run(tag), pattern)
        need(df is not None,
             f"{workload}/{tag}: no readable {pattern} under {p.astra_run(tag)}")
        summ = summarise_run(df)
        summ.update(run_dir=tag, variant=BANDWIDTH_AXIS.variant(tag), bandwidth=bw)
        rows.append(summ)
    summary = pd.DataFrame(rows).sort_values("bandwidth").reset_index(drop=True)
    # Same guard as bandwidth_sweep.py: one line per workload here too, so a
    # sweep that moves a second knob would zigzag instead of drawing a curve.
    need(summary["variant"].nunique() == 1,
         f"{workload}: sweep moves more than one knob: variants "
         f"{sorted(summary['variant'].unique())}. Split into one sweep per "
         f"variant before comparing across workloads.")
    summary.insert(0, "workload", workload)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="bandwidth_sweep",
                    help="sweep sub-directory name to look for under every "
                         "workload (default: bandwidth_sweep)")
    ap.add_argument("--root", default=str(paths.ROOT), type=Path,
                    help=f"project root (default: {paths.ROOT})")
    ap.add_argument("--workloads", nargs="+", default=None,
                    help="glob pattern(s) to keep (default: every workload that "
                         "has the sweep), e.g. --workloads 'llama2_13b_*'")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="glob pattern(s) to drop")
    ap.add_argument("--pattern", default="*.csv",
                    help="glob for the per-node CSVs inside each run dir")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/"
                         "bandwidth_compare/<sweep>)")
    ap.add_argument("--list", action="store_true",
                    help="print discovered workloads and exit, without analysing")
    a = ap.parse_args(argv)

    root = Path(a.root)
    workloads = discover_workloads(root, a.sweep)
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
             f"no workload under {root / 'output' / 'astra_logs'} has a "
             f"{a.sweep!r} sub-directory (or --workloads/--exclude filtered "
             f"all of them out)")

        outdir = (Path(a.out) if a.out else
                  root / "results" / "sweep_analysis" / "bandwidth_compare" / a.sweep)

        frames = []
        print(f"\nScanning {len(workloads)} workload(s):")
        for w in workloads:
            summ = load_workload(root, w, a.sweep, a.pattern)
            base = summ["makespan_ns"].iloc[0]  # lowest bandwidth = baseline
            summ["speedup"] = base / summ["makespan_ns"]
            # Normalised/ratio metrics only -- same reason speedup replaces raw
            # makespan above: a workload with 10x the tokens has a 10x bigger
            # kv_busy_union_ns without being any more fabric-bound, so the
            # absolute ns figure would just re-show "which workload does more
            # work". Dividing by that workload's own makespan makes it comparable.
            summ["kv_exposed_frac"] = summ["kv_exposed_ns"] / summ["makespan_ns"]
            # kv_mean_link_bw (mean per-transfer rate), not kv_agg_bw_bytes_per_ns
            # (bytes / first-to-last-send window): once KV is masked by compute
            # the sends are spread out at the compute's pace, and the window
            # metric conflates that spacing with the wire rate. bandwidth is bx,
            # written into physical_topology.txt as Gbps; bw_bytes_per_ns is
            # GB/s decimal -- an 8x unit gap, not a run difference.
            summ["kv_bw_efficiency"] = (
                summ["kv_mean_link_bw"]
                / (summ["bandwidth"] * BANDWIDTH_GBPS_TO_BYTES_PER_NS))
            frames.append(summ)
            print(f"  + {w:<55} bx={summ['bandwidth'].min():g}.."
                  f"{summ['bandwidth'].max():g}  "
                  f"speedup@max_bx={summ['speedup'].iloc[-1]:.2f}")

        combined = pd.concat(frames, ignore_index=True)
        front = ["workload", "run_dir", "variant", "bandwidth", "makespan_ms", "speedup",
                 "kv_exposed_frac", "kv_over_prefill_compute", "kv_bw_efficiency"]
        combined = combined[[c for c in front if c in combined.columns]
                            + [c for c in combined.columns if c not in front]]
        outdir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(outdir / "summary.csv", index=False)

        written: list[Path] = []

        def line_by_workload(ycol: str, ylabel: str, title: str, fname: str,
                             hline: float | None = None) -> None:
            """One line per workload, x = bandwidth. Skipped when the column is
            absent or entirely NaN, so a metric a workload's runs never produced
            (e.g. no KV transfer at all) just drops out instead of an empty line."""
            if ycol not in combined.columns or not combined[ycol].notna().any():
                return
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for w, grp in combined.groupby("workload"):
                grp = grp.dropna(subset=[ycol]).sort_values("bandwidth")
                if grp.empty:
                    continue
                ax.plot(grp["bandwidth"], grp[ycol], marker="o", label=w)
            if hline is not None:
                ax.axhline(hline, color="k", linestyle=":", alpha=0.4)
            ax.set_xlabel("Simulated link bandwidth (bx)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}\n(sweep: {a.sweep})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            save_fig(fig, outdir, fname, written)

        line_by_workload(
            "speedup", "Speedup (makespan at lowest bw / makespan)",
            "Makespan speedup vs bandwidth, across workloads\n"
            "(each normalised to its own lowest-bw run)",
            "speedup_by_workload.png", hline=1.0)

        line_by_workload(
            "kv_exposed_frac", "KV transfer exposed time / makespan",
            "KV-cache transfer exposure vs bandwidth, across workloads\n"
            "(fraction of the run where KV is in flight AND no compute masks it "
            "anywhere in the system -- time genuinely added to the critical path)",
            "kv_exposed_fraction_by_workload.png")

        line_by_workload(
            "kv_over_prefill_compute", "KV completion / prefill compute completion",
            "Does the fabric gate the prefill→decode handover?, across workloads\n"
            "(>1 means the KV transfer outlasts the prefill compute that feeds it)",
            "kv_over_prefill_compute_by_workload.png", hline=1.0)

        line_by_workload(
            "kv_bw_efficiency", "Mean per-transfer KV rate / nominal link rate",
            "KV bandwidth efficiency vs nominal, across workloads\n"
            "(close to 1 once kv_over_prefill_compute -> ~1: each transfer runs "
            "near the wire rate; a low value at low bandwidth is real contention, "
            "not a metric artifact)",
            "kv_bandwidth_efficiency_by_workload.png", hline=1.0)

        print(f"\nWrote {outdir}:")
        print("  summary.csv")
        for p in written:
            print(f"  {p.name}")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
