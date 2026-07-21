#!/usr/bin/env python3
"""
buffer_compare — same buffer sweep, different MODELS: whose causal chain
(PP skew -> receiving-stage all-reduce -> TTFT) responds the same way to the
per-switch buffer?

The cross-model companion to buffer_sweep, exactly as buffer_compare is to
buffer_sweep_v2. Auto-discovers every workload directory under output/ns3 that
ran the given sweep (buffer_sweep_T1 by default) and overlays them on one
figure per metric. Reuses buffer_sweep.analyse_sweep so a model is scored
here identically to the single-model analysis -- one definition of the v3
metrics, two tools.

What goes on the axes, and why
--------------------------------------------------------------------------------
Two kinds of quantity, in two blocks.

Block A -- FABRIC-domain magnitudes, plotted RAW (absolute units). A PP arrival
skew, a queue depth, a PAUSE count, a link utilisation all live in the network
domain: they are caused by the fabric + buffer, not by how many parameters or
tokens a model has, so they do NOT carry the model's compute scale the way
ttft_ns or a tensor collective does. Their raw value is already the physical
number to compare across models -- no normalisation:

    pp_skew_ms           pp_skew_ns / 1e6: cross-rank arrival misalignment on the
                         receiving stage. A delta (idle ms), not a workload size,
                         so directly comparable across models.
    qpeak_mb             link0_qpeak_bytes / 2^20: peak occupancy at the
                         bottleneck port, in MB. Absolute bytes -- deliberately
                         NOT qpeak_pct, whose denominator is the swept buffer
                         (dividing a queue by the buffer is circular on a buffer
                         sweep).
    pause_frames         link0_pause_frames: PFC PAUSE event count at the
                         bottleneck link. Raw count -- also grows with run
                         duration, so read alongside the KV window.
    line_rate_pct        already a % (link0_eff_pct): KV delivered vs the
                         bottleneck's nominal rate. Absolute bytes cancel.

Block B -- NORMALISED "does the fabric effect reach the user?" quantities. Here
the raw ns WOULD carry compute scale (a 70B's TTFT and collectives dwarf a
13B's), so these are made dimensionless -- self-normalised (divided by another
quantity of the SAME run) or normalised to that model's OWN largest-buffer run:

    ar_first_over_rest   rs_ar_first_ns / rs_ar_rest_mean_ns: the first
                         (skew-gated) all-reduce measured in units of THIS
                         model's own steady-state all-reduce. ~1 means the skew
                         added nothing; >1 is the skew stall.
    ttft_slowdown        ttft_ns / ttft_ns at THIS model's largest buffer: how
                         much the buffer moves TTFT, relative to the most-relaxed
                         (largest-buffer) configuration. Flat ~1 across the sweep
                         means the buffer -> skew -> all-reduce chain does NOT
                         reach TTFT for that model.

Kept in summary.csv but no longer plotted: skew_over_ar_rest (redundant now that
pp_skew_ms is shown raw -- it was the same skew in collective-units) and
kv_gate_over_ttft (decode-start timing, orthogonal to the buffer chain).

Discovery
---------
    <ROOT>/output/ns3/<workload>/<sweep>/<tag>/{fct,pfc,qlen}.txt

Every sub-directory of output/ns3 that contains a `<sweep>` directory is a
model; nothing is hard-coded. Run with --list to see what would be picked up
without analysing anything.

Bottleneck consistency is checked WITHIN each model's sweep (as buffer_sweep
does), never ACROSS models: different topologies number their switches
differently, so one 'sw->peer' string cannot be required to match across
models. --bottleneck is therefore not exposed here; pass it to
buffer_sweep.py directly if one model's auto-detected bottleneck needs
overriding.

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

import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt

from utils import paths, roles
from utils.plots import logx_pow2, save_fig
from utils.roles import Placement
from buffer_sweep import analyse_sweep, Abort, need

NAN = float("nan")


def discover_workloads(root: Path, sweep: str) -> list[str]:
    ns3_root = root / "output" / "ns3"
    if not ns3_root.is_dir():
        return []
    return sorted(p.name for p in ns3_root.iterdir()
                  if p.is_dir() and (p / sweep).is_dir())


def load_workload(root: Path, workload: str, sweep: str,
                  placement: Placement, top_links: int) -> pd.DataFrame:
    """One row per (workload, buffer) run -- mirrors buffer_sweep.main's call
    to analyse_sweep so a model is scored identically whether analysed alone or
    here, then adds only the comparable/normalised columns (see module docstring
    for why the absolute-ns columns already in `s` are not plotted across
    models)."""
    p = paths.SweepPaths(sweep=sweep, workload=workload, root=root)
    need(not p.missing_roots(),
         f"{workload}: derived root(s) do not exist:\n    "
         + "\n    ".join(p.missing_roots()))
    _, s, _ = analyse_sweep(p, placement, top_links=top_links,
                            bn_force=None, verbose=False)

    s = s.copy()
    # --- Block A: fabric-domain quantities, comparable in ABSOLUTE units ------
    # These live in the network domain (a delay, a byte count, an event count, a
    # %), not the compute domain, so they do NOT carry the model's parameter/
    # token scale the way ttft_ns or a tensor collective does -- their raw value
    # is already the physical number to compare across runs. No normalisation.
    s["pp_skew_ms"] = s["pp_skew_ns"] / 1e6          # arrival misalignment (delta)
    s["line_rate_pct"] = s.get("link0_eff_pct")      # already a %
    s["pause_frames"] = s.get("link0_pause_frames")  # PFC PAUSE event count
    qb = s.get("link0_qpeak_bytes")                  # absolute peak occupancy, MB
    qm = s.get("link0_qmean_bytes")                  # -- NOT qpeak_pct: dividing by
    s["qpeak_mb"] = qb / 2**20 if qb is not None else NAN   # the swept buffer is
    s["qmean_mb"] = qm / 2**20 if qm is not None else NAN   # circular on a buffer
    # kept in the CSV for continuity, no longer plotted:                  # sweep.
    s["pause_pct_of_window"] = s.get("link0_pause_pct_of_window")
    s["qpeak_pct"] = s.get("link0_qpeak_pct")
    s["skew_over_ar_rest"] = s["pp_skew_ns"] / s["rs_ar_rest_mean_ns"]
    # --- Block B: normalised "does it propagate to TTFT?" quantities ----------
    # ar_first_over_rest is self-normalised (first gated all-reduce in units of
    # this model's own steady-state collective); ttft_slowdown is normalised to
    # this model's largest-buffer run. These answer the payoff question, not the
    # fabric magnitude one, so they stay dimensionless.
    s["ar_first_over_rest"] = s["rs_ar_first_ns"] / s["rs_ar_rest_mean_ns"]
    # normalised to THIS model's largest-buffer (most relaxed) run.
    tt = s.dropna(subset=["ttft_ns"]).sort_values("buffer_mb")
    ref = float(tt["ttft_ns"].iloc[-1]) if len(tt) else NAN
    s["ttft_slowdown"] = (s["ttft_ns"] / ref
                          if pd.notna(ref) and ref > 0 else NAN)
    s.insert(0, "workload", workload)
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="buffer_sweep_T1",
                    help="sweep sub-directory name to look for under every "
                         "workload (default: buffer_sweep_T1)")
    ap.add_argument("--root", default=str(paths.ROOT), type=Path,
                    help=f"project root (default: {paths.ROOT})")
    ap.add_argument("--workloads", nargs="+", default=None,
                    help="glob pattern(s) to keep (default: every workload that "
                         "has the sweep), e.g. --workloads 'llama2_13b_*'")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="glob pattern(s) to drop")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links analyse_sweep scores "
                         "(only link0 is compared here; default: 6)")
    roles.add_argument(ap)
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/"
                         "buffer_compare/<sweep>)")
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
            print(f"  + {w:<55} buf={summ['buffer_mb'].min():g}.."
                  f"{summ['buffer_mb'].max():g}  "
                  f"bn={summ['bottleneck'].iloc[0]}  "
                  f"skew@min_buf="
                  f"{summ.sort_values('buffer_mb')['pp_skew_ms'].iloc[0]:.2f}ms")

        combined = pd.concat(frames, ignore_index=True)
        front = ["workload", "tag", "bottleneck", "buffer_mb",
                 "pp_skew_ms", "qpeak_mb", "qmean_mb", "pause_frames",
                 "line_rate_pct", "ar_first_over_rest", "ttft_slowdown",
                 "pause_pct_of_window", "qpeak_pct", "skew_over_ar_rest",
                 "kv_gate_over_ttft"]
        combined = combined[[c for c in front if c in combined.columns]
                            + [c for c in combined.columns if c not in front]]
        outdir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(outdir / "summary.csv", index=False)

        written: list[Path] = []

        def line_by_workload(ycol: str, ylabel: str, title: str, fname: str,
                             hline: float | None = None) -> None:
            """One line per model, x = buffer (log2). Skipped when the column is
            absent or entirely NaN, so a metric a model's runs never produced
            (e.g. PP=1 -> no all-reduce ratio) just drops out."""
            if ycol not in combined.columns or not combined[ycol].notna().any():
                return
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for w, grp in combined.groupby("workload"):
                grp = grp.dropna(subset=[ycol]).sort_values("buffer_mb")
                if grp.empty:
                    continue
                ax.plot(grp["buffer_mb"], grp[ycol], marker="o", label=w)
            if hline is not None:
                ax.axhline(hline, color="k", linestyle=":", alpha=0.4)
            logx_pow2(ax, combined, "buffer_mb", "Per-switch buffer (MiB)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}\n(sweep: {a.sweep})")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8)
            save_fig(fig, outdir, fname, written)

        # === Block A: fabric-domain magnitudes, absolute & comparable ========
        line_by_workload(
            "pp_skew_ms", "PP arrival skew (worst wave, ms)",
            "PP arrival misalignment vs buffer, across models\n"
            "(cross-rank arrival delta on the receiving stage -- a network-domain "
            "delay, so the raw ms is directly comparable across models)",
            "pp_skew_ms_by_workload.png")

        line_by_workload(
            "qpeak_mb", "Peak egress-queue occupancy (MB)",
            "How deep does the bottleneck queue get vs buffer, across models\n"
            "(absolute bytes at the bottleneck port -- unlike % of buffer, this is "
            "not circular on a buffer sweep)",
            "qpeak_occupancy_mb_by_workload.png")

        line_by_workload(
            "pause_frames", "PFC PAUSE frames (count)",
            "Backpressure events vs buffer, across models\n"
            "(number of PAUSE frames at the bottleneck link; note this is a raw "
            "count, so it also grows with run duration)",
            "pause_frames_by_workload.png")

        line_by_workload(
            "line_rate_pct", "Effective KV bandwidth (% of line rate)",
            "KV delivered vs nominal bottleneck rate, across models\n"
            "(link0_eff_pct -- independent of how much KV a model moves)",
            "line_rate_efficiency_by_workload.png")

        # === Block B: does the fabric effect propagate to the user? ==========
        line_by_workload(
            "ar_first_over_rest",
            "First gated all-reduce / steady-state all-reduce (x)",
            "How much does the skew stall the FIRST all-reduce?, across models\n"
            "(the first collective in units of that model's own steady-state one; "
            "1 = skew added nothing, >1 = the skew-induced stall)",
            "allreduce_first_over_rest_by_workload.png", hline=1.0)

        line_by_workload(
            "ttft_slowdown", "TTFT / TTFT at largest buffer",
            "Does the buffer chain reach TTFT?, across models\n"
            "(TTFT normalised to each model's own largest-buffer run; flat ~1 "
            "means buffer -> skew -> all-reduce does NOT move TTFT for that model)",
            "ttft_slowdown_by_workload.png", hline=1.0)

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
