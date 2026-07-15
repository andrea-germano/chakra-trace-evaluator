#!/usr/bin/env python3
"""
bandwidth_analyzer.py
=====================

Summarise an ASTRA-sim bandwidth sweep for MLSynth disaggregated-inference traces.

The sweep root contains one sub-directory per simulated run, e.g.

    T2_bx25_dcqcn_buf32/
        stats_sys0.csv
        stats_sys1.csv
        ...

The number after ``bx`` in the directory name is the simulated per-link bandwidth.
Every CSV inside a run directory is one node's stats trace, in the format:

    sys_id,node_id,name,type,comm_size,start_tick,end_tick,duration,
    bw_bytes_per_ns,operation_intensity,compute_utilization,
    memory_utilization,is_memory_bound

The ``name`` column encodes the operation class through the MLSynth naming scheme
(Utils/naming.py):

    COMP_...      compute (GPU)
    TP_...        tensor-parallel all-reduce
    KV_...        prefill -> decode KV-cache transfer
    PP_...        pipeline-parallel activation transfer
    FIRSTTOK_...  first-token handoff (prefill -> decode)
    DECFB_...     decode autoregressive feedback

``pl=p`` / ``pl=d`` inside the name marks the prefill vs decode phase.
Ticks are nanoseconds, so ``bw_bytes_per_ns`` is effectively GB/s (decimal).

Focus of the analysis:
  * KV-cache transfer scaling with bandwidth
  * Pipeline-parallel (PP) transfer scaling with bandwidth
  * Overall makespan and which phase gates it
Tensor parallelism is NOT compared across runs; TP only appears as background
context in the busy-time breakdown.

This analysis lives entirely in the ASTRA-sim CSVs: more bandwidth -> shorter
transfers, monotone, visible in the logical ticks alone. That is what separates it
from buffer_analyzer.py, which has to cross into the ns-3 outputs because the
buffer changes the congestion *regime* and not the drain rate. Different questions,
different tools; they share readers, not conclusions -- see utils/__init__.py.

Outputs (written to <root>/bandwidth_analysis/ by default):
  * summary.csv            one row per run with all aggregated metrics
  * per_node.csv           per-node / per-class breakdown (long format)
  * a set of PNG graphs

Usage
-----
    python3 analyze_bandwidth_sweep.py [ROOT] [-o OUTDIR] [--pattern '*.csv']

If ROOT is omitted the hard-coded sweep path above is used.

Counting each transfer once
---------------------------
Every transfer appears TWICE in the concatenated CSVs: a COMM_SEND row in the
sender's file and a COMM_RECV row in the receiver's. Worse, ASTRA-sim posts
dependency-free RECV nodes eagerly at tick 0, so a recv row's start_tick is 0 and
its duration is the whole wait rather than the transfer. Keeping recv rows
therefore corrupted three things at once:

    bytes    doubled (kv_total_GB read 13.42 GB on the real buffer sweep, where
             the true KV volume is 6.71 GB);
    window   max(end) - min(start) collapsed to "the whole run", because
             min(start_tick) was 0;
    duration kv_mean_duration_ns averaged real send times with wait times.

The window error dominates, and it does not merely shift kv_agg_bw -- it FLATTENS
it. On a synthetic sweep with a known answer (aggregate must equal the nominal bx)
the old metric read 12.2 / 13.9 / 15.0 / 15.5 / 15.9 for bx = 25..400: a 1.3x
spread over a 16x range of nominal bandwidth, i.e. "the fabric does not scale".
The size of the error is workload-dependent (it is the ratio of prefill compute to
transfer time), so it cannot be corrected after the fact.

``astra.sends()`` keeps the send side of every point-to-point class and
``astra.collapse_collectives()`` keeps one representative per TP all-reduce;
``comm_role`` is a grouping
key rather than a filter in per_node.csv, so a receiving rank still shows what it
received without the table double-counting when summed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt

from utils import astra
from utils.plots import plot_series, save_fig
from utils.sweep import (BANDWIDTH_AXIS, discover_runs, order_columns,
                            resolve_outdir, write_table)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_ROOT = (
    "/home/andre/tesi/trace_evaluator/output/astra_logs/"
    "llama2_13b_p-tp2pp2_d-tp2pp2_stream_64reqs_512prompt/bandwidth_sweep"
)

# Where results (summary.csv, per_node.csv, the PNG graphs) are written by
# default. Edit this constant to change the default output location.
# Set it to None to fall back to "<root>/bandwidth_analysis" instead.
DEFAULT_OUTDIR = ("/home/andre/tesi/trace_evaluator/sweep_results/bandwidth_analysis/llama2_13b_p-tp2pp2_d-tp2pp2_stream_64reqs_512prompt")

def _phase_of(pl) -> str:
    return {"p": "prefill", "d": "decode"}.get(pl, "other")


def combo(row) -> str:
    """Combined label for the busy-time breakdown, e.g. PP_prefill, COMP_decode,
    KV, FIRSTTOK, DECFB. KV carries no pool, so it stays a class of its own."""
    if row["op_class"] in {"COMP", "TP", "PP"} and row["phase"] in {"prefill", "decode"}:
        return f"{row['op_class']}_{row['phase']}"
    return row["op_class"]


def load_run(run_dir: Path, pattern: str) -> pd.DataFrame | None:
    """Load and concatenate every CSV in a single run directory, then add the
    combined ``group`` label used by the busy-time breakdown.

    Reading and op-class/phase tagging come from utils.astra so that both
    analyzers classify a node name identically; ``group`` stays here because only
    this sweep plots a per-class breakdown."""
    df = astra.read_run(run_dir, pattern)
    if df is None:
        return None

    df["group"] = df.apply(combo, axis=1)
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _win_bw(sub: pd.DataFrame) -> tuple[float, float, float]:
    """(total_bytes, window_ns, aggregate_bytes_per_ns) for a subset of COMM nodes.
    `sub` must already be de-duplicated by astra.sends(): the window is
    max(end) - min(start), which only means "the transfer window" once the eager
    recv rows are gone."""
    if sub.empty:
        return (np.nan, np.nan, np.nan)
    total_bytes = sub["comm_size"].sum(min_count=1)
    window = sub["end_tick"].max() - sub["start_tick"].min()
    agg = total_bytes / window if window and window > 0 else np.nan
    return (total_bytes, window, agg)


def summarise_run(df: pd.DataFrame) -> dict:
    """Collapse one run's concatenated trace into a flat dict of metrics."""
    out: dict[str, float] = {}

    out["n_nodes"] = int(df["sys_id"].nunique()) if "sys_id" in df else np.nan
    out["n_rows"] = len(df)

    global_start = df["start_tick"].min()
    global_end = df["end_tick"].max()
    makespan = global_end - global_start
    out["makespan_ns"] = makespan
    out["makespan_ms"] = makespan / 1e6

    # ---- phase completion times (when did each class finish?) -------------- #
    def last_end(mask) -> float:
        s = df.loc[mask, "end_tick"]
        return float(s.max()) if len(s) else np.nan

    comp_mask = df["op_class"] == "COMP"
    kv_mask = df["op_class"] == "KV"
    pp_mask = df["op_class"] == "PP"
    pp_pre_mask = pp_mask & (df["phase"] == "prefill")
    pp_dec_mask = pp_mask & (df["phase"] == "decode")

    out["comp_completion_ns"] = last_end(comp_mask)
    out["kv_completion_ns"] = last_end(kv_mask)
    out["pp_completion_ns"] = last_end(pp_mask)
    out["prefill_comp_completion_ns"] = last_end(comp_mask & (df["phase"] == "prefill"))
    out["decode_comp_completion_ns"] = last_end(comp_mask & (df["phase"] == "decode"))

    # gating indicator: fraction of makespan at which KV finishes
    out["kv_bound_ratio"] = (
        out["kv_completion_ns"] / makespan if makespan and makespan > 0 else np.nan
    )

    # ---- KV-cache transfer ------------------------------------------------- #
    kv = astra.sends(df, kv_mask)
    out["kv_count"] = len(kv)                          # transfers, not CSV rows
    kv_bytes, kv_window, kv_agg = _win_bw(kv)
    out["kv_total_bytes"] = kv_bytes
    out["kv_total_GB"] = kv_bytes / 1e9 if pd.notna(kv_bytes) else np.nan
    out["kv_window_ns"] = kv_window
    out["kv_agg_bw_bytes_per_ns"] = kv_agg            # delivered aggregate throughput
    out["kv_mean_link_bw"] = kv["bw_bytes_per_ns"].mean() if len(kv) else np.nan
    out["kv_mean_duration_ns"] = kv["duration"].mean() if len(kv) else np.nan
    out["kv_max_duration_ns"] = kv["duration"].max() if len(kv) else np.nan
    out["kv_total_busy_ns"] = kv["duration"].sum(min_count=1) if len(kv) else np.nan
    # Wall-clock time during which at least one KV send was in flight. Unlike the
    # window it excludes the gaps, and unlike the busy sum it does not count
    # concurrent transfers twice.
    out["kv_busy_union_ns"] = (astra.interval_union(kv["start_tick"], kv["end_tick"])
                               if len(kv) else np.nan)

    # ---- Pipeline-parallel transfer (prefill + decode separately) ---------- #
    for label, mask in (("pp", pp_mask), ("pp_prefill", pp_pre_mask),
                        ("pp_decode", pp_dec_mask)):
        sub = astra.sends(df, mask)
        b, w, agg = _win_bw(sub)
        out[f"{label}_count"] = len(sub)
        out[f"{label}_total_bytes"] = b
        out[f"{label}_agg_bw_bytes_per_ns"] = agg
        out[f"{label}_mean_link_bw"] = sub["bw_bytes_per_ns"].mean() if len(sub) else np.nan
        out[f"{label}_mean_duration_ns"] = sub["duration"].mean() if len(sub) else np.nan
        out[f"{label}_total_busy_ns"] = sub["duration"].sum(min_count=1) if len(sub) else np.nan

    # ---- compute (context) ------------------------------------------------- #
    comp = df[comp_mask]
    out["comp_total_busy_ns"] = comp["duration"].sum(min_count=1) if len(comp) else np.nan
    out["comp_mean_util"] = comp["compute_utilization"].mean() if len(comp) else np.nan

    # ---- TP (context only, not compared) ----------------------------------- #
    # A TP all-reduce is ONE logical collective spread over the tp ranks: one row
    # each, identical apart from the shard id. Summing the rows multiplies it by tp.
    tp = astra.collapse_collectives(df, df["op_class"] == "TP")
    out["tp_total_busy_ns"] = tp["duration"].sum(min_count=1) if len(tp) else np.nan
    out["tp_mean_link_bw"] = tp["bw_bytes_per_ns"].mean() if len(tp) else np.nan

    # ---- summed busy time per group (for the breakdown chart) -------------- #
    # recv rows excluded: their duration is the wait from tick 0, not work done.
    # One row per logical transfer per class: send side for point-to-point, one
    # representative per collective. Compute rows pass through untouched.
    parts = [df[df["is_compute"]]]
    for cls in sorted(set(df.loc[df["is_comm"], "op_class"])):
        u = astra.unique_transfers(df, cls)
        if "group" not in u.columns:
            # collapse_collectives aggregates away the label, but it groups BY pl,
            # so the same combo rule can be re-applied from the surviving key.
            u = u.assign(group=[combo({"op_class": cls, "phase": _phase_of(pl)})
                                for pl in u.get("pl", [None] * len(u))])
        parts.append(u)
    busy = (pd.concat(parts, ignore_index=True)
            .groupby("group")["duration"].sum(min_count=1))
    for g, v in busy.items():
        out[f"busy__{g}_ns"] = v

    return out


def build_per_node(df: pd.DataFrame, bandwidth: float, variant: str) -> pd.DataFrame:
    """Long-format per-node / per-class busy time & bytes, for auditing.

    `comm_role` is a grouping key rather than a filter, so a receiving rank still
    shows what it received -- but summing the table no longer double-counts, as
    long as you filter to comm_role == 'send' first."""
    g = (
        df.groupby(["sys_id", "group", "comm_role"])
        .agg(
            count=("name", "size"),
            total_bytes=("comm_size", "sum"),
            total_busy_ns=("duration", "sum"),
            mean_bw_bytes_per_ns=("bw_bytes_per_ns", "mean"),
        )
        .reset_index()
    )
    g.insert(0, "bandwidth", bandwidth)
    g.insert(1, "variant", variant)
    return g


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_plots(summary: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    s = summary.sort_values("bandwidth")
    multi = s["variant"].nunique() > 1

    def save(fig, name):
        save_fig(fig, outdir, name, written)

    # 1) Makespan vs bandwidth ---------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_series(ax, s, "bandwidth", "makespan_ms", "Makespan")
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("Makespan (ms)")
    ax.set_title("Total simulated execution time vs bandwidth")
    ax.grid(True, alpha=0.3)
    if multi or True:
        ax.legend()
    save(fig, "01_makespan_vs_bandwidth.png")

    # 2) Phase completion crossover ----------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, lbl, mk in (
        ("makespan_ns", "Makespan (overall)", "o"),
        ("kv_completion_ns", "KV transfer completion", "s"),
        ("comp_completion_ns", "Compute completion", "^"),
        ("pp_completion_ns", "PP transfer completion", "d"),
    ):
        if col in s and s[col].notna().any():
            plot_series(ax, s, "bandwidth", col, lbl, marker=mk, scale=1e-6)
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("Completion time (ms)")
    ax.set_title("Which phase gates the makespan?\n(KV above compute => KV-bound)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "02_phase_completion_vs_bandwidth.png")

    # 3) KV effective bandwidth vs nominal ---------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_series(ax, s, "bandwidth", "kv_agg_bw_bytes_per_ns", "KV aggregate delivered", marker="o")
    plot_series(ax, s, "bandwidth", "kv_mean_link_bw", "KV mean per-transfer", marker="s", linestyle="--")
    bmin, bmax = s["bandwidth"].min(), s["bandwidth"].max()
    ax.plot([bmin, bmax], [bmin, bmax], "k:", alpha=0.5, label="ideal (y = x)")
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("Effective bandwidth (bytes/ns \u2248 GB/s)")
    ax.set_title("KV-cache transfer: delivered vs nominal bandwidth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "03_kv_effective_bandwidth.png")

    # 4) KV duration vs bandwidth ------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_series(ax, s, "bandwidth", "kv_mean_duration_ns", "KV mean transfer time", marker="o", scale=1e-6)
    plot_series(ax, s, "bandwidth", "kv_max_duration_ns", "KV max transfer time", marker="s",
            scale=1e-6, linestyle="--")
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("KV transfer time (ms)")
    ax.set_title("KV-cache transfer time vs bandwidth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "04_kv_transfer_time.png")

    # 5) PP effective bandwidth (prefill vs decode) ------------------------- #
    if s[["pp_prefill_agg_bw_bytes_per_ns", "pp_decode_agg_bw_bytes_per_ns"]].notna().any().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_series(ax, s, "bandwidth", "pp_prefill_agg_bw_bytes_per_ns", "PP prefill (aggregate)", marker="o")
        plot_series(ax, s, "bandwidth", "pp_decode_agg_bw_bytes_per_ns", "PP decode (aggregate)", marker="s")
        plot_series(ax, s, "bandwidth", "pp_prefill_mean_link_bw", "PP prefill (per-transfer)",
                marker="^", linestyle="--")
        plot_series(ax, s, "bandwidth", "pp_decode_mean_link_bw", "PP decode (per-transfer)",
                marker="v", linestyle="--")
        bmin, bmax = s["bandwidth"].min(), s["bandwidth"].max()
        ax.plot([bmin, bmax], [bmin, bmax], "k:", alpha=0.5, label="ideal (y = x)")
        ax.set_xlabel("Simulated link bandwidth (bx)")
        ax.set_ylabel("Effective bandwidth (bytes/ns \u2248 GB/s)")
        ax.set_title("Pipeline-parallel transfer: delivered vs nominal bandwidth")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save(fig, "05_pp_effective_bandwidth.png")

    # 6) PP transfer time vs bandwidth -------------------------------------- #
    if s[["pp_prefill_mean_duration_ns", "pp_decode_mean_duration_ns"]].notna().any().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_series(ax, s, "bandwidth", "pp_prefill_mean_duration_ns", "PP prefill mean time",
                marker="o", scale=1e-6)
        plot_series(ax, s, "bandwidth", "pp_decode_mean_duration_ns", "PP decode mean time",
                marker="s", scale=1e-6)
        ax.set_xlabel("Simulated link bandwidth (bx)")
        ax.set_ylabel("PP transfer time (ms)")
        ax.set_title("Pipeline-parallel transfer time vs bandwidth")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save(fig, "06_pp_transfer_time.png")

    # 7) Busy-time breakdown (stacked) -------------------------------------- #
    busy_cols = [c for c in s.columns if c.startswith("busy__")]
    if busy_cols:
        # order so KV / PP stand out, TP & compute as context
        def sort_key(c):
            name = c[len("busy__"):-len("_ns")] if c.endswith("_ns") else c
            order = {"KV": 0, "PP_prefill": 1, "PP_decode": 2, "FIRSTTOK": 3,
                     "DECFB": 4, "COMP_prefill": 5, "COMP_decode": 6,
                     "TP_prefill": 7, "TP_decode": 8}
            return order.get(name, 99)
        busy_cols = sorted(busy_cols, key=sort_key)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = s["bandwidth"].astype(str)
        bottom = np.zeros(len(s))
        for c in busy_cols:
            vals = (s[c].fillna(0).to_numpy()) / 1e6  # ms
            if vals.sum() == 0:
                continue
            label = c[len("busy__"):-len("_ns")] if c.endswith("_ns") else c
            ax.bar(x, vals, bottom=bottom, label=label)
            bottom += vals
        ax.set_xlabel("Simulated link bandwidth (bx)")
        ax.set_ylabel("Summed busy time across all nodes (ms)")
        ax.set_title("Busy-time breakdown by operation class\n"
                     "(summed over nodes; comm classes overlap in wall-clock time)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        save(fig, "07_busy_time_breakdown.png")

    # 8) Speedup vs slowest run --------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    tmp = s.copy()
    for variant, grp in tmp.groupby("variant"):
        grp = grp.dropna(subset=["makespan_ns"]).sort_values("bandwidth")
        if grp.empty:
            continue
        base = grp["makespan_ns"].iloc[0]  # lowest bandwidth = baseline
        speedup = base / grp["makespan_ns"]
        lbl = "Speedup vs lowest bw" if tmp["variant"].nunique() == 1 else f"[{variant}]"
        ax.plot(grp["bandwidth"], speedup, marker="o", label=lbl)
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("Speedup (makespan at lowest bw / makespan)")
    ax.set_title("Makespan speedup vs bandwidth (normalised to lowest bw)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "08_speedup_vs_bandwidth.png")

    # 9) KV-bound ratio ----------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_series(ax, s, "bandwidth", "kv_bound_ratio", "KV completion / makespan", marker="o")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.6)
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("KV completion / makespan")
    ax.set_title("How tightly KV transfer gates the run\n(\u22481 => KV is the bottleneck)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "09_kv_bound_ratio.png")

    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                    help="Sweep root containing the per-bandwidth run sub-directories.")
    ap.add_argument("-o", "--out", default=None,
                    help="Output directory. If omitted, uses the DEFAULT_OUTDIR "
                         "constant (or <root>/bandwidth_analysis when that is None).")
    ap.add_argument("--pattern", default="*.csv",
                    help="Glob for the per-node CSVs inside each run dir (default '*.csv').")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        return 2

    outdir = resolve_outdir(args.out, DEFAULT_OUTDIR, root, "bandwidth_analysis")
    run_dirs = discover_runs(root, outdir, skip_names=("bandwidth_analysis",))
    if not run_dirs:
        print(f"ERROR: no run sub-directories found under {root}", file=sys.stderr)
        return 2

    print(f"Scanning {len(run_dirs)} run directories under:\n  {root}\n")

    rows: list[dict] = []
    per_node_frames: list[pd.DataFrame] = []

    for d in run_dirs:
        bw = BANDWIDTH_AXIS.value(d.name)
        if bw is None:
            print(f"  - skip {d.name!r}: no 'bx<num>' bandwidth token", file=sys.stderr)
            continue
        df = load_run(d, args.pattern)
        if df is None:
            print(f"  - skip {d.name!r}: no readable CSVs", file=sys.stderr)
            continue

        variant = BANDWIDTH_AXIS.variant(d.name)
        summ = summarise_run(df)
        summ["run_dir"] = d.name
        summ["variant"] = variant
        summ["bandwidth"] = bw
        rows.append(summ)
        per_node_frames.append(build_per_node(df, bw, variant))

        print(f"  + {d.name:<28} bw={bw:<7g} nodes={summ['n_nodes']:<3} "
              f"makespan={summ['makespan_ms']:.2f} ms  "
              f"KV/makespan={summ['kv_bound_ratio']:.2f}")

    if not rows:
        print("ERROR: no valid runs parsed.", file=sys.stderr)
        return 2

    summary = pd.DataFrame(rows).sort_values(["variant", "bandwidth"]).reset_index(drop=True)

    # nice column ordering: identifiers first
    summary = order_columns(summary, ["run_dir", "variant", "bandwidth", "n_nodes",
                                      "makespan_ms", "kv_bound_ratio"])
    summary_path = write_table(summary, outdir, "summary.csv")
    per_node = pd.concat(per_node_frames, ignore_index=True)
    per_node_path = write_table(per_node, outdir, "per_node.csv")

    plots = make_plots(summary, outdir)

    # console report
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    report_cols = ["bandwidth", "makespan_ms", "kv_bound_ratio",
                   "kv_agg_bw_bytes_per_ns", "kv_mean_duration_ns",
                   "pp_prefill_agg_bw_bytes_per_ns", "pp_decode_agg_bw_bytes_per_ns"]
    report_cols = [c for c in report_cols if c in summary.columns]
    print("\n================ SUMMARY ================")
    print(summary[report_cols].to_string(index=False))
    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {per_node_path}")
    for p in plots:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())