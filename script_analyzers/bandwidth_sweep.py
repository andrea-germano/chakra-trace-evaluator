#!/usr/bin/env python3
"""
bandwidth_sweep — how does the run scale with link bandwidth?

Paths come from utils.paths: `--sweep bandwidth_sweep` and everything else is
derived. The number after ``bx`` in each run-dir name is the simulated per-link
bandwidth (utils.paths.BANDWIDTH_AXIS).
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

Outputs (results/sweep_analysis/bandwidth/<sweep>/<workload>/ by default):
  * summary.csv            one row per run with all aggregated metrics
  * per_node.csv           per-node / per-class breakdown (long format)
  * six PNG graphs

Nine used to be six + three derivations of the other six, which is a way of
showing the same number three times and calling it corroboration:

    01_makespan            a subset of 02, which already draws makespan_ns
    08_speedup             makespan[0] / makespan -- 01, renormalised
    09_kv_bound_ratio      kv_completion / makespan -- two curves of 02, divided

and the metric behind the last one did not measure its own name. `makespan` is
max(end) over ALL rows, and under disaggregation decode runs AFTER the KV
transfer, so kv_completion/makespan measures what fraction of the run had
elapsed when KV finished -- which is set by how many decode steps you simulate.
Double the tokens and KV becomes "less of a bottleneck". The comparison that
answers the question is kv_completion vs prefill_comp_completion: does decode
wait on the fabric, or on the prefill it is fed by? Both columns already
existed; `kv_over_prefill_compute` is their ratio and 02 draws both curves.

Usage
-----
    python3 bandwidth_sweep.py --sweep bandwidth_sweep
    python3 bandwidth_sweep.py --sweep bandwidth_sweep --workload other_model -o /tmp/x

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

from utils import astra, paths
from utils.paths import BANDWIDTH_AXIS
from utils.plots import plot_series, save_fig

KIND = "bandwidth"


class Abort(Exception):
    pass


def need(cond, msg: str) -> None:
    if not cond:
        raise Abort(msg)


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

    # Does decode wait on the fabric, or on the prefill that feeds it? >1 means
    # the KV transfer outlasts prefill compute, i.e. the fabric gates the handover.
    # NOT kv_completion/makespan: makespan includes every decode step, so that
    # ratio shrinks when you simulate more tokens and says nothing about the link.
    pc = out["prefill_comp_completion_ns"]
    out["kv_over_prefill_compute"] = (
        out["kv_completion_ns"] / pc if pd.notna(pc) and pc > 0 else np.nan)

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

    def save(fig, name):
        save_fig(fig, outdir, name, written)

    # 2) Phase completion crossover ----------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, lbl, mk in (
        ("makespan_ns", "Makespan (overall)", "o"),
        ("kv_completion_ns", "KV transfer completion", "s"),
        ("comp_completion_ns", "Compute completion (all)", "^"),
        ("prefill_comp_completion_ns", "Prefill compute completion", "v"),
        ("pp_completion_ns", "PP transfer completion", "d"),
    ):
        if col in s and s[col].notna().any():
            plot_series(ax, s, "bandwidth", col, lbl, marker=mk, scale=1e-6)
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("Completion time (ms)")
    ax.set_title("What gates the handover to decode?\n"
                 "(KV completion above prefill compute => the fabric gates it)")
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

    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    ap.add_argument("--pattern", default="*.csv",
                    help="glob for the per-node CSVs inside each run dir")
    a = ap.parse_args(argv)

    try:
        p, outdir = paths.from_arguments(a, KIND)
        need(p.astra_root.is_dir(),
             f"derived ASTRA root does not exist:\n    {p.astra_root}\n"
             f"  --sweep {a.sweep!r} or --workload {a.workload!r} is wrong.")
        tags = p.tags("astra")
        need(tags, f"no run sub-directory under {p.astra_root}")
        print(p.describe())
        print(f"  out      {outdir}\n\nScanning {len(tags)} runs:")

        rows, per_node_frames = [], []
        for tag in tags:
            bw = BANDWIDTH_AXIS.value(tag)
            need(bw is not None,
                 f"{tag}: no 'bx<num>' token in the directory name; the swept "
                 f"axis is unreadable.")
            df = load_run(p.astra_run(tag), a.pattern)
            need(df is not None,
                 f"{tag}: no readable {a.pattern} under {p.astra_run(tag)}")
            summ = summarise_run(df)
            summ.update(run_dir=tag, variant=BANDWIDTH_AXIS.variant(tag), bandwidth=bw)
            rows.append(summ)
            per_node_frames.append(build_per_node(df, bw, summ["variant"]))
            print(f"  + {tag:<28} bw={bw:<7g} nodes={summ['n_nodes']:<3} "
                  f"makespan={summ['makespan_ms']:.2f} ms  "
                  f"KV/prefill_comp={summ['kv_over_prefill_compute']:.2f}")

        summary = pd.DataFrame(rows).sort_values("bandwidth").reset_index(drop=True)
        # Every figure draws one line through summary. That is only a line if one
        # knob moves: with two, plot_series joins points from different fabrics in
        # bandwidth order and draws a zigzag that looks like a measurement. Only
        # the old speedup plot grouped by variant -- the other eight did not, and
        # said nothing.
        need(summary["variant"].nunique() == 1,
             f"this sweep moves more than one knob: variants "
             f"{sorted(summary['variant'].unique())}. Every figure here draws a "
             f"single line through summary.csv, so the runs would be joined across "
             f"fabrics in bandwidth order. Split the runs into one sweep per variant.")

        front = ["run_dir", "variant", "bandwidth", "n_nodes", "makespan_ms",
                 "kv_over_prefill_compute"]
        summary = summary[[c for c in front if c in summary.columns]
                          + [c for c in summary.columns if c not in front]]
        outdir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(outdir / "summary.csv", index=False)
        pd.concat(per_node_frames, ignore_index=True).to_csv(
            outdir / "per_node.csv", index=False)
        plots = make_plots(summary, outdir)

        pd.set_option("display.width", 180)
        report = [c for c in ["bandwidth", "makespan_ms", "kv_over_prefill_compute",
                              "kv_agg_bw_bytes_per_ns", "kv_mean_duration_ns",
                              "pp_prefill_agg_bw_bytes_per_ns",
                              "pp_decode_agg_bw_bytes_per_ns"] if c in summary.columns]
        print("\n================ BANDWIDTH SWEEP ================")
        print(summary[report].to_string(index=False))
        print(f"\nWrote {outdir}:")
        for f in ["summary.csv", "per_node.csv", *[q.name for q in plots]]:
            print(f"  {f}")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())