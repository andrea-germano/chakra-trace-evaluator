#!/usr/bin/env python3
"""
analyze_bandwidth_sweep.py
==========================

Summarise an ASTRA-sim bandwidth sweep for MLSynth disaggregated-inference traces.

The sweep root (default:
    /home/andre/tesi/trace_evaluator/output/astra_logs/
        llama2_13b_p-tp2pp2_d-tp2pp2_stream/bandwidth_topologies
) contains one sub-directory per simulated run, e.g.

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

Focus of the analysis (as requested):
  * KV-cache transfer scaling with bandwidth
  * Pipeline-parallel (PP) transfer scaling with bandwidth
  * Overall makespan and which phase gates it
Tensor parallelism is NOT compared across runs; TP only appears as background
context in the busy-time breakdown.

Outputs (written to <root>/bandwidth_analysis/ by default):
  * summary.csv            one row per run with all aggregated metrics
  * per_node.csv           per-node / per-class breakdown (long format)
  * a set of PNG graphs

Usage
-----
    python3 analyze_bandwidth_sweep.py [ROOT] [-o OUTDIR] [--pattern '*.csv']

If ROOT is omitted the hard-coded sweep path above is used.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_ROOT = (
    "/home/andre/tesi/trace_evaluator/output/astra_logs/"
    "llama2_13b_p-tp2pp2_d-tp2pp2_stream_1reqs_32768prompt/bandwidth_sweep"
)

# Where results (summary.csv, per_node.csv, the PNG graphs) are written by
# default. Edit this constant to change the default output location.
# Set it to None to fall back to "<root>/bandwidth_analysis" instead.
DEFAULT_OUTDIR = ("/home/andre/tesi/trace_evaluator/sweep_results/bandwidth_analysis/llama2_13b_p-tp2pp2_d-tp2pp2_stream_1reqs_32768prompt")

NUMERIC_COLS = [
    "comm_size", "start_tick", "end_tick", "duration", "bw_bytes_per_ns",
    "operation_intensity", "compute_utilization", "memory_utilization",
    "is_memory_bound",
]

# Directory name like "T2_bx25_dcqcn_buf32" -> bandwidth = 25
BX_RE = re.compile(r"bx(\d+(?:\.\d+)?)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_bandwidth(dirname: str) -> float | None:
    """Extract the simulated bandwidth encoded after 'bx' in the directory name."""
    m = BX_RE.search(dirname)
    return float(m.group(1)) if m else None


def variant_key(dirname: str) -> str:
    """Everything in the dir name except the bx value, so that if other knobs
    (algorithm, buffer size, ...) also vary they become separate series."""
    return BX_RE.sub("bx*", dirname)


def classify(name: str) -> tuple[str, str]:
    """Return (op_class, phase) for a node name.

    op_class in {COMP, TP, KV, PP, FIRSTTOK, DECFB, OTHER}
    phase    in {prefill, decode, kv_transfer, handoff, other}
    """
    if not isinstance(name, str) or not name:
        return "OTHER", "other"
    head = name.split("_", 1)[0].upper()
    op = head if head in {"COMP", "TP", "KV", "PP", "FIRSTTOK", "DECFB"} else "OTHER"

    if "pl=p" in name:
        phase = "prefill"
    elif "pl=d" in name:
        phase = "decode"
    elif op == "KV":
        phase = "kv_transfer"
    elif op == "FIRSTTOK":
        phase = "handoff"
    elif op == "DECFB":
        phase = "decode"
    else:
        phase = "other"
    return op, phase


def load_run(run_dir: Path, pattern: str) -> pd.DataFrame | None:
    """Load and concatenate every CSV in a single run directory."""
    frames = []
    for csv in sorted(run_dir.glob(pattern)):
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read {csv.name}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        df["__file__"] = csv.name
        frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cls_phase = df["name"].map(classify)
    df["op_class"] = cls_phase.map(lambda t: t[0])
    df["phase"] = cls_phase.map(lambda t: t[1])
    # combined label, e.g. PP_prefill, COMP_decode, KV, FIRSTTOK, DECFB
    def combo(row):
        if row["op_class"] in {"COMP", "TP", "PP"} and row["phase"] in {"prefill", "decode"}:
            return f"{row['op_class']}_{row['phase']}"
        return row["op_class"]
    df["group"] = df.apply(combo, axis=1)
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _win_bw(sub: pd.DataFrame) -> tuple[float, float, float]:
    """(total_bytes, window_ns, aggregate_bytes_per_ns) for a subset of COMM nodes."""
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
    kv = df[kv_mask]
    out["kv_count"] = len(kv)
    kv_bytes, kv_window, kv_agg = _win_bw(kv)
    out["kv_total_bytes"] = kv_bytes
    out["kv_total_GB"] = kv_bytes / 1e9 if pd.notna(kv_bytes) else np.nan
    out["kv_window_ns"] = kv_window
    out["kv_agg_bw_bytes_per_ns"] = kv_agg            # delivered aggregate throughput
    out["kv_mean_link_bw"] = kv["bw_bytes_per_ns"].mean() if len(kv) else np.nan
    out["kv_mean_duration_ns"] = kv["duration"].mean() if len(kv) else np.nan
    out["kv_max_duration_ns"] = kv["duration"].max() if len(kv) else np.nan
    out["kv_total_busy_ns"] = kv["duration"].sum(min_count=1) if len(kv) else np.nan

    # ---- Pipeline-parallel transfer (prefill + decode separately) ---------- #
    for label, mask in (("pp", pp_mask), ("pp_prefill", pp_pre_mask),
                        ("pp_decode", pp_dec_mask)):
        sub = df[mask]
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
    tp = df[df["op_class"] == "TP"]
    out["tp_total_busy_ns"] = tp["duration"].sum(min_count=1) if len(tp) else np.nan
    out["tp_mean_link_bw"] = tp["bw_bytes_per_ns"].mean() if len(tp) else np.nan

    # ---- summed busy time per group (for the breakdown chart) -------------- #
    busy = df.groupby("group")["duration"].sum(min_count=1)
    for g, v in busy.items():
        out[f"busy__{g}_ns"] = v

    return out


def build_per_node(df: pd.DataFrame, bandwidth: float, variant: str) -> pd.DataFrame:
    """Long-format per-node / per-class busy time & bytes, for auditing."""
    g = (
        df.groupby(["sys_id", "group"])
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
def _series(ax, data: pd.DataFrame, ycol: str, label: str,
            marker="o", scale=1.0, linestyle="-"):
    """Plot ycol vs bandwidth for each variant, dropping NaNs."""
    for variant, grp in data.groupby("variant"):
        grp = grp.dropna(subset=[ycol]).sort_values("bandwidth")
        if grp.empty:
            continue
        lbl = label if data["variant"].nunique() == 1 else f"{label} [{variant}]"
        ax.plot(grp["bandwidth"], grp[ycol] * scale, marker=marker,
                linestyle=linestyle, label=lbl)


def make_plots(summary: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    s = summary.sort_values("bandwidth")
    multi = s["variant"].nunique() > 1

    def save(fig, name):
        p = outdir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 1) Makespan vs bandwidth ---------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    _series(ax, s, "makespan_ms", "Makespan")
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
            _series(ax, s, col, lbl, marker=mk, scale=1e-6)
    ax.set_xlabel("Simulated link bandwidth (bx)")
    ax.set_ylabel("Completion time (ms)")
    ax.set_title("Which phase gates the makespan?\n(KV above compute => KV-bound)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "02_phase_completion_vs_bandwidth.png")

    # 3) KV effective bandwidth vs nominal ---------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    _series(ax, s, "kv_agg_bw_bytes_per_ns", "KV aggregate delivered", marker="o")
    _series(ax, s, "kv_mean_link_bw", "KV mean per-transfer", marker="s", linestyle="--")
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
    _series(ax, s, "kv_mean_duration_ns", "KV mean transfer time", marker="o", scale=1e-6)
    _series(ax, s, "kv_max_duration_ns", "KV max transfer time", marker="s",
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
        _series(ax, s, "pp_prefill_agg_bw_bytes_per_ns", "PP prefill (aggregate)", marker="o")
        _series(ax, s, "pp_decode_agg_bw_bytes_per_ns", "PP decode (aggregate)", marker="s")
        _series(ax, s, "pp_prefill_mean_link_bw", "PP prefill (per-transfer)",
                marker="^", linestyle="--")
        _series(ax, s, "pp_decode_mean_link_bw", "PP decode (per-transfer)",
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
        _series(ax, s, "pp_prefill_mean_duration_ns", "PP prefill mean time",
                marker="o", scale=1e-6)
        _series(ax, s, "pp_decode_mean_duration_ns", "PP decode mean time",
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
    _series(ax, s, "kv_bound_ratio", "KV completion / makespan", marker="o")
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

    # Precedence: --out flag > DEFAULT_OUTDIR constant > <root>/bandwidth_analysis
    if args.out:
        outdir = Path(args.out)
    elif DEFAULT_OUTDIR:
        outdir = Path(DEFAULT_OUTDIR)
    else:
        outdir = root / "bandwidth_analysis"

    outdir_resolved = outdir.resolve()
    run_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and p.name != "bandwidth_analysis"
        and p.resolve() != outdir_resolved
    )
    if not run_dirs:
        print(f"ERROR: no run sub-directories found under {root}", file=sys.stderr)
        return 2

    print(f"Scanning {len(run_dirs)} run directories under:\n  {root}\n")

    rows: list[dict] = []
    per_node_frames: list[pd.DataFrame] = []

    for d in run_dirs:
        bw = parse_bandwidth(d.name)
        if bw is None:
            print(f"  - skip {d.name!r}: no 'bx<num>' bandwidth token", file=sys.stderr)
            continue
        df = load_run(d, args.pattern)
        if df is None:
            print(f"  - skip {d.name!r}: no readable CSVs", file=sys.stderr)
            continue

        variant = variant_key(d.name)
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
    front = ["run_dir", "variant", "bandwidth", "n_nodes", "makespan_ms",
             "kv_bound_ratio"]
    cols = front + [c for c in summary.columns if c not in front]
    summary = summary[cols]

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    per_node = pd.concat(per_node_frames, ignore_index=True)
    per_node_path = outdir / "per_node.csv"
    per_node.to_csv(per_node_path, index=False)

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