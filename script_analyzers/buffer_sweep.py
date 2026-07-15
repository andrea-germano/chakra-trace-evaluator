#!/usr/bin/env python3
"""
buffer_analyzer — per-switch buffer sweep for MLSynth disaggregated-inference traces.

The question
--------------------------------------------------------------------------------
Does the size of the switch buffer change when disaggregated decode can start?

It cannot be answered from the ASTRA-sim CSVs alone. At steady state the congested
link drains at line rate whatever the buffer is, so mean KV completion is flat and
the CSVs say "nothing happens". What the buffer actually changes is the
*congestion-control regime* -- whether the queue is held by PFC backpressure or by
DCQCN rate control -- and that lives in the ns-3 outputs. So this analyzer works on
two levels:

    predicted   from physical_topology.txt + config.txt ALONE, ns3_fabric computes
                where the regime flips (a band in MiB). No simulation involved.
    observed    from fct.txt / pfc.txt / qlen.txt, where the flip actually was.

The two agreeing is the result. That is the co-design claim: the tool tells you what
the fabric will do before you build it.

How to read the output
--------------------------------------------------------------------------------
The decisive regime test is `qlen_peak_over_pfc_ceiling`. The PFC ceiling is the
egress occupancy once every ingress port is paused and its headroom has absorbed
the in-flight packets. At the ceiling (~1.0) the queue is held by backpressure;
below it, rate control is what limits. Everything else corroborates.

`slow_mean_incast` is ~N, not ~1. standalone_fct assumes a flow owns its
bottleneck, so N flows sharing it fairly give slowdown N. The meaningful number is
`slow_mean_over_fairshare`: 1.0 means the fabric is as fair as it can be.

Flow taxonomy, used consistently:
    bulk     size >= --bulk-mb                (a KV/PP transfer, not an ACK)
    fabric   traverses at least one switch    (hops > 1)
    direct   host-to-host link, 1 hop         (TP collectives; never congested,
                                               slowdown ~1 by construction)
    incast   bulk AND fabric                  -> the population every statistic
                                                 named *_incast is computed over
Mixing direct flows into the incast statistics turns them into a bimodal mixture:
mean and CV then describe the mixing ratio, which is a constant of the workload,
and look flat vs buffer for the wrong reason. Hence the filter, hence the need for
--topology.

Layout
--------------------------------------------------------------------------------
    <astra_root>/<tag>/   stats_sys*.csv        tag = e.g. T1_bx100_dcqcn_buf8
    <ns3_root>/<tag>/     fct.txt pfc.txt qlen.txt
Matching is by tag; `buf<N>` is the swept axis and the rest is a `variant` key, so a
second moving knob becomes its own series.

--topology takes a path, a {tag} template, or a directory. One file is fine if the
topology is constant across the sweep (it is cached and reused).
--config MUST resolve per tag: BUFFER_SIZE is the swept axis, and a single
config.txt collapses every run onto one buffer value.

Usage
--------------------------------------------------------------------------------
    python3 buffer_analyzer.py [ASTRA_ROOT] [--ns3-root R] [-o OUT]
                               [--topology PATH|DIR|TEMPLATE] [--config ...]
                               [--bulk-mb 1] [--decode-nodes 4,5,6,7]

    python3 -m ns3_fabric <topology> <config>      # inspect the model on its own
    python3 buffer_analyzer.py --print-patch       # the ns-3 qIndex diff
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import astra
from utils import ns3
from utils.fabric import (Bottleneck, FabricModel, Ns3Config, Topology,
                             parse_ns3_config, parse_topology)
from utils import flows as flowlib
from utils.plots import logx_pow2, plot_series, relative_range, save_fig
from utils.sweep import (BUFFER_AXIS, discover_runs, find_aux, find_ns3_run,
                            order_columns, resolve_ns3_root, resolve_outdir,
                            write_table)

NAN = float("nan")

DEFAULT_ASTRA_ROOT = ("/home/andre/tesi/trace_evaluator/output/astra_logs/"
                      "llama2_13b_p-tp2pp2_d-tp2pp2_stream_16reqs_512prompt/buffer_sweep_T1")
DEFAULT_NS3_ROOT = "/home/andre/tesi/trace_evaluator/output/ns3/buffer_sweep_T1"
DEFAULT_OUTDIR = ("/home/andre/tesi/trace_evaluator/results/sweep_analysis/buffer/T1_v2/"
                  "llama2_13b_p-tp2pp2_d-tp2pp2_stream_16req_512prompt")
DEFAULT_TOPOLOGY = ("/home/andre/tesi/trace_evaluator/configs/astra_sim/ns3/"
                    "buffer_sweep_T1/{tag}/physical_topology.txt")
DEFAULT_CONFIG = ("/home/andre/tesi/trace_evaluator/configs/astra_sim/ns3/"
                  "buffer_sweep_T1/{tag}/config.txt")

# Min-max normalisation maps any range onto 0-1, so a 1% wiggle renders as a
# full-scale trend. Series flatter than this are refused, not normalised.
MIN_REL_RANGE = 0.05


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class AstraMetrics:
    n_ranks: float = NAN
    makespan_ms: float = NAN
    kv_completion_ns: float = NAN
    comp_completion_ns: float = NAN
    prefill_comp_completion_ns: float = NAN
    kv_bound_ratio: float = NAN
    kv_flows: float = NAN
    kv_dedup: str = ""
    kv_total_GB: float = NAN
    kv_busy_union_ns: float = NAN
    kv_agg_bw_bytes_per_ns: float = NAN


@dataclass
class FabricMetrics:
    congested_link: str = ""
    congested_rate_gbps: float = NAN
    fanin_ports: float = NAN
    ingress_ports: str = ""
    pfc_thresh_bytes: float = NAN
    pfc_thresh_egress_equiv_bytes: float = NAN
    pfc_thresh_naive_bytes: float = NAN
    pfc_egress_ceiling_bytes: float = NAN
    pfc_shift: float = NAN
    hdrm_pct_of_buffer: float = NAN
    kmin_bytes: float = NAN
    kmax_bytes: float = NAN
    regime_pred: str = "?"


@dataclass
class PfcMetrics:
    pfc_events: float = NAN
    pfc_qidx: str = "n/a"
    pfc_unclosed_pauses: float = NAN
    pfc_bottleneck_pause_pct: float = NAN
    pfc_worst_link_pause_pct: float = NAN
    pfc_worst_link_device: str = ""
    pfc_host_pause_max_pct: float = NAN
    pfc_sw_pause_max_pct: float = NAN
    pfc_paused_devices: float = NAN


@dataclass
class QlenMetrics:
    qlen_samples: float = NAN
    qlen_peak_congested_port_bytes: float = NAN
    qlen_mean_congested_port_bytes: float = NAN
    qlen_peak_over_pfc_ceiling: float = NAN     # the decisive regime test
    qlen_peak_over_kmax: float = NAN
    qlen_peak_over_kmin: float = NAN
    qlen_peak_switch_total_bytes: float = NAN


@dataclass
class FlowMetrics:
    fct_flows: float = NAN
    fct_incast_flows: float = NAN
    fct_direct_flows: float = NAN
    bottleneck_flows: float = NAN
    n_concurrent_peak: float = NAN
    n_concurrent_mean: float = NAN
    slow_mean_incast: float = NAN
    slow_p50_incast: float = NAN
    slow_p99_incast: float = NAN
    slow_max_incast: float = NAN
    slow_cv_incast: float = NAN
    slow_mean_over_fairshare: float = NAN
    slow_p99_over_fairshare: float = NAN
    slow_mean_direct: float = NAN
    incast_window_ns: float = NAN
    run_end_ns: float = NAN


@dataclass
class BarrierMetrics:
    kv_ready_max_ns: float = NAN
    kv_ready_min_ns: float = NAN
    sync_skew_ns: float = NAN
    cross_rank_skew_ns: float = NAN
    decode_ranks_seen: float = NAN
    incast_dst: str = ""


@dataclass
class RunRow:
    run_dir: str = ""
    variant: str = ""
    buffer_mb: float = NAN
    regime_obs: str = "?"
    cc_mode: float = NAN
    ns3_dir: str = ""
    astra: AstraMetrics = field(default_factory=AstraMetrics)
    fabric: FabricMetrics = field(default_factory=FabricMetrics)
    pfc: PfcMetrics = field(default_factory=PfcMetrics)
    qlen: QlenMetrics = field(default_factory=QlenMetrics)
    flows: FlowMetrics = field(default_factory=FlowMetrics)
    barrier: BarrierMetrics = field(default_factory=BarrierMetrics)
    per_node: object = None          # audit frame; excluded from flat()

    def flat(self) -> dict:
        """One CSV row. Field names are declared once, in the dataclasses, so a
        metric can no longer be silently invented or forgotten by a dict literal."""
        out = {}
        for f in fields(self):
            if f.name == "per_node":
                continue
            v = getattr(self, f.name)
            out.update(asdict(v)) if hasattr(v, "__dataclass_fields__") else out.update({f.name: v})
        return out


def isnum(x) -> bool:
    return isinstance(x, (int, float)) and x == x


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Flow annotation and the bottleneck
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Metric computation, one group per function
# --------------------------------------------------------------------------- #
def astra_metrics(df: pd.DataFrame) -> AstraMetrics:
    m = AstraMetrics()
    m.n_ranks = int(df["sys_id"].nunique()) if "sys_id" in df else NAN
    span = df["end_tick"].max() - df["start_tick"].min()
    m.makespan_ms = span / 1e6

    def last_end(mask) -> float:
        s = df.loc[mask, "end_tick"]
        return float(s.max()) if len(s) else NAN

    kv_mask, comp_mask = df["op_class"] == "KV", df["op_class"] == "COMP"
    m.kv_completion_ns = last_end(kv_mask)
    m.comp_completion_ns = last_end(comp_mask)
    m.prefill_comp_completion_ns = last_end(comp_mask & (df["phase"] == "prefill"))
    m.kv_bound_ratio = m.kv_completion_ns / span if span > 0 else NAN

    kv = df[kv_mask]
    if not len(kv):
        return m
    # One row per transfer: the send side. A send and its recv share a node name
    # and land in two different sys files, so the raw rows are exactly double.
    n_raw = len(kv)
    kv = astra.sends(df, kv_mask)
    m.kv_dedup = f"send-only ({n_raw} rows -> {len(kv)} transfers)"
    m.kv_flows = len(kv)
    total = kv["comm_size"].sum(min_count=1)
    window = kv["end_tick"].max() - kv["start_tick"].min()
    m.kv_total_GB = total / 1e9 if pd.notna(total) else NAN
    m.kv_busy_union_ns = astra.interval_union(kv["start_tick"], kv["end_tick"])
    m.kv_agg_bw_bytes_per_ns = total / window if window > 0 else NAN
    return m


def fabric_metrics(model: FabricModel | None, bn: Bottleneck | None,
                   buffer_bytes: int) -> FabricMetrics:
    m = FabricMetrics(pfc_thresh_naive_bytes=buffer_bytes / 8.0)
    if model is None or bn is None:
        return m
    topo = model.topo
    m.congested_link = str(bn)
    m.congested_rate_gbps = bn.rate / 1e9
    m.fanin_ports = bn.f_ports
    m.ingress_ports = ",".join(str(p) for p in bn.ingress_ports)
    m.pfc_shift = topo.shift[bn.switch][bn.egress_port]
    m.pfc_thresh_bytes = model.pfc_threshold(bn, buffer_bytes)
    m.hdrm_pct_of_buffer = 100.0 * (topo.total_hdrm[bn.switch]
                                    + topo.total_rsrv[bn.switch]) / buffer_bytes
    kmin, kmax = model.ecn_band(bn)
    m.kmin_bytes = kmin if kmin is not None else NAN
    m.kmax_bytes = kmax if kmax is not None else NAN
    if bn.f_ports:
        try:
            m.pfc_thresh_egress_equiv_bytes = model.egress_equivalent_threshold(bn, buffer_bytes)
            m.pfc_egress_ceiling_bytes = model.pfc_egress_ceiling(bn, buffer_bytes)
            m.regime_pred = model.regime(bn, buffer_bytes)
        except ValueError as exc:
            print(f"  ! {exc}", file=sys.stderr)
    return m


def pfc_metrics(log: ns3.PfcLog | None, bn: Bottleneck | None, topo: Topology | None,
                window: float, run_end: float) -> PfcMetrics:
    m = PfcMetrics()
    if log is None:
        return m
    m.pfc_events = log.n_events
    m.pfc_qidx = log.qidx_state
    per_dev = log.pause_per_device(clamp_to=int(run_end or log.t_max))
    _, m.pfc_unclosed_pauses = log.pause_totals(int(run_end or log.t_max))
    denom = window if isnum(window) and window > 0 else run_end
    if not denom:
        return m

    def pct(v) -> float:
        return 100.0 * v / denom

    if not per_dev:
        m.pfc_bottleneck_pause_pct = m.pfc_worst_link_pause_pct = 0.0
        m.pfc_host_pause_max_pct = m.pfc_sw_pause_max_pct = 0.0
        m.pfc_paused_devices = 0
        return m
    worst = max(per_dev, key=per_dev.get)
    m.pfc_worst_link_pause_pct = pct(per_dev[worst])
    m.pfc_worst_link_device = f"n{worst[0]}/if{worst[2]}({'sw' if worst[1] == 1 else 'host'})"
    hosts = [v for (_n, nt, _i), v in per_dev.items() if nt == 0]
    sws = [v for (_n, nt, _i), v in per_dev.items() if nt == 1]
    m.pfc_host_pause_max_pct = pct(max(hosts)) if hosts else 0.0
    m.pfc_sw_pause_max_pct = pct(max(sws)) if sws else 0.0
    m.pfc_paused_devices = sum(1 for v in per_dev.values() if v > 0)
    # Only the devices upstream of the congested link are evidence about ITS
    # regime; the global worst can sit on an unrelated link.
    if bn is not None and topo is not None:
        victims = set(bn.pause_victims(topo))
        vv = [v for (n, _nt, i), v in per_dev.items() if (n, i) in victims]
        m.pfc_bottleneck_pause_pct = pct(max(vv)) if vv else 0.0
    return m


def qlen_metrics(log: ns3.QlenLog | None, bn: Bottleneck | None,
                 fab: FabricMetrics, buffer_bytes: int) -> QlenMetrics:
    m = QlenMetrics()
    if log is None or not log.port_max:
        return m
    m.qlen_samples = log.samples
    m.qlen_peak_switch_total_bytes = max(log.switch_total_max.values(), default=NAN)
    if bn is None:
        return m
    peak = log.port_max.get((bn.switch, bn.egress_port), NAN)
    m.qlen_peak_congested_port_bytes = peak
    m.qlen_mean_congested_port_bytes = log.port_mean.get((bn.switch, bn.egress_port), NAN)
    for attr, ref in (("qlen_peak_over_pfc_ceiling", fab.pfc_egress_ceiling_bytes),
                      ("qlen_peak_over_kmax", fab.kmax_bytes),
                      ("qlen_peak_over_kmin", fab.kmin_bytes)):
        if isnum(peak) and isnum(ref) and ref:
            setattr(m, attr, peak / ref)
    return m


def flow_metrics(f: pd.DataFrame | None, bn: Bottleneck | None) -> FlowMetrics:
    m = FlowMetrics()
    if f is None or not len(f):
        return m
    m.fct_flows = len(f)
    m.run_end_ns = float(f["arrival"].max())
    inc = f[f["incast"]]
    m.fct_incast_flows = len(inc)
    m.fct_direct_flows = int((~f["fabric"]).sum())
    if len(inc):
        m.incast_window_ns = float(inc["arrival"].max() - inc["start"].min())
    else:
        print("  ! no incast flows: check --bulk-mb against the real KV flow size "
              "(there is no silent fallback to all flows)", file=sys.stderr)

    if bn is not None:
        iv = flowlib.bottleneck_intervals(f, bn)
        m.bottleneck_flows = len(iv)
        m.n_concurrent_peak, m.n_concurrent_mean = flowlib.concurrency_stats(iv)

    sd = inc.loc[inc["slowdown"].notna(), "slowdown"].to_numpy(float)
    if len(sd):
        m.slow_mean_incast = float(np.mean(sd))
        m.slow_p50_incast = float(np.percentile(sd, 50))
        m.slow_p99_incast = float(np.percentile(sd, 99))
        m.slow_max_incast = float(np.max(sd))
        s = np.std(sd, ddof=1) if len(sd) > 1 else 0.0
        m.slow_cv_incast = float(s / m.slow_mean_incast) if m.slow_mean_incast else NAN
        if isnum(m.n_concurrent_mean) and m.n_concurrent_mean:
            m.slow_mean_over_fairshare = m.slow_mean_incast / m.n_concurrent_mean
            m.slow_p99_over_fairshare = m.slow_p99_incast / m.n_concurrent_mean
    direct = f.loc[~f["fabric"] & f["slowdown"].notna(), "slowdown"]
    m.slow_mean_direct = float(direct.mean()) if len(direct) else NAN
    return m


def barrier_metrics(f: pd.DataFrame | None, decode_nodes: list[int]) -> BarrierMetrics:
    """The first decode step is a synchronisation barrier: it cannot start until
    every KV flow feeding a decode rank has arrived. So KV-ready per rank is its
    latest arrival, and the gate is the worst rank.

    Caveat worth carrying into the write-up: with KV emitted per layer, the spread
    of arrivals on a rank is the duration of that rank's KV stream (staggered by
    prefill compute), not a synchronisation skew. Redefining it over same-layer
    flows is a methodology choice, not a bug fix."""
    m = BarrierMetrics()
    if f is None or not len(f):
        return m
    inc = f[f["incast"]]
    if not len(inc):
        return m
    if decode_nodes:
        dnodes = [d for d in decode_nodes if d in set(inc["dst"])]
    else:
        cnt = inc.groupby("dst").size().sort_values(ascending=False)
        dnodes = [int(d) for d, c in cnt.items() if c > 1] or \
                 ([int(cnt.index[0])] if len(cnt) else [])
    ready, skew = [], []
    for d in dnodes:
        arr = inc.loc[inc["dst"] == d, "arrival"]
        if len(arr):
            ready.append(float(arr.max()))
            skew.append(float(arr.max() - arr.min()))
    if not ready:
        return m
    m.kv_ready_max_ns = max(ready)
    m.kv_ready_min_ns = min(ready)
    m.sync_skew_ns = max(skew)
    m.cross_rank_skew_ns = max(ready) - min(ready)
    m.decode_ranks_seen = len(ready)
    m.incast_dst = ",".join(str(d) for d in dnodes)
    return m


def regime_observed(row: RunRow) -> str:
    """Measured, not assumed. The queue riding at the PFC ceiling means it is held
    by backpressure; below it, rate control limits."""
    pause = row.pfc.pfc_bottleneck_pause_pct
    if not isnum(pause):
        pause = row.pfc.pfc_worst_link_pause_pct
    if not isnum(pause):
        return "?"
    at_ceiling = isnum(row.qlen.qlen_peak_over_pfc_ceiling) and \
        row.qlen.qlen_peak_over_pfc_ceiling >= 0.95
    if at_ceiling and pause > 1.0:
        return "PFC"
    return "MIXED" if pause > 0.5 else "DCQCN"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _band(ax, band: tuple[float, float] | None) -> None:
    """The transition is a band, not a line: PFC below the KMIN crossing, DCQCN
    above the KMAX crossing, mixed in between."""
    if not band:
        return
    lo, hi = band
    ax.axvspan(lo, hi, color="#6a4c93", alpha=0.12, zorder=0,
               label=f"predicted PFC↔DCQCN band ({lo:.1f}–{hi:.1f} MiB)")
    ax.axvline(lo, color="#6a4c93", linestyle=":", lw=1.0)
    ax.axvline(hi, color="#6a4c93", linestyle="--", lw=1.2)


def make_plots(summary: pd.DataFrame, outdir: Path,
               band: tuple[float, float] | None) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    s = summary.sort_values("buffer_mb")

    def save(fig, name, title, ylabel, legend_fs=8):
        ax = fig.axes[0]
        logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
        _band(ax, band)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles = sum((a.get_legend_handles_labels()[0] for a in fig.axes), [])
        labels = sum((a.get_legend_handles_labels()[1] for a in fig.axes), [])
        ax.legend(handles, labels, loc="best", fontsize=legend_fs)
        save_fig(fig, outdir, name, written)

    # 1 REGIME ------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ok = plot_series(ax, s, "buffer_mb", "pfc_bottleneck_pause_pct", "PFC pause on the bottleneck's ingress")
    ok |= plot_series(ax, s, "buffer_mb", "pfc_worst_link_pause_pct", "PFC pause, worst device anywhere",
                marker="v", linestyle="-.")
    if ok:
        ax2 = ax.twinx()
        plot_series(ax2, s, "buffer_mb", "qlen_peak_over_pfc_ceiling", "Peak egress / PFC ceiling",
              marker="s", linestyle="--", color="#d98a00")
        ax2.axhline(1.0, color="#d98a00", linestyle=":", lw=1)
        ax2.set_ylabel("Peak egress / PFC ceiling  (≈1 = held by backpressure)")
        save(fig, "01_regime_vs_buffer.png",
             "Congestion regime vs buffer\n(queue at the PFC ceiling = backpressure; "
             "below it = rate control)", "PFC pause (% of incast window)")
    else:
        plt.close(fig)

    # 2 TRADE-OFF ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_series(ax, s, "buffer_mb", "slow_mean_incast", "Mean slowdown (incast)"):
        plot_series(ax, s, "buffer_mb", "slow_p99_incast", "p99 slowdown (incast)", marker="s", linestyle="--")
        plot_series(ax, s, "buffer_mb", "slow_max_incast", "Max slowdown (incast)", marker="^", linestyle=":")
        mn = s["n_concurrent_mean"].dropna()
        pk = s["n_concurrent_peak"].dropna().unique()
        if len(mn):
            ax.axhline(mn.mean(), color="#2b8a3e", linestyle="-.", lw=1.4,
                       label=f"fair-share reference (mean concurrency ≈ {mn.mean():.0f})")
        if len(pk) == 1 and pk[0]:
            ax.axhline(pk[0], color="#2b8a3e", linestyle=":", lw=1.0,
                       label=f"peak concurrency = {pk[0]:g} (upper bound, not a floor)")
        save(fig, "02_slowdown_mean_vs_tail.png",
             "Mean vs tail vs buffer\n(standalone_fct assumes the flow owns the "
             "bottleneck → reference = concurrency)",
             "Slowdown (fct / standalone_fct)")
    else:
        plt.close(fig)

    # 3 FAIRNESS ----------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_series(ax, s, "buffer_mb", "slow_cv_incast", "CV of incast slowdown (unfairness)"):
        save(fig, "03_fairness_cv_vs_buffer.png",
             "Fairness vs buffer\n(higher CV = victim flows / HOL blocking = PFC signature)",
             "CV = std/mean of slowdown")
    else:
        plt.close(fig)

    # 4 BARRIER ------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_series(ax, s, "buffer_mb", "kv_ready_max_ns", "Decode-start gate = max KV-ready", scale=1e-6):
        ax2 = ax.twinx()
        plot_series(ax2, s, "buffer_mb", "sync_skew_ns", "Spread of KV arrivals on one rank",
              marker="s", linestyle="--", scale=1e-6, color="#d1495b")
        ax2.set_ylabel("Arrival spread (ms)")
        save(fig, "04_barrier_and_skew_vs_buffer.png",
             "Barrier metric vs buffer\n(what actually gates disaggregated decode)",
             "Decode-start gate (ms)")
    else:
        plt.close(fig)

    # 5 MECHANISM ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_series(ax, s, "buffer_mb", "pfc_thresh_bytes", "PFC threshold (exact, ingress)",
             scale=1e-3, color="#1f77b4"):
        plot_series(ax, s, "buffer_mb", "pfc_thresh_egress_equiv_bytes", "  × F_ports (egress-equivalent)",
              marker="D", scale=1e-3, color="#1f77b4", linestyle="--")
        plot_series(ax, s, "buffer_mb", "pfc_thresh_naive_bytes", "buffer/8 (naive — wrong by 86% at 2 MiB)",
              marker="x", scale=1e-3, color="#b0b0b0", linestyle=":")
        plot_series(ax, s, "buffer_mb", "pfc_egress_ceiling_bytes",
              "PFC egress ceiling  Σ(reserve+thresh+headroom)", marker="*",
              scale=1e-3, color="#d1495b", linestyle="-.")
        plot_series(ax, s, "buffer_mb", "qlen_peak_congested_port_bytes", "Measured peak egress",
              marker="s", scale=1e-3, color="#d1495b")
        for col, style, name in (("kmin_bytes", ":", "KMIN"), ("kmax_bytes", "--", "KMAX")):
            v = s[col].dropna().unique()
            if len(v) == 1:
                ax.axhline(v[0] / 1e3, color="#2b8a3e", ls=style, lw=1.2,
                           label=f"{name} = {v[0]/1e3:g} kB")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        save(fig, "05_threshold_vs_ecn_band.png",
             "The mechanism: dynamic PFC threshold vs the ECN band\n"
             "(measured peak at the ceiling → PFC governs)", "Bytes (kB, log)", 7)
    else:
        plt.close(fig)

    # 6 HEADLINE OVERLAY --------------------------------------------------- #
    cols = ["slow_mean_incast", "sync_skew_ns"]
    if s[cols].notna().all(axis=None):
        rng = {c: relative_range(s, c) for c in cols}
        keep = [c for c in cols if rng[c][2] >= MIN_REL_RANGE]
        for c in cols:
            if c not in keep:
                lo, hi, r = rng[c]
                print(f"  - 06 overlay: dropping {c}, it varies by {r:.1%} across the "
                      f"sweep ({lo:.4g}–{hi:.4g}). Normalising that renders noise as "
                      f"signal.", file=sys.stderr)
        if len(keep) < 2:
            print("  - skip 06_aggregate_vs_barrier_overlay: fewer than two series "
                  "carry a real trend.", file=sys.stderr)
            return written
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, grp in s.groupby("variant"):
            grp = grp.sort_values("buffer_mb")
            suff = "" if s["variant"].nunique() == 1 else f" [{variant}]"
            for c, mk, ls in zip(keep, ("o", "s"), ("-", "--")):
                v = grp[c].to_numpy(float)
                lo, hi = np.nanmin(v), np.nanmax(v)
                norm = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
                lo_a, hi_a, r = rng[c]
                ax.plot(grp["buffer_mb"], norm, marker=mk, linestyle=ls,
                        label=f"{c} ({lo_a:.4g}–{hi_a:.4g}, {r:.1%})" + suff)
        save(fig, "06_aggregate_vs_barrier_overlay.png",
             "Aggregate vs barrier, normalised\n(divergence = the trade-off the tool reveals)",
             "Normalised (0–1)")
    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
FRONT_COLS = ["run_dir", "variant", "buffer_mb", "regime_pred", "regime_obs",
              "congested_link", "congested_rate_gbps", "fanin_ports",
              "n_concurrent_peak", "n_concurrent_mean",
              "pfc_thresh_bytes", "pfc_thresh_egress_equiv_bytes",
              "pfc_egress_ceiling_bytes", "kmin_bytes", "kmax_bytes",
              "qlen_peak_congested_port_bytes", "qlen_peak_over_pfc_ceiling",
              "qlen_peak_over_kmax", "pfc_bottleneck_pause_pct",
              "pfc_worst_link_pause_pct", "pfc_worst_link_device", "pfc_qidx",
              "slow_mean_incast", "slow_p99_incast", "slow_cv_incast",
              "slow_mean_over_fairshare", "kv_ready_max_ns", "sync_skew_ns",
              "makespan_ms", "kv_bound_ratio"]
REPORT_COLS = ["buffer_mb", "regime_pred", "regime_obs", "fanin_ports",
               "n_concurrent_mean", "pfc_thresh_egress_equiv_bytes",
               "pfc_egress_ceiling_bytes", "qlen_peak_congested_port_bytes",
               "qlen_peak_over_pfc_ceiling", "pfc_bottleneck_pause_pct",
               "slow_mean_incast", "slow_cv_incast", "kv_ready_max_ns"]


def analyse_run(tag: str, astra_dir: Path, ns3_dir: Path | None,
                topo: Topology | None, cfg: Ns3Config | None,
                buffer_mb: float, bulk_bytes: int,
                decode_nodes: list[int]) -> tuple[RunRow, Bottleneck | None] | None:
    adf = astra.read_run(astra_dir)
    if adf is None:
        return None
    row = RunRow(run_dir=tag, variant=BUFFER_AXIS.variant(tag), buffer_mb=buffer_mb,
                 ns3_dir=str(ns3_dir or ""),
                 cc_mode=cfg.cc_mode if cfg and cfg.cc_mode is not None else NAN)
    row.astra = astra_metrics(adf)

    flows = pfc_log = qlen_log = None
    if ns3_dir is not None:
        raw = ns3.read_fct(ns3_dir / "fct.txt")
        flows = flowlib.annotate(raw, topo, bulk_bytes) if raw is not None else None
        pfc_log = ns3.read_pfc(ns3_dir / "pfc.txt")
        qlen_log = ns3.read_qlen(ns3_dir / "qlen.txt")

    bn = flowlib.find_bottleneck(topo, qlen_log.port_max if qlen_log else None, flows)
    model = FabricModel(topo, cfg) if (topo is not None and cfg is not None) else None
    buffer_bytes = int(buffer_mb * 1024 * 1024)

    row.fabric = fabric_metrics(model, bn, buffer_bytes)
    row.flows = flow_metrics(flows, bn)
    row.pfc = pfc_metrics(pfc_log, bn, topo, row.flows.incast_window_ns,
                          row.flows.run_end_ns)
    row.qlen = qlen_metrics(qlen_log, bn, row.fabric, buffer_bytes)
    row.barrier = barrier_metrics(flows, decode_nodes)
    row.regime_obs = regime_observed(row)
    row.per_node = (adf.groupby(["sys_id", "op_class"])
                    .agg(count=("name", "size"), total_bytes=("comm_size", "sum"),
                         total_busy_ns=("duration", "sum"))
                    .reset_index()
                    .assign(buffer_mb=buffer_mb, variant=BUFFER_AXIS.variant(tag)))
    return row, bn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("astra_root", nargs="?", default=DEFAULT_ASTRA_ROOT)
    ap.add_argument("--ns3-root", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--topology", default=DEFAULT_TOPOLOGY,
                    help="path, directory, or {tag} template")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="path, directory, or {tag} template; MUST resolve per tag")
    ap.add_argument("--bulk-mb", type=float, default=1.0)
    ap.add_argument("--decode-nodes", default=None)
    ap.add_argument("--headroom-factor", type=int, default=3)
    ap.add_argument("--print-patch", action="store_true",
                    help="print the ns-3 diff that adds qIndex to pfc.txt and exit")
    a = ap.parse_args(argv)

    if a.print_patch:
        print(ns3.PFC_QIDX_PATCH)
        return 0

    astra_root = Path(a.astra_root)
    if not astra_root.is_dir():
        print(f"ERROR: astra_root not found: {astra_root}", file=sys.stderr)
        return 2
    ns3_root = resolve_ns3_root(astra_root, a.ns3_root, DEFAULT_NS3_ROOT)
    outdir = resolve_outdir(a.out, DEFAULT_OUTDIR, astra_root, "buffer_analysis")
    decode_nodes = [int(x) for x in a.decode_nodes.split(",")] if a.decode_nodes else []
    bulk_bytes = int(a.bulk_mb * 1024 * 1024)

    tags = discover_runs(astra_root, outdir, skip_names=("buffer_analysis",))
    if not tags:
        print(f"ERROR: no run sub-directories under {astra_root}", file=sys.stderr)
        return 2
    print(f"Scanning {len(tags)} run dirs under:\n  {astra_root}")
    print(f"ns-3 root: {ns3_root or '(none — regime metrics will be empty)'}\n")

    rows: list[RunRow] = []
    topo_cache: dict[str, Topology | None] = {}
    degraded: set[str] = set()
    band = None

    for d in tags:
        tag = d.name
        buf = BUFFER_AXIS.value(tag)
        if buf is None:
            print(f"  - skip {tag!r}: no 'buf<num>' token", file=sys.stderr)
            continue
        ns3_dir = find_ns3_run(ns3_root, tag)
        search = [ns3_dir, ns3_root, astra_root]

        tpath = find_aux(a.topology, tag, "physical_topology.txt", search)
        cpath = find_aux(a.config, tag, "config.txt", search)
        if tpath is None:
            print(f"  ! {tag}: physical_topology.txt NOT RESOLVED — no PFC threshold, "
                  f"no ECN band, and no fabric/direct filter, so the incast statistics "
                  f"will be a MIXTURE of congested KV flows and uncongested direct-link "
                  f"collectives. Pass --topology.", file=sys.stderr)
            degraded.add(tag)
        if cpath is None:
            print(f"  ! {tag}: config.txt NOT RESOLVED — KMIN/KMAX unknown, no regime "
                  f"prediction. Pass --config.", file=sys.stderr)
            degraded.add(tag)

        topo = None
        if tpath:
            key = f"{tpath}|{a.headroom_factor}"
            if key not in topo_cache:
                try:
                    topo_cache[key] = parse_topology(tpath, a.headroom_factor)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! cannot parse {tpath}: {exc}", file=sys.stderr)
                    topo_cache[key] = None
                if (t := topo_cache[key]) and t.ecmp_pairs:
                    print(f"  ! ECMP ties on {len(t.ecmp_pairs)} (node, host) pairs: "
                          f"runtime paths are hash-chosen, so per-flow path "
                          f"attribution is approximate", file=sys.stderr)
            topo = topo_cache[key]
        cfg = parse_ns3_config(cpath) if cpath else None
        if cfg:
            for w in cfg.warnings():
                print(f"  ! {tag}: {w}", file=sys.stderr)
            if cfg.buffer_mb is None:
                print(f"  ! {tag}: no BUFFER_SIZE in {cpath} (a template?) — using "
                      f"'buf{buf:g}' from the dir name", file=sys.stderr)
            elif abs(cfg.buffer_mb - buf) > 1e-6:
                print(f"  ! {tag}: BUFFER_SIZE={cfg.buffer_mb} in config.txt but "
                      f"'buf{buf:g}' in the dir name — trusting config.txt. If every "
                      f"run says this, --config is not resolving per tag.",
                      file=sys.stderr)
                buf = cfg.buffer_mb

        res = analyse_run(tag, d, ns3_dir, topo, cfg, buf, bulk_bytes, decode_nodes)
        if res is None:
            print(f"  - skip {tag!r}: no readable CSVs", file=sys.stderr)
            continue
        row, bn = res
        rows.append(row)
        if band is None and topo is not None and cfg is not None and bn is not None:
            band = FabricModel(topo, cfg).flip_band(bn)

        print(f"  + {tag:<28} buf={row.buffer_mb:<5g} pred={row.fabric.regime_pred:<5} "
              f"obs={row.regime_obs:<5} pause={row.pfc.pfc_bottleneck_pause_pct:6.2f}% "
              f"peak/ceil={row.qlen.qlen_peak_over_pfc_ceiling:5.2f} "
              f"meanSD={row.flows.slow_mean_incast:.1f} "
              f"Fp={row.fabric.fanin_ports:g} N={row.flows.n_concurrent_peak:g}")

    if not rows:
        print("ERROR: no valid runs parsed.", file=sys.stderr)
        return 2

    summary = pd.DataFrame([r.flat() for r in rows])
    summary = summary.sort_values(["variant", "buffer_mb"]).reset_index(drop=True)
    summary = order_columns(summary, FRONT_COLS)
    summary_path = write_table(summary, outdir, "summary.csv")
    # per-node audit table, one row per (rank, op class) -- restored for parity with
    # bandwidth_analyzer, which has always written one.
    per_node = pd.concat([r.per_node for r in rows if r.per_node is not None],
                         ignore_index=True)
    per_node_path = write_table(per_node, outdir, "per_node.csv")
    plots = make_plots(summary, outdir, band)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print("\n================ BUFFER SWEEP SUMMARY ================")
    print(summary[[c for c in REPORT_COLS if c in summary.columns]].to_string(index=False))
    if band:
        print(f"\nPredicted: PFC below {band[0]:.2f} MiB, DCQCN above {band[1]:.2f} MiB.")
        print("The prediction uses topology + config only. Agreement with regime_obs "
              "is the result; disagreement is the finding.")
    if degraded:
        print(f"\nWARNING: {len(degraded)} run(s) ran without topology and/or config. "
              f"Those rows are degraded — see the messages above.")
    if (summary.get("pfc_qidx") == "MISSING").any():
        print("\nWARNING: pfc.txt has no qIndex column. With qos-enabled the PAUSE/"
              "RESUME sequences of different priority groups interleave on one ifindex "
              "and the pause totals are NOT a measurement. Run --print-patch for the "
              "three-line ns-3 diff.")
    print("\nWrote:")
    for p in [summary_path, per_node_path, *plots]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())