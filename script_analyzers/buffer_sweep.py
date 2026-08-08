#!/usr/bin/env python3
"""
buffer_sweep — why does an OVERSUBSCRIBED topology behave differently from
a non-oversubscribed one at the same nominal bandwidth?

A plain buffer-cost sweep answers "what does the buffer cost", deliberately
without explaining the mechanism. This is a mechanism question instead, asked by
comparing two topologies run at the same buffer
values: T1 (4 ToR switches, each oversubscribed ~10:1 into one core switch)
against T2 (one switch, no oversubscription). This script does NOT compare
them itself — it analyses one sweep at a time, and the comparison is done by
eye between two runs (T1, T2, and later other models/topologies). Nothing
here assumes the two runs share a link numbering, a switch count, or even a
topology shape.

Two suspected mechanisms, both traced back to PFC on the oversubscribed link:

    1. KV-cache delivery is not smooth in T1 -- it stalls, apparently in step
       with PFC PAUSE frames.
    2. The same PAUSE-driven skew shows up upstream, in how unevenly the PP
       (pipeline-parallel) activation handoff between prefill stages arrives.
       That skew changes how long the RECEIVING stage's TP all-reduce takes,
       which changes the prefill completion time -- TTFT.

Ten figures, each answering one part of that:

    01  CAUSAL CHAIN TO TTFT       PP arrival skew, the receiving stage's FIRST
                                   (skew-gated) all-reduce, the steady-state mean
                                   of the rest, and TTFT, all vs buffer. Tests
                                   whether skew propagates through the FIRST
                                   all-reduce into TTFT while the steady state
                                   (the control) stays flat.
    02  KV CUMULATIVE ARRIVAL      one panel per buffer value; cumulative KV
                                   bytes arrived per decode rank over time. The
                                   horizontal spread between ranks IS the skew;
                                   a staircase with flat stretches IS a stall.
    03  LINK BANDWIDTH/CONCURRENCY every link any KV flow crosses (not just the
                                   one deepest-queue bottleneck), ranked: does
                                   ONLY the measured link suffer, or several?
    04  BUFFER OCCUPANCY(t)        the bottleneck switch's buffer occupancy as a
                                   % of the swept BUFFER_SIZE, one column per
                                   buffer value, PFC PAUSE spans shaded -- the
                                   pauses line up with a full buffer. (The old
                                   arrival-binned KV throughput row is gone: the
                                   attribution was too coarse to be honest.)
    05  QUEUE(t) PER SWITCH        a grid (rows=switch, cols=buffer), with PFC
                                   PAUSE spans shaded.
    06  OCCUPANCY vs BUFFER        is the added buffer actually used, per switch.
    07  PFC COUNT vs BUFFER        the raw PAUSE-frame count at the bottleneck,
                                   one point per buffer -- just the "how many".
    08  KV TP-GROUP SKEW           per (decode stage, layer): the spread between
                                   the arrival of the KV shards that feed the
                                   SAME TP group -- the wait the first decode
                                   all-reduce of that layer inherits. min / mean
                                   / p99 over the (stage, layer) population, vs
                                   buffer.
    09  KV TIME PER RANK           per decode rank: the total KV transfer time
                                   (first->last arrival) and the signed gap
                                   between the completion of that rank's FIRST
                                   and LAST layer, vs buffer.
    10  DECODE FIRST ALL-REDUCE    per decode PP stage: the entry skew (spread
                                   of the shards' start ticks) of the stage's
                                   FIRST TP all-reduce -- the one gated by its
                                   own KV -- and, beside it, that all-reduce's
                                   duration against the mean duration of the
                                   rest, vs buffer. How much KV skew the decode
                                   pipeline actually inherits.

    11  DECODE KV STALL            how much the decode is stalled waiting for
                                   its KV. Left: token-2 latency after token 1.
                                   Middle: the first decode pass (decode start
                                   -> token 2) against the steady inter-token
                                   gap -- the excess IS the KV stall. Right:
                                   the cause, per stage: KV readiness minus the
                                   arrival of the stage's first input (FIRSTTOK
                                   for stage 0, the it=0 PP activation after);
                                   >0 means the stage outruns its KV and waits.

    (The old 07, decode start as a multiple of TTFT, was dropped; the scalar
    kv_gate_over_ttft is still computed and lands in summary.csv.)

Everything is measured, nothing fitted (same discipline as utils.pp):
fct.txt / pfc.txt / qlen.txt plus, for TTFT, this run's ASTRA-sim trace.

Declared, never inferred:
    --sweep       the one path input; every other path is derived (utils.paths).
    --placement   the rank->role map (utils.roles).
    --bottleneck  optional 'sw->peer' to force which link is treated as the
                  ground-truth bottleneck; it must be among the links this
                  sweep's KV flows actually cross, or the run aborts.
    --top-links   how many KV-crossed links figure 03 (and summary.csv) carry;
                  default 6. The full set is topology-derived and identical at
                  every buffer value of one sweep -- only its congestion
                  ranking can shift, so the set and its display order are
                  fixed once, from the run with the smallest buffer (the most
                  congested, least likely to be a near-tie), never re-ranked
                  per row.

Usage
-----
    python3 buffer_sweep.py --sweep buffer_sweep_T1
    python3 buffer_sweep.py --sweep buffer_sweep_T2
    python3 buffer_sweep.py --sweep buffer_sweep_T1 --top-links 4 -o /tmp/x
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import astra
from utils import flows as flowlib
from utils import ns3, paths, pp, roles
from utils.cli import Abort, drain_warnings, need, warn
from utils.fabric import parse_ns3_config, parse_topology
from utils.measures import (LinkStat, barrier, decode_ar_stats,
                            decode_stall_stats, kv_rank_series, kv_skew_stats,
                            link_metrics, ttft_from, victim_pause_intervals)
from utils.plots import (BLUE, CORAL, GREEN, MS, MUTED, VIOLET,
                         downsample_max, logx_pow2, save_fig, zoom_y)
from utils.roles import Placement
from utils.paths import BUFFER_AXIS, fresh_dir

NAN = float("nan")
KIND = "buffer"


# --------------------------------------------------------------------------- #
# One row per run. (LinkStat, the per-link score, lives in utils.measures.)
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    tag: str = ""
    buffer_mb: float = NAN
    buffer_bytes: float = NAN
    bottleneck: str = ""
    bn_rate_gbps: float = NAN

    # -- 01 causal chain ------------------------------------------------------ #
    ttft_ns: float = NAN                  # token 1 = END OF PREFILL (FIRSTTOK send)
    kv_gate_ns: float = NAN               # decode start = last KV arrival (2nd token)
    kv_gate_over_ttft: float = NAN        # decode start as a multiple of TTFT
    pp_skew_ns: float = NAN               # worst-wave cross-rank PP arrival skew
    pp_skew_mean_ns: float = NAN
    pp_first_ns: float = NAN
    pp_last_ns: float = NAN
    pp_stage: object = None               # destination stage of the worst wave
    pp_n_waves: int = 0
    # All-reduce metrics come from the ASTRA stats CSV (authoritative per-collective
    # duration/bytes), for the receiving stage's PREFILL TP all-reduce.
    rs_ar_first_ns: float = NAN           # duration of the GATED all-reduce (the
                                          # receiving stage's first, skew-stalled)
    rs_ar_rest_mean_ns: float = NAN       # mean duration of the steady-state ones
    rs_ar_first_bw: float = NAN           # gated all-reduce effective bw (bytes/ns
                                          # = GB/s): comm_size / duration
    rs_ar_rest_bw: float = NAN            # steady-state effective bw (mean)
    rs_ar_first_stage_bw: float = NAN     # effective bw of the FIRST prefill stage's
                                          # all-reduce (stage 0, ungated -- starts
                                          # immediately); the reference the steady
                                          # receiving-stage bw is compared against
    rs_ar_n: int = 0                      # receiving-stage prefill TP collectives

    # -- 02 KV skew / smoothness ---------------------------------------------- #
    kv_ready_min_ns: float = NAN
    cross_rank_skew_ns: float = NAN
    kv_stream_duration_ns: float = NAN
    decode_ranks: str = ""

    # -- 11 decode stalled by KV reception ------------------------------------ #
    dec_start_ns: float = NAN             # first decode COMP start (stage 0 wakes
                                          # on the FIRSTTOK ARRIVAL, not its send)
    tok2_ns: float = NAN                  # second token: first DECFB send, max
                                          # over shards (slowest shard's send)
    tok2_latency_ns: float = NAN          # tok2 - dec_start: the first decode pass
    tok2_after_tok1_ns: float = NAN       # tok2 - TTFT: the user-visible gap
    itl_steady_ns: float = NAN            # mean inter-token gap of the remaining
                                          # DECFB iterations (steady-state control)

    # -- 08 KV skew within each TP group -------------------------------------- #
    # Population: one skew per (decode stage, layer) = the spread between the
    # arrivals of the KV shards feeding the same TP group. Groups with a single
    # shard arrival have no skew and are excluded (kv_tp_skew_n counts the rest).
    kv_tp_skew_min_ns: float = NAN
    kv_tp_skew_mean_ns: float = NAN
    kv_tp_skew_p99_ns: float = NAN
    kv_tp_skew_n: int = 0

    # -- not flattened: per-figure raw data ----------------------------------- #
    links: list = field(default_factory=list)          # list[LinkStat]
    kv_rank_series: dict = field(default_factory=dict)  # rank -> (times_ns, cumbytes)
    kv_rank_span: dict = field(default_factory=dict)     # rank -> last-first arrival (ns)
    kv_rank_layer_delta: dict = field(default_factory=dict)  # rank -> signed ns:
                                                        # completion(last layer) -
                                                        # completion(first layer)
    dec_ar: dict = field(default_factory=dict)           # decode stage -> first/steady
                                                        # all-reduce skew + duration
    dec_stall: dict = field(default_factory=dict)        # decode stage -> KV readiness
                                                        # vs first-input arrival
    bn_pause_intervals: list = field(default_factory=list)
    qseries: dict = field(default_factory=dict)         # sw -> (ts_ns, bytes)
    qswitch_peak: dict = field(default_factory=dict)     # sw -> peak total bytes
    qswitch_mean: dict = field(default_factory=dict)     # sw -> mean total bytes
    pause_intervals: dict = field(default_factory=dict)  # sw -> [(start,end)]

    def flat(self) -> dict:
        d = asdict(self)
        for k in ("links", "kv_rank_series", "kv_rank_span", "kv_rank_layer_delta",
                  "dec_ar", "dec_stall", "bn_pause_intervals", "qseries",
                  "qswitch_peak", "qswitch_mean", "pause_intervals"):
            d.pop(k, None)
        for rank in sorted(self.kv_rank_span):
            d[f"kv_rank{rank}_span_ns"] = self.kv_rank_span[rank]
        for rank in sorted(self.kv_rank_layer_delta):
            d[f"kv_rank{rank}_layer_delta_ns"] = self.kv_rank_layer_delta[rank]
        for st in sorted(self.dec_ar):
            m = self.dec_ar[st]
            d[f"dec{st}_ar_first_skew_ns"] = m["first_skew_ns"]
            d[f"dec{st}_ar_first_dur_ns"] = m["first_dur_ns"]
            d[f"dec{st}_ar_rest_skew_mean_ns"] = m["rest_skew_mean_ns"]
            d[f"dec{st}_ar_rest_dur_mean_ns"] = m["rest_dur_mean_ns"]
        for st in sorted(self.dec_stall):
            m = self.dec_stall[st]
            d[f"dec{st}_input_arrival_ns"] = m["input_arrival_ns"]
            d[f"dec{st}_kv_ready_ns"] = m["kv_ready_ns"]
            d[f"dec{st}_kv_lateness_ns"] = m["kv_lateness_ns"]
        for i, ls in enumerate(self.links):
            d[f"link{i}_label"] = ls.label
            d[f"link{i}_window_ns"] = ls.window_ns      # KV window: normaliser for
                                                        # the raw PAUSE-frame count
            d[f"link{i}_eff_pct"] = ls.eff_pct
            d[f"link{i}_delivered_gbps"] = ls.delivered_gbps
            d[f"link{i}_conc_peak"] = ls.conc_peak
            d[f"link{i}_conc_mean"] = ls.conc_mean
            d[f"link{i}_qpeak_pct"] = ls.qpeak_pct
            d[f"link{i}_qpeak_bytes"] = ls.qpeak_bytes    # absolute occupancy:
            d[f"link{i}_qmean_bytes"] = ls.qmean_bytes    # comparable in MB across
                                                          # runs, unlike qpeak_pct
                                                          # (% of the swept buffer)
            d[f"link{i}_pause_frames"] = ls.pause_frames
            d[f"link{i}_pause_pct_of_window"] = ls.pause_pct_of_window
        return d


# --------------------------------------------------------------------------- #
# Measurement helpers. The ones shared with incast_sweep/cc_sweep (barrier,
# kv_rank_series, kv_skew_stats, decode_ar_stats, decode_stall_stats, ttft_from,
# link_metrics, victim_pause_intervals) live in utils.measures; what stays here
# is this sweep's own question.
# --------------------------------------------------------------------------- #
def rs_allreduce_stats(adf: pd.DataFrame | None, placement: Placement, ppr) -> dict:
    """The prefill TP all-reduce, read from the ASTRA stats CSV (the authoritative
    per-collective duration and bytes -- the ns-3 fct.txt only sees the on-wire
    bursts, which under-count the collective's wall-clock ~10x).

    Reported as EFFECTIVE BANDWIDTH, comm_size / duration in bytes/ns (= GB/s,
    the CSV's own bw_bytes_per_ns), for three all-reduces:

        rs_ar_first_bw        the receiving stage's FIRST prefill all-reduce --
                              the one gated by the PP wave, so its duration is
                              stretched by the skew stall and its effective bw is
                              depressed. rs_ar_first_ns keeps its raw duration.
        rs_ar_rest_bw         the mean over that stage's remaining (steady-state)
                              all-reduces -- flat, buffer-independent.
        rs_ar_first_stage_bw  the FIRST prefill stage's all-reduce (stage 0), which
                              starts immediately and is never gated: the reference
                              the steady receiving-stage bw is compared against.

    Stages are the ASTRA `ss` field. With PP=1 there is no wave and no receiving
    stage, so the ONLY prefill stage reports instead: its first all-reduce is
    ungated (nothing to inherit a skew from), but its bw-vs-buffer curve is still
    the quantity the cross-model compare overlays -- expected flat, the control.
    All NaN/0 with no ASTRA run, or no prefill TP (TP=1)."""
    out = {"rs_ar_first_ns": NAN, "rs_ar_rest_mean_ns": NAN, "rs_ar_n": 0,
           "rs_ar_first_bw": NAN, "rs_ar_rest_bw": NAN, "rs_ar_first_stage_bw": NAN}
    if adf is None:
        return out
    tp = adf[(adf["op_class"] == "TP") & (adf["phase"] == "prefill")]
    if not len(tp) or "ss" not in tp.columns:
        return out
    keys = [c for c in ("pl", "ss", "L", "it", "op") if c in tp.columns]
    # One row per collective: slowest shard sets the wall-clock duration, comm_size
    # is the identical per-rank payload. bw is the CSV's effective rate for that
    # collective (bytes moved / how long it took).
    g = (tp.groupby(keys, dropna=False)
           .agg(start=("start_tick", "min"), dur=("duration", "max"),
                cs=("comm_size", "first")).reset_index())
    g = g[g["dur"] > 0].copy()
    if g.empty:
        return out
    g["bw"] = g["cs"] / g["dur"]                       # bytes/ns = GB/s
    g["ss"] = pd.to_numeric(g["ss"], errors="coerce")

    if ppr.available and ppr.stage is not None:
        stage = ppr.stage
    else:
        # PP=1: no wave, no receiving stage. The only prefill stage reports;
        # several stages with no measurable wave stays NaN (ambiguous).
        stages = g["ss"].dropna().unique()
        if len(stages) != 1:
            return out
        stage = stages[0]
    recv = g[g["ss"] == stage].sort_values("start")
    if recv.empty:
        return out
    first = recv.iloc[0]
    out["rs_ar_first_bw"] = float(first["bw"])
    out["rs_ar_first_ns"] = float(first["dur"])
    out["rs_ar_n"] = int(len(recv))
    rest = recv.iloc[1:]
    if len(rest):
        out["rs_ar_rest_bw"] = float(rest["bw"].mean())
        out["rs_ar_rest_mean_ns"] = float(rest["dur"].mean())
    first_stage = g[g["ss"] == 0]
    if len(first_stage):
        out["rs_ar_first_stage_bw"] = float(first_stage["bw"].mean())
    return out


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyse(tag: str, p: paths.SweepPaths, placement: Placement,
           chosen_labels: list[str], want_series: bool = True) -> Row:
    """One run. analyse_sweep has already guaranteed the config inputs exist
    (Abort otherwise) and every output file is on disk (runs missing one are
    skipped before this is called), so nothing here re-checks file presence."""
    buf = BUFFER_AXIS.value(tag)
    ns3_dir = p.ns3_run(tag)
    topo = parse_topology(p.topology(tag))
    cfg = parse_ns3_config(p.config(tag))
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    need(cfg.buffer_mb is not None,
         f"{tag}: no BUFFER_SIZE in {p.config(tag)}.")
    need(abs(cfg.buffer_mb - buf) < 1e-6,
         f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but 'buf{buf:g}' "
         f"in the directory name. One of the two is lying.")

    row = Row(tag=tag, buffer_mb=float(buf), buffer_bytes=float(buf) * 1024 * 1024)

    # The one read of this run's ASTRA trace: TTFT, KV arrivals, PP skew,
    # all-reduce and decode-stall metrics all share this frame.
    adir = p.astra_run(tag)
    adf = astra.read_run(adir)
    need(adf is not None, f"{tag}: no readable stats_sys*.csv under {adir}.")
    row.ttft_ns = ttft_from(adf, tag)

    raw = ns3.read_fct(ns3_dir / "fct.txt")
    need(raw is not None and len(raw), f"{tag}: fct.txt has no parsable rows.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)
    for w in roles.check(f, placement):
        warn(f"{tag}: {w}")
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{tag}: no KV flow after classification.")

    # series (the per-sample queue timeline) feed only the per-tag plots; a
    # cross-model compare (want_series=False) needs just the scalars, so it skips
    # building them -- the big saving on qlen.txt reads across many workloads.
    qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=want_series)
    need(qlen is not None and qlen.port_max, f"{tag}: qlen.txt has no samples.")

    pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
    need(pfc is not None, f"{tag}: pfc.txt unreadable.")
    if pfc.qidx_state == "MISSING":
        warn(f"{tag}: pfc.txt has no qIndex; pause_pct_of_window is "
             f"approximate (see ns3.PFC_QIDX_PATCH). The pause frame COUNT "
             f"is unaffected.")

    run_end = int(f["arrival"].max())

    # -- every link this run's KV flows cross, indexed by the SWEEP-WIDE fixed
    #    label order (chosen_labels), not this run's own congestion ranking --
    #    see the module docstring: the link SET is topology-derived and
    #    invariant across buffer values, only the ranking can shift. ------ #
    links_here = {str(bn): bn for bn in
                 flowlib.candidate_links(topo, qlen.port_max, kv)}
    row.links = []
    per_switch: dict[int, list] = defaultdict(list)
    for label in chosen_labels:
        bn_i = links_here.get(label)
        if bn_i is None:
            warn(f"{tag}: link {label} (present in another run of this sweep) "
                 f"is not crossed by any KV flow here; recorded as NaN.")
            row.links.append(LinkStat(label=label))
            continue
        row.links.append(link_metrics(kv, bn_i, topo, pfc, qlen, row.buffer_bytes))
        # PAUSE on this link's ingress victims is what throttles the inflow
        # into bn_i.switch -- the signal figure 05 overlays on that switch's
        # queue. A switch is never itself the PFC "victim" of ITS OWN egress
        # queue; the upstream neighbour feeding it is (see PfcLog docstring).
        per_switch[bn_i.switch].extend(
            victim_pause_intervals(pfc, bn_i, topo, clamp_to=run_end))
    # two candidate links on one switch can share PAUSE victims; dedupe so the
    # figure-05 shading does not stack the same interval twice.
    row.pause_intervals = {sw: sorted(set(iv)) for sw, iv in per_switch.items()}
    need(row.links, f"{tag}: no candidate link to report.")

    bn = links_here.get(chosen_labels[0])
    need(bn is not None, f"{tag}: the top-ranked link {chosen_labels[0]} is "
                         f"not crossed by any KV flow in THIS run -- it cannot "
                         f"be treated as the bottleneck here.")
    row.bottleneck, row.bn_rate_gbps = str(bn), bn.rate / 1e9

    kv_bn = kv[flowlib.crosses(kv, bn)]
    need(len(kv_bn), f"{tag}: no KV flow crosses the top-ranked link {bn}.")
    row.bn_pause_intervals = victim_pause_intervals(pfc, bn, topo, clamp_to=run_end)

    # KV arrival, PP skew and all-reduce metrics all read the ASTRA stats CSV
    # (per-op end_tick = arrival, cleanly labelled by op/stage/iteration) rather
    # than reconstructing them from fct.txt flows -- identical nanosecond values,
    # none of the flow classification / send-recv dedup / wave-grouping heuristics.
    # (Queue occupancy, PFC and per-physical-link stats above stay on ns-3: ASTRA
    # has no equivalent.) `adf` was read once at the top of this function.
    kv_arr = astra.kv_arrivals(adf)

    for k, v in barrier(kv_arr, placement).items():
        setattr(row, k, v)
    row.kv_gate_over_ttft = (row.kv_gate_ns / row.ttft_ns
                             if pd.notna(row.ttft_ns) and row.ttft_ns > 0 else NAN)

    skew_scal, row.kv_rank_span, row.kv_rank_layer_delta = kv_skew_stats(kv_arr)
    for k, v in skew_scal.items():
        setattr(row, k, v)
    if row.kv_tp_skew_n == 0:
        warn(f"{tag}: no (stage, layer) KV group has >=2 shard arrivals; the "
             f"TP-group skew figure will be empty for this run. (TP=1, or the "
             f"KV names carry no ds=/L= fields.)")
    row.dec_ar = decode_ar_stats(adf)
    if not row.dec_ar:
        warn(f"{tag}: no decode TP all-reduce in the ASTRA stats; the decode "
             f"first-all-reduce skew figure will be empty for this run.")

    stall_scal, row.dec_stall = decode_stall_stats(adf, kv_arr)
    for k, v in stall_scal.items():
        setattr(row, k, v)
    if pd.notna(row.tok2_ns) and pd.notna(row.ttft_ns):
        row.tok2_after_tok1_ns = row.tok2_ns - row.ttft_ns
    if pd.isna(row.tok2_ns):
        warn(f"{tag}: no DECFB send in the ASTRA stats; the decode KV-stall "
             f"figure will be empty for this run.")

    ppr = pp.measure(adf)
    if not ppr.available:
        warn(f"{tag}: no inter-stage PP-prefill activation found; the causal-chain "
             f"figure will be empty for this run. (PP=1, or placement has "
             f"one prefill stage.)")
    row.pp_skew_ns = ppr.skew_ns
    row.pp_skew_mean_ns = ppr.skew_mean_ns
    row.pp_first_ns = ppr.first_ns
    row.pp_last_ns = ppr.last_ns
    row.pp_stage = ppr.stage
    row.pp_n_waves = ppr.n_waves
    for k, v in rs_allreduce_stats(adf, placement, ppr).items():
        setattr(row, k, v)
    if ppr.available and pd.isna(row.rs_ar_first_bw):
        warn(f"{tag}: no prefill TP all-reduce in the ASTRA stats for the "
             f"receiving stage {ppr.stage}; all-reduce bandwidths unavailable.")

    row.kv_rank_series = kv_rank_series(kv_arr, placement)

    for sw, (ts, ys) in qlen.switch_series.items():
        if len(ts) == 0:
            continue
        row.qseries[sw] = downsample_max(ts, ys, 2000)
        row.qswitch_peak[sw] = float(qlen.switch_total_max.get(sw, max(ys)))
        row.qswitch_mean[sw] = float(np.mean(ys))

    return row


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _finish(fig, ax, s, outdir, name, title, ylabel, written, extra_axes=()):
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    for a in extra_axes:
        h, l = a.get_legend_handles_labels()
        handles += h
        labels += l
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8)
    save_fig(fig, outdir, name, written)


def make_plots(rows: list[Row], s: pd.DataFrame, outdir: Path,
              chosen_labels: list[str]) -> list[Path]:
    written: list[Path] = []
    x = s["buffer_mb"]
    runs = sorted(rows, key=lambda r: r.buffer_mb)

    # 01 CAUSAL CHAIN TO TTFT ------------------------------------------------ #
    # One stacked panel per link in the chain, sharing the buffer x-axis. A
    # single twin-axis plot cannot hold these together: TTFT is in ms and moves
    # little, PP skew and the all-reduce span are in µs (~1000x smaller) and
    # move a lot, so on shared axes the coupling is unreadable. Stacked, the
    # chain reads top-to-bottom and the correlation is vertical peak alignment;
    # each panel keeps its own autoscaled scale. Order = causal order:
    # buffer -> PFC/PAUSE -> PP skew -> receiving-stage all-reduce -> TTFT.
    if s["ttft_ns"].notna().any():
        # each panel: (ylabel, [(series, color, style, label), ...])
        panels = []
        if s["pp_skew_ns"].notna().any():
            panels.append(("PP arrival skew (µs)",
                           [(s["pp_skew_ns"] / 1e3, CORAL, "s--", None)]))
        # All-reduce panels are EFFECTIVE BANDWIDTH (bytes/ns = GB/s) from the
        # ASTRA stats CSV, not duration. The gated one's bw is depressed because
        # its duration carries the skew stall; the steady one is compared against
        # the first prefill stage's all-reduce, which starts immediately (ungated).
        if s["rs_ar_first_bw"].notna().any():
            panels.append(("Gated all-reduce\neff. bw (GB/s)",
                           [(s["rs_ar_first_bw"], VIOLET, "^:", None)]))
        steady = []
        if s["rs_ar_rest_bw"].notna().any():
            steady.append((s["rs_ar_rest_bw"], GREEN, "D-.", "receiving stage (steady)"))
        if s["rs_ar_first_stage_bw"].notna().any():
            steady.append((s["rs_ar_first_stage_bw"], BLUE, "o--", "first prefill stage"))
        if steady:
            panels.append(("Steady all-reduce\neff. bw (GB/s)", steady))
        panels.append(("TTFT (ms)", [(s["ttft_ns"] * MS, BLUE, "o-", None)]))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, sharex=True,
                                 figsize=(8.5, 2.0 * n + 1.0))
        axes = np.atleast_1d(axes)
        for i, (ylabel, curves) in enumerate(panels):
            a = axes[i]
            for series, color, style, label in curves:
                a.plot(x, series, style, color=color, label=label)
            zoom_y(a, pd.concat([c[0] for c in curves]))
            a.set_ylabel(ylabel, fontsize=9)
            a.grid(True, alpha=0.3, which="both")
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            if any(c[3] for c in curves):
                a.legend(fontsize=7, loc="best")
            if i != n - 1:
                a.set_xlabel("")
        fig.suptitle("Does PP skew propagate into TTFT?", y=0.99)
        save_fig(fig, outdir, "01_causal_chain_to_ttft.png", written)

    # 02 KV CUMULATIVE ARRIVAL PER DECODE RANK ------------------------------- #
    ranks = sorted({d for r in runs for d in r.kv_rank_series})
    if ranks:
        cmap = plt.get_cmap("tab10")
        ncols = len(runs)
        fig, axes = plt.subplots(1, ncols, figsize=(max(3.0 * ncols, 6), 4.6),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for j, r in enumerate(runs):
            a = axes[j]
            for i, d in enumerate(ranks):
                if d not in r.kv_rank_series:
                    continue
                t, cum = r.kv_rank_series[d]
                total = cum[-1] if len(cum) else 1.0
                a.step(t * MS, 100 * cum / total, where="post",
                      color=cmap(i % 10), label=f"rank {d}")
            if pd.notna(r.kv_gate_ns):
                a.axvline(r.kv_gate_ns * MS, color="k", linestyle=":", alpha=0.5)
            a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
            a.set_xlabel("Time (ms)", fontsize=8)
            a.grid(True, alpha=0.3)
        axes[0].set_ylabel("KV arrived (% of total)")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle("Cumulative KV arrival per decode rank", y=1.02)
        save_fig(fig, outdir, "02_kv_cumulative_arrival_per_rank.png", written)

    # 03 LINK BANDWIDTH & CONCURRENCY, RANKED -------------------------------- #
    if chosen_labels:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
        cmap = plt.get_cmap("tab10")
        for i, label in enumerate(chosen_labels):
            eff_col, conc_col = f"link{i}_eff_pct", f"link{i}_conc_peak"
            if eff_col not in s.columns or not s[eff_col].notna().any():
                continue
            lw = 2.6 if i == 0 else 1.2
            lbl = f"{label}" + ("  (measured bottleneck)" if i == 0 else "")
            axL.plot(x, s[eff_col], marker="o", lw=lw, color=cmap(i % 10), label=lbl)
            if conc_col in s.columns and s[conc_col].notna().any():
                axR.plot(x, s[conc_col], marker="s", lw=lw, color=cmap(i % 10), label=label)
        logx_pow2(axL, s, "buffer_mb", "Per-switch buffer (MiB)")
        axL.set_ylabel("KV bandwidth (% of nominal)")
        axL.set_title("Delivered KV bandwidth per link")
        axL.grid(True, alpha=0.3, which="both")
        axL.legend(fontsize=7)
        logx_pow2(axR, s, "buffer_mb", "Per-switch buffer (MiB)")
        axR.set_ylabel("Peak concurrent KV flows")
        axR.set_title("Peak concurrency per link")
        axR.grid(True, alpha=0.3, which="both")
        axR.legend(fontsize=7)
        fig.suptitle("Congestion across every KV-crossed link", y=1.02)
        save_fig(fig, outdir, "03_link_bandwidth_and_concurrency.png", written)

    # 04 BOTTLENECK BUFFER OCCUPANCY(t) AS % OF THE SWEPT BUFFER, PFC PAUSES -- #
    # One column per buffer value: how full the bottleneck switch's buffer is,
    # as a % of BUFFER_SIZE, with PFC PAUSE spans shaded on top -- the pauses
    # line up with a full buffer. (The old top row, arrival-binned KV
    # throughput, is gone: arrival-time attribution was too coarse to read as
    # a bandwidth.)
    bn_sw = int(rows[0].bottleneck.split("->")[0])
    occ_runs = [r for r in runs if bn_sw in r.qseries and r.buffer_bytes]
    if occ_runs:
        ncols = len(occ_runs)
        fig, axes = plt.subplots(1, ncols, squeeze=False, sharex=True, sharey=True,
                                 figsize=(max(3.0 * ncols, 6), 3.6))
        for j, r in enumerate(occ_runs):
            a = axes[0][j]
            ts, ys = r.qseries[bn_sw]
            occ = np.asarray(ys) / r.buffer_bytes * 100.0
            a.fill_between(np.asarray(ts) * MS, occ, color=VIOLET, alpha=0.5, step="mid")
            a.plot(np.asarray(ts) * MS, occ, color=VIOLET, lw=0.6, alpha=0.8)
            for s0, e0 in r.bn_pause_intervals:
                a.axvspan(s0 * MS, e0 * MS, color=CORAL, alpha=0.25, lw=0)
            a.set_ylim(0, 100)
            a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
            a.set_xlabel("Time (ms)", fontsize=8)
            a.grid(True, alpha=0.3)
        axes[0][0].set_ylabel(f"Buffer occupancy\nswitch {bn_sw} (% of buffer)", fontsize=9)
        fig.suptitle(f"Bottleneck buffer occupancy ({rows[0].bottleneck})"
                     "  —  shaded = PFC PAUSE", y=1.02)
        save_fig(fig, outdir, "04_buffer_occupancy_and_pauses.png", written)

    # 05 QUEUE OCCUPANCY(t) PER SWITCH, WITH PFC PAUSES ---------------------- #
    switches = sorted({sw for r in rows for sw in r.qseries})
    bn_sw = int(rows[0].bottleneck.split("->")[0])
    if switches:
        cmap = plt.get_cmap("viridis")
        bufs = [r.buffer_mb for r in runs]
        cnorm = (matplotlib.colors.LogNorm(vmin=min(bufs), vmax=max(bufs))
                if len(set(bufs)) > 1 else None)
        nrows, ncols = len(switches), len(runs)
        fig, axes = plt.subplots(
            nrows, ncols, squeeze=False, sharex=True, sharey="row",
            figsize=(max(2.1 * ncols + 1.6, 6), max(1.7 * nrows + 1.0, 4)))
        for i, sw in enumerate(switches):
            for j, r in enumerate(runs):
                a = axes[i][j]
                if sw in r.qseries:
                    ts, ys = r.qseries[sw]
                    col = cmap(cnorm(r.buffer_mb)) if cnorm else BLUE
                    t, y = np.asarray(ts) * MS, np.asarray(ys) / 1e3
                    a.fill_between(t, y, color=col, alpha=0.85, lw=0)
                    a.plot(t, y, color="#222222", lw=0.5, alpha=0.6)
                # a full-height translucent span gets visually swallowed by an
                # opaque, tall queue fill exactly where it matters most (pauses
                # correlate with high occupancy) -- a top ribbon in axes-
                # fraction y stays visible regardless of the fill underneath.
                for s0, e0 in r.pause_intervals.get(sw, []):
                    a.axvspan(s0 * MS, e0 * MS, ymin=0.88, ymax=1.0,
                             transform=a.get_xaxis_transform(),
                             color=CORAL, alpha=0.9, lw=0)
                a.grid(True, alpha=0.2)
                if i == 0:
                    a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
                if j == 0:
                    mark = "\n(bottleneck)" if sw == bn_sw else ""
                    a.set_ylabel(f"switch {sw}{mark}\n(kB)", fontsize=8)
                if i == nrows - 1:
                    a.set_xlabel("Time (ms)", fontsize=8)
                    a.locator_params(axis="x", nbins=4)
        fig.suptitle("Queue occupancy over time — rows = switch, columns = buffer",
                    y=1.01)
        save_fig(fig, outdir, "05_queue_occupancy_timeseries_with_pauses.png", written)

    # 06 OCCUPANCY vs BUFFER, PER SWITCH (% of capacity) --------------------- #
    if switches:
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap2 = plt.get_cmap("tab10")
        for i, sw in enumerate(switches):
            xs, peak, mean = [], [], []
            for r in runs:
                if sw not in r.qswitch_peak or not r.buffer_bytes:
                    continue
                xs.append(r.buffer_mb)
                peak.append(100 * r.qswitch_peak[sw] / r.buffer_bytes)
                mean.append(100 * r.qswitch_mean[sw] / r.buffer_bytes)
            if not xs:
                continue
            c = cmap2(i % 10)
            tag = " (bottleneck)" if sw == bn_sw else ""
            ax.plot(xs, peak, "o-", color=c, label=f"switch {sw}{tag} — peak")
            ax.plot(xs, mean, "v--", color=c, alpha=0.5, label=f"switch {sw}{tag} — mean")
        if ax.get_legend_handles_labels()[0]:
            _finish(fig, ax, s, outdir, "06_queue_occupancy_vs_buffer.png",
                    "Is the extra buffer actually used?",
                    "Queue occupancy (% of buffer)", written)
        else:
            plt.close(fig)

    # 07 PFC PAUSE-FRAME COUNT vs BUFFER ------------------------------------- #
    # Just the raw number of PAUSE frames the bottleneck's ingress received,
    # one point per buffer value. Plots 04/05 show WHEN pauses happen; this is
    # the single scalar "how many", so the buffer -> backpressure trend reads
    # off one line. link0 is the measured bottleneck (see chosen_labels[0]).
    if "link0_pause_frames" in s.columns and s["link0_pause_frames"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, s["link0_pause_frames"], "o-", color=CORAL,
               label="PFC PAUSE frames (bottleneck)")
        _finish(fig, ax, s, outdir, "07_pfc_pause_frame_count.png",
                "PFC PAUSE frames at the bottleneck",
                "PAUSE frames (count)", written)

    # 08 KV ARRIVAL SKEW WITHIN EACH TP GROUP -------------------------------- #
    # One skew per (decode stage, layer): the spread between the arrivals of
    # the KV shards feeding the same TP group -- exactly the wait that layer's
    # first decode all-reduce inherits. min/mean/p99 of that population per
    # buffer value.
    if s["kv_tp_skew_mean_ns"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        for col, color, style, label in (
                ("kv_tp_skew_min_ns", GREEN, "v--", "min"),
                ("kv_tp_skew_mean_ns", BLUE, "o-", "mean"),
                ("kv_tp_skew_p99_ns", CORAL, "s-.", "p99")):
            if s[col].notna().any():
                ax.plot(x, s[col] * MS, style, color=color, label=label)
        n = int(s["kv_tp_skew_n"].max())
        _finish(fig, ax, s, outdir, "08_kv_tp_group_skew.png",
                f"KV arrival skew within each TP group "
                f"({n} (stage, layer) groups)",
                "Cross-shard arrival skew (ms)", written)

    # 09 KV TRANSFER TIME AND FIRST/LAST-LAYER GAP, PER DECODE RANK ---------- #
    # Left: each rank's total KV transfer time (first->last arrival). Right:
    # the SIGNED gap between the completion of the rank's first and last layer
    # -- layers arrive out of order, so this is not the span, and a negative
    # value means the last layer landed before the first.
    ranks_kv = sorted({d for r in runs for d in r.kv_rank_span})
    if ranks_kv:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.get_cmap("tab10")
        for i, d in enumerate(ranks_kv):
            c = cmap(i % 10)
            xs = [r.buffer_mb for r in runs if d in r.kv_rank_span]
            ys = [r.kv_rank_span[d] * MS for r in runs if d in r.kv_rank_span]
            axL.plot(xs, ys, "o-", color=c, label=f"rank {d}")
            xs = [r.buffer_mb for r in runs if d in r.kv_rank_layer_delta]
            ys = [r.kv_rank_layer_delta[d] * MS for r in runs
                  if d in r.kv_rank_layer_delta]
            if xs:
                axR.plot(xs, ys, "s--", color=c, label=f"rank {d}")
        for a, ttl, yl in ((axL, "Total KV transfer time per rank",
                            "Last − first KV arrival (ms)"),
                           (axR, "Completion gap: last vs first layer",
                            "compl(last layer) − compl(first layer) (ms)")):
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            a.set_title(ttl)
            a.set_ylabel(yl)
            a.grid(True, alpha=0.3, which="both")
            a.legend(fontsize=8)
        fig.suptitle("KV delivery per decode rank", y=1.02)
        save_fig(fig, outdir, "09_kv_transfer_time_per_rank.png", written)

    # 10 SKEW INHERITED BY THE DECODE STAGES' FIRST ALL-REDUCE --------------- #
    # Per decode PP stage: the FIRST TP all-reduce is gated by that stage's own
    # KV transfer, so whatever KV skew survived the gate shows up twice. LEFT:
    # its ENTRY SKEW -- the spread of the shards' start ticks -- one line per
    # stage. RIGHT: its duration against the mean duration of the stage's
    # remaining all-reduces (the steady-state control): the early shard sits
    # inside the collective waiting for the late one, so inherited skew reads
    # as first-AR duration above the steady mean. If both stay flat while
    # figure 08 moves, the KV skew is being masked (e.g. the transfer outruns
    # the deprioritised FIRSTTOK path).
    dec_stages = sorted({st for r in runs for st in r.dec_ar})
    if dec_stages:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        cmap = plt.get_cmap("tab10")
        US = 1e-3                                       # ns -> µs
        for i, st in enumerate(dec_stages):
            c = cmap(i % 10)
            have = [r for r in runs if st in r.dec_ar]
            xs = [r.buffer_mb for r in have]
            axL.plot(xs, [r.dec_ar[st]["first_skew_ns"] * US for r in have],
                     "o-", color=c, label=f"stage {st}")
            axR.plot(xs, [r.dec_ar[st]["first_dur_ns"] * US for r in have],
                     "o-", color=c, label=f"stage {st} — first AR")
            axR.plot(xs, [r.dec_ar[st]["rest_dur_mean_ns"] * US for r in have],
                     "v--", color=c, alpha=0.5, label=f"stage {st} — mean of the rest")
        for a, ttl, yl in ((axL, "Entry skew of the FIRST all-reduce",
                            "Entry skew (µs)"),
                           (axR, "Duration: first AR vs mean of the rest",
                            "Duration (µs)")):
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            a.set_title(ttl)
            a.set_ylabel(yl)
            a.grid(True, alpha=0.3, which="both")
            a.legend(fontsize=8)
        fig.suptitle("Skew inherited by each decode stage's first TP all-reduce",
                     y=1.02)
        save_fig(fig, outdir, "10_decode_first_allreduce_skew.png", written)

    # 11 HOW MUCH THE DECODE IS STALLED BY ITS KV ---------------------------- #
    # Left: the user-visible gap, token 2 after token 1 (includes the FIRSTTOK
    # message's own transit, which queues behind the KV bulk). Middle: the
    # first decode pass, decode start -> token 2, against the steady-state
    # inter-token gap -- everything above the dashed control is time the
    # pipeline spent waiting for KV. Right: the cause per stage: KV readiness
    # minus the stage's first-input arrival; above 0 the stage outruns its KV
    # and stalls, below 0 the transfer was already complete (fully masked).
    if s["tok2_latency_ns"].notna().any():
        fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8))
        if s["tok2_after_tok1_ns"].notna().any():
            axA.plot(x, s["tok2_after_tok1_ns"] * MS, "o-", color=BLUE)
        axA.set_title("Token 2 after token 1")
        axA.set_ylabel("t(token 2) − TTFT (ms)")

        axB.plot(x, s["tok2_latency_ns"] * MS, "o-", color=CORAL,
                 label="first decode pass")
        if s["itl_steady_ns"].notna().any():
            axB.plot(x, s["itl_steady_ns"] * MS, "v--", color=MUTED,
                     label="steady inter-token gap")
        axB.set_title("First decode pass vs steady state")
        axB.set_ylabel("Decode start → token 2 (ms)")

        dec_st = sorted({st for r in runs for st in r.dec_stall})
        cmap = plt.get_cmap("tab10")
        for i, st in enumerate(dec_st):
            have = [r for r in runs if st in r.dec_stall]
            axC.plot([r.buffer_mb for r in have],
                     [r.dec_stall[st]["kv_lateness_ns"] * MS for r in have],
                     "s-", color=cmap(i % 10), label=f"stage {st}")
        axC.axhline(0.0, color="k", linestyle=":", alpha=0.5)
        axC.set_title("KV lateness at each stage's first input")
        axC.set_ylabel("KV ready − input arrival (ms)")

        for a in (axA, axB, axC):
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            a.grid(True, alpha=0.3, which="both")
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        fig.suptitle("How much the decode is stalled by its KV transfer", y=1.02)
        save_fig(fig, outdir, "11_decode_kv_stall.png", written)

    return written


def decode_worst_stage(s: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-stage decN_* columns (figures 10 and 11) to the WORST
    decode stage, plus the run-wide decode-stall scalars, so the
    cross-model/cross-level tools have one stage-count-independent number per run:

        dec_ar_first_skew_ns    max over stages of the entry skew into the
                                stage's FIRST (KV-gated) TP all-reduce (fig 10).
        dec_ar_first_over_rest  max over stages of first duration / that same
                                stage's steady-state mean -- self-normalised per
                                stage, so PP splits and model scale cancel (fig 10).
        dec_kv_lateness_ns      max over stages of (KV ready - first-input
                                arrival): how late the KV is at the stage that
                                waits most for it (fig 11, right panel).
        decode_kv_stall_ns      tok2_latency - itl_steady: the first decode
                                pass's excess over the steady inter-token gap,
                                i.e. the wall-clock the pipeline actually spent
                                stalled on KV in the first pass (fig 11, middle).

    Stage numbering differs between models (different PP), which is why the
    reduction happens here, once, and not in each caller. NaN columns when the
    sweep has no decode TP all-reduce (TP=1) or no DECFB (no second token)."""
    stages = sorted(int(m.group(1)) for c in s.columns
                    if (m := re.fullmatch(r"dec(\d+)_ar_first_skew_ns", c)))
    fsk = [s[f"dec{st}_ar_first_skew_ns"] for st in stages]
    s["dec_ar_first_skew_ns"] = (pd.concat(fsk, axis=1).max(axis=1)
                                 if fsk else NAN)
    ratios = [s[f"dec{st}_ar_first_dur_ns"] / s[f"dec{st}_ar_rest_dur_mean_ns"]
              for st in stages]
    s["dec_ar_first_over_rest"] = (pd.concat(ratios, axis=1).max(axis=1)
                                   if ratios else NAN)

    lat_stages = sorted(int(m.group(1)) for c in s.columns
                        if (m := re.fullmatch(r"dec(\d+)_kv_lateness_ns", c)))
    lat = [s[f"dec{st}_kv_lateness_ns"] for st in lat_stages]
    s["dec_kv_lateness_ns"] = (pd.concat(lat, axis=1).max(axis=1)
                               if lat else NAN)
    if "tok2_latency_ns" in s.columns and "itl_steady_ns" in s.columns:
        s["decode_kv_stall_ns"] = s["tok2_latency_ns"] - s["itl_steady_ns"]
    else:
        s["decode_kv_stall_ns"] = NAN
    return s


# --------------------------------------------------------------------------- #
REPORT = ["buffer_mb", "ttft_ns", "kv_gate_ns", "kv_gate_over_ttft",
          "pp_skew_ns", "rs_ar_first_ns", "rs_ar_rest_mean_ns", "cross_rank_skew_ns",
          "kv_tp_skew_mean_ns", "kv_tp_skew_p99_ns",
          "tok2_latency_ns", "itl_steady_ns",
          "link0_label", "link0_eff_pct", "link0_conc_peak", "link0_pause_frames"]


def analyse_sweep(p: paths.SweepPaths, placement: Placement,
                  top_links: int = 6, bn_force: str | None = None,
                  verbose: bool = True,
                  want_series: bool = True,
                  tags: list[str] | None = None) -> tuple[list[Row], pd.DataFrame, list[str]]:
    """Score one workload's whole buffer sweep, exactly as buffer_sweep.main
    does -- factored out so buffer_compare gets identical numbers (one
    definition of the metrics, two tools). Returns (rows sorted by buffer, the flat
    summary DataFrame, chosen_labels). Raises Abort on any condition main would.
    Does no I/O beyond reading the sweep; writing figures/CSV stays in the
    caller. `verbose` gates the progress prints so a multi-workload caller stays
    quiet.

    `tags` restricts the analysis to a subset of the sweep's run dirs (default:
    all of p.tags('ns3')). A 2D sweep (e.g. oversubscription x buffer) shares one
    ns3 dir but moves two knobs, so oversub2d_sweep calls this once per
    oversubscription level with that level's buffer tags -- keeping the
    single-variant (one buffer axis) contract intact within each call."""
    need(not p.missing_roots(),
         "derived root(s) do not exist:\n    "
         + "\n    ".join(p.missing_roots())
         + f"\n  --sweep {p.sweep!r} is probably wrong.")
    tags = tags if tags is not None else p.tags("ns3")
    need(tags, f"no run sub-directory under {p.ns3_root}")

    # Config INPUTS and the swept-axis token are what the analysis reasons FROM:
    # any of them missing on any run is an Abort, before touching a single file.
    miss_cfg = [f"{t}: missing {f}" for t in tags
                for f in (p.topology(t), p.config(t)) if not f.is_file()]
    need(not miss_cfg, "config input(s) missing -- fix the configs (or prune "
                       "the stale ns-3 output):\n    " + "\n    ".join(miss_cfg))
    for t in tags:
        need(BUFFER_AXIS.value(t) is not None,
             f"{t}: no 'buf<num>' token in the directory name; the swept axis "
             f"is unreadable.")

    # OUTPUTS are what the simulations write: a run still missing one has just
    # not finished (or died) -- skip it with a name, and analyse the rest.
    usable = []
    for t in tags:
        nd = p.ns3_run(t)
        missing = [n for n, ok in (
            ("fct.txt", (nd / "fct.txt").is_file()),
            ("pfc.txt", (nd / "pfc.txt").is_file()),
            ("qlen.txt", (nd / "qlen.txt").is_file()),
            ("stats_sys*.csv", any(p.astra_run(t).glob("stats_sys*.csv")))) if not ok]
        if missing:
            warn(f"{t}: output(s) missing ({', '.join(missing)}) -- run skipped.")
        else:
            usable.append(t)
    need(usable, "no run has all its outputs on disk yet.")
    tags = usable

    if verbose:
        print(p.describe())
        print(f"  placement\n{placement.describe()}\n")
    if msg := roles.cross_check(placement, p.astra_run(tags[0])):
        warn(msg)

    variants = {BUFFER_AXIS.variant(t) for t in tags}
    need(len(variants) == 1,
         f"this sweep moves more than one knob: variants {sorted(variants)}. "
         f"Split into one sweep per variant.")

    # -- fix the link SET and its display order once, from the smallest
    #    buffer (most congested, least likely to be a near-tie). Every run
    #    then reports THESE labels, in THIS order -- see the module docstring
    #    on why the set itself cannot vary within one sweep. ------------- #
    ref_tag = min(tags, key=lambda t: BUFFER_AXIS.value(t))
    ref_topo = parse_topology(p.topology(ref_tag))
    ref_cfg = parse_ns3_config(p.config(ref_tag))
    ref_raw = ns3.read_fct(p.ns3_run(ref_tag) / "fct.txt")
    need(ref_raw is not None and len(ref_raw),
         f"{ref_tag}: fct.txt has no parsable rows; cannot fix the link set.")
    ref_f = flowlib.annotate(ref_raw, ref_topo, placement, ref_cfg.payload)
    ref_kv = ref_f[ref_f["flow_class"] == "kv"]
    need(len(ref_kv), f"{ref_tag}: no KV flow after classification.")
    ref_qlen = ns3.read_qlen(p.ns3_run(ref_tag) / "qlen.txt", series=False)
    need(ref_qlen is not None and ref_qlen.port_max,
         f"{ref_tag}: qlen.txt has no samples.")
    canonical = flowlib.candidate_links(ref_topo, ref_qlen.port_max, ref_kv)
    need(canonical, f"{ref_tag}: no link is crossed by any KV flow -- "
                    f"classification or topology is wrong.")

    if bn_force:
        sw, peer = (int(x) for x in bn_force.split("->"))
        idx = next((i for i, l in enumerate(canonical)
                   if l.switch == sw and l.peer == peer), None)
        need(idx is not None,
             f"--bottleneck {bn_force}: not among the links this sweep's KV "
             f"flows cross ({[str(l) for l in canonical]}).")
        canonical.insert(0, canonical.pop(idx))

    chosen_labels = [str(l) for l in canonical[:top_links]]
    if verbose:
        print(f"link set ({len(canonical)} total, top {len(chosen_labels)} kept):")
        for lab in chosen_labels:
            print(f"  - {lab}")
        print(f"\nAnalysing {len(tags)} runs:")

    rows = [analyse(t, p, placement, chosen_labels, want_series) for t in tags]
    if verbose:
        for r in sorted(rows, key=lambda r: r.buffer_mb):
            print(f"  + buf{r.buffer_mb:<5g} bn={r.bottleneck:<8} "
                  f"ttft={r.ttft_ns*MS:6.2f}ms gate={r.kv_gate_ns*MS:6.2f}ms  "
                  f"pp_skew={r.pp_skew_ns/1e3 if pd.notna(r.pp_skew_ns) else float('nan'):7.2f}us  "
                  f"eff={r.links[0].eff_pct:5.1f}%")

    bns = {r.bottleneck for r in rows}
    need(len(bns) == 1,
         f"the top-ranked link is not the same on every run: {sorted(bns)}. "
         f"Pass --bottleneck to fix one explicitly.")

    rows = sorted(rows, key=lambda r: r.buffer_mb)
    s = (pd.DataFrame([r.flat() for r in rows])
         .sort_values("buffer_mb").reset_index(drop=True))
    s = decode_worst_stage(s)
    return rows, s, chosen_labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    for act in ap._actions:
        if act.dest == "out":
            act.help = "output dir (default: results/sweep_analysis/" \
                       f"{KIND}/<workload>/<sweep>)"
    roles.add_argument(ap)
    ap.add_argument("--bottleneck", default=None,
                    help="'sw->peer', e.g. '8->12'. Must be among the links "
                         "this sweep's KV flows cross. Default: the deepest "
                         "queue among them, measured on the smallest-buffer run.")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links figure 03 and summary.csv "
                         "carry (default: 6)")
    a = ap.parse_args(argv)

    try:
        p = paths.SweepPaths(sweep=a.sweep, workload=a.workload, root=Path(a.root))
        outdir = (Path(a.out) if a.out else
                  p.root / "results" / "sweep_analysis" / KIND / p.workload / p.sweep)
        placement = Placement.parse(a.placement)
        print(f"  out      {outdir}")
        rows, s, chosen_labels = analyse_sweep(
            p, placement, top_links=a.top_links, bn_force=a.bottleneck, verbose=True)

        fresh_dir(outdir)
        s.to_csv(outdir / "summary.csv", index=False)
        plots = make_plots(rows, s, outdir, chosen_labels)

        pd.set_option("display.width", 220)
        print("\n================ BUFFER SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))
        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", *[q.name for q in plots]]:
            print(f"  {fpath}")
        return drain_warnings(" — the numbers above are conditional on them")
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
