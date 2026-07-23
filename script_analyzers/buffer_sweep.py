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

Seven figures, each answering one part of that:

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
    04  THROUGHPUT / BUFFER(t)    binned KV throughput(t) at the top-ranked link
                                   (top row) over that switch's buffer occupancy
                                   (bottom row, % of BUFFER_SIZE), one column per
                                   buffer value, PFC PAUSE spans shaded on both --
                                   throughput dips line up with a full buffer.
    05  QUEUE(t) PER SWITCH        a grid (rows=switch, cols=buffer), with PFC
                                   PAUSE spans shaded.
    06  OCCUPANCY vs BUFFER        is the added buffer actually used, per switch.
    07  TTFT-NORMALISED COST       decode-start and its "fabric tax" as
                                   multiples of TTFT rather than absolute ms,
                                   since TTFT itself now moves with the buffer
                                   (via the skew chain) instead of being flat.
    08  PFC COUNT vs BUFFER        the raw PAUSE-frame count at the bottleneck,
                                   one point per buffer -- just the "how many".

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
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import astra, intervals
from utils import flows as flowlib
from utils import ns3, paths, pp, roles
from utils.cli import Abort, need
from utils.fabric import Bottleneck, Topology, parse_ns3_config, parse_topology
from utils.plots import downsample_max, logx_pow2, save_fig
from utils.roles import Placement
from utils.paths import BUFFER_AXIS

NAN = float("nan")
KIND = "buffer"
MS = 1e-6                     # ns -> ms

BLUE, CORAL, GREEN, VIOLET, MUTED = \
    "#1f77b4", "#d1495b", "#2b8a3e", "#6a4c93", "#9aa0a6"


# --------------------------------------------------------------------------- #
WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  ! {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# One row per link, one row per run.
# --------------------------------------------------------------------------- #
@dataclass
class LinkStat:
    """One candidate KV-crossed link's stats for one run -- scored the way a
    single bottleneck is (window/floor/efficiency, queue occupancy, PFC pauses),
    plus concurrency, which a single-link analysis never needed with only one
    link to look at."""
    label: str = ""
    switch: int = -1
    egress_port: int = -1
    peer: int = -1
    rate_gbps: float = NAN
    f_ports: int = 0
    kv_bytes: float = NAN
    window_ns: float = NAN
    floor_ns: float = NAN
    delivered_gbps: float = NAN
    eff_pct: float = NAN                  # floor/window, %: delivered vs the hard floor
    qpeak_bytes: float = NAN
    qmean_bytes: float = NAN
    qpeak_pct: float = NAN                # qpeak_bytes / this run's buffer_bytes, %
    conc_peak: float = NAN                # most concurrent KV flows at once
    conc_mean: float = NAN                # mean concurrency each flow actually saw
    pause_frames: float = NAN
    pause_pct_of_window: float = NAN


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

    # -- not flattened: per-figure raw data ----------------------------------- #
    links: list = field(default_factory=list)          # list[LinkStat]
    kv_rank_series: dict = field(default_factory=dict)  # rank -> (times_ns, cumbytes)
    bn_throughput_series: object = None                 # (centres_ns, gbps)
    bn_concurrency_series: object = None                # (times_ns, counts)
    bn_pause_intervals: list = field(default_factory=list)
    qseries: dict = field(default_factory=dict)         # sw -> (ts_ns, bytes)
    qswitch_peak: dict = field(default_factory=dict)     # sw -> peak total bytes
    qswitch_mean: dict = field(default_factory=dict)     # sw -> mean total bytes
    pause_intervals: dict = field(default_factory=dict)  # sw -> [(start,end)]

    def flat(self) -> dict:
        d = asdict(self)
        for k in ("links", "kv_rank_series", "bn_throughput_series",
                  "bn_concurrency_series", "bn_pause_intervals", "qseries",
                  "qswitch_peak", "qswitch_mean", "pause_intervals"):
            d.pop(k, None)
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
# Measurement helpers
# --------------------------------------------------------------------------- #
def union_len(spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Covered length of the union of `spans` clipped to [lo, hi]. The union
    algebra is utils.intervals; only the clip window is this function's own."""
    clipped = [(max(s, lo), min(e, hi)) for s, e in spans if min(e, hi) > max(s, lo)]
    return int(intervals.union_len(clipped))


def pause_stats(pfc: ns3.PfcLog, bn: Bottleneck, topo: Topology,
                lo: int, hi: int) -> dict:
    """Backpressure on the ingress side of `bn`, over [lo, hi]. Kept
    link-generic so it works for every candidate link, not only one."""
    span = max(hi - lo, 1)
    victims = set(bn.pause_victims(topo))

    frames_bn = frames_total = 0
    devices_paused: set[tuple[int, int]] = set()
    for (node, _nt, ifidx, _q), events in pfc.events.items():
        n_in_win = sum(1 for t, typ in events if typ == 1 and lo <= t <= hi)
        frames_total += n_in_win
        if (node, ifidx) in victims:
            frames_bn += n_in_win
            if n_in_win:
                devices_paused.add((node, ifidx))

    iv = pfc.pause_intervals(clamp_to=hi)
    per_dev: dict[tuple[int, int], list] = {}
    for (node, _nt, ifidx, _q), spans in iv.items():
        if (node, ifidx) in victims:
            per_dev.setdefault((node, ifidx), []).extend(spans)
    pct = 0.0
    if per_dev:
        best = max(per_dev, key=lambda k: union_len(per_dev[k], lo, hi))
        pct = 100.0 * union_len(per_dev[best], lo, hi) / span

    return {"pause_frames_bn": float(frames_bn),
            "pause_frames_total": float(frames_total),
            "paused_devices": float(len(devices_paused)),
            "pause_pct_of_window": pct}


def victim_pause_intervals(pfc: ns3.PfcLog, bn: Bottleneck, topo: Topology,
                           clamp_to: int) -> list[tuple[int, int]]:
    """Raw PAUSE intervals (not unioned) on `bn`'s ingress victims, for
    shading a timeline. pause_stats reduces the same population to one %
    number; this keeps the intervals themselves."""
    victims = set(bn.pause_victims(topo))
    out: list[tuple[int, int]] = []
    for (node, _nt, ifidx, _q), spans in pfc.pause_intervals(clamp_to=clamp_to).items():
        if (node, ifidx) in victims:
            out.extend(spans)
    return out


def barrier(kv: pd.DataFrame, placement: Placement) -> dict:
    """The first decode step cannot start until every KV flow feeding a decode
    rank has arrived."""
    out = {"decode_ranks": ",".join(map(str, placement.decode_ranks))}
    ready, dur = {}, {}
    for d in placement.decode_ranks:
        arr = kv.loc[kv["dst"] == d, "arrival"]
        if len(arr):
            ready[d] = float(arr.max())
            dur[d] = float(arr.max() - arr.min())
    need(ready, f"no KV flow arrives at any declared decode rank "
                f"{placement.decode_ranks}: --placement is wrong.")
    if len(ready) < len(placement.decode_ranks):
        warn(f"only {len(ready)}/{len(placement.decode_ranks)} decode ranks "
             f"receive KV; the barrier is over {sorted(ready)}.")
    out["kv_gate_ns"] = max(ready.values())
    out["kv_ready_min_ns"] = min(ready.values())
    out["cross_rank_skew_ns"] = max(ready.values()) - min(ready.values())
    out["kv_stream_duration_ns"] = max(dur.values())
    return out


def ttft_end_of_prefill(tag: str, p: paths.SweepPaths) -> dict:
    """TTFT = the first token, produced at the END OF PREFILL (FIRSTTOK send,
    NOT DECFB -- DECFB is the second token, one decode pipeline late)."""
    adir = p.astra_run(tag)
    if not adir.is_dir():
        warn(f"{tag}: no ASTRA run at {adir}; TTFT (end of prefill) unavailable.")
        return {}
    df = astra.read_run(adir)
    if df is None:
        warn(f"{tag}: no readable stats_sys*.csv under {adir}; TTFT unavailable.")
        return {}
    inst = astra.firsttok_send_instant(df)
    if inst is not None:
        return {"ttft_ns": inst}
    pre = df.loc[(df["op_class"] == "COMP") & (df["phase"] == "prefill"),
                 "end_tick"]
    if len(pre):
        warn(f"{tag}: no FIRSTTOK in the ASTRA trace; using the last prefill "
             f"compute end as end-of-prefill TTFT.")
        return {"ttft_ns": float(pre.max())}
    warn(f"{tag}: no FIRSTTOK and no prefill COMP in the ASTRA trace; TTFT "
         f"unavailable.")
    return {}


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

    Stages are the ASTRA `ss` field. All NaN/0 with no ASTRA run, no wave (PP=1),
    or no prefill TP (TP=1)."""
    out = {"rs_ar_first_ns": NAN, "rs_ar_rest_mean_ns": NAN, "rs_ar_n": 0,
           "rs_ar_first_bw": NAN, "rs_ar_rest_bw": NAN, "rs_ar_first_stage_bw": NAN}
    if adf is None or not ppr.available or ppr.stage is None:
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

    recv = g[g["ss"] == ppr.stage].sort_values("start")
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


def kv_rank_series(kv: pd.DataFrame, placement: Placement) -> dict:
    """rank -> (arrival_times_ns, cumulative_bytes), sorted by arrival. The
    raw material for figure 02: skew is the horizontal spread between ranks'
    curves, smoothness is whether each curve ramps or stair-steps with flats."""
    out = {}
    for d in placement.decode_ranks:
        sub = kv.loc[kv["dst"] == d].sort_values("arrival")
        if not len(sub):
            continue
        out[int(d)] = (sub["arrival"].to_numpy(dtype=float),
                       np.cumsum(sub["size"].to_numpy(dtype=float)))
    return out


def bn_time_series(kv_bn: pd.DataFrame, lo: int, hi: int,
                   n_buckets: int = 80) -> tuple[tuple, tuple]:
    """Binned aggregate KV throughput(t) (Gbit/s) and concurrent-flow count(t)
    at one link, over [lo, hi].

    Throughput is arrival-time attribution: each flow's bytes land in the
    bucket where it ARRIVES (fct.txt gives start+fct, not an intra-flow rate
    curve, so this is the finest honest resolution). A bucket with no arrivals
    reads as a stall even if bytes were in flight -- which is exactly the
    visual signature PFC pausing produces: transfers back up, then complete in
    a burst once RESUME lets them drain. That is the point of the figure."""
    if not len(kv_bn) or hi <= lo:
        return (np.array([]), np.array([])), (np.array([]), np.array([]))
    edges = np.linspace(lo, hi, n_buckets + 1)
    widths = np.diff(edges)
    bytes_per_bucket = np.zeros(n_buckets)
    arr = kv_bn["arrival"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(edges, arr) - 1, 0, n_buckets - 1)
    np.add.at(bytes_per_bucket, idx, kv_bn["size"].to_numpy(dtype=float))
    gbps = bytes_per_bucket * 8.0 / widths        # bytes/ns * 8 = bit/ns = Gbit/s
    centres = (edges[:-1] + edges[1:]) / 2
    t_conc, conc = flowlib.concurrency_series(flowlib.flow_spans(kv_bn))
    return (centres, gbps), (t_conc, conc)


def link_metrics(kv: pd.DataFrame, bn: Bottleneck, topo: Topology,
                 pfc: ns3.PfcLog, qlen: ns3.QlenLog, buffer_bytes: float) -> LinkStat:
    ls = LinkStat(label=str(bn), switch=bn.switch, egress_port=bn.egress_port,
                 peer=bn.peer, rate_gbps=bn.rate / 1e9, f_ports=bn.f_ports)
    kv_bn = kv[flowlib.crosses(kv, bn)]
    if not len(kv_bn):
        return ls
    lo, hi = int(kv_bn["start"].min()), int(kv_bn["arrival"].max())
    ls.window_ns = hi - lo
    ls.kv_bytes = float(kv_bn["size"].sum())
    ls.floor_ns = ls.kv_bytes * 8e9 / bn.rate
    if ls.window_ns > 0:
        ls.delivered_gbps = ls.kv_bytes * 8.0 / ls.window_ns
        ls.eff_pct = 100 * ls.floor_ns / ls.window_ns
    ls.qpeak_bytes = float(qlen.port_max.get((bn.switch, bn.egress_port), NAN))
    ls.qmean_bytes = float(qlen.port_mean.get((bn.switch, bn.egress_port), NAN))
    if buffer_bytes and pd.notna(ls.qpeak_bytes):
        ls.qpeak_pct = 100 * ls.qpeak_bytes / buffer_bytes
    ls.conc_peak, ls.conc_mean = flowlib.concurrency_stats(flowlib.flow_spans(kv_bn))
    pstats = pause_stats(pfc, bn, topo, lo, hi)
    ls.pause_frames = pstats["pause_frames_bn"]
    ls.pause_pct_of_window = pstats["pause_pct_of_window"]
    return ls


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyse(tag: str, p: paths.SweepPaths, placement: Placement,
           chosen_labels: list[str], want_series: bool = True) -> Row:
    buf = BUFFER_AXIS.value(tag)
    need(buf is not None, f"{tag}: no 'buf<num>' token in the directory name; "
                          f"the swept axis is unreadable.")
    tpath, cpath = p.topology(tag), p.config(tag)
    ns3_dir = p.ns3_run(tag)
    for fpath in (tpath, cpath, ns3_dir / "fct.txt", ns3_dir / "pfc.txt",
                 ns3_dir / "qlen.txt"):
        need(fpath.exists(), f"{tag}: missing {fpath}")

    topo = parse_topology(tpath)
    cfg = parse_ns3_config(cpath)
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    need(cfg.buffer_mb is not None,
         f"{tag}: no BUFFER_SIZE in {cpath}.")
    need(abs(cfg.buffer_mb - buf) < 1e-6,
         f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but 'buf{buf:g}' "
         f"in the directory name. One of the two is lying.")

    row = Row(tag=tag, buffer_mb=float(buf), buffer_bytes=float(buf) * 1024 * 1024)

    for k, v in ttft_end_of_prefill(tag, p).items():
        setattr(row, k, v)

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
    row.pause_intervals = dict(per_switch)
    need(row.links, f"{tag}: no candidate link to report.")

    bn = links_here.get(chosen_labels[0])
    need(bn is not None, f"{tag}: the top-ranked link {chosen_labels[0]} is "
                         f"not crossed by any KV flow in THIS run -- it cannot "
                         f"be treated as the bottleneck here.")
    row.bottleneck, row.bn_rate_gbps = str(bn), bn.rate / 1e9

    kv_bn = kv[flowlib.crosses(kv, bn)]
    need(len(kv_bn), f"{tag}: no KV flow crosses the top-ranked link {bn}.")
    lo, hi = int(kv_bn["start"].min()), int(kv_bn["arrival"].max())
    row.bn_throughput_series, row.bn_concurrency_series = bn_time_series(kv_bn, lo, hi)
    row.bn_pause_intervals = victim_pause_intervals(pfc, bn, topo, clamp_to=run_end)

    for k, v in barrier(kv, placement).items():
        setattr(row, k, v)
    row.kv_gate_over_ttft = (row.kv_gate_ns / row.ttft_ns
                             if pd.notna(row.ttft_ns) and row.ttft_ns > 0 else NAN)

    ppr = pp.measure(f, placement)
    if not ppr.available:
        warn(f"{tag}: no inter-stage PP-prefill flow found; the causal-chain "
             f"figure will be empty for this run. (PP=1, or placement has "
             f"one prefill stage.)")
    row.pp_skew_ns = ppr.skew_ns
    row.pp_skew_mean_ns = ppr.skew_mean_ns
    row.pp_first_ns = ppr.first_ns
    row.pp_last_ns = ppr.last_ns
    row.pp_stage = ppr.stage
    row.pp_n_waves = ppr.n_waves
    # All-reduce bandwidths come from the ASTRA stats CSV (per-collective duration
    # and bytes), not the ns-3 fabric flows -- see rs_allreduce_stats.
    adir = p.astra_run(tag)
    adf = astra.read_run(adir) if adir.is_dir() else None
    for k, v in rs_allreduce_stats(adf, placement, ppr).items():
        setattr(row, k, v)
    if ppr.available and pd.isna(row.rs_ar_first_bw):
        warn(f"{tag}: no prefill TP all-reduce in the ASTRA stats for the "
             f"receiving stage {ppr.stage}; all-reduce bandwidths unavailable.")

    row.kv_rank_series = kv_rank_series(kv, placement)

    for sw, (ts, ys) in qlen.switch_series.items():
        if len(ts) == 0:
            continue
        row.qseries[sw] = downsample_max(ts, ys, 2000)
        row.qswitch_peak[sw] = float(qlen.switch_total_max.get(sw, max(ys)))
        row.qswitch_mean[sw] = float(np.mean(ys))

    return row


def _zoom_y(ax, series, pad: float = 0.15) -> None:
    """Autoscale one panel's y-axis to its own data, with a small margin."""
    v = series.dropna()
    if v.empty:
        return
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        band = max(abs(hi) * 0.02, 0.5)
        ax.set_ylim(hi - band, hi + band)
    else:
        ax.set_ylim(lo - pad * span, hi + pad * span)


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
            _zoom_y(a, pd.concat([c[0] for c in curves]))
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

    # 04 KV THROUGHPUT(t) OVER BOTTLENECK BUFFER OCCUPANCY(t), PFC PAUSES ----- #
    # Two stacked rows sharing each column's time axis: TOP = KV throughput at
    # the bottleneck link, BOTTOM = how full that switch's buffer is (% of the
    # swept BUFFER_SIZE). PFC PAUSE spans are shaded across both, so a throughput
    # dip, the buffer that has backed up underneath it, and the pause that links
    # them line up vertically -- the direct picture of "stalls in step with a
    # full buffer / PAUSE". (The old thin twin-axis line was per-flow CONCURRENCY,
    # not throughput; it answered "how many flows" and only muddied this figure.)
    bn_runs = [r for r in runs if r.bn_throughput_series and len(r.bn_throughput_series[0])]
    if bn_runs:
        bn_sw = int(rows[0].bottleneck.split("->")[0])
        ncols = len(bn_runs)
        fig, axes = plt.subplots(2, ncols, squeeze=False, sharex="col", sharey="row",
                                 figsize=(max(3.0 * ncols, 6), 5.4),
                                 gridspec_kw={"height_ratios": [2, 1]})
        for j, r in enumerate(bn_runs):
            top, bot = axes[0][j], axes[1][j]
            # top: KV throughput at the bottleneck link
            t, gbps = r.bn_throughput_series
            top.fill_between(np.asarray(t) * MS, gbps, color=BLUE, alpha=0.6, step="mid")
            top.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
            top.grid(True, alpha=0.3)
            # bottom: bottleneck switch buffer occupancy, as % of the buffer size
            q = r.qseries.get(bn_sw)
            if q and r.buffer_bytes:
                ts, ys = q
                occ = np.asarray(ys) / r.buffer_bytes * 100.0
                bot.fill_between(np.asarray(ts) * MS, occ, color=VIOLET, alpha=0.5, step="mid")
                bot.plot(np.asarray(ts) * MS, occ, color=VIOLET, lw=0.6, alpha=0.8)
            bot.set_ylim(0, 100)
            bot.set_xlabel("Time (ms)", fontsize=8)
            bot.grid(True, alpha=0.3)
            # PFC PAUSE spans on both rows
            for s0, e0 in r.bn_pause_intervals:
                for a in (top, bot):
                    a.axvspan(s0 * MS, e0 * MS, color=CORAL, alpha=0.25, lw=0)
        axes[0][0].set_ylabel("KV throughput\n(Gb/s)", fontsize=9)
        axes[1][0].set_ylabel(f"Buffer occupancy\nswitch {bn_sw} (% of buffer)", fontsize=9)
        fig.suptitle(f"KV throughput over bottleneck buffer occupancy ({rows[0].bottleneck})"
                     "  —  shaded = PFC PAUSE", y=1.01)
        save_fig(fig, outdir, "04_kv_throughput_and_pauses.png", written)

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

    # 07 TTFT-NORMALISED DECODE COST ----------------------------------------- #
    if s["kv_gate_over_ttft"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, s["kv_gate_over_ttft"], "o-", color=BLUE,
               label="decode start / TTFT")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.4)
        ax2 = ax.twinx()
        cost_frac = (s["kv_gate_ns"] - s["ttft_ns"]) / s["ttft_ns"]
        ax2.plot(x, cost_frac, "s--", color=CORAL,
                label="fabric tax")
        ax2.set_ylabel("Fabric tax (×TTFT)")
        _finish(fig, ax, s, outdir, "07_ttft_normalised_decode_cost.png",
                "Decode start relative to TTFT",
                "Decode start (×TTFT)", written, extra_axes=(ax2,))

    # 08 PFC PAUSE-FRAME COUNT vs BUFFER ------------------------------------- #
    # Just the raw number of PAUSE frames the bottleneck's ingress received,
    # one point per buffer value. Plots 04/05 show WHEN pauses happen; this is
    # the single scalar "how many", so the buffer -> backpressure trend reads
    # off one line. link0 is the measured bottleneck (see chosen_labels[0]).
    if "link0_pause_frames" in s.columns and s["link0_pause_frames"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, s["link0_pause_frames"], "o-", color=CORAL,
               label="PFC PAUSE frames (bottleneck)")
        _finish(fig, ax, s, outdir, "08_pfc_pause_frame_count.png",
                "PFC PAUSE frames at the bottleneck",
                "PAUSE frames (count)", written)

    return written


# --------------------------------------------------------------------------- #
REPORT = ["buffer_mb", "ttft_ns", "kv_gate_ns", "kv_gate_over_ttft",
          "pp_skew_ns", "rs_ar_first_ns", "rs_ar_rest_mean_ns", "cross_rank_skew_ns",
          "link0_label", "link0_eff_pct", "link0_conc_peak", "link0_pause_frames"]


def analyse_sweep(p: paths.SweepPaths, placement: Placement,
                  top_links: int = 6, bn_force: str | None = None,
                  verbose: bool = True,
                  want_series: bool = True) -> tuple[list[Row], pd.DataFrame, list[str]]:
    """Score one workload's whole buffer sweep, exactly as buffer_sweep.main
    does -- factored out so buffer_compare gets identical numbers (one
    definition of the metrics, two tools). Returns (rows sorted by buffer, the flat
    summary DataFrame, chosen_labels). Raises Abort on any condition main would.
    Does no I/O beyond reading the sweep; writing figures/CSV stays in the
    caller. `verbose` gates the progress prints so a multi-workload caller stays
    quiet."""
    need(not p.missing_roots(),
         "derived root(s) do not exist:\n    "
         + "\n    ".join(p.missing_roots())
         + f"\n  --sweep {p.sweep!r} is probably wrong.")
    tags = p.tags("ns3")
    need(tags, f"no run sub-directory under {p.ns3_root}")

    if verbose:
        print(p.describe())
        print(f"  placement\n{placement.describe()}\n")
    if (ad := p.astra_run(tags[0])).is_dir():
        if msg := roles.cross_check(placement, ad):
            warn(msg)
    else:
        warn(f"no ASTRA run at {ad}: --placement is taken on trust.")

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

        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        s.to_csv(outdir / "summary.csv", index=False)
        plots = make_plots(rows, s, outdir, chosen_labels)

        pd.set_option("display.width", 220)
        print("\n================ BUFFER SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))
        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", *[q.name for q in plots]]:
            print(f"  {fpath}")
        if WARNINGS:
            print(f"\n{len(WARNINGS)} WARNING(S) — the numbers above are "
                  f"conditional on them:")
            for w in WARNINGS:
                print(f"  ! {w}")
            return 1
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
