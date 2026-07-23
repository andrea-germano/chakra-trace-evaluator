#!/usr/bin/env python3
"""
incast_sweep — the incast variant of the fabric study: many prefill ranks
hand their KV cache to the few decode ranks at once, and the question is how
that INCAST skews KV arrival at the decode stages and what the per-switch buffer
does about it, across topologies of growing incast degree.

Same fabric, DIFFERENT emphasis from buffer_sweep. buffer_sweep held one
topology and chased a PP-skew -> all-reduce -> TTFT chain. Here the topologies
are three (T2.1/T3/T4 = prefill TP2/TP4/TP8: 2/4/8 KV senders converging on each
decode rank) and the thing that matters is downstream of that chain:

  * the KV-cache ARRIVAL SKEW WITHIN each decode stage -- the headline -- and how
    (or whether) the per-switch buffer moves it, per topology. "Within a stage"
    is deliberate: the KV of a decode stage is TP-sharded across that stage's
    ranks, so the skew that gates the stage's all-reduce is the max-min arrival
    over its OWN ranks, not the far larger global spread across the whole decode
    pool (which is mostly the inter-stage pipeline gap; kept only in summary.csv
    as kv_skew_global_ms);
  * TTFT vs buffer, all topologies on ONE axis (TTFT is fixed by prefill, so this
    is the cross-topology comparison); and the makespan (total execution) on its
    OWN axis per topology, shown both as the absolute time and as makespan-minus-
    TTFT, the post-first-token tail the incast actually stretches;
  * and, kept from buffer_sweep but pared down, how the switch buffers fill and
    how many PFC PAUSE frames they draw -- but only for the BUSIEST switches
    (top by PFC and/or by buffer occupancy), because these topologies have far
    too many switches to show them all.

  The PP arrival skew (expected ~0 for these placements) and the KV-skew vs
  (total-execution ÷ TTFT) relationship are NOT given their own figures: on this
  data both are near-constant, so each would be a set of flat lines. They are
  kept instead as summary.csv columns (pp_skew_us, total_over_ttft), and become
  worth a figure again once more buffer points make them move.

Prefill/decode split, per topology
--------------------------------------------------------------------------------
The classification that names a flow 'kv' depends on the rank->role placement,
and the placement differs per topology (prefill TP2/TP4/TP8 -> different rank
sets). It is recovered per level from that level's ASTRA trace (roles.from_astra)
and then VERIFIED against the fabric traffic (roles.check): the summary reports
the KV / 'other' flow counts per topology so a split that silently failed on a
wider-TP topology is visible rather than assumed.

Reused verbatim (one definition of a metric): the ns-3 / ASTRA readers, the flow
classification and bottleneck search (utils.flows/fabric), the PP-skew measure
(utils.pp), and buffer_sweep's own KV barrier, per-link stats and PFC-interval
helpers. Only the incast-specific orchestration and the figures are new.

Output
--------------------------------------------------------------------------------
    <out>/01_kv_arrival_skew_vs_buffer.png      intra-stage KV skew vs buffer (panel/topo,
                                                one line per decode stage)
    <out>/02_ttft_vs_buffer.png                 TTFT, all topologies on one axis
    <out>/03_makespan_vs_buffer.png             makespan & makespan-minus-TTFT, panel/topo
    <out>/04_dropped_packets_vs_buffer.png      packet loss vs buffer, all topologies
    <out>/05_pfc_frames_vs_buffer.png           PFC PAUSE frames vs buffer, all topologies
    <out>/<level>_queue_fill_busy_switches.png     queue(t), busiest switches only
    <out>/<level>_occupancy_and_pfc_vs_buffer.png  occupancy & PFC frames, busiest
    <out>/summary.csv     one row per run (incl. pp_skew_us, total_over_ttft)

Usage
-----
    python3 incast_sweep.py
    python3 incast_sweep.py --levels T3 T4 --top-switches 3
    python3 incast_sweep.py -o /tmp/incast
"""

from __future__ import annotations

import argparse
import re
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
from matplotlib.lines import Line2D

from utils import astra, incast, ns3, pp, roles
from utils import flows as flowlib
from utils.cli import Abort, need
from utils.fabric import parse_ns3_config, parse_topology
from utils.paths import BUFFER_AXIS
from utils.plots import downsample_max, logx_pow2, save_fig
from utils.roles import Placement

# buffer_sweep owns the per-run KV/link/PFC measurement; reuse it so a metric has
# one definition. warn()/WARNINGS is its single warning stream -- share it, and
# drain it once at the end.
from buffer_sweep import (WARNINGS, MS, LinkStat, barrier, link_metrics,
                          ttft_end_of_prefill, victim_pause_intervals, warn,
                          _zoom_y)

NAN = float("nan")
BLUE, CORAL, GREEN, VIOLET, MUTED = \
    "#1f77b4", "#d1495b", "#2b8a3e", "#6a4c93", "#9aa0a6"
LOSS_RED = "#e8000b"          # rings/marks runs that DROPPED packets (not lossless);
                              # a vivid red kept distinct from CORAL's data series


def _mark_lossy(ax, g: pd.DataFrame, xcol: str, ycol: str) -> tuple[bool, bool]:
    """Overlay, on top of a plotted series, the runs that were NOT lossless: a red
    ring on runs that dropped packets, a grey ring on runs whose loss is UNKNOWN
    (no drops.txt captured). Returns (marked_lossy, marked_unknown) so the caller
    can add matching legend proxies."""
    ml = mu = False
    if "lossy" in g.columns:
        gl = g[g["lossy"] == True].dropna(subset=[xcol, ycol])
        if not gl.empty:
            ax.scatter(gl[xcol], gl[ycol], s=170, facecolors="none",
                       edgecolors=LOSS_RED, linewidths=2.6, zorder=6)
            ml = True
    if "loss_captured" in g.columns:
        gu = g[g["loss_captured"] == False].dropna(subset=[xcol, ycol])
        if not gu.empty:
            ax.scatter(gu[xcol], gu[ycol], s=140, facecolors="none",
                       edgecolors=MUTED, linewidths=1.8, zorder=5)
            mu = True
    return ml, mu


def _loss_proxies(marked_lossy: bool, marked_unknown: bool) -> list[Line2D]:
    h: list[Line2D] = []
    if marked_lossy:
        h.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=LOSS_RED,
                        mew=2.6, ms=12, label="dropped packets (not lossless)"))
    if marked_unknown:
        h.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=MUTED,
                        mew=1.8, ms=11, label="loss unknown (no drops.txt)"))
    return h


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    tag: str = ""
    level: str = ""
    incast_degree: int = 0
    buffer_mb: float = NAN
    buffer_bytes: float = NAN
    bottleneck: str = ""

    # -- the two execution times, absolute ns ------------------------------- #
    ttft_ns: float = NAN                  # end of prefill (FIRSTTOK send)
    total_exec_ns: float = NAN            # last op end: whole-workload wall clock

    # -- the incast headline: KV arrival skew WITHIN a decode stage --------- #
    kv_skew_ns: float = NAN               # WORST intra-stage skew: max over decode
                                          # stages of (max-min KV arrival among that
                                          # stage's OWN TP ranks). The per-stage TP
                                          # sync cost -- NOT the global decode-pool
                                          # spread (which is mostly the inter-stage
                                          # pipeline gap, kept below for reference).
    kv_skew_mean_ns: float = NAN          # mean intra-stage skew over decode stages
    kv_skew_global_ns: float = NAN        # max-min over ALL decode ranks (reference)
    kv_skew_stage_ns: dict = field(default_factory=dict)  # stage idx -> intra skew ns
    kv_gate_ns: float = NAN               # decode start = last KV arrival
    kv_ready_min_ns: float = NAN
    kv_stream_duration_ns: float = NAN
    decode_ranks: str = ""

    # -- PP skew (expected ~0 here, plotted to check) ----------------------- #
    pp_skew_ns: float = NAN
    pp_skew_mean_ns: float = NAN
    pp_available: bool = False

    # -- prefill/decode split health, per topology -------------------------- #
    kv_flows: int = 0
    other_flows: int = 0
    split_ok: bool = True

    # -- bottleneck link + fabric totals ------------------------------------ #
    link0: object = None                  # LinkStat
    total_pause_frames: float = NAN

    # -- packet loss ("Headroom full" drops = lossless-fabric violation) ----- #
    dropped_packets: float = NAN          # NaN => UNKNOWN (no drops.txt captured)
    loss_captured: bool = False           # True iff drops.txt existed for this run
    packets_delivered: int = 0            # sum ceil(size/payload) over flows
    dropped_per_switch: dict = field(default_factory=dict)  # switch id -> drops

    # -- per-switch, not flattened ------------------------------------------ #
    qseries: dict = field(default_factory=dict)          # sw -> (ts_ns, bytes) downsampled
    qswitch_peak: dict = field(default_factory=dict)     # sw -> peak total bytes
    qswitch_mean: dict = field(default_factory=dict)     # sw -> mean total bytes
    pfc_per_switch: dict = field(default_factory=dict)   # sw -> PAUSE frame count
    pause_intervals: dict = field(default_factory=dict)  # sw -> [(start,end)]

    # -- derived ------------------------------------------------------------ #
    @property
    def total_over_ttft(self) -> float:
        return (self.total_exec_ns / self.ttft_ns
                if pd.notna(self.total_exec_ns) and pd.notna(self.ttft_ns)
                and self.ttft_ns > 0 else NAN)

    @property
    def lossy(self) -> bool:
        """This run dropped packets (loss is known AND non-zero)."""
        return pd.notna(self.dropped_packets) and self.dropped_packets > 0

    @property
    def drop_rate(self) -> float:
        return (self.dropped_packets / self.packets_delivered
                if pd.notna(self.dropped_packets) and self.packets_delivered > 0
                else NAN)

    def flat(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("qseries", "qswitch_peak", "qswitch_mean",
                          "pfc_per_switch", "pause_intervals", "link0",
                          "dropped_per_switch", "kv_skew_stage_ns")}
        d["total_over_ttft"] = self.total_over_ttft
        d["lossy"] = self.lossy
        d["drop_rate_pct"] = self.drop_rate * 100 if pd.notna(self.drop_rate) else NAN
        d["kv_skew_ms"] = self.kv_skew_ns * MS                  # worst intra-stage
        d["kv_skew_mean_ms"] = self.kv_skew_mean_ns * MS
        d["kv_skew_global_ms"] = self.kv_skew_global_ns * MS    # inter-stage, reference
        for si, v in sorted(self.kv_skew_stage_ns.items()):
            d[f"kv_skew_d{si}_ms"] = v * MS                     # per decode stage
        d["pp_skew_us"] = self.pp_skew_ns / 1e3
        d["ttft_ms"] = self.ttft_ns * MS
        d["total_exec_ms"] = self.total_exec_ns * MS
        d["total_minus_ttft_ms"] = (self.total_exec_ns - self.ttft_ns) * MS
        ls: LinkStat | None = self.link0
        if ls is not None:
            d["bn_eff_pct"] = ls.eff_pct
            d["bn_conc_peak"] = ls.conc_peak
            d["bn_pause_frames"] = ls.pause_frames
            d["bn_qpeak_mb"] = ls.qpeak_bytes / 2**20 if pd.notna(ls.qpeak_bytes) else NAN
        return d


# --------------------------------------------------------------------------- #
# Measurement helpers that are incast's own (buffer_sweep has no total-exec, no
# per-switch PFC census)
# --------------------------------------------------------------------------- #
def astra_times(tag: str, p: incast.IncastPaths) -> dict:
    """TTFT (end of prefill) AND total execution time from one read of this run's
    ASTRA trace. TTFT is the FIRSTTOK send (buffer_sweep's definition, reused via
    firsttok_send_instant); total is the last op end over the whole run -- the
    workload wall clock the incast delay ultimately shows up in."""
    adir = p.astra_run(tag)
    if not adir.is_dir():
        warn(f"{tag}: no ASTRA run at {adir}; TTFT/total unavailable.")
        return {}
    df = astra.read_run(adir)
    if df is None:
        warn(f"{tag}: no readable stats_sys*.csv under {adir}; TTFT/total unavailable.")
        return {}
    out = {"total_exec_ns": float(df["end_tick"].max())}
    inst = astra.firsttok_send_instant(df)
    if inst is not None:
        out["ttft_ns"] = inst
    else:
        pre = df.loc[(df["op_class"] == "COMP") & (df["phase"] == "prefill"),
                     "end_tick"]
        if len(pre):
            warn(f"{tag}: no FIRSTTOK; using last prefill compute end as TTFT.")
            out["ttft_ns"] = float(pre.max())
        else:
            warn(f"{tag}: no FIRSTTOK and no prefill COMP; TTFT unavailable.")
    return out


def pfc_pause_census(pfc: ns3.PfcLog) -> tuple[dict[int, int], float]:
    """(PAUSE frames per switch node, total PAUSE frames). A switch appears in
    pfc.txt as the VICTIM whose egress was held (see ns3.PfcLog), so this counts
    how hard each switch was backpressured -- the 'produces the most PFC packets'
    axis of the busy-switch ranking. Hosts (node_type 0) are excluded: the
    question is about switch buffers."""
    per_sw: dict[int, int] = defaultdict(int)
    total = 0
    for (node, ntype, _ifidx, _q), events in pfc.events.items():
        n_pause = sum(1 for _t, typ in events if typ == 1)
        total += n_pause
        if ntype == 1:
            per_sw[node] += n_pause
    return dict(per_sw), float(total)


def switch_pause_intervals(pfc: ns3.PfcLog, clamp_to: int) -> dict[int, list]:
    """PAUSE [start,end] spans per switch node, for shading a switch's queue
    timeline. The per-device intervals of pfc, collapsed onto the switch id."""
    out: dict[int, list] = defaultdict(list)
    for (node, ntype, _ifidx, _q), spans in pfc.pause_intervals(clamp_to=clamp_to).items():
        if ntype == 1:
            out[node].extend(spans)
    return dict(out)


def kv_stage_skew(kv: pd.DataFrame, placement: Placement) -> dict:
    """KV-cache arrival skew computed WITHIN each decode stage (a TP group),
    not across the whole decode pool.

    The KV cache of a decode stage is tensor-parallel sharded over the ranks of
    THAT stage; the stage's attention / all-reduce cannot progress until its
    SLOWEST rank has its shard. So the sync cost that matters per stage is, over
    the ranks of one stage,

        skew_stage = max_r ready_r - min_r ready_r ,

    with ready_r = the last KV arrival at rank r (the max over the incast fan-in
    of prefill senders onto r). This is deliberately intra-stage: the large
    spread the global barrier reports is mostly BETWEEN stages (an early decode
    stage receives its KV long before a later one) -- a pipeline effect, not the
    per-stage skew the receiving TP group actually pays.

    Returns per_stage {stage_idx: skew_ns} over the stages that had >=2 ranks
    fed, the scalar worst_ns / mean_ns over those stages, the global_ns spread
    (kept only for reference), and short_stages: stages that declare >=2 ranks
    but had fewer than 2 receive any KV (their intra-stage skew is undefined)."""
    ready: dict[int, float] = {}
    for d in placement.decode_ranks:
        arr = kv.loc[kv["dst"] == d, "arrival"]
        if len(arr):
            ready[int(d)] = float(arr.max())

    per_stage: dict[int, float] = {}
    short: list[int] = []
    for si, ranks in enumerate(placement.decode):
        fed = [ready[r] for r in ranks if r in ready]
        if len(fed) >= 2:
            per_stage[si] = max(fed) - min(fed)
        elif len(ranks) >= 2:
            short.append(si)
    worst = max(per_stage.values()) if per_stage else NAN
    mean = float(np.mean(list(per_stage.values()))) if per_stage else NAN
    glob = (max(ready.values()) - min(ready.values())) if ready else NAN
    return {"per_stage": per_stage, "worst_ns": worst, "mean_ns": mean,
            "global_ns": glob, "short_stages": short}


def analyse(tag: str, p: incast.IncastPaths, placement: Placement) -> Row:
    buf = BUFFER_AXIS.value(tag)
    need(buf is not None, f"{tag}: no 'buf<num>' token in the name.")
    tpath, cpath, ns3_dir = p.topology(tag), p.config(tag), p.ns3_run(tag)
    for fpath in (tpath, cpath, ns3_dir / "fct.txt", ns3_dir / "pfc.txt",
                  ns3_dir / "qlen.txt"):
        need(fpath.exists(), f"{tag}: missing {fpath}")

    topo = parse_topology(tpath)
    cfg = parse_ns3_config(cpath)
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    if cfg.buffer_mb is not None and abs(cfg.buffer_mb - buf) > 1e-6:
        warn(f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but "
             f"'buf{buf:g}' in the name; trusting the name.")

    row = Row(tag=tag, level=p.level, incast_degree=incast.prefill_tp(placement),
              buffer_mb=float(buf), buffer_bytes=float(buf) * 1024 * 1024)

    for k, v in astra_times(tag, p).items():
        setattr(row, k, v)

    raw = ns3.read_fct(ns3_dir / "fct.txt")
    need(raw is not None and len(raw), f"{tag}: fct.txt has no parsable rows.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)

    # packet loss ("Headroom full"). Read OPTIONALLY (not via need()): a run
    # recorded before stdout capture has no drops.txt and stays analysable, but
    # its loss is UNKNOWN and must not be shown as lossless.
    ds = ns3.read_drops(ns3_dir / "drops.txt")
    row.loss_captured = ds.captured
    row.dropped_packets = float(ds.total) if ds.captured else NAN
    row.dropped_per_switch = dict(ds.per_switch)
    row.packets_delivered = int(np.ceil(
        f["size"].to_numpy() / max(cfg.payload, 1)).sum())
    if not ds.captured:
        warn(f"{tag}: no drops.txt — packet loss UNKNOWN; re-run with the updated "
             f"generate_log_ns3.sh to record it.")
    elif ds.total:
        warn(f"{tag}: NOT lossless — {ds.total} packet(s) dropped "
             f"('Headroom full').")
    split_warnings = roles.check(f, placement)
    for w in split_warnings:
        warn(f"{tag}: {w}")
    vc = f["flow_class"].value_counts()
    row.kv_flows = int(vc.get("kv", 0) + vc.get("kv_ctrl", 0))
    row.other_flows = int(vc.get("other", 0))
    row.split_ok = not split_warnings
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{tag}: no KV flow after classification -- the prefill/decode "
                  f"split does not match this topology's traffic.")

    qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=True)
    need(qlen is not None and qlen.port_max, f"{tag}: qlen.txt has no samples.")
    pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
    need(pfc is not None, f"{tag}: pfc.txt unreadable.")

    run_end = int(f["arrival"].max())

    # bottleneck = deepest-queue link any KV flow crosses (fabric.candidate_links)
    cands = flowlib.candidate_links(topo, qlen.port_max, kv)
    need(cands, f"{tag}: no link is crossed by any KV flow.")
    bn = cands[0]
    row.bottleneck = str(bn)
    row.link0 = link_metrics(kv, bn, topo, pfc, qlen, row.buffer_bytes)

    # KV arrival timing and PP skew read the ASTRA stats CSV (per-op end_tick =
    # arrival, cleanly labelled) instead of reconstructing them from fct.txt --
    # same nanosecond values, none of the flow-classification / incast fan-in /
    # wave-grouping heuristics. Fabric metrics above (link, PFC, drops, queues)
    # stay on ns-3.
    adir = p.astra_run(tag)
    adf = astra.read_run(adir) if adir.is_dir() else None
    kv_arr = astra.kv_arrivals(adf)

    # barrier gives the decode-start gate and the per-rank ready times; the
    # headline SKEW is intra-stage (computed separately, NOT barrier's global
    # cross_rank_skew, which is dominated by the inter-stage gap).
    b = barrier(kv_arr, placement)
    row.kv_gate_ns = b["kv_gate_ns"]
    row.kv_ready_min_ns = b["kv_ready_min_ns"]
    row.kv_stream_duration_ns = b["kv_stream_duration_ns"]
    row.decode_ranks = b["decode_ranks"]
    sk = kv_stage_skew(kv_arr, placement)
    row.kv_skew_ns = sk["worst_ns"]
    row.kv_skew_mean_ns = sk["mean_ns"]
    row.kv_skew_global_ns = sk["global_ns"]
    row.kv_skew_stage_ns = sk["per_stage"]
    if sk["short_stages"]:
        warn(f"{tag}: decode stage(s) {sk['short_stages']} declare >=2 ranks but "
             f"<2 received KV; their intra-stage skew is omitted.")

    ppr = pp.measure(adf)
    row.pp_available = ppr.available
    row.pp_skew_ns = ppr.skew_ns
    row.pp_skew_mean_ns = ppr.skew_mean_ns
    if not ppr.available:
        warn(f"{tag}: no inter-stage PP-prefill flow (expected for these "
             f"placements); PP skew recorded as NaN/0.")

    # per-switch fabric census
    row.pfc_per_switch, row.total_pause_frames = pfc_pause_census(pfc)
    row.pause_intervals = switch_pause_intervals(pfc, run_end)
    for sw, (ts, ys) in qlen.switch_series.items():
        if len(ts) == 0:
            continue
        row.qseries[sw] = downsample_max(ts, ys, 2000)
        row.qswitch_peak[sw] = float(qlen.switch_total_max.get(sw, max(ys)))
        row.qswitch_mean[sw] = float(np.mean(ys))
    return row


# --------------------------------------------------------------------------- #
# Busiest-switch selection (per topology, fixed across its buffers)
# --------------------------------------------------------------------------- #
def busy_switches(rows: list[Row], k_per_metric: int, cap: int) -> list[int]:
    """The switches worth showing for one topology: the union of the top
    `k_per_metric` by PFC PAUSE frames and the top `k_per_metric` by peak buffer
    occupancy (the user's 'most PFC AND/OR most buffer' -- a union, not a single
    blended score, so a switch that dominates EITHER axis is kept). Occupancy
    order first, so the incast point (the deepest-queue core switch) heads the
    list; capped at `cap` for legible grids. Aggregated as the MAX over the
    topology's buffers so the set is fixed across the sweep, not re-picked per
    column."""
    agg_pfc: dict[int, float] = defaultdict(float)
    agg_occ: dict[int, float] = defaultdict(float)
    for r in rows:
        for sw, c in r.pfc_per_switch.items():
            agg_pfc[sw] = max(agg_pfc[sw], c)
        for sw, v in r.qswitch_peak.items():
            agg_occ[sw] = max(agg_occ[sw], v)
    top_occ = [sw for sw, v in sorted(agg_occ.items(), key=lambda x: -x[1])
               if v > 0][:k_per_metric]
    top_pfc = [sw for sw, v in sorted(agg_pfc.items(), key=lambda x: -x[1])
               if v > 0][:k_per_metric]
    ordered: list[int] = []
    for sw in top_occ + top_pfc:
        if sw not in ordered:
            ordered.append(sw)
    return ordered[:cap]


# --------------------------------------------------------------------------- #
# Per-topology analysis
# --------------------------------------------------------------------------- #
@dataclass
class Level:
    level: str
    degree: int
    rows: list           # list[Row], sorted by buffer
    busy: list           # list[int] switch ids
    label: str           # "T3 (tp4)"


def analyse_level(level: str, root: Path, out_workload: str, config_sweep: str,
                  k_switches: int) -> Level | None:
    p = incast.IncastPaths(level=level, out_workload=out_workload,
                           config_sweep=config_sweep, root=root)
    if p.missing_roots():
        warn(f"{level}: skipped, derived root(s) missing:\n    "
             + "\n    ".join(p.missing_roots()))
        return None
    tags, skipped = p.usable_tags()
    for s in skipped:
        warn(f"{level}: {s} (config/output still being generated?) -- skipped.")
    if not tags:
        warn(f"{level}: no run has all inputs on disk yet; skipped.")
        return None

    placement = recover_placement(p, tags)
    degree = incast.prefill_tp(placement)
    print(f"\n===== {level}  (prefill TP{degree}, incast degree {degree}) =====")
    print(f"  placement {roles.spec_of(placement)}")
    print(f"  buffers   {[BUFFER_AXIS.value(t) for t in tags]}")

    rows = []
    for tag in tags:
        try:
            rows.append(analyse(tag, p, placement))
        except Abort as e:
            warn(f"{level}: run {tag} dropped -- {e}")
    if not rows:
        warn(f"{level}: every run failed to analyse; skipped.")
        return None
    rows.sort(key=lambda r: r.buffer_mb)
    busy = busy_switches(rows, k_switches, cap=2 * k_switches)

    for r in rows:
        flag = "" if r.split_ok else "  ! split check FAILED"
        if r.lossy:
            flag += f"  ** LOSS: {r.dropped_packets:.0f} pkt ({r.drop_rate*100:.2g}%) **"
        elif not r.loss_captured:
            flag += "  (loss unknown: no drops.txt)"
        print(f"  + buf{r.buffer_mb:<4g} bn={r.bottleneck:<8} "
              f"kv_skew(intra)={r.kv_skew_ns*MS:6.2f}ms  ttft={r.ttft_ns*MS:6.1f}ms  "
              f"total={r.total_exec_ns*MS:6.1f}ms  pfc={r.total_pause_frames:.0f}  "
              f"kv_flows={r.kv_flows}{flag}")
    print(f"  busiest switches (top {k_switches} by PFC ∪ by occupancy): {busy}")
    return Level(level=level, degree=degree, rows=rows, busy=busy,
                 label=f"{level} (tp{degree})")


def recover_placement(p: incast.IncastPaths, tags: list[str]) -> Placement:
    """The level's rank->role map, recovered from its ASTRA trace (prefill TP
    width differs per topology, so it is not a single CLI placement). Tries each
    run until one trace is readable."""
    last_err = None
    for tag in tags:
        adir = p.astra_run(tag)
        if not adir.is_dir():
            continue
        try:
            return roles.from_astra(adir)
        except Exception as e:                                  # noqa: BLE001
            last_err = e
    raise Abort(f"{p.level}: no readable ASTRA trace to recover the placement "
                f"from. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# Per-level colour overrides: viridis maps the last level to its pale-yellow
# endpoint, which is nearly invisible on a white background. Override those
# levels with readable, mutually distinct hues (kept CVD-safe against the
# viridis purple/teal of the other levels, and away from the reserved LOSS_RED).
_LEVEL_COLOR_OVERRIDE = {
    "T4": "#e8590c",          # dark orange -- readable, distinct from T2.1/T3
}


def _level_colors(levels: list[Level]) -> dict[str, tuple]:
    cmap = plt.get_cmap("viridis")
    n = max(len(levels) - 1, 1)
    return {lv.level: _LEVEL_COLOR_OVERRIDE.get(lv.level, cmap(i / n))
            for i, lv in enumerate(levels)}


def fig_kv_skew(levels: list[Level], s: pd.DataFrame, outdir: Path,
                written: list[Path]) -> None:
    """01 -- the headline: KV-cache arrival skew WITHIN each decode stage vs the
    per-switch buffer. One panel per topology (own y-scale: the skew shrinks with
    incast degree, so a shared axis would flatten the wider-TP topologies), one
    line per decode stage. Intra-stage = the per-stage TP sync cost (max-min over
    that stage's own ranks); the global decode-pool spread is NOT this figure --
    it is mostly the inter-stage pipeline gap, kept in summary.csv
    (kv_skew_global_ms) instead. Question: does more buffer close the per-stage
    skew, and does the answer depend on the incast degree?"""
    stage_cols = sorted((c for c in s.columns
                         if re.fullmatch(r"kv_skew_d\d+_ms", c)),
                        key=lambda c: int(c[len("kv_skew_d"):-len("_ms")]))
    usable = [lv for lv in levels if stage_cols and any(
        s.loc[s["level"] == lv.level, c].notna().any() for c in stage_cols)]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(1, n, figsize=(max(4.8 * n, 5), 4.6), squeeze=False)
    cmap = plt.get_cmap("tab10")
    anyl = bool(s["lossy"].any()) if "lossy" in s.columns else False
    anyu = bool((~s["loss_captured"]).any()) if "loss_captured" in s.columns else False
    for j, lv in enumerate(usable):
        a = axes[0][j]
        g = s[s["level"] == lv.level].sort_values("buffer_mb")
        for k, col in enumerate(stage_cols):
            gg = g.dropna(subset=[col])
            if gg.empty:
                continue
            si = col[len("kv_skew_d"):-len("_ms")]
            a.plot(gg["buffer_mb"], gg[col], "o-", color=cmap(k),
                   label=f"decode stage d{si}")
            _mark_lossy(a, gg, "buffer_mb", col)
        logx_pow2(a, g, "buffer_mb", "Per-switch buffer (MiB)")
        a.set_title(lv.label, fontsize=10)
        a.grid(True, alpha=0.3, which="both")
        a.set_ylim(bottom=0)
        h, _ = a.get_legend_handles_labels()
        proxies = _loss_proxies(anyl, anyu) if j == 0 else []
        a.legend(handles=h + proxies, fontsize=8)
        if j == 0:
            a.set_ylabel("Intra-stage KV arrival skew (ms)")
    fig.suptitle("KV-cache arrival skew WITHIN each decode stage "
                 "(max−min over the stage's TP ranks)", y=1.02)
    save_fig(fig, outdir, "01_kv_arrival_skew_vs_buffer.png", written)


def fig_ttft(levels: list[Level], s: pd.DataFrame, outdir: Path,
             written: list[Path]) -> None:
    """02 -- TTFT vs buffer, ALL topologies on one axis (one line each). TTFT is
    end of prefill, upstream of the KV/fabric congestion, so it is flat across
    the buffer; the figure's job is the cross-topology comparison -- TTFT falls
    as the prefill TP width grows (more compute parallelism on prefill)."""
    colors = _level_colors(levels)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    drew = anyl = anyu = False
    for lv in levels:
        g = s[s["level"] == lv.level].dropna(subset=["ttft_ms"]).sort_values("buffer_mb")
        if g.empty:
            continue
        ax.plot(g["buffer_mb"], g["ttft_ms"], "o-", color=colors[lv.level],
                label=lv.label)
        ml, mu = _mark_lossy(ax, g, "buffer_mb", "ttft_ms")
        anyl |= ml
        anyu |= mu
        drew = True
    if not drew:
        plt.close(fig)
        return
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title("TTFT vs buffer (all topologies)")
    ax.grid(True, alpha=0.3, which="both")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + _loss_proxies(anyl, anyu), fontsize=8, title="topology")
    save_fig(fig, outdir, "02_ttft_vs_buffer.png", written)


def fig_makespan(levels: list[Level], s: pd.DataFrame, outdir: Path,
                 written: list[Path]) -> None:
    """03 -- total execution time (makespan) vs buffer, one panel per topology.
    Just the makespan, and each panel's y-axis is FITTED to its own data (not
    pinned to zero), so the small buffer-driven variation is actually readable
    instead of a flat line at the top of the frame. The makespan-minus-TTFT
    decomposition is dropped from the figure; it survives in summary.csv as
    total_minus_ttft_ms for anyone who wants it."""
    usable = [lv for lv in levels
              if not s[s["level"] == lv.level].dropna(subset=["total_exec_ms"]).empty]
    if not usable:
        return
    n = len(usable)
    anyl = bool(s["lossy"].any()) if "lossy" in s.columns else False
    anyu = bool((~s["loss_captured"]).any()) if "loss_captured" in s.columns else False
    fig, axes = plt.subplots(1, n, figsize=(max(4.6 * n, 5), 4.6), squeeze=False)
    for j, lv in enumerate(usable):
        a = axes[0][j]
        g = (s[s["level"] == lv.level]
             .dropna(subset=["total_exec_ms"]).sort_values("buffer_mb"))
        a.plot(g["buffer_mb"], g["total_exec_ms"], "s-", color=CORAL,
               label="makespan")
        _mark_lossy(a, g, "buffer_mb", "total_exec_ms")
        logx_pow2(a, g, "buffer_mb", "Per-switch buffer (MiB)")
        _zoom_y(a, g["total_exec_ms"])          # fit y to the data, not to zero
        a.set_title(lv.label, fontsize=10)
        a.grid(True, alpha=0.3, which="both")
        if j == 0:
            a.set_ylabel("Makespan (ms)")
            h, _ = a.get_legend_handles_labels()
            a.legend(handles=h + _loss_proxies(anyl, anyu), fontsize=8)
    fig.suptitle("Makespan vs buffer per topology (y fitted to data)", y=1.02)
    save_fig(fig, outdir, "03_makespan_vs_buffer.png", written)


def fig_pfc_frames(levels: list[Level], s: pd.DataFrame, outdir: Path,
                   written: list[Path]) -> None:
    """05 -- PFC PAUSE frames vs buffer, all topologies on one axis. The direct
    picture of the point that the fabric DOES respond to the buffer even where
    the end-to-end times do not: PFC backpressure explodes at small buffers and
    higher incast degree and collapses to zero as the buffer grows.

    Symlog y-axis on purpose: the counts span 0 to ~10^4 across topologies, and
    ZERO has to be shown -- a pause-free run at large buffer is exactly the
    right-hand end this figure exists to make visible, which a log axis would
    drop and a linear axis would flatten to the x-axis under the T4/buf8 spike."""
    if "total_pause_frames" not in s.columns:
        return
    colors = _level_colors(levels)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    drew = anyl = anyu = False
    for lv in levels:
        g = (s[s["level"] == lv.level]
             .dropna(subset=["total_pause_frames"]).sort_values("buffer_mb"))
        if g.empty:
            continue
        ax.plot(g["buffer_mb"], g["total_pause_frames"], "o-",
                color=colors[lv.level], label=lv.label)
        ml, mu = _mark_lossy(ax, g, "buffer_mb", "total_pause_frames")
        anyl |= ml
        anyu |= mu
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_yscale("symlog", linthresh=10)   # linear within +-10 so 0 shows, log above
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    ax.set_ylabel("PFC PAUSE frames, whole fabric (symlog)")
    ax.set_title("PFC backpressure vs buffer (all topologies)")
    ax.grid(True, alpha=0.3, which="both")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + _loss_proxies(anyl, anyu), fontsize=8, title="topology")
    save_fig(fig, outdir, "05_pfc_frames_vs_buffer.png", written)


def fig_drops(levels: list[Level], s: pd.DataFrame, outdir: Path,
              written: list[Path]) -> None:
    """04 -- dedicated packet-loss panel: dropped packets ('Headroom full') vs the
    per-switch buffer, one line per topology. Lossless runs sit at 0; a run that
    violated the lossless fabric spikes up, is ringed in red, and is labelled with
    its drop count. Only runs with a captured drops.txt are plotted (loss-unknown
    runs are omitted), so a flat line at 0 here is a *certified* lossless sweep,
    not merely an un-measured one. Skipped entirely while no run has a drops.txt."""
    if "dropped_packets" not in s.columns:
        return
    colors = _level_colors(levels)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    drew = anyl = False
    for lv in levels:
        g = (s[s["level"] == lv.level]
             .dropna(subset=["dropped_packets"]).sort_values("buffer_mb"))
        if g.empty:
            continue
        ax.plot(g["buffer_mb"], g["dropped_packets"], "o-", color=colors[lv.level],
                label=lv.label)
        ml, _ = _mark_lossy(ax, g, "buffer_mb", "dropped_packets")
        anyl |= ml
        for _, rr in g[g["dropped_packets"] > 0].iterrows():
            ax.annotate(f"{int(rr['dropped_packets'])}",
                        (rr["buffer_mb"], rr["dropped_packets"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=LOSS_RED, fontweight="bold")
        drew = True
    if not drew:
        plt.close(fig)
        return
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    ax.set_ylabel("Dropped packets ('Headroom full')")
    ax.set_title("Packet loss vs buffer: where the fabric stops being lossless")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, which="both")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + _loss_proxies(anyl, False), fontsize=8,
              title="topology")
    save_fig(fig, outdir, "04_dropped_packets_vs_buffer.png", written)


def fig_queue_fill(lv: Level, outdir: Path, written: list[Path]) -> None:
    """(per topology) -- how the BUSIEST switches' buffers fill over time,
    rows = busy switch, cols = buffer, with PFC PAUSE spans shaded as a top
    ribbon. The pared-down descendant of buffer_sweep's per-switch grid: same
    picture, but only the switches that actually matter here."""
    if not lv.busy:
        return
    runs = lv.rows
    nrows, ncols = len(lv.busy), len(runs)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                             sharey="row",
                             figsize=(max(2.1 * ncols + 1.8, 6),
                                      max(1.7 * nrows + 1.0, 4)))
    cmap = plt.get_cmap("viridis")
    bufs = [r.buffer_mb for r in runs]
    cnorm = (matplotlib.colors.LogNorm(vmin=min(bufs), vmax=max(bufs))
             if len(set(bufs)) > 1 else None)
    for i, sw in enumerate(lv.busy):
        for j, r in enumerate(runs):
            a = axes[i][j]
            if sw in r.qseries:
                ts, ys = r.qseries[sw]
                col = cmap(cnorm(r.buffer_mb)) if cnorm else BLUE
                a.fill_between(np.asarray(ts) * MS, np.asarray(ys) / 1e3,
                               color=col, alpha=0.85, lw=0)
                a.plot(np.asarray(ts) * MS, np.asarray(ys) / 1e3, color="#222222",
                       lw=0.5, alpha=0.6)
            for s0, e0 in r.pause_intervals.get(sw, []):
                a.axvspan(s0 * MS, e0 * MS, ymin=0.88, ymax=1.0,
                          transform=a.get_xaxis_transform(), color=CORAL,
                          alpha=0.9, lw=0)
            a.grid(True, alpha=0.2)
            if i == 0:
                tc = (LOSS_RED if r.lossy
                      else MUTED if not r.loss_captured else "black")
                a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9, color=tc)
            if j == 0:
                a.set_ylabel(f"sw {sw}\n(kB)", fontsize=8)
            if i == nrows - 1:
                a.set_xlabel("Time (ms)", fontsize=8)
                a.locator_params(axis="x", nbins=4)
    lossy = [f"{r.buffer_mb:g}" for r in runs if r.lossy]
    note = f"  [red title = dropped packets: {', '.join(lossy)} MiB]" if lossy else ""
    fig.suptitle(f"{lv.label}: busiest switches' buffer fill over time "
                 f"(PFC PAUSE shaded){note}", y=1.01)
    save_fig(fig, outdir, f"{lv.level}_queue_fill_busy_switches.png", written)


def fig_occ_pfc(lv: Level, outdir: Path, written: list[Path]) -> None:
    """(per topology) -- the busiest switches summarised vs buffer: peak
    occupancy (MB, left) and PFC PAUSE frames drawn (right), one line per busy
    switch. The scalar companion to figure 05."""
    if not lv.busy:
        return
    runs = lv.rows
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")
    for i, sw in enumerate(lv.busy):
        xs, occ, pfc = [], [], []
        for r in runs:
            xs.append(r.buffer_mb)
            occ.append(r.qswitch_peak.get(sw, NAN) / 2**20)
            pfc.append(r.pfc_per_switch.get(sw, 0))
        c = cmap(i % 10)
        axL.plot(xs, occ, "o-", color=c, label=f"sw {sw}")
        axR.plot(xs, pfc, "s-", color=c, label=f"sw {sw}")
    # flag the runs that were not lossless: red line at a lossy buffer, grey at an
    # unknown-loss one (older run without drops.txt).
    lossy_bufs = [r.buffer_mb for r in runs if r.lossy]
    unknown_bufs = [r.buffer_mb for r in runs if not r.lossy and not r.loss_captured]
    for ax in (axL, axR):
        for xb in lossy_bufs:
            ax.axvline(xb, color=LOSS_RED, ls=":", lw=1.6, alpha=0.75, zorder=0)
        for xb in unknown_bufs:
            ax.axvline(xb, color=MUTED, ls=":", lw=1.0, alpha=0.5, zorder=0)
    for ax, ylab, title in ((axL, "Peak queue occupancy (MB)", "Buffer fill"),
                            (axR, "PFC PAUSE frames", "Backpressure")):
        logx_pow2(ax, pd.DataFrame({"buffer_mb": [r.buffer_mb for r in runs]}),
                  "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, title="switch")
    note = " (red dotted = run dropped packets)" if lossy_bufs else ""
    fig.suptitle(f"{lv.label}: busiest switches — occupancy and PFC vs buffer{note}",
                 y=1.02)
    save_fig(fig, outdir, f"{lv.level}_occupancy_and_pfc_vs_buffer.png", written)


# --------------------------------------------------------------------------- #
REPORT = ["level", "incast_degree", "buffer_mb", "bottleneck", "kv_skew_ms",
          "kv_skew_global_ms", "ttft_ms", "total_exec_ms", "total_over_ttft",
          "pp_skew_us", "total_pause_frames", "dropped_packets", "drop_rate_pct",
          "loss_captured", "kv_flows", "other_flows", "split_ok"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(incast.ROOT), type=Path,
                    help=f"project root (default: {incast.ROOT})")
    ap.add_argument("--out-workload", default=incast.OUT_WORKLOAD,
                    help=f"output dir name under output/<domain> (default: "
                         f"{incast.OUT_WORKLOAD})")
    ap.add_argument("--config-sweep", default=incast.CONFIG_SWEEP,
                    help=f"config sub-dir under configs/astra_sim/ns3 (default: "
                         f"{incast.CONFIG_SWEEP})")
    ap.add_argument("--levels", nargs="+", default=None,
                    help="incast levels to analyse, e.g. --levels T3 T4 "
                         "(default: every level found)")
    ap.add_argument("--top-switches", type=int, default=3,
                    help="how many switches to keep PER metric (PFC, occupancy); "
                         "the busy set is their union (default: 3)")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/incast/"
                         "<out-workload>)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    outdir = (Path(a.out) if a.out else
              root / "results" / "sweep_analysis" / "incast" / a.out_workload)

    try:
        levels_found = incast.discover_levels(a.out_workload, root, "ns3")
        need(levels_found,
             f"no incast level under {root / 'output' / 'ns3' / a.out_workload}. "
             f"Is --out-workload right?")
        if a.levels:
            want = set(a.levels)
            missing = want - set(levels_found)
            need(not missing, f"--levels {sorted(missing)} not present; "
                              f"found {levels_found}")
            levels_found = [l for l in levels_found if l in want]

        print(f"  root      {root}")
        print(f"  workload  {a.out_workload}")
        print(f"  out       {outdir}")
        print(f"  levels    {levels_found}")

        levels = []
        for lv in levels_found:
            L = analyse_level(lv, root, a.out_workload, a.config_sweep,
                              a.top_switches)
            if L is not None:
                levels.append(L)
        need(levels, "no incast level produced any analysable run.")
        levels.sort(key=lambda L: L.degree)

        s = pd.DataFrame([r.flat() for L in levels for r in L.rows])
        s = s.sort_values(["incast_degree", "buffer_mb"]).reset_index(drop=True)

        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        front = [c for c in REPORT if c in s.columns]
        s[front + [c for c in s.columns if c not in front]].to_csv(
            outdir / "summary.csv", index=False)

        written: list[Path] = []
        fig_kv_skew(levels, s, outdir, written)
        fig_ttft(levels, s, outdir, written)
        fig_makespan(levels, s, outdir, written)
        fig_drops(levels, s, outdir, written)
        fig_pfc_frames(levels, s, outdir, written)
        for L in levels:
            fig_queue_fill(L, outdir, written)
            fig_occ_pfc(L, outdir, written)

        pd.set_option("display.width", 240)
        print("\n================ INCAST SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))
        # prefill/decode split verdict, per the user's explicit ask
        bad = s[~s["split_ok"]] if "split_ok" in s.columns else s.iloc[0:0]
        if len(bad):
            print(f"\n! prefill/decode split check FAILED on "
                  f"{sorted(set(bad['tag']))} — see warnings.")
        else:
            print("\nprefill/decode split: OK on every analysed run "
                  "(KV flows classified, no 'other', across all topologies).")

        # packet-loss verdict: which runs stopped being lossless, and by how much
        if "lossy" in s.columns:
            lossy = s[s["lossy"] == True]
            unknown = (s[~s["loss_captured"]] if "loss_captured" in s.columns
                       else s.iloc[0:0])
            if len(lossy):
                tot = int(lossy["dropped_packets"].sum())
                print(f"\n! PACKET LOSS (NOT lossless) on {len(lossy)} run(s), "
                      f"{tot} dropped pkt total — flagged RED in the figures:")
                for _, rr in lossy.sort_values("dropped_packets",
                                               ascending=False).iterrows():
                    print(f"    {rr['level']} buf{rr['buffer_mb']:g}: "
                          f"{int(rr['dropped_packets'])} pkt "
                          f"({rr['drop_rate_pct']:.3g}% of delivered)")
            else:
                print("\npacket loss: none on any run with a captured drops.txt "
                      "(fabric lossless).")
            if len(unknown):
                print(f"  (loss UNKNOWN — no drops.txt — on "
                      f"{sorted(set(unknown['tag']))}; re-run to record.)")
        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", *[q.name for q in written]]:
            print(f"  {fpath}")
        if WARNINGS:
            print(f"\n{len(WARNINGS)} WARNING(S):")
            for w in WARNINGS:
                print(f"  ! {w}")
            return 1
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
