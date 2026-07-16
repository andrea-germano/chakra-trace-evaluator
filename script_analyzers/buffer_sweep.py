#!/usr/bin/env python3
"""
buffer_sweep — per-switch buffer sweep, MLSynth disaggregated inference.

The question
--------------------------------------------------------------------------------
Does the switch buffer change when disaggregated decode can start?

It cannot be answered from the ASTRA-sim CSVs, so this analyzer does not read
them: at steady state the congested link drains at line rate whatever the buffer
is, and the CSVs say "nothing happens". What the buffer changes is the
*congestion-control regime* -- whether the queue is held by PFC backpressure or
by DCQCN rate control -- and that lives entirely in the ns-3 outputs. This is an
ns-3 question and the tool is an ns-3 tool.

Two levels, kept physically apart:

    MODEL       from physical_topology.txt + config.txt ALONE, utils.fabric
                computes where the regime must flip (a band, in MiB). No
                simulation involved. Figure 05, and only figure 05.
    MEASURED    from fct.txt / pfc.txt / qlen.txt. Figures 01-04. Every number
                on them is read, none is fitted or estimated.

The two agreeing is the result: the tool says what the fabric will do before you
build it. Disagreeing is also a result -- see `headroom_factor` below.

Declared, never inferred
--------------------------------------------------------------------------------
--sweep       the one path input; every other path is derived (utils.paths).
--placement   the rank->role map (utils.roles). It replaces --bulk-mb: the class
              of a flow is structural, not a size threshold that needs tuning.
--bottleneck  optional. Default is measured (deepest queue in qlen.txt), and it
              must come out the same on every run of the sweep or this aborts:
              a sweep whose curves are stitched together from different switches
              is not a sweep.

What can silently be wrong
--------------------------------------------------------------------------------
`headroom_factor` is compiled into ns-3 (common.h) and does NOT appear in
config.txt, so this tool cannot read it -- it is asserted with --headroom-factor
and every threshold, ceiling and band scales with it. If measured peak egress
exceeds the modelled PFC ceiling, the ceiling is an upper bound by construction
(pfc_threshold is evaluated at shared_used=0) and the excess is proof the
asserted value is wrong. That check runs and prints; it is not decoration.

Usage
--------------------------------------------------------------------------------
    python3 buffer_sweep.py --sweep buffer_sweep_T1
    python3 buffer_sweep.py --sweep buffer_sweep_T1 --placement "p0=0,1 p1=2,3 d0=4,5 d1=6,7"
    python3 buffer_sweep.py --sweep buffer_sweep_T1 --bottleneck 8->12 -o /tmp/x
    python3 -m utils.fabric <topology> <config>     # the model, on its own
    python3 buffer_sweep.py --print-patch           # the ns-3 qIndex diff
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import astra
from utils import flows as flowlib
from utils import ns3, paths, roles
from utils.fabric import Bottleneck, FabricModel, Ns3Config, Topology, \
    parse_ns3_config, parse_topology
from utils.plots import logx_pow2, save_fig
from utils.roles import Placement
from utils.paths import BUFFER_AXIS

NAN = float("nan")
KIND = "buffer"


# --------------------------------------------------------------------------- #
# Fail fast, warn loud
# --------------------------------------------------------------------------- #
class Abort(Exception):
    """A condition under which no number this script could print would mean
    anything. Never caught, never downgraded to a default."""


def need(cond, msg: str) -> None:
    if not cond:
        raise Abort(msg)


WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  ! {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# One row per run
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    tag: str = ""
    buffer_mb: float = NAN
    cc_mode: float = NAN

    # -- the bottleneck, and the model of it (topology + config only) -------- #
    bottleneck: str = ""
    bn_rate_gbps: float = NAN
    f_ports: float = NAN
    ingress_ports: str = ""
    pfc_shift: float = NAN
    pfc_thresh_bytes: float = NAN
    pfc_thresh_x_fports_bytes: float = NAN
    pfc_thresh_naive_bytes: float = NAN
    naive_error_pct: float = NAN
    pfc_ceiling_bytes: float = NAN
    hdrm_rsrv_pct_of_buffer: float = NAN
    kmin_bytes: float = NAN
    kmax_bytes: float = NAN
    regime_model: str = ""

    # -- measured: queue ----------------------------------------------------- #
    qlen_peak_bytes: float = NAN
    qlen_mean_bytes: float = NAN
    qlen_peak_over_kmax: float = NAN
    qlen_peak_over_ceiling: float = NAN      # >1 falsifies --headroom-factor

    # -- measured: PFC ------------------------------------------------------- #
    pfc_qidx: str = ""
    pfc_pause_pct_of_window: float = NAN
    pfc_pause_pct_suspect: float = NAN   # mis-paired without qIndex; an upper bound on the error
    pfc_pause_worst_device: str = ""
    pfc_paused_devices: float = NAN

    # -- measured: flows ----------------------------------------------------- #
    flows_total: float = NAN
    flows_tp: float = NAN
    flows_kv: float = NAN
    flows_pp_prefill: float = NAN
    flows_pp_decode: float = NAN
    flows_other: float = NAN
    kv_at_bottleneck: float = NAN
    kv_bytes_at_bottleneck: float = NAN
    kv_window_ns: float = NAN            # measured: first KV posted -> last arrived
    kv_floor_ns: float = NAN             # bytes / bottleneck rate: nothing beats this
    line_rate_efficiency: float = NAN    # floor / window; 1.0 = saturated throughout
    concurrency_peak: float = NAN
    concurrency_mean: float = NAN
    slow_mean: float = NAN
    slow_p50: float = NAN
    slow_p99: float = NAN
    slow_max: float = NAN

    # -- measured: the barrier ----------------------------------------------- #
    kv_ready_max_ns: float = NAN             # the decode-start gate
    kv_ready_min_ns: float = NAN
    cross_rank_skew_ns: float = NAN          # a real skew: spread ACROSS ranks
    kv_stream_duration_ns: float = NAN       # NOT a skew -- see barrier()
    decode_ranks: str = ""

    slowdowns: object = None                 # raw array, for the box plot

    def flat(self) -> dict:
        d = asdict(self)
        d.pop("slowdowns")
        return d


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def pause_pct(log: ns3.PfcLog, bn: Bottleneck, topo: Topology,
              lo: int, hi: int) -> tuple[float, float, str, int]:
    """Fraction of [lo, hi] each PAUSE victim of `bn` spent paused.

    Two things the old code got wrong and that are not cosmetic:

    * the numerator was accumulated over the whole run and divided by the KV
      window. Different supports; not a percentage of anything. Here the pause
      intervals are clipped to the same window as the denominator.
    * queues of one device overlap in time, so summing over qIndex can exceed
      the device's own paused wall-clock. Here they are unioned.

    When pfc.txt has no qIndex the pairing can be wrong, but only where the two
    queues actually interleave -- so the suspect intervals are flagged and their
    weight returned, rather than the whole file being declared unusable. The
    truth is bracketed by [pct - suspect, pct].

    Only the devices upstream of THIS link are evidence about ITS regime: the
    global worst can sit on an unrelated one (in T1 it usually sits on the other
    leaf, which is a second, independent bottleneck)."""
    victims = set(bn.pause_victims(topo))
    iv = log.pause_intervals_flagged(clamp_to=hi)
    per_dev: dict[tuple[int, int], list] = {}
    for (node, _ntype, ifidx, _q), spans in iv.items():
        if (node, ifidx) in victims:
            per_dev.setdefault((node, ifidx), []).extend(spans)
    if not per_dev:
        return 0.0, 0.0, "", 0
    span = hi - lo
    need(span > 0, "the KV window has zero duration")
    tot = {k: union_len([(a, b) for a, b, _ in v], lo, hi) for k, v in per_dev.items()}
    best = max(tot, key=tot.get)
    sus = union_len([(a, b) for a, b, q in per_dev[best] if q], lo, hi)
    return (100.0 * tot[best] / span, 100.0 * sus / span,
            f"n{best[0]}/if{best[1]}", sum(1 for v in tot.values() if v > 0))


def union_len(spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Measure of the union of `spans` clipped to [lo, hi]."""
    clipped = sorted((max(s, lo), min(e, hi)) for s, e in spans
                     if min(e, hi) > max(s, lo))
    total, cs, ce = 0, None, None
    for s, e in clipped:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs if cs is not None else 0)


def barrier(kv: pd.DataFrame, placement: Placement) -> dict:
    """The first decode step is a synchronisation barrier: it cannot start until
    every KV flow feeding a decode rank has arrived. KV-ready per rank is that
    rank's latest arrival; the gate is the worst rank.

    Two spreads, and they are not the same quantity:

        cross_rank_skew_ns    max(ready) - min(ready) ACROSS decode ranks. A
                              real synchronisation skew: how much earlier the
                              luckiest rank could have started.
        kv_stream_duration_ns max(arrival) - min(arrival) WITHIN one rank. With
                              KV emitted per layer (T1: 20 flows per prefill
                              rank, one per layer) this is the duration of that
                              rank's KV stream, staggered by prefill compute --
                              NOT a skew. The old `sync_skew_ns` was this, under
                              the other name."""
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
    out["kv_ready_max_ns"] = max(ready.values())
    out["kv_ready_min_ns"] = min(ready.values())
    out["cross_rank_skew_ns"] = max(ready.values()) - min(ready.values())
    out["kv_stream_duration_ns"] = max(dur.values())
    return out


def analyse(tag: str, p: paths.SweepPaths, placement: Placement,
            hf: int, bn_force: str | None) -> Row:
    buf = BUFFER_AXIS.value(tag)
    need(buf is not None, f"{tag}: no 'buf<num>' token in the directory name; "
                          f"the swept axis is unreadable.")

    tpath, cpath = p.topology(tag), p.config(tag)
    ns3_dir = p.ns3_run(tag)
    for f in (tpath, cpath, ns3_dir / "fct.txt", ns3_dir / "pfc.txt",
              ns3_dir / "qlen.txt"):
        need(f.exists(), f"{tag}: missing {f}")

    topo = parse_topology(tpath, hf)
    cfg = parse_ns3_config(cpath)
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    need(cfg.buffer_mb is not None,
         f"{tag}: no BUFFER_SIZE in {cpath}. If this is the template, --sweep "
         f"points at the template dir, not the generated configs.")
    need(abs(cfg.buffer_mb - buf) < 1e-6,
         f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but 'buf{buf:g}' "
         f"in the directory name. One of the two is lying.")
    if topo.ecmp_pairs:
        warn(f"{tag}: ECMP ties on {len(topo.ecmp_pairs)} (node, host) pairs: "
             f"runtime paths are hash-chosen, so per-flow path attribution -- "
             f"including the ingress set of the bottleneck -- is approximate.")

    row = Row(tag=tag, buffer_mb=float(buf), cc_mode=cfg.cc_mode if cfg.cc_mode
              is not None else NAN)
    buffer_bytes = int(buf * 1024 * 1024)

    # -- flows ------------------------------------------------------------- #
    raw = ns3.read_fct(ns3_dir / "fct.txt")
    need(raw is not None and len(raw), f"{tag}: fct.txt has no parsable rows.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)
    for w in roles.check(f, placement):
        warn(f"{tag}: {w}")
    counts = f["flow_class"].value_counts()
    row.flows_total = len(f)
    for c in roles.FLOW_CLASSES:
        setattr(row, f"flows_{c}", float(counts.get(c, 0)))
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{tag}: no KV flow after classification.")

    # -- the bottleneck ---------------------------------------------------- #
    qlen = ns3.read_qlen(ns3_dir / "qlen.txt")
    need(qlen is not None and qlen.port_max, f"{tag}: qlen.txt has no samples.")
    if bn_force:
        sw, peer = (int(x) for x in bn_force.split("->"))
        egress = topo.port_facing(sw, peer)
        need(egress is not None, f"--bottleneck {bn_force}: no such link.")
        ing = set()
        for path in kv["path"]:
            for i, (x, y) in enumerate(path or []):
                if (x, y) == (sw, peer) and i > 0:
                    if (q := topo.port_facing(sw, path[i - 1][0])) is not None:
                        ing.add(q)
        bn = Bottleneck(sw, egress, peer, topo.ports[sw][egress].rate,
                        tuple(sorted(ing)))
    else:
        bn = flowlib.find_bottleneck(topo, qlen.port_max, kv)
    need(bn.f_ports, f"{tag}: no KV flow enters {bn} through a known ingress "
                     f"port; F_ports=0 and the PFC threshold is meaningless.")

    row.bottleneck, row.bn_rate_gbps = str(bn), bn.rate / 1e9
    row.f_ports = bn.f_ports
    row.ingress_ports = ",".join(map(str, bn.ingress_ports))
    row.pfc_shift = topo.shift[bn.switch][bn.egress_port]

    # -- the model (topology + config only) -------------------------------- #
    model = FabricModel(topo, cfg)
    row.pfc_thresh_bytes = model.steady_threshold(bn, buffer_bytes)
    row.pfc_thresh_x_fports_bytes = model.egress_equivalent_threshold(bn, buffer_bytes)
    row.pfc_ceiling_bytes = model.pfc_egress_ceiling(bn, buffer_bytes)
    row.pfc_thresh_naive_bytes = buffer_bytes / 8.0
    row.naive_error_pct = 100.0 * (row.pfc_thresh_naive_bytes /
                                   row.pfc_thresh_bytes - 1.0)
    row.hdrm_rsrv_pct_of_buffer = 100.0 * (topo.total_hdrm[bn.switch] +
                                           topo.total_rsrv[bn.switch]) / buffer_bytes
    kmin, kmax = model.ecn_band(bn)
    need(kmin is not None and kmax is not None,
         f"{tag}: no KMIN/KMAX entry for {bn.rate} bit/s in {cpath}. ns-3 would "
         f"NS_ASSERT on this; the map key must equal the link BitRate exactly.")
    row.kmin_bytes, row.kmax_bytes = float(kmin), float(kmax)
    row.regime_model = model.regime(bn, buffer_bytes)

    # -- measured queue ----------------------------------------------------- #
    row.qlen_peak_bytes = float(qlen.port_max[(bn.switch, bn.egress_port)])
    row.qlen_mean_bytes = float(qlen.port_mean[(bn.switch, bn.egress_port)])
    row.qlen_peak_over_kmax = row.qlen_peak_bytes / kmax
    row.qlen_peak_over_ceiling = row.qlen_peak_bytes / row.pfc_ceiling_bytes

    # -- measured flows at the bottleneck ----------------------------------- #
    kv_bn = kv[flowlib.crosses(kv, bn)]
    need(len(kv_bn), f"{tag}: no KV flow crosses {bn}.")
    row.kv_at_bottleneck = len(kv_bn)
    lo, hi = int(kv_bn["start"].min()), int(kv_bn["arrival"].max())
    row.kv_window_ns = hi - lo
    # The floor: this many bytes cannot cross this link faster than this, whatever
    # the buffer, the congestion control or the queue does. On the T1 reference run
    # it is 134.2 ms against a measured 141.9 -- the fabric runs at 94.6% of line
    # rate for the whole transfer, and every regime effect lives in the remaining
    # 5%. That is why the decode gate moves by 2% across a 16x buffer sweep: not
    # noise, arithmetic. Plot the floor next to the measurement or the flat curve
    # looks like a failed experiment instead of a conserved quantity.
    row.kv_bytes_at_bottleneck = float(kv_bn["size"].sum())
    row.kv_floor_ns = row.kv_bytes_at_bottleneck * 8e9 / bn.rate
    row.line_rate_efficiency = row.kv_floor_ns / row.kv_window_ns
    row.concurrency_peak, row.concurrency_mean = \
        flowlib.concurrency_stats(flowlib.intervals(kv_bn))
    sd = kv_bn.loc[kv_bn["slowdown"].notna(), "slowdown"].to_numpy(float)
    need(len(sd), f"{tag}: every KV flow at {bn} has standalone_fct <= 0.")
    row.slowdowns = sd
    row.slow_mean, row.slow_p50 = float(sd.mean()), float(np.percentile(sd, 50))
    row.slow_p99, row.slow_max = float(np.percentile(sd, 99)), float(sd.max())

    # -- measured PFC ------------------------------------------------------- #
    pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
    need(pfc is not None, f"{tag}: pfc.txt unreadable.")
    row.pfc_qidx = pfc.qidx_state
    (row.pfc_pause_pct_of_window, row.pfc_pause_pct_suspect,
     row.pfc_pause_worst_device, n) = pause_pct(pfc, bn, topo, lo, hi)
    row.pfc_paused_devices = n
    if pfc.qidx_state == "MISSING" and row.pfc_pause_pct_suspect > 0:
        warn(f"{tag}: pfc.txt has no qIndex, and the PAUSE/RESUME sequences of "
             f"two priority queues demonstrably interleave on "
             f"{row.pfc_pause_worst_device} (get_pfc drops the qIndex that "
             f"qbb-net-device.cc:383 has in scope). The mis-paired intervals are "
             f"worth {row.pfc_pause_pct_suspect:.2f} of the "
             f"{row.pfc_pause_pct_of_window:.2f} percentage points, so the true "
             f"pause is in [{row.pfc_pause_pct_of_window - row.pfc_pause_pct_suspect:.2f}, "
             f"{row.pfc_pause_pct_of_window:.2f}]%. Apply --print-patch to remove "
             f"the bracket.")

    # -- the model check that matters --------------------------------------- #
    if row.qlen_peak_over_ceiling > 1.0:
        warn(f"{tag}: measured peak egress ({row.qlen_peak_bytes/1e6:.2f} MB) "
             f"exceeds the modelled PFC ceiling ({row.pfc_ceiling_bytes/1e6:.2f} "
             f"MB) by {100*(row.qlen_peak_over_ceiling-1):.0f}%. The ceiling is "
             f"built from the threshold at its FIXED POINT, so it is an upper "
             f"bound and this is impossible. Ruled out already, in the ns-3 "
             f"source: headroom_factor is 3 (common.h:86, and HEADROOM_FACTOR "
             f"overrides it if you set it); ingress and egress account the same "
             f"packets (RemoveFromIngressAdmission runs at dequeue); headroom "
             f"does not overflow (uint64). What is left, and is checkable: "
             f"hdrm_bytes[port][qIndex] is per-queue but its limit headroom[port] "
             f"is per-port, and ConfigNPort reserves ONE per port -- so qCnt "
             f"queues can each claim a full headroom. qlen.txt cannot show it "
             f"because monitor_buffer sums egress_bytes over all queues while "
             f"ShouldSendCN tests one (switch-mmu.cc:103). Emit qIndex from "
             f"monitor_buffer to settle it.")

    # -- the barrier -------------------------------------------------------- #
    for k, v in barrier(kv, placement).items():
        setattr(row, k, v)
    return row


# --------------------------------------------------------------------------- #
# Figures. Four measured, one model. Nothing normalised, nothing fitted.
# --------------------------------------------------------------------------- #
def band_of(topo: Topology, cfg: Ns3Config, bn: Bottleneck) -> tuple[float, float] | None:
    return FabricModel(topo, cfg).flip_band(bn)


def _decorate(fig, s, outdir, name, title, ylabel, band, written, fs=8):
    ax = fig.axes[0]
    logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
    if band:
        lo, hi = band
        ax.axvspan(lo, hi, color="#6a4c93", alpha=0.12, zorder=0,
                   label=f"MODEL: PFC↔DCQCN band ({lo:.1f}–{hi:.1f} MiB)")
        ax.axvline(lo, color="#6a4c93", ls=":", lw=1.0)
        ax.axvline(hi, color="#6a4c93", ls="--", lw=1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    h = sum((a.get_legend_handles_labels()[0] for a in fig.axes), [])
    l = sum((a.get_legend_handles_labels()[1] for a in fig.axes), [])
    ax.legend(h, l, loc="best", fontsize=fs)
    save_fig(fig, outdir, name, written)


def make_plots(rows: list[Row], s: pd.DataFrame, outdir: Path,
               band, bn_label: str, qidx_ok: bool) -> list[Path]:
    written: list[Path] = []
    x = s["buffer_mb"]

    # 01 PFC pause: the regime discriminator, measured ---------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["pfc_pause_pct_of_window"], "o-", color="#1f77b4",
            label=f"PAUSE received by the ingress hosts of {bn_label}")
    sus = s["pfc_pause_pct_suspect"].fillna(0)
    if (sus > 0).any():
        ax.fill_between(x, s["pfc_pause_pct_of_window"] - sus,
                        s["pfc_pause_pct_of_window"], color="#1f77b4", alpha=0.25,
                        label="mis-paired without qIndex (upper bound on the error)")
    caveat = "" if qidx_ok else (f"\npfc.txt has no qIndex: shaded = the "
                                 f"{sus.max():.1f} pp that may be mis-paired")
    _decorate(fig, s, outdir, "01_pfc_pause_vs_buffer.png",
              f"Congestion regime: PFC pause vs buffer{caveat}",
              "Paused fraction of the KV window (%)", band, written)

    # 02 queue vs the ECN band: measured bytes, config constants ------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["qlen_peak_bytes"] / 1e3, "s-", color="#d1495b",
            label="Measured PEAK egress queue")
    ax.plot(x, s["qlen_mean_bytes"] / 1e3, "v--", color="#d1495b", alpha=0.6,
            label="Measured MEAN egress queue")
    for col, ls, nm in (("kmin_bytes", ":", "KMIN"), ("kmax_bytes", "--", "KMAX")):
        v = s[col].unique()
        if len(v) == 1:
            ax.axhline(v[0] / 1e3, color="#2b8a3e", ls=ls, lw=1.2,
                       label=f"{nm} = {v[0]/1e3:g} kB (config.txt)")
    ax.set_yscale("log")
    _decorate(fig, s, outdir, "02_queue_vs_ecn_band.png",
              "Why the regime flips: does the queue ever reach KMAX?\n"
              "(peak below KMIN → ECN never marks → DCQCN is inert by geometry)",
              "Egress queue (kB, log)", band, written)

    # 03 slowdown distribution: no mean/CV, the actual distribution --------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(s["buffer_mb"].to_numpy())
    data = [rows[i].slowdowns for i in order]
    pos = s["buffer_mb"].to_numpy()[order]
    ax.boxplot(data, positions=pos, widths=[b * 0.25 for b in pos],
               showfliers=True, manage_ticks=False,
               medianprops=dict(color="#d1495b"))
    cm = s["concurrency_mean"]
    ax.plot(x, cm, "^--", color="#2b8a3e", lw=1.2,
            label="measured mean concurrency at the bottleneck\n"
                  "(standalone_fct assumes the flow owns the link → this is "
                  "the fair-share reference)")
    _decorate(fig, s, outdir, "03_slowdown_distribution.png",
              f"KV flow slowdown at {bn_label}, full distribution vs buffer",
              "Slowdown (fct / standalone_fct)", band, written)

    # 04 THE HEADLINE. What the KV transfer costs, against what it cannot beat.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["kv_window_ns"] / 1e6, "o-", color="#1f77b4",
            label=f"measured: KV transfer window at {bn_label}")
    floor = s["kv_floor_ns"].mean() / 1e6
    ax.axhline(floor, color="#d1495b", ls="--", lw=1.4,
               label=f"line-rate floor = {s['kv_bytes_at_bottleneck'].mean()/1e9:.2f} GB "
                     f"/ {s['bn_rate_gbps'].iloc[0]:g} Gbps = {floor:.1f} ms")
    ax.plot(x, s["kv_ready_max_ns"] / 1e6, "^:", color="#6b7280", alpha=0.7,
            label="decode-start gate (from t=0, includes the pipeline fill)")
    ax2 = ax.twinx()
    ax2.plot(x, 100 * s["line_rate_efficiency"], "s-.", color="#2a9d5c",
             label="line-rate efficiency = floor / window")
    ax2.set_ylabel("Line-rate efficiency (%)")
    ax2.set_ylim(0, 105)
    _decorate(fig, s, outdir, "04_kv_wait_vs_line_rate_floor.png",
              "What the KV transfer costs decode, against what it cannot beat\n"
              "(the floor is bytes/bandwidth: no buffer moves it, so the gap IS "
              "the whole regime effect)",
              "KV transfer window (ms)", band, written)

    # 04b the skew between decode ranks: a cost of the transfer pattern, not of bytes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["kv_ready_max_ns"] / 1e6, "o-", color="#1f77b4",
            label="last decode rank ready (= the gate)")
    ax.plot(x, s["kv_ready_min_ns"] / 1e6, "o:", color="#1f77b4", alpha=0.45,
            label="first decode rank ready")
    ax.fill_between(x, s["kv_ready_min_ns"] / 1e6, s["kv_ready_max_ns"] / 1e6,
                    color="#1f77b4", alpha=0.12,
                    label="cross-rank skew: ranks idle, waiting for the slowest")
    ax2 = ax.twinx()
    ax2.plot(x, s["concurrency_mean"], "d-.", color="#d98a00",
             label="mean concurrent KV flows on the bottleneck\n"
                   "(per-layer emission: the stream competes with itself)")
    ax2.set_ylabel("Concurrent KV flows")
    _decorate(fig, s, outdir, "04b_skew_and_self_contention.png",
              "Costs of the transfer PATTERN, not of the byte count",
              "Decode rank ready (ms)", band, written)

    # 05 THE MODEL. Topology + config only. Kept apart on purpose. ---------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["pfc_thresh_bytes"] / 1e3, "o-", color="#1f77b4",
            label="PFC threshold at its fixed point, A/(2^s+F), per ingress port")
    ax.plot(x, s["pfc_thresh_x_fports_bytes"] / 1e3, "D--", color="#1f77b4",
            label=f"  × F_ports = {s['f_ports'].iloc[0]:g} (egress-equivalent)")
    err = s["naive_error_pct"].abs().max()
    ax.plot(x, s["pfc_thresh_naive_bytes"] / 1e3, "x:", color="#b0b0b0",
            label=f"buffer/8 (naive — off by up to {err:.0f}%)")
    ax.plot(x, s["pfc_ceiling_bytes"] / 1e3, "*-.", color="#d1495b",
            label="PFC egress ceiling Σ(reserve+thresh+headroom)")
    ax.plot(x, s["qlen_peak_bytes"] / 1e3, "s-", color="#000000", alpha=0.55,
            label="measured peak egress (the only measured line here)")
    for col, ls, nm in (("kmin_bytes", ":", "KMIN"), ("kmax_bytes", "--", "KMAX")):
        v = s[col].unique()
        if len(v) == 1:
            ax.axhline(v[0] / 1e3, color="#2b8a3e", ls=ls, lw=1.2,
                       label=f"{nm} = {v[0]/1e3:g} kB")
    ax.set_yscale("log")
    _decorate(fig, s, outdir, "05_model_pfc_threshold_vs_ecn.png",
              "MODEL (topology + config only): dynamic PFC threshold vs the ECN band\n"
              "band = where F_ports×threshold crosses KMIN and KMAX",
              "Bytes (kB, log)", band, written, fs=7)
    return written


# --------------------------------------------------------------------------- #
REPORT = ["buffer_mb", "regime_model", "pfc_pause_pct_of_window",
          "qlen_peak_over_kmax", "kv_window_ns", "kv_floor_ns",
          "line_rate_efficiency", "concurrency_mean", "slow_mean",
          "kv_ready_max_ns", "cross_rank_skew_ns"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    roles.add_argument(ap)
    ap.add_argument("--bottleneck", default=None,
                    help="'sw->peer', e.g. '8->12'. Default: measured (deepest "
                         "queue in qlen.txt), and required to be identical on "
                         "every run.")
    ap.add_argument("--headroom-factor", type=int, default=3,
                    help="ns-3 common.h::headroom_factor. NOT in config.txt: it "
                         "is compiled in, so this is an ASSERTION and every "
                         "threshold scales with it (default 3).")
    ap.add_argument("--print-patch", action="store_true")
    a = ap.parse_args(argv)

    if a.print_patch:
        print(ns3.PFC_QIDX_PATCH)
        return 0

    try:
        p, outdir = paths.from_arguments(a, KIND)
        need(not p.missing_roots(),
             "derived root(s) do not exist:\n    " + "\n    ".join(p.missing_roots())
             + f"\n  --sweep {a.sweep!r} is probably wrong.")
        placement = Placement.parse(a.placement)
        tags = p.tags("ns3")
        need(tags, f"no run sub-directory under {p.ns3_root}")

        print(p.describe())
        print(f"  out      {outdir}")
        print(f"  placement\n{placement.describe()}")
        print(f"  headroom_factor = {a.headroom_factor}  (ASSERTED, not read)\n")
        # The placement is the one assumption nothing else can catch: get it wrong
        # and every rank-dependent number stays plausible while describing a
        # different machine. MLSynth already wrote it into the ASTRA op names, so
        # compare rather than trust.
        if (ad := p.astra_run(tags[0])).is_dir():
            if msg := roles.cross_check(placement, ad):
                warn(msg)
        else:
            warn(f"no ASTRA run at {ad}: --placement cannot be cross-checked "
                 f"against the trace and is taken on trust.")

        print(f"Analysing {len(tags)} runs:")

        rows = [analyse(t, p, placement, a.headroom_factor, a.bottleneck)
                for t in tags]

        bns = {r.bottleneck for r in rows}
        need(len(bns) == 1,
             f"the deepest queue is not on the same link on every run: {sorted(bns)}. "
             f"The per-run model numbers would come from different switches and "
             f"the curves would be stitched together from incomparable runs. "
             f"Pass --bottleneck to fix one.")
        fps = {r.f_ports for r in rows}
        need(len(fps) == 1, f"F_ports differs across runs: {sorted(fps)}.")

        s = pd.DataFrame([r.flat() for r in rows]).sort_values("buffer_mb")
        s = s.reset_index(drop=True)
        rows = sorted(rows, key=lambda r: r.buffer_mb)

        # the band depends only on (switch, F_ports, shift, KMIN/KMAX), all of
        # which are now known to be constant -- so one band for the whole sweep.
        t0 = parse_topology(p.topology(rows[0].tag), a.headroom_factor)
        c0 = parse_ns3_config(p.config(rows[0].tag))
        sw, peer = (int(v) for v in rows[0].bottleneck.split("->"))
        bn0 = Bottleneck(sw, t0.port_facing(sw, peer), peer, t0.ports[sw][
            t0.port_facing(sw, peer)].rate,
            tuple(int(i) for i in rows[0].ingress_ports.split(",")))
        band = band_of(t0, c0, bn0)

        outdir.mkdir(parents=True, exist_ok=True)
        s.to_csv(outdir / "summary.csv", index=False)
        pd.concat([pd.DataFrame({"buffer_mb": r.buffer_mb,
                                 "slowdown": r.slowdowns}) for r in rows],
                  ignore_index=True).to_csv(outdir / "slowdowns.csv", index=False)
        plots = make_plots(rows, s, outdir, band, rows[0].bottleneck,
                           rows[0].pfc_qidx == "present")

        pd.set_option("display.width", 220)
        print("\n================ BUFFER SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))
        if band:
            print(f"\nMODEL (topology + config only): PFC below {band[0]:.2f} MiB, "
                  f"DCQCN above {band[1]:.2f} MiB, at {rows[0].bottleneck} "
                  f"(F_ports={rows[0].f_ports:g}).")
            print("Agreement with the measured pause is the result; disagreement "
                  "is the finding.")
        print(f"\nWrote {outdir}:")
        for f in ["summary.csv", "slowdowns.csv", *[q.name for q in plots]]:
            print(f"  {f}")
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