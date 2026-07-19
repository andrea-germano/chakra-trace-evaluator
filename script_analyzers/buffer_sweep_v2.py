#!/usr/bin/env python3
"""
buffer_sweep — how does performance move as the per-switch buffer grows, on an
oversubscribed fabric?

This is a deliberate rewrite. The previous version answered a *mechanism*
question ("is the queue held by PFC or by DCQCN?") and to do it drew the modelled
PFC threshold against the ECN band, egress ceilings, naive-vs-real thresholds and
a full slowdown box plot — six dense figures, most of them about the regime and
not about what the buffer *costs*. Useful once, but not the thing you put in
front of someone to say "here is what changing the buffer does".

So this version answers only the outcome question, and answers it with five
figures, each of which makes exactly one point:

    01  PERFORMANCE          two zoomed panels, each on its own scale: TTFT (= end
                             of prefill, the first token; compute-bound, ~flat) and
                             the decode start (= the KV-arrival gate, when token 2
                             can begin; fabric-bound, falls with the buffer). Their
                             difference — annotated on the second panel — is what
                             transferring the KV cache over the oversubscribed
                             fabric costs. NB: TTFT is the end of prefill, NOT
                             DECFB — see ttft_end_of_prefill().
    02  KV ARRIVAL SKEW      cross-rank skew of KV readiness vs buffer: how long
                             the luckiest decode rank sits idle waiting for the
                             slowest. A cost of the transfer pattern, not of bytes.
    03  PAUSES ↔ THROUGHPUT  number of PFC PAUSE frames per run, and the effective
                             (delivered) KV bandwidth, vs buffer — plus a scatter
                             of one against the other. Does backpressure track a
                             throughput loss?
    04  QUEUE(t) PER SWITCH  egress-queue occupancy over time as a small-multiples
                             grid: rows = switches, columns = buffer size, one
                             filled trace per cell (no overlapping lines). Where
                             the queue builds, and whether a bigger buffer just
                             gets filled.
    05  OCCUPANCY vs BUFFER  peak / mean queue as a % of the buffer, per switch.
                             The one-number version of 04: on an oversubscribed
                             link, does added buffer get used or wasted?

Everything is measured. Nothing is fitted, normalised or modelled. Numbers come
from the ns-3 outputs (fct.txt / pfc.txt / qlen.txt) and, for token latency, from
this run's ns-3-backed ASTRA-sim trace. The heavy fabric model (utils.fabric
FabricModel) is intentionally not used here; run `python3 -m utils.fabric <topo>
<config>` or the old regime analyzer if you want the PFC↔DCQCN prediction.

Declared, never inferred (unchanged from before):
    --sweep       the one path input; every other path is derived (utils.paths).
    --placement   the rank->role map (utils.roles); it defines the decode ranks,
                  which are the barrier population behind the KV-arrival gate.
    --bottleneck  optional 'sw->peer'; default is the deepest queue in qlen.txt,
                  required to be the same link on every run or the sweep aborts.

Usage
-----
    python3 buffer_sweep.py --sweep buffer_sweep_T1
    python3 buffer_sweep.py --sweep buffer_sweep_T1 --placement "p0=0,1 p1=2,3 d0=4,5 d1=6,7"
    python3 buffer_sweep.py --sweep buffer_sweep_T1 --bottleneck "8->12" -o /tmp/x
"""

from __future__ import annotations

import argparse
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
from utils import ns3, paths, pp, roles
from utils.fabric import Bottleneck, Topology, parse_ns3_config, parse_topology
from utils.plots import logx_pow2, save_fig
from utils.roles import Placement
from utils.paths import BUFFER_AXIS

NAN = float("nan")
KIND = "buffer"
MS = 1e-6                     # ns -> ms

# palette (kept small and consistent across figures)
BLUE, CORAL, GREEN, VIOLET, MUTED = \
    "#1f77b4", "#d1495b", "#2b8a3e", "#6a4c93", "#9aa0a6"


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
# One row per run. Deliberately small: only what the five figures need.
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    tag: str = ""
    buffer_mb: float = NAN
    buffer_bytes: float = NAN
    bottleneck: str = ""
    bn_rate_gbps: float = NAN

    # -- 01 performance ------------------------------------------------------ #
    ttft_ns: float = NAN                  # token 1 = END OF PREFILL (FIRSTTOK send)
    kv_gate_ns: float = NAN               # decode start = last KV arrival (2nd token)
    kv_ready_min_ns: float = NAN          # first decode rank ready

    # -- 02 skew ------------------------------------------------------------- #
    cross_rank_skew_ns: float = NAN       # max(ready) - min(ready) ACROSS ranks
    kv_stream_duration_ns: float = NAN    # max-min arrival WITHIN a rank (stagger)
    decode_ranks: str = ""

    # -- 02b PP arrival skew (the CAUSE upstream of the KV skew) ------------- #
    pp_skew_ns: float = NAN               # worst wave cross-rank PP arrival skew
    pp_skew_mean_ns: float = NAN          # mean over waves
    pp_first_ns: float = NAN              # earliest PP arrival (worst wave)
    pp_last_ns: float = NAN               # latest PP arrival (worst wave)
    pp_stage: object = None               # dst stage of the worst wave
    pp_n_waves: int = 0
    # not flattened: per-wave and per-flow frames for the PP figures.
    pp_waves: object = field(default=None)      # DataFrame or None
    pp_arrivals: object = field(default=None)   # DataFrame or None
    kv_fct_mean_ns: float = NAN           # mean KV flow FCT (for the flat-gate fig)

    # -- 03 pauses / throughput --------------------------------------------- #
    pause_frames_bn: float = NAN          # PAUSE frames at the bottleneck's victims
    pause_frames_total: float = NAN       # PAUSE frames anywhere in the run
    paused_devices: float = NAN
    pause_pct_of_window: float = NAN      # worst victim, % of KV window under pause
    kv_bytes_at_bottleneck: float = NAN
    kv_window_ns: float = NAN             # first KV posted -> last arrived (at bn)
    kv_floor_ns: float = NAN              # bytes / bottleneck rate: the hard floor
    kv_delivered_gbps: float = NAN        # bytes*8 / window
    line_rate_efficiency: float = NAN     # floor / window in [0, 1]

    # -- 04/05 queue occupancy summary (per bottleneck port) ---------------- #
    qpeak_bytes: float = NAN
    qmean_bytes: float = NAN

    # not flattened: per-switch downsampled series + per-switch peak/mean.
    qseries: dict = field(default_factory=dict)       # sw -> (ts_ns, bytes)
    qswitch_peak: dict = field(default_factory=dict)  # sw -> peak total bytes
    qswitch_mean: dict = field(default_factory=dict)  # sw -> mean total bytes

    def flat(self) -> dict:
        d = asdict(self)
        for k in ("qseries", "qswitch_peak", "qswitch_mean",
                  "pp_waves", "pp_arrivals"):
            d.pop(k, None)
        return d


# --------------------------------------------------------------------------- #
# Measurement helpers (barrier + token latency carried over unchanged in spirit;
# pause accounting simplified to a robust frame count + a window fraction).
# --------------------------------------------------------------------------- #
def union_len(spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Measure of the union of `spans` clipped to [lo, hi]. Queues of one device
    overlap in time, so their paused wall-clock is a union, never a sum."""
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


def pause_stats(pfc: ns3.PfcLog, bn: Bottleneck, topo: Topology,
                lo: int, hi: int) -> dict:
    """Backpressure on the ingress side of THIS bottleneck, over the KV window.

    Two quantities, both about the same devices (the ports upstream of `bn`,
    which are the ones PAUSEd when its ingress fills — QbbNetDevice::Receive fires
    the trace on the victim, so those are the keys to look for in pfc.txt):

        pause_frames   count of PAUSE events (typ==1) inside [lo, hi]. A pure
                       count, so it does NOT depend on PAUSE/RESUME pairing and is
                       unaffected by the missing-qIndex problem. This is "how many
                       pauses did this run have", the thing figure 03 plots.
        pct_of_window  worst victim's paused wall-clock (unioned across its queues,
                       clipped to the window) as a fraction of the window. This
                       one DOES pair PAUSE with RESUME, so without qIndex it is
                       approximate; it is context, not the headline.

    The frame count is also summed across every device in the file (`total`) so a
    run whose backpressure sits on some *other* link is not silently reported as
    pause-free at the bottleneck.
    """
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


def barrier(kv: pd.DataFrame, placement: Placement) -> dict:
    """The first decode step cannot start until every KV flow feeding a decode
    rank has arrived. Per rank, KV-ready is that rank's LATEST arrival; the gate
    is the worst rank.

        kv_gate_ns             max over ranks of (latest arrival): the decode start.
        cross_rank_skew_ns     max(ready) - min(ready) ACROSS ranks. A real skew:
                               how much earlier the luckiest rank could have gone.
        kv_stream_duration_ns  max-min arrival WITHIN one rank — the per-layer KV
                               stream staggered by prefill compute, NOT a skew.
    """
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
    """TTFT = the first token, produced at the END OF PREFILL.

    In disaggregated inference the prefill pool computes the whole prompt and
    samples token 1 from the last stage's logits; MLSynth marks that instant with
    the FIRSTTOK handoff (prefill last stage -> decode stage 0), so FIRSTTOK's
    SEND start_tick is the moment token 1 exists. Fallback when a run has no
    FIRSTTOK: the latest prefill COMP end, which is the same instant.

    This is deliberately NOT DECFB. DECFB carries a token PRODUCED BY DECODE
    (token 2 onward) being fed back to stage 0, so keying TTFT on DECFB(it=0) — as
    the previous version did — actually reported the SECOND token, one full decode
    pipeline late, and made TTFT look artificially buffer-sensitive.

    The decode-start / second-token time is not computed here: it is the
    KV-arrival gate (kv_gate_ns in barrier()), measured from fct.txt, because it
    is gated by KV *receiving* and nothing in the ASTRA trace moves it."""
    adir = p.astra_run(tag)
    if not adir.is_dir():
        warn(f"{tag}: no ASTRA run at {adir}; TTFT (end of prefill) unavailable.")
        return {}
    df = astra.read_run(adir)
    if df is None:
        warn(f"{tag}: no readable stats_sys*.csv under {adir}; TTFT unavailable.")
        return {}
    ft = df.loc[(df["op_class"] == "FIRSTTOK") & (df["comm_role"] == "send"),
                "start_tick"]
    if len(ft):
        return {"ttft_ns": float(ft.max())}
    pre = df.loc[(df["op_class"] == "COMP") & (df["phase"] == "prefill"),
                 "end_tick"]
    if len(pre):
        warn(f"{tag}: no FIRSTTOK in the ASTRA trace; using the last prefill "
             f"compute end as end-of-prefill TTFT.")
        return {"ttft_ns": float(pre.max())}
    warn(f"{tag}: no FIRSTTOK and no prefill COMP in the ASTRA trace; TTFT "
         f"unavailable.")
    return {}


def _downsample(ts, ys, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Max-per-bucket downsample of a queue time series. The peak in each bucket
    is kept (not the mean): the excursions are the point of the plot, and there
    are far more 100 ns samples than the figure has pixels."""
    ts = np.asarray(ts, float)
    ys = np.asarray(ys, float)
    if len(ts) <= n or ts[-1] <= ts[0]:
        return ts, ys
    edges = np.linspace(ts[0], ts[-1], n + 1)
    idx = np.clip(np.searchsorted(edges, ts) - 1, 0, n - 1)
    hi = np.full(n, -1.0)
    np.maximum.at(hi, idx, ys)
    keep = hi >= 0
    centres = (edges[:-1] + edges[1:]) / 2
    return centres[keep], hi[keep]


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyse(tag: str, p: paths.SweepPaths, placement: Placement,
            bn_force: str | None) -> Row:
    buf = BUFFER_AXIS.value(tag)
    need(buf is not None, f"{tag}: no 'buf<num>' token in the directory name; "
                          f"the swept axis is unreadable.")
    tpath, cpath = p.topology(tag), p.config(tag)
    ns3_dir = p.ns3_run(tag)
    for f in (tpath, cpath, ns3_dir / "fct.txt", ns3_dir / "pfc.txt",
              ns3_dir / "qlen.txt"):
        need(f.exists(), f"{tag}: missing {f}")

    topo = parse_topology(tpath)
    cfg = parse_ns3_config(cpath)
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    need(cfg.buffer_mb is not None,
         f"{tag}: no BUFFER_SIZE in {cpath}. If this is the template dir, --sweep "
         f"points at the template rather than the generated configs.")
    need(abs(cfg.buffer_mb - buf) < 1e-6,
         f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but 'buf{buf:g}' "
         f"in the directory name. One of the two is lying.")

    row = Row(tag=tag, buffer_mb=float(buf),
              buffer_bytes=float(buf) * 1024 * 1024)

    # -- TTFT = end of prefill (best effort; NaN -> figure 01 shows only the
    #    KV-gated decode start, which always comes from fct.txt) -------------- #
    for k, v in ttft_end_of_prefill(tag, p).items():
        setattr(row, k, v)

    # -- flows --------------------------------------------------------------- #
    raw = ns3.read_fct(ns3_dir / "fct.txt")
    need(raw is not None and len(raw), f"{tag}: fct.txt has no parsable rows.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)
    for w in roles.check(f, placement):
        warn(f"{tag}: {w}")
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{tag}: no KV flow after classification.")

    # -- the bottleneck (measured, or forced) -------------------------------- #
    qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=True)
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
                     f"port; the ingress (victim) set would be empty.")
    row.bottleneck, row.bn_rate_gbps = str(bn), bn.rate / 1e9

    # -- throughput at the bottleneck ---------------------------------------- #
    kv_bn = kv[flowlib.crosses(kv, bn)]
    need(len(kv_bn), f"{tag}: no KV flow crosses {bn}.")
    lo, hi = int(kv_bn["start"].min()), int(kv_bn["arrival"].max())
    row.kv_window_ns = hi - lo
    row.kv_bytes_at_bottleneck = float(kv_bn["size"].sum())
    row.kv_floor_ns = row.kv_bytes_at_bottleneck * 8e9 / bn.rate
    row.kv_delivered_gbps = (row.kv_bytes_at_bottleneck * 8.0 /
                             row.kv_window_ns if row.kv_window_ns > 0 else NAN)
    row.line_rate_efficiency = (row.kv_floor_ns / row.kv_window_ns
                                if row.kv_window_ns > 0 else NAN)

    # -- pauses -------------------------------------------------------------- #
    pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
    need(pfc is not None, f"{tag}: pfc.txt unreadable.")
    if pfc.qidx_state == "MISSING":
        warn(f"{tag}: pfc.txt has no qIndex; pause_pct_of_window is approximate "
             f"(see ns3.PFC_QIDX_PATCH). The pause frame COUNT is unaffected.")
    for k, v in pause_stats(pfc, bn, topo, lo, hi).items():
        setattr(row, k, v)

    # -- the barrier --------------------------------------------------------- #
    for k, v in barrier(kv, placement).items():
        setattr(row, k, v)

    # -- PP arrival skew: the CAUSE upstream of the KV/TP (measured on the
    #    fabric, from the same annotated fct frame) --------------------------- #
    ppr = pp.measure(f, placement)
    if not ppr.available:
        warn(f"{tag}: no inter-stage PP-prefill flow found; PP skew figures "
             f"will be empty. (PP=1, or placement has one prefill stage.)")
    row.pp_skew_ns = ppr.skew_ns
    row.pp_skew_mean_ns = ppr.skew_mean_ns
    row.pp_first_ns = ppr.first_ns
    row.pp_last_ns = ppr.last_ns
    row.pp_stage = ppr.stage
    row.pp_n_waves = ppr.n_waves
    row.pp_waves = ppr.waves
    row.pp_arrivals = ppr.arrivals

    # -- mean KV FCT: the aggregate that DOES move with the buffer, to sit
    #    against the flat gate (fig 07) -------------------------------------- #
    row.kv_fct_mean_ns = float(kv["fct"].mean()) if len(kv) else NAN

    # -- queue occupancy: bottleneck port summary + per-switch series -------- #
    row.qpeak_bytes = float(qlen.port_max[(bn.switch, bn.egress_port)])
    row.qmean_bytes = float(qlen.port_mean[(bn.switch, bn.egress_port)])
    for sw, (ts, ys) in qlen.switch_series.items():
        if not ts:
            continue
        row.qswitch_peak[sw] = float(qlen.switch_total_max.get(sw, max(ys)))
        row.qswitch_mean[sw] = float(np.mean(ys))
        row.qseries[sw] = _downsample(ts, ys)
    return row


def _zoom_y(ax, series, pad: float = 0.15) -> None:
    """Autoscale one panel's y-axis to its own data, with a small margin — and a
    visible band when the series is flat, so a constant line (e.g. a compute-bound
    TTFT that does not move with the buffer) reads as flat instead of collapsing
    onto the frame."""
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
               bn_label: str) -> list[Path]:
    written: list[Path] = []
    x = s["buffer_mb"]

    # 01 PERFORMANCE: two zoomed panels, each on its own y-scale ----------- #
    # On one shared axis the flat TTFT and the moving decode-start crush each
    # other; split so each trend fills its panel.
    fig, (axP, axD) = plt.subplots(1, 2, figsize=(13, 5))

    # panel A — TTFT = end of prefill (token 1). Compute-bound, so ~flat.
    if s["ttft_ns"].notna().any():
        axP.plot(x, s["ttft_ns"] * MS, "s-", color=CORAL,
                 label="TTFT (token 1)")
        _zoom_y(axP, s["ttft_ns"] * MS)
    else:
        axP.text(0.5, 0.5, "no ASTRA trace:\nTTFT unavailable", ha="center",
                 va="center", transform=axP.transAxes, color=MUTED)
    logx_pow2(axP, s, "buffer_mb", "Per-switch buffer (MiB)")
    axP.set_ylabel("TTFT (ms)")
    axP.set_title("TTFT = end of prefill (token 1)\n"
                  "compute-bound → ~flat across the buffer")
    axP.grid(True, alpha=0.3, which="both")

    # panel B — decode start = last KV arrival (token 2 begins). Fabric-bound.
    axD.plot(x, s["kv_gate_ns"] * MS, "o-", color=BLUE,
             label="decode start (token 2 begins)")
    _zoom_y(axD, s["kv_gate_ns"] * MS)
    logx_pow2(axD, s, "buffer_mb", "Per-switch buffer (MiB)")
    axD.set_ylabel("Decode start (ms)")
    axD.set_title("Decode start = last KV arrival (KV-gated)\n"
                  "fabric-bound → falls as the buffer absorbs the incast")
    axD.grid(True, alpha=0.3, which="both")

    # keep the key number (what the fabric costs) visible without a shared axis
    if s["ttft_ns"].notna().any():
        cost = (s["kv_gate_ns"] - s["ttft_ns"]) * MS
        axD.text(0.02, 0.02,
                 f"KV transfer cost (decode start − TTFT): "
                 f"{cost.min():.1f}–{cost.max():.1f} ms",
                 transform=axD.transAxes, fontsize=8, color=MUTED,
                 va="bottom", ha="left")

    fig.suptitle("Prefill finish vs decode start — each on its own scale", y=1.02)
    save_fig(fig, outdir, "01_performance_vs_buffer.png", written)

    # 02 KV ARRIVAL SKEW ---------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, s["cross_rank_skew_ns"] * MS, "o-", color=BLUE,
            label="cross-rank skew = last-ready − first-ready")
    ax.fill_between(x, 0, s["cross_rank_skew_ns"] * MS, color=BLUE, alpha=0.12,
                    label="decode ranks idle, waiting for the slowest")
    ax2 = ax.twinx()
    ax2.plot(x, s["kv_stream_duration_ns"] * MS, "d:", color=MUTED,
             label="within-rank KV stream duration (stagger, not a skew)")
    ax2.set_ylabel("Within-rank stream duration (ms)")
    _finish(fig, ax, s, outdir, "02_kv_arrival_skew_vs_buffer.png",
            "KV-arrival skew across decode ranks vs buffer\n"
            "(idle time the barrier imposes on all but the slowest rank)",
            "Cross-rank skew (ms)", written, extra_axes=(ax2,))

    # 03 PAUSES <-> EFFECTIVE BANDWIDTH ------------------------------------- #
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # left: both trends against the common driver (buffer)
    axL.plot(x, s["pause_frames_bn"], "o-", color=CORAL,
             label=f"PAUSE frames at ingress of {bn_label}")
    if (s["pause_frames_total"] > s["pause_frames_bn"]).any():
        axL.plot(x, s["pause_frames_total"], "o:", color=CORAL, alpha=0.4,
                 label="PAUSE frames anywhere in the run")
    axLr = axL.twinx()
    axLr.plot(x, 100 * s["line_rate_efficiency"], "s-", color=GREEN,
              label="effective KV bandwidth (% of line rate)")
    axLr.set_ylabel("Effective KV bandwidth (% of line rate)")
    axLr.set_ylim(0, 105)
    logx_pow2(axL, s, "buffer_mb", "Per-switch buffer (MiB)")
    axL.set_ylabel("PFC PAUSE frames (count)")
    axL.set_title("Backpressure and throughput vs buffer")
    axL.grid(True, alpha=0.3, which="both")
    h, l = axL.get_legend_handles_labels()
    h2, l2 = axLr.get_legend_handles_labels()
    axL.legend(h + h2, l + l2, loc="best", fontsize=8)

    # right: the direct relationship, colour = buffer
    bnorm = matplotlib.colors.LogNorm(vmin=s["buffer_mb"].min(),
                                      vmax=s["buffer_mb"].max())
    sc = axR.scatter(s["pause_frames_bn"], 100 * s["line_rate_efficiency"],
                     c=s["buffer_mb"], cmap="viridis", s=90, norm=bnorm, zorder=3)
    for _, r in s.iterrows():
        axR.annotate(f"{r['buffer_mb']:g}",
                     (r["pause_frames_bn"], 100 * r["line_rate_efficiency"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    cb = fig.colorbar(sc, ax=axR)
    cb.set_label("Buffer (MiB)")
    axR.set_xlabel("PFC PAUSE frames at the bottleneck (count)")
    axR.set_ylabel("Effective KV bandwidth (% of line rate)")
    axR.set_title("Does backpressure cost throughput?")
    axR.grid(True, alpha=0.3)
    save_fig(fig, outdir, "03_pauses_vs_effective_bandwidth.png", written)

    # 04 QUEUE OCCUPANCY OVER TIME: grid, rows=switch, cols=buffer ---------- #
    # Overlaying one line per buffer per switch made same-shaped curves sit on
    # top of each other. A small-multiples grid removes the overlap entirely:
    # read a ROW to see a switch's queue shrink/grow with buffer, a COLUMN to
    # compare switches at one buffer. y is shared within a row (per switch), so
    # the bottleneck row keeps its larger scale without flattening the others.
    switches = sorted({sw for r in rows for sw in r.qseries})
    if switches:
        runs = sorted(rows, key=lambda r: r.buffer_mb)
        bufs = [r.buffer_mb for r in runs]
        cmap = plt.get_cmap("viridis")
        cnorm = (matplotlib.colors.LogNorm(vmin=min(bufs), vmax=max(bufs))
                 if len(set(bufs)) > 1 else None)
        bn_sw = int(bn_label.split("->")[0])
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
                    t = np.asarray(ts) * MS
                    y = np.asarray(ys) / 1e3
                    a.fill_between(t, y, color=col, alpha=0.85, lw=0)
                    a.plot(t, y, color="#222222", lw=0.5, alpha=0.6)
                a.grid(True, alpha=0.2)
                if i == 0:
                    a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
                if j == 0:
                    mark = "\n(bottleneck)" if sw == bn_sw else ""
                    a.set_ylabel(f"switch {sw}{mark}\nqueue (kB)", fontsize=8)
                if i == nrows - 1:
                    a.set_xlabel("time (ms)", fontsize=8)
                    a.locator_params(axis="x", nbins=4)
        fig.suptitle("Egress-queue occupancy over time — rows = switches, "
                     "columns = buffer size\n(per-switch y shared across buffers; "
                     "no overlapping lines)", y=1.01)
        save_fig(fig, outdir, "04_queue_occupancy_timeseries_per_switch.png",
                 written)

    # 05 OCCUPANCY vs BUFFER, PER SWITCH (% of capacity) -------------------- #
    if switches:
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("tab10")
        for i, sw in enumerate(switches):
            xs, peak, mean = [], [], []
            for r in sorted(rows, key=lambda r: r.buffer_mb):
                if sw not in r.qswitch_peak or not r.buffer_bytes:
                    continue
                xs.append(r.buffer_mb)
                peak.append(100 * r.qswitch_peak[sw] / r.buffer_bytes)
                mean.append(100 * r.qswitch_mean[sw] / r.buffer_bytes)
            if not xs:
                continue
            c = cmap(i % 10)
            tag = " (bottleneck)" if sw == int(bn_label.split("->")[0]) else ""
            ax.plot(xs, peak, "o-", color=c, label=f"switch {sw}{tag} — peak")
            ax.plot(xs, mean, "v--", color=c, alpha=0.5,
                    label=f"switch {sw}{tag} — mean")
        _finish(fig, ax, s, outdir, "05_queue_occupancy_vs_buffer.png",
                "Is the extra buffer used? Peak / mean egress occupancy as a %\n"
                "of the buffer, per switch (flat-and-low means added buffer is wasted)",
                "Queue occupancy (% of buffer)", written)

    # 06 PP ARRIVAL SKEW vs BUFFER — the CAUSE ------------------------------ #
    # The KV skew of fig 02 is downstream. This is the driver: how far apart the
    # ranks of a stage see the SAME PP activation. RS is gated on the local wake,
    # AG on max(this skew, W); everything the buffer does to TP/KV enters here.
    if s["pp_skew_ns"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, s["pp_skew_ns"] * MS, "o-", color=VIOLET,
                label="worst-wave PP arrival skew  Δ = last − first")
        if s["pp_skew_mean_ns"].notna().any() and \
           not np.allclose(s["pp_skew_mean_ns"].dropna().to_numpy(),
                           s["pp_skew_ns"].dropna().to_numpy()):
            ax.plot(x, s["pp_skew_mean_ns"] * MS, "d:", color=MUTED,
                    label="mean over waves")
        ax.fill_between(x, 0, s["pp_skew_ns"] * MS, color=VIOLET, alpha=0.12,
                        label="skew the receiving stage's all-reduce inherits")
        _finish(fig, ax, s, outdir, "06_pp_arrival_skew_vs_buffer.png",
                "Pipeline-parallel activation arrival skew vs buffer\n"
                "(Δ that gates the receiving stage's all-reduce — the cause "
                "upstream of the TP/KV variation)",
                "PP cross-rank arrival skew (ms)", written)

    # 07 FLAT GATE vs MOVING MEAN — the counter-intuitive result ------------ #
    # kv_gate_ns (the decode barrier) is bandwidth/volume bound and stays put;
    # the mean KV FCT rises with the buffer (bufferbloat under DCQCN). Same axis,
    # two scales: the aggregate moves, the gate does not.
    if s["kv_fct_mean_ns"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, s["kv_gate_ns"] * MS, "o-", color=BLUE,
                label="decode gate = max KV arrival (flat: volume/bandwidth)")
        ax2 = ax.twinx()
        ax2.plot(x, s["kv_fct_mean_ns"] * MS, "s-", color=CORAL,
                 label="mean KV flow FCT (rises: bufferbloat)")
        ax2.set_ylabel("Mean KV flow FCT (ms)")
        _finish(fig, ax, s, outdir, "07_flat_gate_vs_moving_mean.png",
                "The decode gate is invariant while the mean KV FCT grows\n"
                "(the CC regime moves the aggregate, not the barrier)",
                "Decode gate = max KV arrival (ms)", written, extra_axes=(ax2,))

    return written


# --------------------------------------------------------------------------- #
REPORT = ["buffer_mb", "ttft_ns", "kv_gate_ns", "pp_skew_ns",
          "cross_rank_skew_ns", "kv_fct_mean_ns", "pause_frames_bn",
          "pause_pct_of_window", "line_rate_efficiency",
          "kv_delivered_gbps", "kv_window_ns"]


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
                    help="'sw->peer', e.g. '8->12'. Default: measured (deepest "
                         "queue in qlen.txt), required identical on every run.")
    a = ap.parse_args(argv)

    try:
        p = paths.SweepPaths(sweep=a.sweep, workload=a.workload, root=Path(a.root))
        outdir = (Path(a.out) if a.out else
                  p.root / "results" / "sweep_analysis" / KIND / p.workload / p.sweep)
        need(not p.missing_roots(),
             "derived root(s) do not exist:\n    "
             + "\n    ".join(p.missing_roots())
             + f"\n  --sweep {a.sweep!r} is probably wrong.")
        placement = Placement.parse(a.placement)
        tags = p.tags("ns3")
        need(tags, f"no run sub-directory under {p.ns3_root}")

        print(p.describe())
        print(f"  out      {outdir}")
        print(f"  placement\n{placement.describe()}\n")
        # placement is the one assumption nothing else can catch: cross-check it
        # against the names MLSynth already wrote into the ASTRA trace.
        if (ad := p.astra_run(tags[0])).is_dir():
            if msg := roles.cross_check(placement, ad):
                warn(msg)
        else:
            warn(f"no ASTRA run at {ad}: --placement is taken on trust.")

        # one moving knob only: two would join points from different fabrics in
        # buffer order and draw a line that is really a zigzag.
        variants = {BUFFER_AXIS.variant(t) for t in tags}
        need(len(variants) == 1,
             f"this sweep moves more than one knob: variants {sorted(variants)}. "
             f"Split into one sweep per variant.")

        print(f"Analysing {len(tags)} runs:")
        rows = [analyse(t, p, placement, a.bottleneck) for t in tags]
        for r in sorted(rows, key=lambda r: r.buffer_mb):
            print(f"  + buf{r.buffer_mb:<5g} bn={r.bottleneck:<8} "
                  f"ttft={r.ttft_ns*MS:6.2f}ms gate={r.kv_gate_ns*MS:6.2f}ms  "
                  f"pause_frames={r.pause_frames_bn:<5g} "
                  f"eff={100*r.line_rate_efficiency:5.1f}%")

        bns = {r.bottleneck for r in rows}
        need(len(bns) == 1,
             f"the deepest queue is not on the same link on every run: "
             f"{sorted(bns)}. The curves would stitch together incomparable "
             f"switches. Pass --bottleneck to fix one.")

        rows = sorted(rows, key=lambda r: r.buffer_mb)
        s = (pd.DataFrame([r.flat() for r in rows])
             .sort_values("buffer_mb").reset_index(drop=True))

        if outdir.exists():
            shutil.rmtree(outdir)   # a stale figure next to a fresh one is
                                    # indistinguishable from a real result.
        outdir.mkdir(parents=True, exist_ok=True)
        s.to_csv(outdir / "summary.csv", index=False)
        plots = make_plots(rows, s, outdir, rows[0].bottleneck)

        pd.set_option("display.width", 220)
        print("\n================ BUFFER SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))
        print(f"\nWrote {outdir}:")
        for f in ["summary.csv", *[q.name for q in plots]]:
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