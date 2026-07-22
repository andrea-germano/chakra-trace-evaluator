#!/usr/bin/env python3
"""
ns3_analyzer — one ns-3 run, read out of the ns-3 files alone.

Why this exists next to buffer_sweep
--------------------------------------------------------------------------------
The sweep collapses each run to a row of scalars: peak queue, pause %, mean
slowdown. That is the right shape for "how does X vary with the buffer", and it
is the wrong shape for "what actually happened". This tool plots the time axis
the sweep throws away, on one shared clock, for one run — plus an overview that
needs no time axis at all: who queued, and who got throttled.

The placement (KV/TP/PP flow classes) is ON by default, so a bare invocation
gives the full KV layer: the class breakdown, the decode barrier, the KV gantt,
the KV-arrival timeline. That layer rests on a class-agnostic backbone that does
not need it, though: everything computable from fct/pfc/qlen + topology + config
alone is computed unconditionally — per-switch and per-port queue occupancy over
time, occupancy as a fraction of the physical buffer, the ECN band residency,
every PAUSE frame and its duty cycle, offered/delivered load and flow
concurrency, the FCT/slowdown distribution split by hop count. So the placement
can be turned OFF (`--placement ""`) or simply FAIL to match the traffic, and
the run is still analysed class-agnostically instead of aborting. The class
layer is a bonus on top, not a precondition.

Deliberately, NO ASTRA. Nothing here reads stats_sys*.csv. The one thing that
genuinely needs it — TTFT, i.e. the end of prefill — is simply not reported; the
decode-start GATE (max KV arrival) is fct-only and is kept. topology and
config.txt are ns-3 INPUTS, not simulator output, so the fabric model (PFC
threshold, ceiling, regime, flip band) rides along: it is what turns a plot of
the queue into a plot of the queue against where the queue was predicted to sit.

Figures (all ns-3-only; several lifted from buffer_sweep and de-swept)
--------------------------------------------------------------------------------
  * 01  per-switch buffer occupancy over time, and as % of the physical buffer
        — "is the added buffer used, or is it standing empty on an oversubscribed
        link?" A flat, low occupancy at buf32 is a buffer being wasted.
  * 02  PAUSE emission over time — cumulative frames per device and the raster —
        so backpressure can be read on the same clock as the queue that caused it.
  * 03  overview: peak/mean occupancy of every switch port, and the total PAUSE
        frame count per device, side by side. The two-bar answer to "who queued
        and who got throttled", before any time axis.
  * 09  offered/delivered load and flow concurrency over time — the workload's
        own shape, independent of any queue.
  * occupancy-% of buffer, ECN-band residency and time-above-KMAX per switch;
        line-rate efficiency of the busiest link; PAUSE duty cycle and the
        suspect-pairing bound — all in the report, all from the ns-3 files.

Usage
--------------------------------------------------------------------------------
    python3 ns3_analyzer.py --sweep buffer_sweep_T1 --tag T1_bx200_dcqcn_buf4
    python3 ns3_analyzer.py --sweep buffer_sweep_T1 --tag ... --placement ""
    python3 ns3_analyzer.py --sweep buffer_sweep_T1 --tag ... --bottleneck 8->12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt

from utils import flows as flowlib
from utils import ns3, paths, roles
from utils.cli import Abort, need
from utils.fabric import Bottleneck, FabricModel, parse_ns3_config, parse_topology
from utils.plots import downsample_max, save_fig
from utils.roles import Placement

KIND = "run"
COOL, CORAL, AMBER, GREEN, VIOLET, MUTED = (
    "#2b6cb0", "#d1495b", "#d98a00", "#2a9d5c", "#6a4c93", "#6b7280")
_BASE_COLOR = {"kv": CORAL, "tp": MUTED, "pp_prefill": COOL,
               "pp_decode": AMBER, "other": VIOLET}
# A stable palette for switches, so switch N is the same colour in every figure.
_SW_CYCLE = [COOL, CORAL, GREEN, AMBER, VIOLET, "#0f766e", "#b45309", "#7c3aed"]


def class_style(c: str) -> tuple[str, str]:
    """(colour, linestyle) for a flow class. A '_ctrl' suffix keeps the base hue
    and goes dashed: same conversation, notification instead of payload."""
    base = c[:-5] if c.endswith("_ctrl") else c
    return _BASE_COLOR.get(base, VIOLET), ("--" if c.endswith("_ctrl") else "-")


def sw_color(sw: int, switches: list[int]) -> str:
    return _SW_CYCLE[switches.index(sw) % len(_SW_CYCLE)]


class Report:
    """Everything printed is also written to report.txt: one formatting path, so
    the file and the terminal cannot disagree."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def head(self, t):
        self("\n" + "=" * 78)
        self(t)
        self("=" * 78)

    def save(self, p: Path):
        p.write_text("\n".join(self.lines) + "\n")


def fmt_b(b) -> str:
    b = float(b)
    for u, d in (("MB", 1e6), ("kB", 1e3)):
        if b >= d:
            return f"{b/d:.2f} {u}"
    return f"{b:.0f} B"


def fmt_ms(ns) -> str:
    return f"{float(ns) * 1e-6:.3f} ms"


def paused_mask(ts: np.ndarray, pfc, clamp: int) -> np.ndarray:
    """Boolean over `ts`: was ANY ingress port paused at that instant. Class-
    agnostic — unions every device's PAUSE intervals, not just a bottleneck's."""
    m = np.zeros(len(ts), bool)
    if pfc is None:
        return m
    for spans in pfc.pause_intervals(clamp_to=clamp).values():
        for a, b in spans:
            i, j = np.searchsorted(ts, [a, b])
            m[i:j] = True
    return m


# --------------------------------------------------------------------------- #
# Class-agnostic figures — computed from fct/pfc/qlen alone.
# --------------------------------------------------------------------------- #
def plot_queue_timeline(qlen, pfc, topo, model, buffer_bytes, bn, clamp,
                        out, written):
    """Buffer occupancy over time, the whole-run picture, no placement needed.

    Top panel: each SWITCH's total egress bytes (summed over its ports — the
    closest observable proxy for how full its shared buffer is) on the left axis,
    and the same as a percentage of the physical BUFFER_SIZE on the right. This is
    the "is the buffer used or wasted" figure: a line that never climbs off the
    floor is a buffer bigger than the run needs.

    Bottom panel: the busiest ports individually, in kB, against the analysed
    link's ECN band. A per-switch total cannot be compared to KMIN/KMAX (those are
    per egress port); a per-port line can, and this is where DCQCN either marks or
    sits inert below KMIN regardless of the config."""
    switches = sorted(qlen.switch_series)
    ms = 1e-6
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 2.2]))

    a = ax[0]
    for sw in switches:
        ts, ys = downsample_max(*qlen.switch_series[sw])
        if not len(ts):
            continue
        a.plot(ts * ms, ys / 1e6, lw=0.9, color=sw_color(sw, switches),
               drawstyle="steps-post",
               label=f"sw{sw} (peak {fmt_b(qlen.switch_total_max.get(sw,0))})")
    a.set_ylabel("Queued bytes per switch (MB)")
    a.set_title("Buffer occupancy over time")
    if buffer_bytes:
        a2 = a.twinx()
        a2.set_ylim(0, 100 * a.get_ylim()[1] * 1e6 / buffer_bytes)
        a2.set_ylabel("% of buffer")
    a.legend(fontsize=7, loc="upper right", ncol=2)

    a = ax[1]
    items = sorted(qlen.port_max.items(), key=lambda kv: -kv[1])[:6]
    for (sw, pt), _ in items:
        if (sw, pt) not in qlen.port_series:
            continue
        ts, ys = downsample_max(*qlen.port_series[(sw, pt)])
        mark = "  <== analysed" if (sw, pt) == (bn.switch, bn.egress_port) else ""
        a.plot(ts * ms, ys / 1e3, lw=0.9, drawstyle="steps-post",
               label=f"sw{sw}:{topo.port_label(sw, pt)}{mark}")
    kmin, kmax = model.ecn_band(bn)
    if kmax:
        a.axhline(kmax / 1e3, color=GREEN, ls="--", lw=1.1,
                  label=f"KMAX@{bn.rate/1e9:g}G = {kmax/1e3:g} kB")
    if kmin:
        a.axhline(kmin / 1e3, color=GREEN, ls=":", lw=1.1,
                  label=f"KMIN@{bn.rate/1e9:g}G = {kmin/1e3:g} kB")
    a.set_ylabel("Egress queue (kB)")
    a.set_xlabel("Simulated time (ms)")
    a.legend(fontsize=7, loc="upper right", ncol=2)
    for a in ax:
        a.grid(True, alpha=0.25)
    save_fig(fig, out, "01_queue_timeline_by_switch.png", written)


def plot_pause_timeline(pfc, topo, bn, clamp, out, written):
    """PAUSE emission over time — the backpressure the queue figure implies, made
    explicit, on the same clock.

    Top: cumulative count of PAUSE frames per device. A staircase that climbs in
    one burst is a single stall; a constant slope is sustained fluttering, and the
    total alone cannot tell them apart. Bottom: the raster of PAUSE intervals, so a
    stall's DURATION is visible, not just its onset. Victims of the analysed
    bottleneck are drawn in coral; everyone else muted, because backpressure
    sitting on a device that is NOT this bottleneck's ingress is itself a finding."""
    victims = set(bn.pause_victims(topo)) if bn else set()
    # frames = PAUSE events (typ==1); qIndex-independent, so robust even when the
    # 6th column is missing and interval PAIRING is not.
    frames: dict[tuple[int, int], list[int]] = {}
    for (n, _nt, i, _q), evs in pfc.events.items():
        frames.setdefault((n, i), []).extend(t for t, typ in evs if typ == 1)
    if not frames:
        return
    ms = 1e-6
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2.4]))

    a = ax[0]
    for (n, i), ts in sorted(frames.items(), key=lambda kv: -len(kv[1])):
        t = np.sort(np.asarray(ts, float))
        isv = (n, i) in victims
        a.step(t * ms, np.arange(1, len(t) + 1), where="post",
               lw=1.6 if isv else 0.9, color=CORAL if isv else MUTED,
               alpha=1.0 if isv else 0.55,
               label=f"n{n}/if{i} ({len(t)} frames)"
                     + ("  ←ingress of bottleneck" if isv else ""))
    a.set_ylabel("PAUSE frames (cumulative)")
    a.set_title("PFC PAUSE emission over time")
    a.legend(fontsize=7, loc="upper left", ncol=2)

    a = ax[1]
    iv = pfc.pause_intervals(clamp_to=clamp)
    rows: dict[tuple[int, int], list] = {}
    for (n, _nt, i, _q), spans in iv.items():
        rows.setdefault((n, i), []).extend(spans)
    order = sorted(rows, key=lambda k: (k not in victims, k))
    for r, k in enumerate(order):
        a.broken_barh([(s * ms, max((e - s) * ms, 1e-4)) for s, e in rows[k]],
                      (r - 0.4, 0.8), color=CORAL if k in victims else MUTED,
                      alpha=0.85 if k in victims else 0.4)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([f"n{n}/if{i}" + ("  ←" if (n, i) in victims else "")
                       for n, i in order], fontsize=7)
    a.set_ylabel("PAUSE received")
    a.set_xlabel("Simulated time (ms)")
    a.invert_yaxis()
    for a in ax:
        a.grid(True, alpha=0.25)
    save_fig(fig, out, "02_pause_timeline.png", written)


def plot_overview(qlen, pfc, topo, bn, kmax, clamp, out, written):
    """The two overview bars, no time axis: who queued, and who got throttled.

    Left: peak and mean egress occupancy of every switch port that ever held a
    queue. Right: total PAUSE frames received per device. Read together they say
    whether the queue and the backpressure are on the two ends of the same link
    (the expected coupling) or somewhere they should not be."""
    fig, ax = plt.subplots(1, 2, figsize=(13, max(3.5, 0.34 * max(
        len(qlen.port_max), 1) + 2)))

    items = sorted(qlen.port_max.items(), key=lambda kv: -kv[1])
    lbl = [f"sw{s}:{topo.port_label(s, p)}" for (s, p), _ in items]
    peak = [v / 1e3 for _, v in items]
    mean = [qlen.port_mean[k] / 1e3 for k, _ in items]
    col = [CORAL if (s, p) == (bn.switch, bn.egress_port) else COOL
           for (s, p), _ in items]
    y = np.arange(len(items))
    ax[0].barh(y + 0.2, peak, height=0.4, color=col, label="peak")
    ax[0].barh(y - 0.2, mean, height=0.4, color=col, alpha=0.45, label="mean")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(lbl, fontsize=8)
    ax[0].invert_yaxis()
    if kmax:
        ax[0].axvline(kmax / 1e3, color=GREEN, ls="--", lw=1.1,
                      label=f"KMAX = {kmax/1e3:g} kB")
    ax[0].set_xlabel("Egress queue (kB)")
    ax[0].set_title("Queue occupancy per switch port")
    ax[0].grid(True, alpha=0.3, axis="x")
    ax[0].legend(fontsize=8)

    frames: dict[tuple[int, int], int] = {}
    for (n, _nt, i, _q), evs in pfc.events.items():
        frames[(n, i)] = frames.get((n, i), 0) + sum(1 for _, typ in evs if typ == 1)
    victims = set(bn.pause_victims(topo))
    if frames:
        fit = sorted(frames.items(), key=lambda kv: -kv[1])
        fl = [f"n{n}/if{i}" for (n, i), _ in fit]
        fv = [v for _, v in fit]
        fc = [CORAL if k in victims else MUTED for k, _ in fit]
        yy = np.arange(len(fit))
        ax[1].barh(yy, fv, color=fc)
        ax[1].set_yticks(yy)
        ax[1].set_yticklabels(fl, fontsize=8)
        ax[1].invert_yaxis()
        ax[1].set_xlabel("PAUSE frames received")
        ax[1].set_title("Backpressure per device")
        ax[1].grid(True, alpha=0.3, axis="x")
    else:
        ax[1].text(0.5, 0.5, "no PAUSE in pfc.txt\n(DCQCN-only regime)",
                   ha="center", va="center", transform=ax[1].transAxes, fontsize=11)
        ax[1].set_axis_off()
    save_fig(fig, out, "03_overview.png", written)


def plot_load_and_concurrency(fabric_flows, out, written):
    """The workload's own shape, before any queue: delivered load and concurrency.

    Delivered throughput bins each fabric flow's bytes at its ARRIVAL time (80
    buckets) — an empty bucket is a stall, and stalls line up with PAUSE spans in
    figure 02. Concurrency is the sweep-line count of flows alive at once: its
    peak is the upper bound on fair-share slowdown, so a slowdown ECDF should be
    read against it, not against 1."""
    if not len(fabric_flows):
        return
    start = fabric_flows["start"].to_numpy(float)
    arr = fabric_flows["arrival"].to_numpy(float)
    size = fabric_flows["size"].to_numpy(float)
    lo, hi = start.min(), arr.max()
    ms = 1e-6
    nb = 80
    edges = np.linspace(lo, hi, nb + 1)
    width = (hi - lo) / nb
    bytes_in = np.zeros(nb)
    bi = np.clip(np.searchsorted(edges, arr) - 1, 0, nb - 1)
    np.add.at(bytes_in, bi, size)
    gbps = bytes_in * 8 / width         # bytes/ns * 8 = Gbit/s

    ct, cc = flowlib.concurrency_series(
        list(zip(start.astype(int), arr.astype(int))))

    fig, ax = plt.subplots(figsize=(11, 5))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers * ms, gbps, width=width * ms * 0.95, color=COOL, alpha=0.55,
           label="delivered throughput (Gb/s, by arrival)")
    ax.set_xlabel("Simulated time (ms)")
    ax.set_ylabel("Delivered throughput (Gb/s)", color=COOL)
    ax.grid(True, alpha=0.25)
    a2 = ax.twinx()
    if len(ct):
        a2.step(ct * ms, cc, where="post", color=CORAL, lw=1.4,
                label=f"concurrent flows (peak {int(cc.max())})")
    a2.set_ylabel("Concurrent flows", color=CORAL)
    ax.set_title("Delivered load and flow concurrency over time")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    save_fig(fig, out, "09_load_and_concurrency.png", written)


def plot_slowdown_ecdf(f, has_class, mtu, out, written):
    """Slowdown distribution — the full ECDF, no mean, no CV.

    With a placement: one curve per fabric flow class (TP excluded — it is 96% of
    the flows on dedicated 1-hop links at slowdown ~1 and would drown the ink).
    Without one: split the multi-hop flows into bulk (> MTU) and control (<= MTU,
    one packet), the only two structural populations the topology + config give for
    free. The split is on the MTU, not on a median: a single 8-byte control message
    stuck behind a bulk transfer must stay its own curve, not be averaged in with
    the bulk it is queued behind. 1-hop flows are excluded (they cannot queue)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if has_class:
        for c in roles.FLOW_CLASSES:
            if c == "tp":
                continue
            v = f.loc[(f["flow_class"] == c) & f["slowdown"].notna(), "slowdown"]
            if not len(v):
                continue
            v = np.sort(v.to_numpy(float))
            col, ls = class_style(c)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col,
                    linestyle=ls, lw=1.6, label=f"{c}  (n={len(v)})")
    else:
        multi = f[(f["hops"] > 1) & f["slowdown"].notna()]
        for lab, sel, col, ls in (
            ("multi-hop bulk (> MTU)", multi["size"] > mtu, CORAL, "-"),
            ("multi-hop control (<= MTU)", multi["size"] <= mtu, COOL, "--")):
            v = np.sort(multi.loc[sel, "slowdown"].to_numpy(float))
            if len(v) < 2:
                continue
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                    color=col, linestyle=ls, lw=1.6, label=f"{lab}  (n={len(v)})")
    ax.set_xscale("log")
    ax.set_xlabel("Slowdown (×ideal, log)")
    ax.set_ylabel("ECDF")
    ax.set_title("Slowdown distribution across fabric flows")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save_fig(fig, out, "08_slowdown_ecdf.png", written)


def plot_queue_ecdf(qlen, pfc, bn, kmin, kmax, clamp, out, written):
    """Where the queue LIVES at the bottleneck, distributionally, split by whether
    an ingress was paused at that instant. peak/KMAX is one order statistic of a
    million-sample distribution and the least robust one; this is the mechanism."""
    if (bn.switch, bn.egress_port) not in qlen.port_series:
        return
    ts, ys = qlen.port_series[(bn.switch, bn.egress_port)]
    ts, ys = np.asarray(ts), np.asarray(ys, dtype=float)
    paused = paused_mask(ts, pfc, clamp)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, lab, col in ((ys, f"all samples (n={len(ys):,})", COOL),
                        (ys[paused], f"while an ingress is PAUSED (n={paused.sum():,})", CORAL),
                        (ys[~paused], f"while none is (n={(~paused).sum():,})", GREEN)):
        if len(v) < 2:
            continue
        v = np.sort(v)
        ax.step(v / 1e3, np.arange(1, len(v) + 1) / len(v), where="post",
                color=col, lw=1.5, label=lab)
    if kmin:
        ax.axvline(kmin / 1e3, color=MUTED, ls=":", lw=1.4, label=f"KMIN = {kmin/1e3:g} kB")
    if kmax:
        ax.axvline(kmax / 1e3, color=MUTED, ls="--", lw=1.4, label=f"KMAX = {kmax/1e3:g} kB")
        ax.axvspan(kmin / 1e3, kmax / 1e3, color=AMBER, alpha=0.08,
                   label="ECN ramp: marks with probability p(q)")
    ax.set_xlabel("Egress queue (kB)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Queue occupancy vs the ECN band at {bn}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, out, "06_queue_ecdf_vs_ecn_band.png", written)


def plot_pause_ecdf(pfc, bn, topo, clamp, out, written):
    """Distribution of individual PAUSE durations, log x. A pause total is a sum
    over thousands of events; it cannot say whether that is one long stall or
    continuous fluttering, and those are different fabrics."""
    fl = pfc.pause_intervals_flagged(clamp_to=clamp)
    if not fl:
        return
    victims = set(bn.pause_victims(topo))
    dev: dict[tuple, list] = {}
    for (n, _nt, i, _q), sp in fl.items():
        dev.setdefault((n, i), []).extend(sp)
    fig, ax = plt.subplots(figsize=(8, 5))
    any_drawn = False
    for k in sorted(dev):
        d = np.array([e - s for s, e, _ in dev[k] if e > s], float)
        if len(d) < 2:
            continue
        any_drawn = True
        d = np.sort(d)
        isv = k in victims
        ax.step(d, np.arange(1, len(d) + 1) / len(d), where="post", lw=1.6 if isv else 1.0,
                color=CORAL if isv else MUTED, alpha=1.0 if isv else 0.5,
                label=f"n{k[0]}/if{k[1]}" + ("  ←ingress of the bottleneck" if isv else ""))
        sus = np.array([e - s for s, e, q in dev[k] if q and e > s], float)
        if len(sus):
            ax.plot(sus, [np.searchsorted(d, x) / len(d) for x in sus], "x",
                    color="k", ms=7, mew=1.6,
                    label="mis-paired (no qIndex)" if k == sorted(dev)[0] else None)
    if not any_drawn:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("PAUSE interval duration (ns, log)")
    ax.set_ylabel("ECDF")
    ax.set_title("Distribution of PFC pause durations")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, out, "07_pause_duration_ecdf.png", written)


def plot_ports(qlen, topo, bn, kmax, out, written):
    """Peak queue on every switch port that held one — answers 'is the port I
    think is the bottleneck the one that actually built the queue'."""
    items = sorted(qlen.port_max.items(), key=lambda kv: -kv[1])
    lbl = [f"sw{s}:{topo.port_label(s, p)}" for (s, p), _ in items]
    val = [v / 1e3 for _, v in items]
    col = [CORAL if (s, p) == (bn.switch, bn.egress_port) else COOL
           for (s, p), _ in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.34 * len(items) + 1.6)))
    ax.barh(range(len(val)), val, color=col)
    ax.set_yticks(range(len(lbl)))
    ax.set_yticklabels(lbl, fontsize=8)
    ax.invert_yaxis()
    if kmax:
        ax.axvline(kmax / 1e3, color=GREEN, ls="--", lw=1.2, label=f"KMAX = {kmax/1e3:g} kB")
        ax.legend(fontsize=8)
    ax.set_xlabel("Peak egress queue (kB)")
    ax.set_title("Peak queue per switch port")
    ax.grid(True, alpha=0.3, axis="x")
    save_fig(fig, out, "04_queue_peak_by_port.png", written)


def plot_congestion_heatmap(qlen, topo, model, bn, out, written):
    """Peak egress queue / that port's own KMAX, for EVERY switch port — the
    topology's adjacency matrix coloured by congestion. Gray = no such link;
    near-white = a real port that never queued (itself a finding)."""
    switches = sorted(topo.switches)
    peers = sorted({port.peer for sw in switches for port in topo.ports.get(sw, {}).values()})
    ratio = np.full((len(switches), len(peers)), np.nan)
    label = np.full((len(switches), len(peers)), "", dtype=object)
    for i, sw in enumerate(switches):
        for pt, port in topo.ports.get(sw, {}).items():
            peak = qlen.port_max.get((sw, pt), 0)
            j = peers.index(port.peer)
            kmax = model.cfg.kmax.get(port.rate)
            label[i, j] = fmt_b(peak) + (f"\n{peak/kmax:.2f}x" if kmax else "\n(no KMAX)")
            ratio[i, j] = (peak / kmax) if kmax else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 0.65 * len(peers) + 2),
                                    max(4, 0.5 * len(switches) + 2)))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("congestion", ["#fbeaea", CORAL])
    cmap.set_bad("#ececec")
    vmax = max(1.0, np.nanmax(ratio)) if np.isfinite(ratio).any() else 1.0
    im = ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(switches)):
        for j in range(len(peers)):
            if label[i, j]:
                ax.text(j, i, label[i, j], ha="center", va="center", fontsize=6.5)
    bi, bj = switches.index(bn.switch), peers.index(bn.peer)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.2))
    ax.set_xticks(range(len(peers)))
    ax.set_xticklabels([("sw" if topo.is_switch(p) else "h") + str(p) for p in peers], fontsize=8)
    ax.set_yticks(range(len(switches)))
    ax.set_yticklabels([f"sw{s}" for s in switches], fontsize=8)
    ax.set_xlabel("Port peer (host / switch)")
    ax.set_ylabel("Switch")
    ax.set_title("Peak queue per port, relative to KMAX")
    fig.colorbar(im, ax=ax).set_label("Peak queue (×KMAX)")
    save_fig(fig, out, "05_congestion_heatmap.png", written)


# --------------------------------------------------------------------------- #
# Class-dependent figures — only drawn when a placement matched the traffic.
# --------------------------------------------------------------------------- #
def plot_kv_timeline(qlen, pfc, f, kv, bn, topo, model, buffer_bytes,
                     kmin, kmax, clamp, out, written):
    """The headline: egress queue, backpressure and KV arrivals on ONE clock."""
    if (bn.switch, bn.egress_port) not in qlen.port_series:
        return
    ts, ys = downsample_max(*qlen.port_series[(bn.switch, bn.egress_port)], n=6000)
    lo, hi = int(kv["start"].min()), int(kv["arrival"].max())
    ms = 1e-6
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.6, 2.2]))
    a = ax[0]
    a.plot(ts * ms, ys / 1e3, lw=0.8, color=COOL, drawstyle="steps-post",
           label=f"egress queue at {bn} (port {topo.port_label(bn.switch, bn.egress_port)})")
    if kmax:
        a.axhline(kmax / 1e3, color=GREEN, ls="--", lw=1.2, label=f"KMAX = {kmax/1e3:g} kB")
    if kmin:
        a.axhline(kmin / 1e3, color=GREEN, ls=":", lw=1.2, label=f"KMIN = {kmin/1e3:g} kB")
    ceil = model.pfc_egress_ceiling(bn, buffer_bytes)
    a.axhline(ceil / 1e3, color=CORAL, ls="-.", lw=1.2,
              label=f"MODEL PFC ceiling = {ceil/1e3:.0f} kB")
    a.set_ylabel("Egress queue (kB)")
    a.legend(fontsize=7, loc="upper right", ncol=2)
    a.set_title("Queue, backpressure and KV arrivals on one clock")

    a = ax[1]
    victims = sorted(set(bn.pause_victims(topo)))
    iv = pfc.pause_intervals(clamp_to=clamp)
    rows = {}
    for (node, _nt, ifidx, _q), spans in iv.items():
        rows.setdefault((node, ifidx), []).extend(spans)
    order = victims + [k for k in sorted(rows) if k not in victims]
    for i, k in enumerate(order):
        a.broken_barh([(s * ms, (e - s) * ms) for s, e in rows.get(k, [])], (i - 0.4, 0.8),
                      color=CORAL if k in victims else MUTED,
                      alpha=0.85 if k in victims else 0.35)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([f"n{n}/if{i}" + ("  ←ingress of the bottleneck" if (n, i) in victims else "")
                       for n, i in order], fontsize=7)
    a.set_ylabel("PAUSE received")
    a.invert_yaxis()

    a = ax[2]
    for r in sorted(set(kv["dst"])):
        arr = np.sort(kv.loc[kv["dst"] == r, "arrival"].to_numpy()) * ms
        a.step(arr, np.arange(1, len(arr) + 1), where="post", lw=1.2,
               label=f"KV arrived at decode rank {r} ({len(arr)} flows)")
        a.plot(arr[-1], len(arr), "o", ms=5)
    gate = kv["arrival"].max() * ms
    a.axvline(gate, color=CORAL, ls="--", lw=1.4, label=f"decode-start gate = {gate:.2f} ms")
    a.axvspan(lo * ms, hi * ms, color=VIOLET, alpha=0.07, label=f"KV window ({(hi-lo)*ms:.2f} ms)")
    a.set_ylabel("KV flows arrived (cumulative)")
    a.set_xlabel("Simulated time (ms)")
    a.legend(fontsize=7, loc="upper left")
    for a in ax:
        a.grid(True, alpha=0.25)
    save_fig(fig, out, "10_kv_timeline.png", written)


def plot_gantt(kv_bn, bn, out, written):
    """One bar per KV flow at the bottleneck, start -> arrival. Parallel bars =
    sharing the link; a staircase = serialising."""
    d = kv_bn.sort_values("start").reset_index(drop=True)
    ranks = sorted(set(d["dst"]))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.13 * len(d) + 2)))
    for i, r in d.iterrows():
        ax.barh(i, (r["arrival"] - r["start"]) * 1e-6, left=r["start"] * 1e-6,
                height=0.8, color=cmap(ranks.index(r["dst"]) % 10))
    for j, r in enumerate(ranks):
        ax.barh(0, 0, color=cmap(j % 10), label=f"→ decode rank {r}")
    ax.set_xlabel("Simulated time (ms)")
    ax.set_ylabel("KV flow (sorted by start)")
    ax.set_title(f"KV flow timeline at the bottleneck {bn}")
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(fontsize=8)
    save_fig(fig, out, "11_kv_gantt.png", written)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    for act in ap._actions:
        if act.dest == "out":
            act.help = "output dir (default: results/ns3_graphs/<workload>/<sweep>/<tag>)"
    roles.add_argument(ap)
    ap.add_argument("--tag", required=True, help="run sub-directory, e.g. 'T1_bx200_dcqcn_buf8'")
    ap.add_argument("--bottleneck", default=None,
                    help="'sw->peer'. Default: measured from qlen.txt.")
    ap.add_argument("--headroom-factor", type=int, default=None,
                    help="ns-3 common.h::headroom_factor. Default: HEADROOM_FACTOR "
                         "from config.txt, else 3.")
    a = ap.parse_args(argv)

    try:
        p = paths.SweepPaths(sweep=a.sweep, workload=a.workload, root=Path(a.root))
        outbase = (Path(a.out) if a.out else
                   p.root / "results" / "ns3_graphs" / p.workload / p.sweep)
        outdir = outbase / a.tag
        need(not p.missing_roots(),
             "derived root(s) do not exist:\n    " + "\n    ".join(p.missing_roots()))
        ns3_dir, tpath, cpath = p.ns3_run(a.tag), p.topology(a.tag), p.config(a.tag)
        for q in (ns3_dir / "fct.txt", ns3_dir / "qlen.txt", tpath, cpath):
            need(q.exists(), f"missing {q}\n  --tag {a.tag!r} may be wrong; "
                             f"{p.ns3_root} has: {sorted(x.name for x in p.ns3_root.iterdir())[:8]}")

        cfg = parse_ns3_config(cpath)
        hf = a.headroom_factor if a.headroom_factor is not None else cfg.headroom_factor
        need(cfg.buffer_mb is not None, f"no BUFFER_SIZE in {cpath}.")
        topo = parse_topology(tpath, hf)
        buffer_bytes = cfg.buffer_bytes

        say = Report()
        say.head(f"ns-3 run: {a.tag}   (ns3_analyzer — ns-3 files only, no ASTRA)")
        say(f"  ns3      {ns3_dir}")
        say(f"  topology {tpath}")
        say(f"  config   {cpath}")
        say(f"  out      {outdir}")
        say(f"  BUFFER_SIZE = {cfg.buffer_mb:g} MiB   CC_MODE = {cfg.cc_mode}   "
            f"ENABLE_QCN = {cfg.enable_qcn}   headroom_factor = {hf}"
            f"{'  (from config.txt)' if 'HEADROOM_FACTOR' in cpath.read_text() else '  (ASSERTED)'}")
        say(f"  topology: {len(topo.hosts)} hosts, {len(topo.switches)} switches "
            f"{sorted(topo.switches)}, nic_rate = {topo.nic_rate/1e9:g} Gbps")
        for w in cfg.warnings():
            say(f"  ! {w}")

        # -- flows ---------------------------------------------------------- #
        raw = ns3.read_fct(ns3_dir / "fct.txt")
        need(raw is not None and len(raw), "fct.txt has no parsable rows.")
        d = raw.attrs["diagnostics"]
        say.head("fct.txt — parse sanity, before any number computed from it")
        say(f"  lines {d['raw_lines']} | parsed {len(raw)} | skipped(<8 cols) "
            f"{d['skipped_short']} | skipped(non-numeric) {d['skipped_badnum']} | "
            f"column counts {d['ncol_hist']}")
        say(f"  first line : {d['sample']}")
        say(f"  expected   : sip dip sport dport size(B) start(ns) fct(ns) standalone_fct(ns)")
        say(f"  min slowdown = {d['slow_min']:.4f}  (headers billed in standalone_fct "
            f"but not in the size column put an uncongested flow a % below 1)")
        if d["slow_lt09"]:
            say(f"  ! {d['slow_lt09']} flows have slowdown < 0.9, which headers cannot "
                f"explain: columns 5-8 are probably not where expected.")
        if d["sfct_nonpos"]:
            say(f"  ! {d['sfct_nonpos']} flows have standalone_fct <= 0 (no slowdown).")

        # Placement is OPTIONAL. Try it; if it does not produce KV, fall back to
        # a class-agnostic reading rather than aborting.
        placement = None
        if a.placement.strip():
            try:
                placement = Placement.parse(a.placement)
            except ValueError as e:
                say(f"  ! --placement ignored ({e}); analysing class-agnostically.")
        f = flowlib.annotate(raw, topo, placement, cfg.payload)
        has_class = "flow_class" in f.columns
        if has_class:
            warns = roles.check(f, placement)
            if any("no flow is classified 'kv'" in w for w in warns):
                say("  ! --placement does not match the traffic (no KV flow). "
                    "Falling back to class-agnostic analysis.")
                f, has_class, placement = (
                    flowlib.annotate(raw, topo, None, cfg.payload), False, None)
            else:
                for w in warns:
                    say(f"  ! {w}")

        fabric_flows = f[f["hops"] > 1]
        direct = f[f["hops"] == 1]
        say.head("Flows overview (structural, no ASTRA — hops from topology)")
        say(f"  total flows {len(f)} | bytes {fmt_b(f['size'].sum())} | "
            f"span [{fmt_ms(f['start'].min())}, {fmt_ms(f['arrival'].max())}]")
        say(f"  1-hop (host-to-host, cannot queue) : {len(direct):>5}  "
            f"bytes {fmt_b(direct['size'].sum())}")
        say(f"  multi-hop (crosses >=1 switch)     : {len(fabric_flows):>5}  "
            f"bytes {fmt_b(fabric_flows['size'].sum())}")
        say("\n  flow-size histogram (distinct payload sizes, all flows):")
        for sz, cnt in raw["size"].value_counts().sort_index(ascending=False).head(10).items():
            tag = "  (<=MTU, 1 packet -> control)" if sz <= cfg.payload else ""
            say(f"    {fmt_b(sz):>10}  x {cnt}{tag}")
        conc_peak, conc_mean = flowlib.concurrency_stats(flowlib.flow_spans(fabric_flows))
        say(f"\n  fabric-flow concurrency: peak {conc_peak:.0f}  "
            f"mean-experienced {conc_mean:.2f}  (upper bound on fair-share slowdown)")
        if len(fabric_flows):
            sd = fabric_flows.loc[fabric_flows["slowdown"].notna(), "slowdown"]
            say(f"  fabric-flow slowdown: p50 {np.percentile(sd,50):.2f}  "
                f"p99 {np.percentile(sd,99):.2f}  max {sd.max():.2f}")

        if has_class:
            say.head("Flows by class (from --placement)")
            for c in roles.FLOW_CLASSES:
                g = f[f["flow_class"] == c]
                if not len(g):
                    continue
                sd = g.loc[g["slowdown"].notna(), "slowdown"]
                say(f"  {c:<15} n={len(g):<5} bytes={fmt_b(g['size'].sum()):>9}"
                    + (f"  slowdown p50={np.percentile(sd,50):7.2f} "
                       f"p99={np.percentile(sd,99):7.2f} max={sd.max():7.2f}" if len(sd) else ""))

        # -- the bottleneck ------------------------------------------------- #
        qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=True)
        need(qlen is not None and qlen.port_max, "qlen.txt has no samples.")
        # The population whose paths define the ingress ports: KV if we have it,
        # else every fabric flow (a superset, so F_ports is an upper bound).
        congesting = f[f["flow_class"] == "kv"] if has_class else fabric_flows
        if a.bottleneck:
            sw, peer = (int(x) for x in a.bottleneck.split("->"))
            eg = topo.port_facing(sw, peer)
            need(eg is not None, f"--bottleneck {a.bottleneck}: no such link.")
            ing = {q for path in congesting["path"] for i, (x, y) in enumerate(path or [])
                   if (x, y) == (sw, peer) and i > 0
                   and (q := topo.port_facing(sw, path[i - 1][0])) is not None}
            bn = Bottleneck(sw, eg, peer, topo.ports[sw][eg].rate, tuple(sorted(ing)))
        else:
            bn = flowlib.find_bottleneck(topo, qlen.port_max, congesting)
        model = FabricModel(topo, cfg)
        kmin, kmax = model.ecn_band(bn)

        say.head(f"Bottleneck {bn} — measured (deepest queue), model on top")
        say(f"  egress {topo.port_label(bn.switch, bn.egress_port)} @ {bn.rate/1e9:g} Gbps | "
            f"F_ports = {bn.f_ports} (ingress {[topo.port_label(bn.switch, i) for i in bn.ingress_ports]})")
        say(f"  PAUSE victims to look for in pfc.txt: {sorted(bn.pause_victims(topo))}")
        peak = qlen.port_max[(bn.switch, bn.egress_port)]
        if bn.f_ports:
            thr = model.steady_threshold(bn, buffer_bytes)
            ceil = model.pfc_egress_ceiling(bn, buffer_bytes)
            say(f"  MODEL  steady PFC threshold = {fmt_b(thr)} | x F_ports = "
                f"{fmt_b(thr*bn.f_ports)} | ceiling = {fmt_b(ceil)} | "
                f"regime = {model.regime(bn, buffer_bytes)}")
            band = model.flip_band(bn)
            if band:
                say(f"  MODEL  predicted flip band: PFC below {band[0]:.2f} MiB, "
                    f"DCQCN above {band[1]:.2f} MiB")
        else:
            ceil = None
            say("  MODEL  no known ingress port for this link -> F_ports=0; "
                "threshold/ceiling not modelled.")
        say(f"  ECN    KMIN = {fmt_b(kmin) if kmin else 'n/a'}   "
            f"KMAX = {fmt_b(kmax) if kmax else 'n/a'}")
        say(f"  MEAS   peak egress = {fmt_b(peak)}"
            + (f"  (= {peak/kmax:.2f} x KMAX" if kmax else "")
            + (f", {peak/ceil:.2f} x ceiling)" if ceil else (")" if kmax else "")))
        if ceil and peak > ceil:
            say("  ! peak exceeds the modelled PFC ceiling (an upper bound). Either "
                "--headroom-factor is wrong, or per-port vs per-queue headroom "
                "accounting differs (qlen.txt sums queues; ShouldSendCN tests one).")

        # -- occupancy per switch/port -------------------------------------- #
        say.head("Queue occupancy per switch/port (qlen.txt — egress bytes, per port)")
        say(f"  monitored window [{fmt_ms(qlen.t_min)}, {fmt_ms(qlen.t_max)}] at 100 ns")
        say(f"\n    {'switch':<8} {'peak total':>12} {'% buffer':>9} {'ports seen':>10}")
        for sw in sorted(qlen.switch_total_max, key=lambda s: -qlen.switch_total_max[s]):
            pk = qlen.switch_total_max[sw]
            pct = 100 * pk / buffer_bytes if buffer_bytes else float("nan")
            nports = sum(1 for (s, _p) in qlen.port_max if s == sw)
            say(f"    sw{sw:<6} {fmt_b(pk):>12} {pct:>8.1f}% {nports:>10}")
        say(f"\n    {'port':<20} {'peak':>10} {'mean':>10} {'samples':>8} "
            f"{'>KMAX':>7} {'in band':>8}")
        for (s, pt), v in sorted(qlen.port_max.items(), key=lambda kv: -kv[1]):
            kx = model.cfg.kmax.get(topo.ports[s][pt].rate)
            kn = model.cfg.kmin.get(topo.ports[s][pt].rate)
            ab, ib = "-", "-"
            if kx and (s, pt) in qlen.port_series:
                ys = np.asarray(qlen.port_series[(s, pt)][1], float)
                ab = f"{100*(ys>kx).mean():.1f}%"
                ib = f"{100*((ys>=(kn or 0))&(ys<=kx)).mean():.1f}%"
            mark = "  <== analysed" if (s, pt) == (bn.switch, bn.egress_port) else ""
            say(f"    sw{s}:{topo.port_label(s, pt):<16} {fmt_b(v):>10} "
                f"{fmt_b(qlen.port_mean[(s,pt)]):>10} {qlen.port_count[(s,pt)]:>8} "
                f"{ab:>7} {ib:>8}{mark}")

        # line-rate efficiency of the busiest link, class-agnostic
        busiest = bn if len(congesting) else None
        cross = f[flowlib.crosses(f, bn)] if busiest else f.iloc[:0]
        if len(cross):
            b_bytes = cross["size"].sum()
            win = int(cross["arrival"].max() - cross["start"].min())
            floor = b_bytes * 8e9 / bn.rate
            say(f"\n  busiest link {bn}: {len(cross)} flows, {fmt_b(b_bytes)} across it in "
                f"{fmt_ms(win)}")
            say(f"    line-rate floor {fmt_ms(floor)} | delivered {b_bytes*8/win:.1f} Gb/s "
                f"| efficiency {100*floor/win:.1f}% of line rate")

        # -- PFC ------------------------------------------------------------ #
        pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
        clamp = int(f["arrival"].max())
        say.head("pfc.txt — backpressure")
        if pfc is None or pfc.n_events == 0:
            say("  no PAUSE events: PFC never fired. This run is DCQCN-governed "
                "(rate control absorbed the incast without backpressure).")
        else:
            say(f"  events {pfc.n_events} | qIndex column: {pfc.qidx_state}")
            frames_by_dev: dict[tuple, int] = {}
            for (n, nt, i, _q), evs in pfc.events.items():
                frames_by_dev[(n, nt, i)] = frames_by_dev.get((n, nt, i), 0) \
                    + sum(1 for _, typ in evs if typ == 1)
            total_frames = sum(frames_by_dev.values())
            say(f"  total PAUSE frames emitted: {total_frames} "
                f"across {len(frames_by_dev)} device(s)")
            if pfc.qidx_state == "MISSING":
                say("  ! no qIndex column: PAUSE/RESUME of different queues share a "
                    "(node,ifindex) key and pairing may mis-attribute. Frame COUNTS "
                    "above are exact (qIndex-independent); durations are bracketed below.")
            totals, unclosed = pfc.pause_totals(clamp_to=clamp)
            per_dev = pfc.pause_per_device(clamp_to=clamp)
            fl = pfc.pause_intervals_flagged(clamp_to=clamp)
            sus_by_dev: dict[tuple, int] = {}
            for (n, nt, i, _q), sp in fl.items():
                sus_by_dev[(n, i)] = sus_by_dev.get((n, i), 0) + sum(e - s for s, e, q in sp if q)
            window = clamp - int(f["start"].min())
            victims = set(bn.pause_victims(topo))
            say(f"  unclosed PAUSEs (held to run end): {unclosed}")
            say(f"\n    {'device':<16} {'frames':>7} {'paused':>11} {'duty':>7} "
                f"{'suspect':>10}")
            for (n, nt, i), frm in sorted(frames_by_dev.items(), key=lambda kv: -kv[1]):
                pt = per_dev.get((n, nt, i), 0)
                duty = 100 * pt / window if window else 0
                sus = sus_by_dev.get((n, i), 0)
                mark = "  <== ingress of bottleneck" if (n, i) in victims else ""
                say(f"    n{n}/if{i} ({'sw' if nt else 'host'}){'':<3} {frm:>7} "
                    f"{fmt_ms(pt):>11} {duty:>6.1f}% {fmt_ms(sus):>10}{mark}")

        # -- barrier (class-dependent) -------------------------------------- #
        kv = f[f["flow_class"] == "kv"] if has_class else f.iloc[:0]
        if has_class and len(kv):
            say.head("Barrier: what gates decode (fct-only; TTFT would need ASTRA)")
            for r in placement.decode_ranks:
                arr = kv.loc[kv["dst"] == r, "arrival"]
                if not len(arr):
                    say(f"  decode rank {r}: receives NO KV flow")
                    continue
                say(f"  decode rank {r}: {len(arr):>3} KV flows | first {fmt_ms(arr.min())} | "
                    f"KV-ready {fmt_ms(arr.max())} | stream {fmt_ms(arr.max()-arr.min())}")
            ready = [kv.loc[kv["dst"] == r, "arrival"].max() for r in placement.decode_ranks
                     if len(kv.loc[kv["dst"] == r, "arrival"])]
            gate = max(ready)
            say(f"\n  decode-start gate = max KV-ready = {fmt_ms(gate)} | "
                f"cross-rank skew = {fmt_ms(gate - min(ready))}")

        # -- outputs --------------------------------------------------------- #
        outdir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        # class-agnostic figures (always)
        plot_queue_timeline(qlen, pfc, topo, model, buffer_bytes, bn, clamp, outdir, written)
        if pfc is not None and pfc.n_events:
            plot_pause_timeline(pfc, topo, bn, clamp, outdir, written)
        plot_overview(qlen, pfc if pfc is not None else ns3.PfcLog(), topo, bn, kmax, clamp, outdir, written)
        plot_ports(qlen, topo, bn, kmax, outdir, written)
        plot_congestion_heatmap(qlen, topo, model, bn, outdir, written)
        plot_queue_ecdf(qlen, pfc, bn, kmin, kmax, clamp, outdir, written)
        if pfc is not None and pfc.n_events:
            plot_pause_ecdf(pfc, bn, topo, clamp, outdir, written)
        plot_slowdown_ecdf(f, has_class, cfg.payload, outdir, written)
        plot_load_and_concurrency(fabric_flows, outdir, written)
        # class-dependent figures
        if has_class and len(kv):
            kv_bn = kv[flowlib.crosses(kv, bn)]
            plot_kv_timeline(qlen, pfc if pfc is not None else ns3.PfcLog(), f, kv, bn,
                             topo, model, buffer_bytes, kmin, kmax, clamp, outdir, written)
            if len(kv_bn):
                plot_gantt(kv_bn, bn, outdir, written)

        # CSVs
        f.drop(columns=["path"]).to_csv(outdir / "flows.csv", index=False)
        pd.DataFrame([{"switch": s, "port": pt, "label": topo.port_label(s, pt),
                       "rate_gbps": topo.ports[s][pt].rate / 1e9,
                       "peak_bytes": v, "mean_bytes": qlen.port_mean[(s, pt)],
                       "samples": qlen.port_count[(s, pt)],
                       "is_bottleneck": (s, pt) == (bn.switch, bn.egress_port)}
                      for (s, pt), v in qlen.port_max.items()]
                     ).sort_values("peak_bytes", ascending=False).to_csv(outdir / "ports.csv", index=False)
        if pfc is not None and pfc.n_events:
            per_dev = pfc.pause_per_device(clamp_to=clamp)
            frm: dict[tuple, int] = {}
            for (n, nt, i, _q), evs in pfc.events.items():
                frm[(n, nt, i)] = frm.get((n, nt, i), 0) + sum(1 for _, t in evs if t == 1)
            pd.DataFrame([{"node": n, "node_type": nt, "ifindex": i,
                           "pause_frames": c, "paused_ns": per_dev.get((n, nt, i), 0)}
                          for (n, nt, i), c in frm.items()]
                         ).sort_values("pause_frames", ascending=False).to_csv(
                outdir / "pause.csv", index=False)

        say(f"\nWrote {outdir}:")
        for q in ["report.txt", "flows.csv", "ports.csv",
                  *(["pause.csv"] if (pfc is not None and pfc.n_events) else []),
                  *[w.name for w in written]]:
            say(f"  {q}")
        (outdir / "report.txt").write_text("\n".join(say.lines) + "\n")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
