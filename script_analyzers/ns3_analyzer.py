#!/usr/bin/env python3
"""
ns3_analyzer — one ns-3 run, in the time domain.

Why this exists next to buffer_sweep
--------------------------------------------------------------------------------
The sweep collapses each run to a row of scalars: peak queue, pause %, mean
slowdown. That is the right shape for "how does X vary with the buffer", and it
is the wrong shape for "what actually happened". Every question that made the
sweep results hard to read is a question about WHEN:

    does the queue peak while PFC is pausing, or before it?
    is the KV window the same window as the pause window?
    do the KV flows share the link, or serialise into a queue of arrivals?
    is the port I think is the bottleneck the port that built the queue?

None of those survive a max(). So this tool plots the time axis the sweep throws
away, on one shared clock, for one run. It computes no verdict and fits nothing.

What was removed, and why
--------------------------------------------------------------------------------
* `classify()`. It returned "DCQCN / MIXED / PFC" from hand-picked cut-offs
  (0.95, 1.0, 0.5, 5.0). Its own docstring already recorded that a previous
  version had been wrong on four runs out of five. A cut-off is not a
  measurement: the pause fraction and the queue are plotted, and the reader
  applies the model's band from figure 05 of the sweep.
* Six local parsers (`parse_fct`, `parse_pfc`, `parse_qlen`, `ip_to_node`,
  `load_topology`, `port_label`) that re-implemented utils.ns3 and utils.fabric.
  Two implementations of one format drift, and when they do you get a per-run
  figure that silently disagrees with the sweep and no way to tell which lied.
* `--bulk-mb`. Flow class is structural now (utils.roles), not a size threshold.
* `--buffer-mb` (default 16!). It is in config.txt; it is read.
* The `HAVE_MPL` fallback and `_resolve_aux`'s CWD-relative directory search.

Usage
--------------------------------------------------------------------------------
    python3 ns3_analyzer.py --sweep buffer_sweep_T1 --tag T1_bx200_dcqcn_buf8
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
import matplotlib.pyplot as plt

from utils import flows as flowlib
from utils import ns3, paths, roles
from utils.fabric import Bottleneck, FabricModel, parse_ns3_config, parse_topology
from utils.plots import save_fig
from utils.roles import Placement

KIND = "run"
COOL, CORAL, AMBER, GREEN, VIOLET, MUTED = (
    "#2b6cb0", "#d1495b", "#d98a00", "#2a9d5c", "#6a4c93", "#6b7280")
_BASE_COLOR = {"kv": CORAL, "tp": MUTED, "pp_prefill": COOL,
               "pp_decode": AMBER, "other": VIOLET}


def class_style(c: str) -> tuple[str, str]:
    """(colour, linestyle). A '_ctrl' class keeps its base colour and goes
    dashed: it is the same conversation, carrying a notification instead of a
    payload, and on T1 it is four orders of magnitude slower than the payload it
    announces. Same hue so the pair is read together; dashed so it is never
    mistaken for the bulk."""
    base = c[:-5] if c.endswith("_ctrl") else c
    return _BASE_COLOR.get(base, VIOLET), ("--" if c.endswith("_ctrl") else "-")


class Abort(Exception):
    pass


def need(cond, msg: str) -> None:
    if not cond:
        raise Abort(msg)


class Report:
    """Everything printed is also written to report.txt. There is no second
    formatting path, so the file and the terminal cannot disagree."""

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


# --------------------------------------------------------------------------- #
# Figures. Time on the x axis, because that is the whole point.
# --------------------------------------------------------------------------- #
def plot_timeline(say, qlen, pfc, f, kv, bn, topo, model, buffer_bytes,
                  kmin, kmax, out, written):
    """The headline: queue, backpressure and KV arrivals on ONE clock.

    Reading it: if the queue plateau and the pause bars line up, backpressure is
    what is holding the queue. If the queue rides below KMIN, ECN never marks and
    DCQCN is inert regardless of what the config says. If the pause bars extend
    past the KV window, the denominator of any pause percentage is wrong."""
    ts, ys = qlen.port_series[(bn.switch, bn.egress_port)]
    # 2.75M samples at 100 ns is more points than the figure has pixels: keep the
    # max per bucket, because the peak is the quantity being compared to KMAX and
    # averaging would hide exactly the excursion that matters.
    if len(ts) > 6000:
        ts, ys = np.asarray(ts), np.asarray(ys)
        edges = np.linspace(ts[0], ts[-1], 6001)
        idx = np.clip(np.searchsorted(edges, ts) - 1, 0, 5999)
        hi_ = np.full(6000, -1.0)
        np.maximum.at(hi_, idx, ys)
        keep = hi_ >= 0
        ts, ys = ((edges[:-1] + edges[1:]) / 2)[keep], hi_[keep]
    lo, hi = int(kv["start"].min()), int(kv["arrival"].max())
    ms = 1e-6

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.6, 2.2]))

    a = ax[0]
    a.plot(np.array(ts) * ms, np.array(ys) / 1e3, lw=0.8, color=COOL,
           drawstyle="steps-post", label=f"egress queue at {bn} (port "
                                         f"{topo.port_label(bn.switch, bn.egress_port)})")
    a.axhline(kmax / 1e3, color=GREEN, ls="--", lw=1.2, label=f"KMAX = {kmax/1e3:g} kB")
    a.axhline(kmin / 1e3, color=GREEN, ls=":", lw=1.2, label=f"KMIN = {kmin/1e3:g} kB")
    ceil = model.pfc_egress_ceiling(bn, buffer_bytes)
    a.axhline(ceil / 1e3, color=CORAL, ls="-.", lw=1.2,
              label=f"MODEL PFC ceiling = {ceil/1e3:.0f} kB")
    a.set_ylabel("Egress queue (kB)")
    a.legend(fontsize=7, loc="upper right", ncol=2)
    a.set_title(f"{out.name}: queue, backpressure and KV arrivals on one clock")

    a = ax[1]
    victims = sorted(set(bn.pause_victims(topo)))
    iv = pfc.pause_intervals(clamp_to=int(f["arrival"].max()))
    rows = {}
    for (node, _nt, ifidx, q), spans in iv.items():
        rows.setdefault((node, ifidx), []).extend(spans)
    order = victims + [k for k in sorted(rows) if k not in victims]
    for i, k in enumerate(order):
        spans = rows.get(k, [])
        a.broken_barh([(s * ms, (e - s) * ms) for s, e in spans], (i - 0.4, 0.8),
                      color=CORAL if k in victims else MUTED,
                      alpha=0.85 if k in victims else 0.35)
    a.set_yticks(range(len(order)))
    a.set_yticklabels([f"n{n}/if{i}" + ("  ←ingress of the bottleneck"
                                        if (n, i) in victims else "")
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
    a.axvline(gate, color=CORAL, ls="--", lw=1.4,
              label=f"decode-start gate = {gate:.2f} ms")
    a.axvspan(lo * ms, hi * ms, color=VIOLET, alpha=0.07,
              label=f"KV window ({(hi-lo)*ms:.2f} ms)")
    a.set_ylabel("KV flows arrived (cumulative)")
    a.set_xlabel("Simulated time (ms)")
    a.legend(fontsize=7, loc="upper left")
    for a in ax:
        a.grid(True, alpha=0.25)
    save_fig(fig, out, "01_timeline.png", written)


def plot_slowdown_ecdf(f, out, written):
    """The full distribution per flow class. No mean, no CV, no p99: an ECDF is
    the data. The TP curve at ~1 is the check that the classification works --
    if it is not at 1, a 'direct' flow is crossing a switch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in roles.FLOW_CLASSES:
        v = f.loc[(f["flow_class"] == c) & f["slowdown"].notna(), "slowdown"]
        if not len(v):
            continue
        v = np.sort(v.to_numpy(float))
        col, ls = class_style(c)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col,
                linestyle=ls, lw=1.6, label=f"{c}  (n={len(v)})")
    ax.set_xscale("log")
    ax.set_xlabel("Slowdown = fct / standalone_fct  (log)")
    ax.set_ylabel("ECDF")
    ax.set_title("Slowdown by flow class\n(standalone_fct assumes the flow owns "
                 "its bottleneck → N flows sharing it fairly give slowdown N)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save_fig(fig, out, "02_slowdown_ecdf.png", written)


def plot_queue_ecdf(qlen, pfc, f, bn, kmin, kmax, out, written):
    """The regime, distributionally instead of by peak.

    `peak / KMAX` is ONE order statistic of a 1.4M-sample distribution, and it is
    the least robust one: a single 100 ns excursion decides it. What the run
    actually says is where the queue LIVES. On the reference buf4 run:

        96.8% of samples inside [KMIN, KMAX]   -> ECN marks, probabilistically
         0.0% of samples above KMAX            -> it never saturates
        peak = 1.03 x KMAX                     -> the peak says "at KMAX"

    Those are three different stories and only the first two are true. The peak
    is a tail event; the mixed regime is the distribution.

    Split by whether the ingress was paused at that instant: if backpressure is
    what holds the queue up, the paused curve sits to the right. That is the
    mechanism, and it costs one extra line."""
    ts, ys = qlen.port_series[(bn.switch, bn.egress_port)]
    ts, ys = np.asarray(ts), np.asarray(ys, dtype=float)
    iv = pfc.pause_intervals(clamp_to=int(f["arrival"].max()))
    paused = np.zeros(len(ts), bool)
    for (node, _nt, ifidx, _q), spans in iv.items():
        for a, b in spans:
            i, j = np.searchsorted(ts, [a, b])
            paused[i:j] = True

    fig, ax = plt.subplots(figsize=(8, 5))
    for v, lab, col in ((ys, f"all samples (n={len(ys):,})", COOL),
                        (ys[paused], f"while an ingress is PAUSED (n={paused.sum():,})", CORAL),
                        (ys[~paused], f"while none is (n={(~paused).sum():,})", GREEN)):
        if len(v) < 2:
            continue
        v = np.sort(v)
        ax.step(v / 1e3, np.arange(1, len(v) + 1) / len(v), where="post",
                color=col, lw=1.5, label=lab)
    ax.axvline(kmin / 1e3, color=MUTED, ls=":", lw=1.4, label=f"KMIN = {kmin/1e3:g} kB")
    ax.axvline(kmax / 1e3, color=MUTED, ls="--", lw=1.4, label=f"KMAX = {kmax/1e3:g} kB")
    ax.axvspan(kmin / 1e3, kmax / 1e3, color=AMBER, alpha=0.08,
               label="ECN ramp: marks with probability p(q)")
    inband = 100 * ((ys >= kmin) & (ys <= kmax)).mean()
    above = 100 * (ys > kmax).mean()
    ax.set_xlabel("Egress queue (kB)")
    ax.set_ylabel("ECDF over qlen samples")
    ax.set_title(f"Where the queue LIVES at {bn}\n"
                 f"{inband:.1f}% of samples in the ECN ramp, {above:.1f}% above KMAX "
                 f"(peak = {ys.max()/kmax:.2f} x KMAX)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, out, "05_queue_ecdf_vs_ecn_band.png", written)


def plot_pause_ecdf(pfc, bn, topo, out, written, clamp: int):
    """Distribution of individual PAUSE durations, log x.

    A pause total is a sum over thousands of events with a median of 9 us; the
    sum cannot tell you whether that is one long stall or continuous fluttering,
    and those are different fabrics. It also cannot show the two artefacts: on
    the reference run the mis-paired intervals appear as isolated points near
    3.6 ms, four orders of magnitude off a p99.9 of 33 us. On a log ECDF you see
    them without being told they are there -- which is the point of plotting a
    distribution rather than reporting its mean."""
    fl = pfc.pause_intervals_flagged(clamp_to=clamp)
    victims = set(bn.pause_victims(topo))
    dev: dict[tuple, list] = {}
    for (n, nt, i, _q), sp in fl.items():
        dev.setdefault((n, i), []).extend(sp)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in sorted(dev):
        d = np.array([e - s for s, e, _ in dev[k] if e > s], float)
        if len(d) < 2:
            continue
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
    ax.set_xscale("log")
    ax.set_xlabel("Duration of one PAUSE interval (ns, log)")
    ax.set_ylabel("ECDF over intervals")
    ax.set_title("PFC pause durations: one long stall, or continuous fluttering?\n"
                 "(a pause total cannot tell them apart; x = intervals the missing "
                 "qIndex may have mis-paired)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, out, "06_pause_duration_ecdf.png", written)


def plot_ports(qlen, topo, bn, kmax, out, written):
    """Peak queue on every switch port that ever held one. This is the figure
    that answers 'is the port I think is the bottleneck the port that actually
    built the queue' -- the sweep asserts one link and shows you nothing else."""
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
    ax.axvline(kmax / 1e3, color=GREEN, ls="--", lw=1.2, label=f"KMAX = {kmax/1e3:g} kB")
    ax.set_xlabel("Peak egress queue (kB)")
    ax.set_title(f"Where the queues actually are (red = {bn}, the analysed link)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    save_fig(fig, out, "03_queue_peak_by_port.png", written)


def plot_gantt(kv_bn, bn, out, written):
    """One bar per KV flow at the bottleneck, start -> arrival, sorted by start.

    A block of parallel bars means the flows share the link; a staircase means
    they serialise. That distinction is invisible in a mean slowdown and it is
    exactly what the barrier depends on."""
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
    ax.set_title(f"KV flows crossing {bn}: parallel = sharing, staircase = serialising")
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(fontsize=8)
    save_fig(fig, out, "04_kv_gantt.png", written)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    roles.add_argument(ap)
    ap.add_argument("--tag", required=True, help="run sub-directory, e.g. "
                                                 "'T1_bx200_dcqcn_buf8'")
    ap.add_argument("--bottleneck", default=None, help="'sw->peer'. Default: "
                                                       "measured from qlen.txt.")
    ap.add_argument("--headroom-factor", type=int, default=None,
                    help="ns-3 common.h::headroom_factor. Default: HEADROOM_FACTOR "
                         "from config.txt, else 3.")
    a = ap.parse_args(argv)

    try:
        p, outbase = paths.from_arguments(a, KIND)
        outdir = outbase / a.tag
        need(not p.missing_roots(),
             "derived root(s) do not exist:\n    " + "\n    ".join(p.missing_roots()))
        ns3_dir, tpath, cpath = p.ns3_run(a.tag), p.topology(a.tag), p.config(a.tag)
        for q in (ns3_dir / "fct.txt", ns3_dir / "pfc.txt", ns3_dir / "qlen.txt",
                  tpath, cpath):
            need(q.exists(), f"missing {q}\n  --tag {a.tag!r} may be wrong; "
                             f"{p.ns3_root} has: {sorted(x.name for x in p.ns3_root.iterdir())[:8]}")

        cfg = parse_ns3_config(cpath)
        hf = a.headroom_factor if a.headroom_factor is not None else cfg.headroom_factor
        need(cfg.buffer_mb is not None, f"no BUFFER_SIZE in {cpath}.")
        topo = parse_topology(tpath, hf)
        placement = Placement.parse(a.placement)
        buffer_bytes = cfg.buffer_bytes

        say = Report()
        say.head(f"ns-3 run: {a.tag}")
        say(f"  ns3      {ns3_dir}")
        say(f"  topology {tpath}")
        say(f"  config   {cpath}")
        say(f"  out      {outdir}")
        say(f"  BUFFER_SIZE = {cfg.buffer_mb:g} MiB (read)   CC_MODE = {cfg.cc_mode}"
            f"   headroom_factor = {hf}"
            f"{'  (read from config.txt)' if 'HEADROOM_FACTOR' in cpath.read_text() else '  (ASSERTED — not in config.txt)'}")
        say(f"  placement{placement.describe()}")
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
        say(f"  min slowdown = {d['slow_min']:.4f}  (entry.h bills standalone_fct "
            f"over payload+headers while the size column is payload only, so an "
            f"uncongested flow sits a percent or two below 1 — that is not a "
            f"misparse)")
        if d["slow_lt09"]:
            say(f"  ! {d['slow_lt09']} flows have slowdown < 0.9, which headers "
                f"cannot explain: columns 5-8 are probably not where expected.")
        if d["sfct_nonpos"]:
            say(f"  ! {d['sfct_nonpos']} flows have standalone_fct <= 0 (no slowdown).")

        f = flowlib.annotate(raw, topo, placement, cfg.payload)
        for w in roles.check(f, placement):
            say(f"  ! {w}")
        kv = f[f["flow_class"] == "kv"]
        need(len(kv), "no KV flow after classification: --placement is wrong.")

        say.head("Flows by class (structural, from --placement — no size threshold)")
        for c in roles.FLOW_CLASSES:
            g = f[f["flow_class"] == c]
            if not len(g):
                continue
            sd = g.loc[g["slowdown"].notna(), "slowdown"]
            say(f"  {c:<11} n={len(g):<5} bytes={fmt_b(g['size'].sum()):>9}  "
                f"slowdown p50={np.percentile(sd,50):7.2f} p99={np.percentile(sd,99):7.2f} "
                f"max={sd.max():7.2f}" if len(sd) else f"  {c:<11} n={len(g)}")

        # -- the bottleneck ------------------------------------------------- #
        qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=True)
        need(qlen is not None and qlen.port_max, "qlen.txt has no samples.")
        if a.bottleneck:
            sw, peer = (int(x) for x in a.bottleneck.split("->"))
            eg = topo.port_facing(sw, peer)
            need(eg is not None, f"--bottleneck {a.bottleneck}: no such link.")
            ing = {q for path in kv["path"] for i, (x, y) in enumerate(path or [])
                   if (x, y) == (sw, peer) and i > 0
                   and (q := topo.port_facing(sw, path[i - 1][0])) is not None}
            bn = Bottleneck(sw, eg, peer, topo.ports[sw][eg].rate, tuple(sorted(ing)))
        else:
            bn = flowlib.find_bottleneck(topo, qlen.port_max, kv)
        need(bn.f_ports, f"no KV flow enters {bn} through a known ingress port.")
        model = FabricModel(topo, cfg)
        kmin, kmax = model.ecn_band(bn)
        need(kmin is not None, f"no KMIN/KMAX for {bn.rate} bit/s in config.txt.")

        say.head(f"Bottleneck {bn} — measured (deepest queue), model on top")
        say(f"  egress {topo.port_label(bn.switch, bn.egress_port)} @ "
            f"{bn.rate/1e9:g} Gbps | F_ports = {bn.f_ports} "
            f"(ingress {[topo.port_label(bn.switch, i) for i in bn.ingress_ports]})")
        say(f"  PAUSE victims in pfc.txt: {sorted(bn.pause_victims(topo))}")
        peak = qlen.port_max[(bn.switch, bn.egress_port)]
        thr = model.steady_threshold(bn, buffer_bytes)
        ceil = model.pfc_egress_ceiling(bn, buffer_bytes)
        say(f"  MODEL  steady PFC threshold A/(2^{topo.shift[bn.switch][bn.egress_port]}"
            f"+F) = {fmt_b(thr)} | x F_ports = {fmt_b(thr*bn.f_ports)} | "
            f"ceiling = {fmt_b(ceil)} | regime = {model.regime(bn, buffer_bytes)}")
        say(f"  ECN    KMIN = {fmt_b(kmin)}   KMAX = {fmt_b(kmax)}")
        say(f"  MEAS   peak egress = {fmt_b(peak)}  "
            f"(= {peak/kmax:.2f} x KMAX, {peak/ceil:.2f} x ceiling)")
        if peak > ceil:
            say(f"  ! peak exceeds the modelled ceiling, which is an UPPER bound "
                f"(the threshold is at its fixed point), so this is impossible and "
                f"the model is wrong. Ruled out in the ns-3 source: headroom_factor "
                f"= 3 (common.h:86); ingress and egress account the same packets "
                f"(RemoveFromIngressAdmission runs at dequeue); no overflow in "
                f"headroom. Still open: hdrm_bytes[port][qIndex] is per-queue while "
                f"its limit headroom[port] is per-port and only ONE is reserved per "
                f"port. qlen.txt cannot show it — monitor_buffer sums egress_bytes "
                f"over all queues, ShouldSendCN tests one (switch-mmu.cc:103).")

        say.head("Every switch port that held a queue")
        say(f"    {'port':<20} {'peak':>10} {'mean':>10} {'samples':>8}")
        for (s, pt), v in sorted(qlen.port_max.items(), key=lambda kv: -kv[1]):
            mark = "  <== analysed" if (s, pt) == (bn.switch, bn.egress_port) else ""
            say(f"    sw{s}:{topo.port_label(s, pt):<16} {fmt_b(v):>10} "
                f"{fmt_b(qlen.port_mean[(s,pt)]):>10} {qlen.port_count[(s,pt)]:>8}{mark}")

        # -- PFC ------------------------------------------------------------ #
        pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
        need(pfc is not None, "pfc.txt unreadable.")
        say.head("pfc.txt")
        say(f"  events {pfc.n_events} | qIndex column: {pfc.qidx_state}")
        fl = pfc.pause_intervals_flagged(clamp_to=int(f["arrival"].max()))
        dev = {}
        for (n, nt, i, _q), sp in fl.items():
            dev.setdefault((n, nt, i), []).extend(sp)
        if pfc.qidx_state == "MISSING":
            say("  ! no qIndex column. get_pfc drops the qIndex that "
                "qbb-net-device.cc:383 has in scope, so PAUSE/RESUME of queue 0 "
                "(ACK/CNP, with ACK_HIGH_PRIO) and of the data queue land on the "
                "same (node, ifindex) key. That only mis-pairs where the two "
                "sequences actually INTERLEAVE, which leaves a fingerprint: two "
                "same-type events in a row. Measured below, per device.")
            say(f"\n    {'device':<16} {'intervals':>9} {'suspect':>8} {'total':>11} "
                f"{'suspect':>10} {'max error':>10}")
            for k in sorted(dev):
                iv = dev[k]
                tot = sum(e - s_ for s_, e, _ in iv)
                sus = sum(e - s_ for s_, e, q in iv if q)
                mark = "  <== ingress of the bottleneck" if (k[0], k[2]) in set(
                    bn.pause_victims(topo)) else ""
                say(f"    n{k[0]}/if{k[2]} ({'sw' if k[1] else 'host'}){'':<3} "
                    f"{len(iv):>9} {sum(1 for *_, q in iv if q):>8} {tot*1e-6:>8.2f} ms "
                    f"{sus*1e-6:>7.3f} ms {100*sus/tot if tot else 0:>9.1f}%{mark}")
            say("\n    'suspect' brackets the error: the truth is in "
                "[total - suspect, total]. Apply --print-patch to remove the bracket.")
        lo, hi = int(kv["start"].min()), int(kv["arrival"].max())
        totals, unclosed = pfc.pause_totals(clamp_to=int(f["arrival"].max()))
        say(f"  KV window at the bottleneck: [{lo*1e-6:.2f}, {hi*1e-6:.2f}] ms "
            f"({(hi-lo)*1e-6:.2f} ms) | unclosed PAUSEs: {unclosed}")
        if totals:
            say(f"\n    {'device':<12} {'q':>3} {'paused (total run)':>19} "
                f"{'of which in the KV window':>26}")
            for (n, nt, i, q), tp in sorted(totals.items(), key=lambda x: -x[1])[:12]:
                spans = pfc.pause_intervals(clamp_to=int(f["arrival"].max()))[(n, nt, i, q)]
                inw = sum(max(0, min(e, hi) - max(s, lo)) for s, e in spans)
                v = "  <== ingress of the bottleneck" if (n, i) in set(
                    bn.pause_victims(topo)) else ""
                say(f"    n{n}/if{i} ({'sw' if nt else 'host'}){'':<2} {q:>3} "
                    f"{tp*1e-6:>15.3f} ms {inw*1e-6:>19.3f} ms "
                    f"({100*inw/(hi-lo):5.1f}%){v}")

        # -- the barrier ----------------------------------------------------- #
        say.head("Barrier: what gates decode")
        for r in placement.decode_ranks:
            arr = kv.loc[kv["dst"] == r, "arrival"]
            if not len(arr):
                say(f"  decode rank {r}: receives NO KV flow")
                continue
            say(f"  decode rank {r}: {len(arr):>3} KV flows | first "
                f"{arr.min()*1e-6:8.3f} ms | KV-ready {arr.max()*1e-6:8.3f} ms | "
                f"stream duration {(arr.max()-arr.min())*1e-6:8.3f} ms")
        gate = kv["arrival"].max()
        say(f"\n  decode-start gate = max KV-ready = {gate*1e-6:.3f} ms")

        # -- outputs --------------------------------------------------------- #
        outdir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        kv_bn = kv[flowlib.crosses(kv, bn)]
        need(len(kv_bn), f"no KV flow crosses {bn}.")
        plot_timeline(say, qlen, pfc, f, kv, bn, topo, model, buffer_bytes,
                      kmin, kmax, outdir, written)
        plot_queue_ecdf(qlen, pfc, f, bn, kmin, kmax, outdir, written)
        plot_pause_ecdf(pfc, bn, topo, outdir, written, int(f["arrival"].max()))
        plot_slowdown_ecdf(f, outdir, written)
        plot_ports(qlen, topo, bn, kmax, outdir, written)
        plot_gantt(kv_bn, bn, outdir, written)
        f.drop(columns=["path"]).to_csv(outdir / "flows.csv", index=False)
        pd.DataFrame([{"switch": s, "port": pt,
                       "label": topo.port_label(s, pt), "peak_bytes": v,
                       "mean_bytes": qlen.port_mean[(s, pt)],
                       "samples": qlen.port_count[(s, pt)],
                       "is_bottleneck": (s, pt) == (bn.switch, bn.egress_port)}
                      for (s, pt), v in qlen.port_max.items()]
                     ).sort_values("peak_bytes", ascending=False).to_csv(
            outdir / "ports.csv", index=False)
        say(f"\nWrote {outdir}:")
        for q in ["report.txt", "flows.csv", "ports.csv", *[w.name for w in written]]:
            say(f"  {q}")
        (outdir / "report.txt").write_text("\n".join(say.lines) + "\n")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())