#!/usr/bin/env python3
"""
ns3_analyzer.py  —  Analyze the ns-3 outputs (fct.txt, pfc.txt, qlen.txt) of ONE
run from the astra-network-ns3 fork (branch qos-impl) and render a set of figures
that make the congestion regime (DCQCN vs PFC) and the slowdown tail immediately
visible.

--------------------------------------------------------------------------------
FILE FORMATS (verified against scratch/common.h on branch qos-impl)
--------------------------------------------------------------------------------
fct.txt   (writer: qp_finish, astra-sim frontend; classic HPCC layout)
    sip dip sport dport size(B) start(ns) fct(ns) standalone_fct(ns)
    - sip/dip are 8-hex-digit Ipv4Address integers ("0b000005"); dotted-decimal
      ("11.0.0.5") is also accepted.  node_id = (ip >> 8) & 0xffff.
    - slowdown = fct / standalone_fct   (1.0 == no congestion).

pfc.txt   (writer: get_pfc, common.h:141  ->  "%lu %u %u %u %u")
    time(ns) node_id node_type if_index type
    - node_type : 0 = host/server, 1 = switch
    - type      : 0 = RESUME, 1 = PAUSE   (trace src "QbbPfc", qbb-net-device.cc:227)

qlen.txt  (writer: monitor_buffer, common.h:159)
    time <ns> <switch_node_id> j <port> <bytes> j <port> <bytes> ...
    - one line per switch per sample (interval = QLEN_MON_INTERVAL ns).
    - <bytes> is summed over ALL priority queues of that egress port.
    - a port is only emitted when its occupancy >= 1000 B.
    - switch_node_id == node id (nodes are created in order 0..N-1).

topology (enables port -> neighbour labelling): auto-detected at
    configs/astra_sim/ns3/<run>/physical_topology.txt (override with --topology).
    line1: node_num switch_num link_num
    line2: <switch node ids ...>
    then link_num lines: src dst rate delay error
    ns-3 assigns link device ifIndex 1..K per node in link-file order (0 = loopback),
    so a switch port j maps to the j-th link (file order) that touches that switch.
    A host-facing switch port is where incast lands.

BUFFER_SIZE is per-switch, in MiB (common.h:746 -> *1024*1024).  Crucially it is a
    SHARED pool across all ports and queues of the switch: switch-mmu.cc keeps a
    single shared_used_bytes counter and the dynamic PFC threshold is
    (buffer_size - headroom - reserve - shared_used_bytes) >> a_shift.  So the metric
    that governs PFC is the per-SWITCH aggregate occupancy (sum of its ports at each
    sample = one qlen.txt line), NOT any single port versus the full buffer.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 ns3_analyzer.py <run>  [options]

<run> may be:
    * a subdirectory name looked up under  ../output/ns3/<run>  (relative to this
      script or current working directory), or
    * a direct path to a folder that contains fct.txt / pfc.txt / qlen.txt.

Options:
    --out-dir DIR     where to write the report + figures.
                      default: ../results/ns3_graphs/<run basename>
    --buffer-mb F     per-switch buffer in MiB (default 16)
    --bulk-mb F       flow-size threshold (MB) to count as "bulk" KV/PP (default 1)
    --topology FILE   topology file, to label switch ports with their neighbour
    --top N           how many worst links/ports to show in tables & plots (10)
    --dpi N           figure DPI (default 130)
    --no-plots        text report only (skip figures)
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from collections import defaultdict

from utils import ns3 as ns3io
from utils import flows as flowlib
from utils.fabric import FabricModel, parse_ns3_config, parse_topology
from utils.plots import save_fig
from utils.sweep import (BUFFER_AXIS, find_aux, find_under_roots,
                            project_roots)

# Matplotlib is optional: if it is missing we still print the text report.
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch
    HAVE_MPL = True
except Exception as _e:            # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = str(_e)

# ============================================================================ #
#  Visual theme  — one coherent, colour-blind-safe palette used everywhere.
# ============================================================================ #
INK      = "#1c2333"   # near-black text / axes
MUTED    = "#6b7280"   # secondary text / grid
PAPER    = "#ffffff"   # figure background
PANEL    = "#f6f7f9"   # light panel fill
GRID     = "#e3e6eb"
COOL     = "#2b6cb0"   # primary (healthy / DCQCN)  – blue
TEAL     = "#0e7c7b"   # secondary                  – teal
AMBER    = "#d98a00"   # warning                    – amber
CORAL    = "#d1495b"   # danger / PFC / tail        – red
VIOLET   = "#6a4c93"   # accent
OKGREEN  = "#2a9d5c"   # good

def _apply_theme():
    if not HAVE_MPL:
        return
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
    })

# heat colormap: quiet paper -> teal -> amber -> coral (occupancy severity)
def _heat_cmap():
    return LinearSegmentedColormap.from_list(
        "occ", [PANEL, "#bfe3e0", TEAL, AMBER, CORAL])

# ============================================================================ #
#  Small numeric helpers (pure python, used by the text report)
# ============================================================================ #
def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k); c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")

def stdev(vals, m=None):
    if len(vals) < 2:
        return 0.0
    m = mean(vals) if m is None else m
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

def fmt_bytes(b):
    b = float(b)
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} TB"

def fmt_ns(ns):
    ns = float(ns)
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns/1e3:.2f} us"
    if ns < 1e9:
        return f"{ns/1e6:.3f} ms"
    return f"{ns/1e9:.3f} s"

def ip_to_node(tok):
    """node id from an fct.txt address token. See sweeplib.ns3.ip_to_node."""
    return ns3io.ip_to_node(tok)

# ============================================================================ #
#  Topology  (optional) : switch port -> neighbour node id
# ============================================================================ #
def load_topology(path):
    """Return (Topology|None, switch_ids:set, portmap:{(node,ifindex):neighbor}).

    The Topology object is what makes the regime verdict quantitative: it carries
    the headroom, the reserve and the per-port pfc_a_shift, hence the PFC
    threshold and the ceiling the egress queue settles at. `portmap` is kept for
    the human-readable port labels the report already prints."""
    try:
        topo = parse_topology(Path(path))
    except Exception as exc:
        print(f"  ! cannot parse topology {path}: {exc}", file=sys.stderr)
        return None, set(), {}
    portmap = {(n, i): pt.peer for n, ports in topo.ports.items() for i, pt in ports.items()}
    return topo, set(topo.switches), portmap

def port_label(node, ifidx, portmap, switch_ids):
    """'p3->sw12' / 'p2->h0'. Thin shim over Topology.port_label so the plotting
    code keeps its existing signature; the convention lives in sweeplib.fabric."""
    nb = portmap.get((node, ifidx))
    if nb is None:
        return f"p{ifidx}"
    return f"p{ifidx}->{'sw' if nb in switch_ids else 'h'}{nb}"

# ============================================================================ #
#  Parsers  — return plain dicts reused by both the text report and the plots
# ============================================================================ #
def parse_fct(path, bulk_bytes, topo=None):
    """Adapter over sweeplib.ns3.read_fct + sweeplib.flows.annotate.

    Same dict shape the report and the plots already consume, but the flows are
    now split by the taxonomy: `slow_bulk` is the INCAST population (bulk AND
    crossing a switch), not a bimodal mixture of congested KV transfers and
    tensor-parallel all-reduces on dedicated host-to-host links. Those direct
    flows have slowdown ~1 by construction; blending them in makes the mean and
    the CV describe the mixing ratio, which is a constant of the workload, and so
    look flat no matter what the fabric does. `slow_direct` keeps them visible.

    With no topology every flow is treated as fabric and nothing is filtered --
    the caller warns."""
    df = ns3io.read_fct(Path(path))
    d = dict(n=0, sizes=[], starts=[], fcts=[], sfcts=[], slow=[], slow_bulk=[],
             slow_direct=[], sd_size=[], sd_start=[], worst=[], run_end=0,
             per_dst_bulk=defaultdict(int), per_dst_bytes=defaultdict(int),
             per_dst_slow=defaultdict(list), n_fabric=0, n_direct=0,
             filtered=topo is not None,
             raw_lines=0, skipped_short=0, skipped_badnum=0, sfct_nonpos=0,
             slow_lt1=0, ncols=None, sample=None, ncol_hist=defaultdict(int))
    if df is None:
        return d
    d.update({k: v for k, v in df.attrs.get("diagnostics", {}).items()})
    f = flowlib.annotate(df, topo, bulk_bytes)
    d["n"] = len(f)
    d["sizes"] = f["size"].tolist()
    d["starts"] = f["start"].tolist()
    d["fcts"] = f["fct"].tolist()
    d["sfcts"] = f["sfct"].tolist()
    d["run_end"] = int(f["arrival"].max())
    d["n_fabric"] = int(f["fabric"].sum())
    d["n_direct"] = int((~f["fabric"]).sum())
    ok = f[f["slowdown"].notna()]
    d["slow"] = ok["slowdown"].tolist()
    d["sd_size"] = ok["size"].tolist()
    d["sd_start"] = ok["start"].tolist()
    d["worst"] = list(zip(ok["slowdown"], ok["size"], ok["src"], ok["dst"],
                          ok["fct"], ok["start"]))
    d["slow_bulk"] = ok.loc[ok["incast"], "slowdown"].tolist()
    d["slow_direct"] = ok.loc[~ok["fabric"], "slowdown"].tolist()
    for dst, sd in zip(ok["dst"], ok["slowdown"]):
        d["per_dst_slow"][dst].append(sd)
    inc = ok[ok["incast"]]
    for dst, size in zip(inc["dst"], inc["size"]):
        d["per_dst_bulk"][dst] += 1
        d["per_dst_bytes"][dst] += size
    d["_flows"] = f            # the annotated frame, for find_bottleneck
    return d

def parse_pfc(path):
    """Adapter over sweeplib.ns3.read_pfc.

    Keys now include the qIndex (an optional 6th column). m_tracePfc fires per
    qIndex but get_pfc does not print it, so with qos-enabled the PAUSE/RESUME
    sequences of different priority groups interleave on one ifindex and a state
    machine keyed on (node, ifindex) mis-pairs them. `qidx_state` says whether
    the column is there; when it is MISSING the pause totals are not a
    measurement. See sweeplib.ns3.PFC_QIDX_PATCH for the three-line ns-3 diff."""
    log = ns3io.read_pfc(Path(path))
    if log is None:
        return dict(ev={}, raw=[], n=0, tmin=0, tmax=0, qidx_state="n/a")
    raw = [(t, node, ntype, ifidx, typ)
           for (node, ntype, ifidx, _q), evs in log.events.items()
           for (t, typ) in evs]
    raw.sort()
    return dict(ev=log.events, raw=raw, n=log.n_events,
                tmin=min((r[0] for r in raw), default=0), tmax=log.t_max,
                qidx_state=log.qidx_state, log=log)

def pfc_intervals(pfcd, clamp_end):
    """Closed [start,end] pause intervals per key, plus totals.

    Delegates to sweeplib.ns3.PfcLog: the PAUSE/RESUME state machine exists once.
    Two copies of it is how the four analyzers ended up parsing the same file
    three different ways."""
    log = pfcd.get("log")
    if log is None:
        return {}, {}
    intervals = log.pause_intervals(clamp_end)
    totals, _ = log.pause_totals(clamp_end)
    return intervals, totals

def parse_qlen(path):
    """Adapter over sweeplib.ns3.read_qlen(series=True).

    monitor_buffer dumps ``sum_k egress_bytes[port][k]`` for ONE port, and only
    while that port's queue is >= 1000 B. So this is PER-EGRESS-PORT occupancy --
    exactly what ShouldSendCN() compares against kmin/kmax[ifindex]. It is NOT
    the shared pool: that is ingress-side accounting (shared_used_bytes) and is
    not observable from this file at all. The per-switch sum is therefore "total
    bytes queued for egress on that switch", a useful proxy for how full the
    switch is, but it must not be compared against the PFC threshold."""
    q = ns3io.read_qlen(Path(path), series=True)
    if q is None:
        return dict(series={}, hist={}, mx={}, cnt={}, ssum={}, sw_series={},
                    sw_hist={}, sw_mx={}, sw_cnt={}, sw_ssum={}, tmin=0, tmax=0)
    return dict(series=q.port_series, hist=q.port_hist, mx=q.port_max,
                cnt=q.port_count,
                ssum={k: q.port_mean[k] * q.port_count[k] for k in q.port_count},
                sw_series=q.switch_series, sw_hist=q.switch_hist,
                sw_mx=q.switch_total_max, sw_cnt=q.switch_count,
                sw_ssum={s: sum(q.switch_series[s][1]) for s in q.switch_count},
                tmin=q.t_min, tmax=q.t_max)

def hpct(h, total, pp):
    """percentile from a {kb:count} histogram (returns bytes)."""
    target = pp * total; acc = 0
    for kb in sorted(h):
        acc += h[kb]
        if acc >= target:
            return kb * 1000
    return 0

# ============================================================================ #
#  A tiny tee: print to console AND collect into the report file.
# ============================================================================ #
class Report:
    def __init__(self):
        self.lines = []
    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)
    def header(self, title):
        self("\n" + "=" * 74); self(title); self("=" * 74)
    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")

# ============================================================================ #
#  TEXT REPORT
# ============================================================================ #
def report_fct(say, d, bulk_bytes):
    # ---- parse sanity: catch a wrong column layout before trusting anything ----
    say("  --- parse sanity (fct.txt layout is the astra-sim frontend's, not this "
        "repo's) ---")
    cols = sorted(d["ncol_hist"].items(), key=lambda x: -x[1])
    col_desc = ", ".join(f"{c}c×{n}" for c, n in cols[:4]) if cols else "-"
    say(f"    lines read {d['raw_lines']} | parsed {d['n']} | "
        f"skipped(<8 cols) {d['skipped_short']} | skipped(non-numeric) "
        f"{d['skipped_badnum']} | column-counts: {col_desc}")
    if d["sample"]:
        say(f"    first line: {d['sample']}")
        say("    expected  : sip  dip  sport dport size(B) start(ns) fct(ns) "
            "standalone_fct(ns)")
    if d["sizes"]:
        say(f"    ranges: size [{min(d['sizes'])}..{max(d['sizes'])}] B | "
            f"fct [{min(d['fcts'])}..{max(d['fcts'])}] ns | "
            f"sfct [{min(d['sfcts'])}..{max(d['sfcts'])}] ns")
    warned = False
    if d["slow_lt1"] > 0.02 * max(d["n"], 1):
        say(f"    [!] {d['slow_lt1']} flows have slowdown < 1 (fct < standalone_fct). "
            "Slowdown < 1 is physically impossible, so columns 5-8 (size/start/fct/"
            "standalone) are likely NOT where I expect — check the first line above.")
        warned = True
    if d["skipped_short"] > 0.5 * max(d["raw_lines"], 1):
        say("    [!] more than half the lines have <8 columns: the fct.txt format "
            "differs from what I parse. Send me one line and I'll adjust the indices.")
        warned = True
    if d["sfct_nonpos"] > 0:
        say(f"    note: {d['sfct_nonpos']} flows had standalone_fct <= 0 "
            "(excluded from slowdown).")
    if not warned:
        say("    OK: columns look consistent and slowdown >= 1 as expected.")
    say("")

    if d["n"] == 0:
        say("  (no valid flows)"); return
    say(f"  total flows          : {d['n']}")
    say(f"  total bytes          : {fmt_bytes(sum(d['sizes']))}")
    ss = sorted(d["sizes"])
    say(f"  flow size            : min {fmt_bytes(ss[0])} | "
        f"p50 {fmt_bytes(pct(ss,0.5))} | max {fmt_bytes(ss[-1])}")
    say(f"  bulk flows (>= {fmt_bytes(bulk_bytes)}) : {len(d['slow_bulk'])}")

    def block(label, vals):
        if not vals:
            say(f"  {label}: (none)"); return None
        v = sorted(vals); m = mean(v); cv = stdev(v, m) / m if m else 0.0
        say(f"  {label}:")
        say(f"      mean {m:7.3f} | p50 {pct(v,0.5):7.3f} | p90 {pct(v,0.9):7.3f} "
            f"| p99 {pct(v,0.99):7.3f} | max {v[-1]:8.3f}")
        say(f"      CV (std/mean) = {cv:5.3f}   <-- tail: high = unfair (PFC/HOL regime)")
        return dict(mean=m, p50=pct(v,0.5), p90=pct(v,0.9), p99=pct(v,0.99),
                    max=v[-1], cv=cv)

    say("\n  SLOWDOWN = fct / standalone_fct   (1.0 = no congestion)")
    s_all = block("all flows      ", d["slow"])
    s_bulk = block("bulk only      ", d["slow_bulk"])

    say("\n  Top 8 most slowed-down flows (decode-start straggler candidates):")
    say(f"    {'slowdown':>9}  {'size':>10}  {'src->dst':>10}  {'fct':>10}")
    for sd, size, src, dst, fct, start in sorted(d["worst"], reverse=True)[:8]:
        say(f"    {sd:9.2f}  {fmt_bytes(size):>10}  {src:>4}->{dst:<4}  {fmt_ns(fct):>10}")

    if d["per_dst_bulk"]:
        say("\n  Incast: destinations with the most concurrent incoming bulk flows:")
        say(f"    {'dst_node':>8}  {'#bulk':>6}  {'total bytes':>12}")
        for dst, c in sorted(d["per_dst_bulk"].items(), key=lambda x: -x[1])[:6]:
            say(f"    {dst:>8}  {c:>6}  {fmt_bytes(d['per_dst_bytes'][dst]):>12}")
    return dict(all=s_all, bulk=s_bulk)

def report_pfc(say, pfcd, totals, run_span, portmap, switch_ids, topo=None, bn=None):
    if pfcd["n"] == 0:
        say("  (no PFC events — no pause: good sign, DCQCN regime)")
        return dict(events=0, paused_links=0, tot_pause=0, max_frac=0.0,
                    bottleneck_frac=0.0)
    if pfcd.get("qidx_state") == "MISSING":
        say("  !! pfc.txt has no qIndex column. m_tracePfc fires per queue but "
            "get_pfc does not print it, so with qos-enabled the PAUSE/RESUME "
            "sequences of different priority groups interleave on one ifindex and "
            "the state machine mis-pairs them. The numbers below are NOT a "
            "measurement. See sweeplib.ns3.PFC_QIDX_PATCH.")
    span = run_span or (pfcd["tmax"] - pfcd["tmin"]) or 1
    tot_pause = sum(totals.values())
    paused_links = sum(1 for v in totals.values() if v > 0)
    say(f"  total PFC events     : {pfcd['n']}")
    say(f"  links (node,port) that saw a pause : {paused_links}")
    say(f"  aggregate pause-time : {fmt_ns(tot_pause)}")
    say(f"  run window           : {fmt_ns(span)}")
    say("\n  Top links by pause-time (fraction of the run spent in PAUSE):")
    say(f"    {'node':>5} {'type':>6} {'port':>16}  {'pause-time':>12}  {'% run':>7}")
    max_frac = 0.0
    # Keys are (node, node_type, ifindex, qIndex). Queues of one device overlap in
    # time, so they are shown per queue and the device max is what is reported --
    # summing across devices and dividing by a time window can trivially exceed
    # 100%, which is what "283% paused" looked like.
    for key, tp in sorted(totals.items(), key=lambda x: -x[1])[:10]:
        node, ntype, ifidx, qidx = key
        if tp == 0:
            continue
        kind = "switch" if ntype == 1 else "host"
        lbl = port_label(node, ifidx, portmap, switch_ids)
        q = f" q{qidx}" if qidx >= 0 else ""
        frac = 100 * tp / span; max_frac = max(max_frac, frac)
        say(f"    {node:>5} {kind:>6} {lbl + q:>16}  {fmt_ns(tp):>12}  {frac:6.2f}%")
    # Pause on the devices upstream of the CONGESTED link specifically. The global
    # worst can sit on a completely unrelated link, so it is not evidence about
    # this bottleneck's regime. The trace fires on the device RECEIVING the pause
    # frame -- the victim -- so the keys to look for are the peers' ports.
    bfrac = float("nan")
    if topo is not None and bn is not None:
        victims = set(bn.pause_victims(topo))
        vals = [tp for (node, _nt, ifidx, *_), tp in totals.items()
                if (node, ifidx) in victims]
        bfrac = 100 * max(vals) / span if vals else 0.0
        say(f"\n  Pause on the bottleneck's own ingress ({sorted(victims)}): "
            f"{bfrac:.2f}% of the run  <== this, not the global worst, is what "
            f"{bn} 's regime rests on")
    return dict(events=pfcd["n"], paused_links=paused_links, tot_pause=tot_pause,
                max_frac=max_frac, bottleneck_frac=bfrac, span=span)

def report_qlen(say, qd, buffer_bytes, portmap, switch_ids, top, model=None, bn=None):
    if not qd["cnt"]:
        say("  (no congested port recorded — queues always < 1KB: DCQCN regime)")
        return dict(ports=0, switches=0, max_pct=0.0, near_buffer=0)
    say(f"  buffer per switch    : {fmt_bytes(buffer_bytes)}  (SHARED across all ports)")
    say(f"  switches with traffic: {len(qd['sw_cnt'])}   congested ports: {len(qd['cnt'])}")
    win = qd["tmax"] - qd["tmin"]
    say(f"  sampled window       : {fmt_ns(qd['tmin'])} .. {fmt_ns(qd['tmax'])}  "
        f"(span {fmt_ns(win)})")
    say("    NB: only ports with >= 1KB are logged, and only within "
        "[QLEN_MON_START, QLEN_MON_END]. A short window or a well-behaved DCQCN run "
        "both make this look sparse.")

    # --- the regime test: is the congested port's queue at the PFC ceiling? ---
    # The ceiling is Sum over ingress ports of (reserve + threshold + headroom):
    # where the egress queue settles once PFC has paused every ingress port and
    # their headroom has absorbed the packets already in flight. At the ceiling
    # the queue is held by backpressure; below it, the rate control is limiting.
    #
    # Comparing the queue against a fraction of the BUFFER instead is not just
    # imprecise, it is unreachable: on an oversubscribed leaf whose host ports
    # carry a large rate*delay headroom, the ceiling itself is ~50% of a 2 MiB
    # buffer and ~27% of a 32 MiB one, so an "80% of buffer" flag can never fire
    # even in full PFC.
    ceiling = None
    if model is not None and bn is not None and bn.f_ports:
        ceiling = model.pfc_egress_ceiling(bn, buffer_bytes)
        peak = qd["mx"].get((bn.switch, bn.egress_port))
        kmin, kmax = model.ecn_band(bn)
        say(f"\n  Congested link {bn}  (egress p{bn.egress_port} @ "
            f"{bn.rate/1e9:g} Gbps, F_ports={bn.f_ports})")
        say(f"    PFC threshold (per ingress port) : {fmt_bytes(model.pfc_threshold(bn, buffer_bytes))}")
        say(f"    PFC egress ceiling               : {fmt_bytes(ceiling)}"
            f"   ({100*ceiling/buffer_bytes:.1f}% of buffer)")
        if kmin and kmax:
            say(f"    ECN band                         : KMIN {fmt_bytes(kmin)} .. "
                f"KMAX {fmt_bytes(kmax)}")
        if peak:
            r = peak / ceiling if ceiling else float("nan")
            verdict = ("at the ceiling -> held by BACKPRESSURE" if r >= 0.95
                       else "below the ceiling -> RATE CONTROL is limiting")
            say(f"    measured peak egress             : {fmt_bytes(peak)}"
                f"   peak/ceiling = {r:.3f}   <== {verdict}")
    else:
        say("\n  (no topology/config: no PFC threshold, no ceiling, no regime test. "
            "Pass --topology and --config.)")

    # --- context: total bytes queued for egress, per switch ------------------
    say("\n  Total bytes queued for EGRESS per switch (sum over its ports).")
    say("    NB: this is NOT the shared pool. monitor_buffer dumps egress_bytes;")
    say("        the shared pool is ingress accounting (shared_used_bytes) and is")
    say("        not observable from qlen.txt. Do not compare it to the threshold.")
    say(f"    {'switch':>6}  {'max':>10} {'% buf':>6}  {'mean':>10}  {'p99':>10}")
    near_buffer = 0; max_pct = 0.0
    for sw in sorted(qd["sw_cnt"], key=lambda s: -qd["sw_mx"][s]):
        m = qd["sw_mx"][sw]; av = qd["sw_ssum"][sw] / qd["sw_cnt"][sw]
        p99 = hpct(qd["sw_hist"][sw], qd["sw_cnt"][sw], 0.99)
        pctbuf = 100 * m / buffer_bytes; max_pct = max(max_pct, pctbuf)
        # Flag against the PFC ceiling, which is where the queue physically stops,
        # rather than against the buffer, which it can never approach.
        ref = ceiling if ceiling else buffer_bytes
        pctref = 100 * m / ref
        flag = "  <== at the PFC ceiling" if pctref >= 95 else ""
        if pctref >= 95:
            near_buffer += 1
        say(f"    {sw:>6}  {fmt_bytes(m):>10} {pctbuf:5.1f}%  "
            f"{fmt_bytes(av):>10}  {fmt_bytes(p99):>10}{flag}")

    # --- secondary view: which ports fill the pool ---------------------------
    say("\n  Per-port occupancy (share of the switch's shared buffer), sorted by MAX:")
    say(f"    {'switch':>6} {'port':>16}  {'max':>10} {'% buf':>6}  {'mean':>10}  {'p99':>10}")
    for key in sorted(qd["cnt"], key=lambda k: -qd["mx"][k])[:max(top, 15)]:
        sw, port = key
        m = qd["mx"][key]; av = qd["ssum"][key] / qd["cnt"][key]
        p99 = hpct(qd["hist"][key], qd["cnt"][key], 0.99)
        pctbuf = 100 * m / buffer_bytes
        lbl = port_label(sw, port, portmap, switch_ids)
        flag = "  <== dominates the pool" if pctbuf >= 50 else ""
        say(f"    {sw:>6} {lbl:>16}  {fmt_bytes(m):>10} {pctbuf:5.1f}%  "
            f"{fmt_bytes(av):>10}  {fmt_bytes(p99):>10}{flag}")

    say(f"\n  Switches whose egress queue reached the PFC ceiling : {near_buffer}  "
        f"(>=1 => backpressure-limited; zero => rate-control-limited)")
    say("  NB: the monitor logs a port only when its queue >= 1KB, so 'mean' is over")
    say("      the active phase, not the whole run. MAX remains reliable.")
    peak_over_ceiling = float("nan")
    if ceiling and bn is not None:
        pk = qd["mx"].get((bn.switch, bn.egress_port))
        if pk:
            peak_over_ceiling = pk / ceiling
    return dict(ports=len(qd["cnt"]), switches=len(qd["sw_cnt"]),
                max_pct=max_pct, near_buffer=near_buffer,
                peak_over_ceiling=peak_over_ceiling, ceiling=ceiling)

# ============================================================================ #
#  REGIME VERDICT  (quantitative DCQCN vs PFC classification)
# ============================================================================ #
def classify(fct_s, pfc_s, qlen_s, model=None, bn=None, buffer_bytes=None):
    """DCQCN vs PFC, from measurements rather than from thresholds picked by hand.

    The decisive test is peak_over_ceiling: the egress queue riding at the PFC
    ceiling means backpressure is holding it; below the ceiling, the rate control
    is what limits. Pause time on the bottleneck's own ingress corroborates.

    The previous version summed four boolean signals, two of which -- cv >= 0.7
    and p99 >= 3.0 -- were computed over an unfiltered bimodal mixture of
    congested KV flows (slowdown ~36) and tensor-parallel all-reduces on
    dedicated host links (slowdown ~1). Those two fired at EVERY buffer size,
    which alone met the "pfc_signals >= 2" bar, so the verdict was "PFC" on every
    run of a sweep -- including one with literally zero PFC events. Meanwhile the
    third signal, "pool >= 80% of buffer", could never fire at all, because the
    physical ceiling is ~50% of a 2 MiB buffer. Three of the four inputs were
    broken; the verdict was right once out of five, by luck.

    cv and p99 remain, now over the incast alone, but only as corroboration: they
    cannot by themselves produce a PFC verdict."""
    ceil_ratio = qlen_s.get("peak_over_ceiling", float("nan"))
    pause_frac = pfc_s.get("bottleneck_frac", pfc_s.get("max_frac", 0.0))
    near = qlen_s.get("near_buffer", 0)
    max_buf = qlen_s.get("max_pct", 0.0)
    cv = 0.0; p99 = 1.0
    if fct_s and fct_s.get("bulk"):
        cv = fct_s["bulk"]["cv"]; p99 = fct_s["bulk"]["p99"]
    elif fct_s and fct_s.get("all"):
        cv = fct_s["all"]["cv"]; p99 = fct_s["all"]["p99"]

    at_ceiling = ceil_ratio == ceil_ratio and ceil_ratio >= 0.95
    have_model = ceil_ratio == ceil_ratio

    if not have_model:
        # Without the topology there is no ceiling to compare against, so fall
        # back to pause time alone and say the verdict is provisional.
        if pause_frac >= 5.0:
            verdict, color = "PFC (provisional: no topology, pause-time only)", CORAL
        elif pause_frac > 0.5:
            verdict, color = "MIXED (provisional: no topology, pause-time only)", AMBER
        else:
            verdict, color = "DCQCN (provisional: no topology, pause-time only)", OKGREEN
    elif at_ceiling and pause_frac > 1.0:
        verdict, color = "PFC / backpressure-limited", CORAL
    elif pause_frac > 0.5 or at_ceiling:
        verdict, color = "MIXED - DCQCN holding, PFC starting to bite", AMBER
    else:
        verdict, color = "DCQCN regime (rate-control-limited)", OKGREEN
    return dict(verdict=verdict, color=color, max_buf_pct=max_buf,
                pause_frac=pause_frac, near_buffer=near, cv=cv, p99=p99,
                peak_over_ceiling=ceil_ratio)

# ============================================================================ #
#  PLOTS
# ============================================================================ #
def _finish(fig, path, dpi):
    """Shim over sweeplib.plots.save_fig, kept because the plot functions pass a
    full path rather than (dir, name)."""
    p = Path(path)
    save_fig(fig, p.parent, p.name)
    return os.path.basename(path)

def _empty_panel(ax, msg):
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED,
            fontsize=11, style="italic", transform=ax.transAxes)

def plot_fct(d, bulk_bytes, out, dpi):
    """CDF of slowdown, slowdown-vs-size scatter, per-dst incast, top stragglers."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("FCT — flow slowdown, tail & incast", fontsize=15, fontweight="bold")

    # (a) CDF of slowdown -----------------------------------------------------
    ax = axes[0, 0]
    def cdf(ax, vals, color, label):
        if not vals:
            return
        v = np.sort(np.asarray(vals, float))
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, color=color, lw=2.2, label=f"{label} (n={len(v)})")
        ax.plot([v[-1]], [1.0], "o", color=color, ms=5)
    cdf(ax, d["slow"], COOL, "all flows")
    cdf(ax, d["slow_bulk"], CORAL, f"bulk ≥ {fmt_bytes(bulk_bytes)}")
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    for q in (0.9, 0.99):
        ax.axhline(q, color=GRID, lw=1)
        ax.text(ax.get_xlim()[1], q, f" p{int(q*100)}", va="center",
                ha="left", fontsize=8, color=MUTED)
    ax.set_xlabel("slowdown  (fct / standalone_fct)")
    ax.set_ylabel("cumulative fraction of flows")
    ax.set_title("(a) slowdown CDF — watch the right tail")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right")

    # (b) slowdown vs size ----------------------------------------------------
    ax = axes[0, 1]
    if d["sd_size"]:
        x = np.asarray(d["sd_size"], float); y = np.asarray(d["slow"], float)
        big = x >= bulk_bytes
        ax.scatter(x[~big], y[~big], s=10, alpha=0.35, color=COOL,
                   edgecolors="none", label="small")
        ax.scatter(x[big], y[big], s=16, alpha=0.6, color=CORAL,
                   edgecolors="none", label="bulk")
        ax.set_xscale("log"); ax.axhline(1.0, color=MUTED, ls=":", lw=1)
        ax.axvline(bulk_bytes, color=AMBER, ls="--", lw=1)
        ax.set_xlabel("flow size (bytes, log)")
        ax.set_ylabel("slowdown")
        ax.set_title("(b) slowdown vs flow size")
        ax.legend(loc="upper left")
    else:
        _empty_panel(ax, "no slowdown data")

    # (c) per-destination incast ---------------------------------------------
    ax = axes[1, 0]
    if d["per_dst_bulk"]:
        items = sorted(d["per_dst_bulk"].items(), key=lambda x: -x[1])[:12]
        dsts = [str(k) for k, _ in items]
        cnts = [v for _, v in items]
        byts = [d["per_dst_bytes"][k] / (1024 * 1024) for k, _ in items]
        ypos = np.arange(len(dsts))
        b = ax.barh(ypos, cnts, color=CORAL, alpha=0.85)
        ax.set_yticks(ypos); ax.set_yticklabels([f"node {x}" for x in dsts])
        ax.invert_yaxis()
        ax.set_xlabel("# concurrent incoming bulk flows")
        ax.set_title("(c) incast victims (bulk fan-in per destination)")
        for rect, mb in zip(b, byts):
            ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2,
                    f" {mb:.0f} MB", va="center", ha="left", fontsize=8, color=INK)
    else:
        _empty_panel(ax, "no bulk flows -> no incast fan-in")

    # (d) worst stragglers ----------------------------------------------------
    ax = axes[1, 1]
    if d["worst"]:
        top = sorted(d["worst"], reverse=True)[:10]
        labels = [f"{s}->{t}  ({fmt_bytes(sz)})" for _, sz, s, t, _, _ in top]
        vals = [sd for sd, *_ in top]
        ypos = np.arange(len(vals))
        cmap = _heat_cmap(); vmax = max(vals)
        cols = [cmap(0.3 + 0.7 * (v / vmax)) for v in vals]
        ax.barh(ypos, vals, color=cols)
        ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(1.0, color=MUTED, ls=":", lw=1)
        ax.set_xlabel("slowdown")
        ax.set_title("(d) worst stragglers (gate the decode-start)")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f}x", va="center", ha="left", fontsize=8)
    else:
        _empty_panel(ax, "no flow data")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _finish(fig, out, dpi)

def plot_pfc(pfcd, intervals, totals, run_span, portmap, switch_ids, top, out, dpi):
    """pause-fraction bars + pause timeline (Gantt) + event-rate over time."""
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    fig.suptitle("PFC — backpressure: where, how much, and when",
                 fontsize=15, fontweight="bold")
    span = run_span or (pfcd["tmax"] - pfcd["tmin"]) or 1
    t0 = pfcd["tmin"]

    ranked = [k for k in sorted(totals, key=lambda k: -totals[k]) if totals[k] > 0][:top]

    # (a) top links by pause fraction ----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    if ranked:
        labs, fracs, cols = [], [], []
        for key in ranked:
            node, ntype, ifidx, qidx = key
            kind = "sw" if ntype == 1 else "h"
            lbl = f"{kind}{node}:{port_label(node, ifidx, portmap, switch_ids)}"
            if qidx >= 0:
                lbl += f" q{qidx}"
            labs.append(lbl); fr = 100 * totals[key] / span
            fracs.append(fr)
            cols.append(CORAL if ntype == 1 else VIOLET)
        ypos = np.arange(len(labs))
        ax.barh(ypos, fracs, color=cols)
        ax.set_yticks(ypos); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("% of run spent in PAUSE")
        ax.set_title("(a) top paused links")
        for i, v in enumerate(fracs):
            ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
        ax.legend(handles=[Patch(color=CORAL, label="switch port"),
                           Patch(color=VIOLET, label="host port")],
                  loc="lower right")
    else:
        _empty_panel(ax, "no pauses recorded (DCQCN regime)")

    # (b) event rate over time -----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if pfcd["raw"]:
        times = np.array([r[0] for r in pfcd["raw"]], float)
        pauses = np.array([r[4] == 1 for r in pfcd["raw"]])
        nb = 80
        edges = np.linspace(t0, t0 + span, nb + 1)
        hp, _ = np.histogram(times[pauses], bins=edges)
        hr, _ = np.histogram(times[~pauses], bins=edges)
        ctr = (edges[:-1] + edges[1:]) / 2
        w = (edges[1] - edges[0]) * 0.9
        ax.bar((ctr - t0) / 1e3, hp, width=w / 1e3, color=CORAL, label="PAUSE")
        ax.bar((ctr - t0) / 1e3, -hr, width=w / 1e3, color=COOL, label="RESUME")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("time since first PFC event (us)")
        ax.set_ylabel("events / bin")
        ax.set_title("(b) PFC episodes over time")
        ax.legend(loc="upper right")
    else:
        _empty_panel(ax, "no PFC events")

    # (c) pause timeline / Gantt ---------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    if ranked:
        for row, key in enumerate(ranked):
            node, ntype, ifidx, qidx = key
            col = CORAL if ntype == 1 else VIOLET
            for (a, b) in intervals[key]:
                ax.barh(row, (b - a) / 1e3, left=(a - t0) / 1e3, height=0.7,
                        color=col, alpha=0.85)
            kind = "sw" if ntype == 1 else "h"
            q = f" q{qidx}" if qidx >= 0 else ""
            ax.text(-0.005 * (span / 1e3), row,
                    f"{kind}{node}:{port_label(node, ifidx, portmap, switch_ids)}{q} ",
                    va="center", ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(ranked) - 0.4); ax.invert_yaxis()
        ax.set_xlabel("time since first PFC event (us)")
        ax.set_title("(c) PAUSE intervals per link — vertical alignment = backpressure "
                     "propagating through the fabric")
        ax.grid(axis="y", visible=False)
    else:
        _empty_panel(ax, "no pause intervals to draw")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _finish(fig, out, dpi)

def _downsample(ts, ys, n=3000):
    ts = np.asarray(ts, float); ys = np.asarray(ys, float)
    order = np.argsort(ts); ts, ys = ts[order], ys[order]
    if len(ts) > n:
        idx = np.linspace(0, len(ts) - 1, n).astype(int)
        ts, ys = ts[idx], ys[idx]
    return ts, ys

def plot_qlen(qd, buffer_bytes, portmap, switch_ids, top, out, dpi):
    """The switch buffer is a SHARED pool, so the decisive view is per-switch
    aggregate occupancy vs buffer.  Panels:
      (a) per-switch shared-pool max/p99/mean vs buffer
      (b) per-switch shared-pool occupancy over time
      (c) hottest switch: how its ports compose the shared pool over time
      (d) per-port occupancy heatmap (locate the exact hot port)."""
    fig = plt.figure(figsize=(13.5, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.34, wspace=0.22)
    fig.suptitle("QLEN — shared-buffer occupancy (buffer is per-switch, shared "
                 "across ports)", fontsize=15, fontweight="bold")
    buf_kb = buffer_bytes / 1024
    t0 = qd["tmin"]; span = max(qd["tmax"] - qd["tmin"], 1)

    sw_ranked = sorted(qd["sw_cnt"], key=lambda s: -qd["sw_mx"][s])
    port_ranked = sorted(qd["cnt"], key=lambda k: -qd["mx"][k])[:top]

    # (a) per-switch shared-pool vs buffer -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    if sw_ranked:
        sws = sw_ranked[:max(top, 8)]
        mx = np.array([qd["sw_mx"][s] for s in sws]) / 1024
        p99 = np.array([hpct(qd["sw_hist"][s], qd["sw_cnt"][s], 0.99) for s in sws]) / 1024
        av = np.array([qd["sw_ssum"][s] / qd["sw_cnt"][s] for s in sws]) / 1024
        x = np.arange(len(sws)); w = 0.26
        ax.bar(x - w, mx, w, color=CORAL, label="max")
        ax.bar(x, p99, w, color=AMBER, label="p99")
        ax.bar(x + w, av, w, color=COOL, label="mean (active)")
        ax.axhline(buf_kb, color=INK, ls="--", lw=1.4)
        ax.text(len(sws) - 0.5, buf_kb, " buffer", va="bottom", ha="right",
                fontsize=9, color=INK)
        ax.axhline(0.8 * buf_kb, color=CORAL, ls=":", lw=1)
        ax.text(len(sws) - 0.5, 0.8 * buf_kb, " 80% (PFC risk)", va="bottom",
                ha="right", fontsize=8, color=CORAL)
        ax.set_xticks(x); ax.set_xticklabels([f"sw{s}" for s in sws], fontsize=9)
        ax.set_ylabel("shared-pool occupancy (KB)")
        ax.set_title("(a) per-switch pool vs buffer — bars near the line = overshoot")
        ax.legend(loc="upper right")
    else:
        _empty_panel(ax, "no congested switches (DCQCN regime)")

    # (b) per-switch shared-pool over time -----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if sw_ranked:
        for s in sw_ranked[:8]:
            ts, bs = _downsample(*qd["sw_series"][s])
            ax.plot((ts - t0) / 1e3, bs / 1024, lw=1.3, label=f"sw{s}")
        ax.axhline(buf_kb, color=INK, ls="--", lw=1.2)
        ax.text((span) / 1e3, buf_kb, " buffer", va="bottom", ha="right",
                fontsize=8, color=INK)
        ax.set_xlabel("time since first sample (us)")
        ax.set_ylabel("shared-pool occupancy (KB)")
        ax.set_title("(b) pool fill over time — standing vs transient")
        ax.legend(fontsize=8, ncol=2, loc="upper right")
    else:
        _empty_panel(ax, "no queue samples")

    # (c) composition of the hottest switch's shared pool --------------------
    ax = fig.add_subplot(gs[1, 0])
    if sw_ranked:
        hot = sw_ranked[0]
        ports = [k for k in port_ranked if k[0] == hot]
        if not ports:
            ports = sorted([k for k in qd["cnt"] if k[0] == hot],
                           key=lambda k: -qd["mx"][k])
        ports = ports[:8]
        nb = 200
        edges = np.linspace(t0, t0 + span, nb + 1)
        ctr = (edges[:-1] + edges[1:]) / 2
        stacks = []
        for k in ports:
            ts = np.asarray(qd["series"][k][0], float)
            bs = np.asarray(qd["series"][k][1], float)
            idx = np.clip(np.digitize(ts, edges) - 1, 0, nb - 1)
            binned = np.zeros(nb)
            for b, v in zip(idx, bs):
                if v > binned[b]:
                    binned[b] = v            # max per bin
            stacks.append(binned / 1024)
        palette = [CORAL, AMBER, TEAL, COOL, VIOLET, OKGREEN, "#b07aa1", "#9c755f"]
        labs = [port_label(hot, p, portmap, switch_ids) for (_, p) in ports]
        ax.stackplot((ctr - t0) / 1e3, *stacks, labels=labs,
                     colors=[palette[i % len(palette)] for i in range(len(stacks))],
                     alpha=0.9)
        ax.axhline(buf_kb, color=INK, ls="--", lw=1.2)
        ax.set_xlabel("time since first sample (us)")
        ax.set_ylabel("occupancy (KB)")
        ax.set_title(f"(c) sw{hot}: which ports fill the shared pool")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    else:
        _empty_panel(ax, "no queue samples")

    # (d) per-port occupancy heatmap -----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if port_ranked:
        nb = 160
        edges = np.linspace(t0, t0 + span, nb + 1)
        rows = port_ranked[:min(len(port_ranked), 16)]
        M = np.full((len(rows), nb), np.nan)
        for r, k in enumerate(rows):
            ts = np.asarray(qd["series"][k][0], float)
            bs = np.asarray(qd["series"][k][1], float)
            idx = np.clip(np.digitize(ts, edges) - 1, 0, nb - 1)
            for b, val in zip(idx, bs):
                cur = M[r, b]
                M[r, b] = val if (np.isnan(cur) or val > cur) else cur
        cmap = _heat_cmap(); cmap.set_bad(PANEL)
        im = ax.imshow(M / 1024, aspect="auto", cmap=cmap, origin="upper",
                       extent=[0, span / 1e3, len(rows), 0],
                       norm=Normalize(vmin=0, vmax=buf_kb))
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels([f"sw{sw}:{port_label(sw, p, portmap, switch_ids)}"
                            for (sw, p) in rows], fontsize=7)
        ax.set_xlabel("time since first sample (us)")
        ax.set_title("(d) per-port heatmap — hot rows are the bottleneck ports")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("KB (full scale = buffer)", fontsize=8)
        ax.grid(False)
    else:
        _empty_panel(ax, "no queue samples")

    return _finish(fig, out, dpi)

def plot_overview(regime, fct_s, pfc_s, qlen_s, run_name, out, dpi):
    """One-glance verdict card with the four decisive indicators."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    ax.text(0.0, 1.02, "ns-3 run verdict", fontsize=13, color=MUTED,
            transform=ax.transAxes, va="bottom")
    ax.text(0.0, 0.90, run_name, fontsize=17, fontweight="bold",
            transform=ax.transAxes, va="bottom")

    # verdict banner
    ax.add_patch(plt.Rectangle((0.0, 0.62), 1.0, 0.2, transform=ax.transAxes,
                               color=regime["color"], alpha=0.16, lw=0))
    ax.add_patch(plt.Rectangle((0.0, 0.62), 0.012, 0.2, transform=ax.transAxes,
                               color=regime["color"], lw=0))
    ax.text(0.03, 0.72, regime["verdict"], fontsize=17, fontweight="bold",
            color=regime["color"], transform=ax.transAxes, va="center")

    # indicator tiles
    def tile(x, title, value, sub, ok):
        col = OKGREEN if ok == "good" else (AMBER if ok == "warn" else CORAL)
        ax.add_patch(plt.Rectangle((x, 0.10), 0.22, 0.42, transform=ax.transAxes,
                                   facecolor=PANEL, edgecolor=GRID, lw=1))
        ax.text(x + 0.02, 0.44, title, fontsize=9.5, color=MUTED,
                transform=ax.transAxes, va="center")
        ax.text(x + 0.02, 0.30, value, fontsize=20, fontweight="bold", color=col,
                transform=ax.transAxes, va="center")
        ax.text(x + 0.02, 0.17, sub, fontsize=8.5, color=MUTED,
                transform=ax.transAxes, va="center")

    mb = regime["max_buf_pct"]
    tile(0.02, "peak shared pool / buffer", f"{mb:.0f}%",
         "≥80% → overshoot/PFC",
         "good" if mb < 50 else ("warn" if mb < 80 else "bad"))
    pf = regime["pause_frac"]
    tile(0.26, "worst link PAUSE", f"{pf:.1f}%",
         "0% → no backpressure",
         "good" if pf == 0 else ("warn" if pf < 1 else "bad"))
    cv = regime["cv"]
    tile(0.50, "slowdown CV (bulk)", f"{cv:.2f}",
         "high → unfair tail",
         "good" if cv < 0.3 else ("warn" if cv < 0.7 else "bad"))
    p99 = regime["p99"]
    tile(0.74, "p99 slowdown (bulk)", f"{p99:.1f}x",
         "1.0 → congestion-free",
         "good" if p99 < 1.5 else ("warn" if p99 < 3 else "bad"))

    ax.text(0.0, 0.02,
            "DCQCN keeps queues shallow and pauses near zero; PFC shows up as "
            "queues pinned near the buffer, long PAUSE fractions, and a heavy "
            "slowdown tail (high CV / p99).",
            fontsize=8.5, color=MUTED, transform=ax.transAxes, va="bottom")
    return _finish(fig, out, dpi)

# ============================================================================ #
#  MAIN
# ============================================================================ #
def buffer_mb_from_name(run_name):
    """Per-switch buffer (MiB) from the run name, e.g. 'T1_bx200_dcqcn_buf2' -> 2.0.
    Anchored on 'buf' so 'bx200' is not mistaken for it. config.txt overrides this
    when present: the name is a convention, the config is fact."""
    return BUFFER_AXIS.value(run_name)

def resolve_input(folder):
    if os.path.isdir(folder):
        return os.path.abspath(folder)
    
    cwd_cand = os.path.join(os.getcwd(), "..", "output", "ns3", folder)
    if os.path.isdir(cwd_cand):
        return os.path.abspath(cwd_cand)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_cand = os.path.join(base_dir, "..", "output", "ns3", folder)
    if os.path.isdir(script_cand):
        return os.path.abspath(script_cand)
    return None

def _resolve_aux(explicit, run_name, folder, filename):
    """Locate a per-run auxiliary file, no flag needed. Priority:
      1. the explicit override (a path, a {tag} template, or a directory);
      2. ../configs/astra_sim/ns3/<run_name>/<filename> relative to CWD or script;
      3. original project roots search.
    """
    if explicit:
        hit = find_aux(explicit, run_name, filename, [])
        if hit:
            return str(hit)
            
    # 1. Cerca in ../configs rispetto alla directory corrente del terminale (CWD)
    cwd_cand = os.path.abspath(os.path.join(os.getcwd(), "..", "configs", "astra_sim", "ns3", run_name, filename))
    if os.path.isfile(cwd_cand):
        return cwd_cand
        
    # 2. Cerca in ../configs rispetto alla posizione dello script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_cand = os.path.abspath(os.path.join(base_dir, "..", "configs", "astra_sim", "ns3", run_name, filename))
    if os.path.isfile(script_cand):
        return script_cand

    # 3. Fallback sulla ricerca "root" originale dello script
    roots = project_roots(folder, os.getcwd(), base_dir)
    hit = find_under_roots(roots, os.path.join("configs", "astra_sim", "ns3",
                                               run_name, filename))
    if hit:
        return str(hit)
        
    return None

def resolve_topology(explicit, run_name, folder):
    return _resolve_aux(explicit, run_name, folder, "physical_topology.txt")

def resolve_config(explicit, run_name, folder):
    return _resolve_aux(explicit, run_name, folder, "config.txt")

def main():
    ap = argparse.ArgumentParser(
        description="Analyze & visualize ns-3 output (fct/pfc/qlen) for one run.")
    ap.add_argument("folder",
                    help="run subdir under ../output/ns3/, or a direct path to the run folder")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: ../results/ns3_graphs/<run basename>)")
    ap.add_argument("--buffer-mb", type=float, default=None,
                    help="per-switch BUFFER_SIZE in MiB. If omitted, taken from the "
                         "run name (the number after 'buf', e.g. ...buf2 -> 2 MiB); "
                         "falls back to 16 if the name has no 'buf' token.")
    ap.add_argument("--bulk-mb", type=float, default=1.0,
                    help="threshold in MB for a 'bulk' flow (KV/PP) (default 1)")
    ap.add_argument("--topology", default=None,
                    help="physical_topology.txt. Drives the port labels, the PFC "
                         "threshold and ceiling, and the fabric/direct flow filter. "
                         "Without it the regime verdict is provisional and the "
                         "slowdown statistics are a mixture. If omitted, "
                         "auto-detected at configs/astra_sim/ns3/<run>/physical_topology.txt")
    ap.add_argument("--config", default=None,
                    help="ns-3 config.txt for this run (KMIN/KMAX + authoritative "
                         "BUFFER_SIZE). Default: <run folder>/config.txt")
    ap.add_argument("--top", type=int, default=10,
                    help="how many worst links/ports to show (default 10)")
    ap.add_argument("--dpi", type=int, default=130, help="figure DPI (default 130)")
    ap.add_argument("--no-plots", action="store_true", help="text report only")
    args = ap.parse_args()

    folder = resolve_input(args.folder)
    if folder is None:
        print(f"Error: could not find run '{args.folder}' "
              f"(neither a folder nor output/ns3/{args.folder}).")
        sys.exit(1)

    run_name = os.path.basename(folder.rstrip("/"))
    # Salva in ../results/ns3_graphs/
    out_dir = args.out_dir or os.path.abspath(os.path.join(os.getcwd(), "..", "results", "ns3_graphs", run_name))
    os.makedirs(out_dir, exist_ok=True)

    # buffer size: explicit flag wins; else parse the 'buf<N>' token in the name;
    # else fall back to 16 MiB.
    if args.buffer_mb is not None:
        buffer_mb, buf_src = args.buffer_mb, "--buffer-mb"
    else:
        parsed = buffer_mb_from_name(run_name)
        if parsed is not None:
            buffer_mb, buf_src = parsed, "run name ('buf' token)"
        else:
            buffer_mb, buf_src = 16.0, "default (no 'buf' in name)"
    buffer_bytes = int(buffer_mb * 1024 * 1024)
    bulk_bytes = int(args.bulk_mb * 1024 * 1024)

    topo, portmap, switch_ids, model = None, {}, set(), None
    topo_path = resolve_topology(args.topology, run_name, folder)
    if topo_path:
        topo, switch_ids, portmap = load_topology(topo_path)
    else:
        print("[warn] no physical_topology.txt resolved: no PFC threshold, no "
              "ceiling, no fabric/direct filter. The slowdown statistics will be "
              "a MIXTURE of congested transfers and uncongested direct-link "
              "collectives, and the regime verdict is provisional. Pass --topology.",
              file=sys.stderr)
    cfg_path = resolve_config(args.config, run_name, folder)
    if cfg_path and topo is not None:
        cfg = parse_ns3_config(Path(cfg_path))
        for w in cfg.warnings():
            print(f"[warn] {w}", file=sys.stderr)
        if cfg.buffer_mb is not None and abs(cfg.buffer_mb - buffer_mb) > 1e-6:
            print(f"[warn] BUFFER_SIZE={cfg.buffer_mb} in config.txt but "
                  f"'buf{buffer_mb:g}' in the run name -- trusting config.txt",
                  file=sys.stderr)
            buffer_mb = cfg.buffer_mb
            buffer_bytes = int(buffer_mb * 1024 * 1024)
        model = FabricModel(topo, cfg)
    elif topo is not None:
        print("[warn] no config.txt resolved: KMIN/KMAX unknown, no ECN band.",
              file=sys.stderr)

    make_plots = HAVE_MPL and not args.no_plots
    if not HAVE_MPL and not args.no_plots:
        print("[note] matplotlib/numpy unavailable -> text report only "
              f"({_MPL_ERR}).")
    if make_plots:
        _apply_theme()

    say = Report()
    say(f"\nRun: {folder}")
    say(f"Buffer: {buffer_mb:g} MiB per switch  (source: {buf_src})")
    say(f"Output: {out_dir}")
    if topo_path:
        say(f"Topology: {topo_path}  ({len(switch_ids)} switches)")
    else:
        say("Topology: not found — ports shown by index "
            "(pass --topology or place it at configs/astra_sim/ns3/<run>/physical_topology.txt)")

    # ---- FCT ----------------------------------------------------------------
    say.header("FCT  (fct.txt) — per-flow slowdown, tail, incast")
    fct_path = os.path.join(folder, "fct.txt")
    fct_d, fct_s = None, None
    if os.path.isfile(fct_path):
        fct_d = parse_fct(fct_path, bulk_bytes, topo)
        fct_s = report_fct(say, fct_d, bulk_bytes)
    else:
        say("  file not found")
    run_end = fct_d["run_end"] if fct_d else 0

    # ---- QLEN (parsed first: it identifies the bottleneck, which both the PFC
    #      attribution and the regime test need) -----------------------------
    qlen_path = os.path.join(folder, "qlen.txt")
    qlen_d = qlen_s = None
    if os.path.isfile(qlen_path):
        qlen_d = parse_qlen(qlen_path)

    # The congested directed link, plus the ingress PORTS feeding it. Ground truth
    # is qlen.txt (the port that actually built a queue); the ingress ports come
    # from the flows' paths. PFC accounting is per (port, qIndex), so it is a port
    # count -- dozens of flows entering through two host ports load two counters.
    bn = None
    if topo is not None:
        bn = flowlib.find_bottleneck(topo, qlen_d["mx"] if qlen_d else None,
                                     fct_d["_flows"] if fct_d is not None else None)

    # ---- PFC ----------------------------------------------------------------
    say.header("PFC  (pfc.txt) — pause-time per link")
    pfc_path = os.path.join(folder, "pfc.txt")
    pfc_d = pfc_s = None; intervals = totals = None
    if os.path.isfile(pfc_path):
        pfc_d = parse_pfc(pfc_path)
        clamp = run_end or pfc_d["tmax"]
        intervals, totals = pfc_intervals(pfc_d, clamp)
        pfc_s = report_pfc(say, pfc_d, totals, run_end, portmap, switch_ids, topo, bn)
    else:
        say("  file not found")

    # ---- QLEN ---------------------------------------------------------------
    say.header("QLEN (qlen.txt) — egress occupancy vs the PFC ceiling and the ECN band")
    if qlen_d is not None:
        qlen_s = report_qlen(say, qlen_d, buffer_bytes, portmap, switch_ids, args.top,
                             model, bn)
    else:
        say("  file not found")

    # ---- verdict ------------------------------------------------------------
    regime = classify(fct_s, pfc_s or {}, qlen_s or {}, model, bn, buffer_bytes)
    say.header("VERDICT")
    say(f"  {regime['verdict']}")
    pc = regime.get("peak_over_ceiling", float("nan"))
    say(f"    peak egress / PFC ceiling : {pc:.3f}   "
        f"({'held by backpressure' if pc == pc and pc >= 0.95 else 'rate control limiting' if pc == pc else 'no topology/config'})")
    say(f"    pause on the bottleneck   : {regime['pause_frac']:.2f}% of run")
    say(f"    peak egress / buffer      : {regime['max_buf_pct']:.1f}%   "
        f"(context only: the ceiling, not the buffer, is where the queue stops)")
    say(f"    slowdown CV (incast)      : {regime['cv']:.3f}   "
        f"{'' if (fct_s or {}).get('filtered', True) else '<< UNFILTERED: a fabric/direct mixture'}")
    say(f"    p99 slowdown (incast)     : {regime['p99']:.2f}x")

    say("\n" + "-" * 74)
    say("Reading: queues near buffer + high pause-time + high-CV slowdown => PFC.")
    say("         low queues + pause ~0 + tight slowdown (low CV)          => DCQCN.")
    say("-" * 74)

    # ---- write report + machine-readable summary ----------------------------
    say.save(os.path.join(out_dir, "report.txt"))
    summary = dict(run=run_name, folder=folder,
                   buffer_mb=buffer_mb, buffer_source=buf_src, bulk_mb=args.bulk_mb,
                   fct=fct_s, pfc=pfc_s, qlen=qlen_s, regime=regime)
    with open(os.path.join(out_dir, "summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2, default=lambda o: None)

    # ---- figures ------------------------------------------------------------
    made = []
    if make_plots:
        try:
            made.append(plot_overview(regime, fct_s, pfc_s, qlen_s, run_name,
                                      os.path.join(out_dir, "00_overview.png"), args.dpi))
        except Exception as e:
            print(f"[warn] overview plot failed: {e}")
        if fct_d and fct_d["n"]:
            try:
                made.append(plot_fct(fct_d, bulk_bytes,
                                     os.path.join(out_dir, "01_fct.png"), args.dpi))
            except Exception as e:
                traceback.print_exc()
                print(f"[warn] fct plot failed: {e}")
        if pfc_d and pfc_d["n"]:
            try:
                made.append(plot_pfc(pfc_d, intervals, totals, run_end, portmap,
                                     switch_ids, args.top,
                                     os.path.join(out_dir, "02_pfc.png"), args.dpi))
            except Exception as e:
                traceback.print_exc()
                print(f"[warn] pfc plot failed: {e}")
        if qlen_d and qlen_d["cnt"]:
            try:
                made.append(plot_qlen(qlen_d, buffer_bytes, portmap, switch_ids,
                                      args.top,
                                      os.path.join(out_dir, "03_qlen.png"), args.dpi))
            except Exception as e:
                traceback.print_exc()
                print(f"[warn] qlen plot failed: {e}")

    print(f"\nWrote report.txt + summary.json to {out_dir}")
    if made:
        print("Figures:")
        for m in made:
            print("  " + os.path.basename(m))
    elif not make_plots:
        print("(figures skipped)")
    print()

if __name__ == "__main__":
    main()