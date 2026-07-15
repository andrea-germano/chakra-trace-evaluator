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
    * a subdirectory name looked up under  ./output/ns3/<run>  (relative to this
      script, as in the original tool), or
    * a direct path to a folder that contains fct.txt / pfc.txt / qlen.txt.

Options:
    --out-dir DIR     where to write the report + figures.
                      default: ./ns3_graphs/<run basename>  (in the CWD)
    --buffer-mb F     per-switch buffer in MiB (default 16)
    --bulk-mb F       flow-size threshold (MB) to count as "bulk" KV/PP (default 1)
    --topology FILE   topology file, to label switch ports with their neighbour
    --top N           how many worst links/ports to show in tables & plots (10)
    --dpi N           figure DPI (default 130)
    --no-plots        text report only (skip figures)
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

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
    """node id from an fct.txt address token (hex '0b000005' or dotted '11.0.0.5').
    Mapping (common.h:139): node = (ip >> 8) & 0xffff."""
    tok = tok.strip()
    try:
        if "." in tok:
            parts = [int(x) for x in tok.split(".")]
            ip = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]
        else:
            ip = int(tok, 16)
    except (ValueError, IndexError):
        return -1
    return (ip >> 8) & 0xffff

# ============================================================================ #
#  Topology  (optional) : switch port -> neighbour node id
# ============================================================================ #
def parse_topology(path):
    """Return (switch_ids:set, portmap:{(node,ifindex):neighbor}, host_ids:set).
    ifIndex counting starts at 1 (0 = loopback) and follows link-file order."""
    switch_ids, portmap = set(), {}
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return switch_ids, portmap, set()
    it = iter(toks)
    try:
        node_num = int(next(it)); switch_num = int(next(it)); link_num = int(next(it))
        for _ in range(switch_num):
            switch_ids.add(int(next(it)))
        nxt = defaultdict(lambda: 1)      # per node: next ifIndex to assign
        for _ in range(link_num):
            src = int(next(it)); dst = int(next(it))
            next(it); next(it); next(it)  # rate, delay, error
            portmap[(src, nxt[src])] = dst; nxt[src] += 1
            portmap[(dst, nxt[dst])] = src; nxt[dst] += 1
    except (StopIteration, ValueError):
        pass
    all_nodes = {n for (n, _) in portmap}
    host_ids = all_nodes - switch_ids
    return switch_ids, portmap, host_ids

def port_label(node, ifidx, portmap, switch_ids):
    nb = portmap.get((node, ifidx))
    if nb is None:
        return f"p{ifidx}"
    kind = "sw" if nb in switch_ids else "h"
    return f"p{ifidx}->{kind}{nb}"

# ============================================================================ #
#  Parsers  — return plain dicts reused by both the text report and the plots
# ============================================================================ #
def parse_fct(path, bulk_bytes):
    d = dict(n=0, sizes=[], starts=[], fcts=[], sfcts=[], slow=[], slow_bulk=[],
             sd_size=[], sd_start=[], worst=[], run_end=0,
             per_dst_bulk=defaultdict(int), per_dst_bytes=defaultdict(int),
             per_dst_slow=defaultdict(list),
             # --- sanity / self-diagnosis ---
             raw_lines=0, skipped_short=0, skipped_badnum=0, sfct_nonpos=0,
             slow_lt1=0, ncols=None, sample=None, ncol_hist=defaultdict(int))
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d["raw_lines"] += 1
            p = line.split()
            d["ncol_hist"][len(p)] += 1
            if d["sample"] is None:
                d["sample"] = line.rstrip("\n"); d["ncols"] = len(p)
            if len(p) < 8:
                d["skipped_short"] += 1
                continue
            try:
                size = int(p[4]); start = int(p[5]); fct = int(p[6]); sfct = int(p[7])
            except ValueError:
                d["skipped_badnum"] += 1
                continue
            d["n"] += 1
            d["sizes"].append(size); d["starts"].append(start); d["fcts"].append(fct)
            d["sfcts"].append(sfct)
            d["run_end"] = max(d["run_end"], start + fct)
            src = ip_to_node(p[0]); dst = ip_to_node(p[1])
            if sfct > 0:
                sd = fct / sfct
                if sd < 0.999:
                    d["slow_lt1"] += 1
                d["slow"].append(sd); d["sd_size"].append(size); d["sd_start"].append(start)
                d["worst"].append((sd, size, src, dst, fct, start))
                d["per_dst_slow"][dst].append(sd)
                if size >= bulk_bytes:
                    d["slow_bulk"].append(sd)
                    d["per_dst_bulk"][dst] += 1
                    d["per_dst_bytes"][dst] += size
            else:
                d["sfct_nonpos"] += 1
    return d

def parse_pfc(path):
    """events[(node,ntype,ifidx)] = sorted [(time,type)]; also raw stream for timeline."""
    ev = defaultdict(list)
    raw = []                 # (time, node, ntype, ifidx, type)
    tmin = None; tmax = 0; n = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            try:
                t = int(p[0]); node = int(p[1]); ntype = int(p[2])
                ifidx = int(p[3]); typ = int(p[4])
            except ValueError:
                continue
            n += 1
            ev[(node, ntype, ifidx)].append((t, typ))
            raw.append((t, node, ntype, ifidx, typ))
            tmin = t if tmin is None else min(tmin, t)
            tmax = max(tmax, t)
    return dict(ev=ev, raw=raw, n=n, tmin=tmin or 0, tmax=tmax)

def pfc_intervals(ev, clamp_end):
    """Turn PAUSE/RESUME events into closed [start,end] pause intervals per link.
    Unclosed pauses (still paused at capture end) are clamped to clamp_end."""
    intervals = {}
    totals = {}
    for key, events in ev.items():
        events.sort()
        iv = []; start = None
        for t, typ in events:
            if typ == 1 and start is None:      # PAUSE
                start = t
            elif typ == 0 and start is not None:  # RESUME
                iv.append((start, t)); start = None
        if start is not None:                    # still paused at end
            iv.append((start, max(clamp_end, start)))
        intervals[key] = iv
        totals[key] = sum(b - a for a, b in iv)
    return intervals, totals

def parse_qlen(path):
    """Full parse into per-(switch,port) time series (for plots) plus histograms
    (robust percentiles on huge files).

    The switch buffer is a single SHARED pool (switch-mmu.cc: shared_used_bytes is
    switch-wide, and GetPfcThreshold subtracts it from buffer_size).  Each qlen.txt
    line is one switch at one timestamp listing all its congested ports, so the sum
    over that line is the switch's shared-pool occupancy at that sample.  We track
    that aggregate per switch as well as the per-port detail."""
    series = defaultdict(lambda: ([], []))         # (sw,port) -> (times[], bytes[])
    hist = defaultdict(lambda: defaultdict(int))   # (sw,port) -> {kb: count}
    mx = defaultdict(int); cnt = defaultdict(int); ssum = defaultdict(int)
    # per-switch shared-pool aggregate (sum over ports per sample):
    sw_series = defaultdict(lambda: ([], []))      # sw -> (times[], total_bytes[])
    sw_hist = defaultdict(lambda: defaultdict(int))
    sw_mx = defaultdict(int); sw_cnt = defaultdict(int); sw_ssum = defaultdict(int)
    tmin = None; tmax = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3 or p[0] != "time":
                continue
            try:
                t = int(p[1]); sw = int(p[2])
            except ValueError:
                continue
            tmin = t if tmin is None else min(tmin, t); tmax = max(tmax, t)
            i = 3; line_total = 0
            while i + 2 <= len(p):
                if p[i] == "j":
                    try:
                        port = int(p[i + 1]); b = int(p[i + 2])
                    except ValueError:
                        i += 3; continue
                    key = (sw, port)
                    ts, bs = series[key]; ts.append(t); bs.append(b)
                    hist[key][b // 1000] += 1
                    cnt[key] += 1; ssum[key] += b
                    if b > mx[key]:
                        mx[key] = b
                    line_total += b
                    i += 3
                else:
                    i += 1
            # record the switch-wide shared-pool sample
            sts, sbs = sw_series[sw]; sts.append(t); sbs.append(line_total)
            sw_hist[sw][line_total // 1000] += 1
            sw_cnt[sw] += 1; sw_ssum[sw] += line_total
            if line_total > sw_mx[sw]:
                sw_mx[sw] = line_total
    return dict(series=series, hist=hist, mx=mx, cnt=cnt, ssum=ssum,
                sw_series=sw_series, sw_hist=sw_hist, sw_mx=sw_mx,
                sw_cnt=sw_cnt, sw_ssum=sw_ssum,
                tmin=tmin or 0, tmax=tmax)

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

def report_pfc(say, pfcd, totals, run_span, portmap, switch_ids):
    if pfcd["n"] == 0:
        say("  (no PFC events — no pause: good sign, DCQCN regime)")
        return dict(events=0, paused_links=0, tot_pause=0, max_frac=0.0)
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
    for (node, ntype, ifidx), tp in sorted(totals.items(), key=lambda x: -x[1])[:10]:
        if tp == 0:
            continue
        kind = "switch" if ntype == 1 else "host"
        lbl = port_label(node, ifidx, portmap, switch_ids)
        frac = 100 * tp / span; max_frac = max(max_frac, frac)
        say(f"    {node:>5} {kind:>6} {lbl:>16}  {fmt_ns(tp):>12}  {frac:6.2f}%")
    return dict(events=pfcd["n"], paused_links=paused_links,
                tot_pause=tot_pause, max_frac=max_frac, span=span)

def report_qlen(say, qd, buffer_bytes, portmap, switch_ids, top):
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

    # --- primary view: switch-wide shared-pool occupancy (drives PFC) --------
    say("\n  Shared-pool occupancy per SWITCH (sum of all ports = what PFC watches):")
    say(f"    {'switch':>6}  {'max':>10} {'% buf':>6}  {'mean':>10}  {'p99':>10}")
    near_buffer = 0; max_pct = 0.0
    for sw in sorted(qd["sw_cnt"], key=lambda s: -qd["sw_mx"][s]):
        m = qd["sw_mx"][sw]; av = qd["sw_ssum"][sw] / qd["sw_cnt"][sw]
        p99 = hpct(qd["sw_hist"][sw], qd["sw_cnt"][sw], 0.99)
        pctbuf = 100 * m / buffer_bytes; max_pct = max(max_pct, pctbuf)
        flag = "  <== pool near buffer (PFC/overshoot)" if pctbuf >= 80 else ""
        if pctbuf >= 80:
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

    say(f"\n  Switches whose shared pool hit >= 80% of buffer : {near_buffer}  "
        f"(>=1 => PFC/overshoot regime; zero => DCQCN regime)")
    say("  NB: the monitor logs a port only when its queue >= 1KB, so 'mean' is over")
    say("      the active phase, not the whole run. MAX remains reliable.")
    return dict(ports=len(qd["cnt"]), switches=len(qd["sw_cnt"]),
                max_pct=max_pct, near_buffer=near_buffer)

# ============================================================================ #
#  REGIME VERDICT  (quantitative DCQCN vs PFC classification)
# ============================================================================ #
def classify(fct_s, pfc_s, qlen_s):
    max_buf = qlen_s.get("max_pct", 0.0)
    pause_frac = pfc_s.get("max_frac", 0.0)
    near = qlen_s.get("near_buffer", 0)
    cv = 0.0; p99 = 1.0
    if fct_s and fct_s.get("bulk"):
        cv = fct_s["bulk"]["cv"]; p99 = fct_s["bulk"]["p99"]
    elif fct_s and fct_s.get("all"):
        cv = fct_s["all"]["cv"]; p99 = fct_s["all"]["p99"]

    pfc_signals = 0
    pfc_signals += near > 0 or max_buf >= 80
    pfc_signals += pause_frac >= 1.0
    pfc_signals += cv >= 0.7
    pfc_signals += p99 >= 3.0

    if pfc_signals >= 2 or max_buf >= 90 or pause_frac >= 5:
        verdict, color = "PFC / buffer-overshoot regime", CORAL
    elif pfc_signals == 1 or max_buf >= 50 or pause_frac > 0:
        verdict, color = "MIXED — DCQCN holding, PFC starting to bite", AMBER
    else:
        verdict, color = "DCQCN regime (healthy)", OKGREEN
    return dict(verdict=verdict, color=color, max_buf_pct=max_buf,
                pause_frac=pause_frac, near_buffer=near, cv=cv, p99=p99)

# ============================================================================ #
#  PLOTS
# ============================================================================ #
def _finish(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

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
        for (node, ntype, ifidx) in ranked:
            kind = "sw" if ntype == 1 else "h"
            lbl = f"{kind}{node}:{port_label(node, ifidx, portmap, switch_ids)}"
            labs.append(lbl); fr = 100 * totals[(node, ntype, ifidx)] / span
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
            node, ntype, ifidx = key
            col = CORAL if ntype == 1 else VIOLET
            for (a, b) in intervals[key]:
                ax.barh(row, (b - a) / 1e3, left=(a - t0) / 1e3, height=0.7,
                        color=col, alpha=0.85)
            kind = "sw" if ntype == 1 else "h"
            ax.text(-0.005 * (span / 1e3), row,
                    f"{kind}{node}:{port_label(node, ifidx, portmap, switch_ids)} ",
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
    """Extract the per-switch buffer (MiB) from the run name, e.g.
    'T1_bx200_dcqcn_buf2' -> 2.0.  The token always follows 'buf' but is not
    necessarily at the end; 'bx200' etc. are ignored because we anchor on 'buf'
    immediately followed by digits.  Returns None if absent."""
    m = re.search(r"buf(\d+(?:\.\d+)?)", run_name, re.IGNORECASE)
    return float(m.group(1)) if m else None

def resolve_input(folder):
    if os.path.isdir(folder):
        return os.path.abspath(folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(base_dir, "output", "ns3", folder)
    if os.path.isdir(cand):
        return os.path.abspath(cand)
    return None

def resolve_topology(explicit, run_name, folder):
    """Locate the physical topology file automatically (no flag needed).  Priority:
      1. --topology (explicit override);
      2. configs/astra_sim/ns3/<run_name>/physical_topology.txt under any ancestor
         of the run folder — the run lives at <root>/output/ns3/<run> and configs
         at <root>/configs/..., so they share a project root; we also try the CWD
         and this script's directory as fallback roots;
      3. physical_topology.txt sitting inside the run folder itself."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    rel = os.path.join("configs", "astra_sim", "ns3", run_name, "physical_topology.txt")

    def ancestors(start, depth=8):
        d = os.path.abspath(start)
        for _ in range(depth):
            yield d
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd

    roots, seen = [], set()
    # run folder first (most reliable), then CWD, then the script's directory
    for base in (folder, os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        for d in ancestors(base):
            if d not in seen:
                seen.add(d); roots.append(d)

    for root in roots:
        cand = os.path.join(root, rel)
        if os.path.isfile(cand):
            return cand
    inside = os.path.join(folder, "physical_topology.txt")
    if os.path.isfile(inside):
        return inside
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Analyze & visualize ns-3 output (fct/pfc/qlen) for one run.")
    ap.add_argument("folder",
                    help="run subdir under output/ns3/, or a direct path to the run folder")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: ./results/ns3/<run basename>)")
    ap.add_argument("--buffer-mb", type=float, default=None,
                    help="per-switch BUFFER_SIZE in MiB. If omitted, taken from the "
                         "run name (the number after 'buf', e.g. ...buf2 -> 2 MiB); "
                         "falls back to 16 if the name has no 'buf' token.")
    ap.add_argument("--bulk-mb", type=float, default=1.0,
                    help="threshold in MB for a 'bulk' flow (KV/PP) (default 1)")
    ap.add_argument("--topology", default=None,
                    help="topology file for port->neighbour labels. If omitted, "
                         "auto-detected at configs/astra_sim/ns3/<run>/physical_topology.txt")
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
    out_dir = args.out_dir or os.path.join(os.getcwd(), "results", "ns3_graphs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # buffer size: explicit flag wins; else parse the 'buf<N>' token in the name;
    # else fall back to 16 MiB.
    if args.buffer_mb is not None:
        buffer_mb, buf_src = args.buffer_mb, "--buffer-mb"
    else:
        parsed = buffer_mb_from_name(run_name)
        if parsed is not None:
            buffer_mb, buf_src = parsed, f"run name ('buf' token)"
        else:
            buffer_mb, buf_src = 16.0, "default (no 'buf' in name)"
    buffer_bytes = int(buffer_mb * 1024 * 1024)
    bulk_bytes = int(args.bulk_mb * 1024 * 1024)

    portmap, switch_ids = {}, set()
    topo_path = resolve_topology(args.topology, run_name, folder)
    if topo_path:
        switch_ids, portmap, _ = parse_topology(topo_path)

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
        fct_d = parse_fct(fct_path, bulk_bytes)
        fct_s = report_fct(say, fct_d, bulk_bytes)
    else:
        say("  file not found")
    run_end = fct_d["run_end"] if fct_d else 0

    # ---- PFC ----------------------------------------------------------------
    say.header("PFC  (pfc.txt) — pause-time per link")
    pfc_path = os.path.join(folder, "pfc.txt")
    pfc_d = pfc_s = None; intervals = totals = None
    if os.path.isfile(pfc_path):
        pfc_d = parse_pfc(pfc_path)
        clamp = run_end or pfc_d["tmax"]
        intervals, totals = pfc_intervals(pfc_d["ev"], clamp)
        pfc_s = report_pfc(say, pfc_d, totals, run_end, portmap, switch_ids)
    else:
        say("  file not found")

    # ---- QLEN ---------------------------------------------------------------
    say.header("QLEN (qlen.txt) — queue occupancy vs buffer (DCQCN vs PFC regime)")
    qlen_path = os.path.join(folder, "qlen.txt")
    qlen_d = qlen_s = None
    if os.path.isfile(qlen_path):
        qlen_d = parse_qlen(qlen_path)
        qlen_s = report_qlen(say, qlen_d, buffer_bytes, portmap, switch_ids, args.top)
    else:
        say("  file not found")

    # ---- verdict ------------------------------------------------------------
    regime = classify(fct_s, pfc_s or {}, qlen_s or {})
    say.header("VERDICT")
    say(f"  {regime['verdict']}")
    say(f"    peak queue / buffer : {regime['max_buf_pct']:.1f}%")
    say(f"    worst link pause    : {regime['pause_frac']:.2f}% of run")
    say(f"    slowdown CV (bulk)  : {regime['cv']:.3f}")
    say(f"    p99 slowdown (bulk) : {regime['p99']:.2f}x")

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
                print(f"[warn] fct plot failed: {e}")
        if pfc_d and pfc_d["n"]:
            try:
                made.append(plot_pfc(pfc_d, intervals, totals, run_end, portmap,
                                     switch_ids, args.top,
                                     os.path.join(out_dir, "02_pfc.png"), args.dpi))
            except Exception as e:
                print(f"[warn] pfc plot failed: {e}")
        if qlen_d and qlen_d["cnt"]:
            try:
                made.append(plot_qlen(qlen_d, buffer_bytes, portmap, switch_ids,
                                      args.top,
                                      os.path.join(out_dir, "03_qlen.png"), args.dpi))
            except Exception as e:
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