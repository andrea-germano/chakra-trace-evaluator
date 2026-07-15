#!/usr/bin/env python3
"""
buffer_analyzer.py  (v2)
========================

Summarise an ASTRA-sim + ns-3 **buffer-size sweep** for MLSynth disaggregated-
inference traces.  Buffer-sweep sibling of ``bandwidth_analyzer.py``: same
discovery / tag / summary.csv / per_node.csv / PNG structure, different question.

Why the buffer sweep needs its own analyzer
--------------------------------------------------------------------------------
The bandwidth sweep lives entirely in the ASTRA-sim ``stats_sys*.csv`` (logical
ticks).  The buffer sweep does not: at steady state the congested downlink drains
at line rate regardless of buffer, so mean KV completion is nearly flat.  The real
story is the **congestion-control regime** (DCQCN vs PFC), which lives in the ns-3
outputs:

    pfc.txt   -> which devices are being PAUSEd, and for how long   (PFC signature)
    qlen.txt  -> per-egress-port queue build-up vs KMIN/KMAX        (ECN axis)
    fct.txt   -> per-flow slowdown: mean vs p99/CV                  (the trade-off)

What changed vs v1 (all of it verified against the qos-impl sources)
--------------------------------------------------------------------------------
v1 made five modelling assumptions that do not hold for this fork.  This version
derives them from ``physical_topology.txt`` + ``config.txt`` instead:

1. **PFC threshold.**  ``switch-mmu.cc``:

       GetPfcThreshold(port) = (buffer_size - total_hdrm - total_rsrv
                                - shared_used_bytes) >> pfc_a_shift[port]

   ``buffer/8`` ignores ``total_hdrm + total_rsrv``.  On a leaf with two 1024 Gbps
   host ports (headroom = rate*delay/8e9*3 = 384 kB each) plus a 100 Gbps uplink,
   that is ~968 kB -- i.e. **46 % of a 2 MiB buffer**, an 86 % error on the
   threshold exactly where the regime flips.  ``pfc_a_shift`` is also per-port and
   rate-dependent (``common.h``: decremented while ``rate > nic_rate``, where
   ``nic_rate`` is the rate of *device 1 of the lowest-id host*, i.e. of the first
   link mentioning that host in the topology file).  Both are computed here.

2. **qlen.txt is egress, not shared pool.**  ``monitor_buffer`` dumps
   ``sum_k egress_bytes[port][k]`` **per port** (only ports with >= 1000 B are
   emitted).  The shared pool is ingress-side accounting (``shared_used_bytes``)
   and is **not observable** from qlen.txt at all.  Summing over ports destroys
   the one quantity that is directly comparable to KMIN/KMAX, since
   ``ShouldSendCN`` tests ``egress_bytes[ifindex][qIndex] > kmax[ifindex]``.
   We therefore track **per-(switch, port)** peaks and compare the congested port
   against its own KMIN/KMAX (keyed by that port's link rate, as ns-3 does).

3. **Ingress/egress fan-in.**  PFC watches per-*ingress*-port occupancy, ECN
   watches per-*egress*-port occupancy.  On an F-to-1 constriction the same bytes
   sit on F ingress counters but on 1 egress counter, so PFC fires at an
   egress-equivalent of ``F * threshold``.  The regime flip is therefore at
   ``F*threshold == KMAX``, not ``threshold == KMAX``.  F is measured (max
   concurrency of bulk flows on the congested directed link), not assumed.

4. **pfc.txt has no qIndex column.**  ``m_tracePfc`` fires per qIndex, but
   ``get_pfc`` prints only (time, node, node_type, ifindex, type).  With
   ``qos-enabled`` (KV and TP collectives on different priority groups) the
   PAUSE/RESUME sequences of different queues interleave on the same ifindex and a
   state machine keyed on (node, ifindex) silently mis-pairs them.  This parser
   reads an **optional 6th column** with the qIndex; see ``PFC_QIDX_PATCH`` at the
   bottom of this file for the three-line ns-3 diff that emits it.  Without the
   patch the parser still runs but reports ``pfc_qidx=MISSING`` and the
   pause totals must be treated as a lower/upper bound, not a measurement.

   Also: the trace fires on the device that **receives** the PAUSE frame, i.e. on
   the *victim*.  ``pfc_sw_pause_*`` = time switch egress ports were held by
   downstream backpressure; ``pfc_host_pause_*`` = hosts held by their own leaf
   (the deepest point of the backpressure tree).  v1's docstring had this backwards.

5. **Slowdown floor.**  ``standalone_fct = base_rtt + total_bytes*8e9/pairBw``
   with ``pairBw`` = min link rate along the BFS path (``entry.h``,
   ``common.h::CalculateRoute``).  It assumes the flow owns the bottleneck, so an
   F-to-1 incast has a slowdown **floor of F even on a perfect fabric**.  Plots
   reference F, not 1.  Flows on the direct host-host links (1 hop, never
   congested) are also excluded from the incast statistics: they have slowdown ~1
   by construction and only dilute the mean and the CV.

Other fixes: pause clamped to the run end (not to the last PFC event); pause
normalised by the **incast window** rather than the whole run (the ns-3 clock also
advances during prefill compute, which diluted v1's percentages); no silent
fallback from bulk flows to all flows; qlen token scan no longer IndexErrors on a
truncated final line.

Note: ``QLEN_MON_END`` is dead code in ``common.h`` (parsed, never used --
``monitor_buffer`` reschedules unconditionally), so qlen.txt always covers the run
from ``QLEN_MON_START`` at a fixed 100 ns interval.

Sweep layout (unchanged)
--------------------------------------------------------------------------------
    <astra_root>/buffer_sweep/T2_bx1_dcqcn_buf2/  stats_sys*.csv ...
    <ns3_root>/buffer_sweep/T2_bx1_dcqcn_buf2/    fct.txt pfc.txt qlen.txt ...

Matching is by run-directory basename; the ``buf<N>`` token is the swept axis and
everything else is kept as a ``variant`` key.

Usage
-----
    python3 buffer_analyzer.py [ASTRA_ROOT] [--ns3-root NS3_ROOT] [-o OUTDIR]
        [--topology PATH|TEMPLATE] [--config PATH|TEMPLATE]
        [--pattern '*.csv'] [--bulk-mb 1] [--decode-nodes 4,5]
        [--headroom-factor 3]

``--topology`` / ``--config`` accept a plain path or a template containing
``{tag}`` (replaced by the run-dir basename), e.g.
``--config '/home/andre/tesi/.../conf/{tag}/config.txt'``.  If omitted we look for
``physical_topology.txt`` / ``config.txt`` inside each ns-3 run dir, then next to
the sweep root.  Without them the topology-derived metrics are skipped and the
analyzer degrades to the fct/pfc statistics only (and says so).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_ASTRA_ROOT = ("/home/andre/tesi/trace_evaluator/output/astra_logs/"
                      "llama2_13b_p-tp2pp2_d-tp2pp2_stream_16reqs_512prompt/buffer_sweep_T2")
DEFAULT_NS3_ROOT = "/home/andre/tesi/trace_evaluator/output/ns3/buffer_sweep_T2"
DEFAULT_OUTDIR = ("/home/andre/tesi/trace_evaluator/sweep_results/buffer_analysis_T2/"
                  "llama2_13b_p-tp2pp2_d-tp2pp2_stream_16req_512prompt")
DEFAULT_TOPOLOGY: str | None = ("/home/andre/tesi/trace_evaluator/configs/astra_sim/ns3/"
                                "buffer_sweep_T2/T2_bx100_dcqcn_buf2/physical_topology.txt")
DEFAULT_CONFIG: str | None = ("/home/andre/tesi/trace_evaluator/configs/astra_sim/ns3/"
                              "buffer_sweep_T2/{tag}/config.txt")
# NB: a single topology file is fine (it is cached and reused for every run), but
# --config MUST be per-tag: BUFFER_SIZE is the swept axis, and a single config.txt
# would make every run collapse onto one buffer value.

# ns-3 constants (switch-mmu.cc / common.h). Only used when config.txt is absent.
RESERVE_BYTES = 4 * 1024                 # SwitchMmu::reserve
DEFAULT_HEADROOM_FACTOR = 3              # common.h::headroom_factor
DEFAULT_PFC_SHIFT = 3                    # common.h: "uint32_t shift = 3; // by default 1/8"

NUMERIC_COLS = [
    "comm_size", "start_tick", "end_tick", "duration", "bw_bytes_per_ns",
    "operation_intensity", "compute_utilization", "memory_utilization",
    "is_memory_bound",
]

BUF_RE = re.compile(r"buf(\d+(?:\.\d+)?)", re.IGNORECASE)

PFC_QIDX_PATCH = r"""
--- a/src/point-to-point/model/qbb-net-device.h
-    TracedCallback<uint32_t> m_tracePfc;              // 0: resume, 1: pause
+    TracedCallback<uint32_t, uint32_t> m_tracePfc;    // (qIndex, type)
--- a/src/point-to-point/model/qbb-net-device.cc   (QbbNetDevice::Receive)
-                m_tracePfc(1);
+                m_tracePfc(qIndex, 1);
-                m_tracePfc(0);
+                m_tracePfc(qIndex, 0);
--- a/scratch/common.h
-void get_pfc(FILE *fout, Ptr<QbbNetDevice> dev, uint32_t type) {
-  fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(),
-          dev->GetNode()->GetId(), dev->GetNode()->GetNodeType(),
-          dev->GetIfIndex(), type);
+void get_pfc(FILE *fout, Ptr<QbbNetDevice> dev, uint32_t qIndex, uint32_t type) {
+  fprintf(fout, "%lu %u %u %u %u %u\n", Simulator::Now().GetTimeStep(),
+          dev->GetNode()->GetId(), dev->GetNode()->GetNodeType(),
+          dev->GetIfIndex(), type, qIndex);
   }
The qIndex is appended as a 6th column so existing parsers keying on p[0..4]
keep working.
"""


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #
_RATE_RE = re.compile(r"^\s*([\d.]+)\s*([kKmMgGtT]?)b(?:ps|/s)\s*$")
_TIME_RE = re.compile(r"^\s*([\d.]+)\s*(s|ms|us|ns)\s*$")
_RATE_MUL = {"": 1, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
_TIME_MUL = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}


def parse_rate(tok: str) -> int:
    """'1024Gbps' / '100Gb/s' -> bits per second (int, as ns-3 stores it)."""
    m = _RATE_RE.match(tok)
    if not m:
        raise ValueError(f"cannot parse data rate {tok!r}")
    return int(float(m.group(1)) * _RATE_MUL[m.group(2).lower()])


def parse_delay_ns(tok: str) -> int:
    """'0.005ms' -> 5000 (ns). ns-3's default time resolution is ns, and
    GetDelay().GetTimeStep() returns ns."""
    m = _TIME_RE.match(tok)
    if not m:
        raise ValueError(f"cannot parse delay {tok!r}")
    return int(round(float(m.group(1)) * _TIME_MUL[m.group(2)]))


def ip_to_node(tok: str) -> int:
    """node id from an fct.txt address token. entry.h prints the raw IP with
    %08x and common.h maps node<->ip as
        node_id_to_ip(id) = 0x0b000001 + (id/256)*0x10000 + (id%256)*0x100
        ip_to_node_id(ip) = (ip >> 8) & 0xffff
    so node 5 appears as '0b000501' (not '0b000005')."""
    tok = tok.strip()
    try:
        if "." in tok:
            a, b, c, d = (int(x) for x in tok.split("."))
            ip = (a << 24) | (b << 16) | (c << 8) | d
        else:
            ip = int(tok, 16)
    except (ValueError, IndexError):
        return -1
    return (ip >> 8) & 0xFFFF


def parse_buffer(dirname: str) -> float | None:
    m = BUF_RE.search(dirname)
    return float(m.group(1)) if m else None


def variant_key(dirname: str) -> str:
    return BUF_RE.sub("buf*", dirname)


def resolve_template(spec: str | None, tag: str, filename: str) -> Path | None:
    """Accept a file path, a template containing {tag}, or a directory (in which
    case <dir>/<tag>/<filename> is tried before <dir>/<filename>)."""
    if not spec:
        return None
    p = Path(spec.replace("{tag}", tag))
    if p.is_file():
        return p
    if p.is_dir():
        for cand in (p / tag / filename, p / filename):
            if cand.is_file():
                return cand
    return None


# --------------------------------------------------------------------------- #
# Topology model  (replicates common.h::SetupNetwork + CalculateRoute)
# --------------------------------------------------------------------------- #
@dataclass
class Link:
    a: int
    b: int
    rate: int          # bits/s
    delay_ns: int
    if_a: int          # ifindex of the device on node a
    if_b: int


@dataclass
class Topology:
    n_nodes: int
    switches: set[int]
    links: list[Link]
    # node -> ifindex -> (peer, rate, delay_ns)
    ports: dict[int, dict[int, tuple[int, int, int]]] = field(default_factory=dict)
    nic_rate: int = 0
    headroom_factor: int = DEFAULT_HEADROOM_FACTOR

    # -- derived, filled by _finalise ------------------------------------- #
    hdrm: dict[int, dict[int, int]] = field(default_factory=dict)   # sw -> port -> bytes
    total_hdrm: dict[int, int] = field(default_factory=dict)
    total_rsrv: dict[int, int] = field(default_factory=dict)
    shift: dict[int, dict[int, int]] = field(default_factory=dict)  # sw -> port -> pfc_a_shift
    pair_bw: dict[tuple[int, int], int] = field(default_factory=dict)
    next_hop: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    dist: dict[int, dict[int, int]] = field(default_factory=dict)
    ecmp_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def hosts(self) -> list[int]:
        return [i for i in range(self.n_nodes) if i not in self.switches]

    def is_switch(self, n: int) -> bool:
        return n in self.switches


def parse_topology(path: Path, headroom_factor: int = DEFAULT_HEADROOM_FACTOR) -> Topology:
    """physical_topology.txt:
           <node_num> <switch_num> <link_num>
           <switch ids...>
           <src> <dst> <rate> <delay> <error_rate>   x link_num
    Device indices are assigned in file order (device 0 is the loopback added by
    InternetStackHelper), exactly as qbb.Install() does in common.h."""
    toks = path.read_text().split()
    it = iter(toks)
    n_nodes, n_sw, n_link = int(next(it)), int(next(it)), int(next(it))
    switches = {int(next(it)) for _ in range(n_sw)}

    ports: dict[int, dict[int, tuple[int, int, int]]] = defaultdict(dict)
    next_if: dict[int, int] = defaultdict(lambda: 1)   # 0 = loopback
    links: list[Link] = []
    for _ in range(n_link):
        src, dst = int(next(it)), int(next(it))
        rate = parse_rate(next(it))
        delay = parse_delay_ns(next(it))
        next(it)                                        # error_rate (unused here)
        if_a, if_b = next_if[src], next_if[dst]
        next_if[src] += 1
        next_if[dst] += 1
        ports[src][if_a] = (dst, rate, delay)
        ports[dst][if_b] = (src, rate, delay)
        links.append(Link(src, dst, rate, delay, if_a, if_b))

    topo = Topology(n_nodes=n_nodes, switches=switches, links=links,
                    ports=dict(ports), headroom_factor=headroom_factor)
    _finalise(topo)
    return topo


def _finalise(topo: Topology) -> None:
    # --- nic_rate: common.h::get_nic_rate walks the NodeContainer in id order
    #     and returns the rate of device 1 of the FIRST host, i.e. of the first
    #     link in the topology file that mentions the lowest-id host.
    first_host = next((i for i in range(topo.n_nodes) if i not in topo.switches), None)
    topo.nic_rate = topo.ports[first_host][1][1] if first_host is not None else 0

    # --- per-switch headroom / reserve / pfc_a_shift ---------------------- #
    for sw in topo.switches:
        pmap = topo.ports.get(sw, {})
        topo.hdrm[sw] = {}
        topo.shift[sw] = {}
        for port, (_peer, rate, delay) in pmap.items():
            # common.h: uint32_t headroom = rate * delay / 8 / 1000000000 * headroom_factor
            # (uint64 integer arithmetic -- replicate the truncation exactly)
            h = (rate * delay) // 8 // 1_000_000_000 * topo.headroom_factor
            topo.hdrm[sw][port] = h
            s, r = DEFAULT_PFC_SHIFT, rate
            while r > topo.nic_rate and s > 0:
                s -= 1
                r //= 2
            topo.shift[sw][port] = s
        # ConfigNPort(GetNDevices()-1) -> sums headroom+reserve over real ports
        topo.total_hdrm[sw] = sum(topo.hdrm[sw].values())
        topo.total_rsrv[sw] = len(pmap) * RESERVE_BYTES

    # --- BFS routing (common.h::CalculateRoute): only switches are expanded,
    #     bw[] carries the min rate along the path -> that is entry.h's pairBw.
    INF = 1 << 62
    for host in topo.hosts:
        dis = {host: 0}
        bw = {host: INF}
        q = deque([host])
        while q:
            now = q.popleft()
            for _port, (nxt, rate, _d) in topo.ports.get(now, {}).items():
                if nxt not in dis:
                    dis[nxt] = dis[now] + 1
                    bw[nxt] = min(bw[now], rate)
                    if topo.is_switch(nxt):
                        q.append(nxt)
                if dis[nxt] == dis[now] + 1:
                    topo.next_hop.setdefault(nxt, {}).setdefault(host, [])
                    if now not in topo.next_hop[nxt][host]:
                        topo.next_hop[nxt][host].append(now)
        for node, d in dis.items():
            topo.dist.setdefault(node, {})[host] = d
            topo.pair_bw[(node, host)] = bw[node]

    # --- flag ECMP ties: SetRoutingEntries installs every equal-cost next hop,
    #     so the runtime path is chosen by hash and is NOT reconstructible here.
    for node, per_host in topo.next_hop.items():
        for host, hops in per_host.items():
            if len(hops) > 1 and topo.is_switch(node):
                topo.ecmp_pairs.append((node, host))


def path_links(topo: Topology, src: int, dst: int) -> list[tuple[int, int]] | None:
    """Directed (node, next) links on the src->dst shortest path. None if the
    path is ambiguous (ECMP) or unreachable."""
    cur, out, guard = src, [], 0
    while cur != dst:
        hops = topo.next_hop.get(cur, {}).get(dst, [])
        if len(hops) != 1:
            return None
        out.append((cur, hops[0]))
        cur = hops[0]
        guard += 1
        if guard > topo.n_nodes:
            return None
    return out


def port_of(topo: Topology, node: int, peer: int) -> int | None:
    for port, (p, _r, _d) in topo.ports.get(node, {}).items():
        if p == peer:
            return port
    return None


def pfc_threshold(topo: Topology, sw: int, port: int, buffer_bytes: int,
                  shared_used: int = 0) -> float:
    """switch-mmu.cc::GetPfcThreshold, with shared_used_bytes = 0 (the maximum;
    the real threshold shrinks as the shared pool fills, so this is an upper
    bound and the PFC regime is entered *earlier* than this suggests)."""
    avail = buffer_bytes - topo.total_hdrm[sw] - topo.total_rsrv[sw] - shared_used
    if avail <= 0:
        return 0.0
    return float(int(avail) >> topo.shift[sw][port])


# --------------------------------------------------------------------------- #
# ns-3 config.txt
# --------------------------------------------------------------------------- #
@dataclass
class Ns3Config:
    buffer_mb: float | None = None
    cc_mode: int | None = None
    enable_qcn: int | None = None
    dynamic_pfc: int | None = None
    kmin: dict[int, int] = field(default_factory=dict)   # rate(bit/s) -> bytes
    kmax: dict[int, int] = field(default_factory=dict)
    pmax: dict[int, float] = field(default_factory=dict)
    topology_file: str | None = None


def parse_ns3_config(path: Path) -> Ns3Config:
    """ConfigEcn multiplies the KMIN/KMAX map values by 1000 (decimal kB), and
    ConfigBufferSize multiplies BUFFER_SIZE by 1024*1024 (MiB). Keys are link
    rates in bit/s and must match GetBitRate() exactly or ns-3 NS_ASSERTs."""
    cfg = Ns3Config()
    for raw in path.read_text().splitlines():
        p = raw.split()
        if not p:
            continue
        key = p[0].upper()
        try:
            if key == "BUFFER_SIZE":
                cfg.buffer_mb = float(p[1])
            elif key == "CC_MODE":
                cfg.cc_mode = int(p[1])
            elif key == "ENABLE_QCN":
                cfg.enable_qcn = int(p[1])
            elif key == "USE_DYNAMIC_PFC_THRESHOLD":
                cfg.dynamic_pfc = int(p[1])
            elif key == "TOPOLOGY_FILE":
                cfg.topology_file = p[1]
            elif key in ("KMIN_MAP", "KMAX_MAP", "PMAX_MAP"):
                cnt = int(p[1])
                tgt = {"KMIN_MAP": cfg.kmin, "KMAX_MAP": cfg.kmax, "PMAX_MAP": cfg.pmax}[key]
                for i in range(cnt):
                    rate = int(p[2 + 2 * i])
                    val = p[3 + 2 * i]
                    tgt[rate] = float(val) if key == "PMAX_MAP" else int(float(val) * 1000)
        except (IndexError, ValueError):
            print(f"  ! malformed line in {path.name}: {raw.strip()!r}", file=sys.stderr)
    return cfg


# --------------------------------------------------------------------------- #
# ASTRA-sim CSV side  (magnitude: KV completion, makespan, gating)
# --------------------------------------------------------------------------- #
def classify_op(name: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        return "OTHER", "other"
    head = name.split("_", 1)[0].upper()
    op = head if head in {"COMP", "TP", "KV", "PP", "FIRSTTOK", "DECFB"} else "OTHER"
    if op == "KV":
        phase = "kv_transfer"          # checked before pl= so KV never becomes 'prefill'
    elif "pl=p" in name:
        phase = "prefill"
    elif "pl=d" in name:
        phase = "decode"
    elif op == "FIRSTTOK":
        phase = "handoff"
    elif op == "DECFB":
        phase = "decode"
    else:
        phase = "other"
    return op, phase


def load_astra_run(run_dir: Path, pattern: str) -> pd.DataFrame | None:
    frames = []
    for csv in sorted(run_dir.glob(pattern)):
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read {csv.name}: {exc}", file=sys.stderr)
            continue
        if df.empty or "name" not in df.columns:
            continue
        df["__file__"] = csv.name
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    cp = df["name"].map(classify_op)
    df["op_class"] = cp.map(lambda t: t[0])
    df["phase"] = cp.map(lambda t: t[1])
    return df


def _comm_direction(df: pd.DataFrame):
    """'send'/'recv'/'other' per row, from the Chakra node type. Handles both the
    string form (COMM_SEND_NODE/COMM_RECV_NODE) and the numeric enum (4/5)."""
    if "type" not in df.columns:
        return None
    t = df["type"]
    if t.dtype == object:
        u = t.astype(str).str.upper()
        return pd.Series(np.where(u.str.contains("SEND"), "send",
                                  np.where(u.str.contains("RECV"), "recv", "other")),
                         index=df.index)
    tn = pd.to_numeric(t, errors="coerce")
    return pd.Series(np.where(tn == 4, "send", np.where(tn == 5, "recv", "other")),
                     index=df.index)


def summarise_astra(df: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    out["n_nodes"] = int(df["sys_id"].nunique()) if "sys_id" in df else np.nan
    out["n_rows"] = len(df)

    gstart, gend = df["start_tick"].min(), df["end_tick"].max()
    makespan = gend - gstart
    out["makespan_ns"] = makespan
    out["makespan_ms"] = makespan / 1e6

    def last_end(mask) -> float:
        s = df.loc[mask, "end_tick"]
        return float(s.max()) if len(s) else np.nan

    kv_mask = df["op_class"] == "KV"
    comp_mask = df["op_class"] == "COMP"
    out["kv_completion_ns"] = last_end(kv_mask)
    out["comp_completion_ns"] = last_end(comp_mask)
    out["prefill_comp_completion_ns"] = last_end(comp_mask & (df["phase"] == "prefill"))
    out["kv_bound_ratio"] = (out["kv_completion_ns"] / makespan
                             if makespan and makespan > 0 else np.nan)

    kv = df[kv_mask]
    out["kv_count"] = len(kv)
    if len(kv):
        # Each transfer produces BOTH a COMM_SEND and a COMM_RECV row (in the
        # sender's and the receiver's stats_sys*.csv), so summing comm_size over
        # every row double-counts every byte. Keep the send side only.
        d = _comm_direction(kv)
        if d is not None and (d == "send").any() and (d == "recv").any():
            kv = kv[d == "send"]
            out["kv_flows"] = len(kv)
            out["kv_dedup"] = "send-only"
        else:
            out["kv_flows"] = len(kv)
            out["kv_dedup"] = "raw (no type column: bytes may be double-counted)"
        total_bytes = kv["comm_size"].sum(min_count=1)
        # interval UNION, not sum: ASTRA-sim posts dependency-free RECV nodes
        # eagerly at tick=0, so summing durations double-counts overlap.
        iv = sorted(zip(kv["start_tick"], kv["end_tick"]))
        union, cur_s, cur_e = 0.0, None, None
        for s0, e0 in iv:
            if cur_s is None:
                cur_s, cur_e = s0, e0
            elif s0 <= cur_e:
                cur_e = max(cur_e, e0)
            else:
                union += cur_e - cur_s
                cur_s, cur_e = s0, e0
        if cur_s is not None:
            union += cur_e - cur_s
        window = kv["end_tick"].max() - kv["start_tick"].min()
        out["kv_total_GB"] = total_bytes / 1e9 if pd.notna(total_bytes) else np.nan
        out["kv_busy_union_ns"] = union
        out["kv_agg_bw_bytes_per_ns"] = (total_bytes / window
                                         if window and window > 0 else np.nan)
        out["kv_mean_duration_ns"] = kv["duration"].mean()
        out["kv_max_duration_ns"] = kv["duration"].max()
        kv_end_per_sys = kv.groupby("sys_id")["end_tick"].max()
        out["kv_end_spread_ns"] = (float(kv_end_per_sys.max() - kv_end_per_sys.min())
                                   if len(kv_end_per_sys) > 1 else 0.0)
    else:
        for k in ("kv_total_GB", "kv_busy_union_ns", "kv_agg_bw_bytes_per_ns",
                  "kv_mean_duration_ns", "kv_max_duration_ns", "kv_end_spread_ns"):
            out[k] = np.nan
    return out


# --------------------------------------------------------------------------- #
# ns-3 parsers
# --------------------------------------------------------------------------- #
def parse_fct(path: Path) -> pd.DataFrame | None:
    """entry.h::qp_finish_print_log
         %08x %08x %u %u %lu %lu %lu %lu
         sip dip sport dport size(B) start(ns) fct(ns) standalone_fct(ns)
    standalone_fct = base_rtt + total_bytes*8e9/pairBw, with pairBw = the min
    link rate along the BFS path -> slowdown = fct/standalone_fct is the
    'alone on the bottleneck' normalisation (floor = fan-in, not 1)."""
    if not path.is_file():
        return None
    rows = []
    for line in path.open():
        p = line.split()
        if len(p) < 8:
            continue
        try:
            rows.append((ip_to_node(p[0]), ip_to_node(p[1]), int(p[2]), int(p[3]),
                         int(p[4]), int(p[5]), int(p[6]), int(p[7])))
        except ValueError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["src", "dst", "sport", "dport",
                                     "size", "start", "fct", "sfct"])
    df["arrival"] = df["start"] + df["fct"]
    df["slowdown"] = np.where(df["sfct"] > 0, df["fct"] / df["sfct"], np.nan)
    return df


def parse_pfc(path: Path) -> dict | None:
    """common.h::get_pfc -> time node node_type ifindex type [qIndex]
       node_type 0=host 1=switch ; type 0=RESUME 1=PAUSE
    The trace fires on the device that RECEIVES the PAUSE frame, i.e. the one
    being paused.  qIndex is present only with PFC_QIDX_PATCH applied; without it
    the per-queue sequences are interleaved and the state machine mis-pairs."""
    if not path.is_file():
        return None
    ev: dict[tuple, list] = defaultdict(list)
    tmax, n, has_q = 0, 0, True
    for line in path.open():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            t, node, ntype, ifidx, typ = (int(p[i]) for i in range(5))
            qidx = int(p[5]) if len(p) > 5 else -1
        except ValueError:
            continue
        if qidx < 0:
            has_q = False
        n += 1
        tmax = max(tmax, t)
        ev[(node, ntype, ifidx, qidx)].append((t, typ))
    if n == 0:
        return dict(n=0, totals={}, tmax=0, qidx_resolved=True, unclosed=0)
    return dict(n=n, events=dict(ev), tmax=tmax, qidx_resolved=has_q)


def pfc_totals(pfc: dict, clamp_to: int) -> tuple[dict, int]:
    """Per (node, ntype, ifidx, qidx) total paused time. There is no pause
    timeout on the receiving side of this fork (QbbNetDevice::Receive only sets
    m_paused[qIndex]; nothing schedules a PauseFinish), so PAUSE/RESUME strictly
    alternate per queue and an unclosed PAUSE really lasts to the end of the run
    -> clamp to the run end, not to the last PFC event."""
    totals, unclosed = {}, 0
    for key, events in pfc.get("events", {}).items():
        events.sort()
        tot, start = 0, None
        for t, typ in events:
            if typ == 1 and start is None:
                start = t
            elif typ == 0 and start is not None:
                tot += t - start
                start = None
        if start is not None:
            unclosed += 1
            tot += max(clamp_to, start) - start
        totals[key] = tot
    return totals, unclosed


def parse_qlen(path: Path) -> dict | None:
    """common.h::monitor_buffer
         time <t> <switch_id> j <port> <bytes> j <port> <bytes> ...
    <bytes> is sum_k egress_bytes[port][k] for ONE port; a port appears only when
    its queue is >= 1000 B.  This is per-egress-port occupancy -- the quantity
    ShouldSendCN() compares against kmin/kmax[ifindex].  It is NOT the shared
    pool (that is ingress-side and unobservable from this file)."""
    if not path.is_file():
        return None
    port_max: dict[tuple[int, int], int] = defaultdict(int)
    port_sum: dict[tuple[int, int], int] = defaultdict(int)
    port_cnt: dict[tuple[int, int], int] = defaultdict(int)
    sw_total_max: dict[int, int] = defaultdict(int)
    samples = 0
    for line in path.open():
        p = line.split()
        if len(p) < 3 or p[0] != "time":
            continue
        try:
            sw = int(p[2])
        except ValueError:
            continue
        i, line_total = 3, 0
        # i+2 < len(p), not <= : a truncated final line (killed run) must not
        # IndexError here.
        while i < len(p):
            if p[i] == "j" and i + 2 < len(p):
                try:
                    port, b = int(p[i + 1]), int(p[i + 2])
                except ValueError:
                    i += 1
                    continue
                key = (sw, port)
                port_cnt[key] += 1
                port_sum[key] += b
                if b > port_max[key]:
                    port_max[key] = b
                line_total += b
                i += 3
            else:
                i += 1
        samples += 1
        if line_total > sw_total_max[sw]:
            sw_total_max[sw] = line_total
    if samples == 0:
        return dict(samples=0, port_max={}, port_mean={}, sw_total_max={})
    port_mean = {k: port_sum[k] / port_cnt[k] for k in port_cnt}
    return dict(samples=samples, port_max=dict(port_max), port_mean=port_mean,
                port_cnt=dict(port_cnt), sw_total_max=dict(sw_total_max))


# --------------------------------------------------------------------------- #
# Flow classification + fan-in
# --------------------------------------------------------------------------- #
def annotate_flows(fct: pd.DataFrame, topo: Topology | None,
                   bulk_bytes: int) -> pd.DataFrame:
    """Tag every flow with hop count, bottleneck rate and its path's directed
    links, so that direct host-host TP traffic (1 hop, 4800 Gbps, never
    congested -> slowdown ~1 by construction) can be excluded from the incast
    statistics instead of diluting the mean and the CV."""
    fct = fct.copy()
    fct["bulk"] = fct["size"] >= bulk_bytes
    if topo is None:
        fct["hops"] = np.nan
        fct["bottleneck_bps"] = np.nan
        fct["fabric"] = True
        fct["path"] = [None] * len(fct)
        return fct
    hops, bw, paths = [], [], []
    for src, dst in zip(fct["src"], fct["dst"]):
        hops.append(topo.dist.get(src, {}).get(dst, np.nan))
        bw.append(topo.pair_bw.get((src, dst), np.nan))
        paths.append(path_links(topo, src, dst))
    fct["hops"] = hops
    fct["bottleneck_bps"] = bw
    fct["path"] = paths
    fct["fabric"] = fct["hops"] > 1          # traverses at least one switch
    return fct


def max_concurrency(intervals: list[tuple[int, int]]) -> int:
    """Peak number of simultaneously active flows."""
    if not intervals:
        return 0
    ev = [(s, 1) for s, _e in intervals] + [(e, -1) for _s, e in intervals]
    ev.sort(key=lambda x: (x[0], x[1]))      # closes before opens at a tie
    cur = best = 0
    for _t, d in ev:
        cur += d
        best = max(best, cur)
    return best


def concurrency_stats(intervals: list[tuple[int, int]]) -> tuple[int, float]:
    """(peak, mean_experienced).

    ``peak`` is the max number of flows alive at any instant.  ``mean_experienced``
    is the average, over flows, of the time-averaged concurrency during that
    flow's *own* lifetime -- i.e. the fair-share slowdown a flow should actually
    expect.  The peak is only an upper bound: reporting it as a 'floor' is
    wrong-signed, which is why the measured mean slowdown sits *below* it."""
    if not intervals:
        return 0, float("nan")
    peak = max_concurrency(intervals)
    ev = sorted([(s, 1) for s, _e in intervals] + [(e, -1) for _s, e in intervals],
                key=lambda x: (x[0], x[1]))
    segs, cur = [], 0
    for i in range(len(ev) - 1):
        cur += ev[i][1]
        if ev[i + 1][0] > ev[i][0]:
            segs.append((ev[i][0], ev[i + 1][0], cur))
    means = []
    for s, e in intervals:
        if e <= s:
            continue
        acc = 0.0
        for t0, t1, k in segs:
            lo, hi = max(t0, s), min(t1, e)
            if hi > lo:
                acc += k * (hi - lo)
        means.append(acc / (e - s))
    return peak, (float(np.mean(means)) if means else float("nan"))


def ingress_ports_of(topo: Topology, sw: int, peer: int,
                     fct: pd.DataFrame) -> tuple[set[int], set[int]]:
    """Ingress ports of `sw` through which the bulk fabric flows leaving on the
    sw->peer egress port arrive, plus the *devices* that get PAUSEd when those
    ports fill.

    This is the quantity the PFC threshold is compared against: SwitchMmu keys
    ingress accounting on (port, qIndex), NOT on flows.  42 flows entering
    through 2 host ports load 2 ingress counters, so the egress-equivalent of the
    threshold is 2*threshold -- not 42*threshold, which would exceed the physical
    buffer by 3-5x and is how v2 broke its own regime prediction.

    The PAUSE trace fires on the device that *receives* the pause frame, so the
    devices to watch are the peers' ports facing `sw`."""
    ports: set[int] = set()
    for path, bulk, fab in zip(fct["path"], fct["bulk"], fct["fabric"]):
        if not path or not bulk or not fab:
            continue
        for i, (a, b) in enumerate(path):
            if (a, b) == (sw, peer) and i > 0:
                prev = path[i - 1][0]
                p = port_of(topo, sw, prev)
                if p is not None:
                    ports.add(p)
    victims: set[int] = set()
    for p in ports:
        upstream = topo.ports[sw][p][0]
        vp = port_of(topo, upstream, sw)
        if vp is not None:
            victims.add((upstream, vp))
    return ports, victims


def congested_link(topo: Topology | None, qlen: dict | None,
                   fct: pd.DataFrame | None) -> tuple | None:
    """Return (sw, port, peer, rate) for the directed link that actually built a
    queue.  Ground truth is qlen.txt (peak per-port egress occupancy); if qlen is
    missing we fall back to the link carrying the most bulk fabric bytes."""
    if topo is not None and qlen and qlen.get("port_max"):
        (sw, port), _peak = max(qlen["port_max"].items(), key=lambda kv: kv[1])
        info = topo.ports.get(sw, {}).get(port)
        if info:
            peer, rate, _d = info
            return sw, port, peer, rate
    if topo is not None and fct is not None:
        load: dict[tuple[int, int], int] = defaultdict(int)
        for path, size, is_bulk, fab in zip(fct["path"], fct["size"],
                                            fct["bulk"], fct["fabric"]):
            if not path or not is_bulk or not fab:
                continue
            for a, b in path:
                if topo.is_switch(a):
                    load[(a, b)] += size
        if load:
            (a, b), _ = max(load.items(), key=lambda kv: kv[1])
            port = port_of(topo, a, b)
            if port is not None:
                return a, port, b, topo.ports[a][port][1]
    return None


# --------------------------------------------------------------------------- #
# Per-run summarisation
# --------------------------------------------------------------------------- #
def summarise_ns3(fct: pd.DataFrame | None, pfc: dict | None, qlen: dict | None,
                  topo: Topology | None, cfg: Ns3Config | None,
                  buffer_mb: float, decode_nodes: list[int]) -> dict:
    out: dict = {}
    buffer_bytes = int(buffer_mb * 1024 * 1024)
    out["buffer_bytes"] = buffer_bytes
    out["pfc_thresh_naive_bytes"] = buffer_bytes / 8.0      # v1's assumption, kept for contrast

    # ---- which link is congested, and what governs it ---------------------- #
    cl = congested_link(topo, qlen, fct)
    F_ports = np.nan
    n_peak, n_mean = np.nan, np.nan
    victims: set = set()
    if cl and topo is not None:
        sw, port, peer, rate = cl
        out["congested_link"] = f"{sw}->{peer}"
        out["congested_port"] = port
        out["congested_rate_gbps"] = rate / 1e9
        out["pfc_thresh_bytes"] = pfc_threshold(topo, sw, port, buffer_bytes)
        out["pfc_shift"] = topo.shift[sw][port]
        out["sw_total_hdrm_bytes"] = topo.total_hdrm[sw]
        out["sw_total_rsrv_bytes"] = topo.total_rsrv[sw]
        out["hdrm_pct_of_buffer"] = 100.0 * (topo.total_hdrm[sw] + topo.total_rsrv[sw]) / buffer_bytes
        if cfg and cfg.kmin and rate in cfg.kmin:
            out["kmin_bytes"] = cfg.kmin[rate]
            out["kmax_bytes"] = cfg.kmax.get(rate, np.nan)
        if fct is not None:
            iports, victims = ingress_ports_of(topo, sw, peer, fct)
            F_ports = len(iports) or np.nan
            out["fanin_ports"] = F_ports
            out["ingress_ports"] = ",".join(str(p) for p in sorted(iports))
            iv = [(s, a) for path, s, a, bulk, fab in
                  zip(fct["path"], fct["start"], fct["arrival"], fct["bulk"], fct["fabric"])
                  if path and bulk and fab and (sw, peer) in path]
            n_peak, n_mean = concurrency_stats(iv)
            out["n_concurrent_peak"] = n_peak
            out["n_concurrent_mean"] = n_mean
            out["bottleneck_flows"] = len(iv)
            # The PFC ceiling on the egress queue: once every ingress port is
            # paused, each can still hold reserve + threshold + its own headroom.
            # Measured peak egress ~= this ceiling  <=>  the run is PFC-limited.
            ceil = sum(RESERVE_BYTES + out["pfc_thresh_bytes"] + topo.hdrm[sw][p]
                       for p in iports)
            out["pfc_egress_ceiling_bytes"] = ceil if iports else np.nan
    else:
        for k in ("congested_link", "congested_port", "congested_rate_gbps",
                  "pfc_thresh_bytes", "pfc_shift", "sw_total_hdrm_bytes",
                  "sw_total_rsrv_bytes", "hdrm_pct_of_buffer", "kmin_bytes",
                  "kmax_bytes", "fanin_ports", "ingress_ports", "n_concurrent_peak",
                  "n_concurrent_mean", "bottleneck_flows", "pfc_egress_ceiling_bytes"):
            out.setdefault(k, np.nan)

    # PFC watches per-INGRESS-PORT occupancy, ECN per-EGRESS-port occupancy. The
    # same bytes sit on F_ports ingress counters but on one egress counter, so
    # PFC fires at an egress-equivalent of F_ports * threshold.
    thr = out.get("pfc_thresh_bytes", np.nan)
    if thr == thr and F_ports == F_ports and F_ports:
        eq = thr * F_ports
        if eq > buffer_bytes:
            # Cannot happen physically: the threshold is carved out of the buffer.
            # Guard against ever re-introducing the flows-vs-ports confusion.
            print(f"  ! egress-equivalent threshold ({eq/1e6:.1f} MB) exceeds the "
                  f"physical buffer ({buffer_bytes/1e6:.1f} MB) — F_ports={F_ports} "
                  f"is not a port count", file=sys.stderr)
        out["pfc_thresh_egress_equiv_bytes"] = eq
        kmax, kmin = out.get("kmax_bytes", np.nan), out.get("kmin_bytes", np.nan)
        if kmax == kmax:
            out["thresh_over_kmax"] = eq / kmax
        if kmin == kmin:
            out["thresh_over_kmin"] = eq / kmin

    # ---- incast window: pause% must be normalised by the window in which the
    #      incast actually happens.  The ns-3 clock is the global clock in
    #      ASTRA-sim and advances during prefill compute too, so normalising by
    #      the whole run dilutes every percentage by the prompt length.
    win_lo, win_hi, run_end = np.nan, np.nan, 0
    if fct is not None and len(fct):
        run_end = int(fct["arrival"].max())
        inc = fct[fct["bulk"] & fct["fabric"]]
        if len(inc):
            win_lo, win_hi = int(inc["start"].min()), int(inc["arrival"].max())
    window = (win_hi - win_lo) if (win_hi == win_hi and win_hi > win_lo) else np.nan
    out["incast_window_ns"] = window
    out["run_end_ns"] = run_end

    # ---- PFC ------------------------------------------------------------- #
    # An existing-but-empty pfc.txt is a measurement (zero PAUSE = DCQCN), not a
    # missing file: it must yield 0.0, not NaN, or the DCQCN end of the sweep
    # disappears from the regime plot.
    if pfc is not None:
        totals, unclosed = pfc_totals(pfc, clamp_to=run_end or pfc.get("tmax", 0))
        out["pfc_events"] = pfc["n"]
        # tri-state: an empty pfc.txt has nothing to resolve, which is not the
        # same as "the qIndex column is there".
        out["pfc_qidx"] = ("n/a" if pfc["n"] == 0
                           else ("present" if pfc.get("qidx_resolved") else "MISSING"))
        out["pfc_unclosed_pauses"] = unclosed
        denom = window if window == window and window > 0 else (run_end or np.nan)

        def _pct(v):
            return 100.0 * v / denom if denom and denom == denom else np.nan

        # Per DEVICE (summing across devices and dividing by a time window can
        # trivially exceed 100%, which is what made v2 report 283%).
        per_dev: dict[tuple, int] = defaultdict(int)
        for (node, ntype, ifidx, _q), v in totals.items():
            per_dev[(node, ntype, ifidx)] += v      # queues of one device do overlap
        worst_key = max(per_dev, key=per_dev.get, default=None)
        out["pfc_worst_link_pause_ns"] = per_dev.get(worst_key, 0) if worst_key else 0
        out["pfc_worst_link_pause_pct"] = _pct(out["pfc_worst_link_pause_ns"])
        out["pfc_worst_link_device"] = (f"n{worst_key[0]}/if{worst_key[2]}"
                                        f"{'(sw)' if worst_key[1] == 1 else '(host)'}"
                                        if worst_key else "")
        hostv = [v for (_n, nt, _i), v in per_dev.items() if nt == 0]
        swv = [v for (_n, nt, _i), v in per_dev.items() if nt == 1]
        out["pfc_host_pause_max_pct"] = _pct(max(hostv)) if hostv else 0.0
        out["pfc_sw_pause_max_pct"] = _pct(max(swv)) if swv else 0.0
        out["pfc_paused_devices"] = sum(1 for v in per_dev.values() if v > 0)
        # Pause on the devices upstream of the CONGESTED link specifically. The
        # global worst can sit on a completely different link, so it is not
        # evidence about the bottleneck's regime.
        if victims:
            vv = [v for (n, nt, i), v in per_dev.items() if (n, i) in victims]
            out["pfc_bottleneck_pause_pct"] = _pct(max(vv)) if vv else 0.0
            out["pfc_bottleneck_devices"] = len(victims)
        else:
            out["pfc_bottleneck_pause_pct"] = np.nan
    else:
        for k in ("pfc_events", "pfc_unclosed_pauses", "pfc_worst_link_pause_ns",
                  "pfc_worst_link_pause_pct", "pfc_host_pause_max_pct",
                  "pfc_sw_pause_max_pct", "pfc_bottleneck_pause_pct",
                  "pfc_paused_devices"):
            out[k] = np.nan
        out["pfc_qidx"] = "n/a"
        out["pfc_worst_link_device"] = ""

    # ---- qlen (ECN axis) -------------------------------------------------- #
    if qlen and qlen.get("port_max"):
        if cl:
            key = (cl[0], cl[1])
            peak = qlen["port_max"].get(key, np.nan)
            out["qlen_peak_congested_port_bytes"] = peak
            kmax = out.get("kmax_bytes", np.nan)
            kmin = out.get("kmin_bytes", np.nan)
            if peak == peak and kmax == kmax and kmax:
                out["qlen_peak_over_kmax"] = peak / kmax
            if peak == peak and kmin == kmin and kmin:
                out["qlen_peak_over_kmin"] = peak / kmin
            out["qlen_mean_congested_port_bytes"] = qlen["port_mean"].get(key, np.nan)
        out["qlen_peak_switch_total_bytes"] = max(qlen["sw_total_max"].values(), default=np.nan)
        out["qlen_peak_switch_pct_buf"] = (100.0 * out["qlen_peak_switch_total_bytes"]
                                           / buffer_bytes)
        out["qlen_samples"] = qlen["samples"]
    else:
        for k in ("qlen_peak_congested_port_bytes", "qlen_peak_over_kmax",
                  "qlen_peak_over_kmin", "qlen_mean_congested_port_bytes",
                  "qlen_peak_switch_total_bytes", "qlen_peak_switch_pct_buf",
                  "qlen_samples"):
            out[k] = np.nan

    # ---- the trade-off: mean vs tail, on the INCAST flows only ------------ #
    if fct is not None and len(fct):
        inc = fct[fct["bulk"] & fct["fabric"] & fct["slowdown"].notna()]
        out["fct_flows"] = len(fct)
        out["fct_bulk_fabric_flows"] = len(inc)
        out["fct_direct_flows"] = int((~fct["fabric"]).sum())
        if len(inc) == 0:
            print("  ! no bulk fabric flows: check --bulk-mb against the actual "
                  "KV flow size (no silent fallback to all flows)", file=sys.stderr)
        sd = inc["slowdown"].to_numpy(float)
        out["slow_mean_bulk"] = float(np.mean(sd)) if len(sd) else np.nan
        out["slow_p50_bulk"] = float(np.percentile(sd, 50)) if len(sd) else np.nan
        out["slow_p99_bulk"] = float(np.percentile(sd, 99)) if len(sd) else np.nan
        out["slow_max_bulk"] = float(np.max(sd)) if len(sd) else np.nan
        m = np.mean(sd) if len(sd) else np.nan
        s = np.std(sd, ddof=1) if len(sd) > 1 else 0.0
        out["slow_cv_bulk"] = float(s / m) if m and m == m else np.nan
        # Slowdown normalised by the fair-share expectation. standalone_fct
        # assumes the flow owns the bottleneck, so N flows sharing it fairly give
        # slowdown ~ N. The reference is the concurrency each flow *experiences*
        # (n_concurrent_mean); the peak is only an upper bound, which is why the
        # measured mean legitimately sits below it.
        if n_mean == n_mean and n_mean:
            out["slow_mean_over_fairshare"] = out["slow_mean_bulk"] / n_mean
            out["slow_p99_over_fairshare"] = out["slow_p99_bulk"] / n_mean
        direct = fct[~fct["fabric"] & fct["slowdown"].notna()]["slowdown"]
        out["slow_mean_direct"] = float(direct.mean()) if len(direct) else np.nan

        # ---- barrier: when can decode start? ---------------------------- #
        bulk_fab = fct[fct["bulk"] & fct["fabric"]]
        if decode_nodes:
            dnodes = [d for d in decode_nodes if d in set(bulk_fab["dst"])]
        else:
            cnt = bulk_fab.groupby("dst").size().sort_values(ascending=False)
            dnodes = [int(d) for d, c in cnt.items() if c > 1] or \
                     ([int(cnt.index[0])] if len(cnt) else [])
        rank_ready, rank_skew = [], []
        for d in dnodes:
            arr = bulk_fab.loc[bulk_fab["dst"] == d, "arrival"]
            if not len(arr):
                continue
            rank_ready.append(float(arr.max()))
            rank_skew.append(float(arr.max() - arr.min()))
        if rank_ready:
            out["kv_ready_max_ns"] = max(rank_ready)     # decode-start gate
            out["kv_ready_min_ns"] = min(rank_ready)
            out["sync_skew_ns"] = max(rank_skew)         # worst per-rank incast spread
            out["cross_rank_skew_ns"] = max(rank_ready) - min(rank_ready)
            out["decode_ranks_seen"] = len(rank_ready)
        else:
            for k in ("kv_ready_max_ns", "kv_ready_min_ns", "sync_skew_ns",
                      "cross_rank_skew_ns"):
                out[k] = np.nan
            out["decode_ranks_seen"] = 0
        out["incast_dst"] = ",".join(str(d) for d in dnodes)
    else:
        for k in ("fct_flows", "fct_bulk_fabric_flows", "fct_direct_flows",
                  "slow_mean_bulk", "slow_p50_bulk", "slow_p99_bulk", "slow_max_bulk",
                  "slow_cv_bulk", "slow_mean_over_fairshare", "slow_p99_over_fairshare",
                  "slow_mean_direct", "kv_ready_max_ns", "kv_ready_min_ns",
                  "sync_skew_ns", "cross_rank_skew_ns", "decode_ranks_seen"):
            out[k] = np.nan
        out["incast_dst"] = ""
    return out


def regime_predicted(row: dict) -> str:
    """Physics, not statistics: compare the egress-equivalent PFC threshold with
    the ECN band [KMIN, KMAX] on the congested port."""
    t = row.get("pfc_thresh_egress_equiv_bytes", np.nan)
    kmin, kmax = row.get("kmin_bytes", np.nan), row.get("kmax_bytes", np.nan)
    if not (t == t and kmin == kmin and kmax == kmax):
        return "?"
    if t < kmin:
        return "PFC"          # PFC fires before ECN even starts marking
    if t > kmax:
        return "DCQCN"        # ECN saturates long before PFC can trigger
    return "MIXED"


def regime_observed(row: dict) -> str:
    """Measured, not assumed. The decisive test is whether the egress queue on the
    congested port rides at the PFC ceiling -- sum over ingress ports of
    (reserve + threshold + headroom). At the ceiling the queue is held by
    backpressure; below it, the rate control is what is limiting."""
    pause = row.get("pfc_bottleneck_pause_pct", np.nan)
    if pause != pause:
        pause = row.get("pfc_worst_link_pause_pct", np.nan)
    peak = row.get("qlen_peak_congested_port_bytes", np.nan)
    ceil = row.get("pfc_egress_ceiling_bytes", np.nan)
    at_ceiling = (peak == peak and ceil == ceil and ceil > 0 and peak / ceil >= 0.95)
    if pause != pause:
        return "?"
    if at_ceiling and pause > 1.0:
        return "PFC"
    if pause > 0.5:
        return "MIXED"
    return "DCQCN"


def flip_buffers(topo: Topology, sw: int, port: int, kmin: float, kmax: float,
                 f_ports: float) -> tuple[float, float]:
    """Buffers (MiB) at which F*threshold crosses KMIN and KMAX:
           F_ports * ((B - hdrm - rsrv) >> shift) == K
       ->  B = K*2^shift/F_ports + hdrm + rsrv
    Left of the KMIN crossing the fabric is PFC-dominated; right of the KMAX
    crossing it is ECN/DCQCN-dominated; between them, mixed."""
    base = topo.total_hdrm[sw] + topo.total_rsrv[sw]
    mul = 1 << topo.shift[sw][port]
    lo = (kmin * mul / f_ports + base) / (1024 * 1024)
    hi = (kmax * mul / f_ports + base) / (1024 * 1024)
    return lo, hi


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def resolve_ns3_root(astra_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    if DEFAULT_NS3_ROOT and Path(DEFAULT_NS3_ROOT).is_dir():
        return Path(DEFAULT_NS3_ROOT)
    if any(astra_root.rglob("fct.txt")):
        return astra_root
    return None


def find_ns3_run(ns3_root: Path | None, run_name: str) -> Path | None:
    if ns3_root is None:
        return None
    cand = ns3_root / run_name
    if (cand / "fct.txt").is_file() or (cand / "pfc.txt").is_file():
        return cand
    for d in ns3_root.rglob(run_name):
        if d.is_dir() and ((d / "fct.txt").is_file() or (d / "pfc.txt").is_file()):
            return d
    return None


def find_aux(spec: str | None, tag: str, ns3_dir: Path | None,
             roots: list[Path], filename: str) -> Path | None:
    p = resolve_template(spec, tag, filename)
    if p:
        return p
    for base in ([ns3_dir] if ns3_dir else []) + roots:
        if base is None:
            continue
        for cand in (base / filename, base / tag / filename, base.parent / filename):
            if cand.is_file():
                return cand
    return None


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _series(ax, data: pd.DataFrame, ycol: str, label: str, marker="o",
            scale=1.0, linestyle="-", color=None):
    if ycol not in data.columns:
        return
    for variant, grp in data.groupby("variant"):
        grp = grp.dropna(subset=[ycol]).sort_values("buffer_mb")
        if grp.empty:
            continue
        lbl = label if data["variant"].nunique() == 1 else f"{label} [{variant}]"
        ax.plot(grp["buffer_mb"], grp[ycol] * scale, marker=marker,
                linestyle=linestyle, label=lbl, color=color)


def _logx(ax, s: pd.DataFrame):
    ax.set_xscale("log", base=2)
    xs = sorted(s["buffer_mb"].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("Per-switch buffer (MiB)")


def _flip_band(ax, band: tuple[float, float] | None):
    """The regime transition is a BAND, not a line: PFC below the KMIN crossing,
    DCQCN above the KMAX crossing, mixed in between."""
    if not band:
        return
    lo, hi = band
    ax.axvspan(lo, hi, color="#6a4c93", alpha=0.12, zorder=0,
               label=f"predicted PFC↔DCQCN band ({lo:.1f}–{hi:.1f} MiB)")
    ax.axvline(lo, color="#6a4c93", ls=":", lw=1.0)
    ax.axvline(hi, color="#6a4c93", ls="--", lw=1.2)


def make_plots(summary: pd.DataFrame, outdir: Path,
               band: tuple[float, float] | None, topo_ok: bool) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    s = summary.sort_values("buffer_mb").copy()
    # Without topology/config (or without ns-3 outputs) whole metric families are
    # absent; fill them with NaN so every panel degrades to "skipped" instead of
    # raising a KeyError halfway through the sweep.
    for c in ("pfc_worst_link_pause_pct", "pfc_host_pause_max_pct", "pfc_bottleneck_pause_pct",
              "qlen_peak_over_kmax", "pfc_egress_ceiling_bytes", "n_concurrent_peak",
              "n_concurrent_mean",
              "qlen_peak_congested_port_bytes", "slow_mean_bulk", "slow_p99_bulk",
              "slow_max_bulk", "slow_cv_bulk", "kv_ready_max_ns", "sync_skew_ns",
              "pfc_thresh_bytes", "pfc_thresh_egress_equiv_bytes",
              "pfc_thresh_naive_bytes", "kmin_bytes", "kmax_bytes"):
        if c not in s.columns:
            s[c] = np.nan

    def save(fig, name):
        p = outdir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 1) REGIME ------------------------------------------------------------- #
    if s["pfc_worst_link_pause_pct"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        _series(ax, s, "pfc_worst_link_pause_pct", "Worst-link PFC pause (% of incast window)")
        _series(ax, s, "pfc_host_pause_max_pct", "Worst host PAUSE (% of incast window)",
                marker="v", linestyle="-.")
        ax2 = ax.twinx()
        for _v, grp in s.groupby("variant"):
            grp = grp.dropna(subset=["qlen_peak_over_kmax"]).sort_values("buffer_mb")
            if not grp.empty:
                ax2.plot(grp["buffer_mb"], grp["qlen_peak_over_kmax"], marker="s",
                         linestyle="--", color="#d98a00",
                         label="Peak egress on congested port / KMAX")
        ax2.axhline(1.0, color="#d98a00", ls=":", lw=1)
        _logx(ax, s)
        _flip_band(ax, band)
        ax.set_ylabel("PFC pause (% of incast window)")
        ax2.set_ylabel("Peak egress / KMAX  (>1 = ECN saturated)")
        ax.set_title("Congestion regime vs buffer\n"
                     "(pause>0 = lossless backpressure; peak/KMAX>1 = ECN governs)")
        ax.grid(True, alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        save(fig, "01_regime_vs_buffer.png")

    # 2) THE TRADE-OFF ------------------------------------------------------ #
    if s["slow_mean_bulk"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        _series(ax, s, "slow_mean_bulk", "Mean slowdown (incast)")
        _series(ax, s, "slow_p99_bulk", "p99 slowdown (incast)", marker="s", linestyle="--")
        _series(ax, s, "slow_max_bulk", "Max slowdown (incast)", marker="^", linestyle=":")
        _logx(ax, s)
        _flip_band(ax, band)
        pk = s["n_concurrent_peak"].dropna().unique()
        mn = s["n_concurrent_mean"].dropna()
        if len(mn):
            ax.axhline(mn.mean(), color="#2b8a3e", ls="-.", lw=1.4,
                       label=f"fair-share reference (mean concurrency ≈ {mn.mean():.0f})")
        if len(pk) == 1 and pk[0]:
            ax.axhline(pk[0], color="#2b8a3e", ls=":", lw=1.0,
                       label=f"peak concurrency = {pk[0]:g} (upper bound, not a floor)")
        ax.axhline(1.0, color="#b0b0b0", ls=":", lw=1,
                   label="slowdown = 1 (unreachable under incast)")
        ax.set_ylabel("Slowdown (fct / standalone_fct)")
        ax.set_title("Mean vs tail vs buffer\n"
                     "(standalone_fct assumes the flow owns the bottleneck → reference = concurrency)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "02_slowdown_mean_vs_tail.png")

    # 3) FAIRNESS ----------------------------------------------------------- #
    if s["slow_cv_bulk"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        _series(ax, s, "slow_cv_bulk", "CV of incast slowdown (unfairness)")
        _logx(ax, s)
        _flip_band(ax, band)
        ax.set_ylabel("CV = std/mean of slowdown")
        ax.set_title("Fairness vs buffer\n(higher CV = victim flows / HOL blocking = PFC signature)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "03_fairness_cv_vs_buffer.png")

    # 4) BARRIER ------------------------------------------------------------ #
    if s[["kv_ready_max_ns", "sync_skew_ns"]].notna().any().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        _series(ax, s, "kv_ready_max_ns", "Decode-start gate = max KV-ready", scale=1e-6)
        ax2 = ax.twinx()
        for _v, grp in s.groupby("variant"):
            grp = grp.dropna(subset=["sync_skew_ns"]).sort_values("buffer_mb")
            if not grp.empty:
                ax2.plot(grp["buffer_mb"], grp["sync_skew_ns"] * 1e-6, marker="s",
                         linestyle="--", color="#d1495b",
                         label="Sync skew (spread of KV arrivals on one rank)")
        _logx(ax, s)
        _flip_band(ax, band)
        ax.set_ylabel("Decode-start gate (ms)")
        ax2.set_ylabel("Sync skew (ms)")
        ax.set_title("Barrier metric vs buffer\n(what actually gates disaggregated decode)")
        ax.grid(True, alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        save(fig, "04_barrier_and_skew_vs_buffer.png")

    # 5) THE MECHANISM: threshold vs the ECN band --------------------------- #
    if topo_ok and s["pfc_thresh_bytes"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        _series(ax, s, "pfc_thresh_bytes", "PFC threshold (exact, ingress)",
                scale=1e-3, color="#1f77b4")
        _series(ax, s, "pfc_thresh_egress_equiv_bytes",
                "PFC threshold × F (egress-equivalent)", marker="D",
                scale=1e-3, color="#1f77b4", linestyle="--")
        _series(ax, s, "pfc_thresh_naive_bytes", "buffer/8 (naive)", marker="x",
                scale=1e-3, color="#b0b0b0", linestyle=":")
        _series(ax, s, "qlen_peak_congested_port_bytes", "Measured peak egress",
                marker="s", scale=1e-3, color="#d1495b")
        # The decisive test: peak egress riding at the PFC ceiling == the queue is
        # held by backpressure; below it == the rate control is what limits.
        _series(ax, s, "pfc_egress_ceiling_bytes",
                "PFC egress ceiling  Σ(reserve+thresh+headroom)", marker="*",
                scale=1e-3, color="#d1495b", linestyle="-.")
        kmin = s["kmin_bytes"].dropna().unique()
        kmax = s["kmax_bytes"].dropna().unique()
        if len(kmin) == 1:
            ax.axhline(kmin[0] / 1e3, color="#2b8a3e", ls=":", lw=1.2,
                       label=f"KMIN = {kmin[0]/1e3:g} kB")
        if len(kmax) == 1:
            ax.axhline(kmax[0] / 1e3, color="#2b8a3e", ls="--", lw=1.2,
                       label=f"KMAX = {kmax[0]/1e3:g} kB")
        _logx(ax, s)
        _flip_band(ax, band)
        ax.set_yscale("log")
        ax.set_ylabel("Bytes (kB, log)")
        ax.set_title("The mechanism: dynamic PFC threshold vs the ECN band\n"
                     "(threshold below KMIN → PFC fires before ECN marks)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7)
        save(fig, "05_threshold_vs_ecn_band.png")

    # 6) HEADLINE OVERLAY --------------------------------------------------- #
    if s[["slow_mean_bulk", "sync_skew_ns"]].notna().all(axis=None):
        # Min-max normalisation maps ANY range onto 0-1, so a 1% wiggle looks like
        # a full-scale trend. Refuse to normalise a series whose relative range is
        # below MIN_REL_RANGE: there is nothing there to plot.
        MIN_REL_RANGE = 0.05
        ranges = {}
        for col in ("slow_mean_bulk", "sync_skew_ns"):
            v = s[col].to_numpy(float)
            m = np.nanmean(v)
            ranges[col] = (np.nanmin(v), np.nanmax(v),
                           (np.nanmax(v) - np.nanmin(v)) / m if m else 0.0)
        if all(r[2] < MIN_REL_RANGE for r in ranges.values()):
            print(f"  - skip 06_aggregate_vs_barrier_overlay: both series vary by "
                  f"less than {MIN_REL_RANGE:.0%} across the sweep "
                  f"(slowdown {ranges['slow_mean_bulk'][0]:.2f}–{ranges['slow_mean_bulk'][1]:.2f}, "
                  f"skew {ranges['sync_skew_ns'][0]/1e6:.1f}–{ranges['sync_skew_ns'][1]/1e6:.1f} ms). "
                  f"Normalising that would render noise as signal.", file=sys.stderr)
            return written
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, grp in s.groupby("variant"):
            grp = grp.sort_values("buffer_mb")

            def _norm(col):
                v = grp[col].to_numpy(float)
                lo, hi = np.nanmin(v), np.nanmax(v)
                return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

            suff = "" if s["variant"].nunique() == 1 else f" [{variant}]"
            r = ranges["slow_mean_bulk"]
            ax.plot(grp["buffer_mb"], _norm("slow_mean_bulk"), marker="o",
                    label=f"Mean slowdown ({r[0]:.2f}–{r[1]:.2f}, {r[2]:.1%})" + suff)
            r = ranges["sync_skew_ns"]
            ax.plot(grp["buffer_mb"], _norm("sync_skew_ns"), marker="s", linestyle="--",
                    label=f"Sync skew ({r[0]/1e6:.0f}–{r[1]/1e6:.0f} ms, {r[2]:.1%})" + suff)
        _logx(ax, s)
        _flip_band(ax, band)
        ax.set_ylabel("Normalised (0–1)")
        ax.set_title("Aggregate vs barrier, normalised\n(divergence = the trade-off the tool reveals)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, "06_aggregate_vs_barrier_overlay.png")

    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("astra_root", nargs="?", default=DEFAULT_ASTRA_ROOT)
    ap.add_argument("--ns3-root", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--topology", default=DEFAULT_TOPOLOGY,
                    help="physical_topology.txt; a path, or a template with {tag}.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="ns-3 config.txt; a path, or a template with {tag}.")
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--bulk-mb", type=float, default=1.0,
                    help="Flow-size threshold (MB) for a flow to count as bulk KV/PP.")
    ap.add_argument("--decode-nodes", default=None,
                    help="Comma-separated decode host node ids (KV incast dsts). "
                         "If omitted, inferred as the fabric incast receivers.")
    ap.add_argument("--headroom-factor", type=int, default=DEFAULT_HEADROOM_FACTOR,
                    help="common.h::headroom_factor (default 3; set it if you "
                         "override HEADROOM_FACTOR in config.txt).")
    ap.add_argument("--print-patch", action="store_true",
                    help="Print the ns-3 diff that adds qIndex to pfc.txt and exit.")
    args = ap.parse_args(argv)

    if args.print_patch:
        print(PFC_QIDX_PATCH)
        return 0

    astra_root = Path(args.astra_root)
    if not astra_root.is_dir():
        print(f"ERROR: astra_root not found: {astra_root}", file=sys.stderr)
        return 2
    ns3_root = resolve_ns3_root(astra_root, args.ns3_root)
    if ns3_root is None:
        print("WARNING: no ns-3 root resolved — regime/tail/barrier metrics will be "
              "empty. Pass --ns3-root.", file=sys.stderr)

    outdir = Path(args.out) if args.out else (Path(DEFAULT_OUTDIR) if DEFAULT_OUTDIR
                                              else astra_root / "buffer_analysis")
    decode_nodes = [int(x) for x in args.decode_nodes.split(",")] if args.decode_nodes else []
    bulk_bytes = int(args.bulk_mb * 1024 * 1024)

    outdir_resolved = outdir.resolve()
    run_dirs = sorted(p for p in astra_root.iterdir()
                      if p.is_dir() and p.name != "buffer_analysis"
                      and p.resolve() != outdir_resolved)
    if not run_dirs:
        print(f"ERROR: no run sub-directories under {astra_root}", file=sys.stderr)
        return 2

    print(f"Scanning {len(run_dirs)} run dirs under:\n  {astra_root}")
    print(f"ns-3 root: {ns3_root if ns3_root else '(none)'}\n")

    rows: list[dict] = []
    per_node_frames: list[pd.DataFrame] = []
    topo_cache: dict[str, Topology] = {}
    unresolved: list[str] = []
    band: tuple[float, float] | None = None
    topo_ok = False

    for d in run_dirs:
        buf = parse_buffer(d.name)
        if buf is None:
            print(f"  - skip {d.name!r}: no 'buf<num>' token", file=sys.stderr)
            continue
        adf = load_astra_run(d, args.pattern)
        if adf is None:
            print(f"  - skip {d.name!r}: no readable CSVs", file=sys.stderr)
            continue

        summ = summarise_astra(adf)
        ns3_dir = find_ns3_run(ns3_root, d.name)

        # -- topology + config -------------------------------------------- #
        roots = [ns3_root, astra_root] if ns3_root else [astra_root]
        tpath = find_aux(args.topology, d.name, ns3_dir, roots, "physical_topology.txt")
        cpath = find_aux(args.config, d.name, ns3_dir, roots, "config.txt")
        topo = None
        if tpath:
            key = f"{tpath}|{args.headroom_factor}"
            if key not in topo_cache:
                try:
                    topo_cache[key] = parse_topology(tpath, args.headroom_factor)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! cannot parse topology {tpath}: {exc}", file=sys.stderr)
                    topo_cache[key] = None
            topo = topo_cache[key]
        if tpath is None:
            print(f"  ! {d.name}: physical_topology.txt NOT RESOLVED — no PFC "
                  f"threshold, no ECN band, no fabric/direct flow filter. The "
                  f"slowdown statistics will be a MIXTURE of congested KV flows and "
                  f"uncongested direct-link collectives. Pass --topology.",
                  file=sys.stderr)
            unresolved.append(d.name)
        cfg = parse_ns3_config(cpath) if cpath else None
        if cpath is None:
            print(f"  ! {d.name}: config.txt NOT RESOLVED — KMIN/KMAX unknown, "
                  f"regime prediction unavailable. Pass --config.", file=sys.stderr)
            unresolved.append(d.name)
        if cfg and cfg.buffer_mb is not None and abs(cfg.buffer_mb - buf) > 1e-6:
            print(f"  ! {d.name}: BUFFER_SIZE={cfg.buffer_mb} in config.txt but "
                  f"'buf{buf:g}' in the dir name — trusting config.txt", file=sys.stderr)
            buf = cfg.buffer_mb
        if topo is not None:
            topo_ok = True
            if topo.ecmp_pairs:
                print(f"  ! ECMP ties present ({len(topo.ecmp_pairs)} node/host pairs): "
                      f"runtime paths are hash-chosen and per-flow path attribution "
                      f"is approximate", file=sys.stderr)

        # -- ns-3 outputs -------------------------------------------------- #
        fct = pfc = qlen = None
        if ns3_dir is not None:
            fct = parse_fct(ns3_dir / "fct.txt")
            if fct is not None:
                fct = annotate_flows(fct, topo, bulk_bytes)
            pfc = parse_pfc(ns3_dir / "pfc.txt")
            qlen = parse_qlen(ns3_dir / "qlen.txt")
        summ.update(summarise_ns3(fct, pfc, qlen, topo, cfg, buf, decode_nodes))

        summ["run_dir"] = d.name
        summ["variant"] = variant_key(d.name)
        summ["buffer_mb"] = buf
        summ["ns3_dir"] = str(ns3_dir) if ns3_dir else ""
        summ["cc_mode"] = cfg.cc_mode if cfg else np.nan
        if cfg and cfg.cc_mode == 12:
            print(f"  ! {d.name}: CC_MODE 12 has no handler in any RdmaHw/SwitchNode "
                  f"branch — this run is PFC-only lossless with NO rate-based CC",
                  file=sys.stderr)
        summ["regime_pred"] = regime_predicted(summ)
        summ["regime_obs"] = regime_observed(summ)
        rows.append(summ)

        # The flip band is a property of the topology + ECN map + fan-in, not of
        # the buffer, so it is computed once from the first run that has them all.
        if (band is None and topo is not None
                and summ.get("kmin_bytes") == summ.get("kmin_bytes")
                and summ.get("kmax_bytes") == summ.get("kmax_bytes")
                and summ.get("congested_link")):
            F = summ.get("fanin_ports") or 1
            sw = int(str(summ["congested_link"]).split("->")[0])
            band = flip_buffers(topo, sw, int(summ["congested_port"]),
                                summ["kmin_bytes"], summ["kmax_bytes"], float(F))

        pn = (adf.groupby(["sys_id", "op_class"])
              .agg(count=("name", "size"), total_bytes=("comm_size", "sum"),
                   total_busy_ns=("duration", "sum")).reset_index())
        pn.insert(0, "buffer_mb", buf)
        pn.insert(1, "variant", summ["variant"])
        per_node_frames.append(pn)

        print(f"  + {d.name:<28} buf={buf:<5g} pred={summ['regime_pred']:<5} "
              f"obs={summ['regime_obs']:<5} pause={summ.get('pfc_worst_link_pause_pct', float('nan')):6.2f}% "
              f"meanSD={summ.get('slow_mean_bulk', float('nan')):.2f} "
              f"p99SD={summ.get('slow_p99_bulk', float('nan')):.2f} "
              f"Fp={summ.get('fanin_ports', float('nan'))} N={summ.get('n_concurrent_peak', float('nan'))}")

    if not rows:
        print("ERROR: no valid runs parsed.", file=sys.stderr)
        return 2

    summary = pd.DataFrame(rows).sort_values(["variant", "buffer_mb"]).reset_index(drop=True)
    front = ["run_dir", "variant", "buffer_mb", "regime_pred", "regime_obs",
             "congested_link", "congested_rate_gbps", "fanin_ports",
             "n_concurrent_peak", "n_concurrent_mean",
             "pfc_thresh_bytes", "pfc_thresh_egress_equiv_bytes",
             "pfc_thresh_naive_bytes", "kmin_bytes", "kmax_bytes",
             "qlen_peak_congested_port_bytes", "qlen_peak_over_kmax",
             "pfc_bottleneck_pause_pct", "pfc_worst_link_pause_pct",
             "pfc_worst_link_device", "pfc_host_pause_max_pct", "pfc_qidx",
             "slow_mean_bulk", "slow_p99_bulk", "slow_cv_bulk", "slow_mean_over_fairshare",
             "kv_ready_max_ns", "sync_skew_ns", "makespan_ms", "kv_bound_ratio"]
    summary = summary[[c for c in front if c in summary.columns] +
                      [c for c in summary.columns if c not in front]]

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    per_node = pd.concat(per_node_frames, ignore_index=True)
    per_node_path = outdir / "per_node.csv"
    per_node.to_csv(per_node_path, index=False)

    plots = make_plots(summary, outdir, band, topo_ok)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    report = [c for c in ["buffer_mb", "regime_pred", "regime_obs", "fanin_ports",
                          "pfc_thresh_bytes", "pfc_thresh_egress_equiv_bytes",
                          "qlen_peak_congested_port_bytes", "pfc_egress_ceiling_bytes",
                          "pfc_bottleneck_pause_pct", "slow_mean_bulk", "slow_p99_bulk", "slow_cv_bulk",
                          "sync_skew_ns"] if c in summary.columns]
    print("\n================ BUFFER SWEEP SUMMARY ================")
    print(summary[report].to_string(index=False))
    if band:
        print(f"\nPredicted regime band: PFC below {band[0]:.2f} MiB, "
              f"DCQCN above {band[1]:.2f} MiB (F*threshold crossing KMIN and KMAX).")
    if "pfc_qidx" in summary and (summary["pfc_qidx"] == "MISSING").any():
        print("\nWARNING: pfc.txt has no qIndex column. With qos-enabled the PAUSE/"
              "RESUME sequences of different priority groups interleave on the same "
              "ifindex and the pause totals are unreliable. Run with --print-patch "
              "for the three-line ns-3 diff that fixes it.")
    if unresolved:
        print(f"\nWARNING: {len(set(unresolved))} run(s) ran WITHOUT topology and/or "
              f"config. Those rows are degraded — see the per-run messages above.")
    print("\nWrote:")
    for p in [summary_path, per_node_path, *plots]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())