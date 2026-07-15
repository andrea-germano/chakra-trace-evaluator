#!/usr/bin/env python3
"""
utils.ns3 — readers for the ns-3 output files (fct.txt, pfc.txt, qlen.txt)

One function per format, each documented against the code that produces it. No
interpretation happens here: these return the data, and each analyzer decides what
it means. That split is why the buffer sweep and the bandwidth sweep can share a
reader without sharing a question.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Addressing
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def ip_to_node(tok: str) -> int:
    """common.h:
        node_id_to_ip(id) = 0x0b000001 + (id/256)*0x10000 + (id%256)*0x100
        ip_to_node_id(ip) = (ip >> 8) & 0xffff
    entry.h prints the raw IP with %08x, so node 5 appears as '0b000501'."""
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


# --------------------------------------------------------------------------- #
# fct.txt
# --------------------------------------------------------------------------- #
def read_fct(path: Path) -> pd.DataFrame | None:
    """entry.h::qp_finish_print_log
        %08x %08x %u %u %lu %lu %lu %lu
        sip dip sport dport size(B) start(ns) fct(ns) standalone_fct(ns)

    standalone_fct = base_rtt + total_bytes*8e9/pairBw, where pairBw is the MIN
    link rate along the BFS path. So slowdown = fct/standalone_fct normalises
    against "this flow alone on its bottleneck" -- meaning N flows sharing that
    bottleneck fairly give slowdown ~ N. The floor is the concurrency, not 1."""
    if not path.is_file():
        return None
    rows = []
    for line in path.open():
        p = line.split()
        if len(p) < 8:
            continue
        try:
            rows.append((ip_to_node(p[0]), ip_to_node(p[1]),
                         int(p[4]), int(p[5]), int(p[6]), int(p[7])))
        except ValueError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["src", "dst", "size", "start", "fct", "sfct"])
    df["arrival"] = df["start"] + df["fct"]
    df["slowdown"] = np.where(df["sfct"] > 0, df["fct"] / df["sfct"], np.nan)
    return df


# --------------------------------------------------------------------------- #
# pfc.txt
# --------------------------------------------------------------------------- #
@dataclass
class PfcLog:
    """common.h::get_pfc -> time node node_type ifindex type [qIndex]
       node_type 0=host 1=switch ; type 0=RESUME 1=PAUSE

    Two things the format hides:

    1. The trace fires in QbbNetDevice::Receive, on the device that RECEIVES the
       pause frame -- the VICTIM being paused, not the device doing the pausing.
       A switch row means "this switch's egress port was held by downstream".

    2. m_tracePfc fires per qIndex, but get_pfc does not print it. With
       qos-enabled the PAUSE/RESUME sequences of different priority groups
       interleave on one ifindex and any state machine keyed on (node, ifindex)
       mis-pairs them. `qidx` is read from an optional 6th column; when absent,
       `qidx_state` reports MISSING and the totals are not a measurement.
       See PFC_QIDX_PATCH for the three-line ns-3 diff that emits it."""
    n_events: int = 0
    t_max: int = 0
    qidx_state: str = "n/a"                 # present | MISSING | n/a
    events: dict[tuple, list] = field(default_factory=dict)

    def pause_totals(self, clamp_to: int) -> tuple[dict[tuple, int], int]:
        """Total paused time per (node, node_type, ifindex, qIndex).

        There is no pause timeout on the receiving side of this fork (Receive
        only sets m_paused[qIndex]; nothing schedules a PauseFinish), so
        PAUSE/RESUME strictly alternate per queue and an unclosed PAUSE really
        does last to the end of the run -- clamp to the run end, not to the last
        PFC event, which may be far earlier."""
        totals, unclosed = {}, 0
        for key, events in self.events.items():
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

    def pause_per_device(self, clamp_to: int) -> dict[tuple[int, int, int], int]:
        """Collapsed to (node, node_type, ifindex). Queues of one device overlap
        in time, so this is a sum over queues and can exceed the device's own
        paused wall-clock -- but summing across DEVICES and dividing by a time
        window is what produces nonsense like "283% paused", so callers must take
        a max or a per-device value, never a total."""
        per: dict[tuple[int, int, int], int] = defaultdict(int)
        totals, _ = self.pause_totals(clamp_to)
        for (node, ntype, ifidx, _q), v in totals.items():
            per[(node, ntype, ifidx)] += v
        return dict(per)


def read_pfc(path: Path) -> PfcLog | None:
    if not path.is_file():
        return None
    ev: dict[tuple, list] = defaultdict(list)
    n, t_max, has_q = 0, 0, True
    for line in path.open():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            t, node, ntype, ifidx, typ = (int(p[i]) for i in range(5))
            qidx = int(p[5]) if len(p) > 5 else -1
        except ValueError:
            continue
        has_q &= qidx >= 0
        n += 1
        t_max = max(t_max, t)
        ev[(node, ntype, ifidx, qidx)].append((t, typ))
    # An existing-but-empty pfc.txt is a measurement (zero PAUSE = DCQCN), not a
    # missing file, and has no qIndex question to answer.
    return PfcLog(n_events=n, t_max=t_max, events=dict(ev),
                  qidx_state="n/a" if n == 0 else ("present" if has_q else "MISSING"))


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
qIndex is appended as a 6th column, so parsers keying on p[0..4] keep working.
"""


# --------------------------------------------------------------------------- #
# qlen.txt
# --------------------------------------------------------------------------- #
@dataclass
class QlenLog:
    """common.h::monitor_buffer
        time <t> <switch_id> j <port> <bytes> j <port> <bytes> ...

    <bytes> is sum_k egress_bytes[port][k] for ONE port, and a port is emitted
    only while its queue is >= 1000 B. So this observes PER-EGRESS-PORT
    occupancy: precisely the quantity ShouldSendCN() compares against
    kmin/kmax[ifindex]. It is NOT the shared pool -- that is ingress-side
    accounting (shared_used_bytes) and is not observable from this file at all.
    Summing across ports would destroy the only directly comparable quantity.

    QLEN_MON_END is dead code in common.h (parsed, never used: monitor_buffer
    reschedules unconditionally), so this always covers the run from
    QLEN_MON_START at a fixed 100 ns interval."""
    samples: int = 0
    port_max: dict[tuple[int, int], int] = field(default_factory=dict)
    port_mean: dict[tuple[int, int], float] = field(default_factory=dict)
    switch_total_max: dict[int, int] = field(default_factory=dict)

    def busiest_port(self) -> tuple[int, int] | None:
        if not self.port_max:
            return None
        return max(self.port_max.items(), key=lambda kv: kv[1])[0]


def read_qlen(path: Path) -> QlenLog | None:
    if not path.is_file():
        return None
    pmax: dict[tuple[int, int], int] = defaultdict(int)
    psum: dict[tuple[int, int], int] = defaultdict(int)
    pcnt: dict[tuple[int, int], int] = defaultdict(int)
    swmax: dict[int, int] = defaultdict(int)
    samples = 0
    for line in path.open():
        p = line.split()
        if len(p) < 3 or p[0] != "time":
            continue
        try:
            sw = int(p[2])
        except ValueError:
            continue
        i, total = 3, 0
        while i < len(p):
            # i+2 < len(p), not <=: a truncated final line from a killed run
            # must not raise here.
            if p[i] == "j" and i + 2 < len(p):
                try:
                    port, b = int(p[i + 1]), int(p[i + 2])
                except ValueError:
                    i += 1
                    continue
                key = (sw, port)
                pcnt[key] += 1
                psum[key] += b
                pmax[key] = max(pmax[key], b)
                total += b
                i += 3
            else:
                i += 1
        samples += 1
        swmax[sw] = max(swmax[sw], total)
    if samples == 0:
        return QlenLog()
    return QlenLog(samples=samples, port_max=dict(pmax),
                   port_mean={k: psum[k] / pcnt[k] for k in pcnt},
                   switch_total_max=dict(swmax))