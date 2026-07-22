#!/usr/bin/env python3
"""
utils.ns3 — readers for the ns-3 output files (fct.txt, pfc.txt, qlen.txt)

One function per format, each documented against the code that produces it. No
interpretation happens here: these return the data, and each analyzer decides what
it means. That split is why the buffer sweep and the bandwidth sweep can share a
reader without sharing a question.
"""

from __future__ import annotations

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
    # `slow_min` replaces a `fct/sfct < 0.999` counter that fired on 1694 of the
    # 2994 flows of the T1 reference run and would have reported the columns as
    # misparsed. They are not: entry.h computes standalone_fct over total_bytes
    # (payload + per-packet headers) while the size column is the payload alone,
    # so an uncongested flow legitimately lands a percent or two below 1. A wrong
    # column layout does not miss by 1.3% -- it misses by orders of magnitude.
    # So: report the minimum, and flag only what physics cannot explain.
    diag = dict(raw_lines=0, skipped_short=0, skipped_badnum=0, sfct_nonpos=0,
                slow_min=float("inf"), slow_lt09=0, ncols=None, sample=None,
                ncol_hist=defaultdict(int))
    for line in path.open():
        if not line.strip():
            continue
        diag["raw_lines"] += 1
        p = line.split()
        diag["ncol_hist"][len(p)] += 1
        if diag["sample"] is None:
            diag["sample"], diag["ncols"] = line.rstrip("\n"), len(p)
        if len(p) < 8:
            diag["skipped_short"] += 1
            continue
        try:
            size, start, fct, sfct = int(p[4]), int(p[5]), int(p[6]), int(p[7])
        except ValueError:
            diag["skipped_badnum"] += 1
            continue
        if sfct <= 0:
            diag["sfct_nonpos"] += 1
        else:
            diag["slow_min"] = min(diag["slow_min"], fct / sfct)
            if fct / sfct < 0.9:
                diag["slow_lt09"] += 1
        rows.append((ip_to_node(p[0]), ip_to_node(p[1]), int(p[2]), int(p[3]),
                     size, start, fct, sfct))
    if not rows:
        return None
    # sport/dport are kept, not dropped. They are the ONLY thing that tells two
    # concurrent QPs on the same (src, dst) apart, and without them a pair of flows
    # sharing one NIC round-robin looks identical to one slow flow. On the T1
    # reference run that is exactly what happens: rank 3 posts its 40 MB all-reduce
    # chunks in PAIRS 37,570 ns apart while rank 2 posts them singly, so
    # RdmaEgressQueue halves the rate and fct goes 72,902 -> 107,234 on a dedicated
    # 4800 Gbps link that cannot queue. Diagnosing it needed the ports, and getting
    # them meant re-parsing the file by hand outside this module.
    df = pd.DataFrame(rows, columns=["src", "dst", "sport", "dport",
                                     "size", "start", "fct", "sfct"])
    df["arrival"] = df["start"] + df["fct"]
    df["slowdown"] = np.where(df["sfct"] > 0, df["fct"] / df["sfct"], np.nan)
    # Parse diagnostics ride along: a wrong column layout must be visible before
    # any number computed from it is trusted.
    diag["ncol_hist"] = dict(diag["ncol_hist"])
    df.attrs["diagnostics"] = diag
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

    def pause_intervals_flagged(self, clamp_to: int) -> dict[tuple, list[tuple[int, int, bool]]]:
        """Closed intervals as `pause_totals` pairs them, each flagged suspect or
        not. This is the honest answer to "is pfc.txt without qIndex usable".

        Without qIndex, PAUSE and RESUME of different queues land on the same
        (node, ifindex) key and this state machine mis-pairs them -- but only
        where the two queues' sequences actually interleave, which is a property
        of the run, not of the format. An interleaving leaves a fingerprint:
        two same-type events in a row. So instead of declaring the whole file
        unusable a priori, flag the intervals adjacent to a fingerprint and let
        the caller report a bound: [total - suspect, total] brackets the truth.

        Flagged when:
          * a second PAUSE arrives while one is open (its RESUME will close the
            wrong interval, stretching it to the later queue's release), or
          * a RESUME arrives with nothing open (a PAUSE was swallowed earlier)."""
        out: dict[tuple, list] = {}
        for key, events in self.events.items():
            events.sort()
            iv: list[tuple[int, int, bool]] = []
            start, suspect = None, False
            for t, typ in events:
                if typ == 1:
                    if start is None:
                        start, suspect = t, False
                    else:
                        suspect = True
                elif start is not None:
                    iv.append((start, t, suspect))
                    start = None
                elif iv:
                    iv[-1] = (iv[-1][0], iv[-1][1], True)
            if start is not None:
                iv.append((start, max(clamp_to, start), True))
            out[key] = iv
        return out

    def pause_intervals(self, clamp_to: int) -> dict[tuple, list[tuple[int, int]]]:
        """Closed [start, end] pause intervals per (node, node_type, ifindex, qIndex),
        for timelines. Same state machine as pause_totals."""
        out: dict[tuple, list] = {}
        for key, events in self.events.items():
            events.sort()
            iv, start = [], None
            for t, typ in events:
                if typ == 1 and start is None:
                    start = t
                elif typ == 0 and start is not None:
                    iv.append((start, t))
                    start = None
            if start is not None:
                iv.append((start, max(clamp_to, start)))
            out[key] = iv
        return out

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
    port_count: dict[tuple[int, int], int] = field(default_factory=dict)
    switch_total_max: dict[int, int] = field(default_factory=dict)
    t_min: int = 0
    t_max: int = 0
    # Filled only when read_qlen(..., series=True): the raw per-sample time series,
    # as numpy arrays (ts, bytes). Arrays rather than Python lists so a 10M-sample
    # run does not pay ~28 bytes per int. Consumers already np.asarray them.
    port_series: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    switch_series: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    switch_count: dict[int, int] = field(default_factory=dict)


def read_qlen(path: Path, series: bool = False) -> QlenLog | None:
    """Read qlen.txt into a QlenLog. `series=True` also retains the per-sample
    time series (as numpy arrays), which plots need and a sweep does not."""
    if not path.is_file():
        return None
    return _read_qlen_fast(path, series)


# --------------------------------------------------------------------------- #
# Fast qlen parser (numba, a hard dependency). The format is
# `time <t> <sw> [j <port> <bytes>]*`; a compiled single-pass byte scan reads it
# an order of magnitude faster than a Python loop and with no temporary table --
# the same logic, just JIT-compiled and cached to disk.
# --------------------------------------------------------------------------- #
from numba import njit
from numba.core import types
from numba.typed import Dict

_PKEY = 1 << 20                         # pack (sw, port) -> sw * _PKEY + port


def _unpack(k: int) -> tuple[int, int]:
    return (int(k // _PKEY), int(k % _PKEY))


@njit(cache=True)
def _qlen_scan_scalar(buf):
    """One byte-pass -> the scalar aggregates only. Keyed sw*_PKEY+port (ports)
    and sw (switches); no per-sample arrays, so RAM stays negligible -- this is
    the path the cross-model compares use over many huge files."""
    n = buf.size
    pmax = Dict.empty(types.int64, types.int64)
    psum = Dict.empty(types.int64, types.int64)
    pcnt = Dict.empty(types.int64, types.int64)
    swmax = Dict.empty(types.int64, types.int64)
    swcnt = Dict.empty(types.int64, types.int64)
    samples = 0
    tmin = -1
    tmax = 0
    i = 0
    while i < n:
        while i < n and (buf[i] == 32 or buf[i] == 10):
            i += 1
        if i >= n:
            break
        if not (i + 4 <= n and buf[i] == 116 and buf[i + 1] == 105
                and buf[i + 2] == 109 and buf[i + 3] == 101):   # not 'time'
            while i < n and buf[i] != 10:
                i += 1
            continue
        i += 4
        while i < n and buf[i] == 32:
            i += 1
        t = 0
        while i < n and 48 <= buf[i] <= 57:
            t = t * 10 + (buf[i] - 48)
            i += 1
        while i < n and buf[i] == 32:
            i += 1
        sw = 0
        while i < n and 48 <= buf[i] <= 57:
            sw = sw * 10 + (buf[i] - 48)
            i += 1
        total = 0
        while i < n and buf[i] != 10:
            while i < n and buf[i] == 32:
                i += 1
            if i < n and buf[i] == 106:                         # 'j'
                i += 1
                while i < n and buf[i] == 32:
                    i += 1
                port = 0
                got_p = False
                while i < n and 48 <= buf[i] <= 57:
                    port = port * 10 + (buf[i] - 48)
                    i += 1
                    got_p = True
                while i < n and buf[i] == 32:
                    i += 1
                b = 0
                got_b = False
                while i < n and 48 <= buf[i] <= 57:
                    b = b * 10 + (buf[i] - 48)
                    i += 1
                    got_b = True
                if got_p and got_b:
                    key = sw * _PKEY + port
                    if key in pmax:
                        if b > pmax[key]:
                            pmax[key] = b
                        psum[key] += b
                        pcnt[key] += 1
                    else:
                        pmax[key] = b
                        psum[key] = b
                        pcnt[key] = 1
                    total += b
            elif i < n and buf[i] != 10:
                while i < n and buf[i] != 32 and buf[i] != 10:
                    i += 1
        if i < n and buf[i] == 10:
            i += 1
        samples += 1
        if tmin < 0 or t < tmin:
            tmin = t
        if t > tmax:
            tmax = t
        if sw in swmax:
            if total > swmax[sw]:
                swmax[sw] = total
            swcnt[sw] += 1
        else:
            swmax[sw] = total
            swcnt[sw] = 1
    return pmax, psum, pcnt, swmax, swcnt, samples, tmin, tmax

@njit(cache=True)
def _qlen_scan_count(buf):
    """One byte-pass -> (n_rows, n_entries), the sizes the fill pass needs."""
    n = buf.size
    nrows = 0
    nent = 0
    i = 0
    while i < n:
        while i < n and (buf[i] == 32 or buf[i] == 10):
            i += 1
        if i >= n:
            break
        if not (i + 4 <= n and buf[i] == 116 and buf[i + 1] == 105
                and buf[i + 2] == 109 and buf[i + 3] == 101):
            while i < n and buf[i] != 10:
                i += 1
            continue
        i += 4
        while i < n and buf[i] == 32:
            i += 1
        while i < n and 48 <= buf[i] <= 57:
            i += 1
        while i < n and buf[i] == 32:
            i += 1
        while i < n and 48 <= buf[i] <= 57:
            i += 1
        while i < n and buf[i] != 10:
            while i < n and buf[i] == 32:
                i += 1
            if i < n and buf[i] == 106:
                i += 1
                while i < n and buf[i] == 32:
                    i += 1
                got_p = False
                while i < n and 48 <= buf[i] <= 57:
                    i += 1
                    got_p = True
                while i < n and buf[i] == 32:
                    i += 1
                got_b = False
                while i < n and 48 <= buf[i] <= 57:
                    i += 1
                    got_b = True
                if got_p and got_b:
                    nent += 1
            elif i < n and buf[i] != 10:
                while i < n and buf[i] != 32 and buf[i] != 10:
                    i += 1
        if i < n and buf[i] == 10:
            i += 1
        nrows += 1
    return nrows, nent

@njit(cache=True)
def _qlen_scan_fill(buf, sw_row, t_row, tot_row, sw_j, port_j, ts_j, b_j):
    """One byte-pass -> fill the preallocated row- and entry-level arrays, in
    file order (so each key's samples come out chronological)."""
    n = buf.size
    r = 0
    e = 0
    i = 0
    while i < n:
        while i < n and (buf[i] == 32 or buf[i] == 10):
            i += 1
        if i >= n:
            break
        if not (i + 4 <= n and buf[i] == 116 and buf[i + 1] == 105
                and buf[i + 2] == 109 and buf[i + 3] == 101):
            while i < n and buf[i] != 10:
                i += 1
            continue
        i += 4
        while i < n and buf[i] == 32:
            i += 1
        t = 0
        while i < n and 48 <= buf[i] <= 57:
            t = t * 10 + (buf[i] - 48)
            i += 1
        while i < n and buf[i] == 32:
            i += 1
        sw = 0
        while i < n and 48 <= buf[i] <= 57:
            sw = sw * 10 + (buf[i] - 48)
            i += 1
        total = 0
        while i < n and buf[i] != 10:
            while i < n and buf[i] == 32:
                i += 1
            if i < n and buf[i] == 106:
                i += 1
                while i < n and buf[i] == 32:
                    i += 1
                port = 0
                got_p = False
                while i < n and 48 <= buf[i] <= 57:
                    port = port * 10 + (buf[i] - 48)
                    i += 1
                    got_p = True
                while i < n and buf[i] == 32:
                    i += 1
                b = 0
                got_b = False
                while i < n and 48 <= buf[i] <= 57:
                    b = b * 10 + (buf[i] - 48)
                    i += 1
                    got_b = True
                if got_p and got_b:
                    sw_j[e] = sw
                    port_j[e] = port
                    ts_j[e] = t
                    b_j[e] = b
                    e += 1
                    total += b
            elif i < n and buf[i] != 10:
                while i < n and buf[i] != 32 and buf[i] != 10:
                    i += 1
        if i < n and buf[i] == 10:
            i += 1
        sw_row[r] = sw
        t_row[r] = t
        tot_row[r] = total
        r += 1


def _read_qlen_fast(path: Path, series: bool) -> QlenLog:
    """numba byte-scan of qlen.txt. Scalar-only (`series=False`) accumulates in
    compiled dicts (tiny RAM); `series=True` sizes the per-sample arrays with a
    count pass, then fills them, and numpy groups them into the per-key series.
    The file bytes are read once into an owned array (freed before the series
    aggregation); an empty file yields an empty QlenLog."""
    buf = np.fromfile(path, dtype=np.uint8)
    if not series:
        pmax, psum, pcnt, swmax, swcnt, samples, tmin, tmax = _qlen_scan_scalar(buf)
        if samples == 0:
            return QlenLog()
        return QlenLog(
            samples=int(samples),
            port_max={_unpack(k): int(v) for k, v in pmax.items()},
            port_mean={_unpack(k): psum[k] / pcnt[k] for k in pcnt},
            port_count={_unpack(k): int(v) for k, v in pcnt.items()},
            switch_total_max={int(k): int(v) for k, v in swmax.items()},
            switch_count={int(k): int(v) for k, v in swcnt.items()},
            t_min=int(tmin if tmin > 0 else 0), t_max=int(tmax))

    n_rows, n_ent = _qlen_scan_count(buf)
    if n_rows == 0:
        return QlenLog()
    sw_row = np.empty(n_rows, np.int64)
    t_row = np.empty(n_rows, np.int64)
    tot_row = np.empty(n_rows, np.int64)
    sw_j = np.empty(n_ent, np.int64)
    port_j = np.empty(n_ent, np.int64)
    ts_j = np.empty(n_ent, np.int64)
    b_j = np.empty(n_ent, np.int64)
    _qlen_scan_fill(buf, sw_row, t_row, tot_row, sw_j, port_j, ts_j, b_j)
    del buf                             # file bytes no longer needed; free before aggregating

    # aggregate the flat arrays with numpy/pandas
    if n_ent:
        dfp = pd.DataFrame({"sw": sw_j, "port": port_j, "b": b_j})
        gp = dfp.groupby(["sw", "port"])["b"]
        agg = gp.agg(["max", "sum", "count"])
        port_max = {(int(s), int(p)): int(row["max"]) for (s, p), row in agg.iterrows()}
        port_count = {(int(s), int(p)): int(row["count"]) for (s, p), row in agg.iterrows()}
        port_mean = {(int(s), int(p)): row["sum"] / row["count"]
                     for (s, p), row in agg.iterrows()}
        port_series = {(int(s), int(p)): (ts_j[idx], b_j[idx])
                       for (s, p), idx in dfp.groupby(["sw", "port"]).indices.items()}
    else:
        port_max = port_count = port_mean = port_series = {}
    dfr = pd.DataFrame({"sw": sw_row, "tot": tot_row})
    sagg = dfr.groupby("sw")["tot"].agg(["max", "count"])
    switch_total_max = {int(s): int(row["max"]) for s, row in sagg.iterrows()}
    switch_count = {int(s): int(row["count"]) for s, row in sagg.iterrows()}
    switch_series = {int(s): (t_row[idx], tot_row[idx])
                     for s, idx in dfr.groupby("sw").indices.items()}

    return QlenLog(
        samples=int(n_rows), port_max=port_max, port_mean=port_mean,
        port_count=port_count, switch_total_max=switch_total_max,
        switch_count=switch_count, t_min=int(t_row.min()), t_max=int(t_row.max()),
        port_series=port_series, switch_series=switch_series)
