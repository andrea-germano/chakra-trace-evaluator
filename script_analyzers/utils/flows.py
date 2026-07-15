#!/usr/bin/env python3
"""
utils.flows — what an fct.txt row means once you know the topology.

fct.txt is a flat list of finished RDMA queue pairs. Without the topology it is
impossible to tell a congested KV transfer from a tensor-parallel all-reduce
riding a dedicated host-to-host link -- and mixing them is not a rounding error,
it is a **bimodal mixture** whose mean and CV describe the mixing ratio rather
than the fabric. On the reference run that produced a mean slowdown of 8.1 and a
CV of 1.74, both of which sat perfectly flat across the whole buffer sweep: the
mixture proportion is a constant of the workload, so the statistics could not
move no matter what the fabric did. Filtered to the incast alone, the same run
gives CV 0.17.

The taxonomy, used verbatim by every analyzer:

    bulk     size >= a threshold          a real transfer, not an ACK
    fabric   hops > 1                     traverses at least one switch
    direct   hops == 1                    host-to-host link: never congested, so
                                          slowdown ~1 by construction
    incast   bulk AND fabric              the population every *_incast statistic
                                          is computed over

What slowdown means
--------------------------------------------------------------------------------
entry.h computes ``standalone_fct = base_rtt + total_bytes * 8e9 / pairBw`` with
pairBw = the MIN link rate along the BFS path, i.e. "this flow alone owning its
bottleneck". N flows sharing that bottleneck fairly therefore give slowdown ~ N.
The reference is the concurrency, never 1.

(Note that ``total_bytes`` there includes the per-packet header overhead while the
size column is the payload alone, so size*8/sfct recovers pairBw only to within a
percent or so. The ratio fct/standalone_fct is unaffected: it is a ratio of two
times.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fabric import Bottleneck, Topology


def annotate(fct: pd.DataFrame, topo: Topology | None, bulk_bytes: int) -> pd.DataFrame:
    """Attach hops / path / bulk / fabric / incast.

    With no topology every flow is optimistically 'fabric' and nothing is
    filtered: that is the degraded mode, and callers are expected to say so out
    loud rather than quietly report a mixture."""
    f = fct.copy()
    f["bulk"] = f["size"] >= bulk_bytes
    if topo is None:
        f["hops"] = np.nan
        f["path"] = None
        f["fabric"] = True
    else:
        f["hops"] = [topo.dist.get(s, {}).get(d, np.nan) for s, d in zip(f["src"], f["dst"])]
        f["path"] = [topo.path(s, d) for s, d in zip(f["src"], f["dst"])]
        f["fabric"] = f["hops"] > 1
    f["incast"] = f["bulk"] & f["fabric"]
    return f


def concurrency_stats(intervals: list[tuple[int, int]]) -> tuple[float, float]:
    """(peak, mean_experienced).

    `peak` is the most flows alive at any instant: an UPPER bound on the
    fair-share slowdown, which is why a measured mean legitimately sits below it.
    Reporting it as a floor is wrong-signed.
    `mean_experienced` averages, over flows, the time-averaged concurrency during
    each flow's own lifetime -- the fair share a flow should actually expect, and
    the right denominator for a "how close to ideal" ratio."""
    if not intervals:
        return float("nan"), float("nan")
    ev = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
                key=lambda x: (x[0], x[1]))        # closes before opens at a tie
    peak, cur, segs = 0, 0, []
    for i in range(len(ev) - 1):
        cur += ev[i][1]
        peak = max(peak, cur)
        if ev[i + 1][0] > ev[i][0]:
            segs.append((ev[i][0], ev[i + 1][0], cur))
    means = []
    for s, e in intervals:
        if e <= s:
            continue
        acc = sum(k * (min(t1, e) - max(t0, s))
                  for t0, t1, k in segs if min(t1, e) > max(t0, s))
        means.append(acc / (e - s))
    return float(peak), (float(np.mean(means)) if means else float("nan"))


def find_bottleneck(topo: Topology | None, port_max: dict[tuple[int, int], int] | None,
                    flows: pd.DataFrame | None) -> Bottleneck | None:
    """The congested directed link, plus the ingress ports feeding it.

    Ground truth is qlen.txt: the port that actually built a queue. Falls back to
    the directed link carrying the most incast bytes when qlen is unavailable.

    The ingress ports come from the flows' paths and are returned as a tuple of
    PORT indices, which is what the PFC threshold is compared against: SwitchMmu
    keys ingress accounting on (port, qIndex), not on flows. Dozens of flows
    entering through two host ports load two ingress counters."""
    if topo is None:
        return None
    sw = port = None
    if port_max:
        (sw, port) = max(port_max.items(), key=lambda kv: kv[1])[0]
        if port not in topo.ports.get(sw, {}):
            sw = port = None
    if sw is None and flows is not None and "path" in flows.columns:
        load: dict[tuple[int, int], int] = {}
        for path, size, inc in zip(flows["path"], flows["size"], flows["incast"]):
            if path and inc:
                for a, b in path:
                    if topo.is_switch(a):
                        load[(a, b)] = load.get((a, b), 0) + size
        if load:
            (a, b), _ = max(load.items(), key=lambda kv: kv[1])
            sw, port = a, topo.port_facing(a, b)
    if sw is None or port is None:
        return None

    peer = topo.ports[sw][port].peer
    ingress: set[int] = set()
    if flows is not None and "path" in flows.columns:
        for path, inc in zip(flows["path"], flows["incast"]):
            if not path or not inc:
                continue
            for i, (a, b) in enumerate(path):
                if (a, b) == (sw, peer) and i > 0:
                    if (p := topo.port_facing(sw, path[i - 1][0])) is not None:
                        ingress.add(p)
    return Bottleneck(sw, port, peer, topo.ports[sw][port].rate, tuple(sorted(ingress)))


def bottleneck_intervals(flows: pd.DataFrame, bn: Bottleneck) -> list[tuple[int, int]]:
    """(start, arrival) of every incast flow crossing the bottleneck link."""
    return [(s, a) for path, s, a, inc in
            zip(flows["path"], flows["start"], flows["arrival"], flows["incast"])
            if path and inc and (bn.switch, bn.peer) in path]