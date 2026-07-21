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

    hops == 1                            host-to-host link: never congested, so
                                         slowdown ~1 by construction. Class 'tp'.
    hops  > 1                            traverses at least one switch.
    flow_class                           structural, from utils.roles: the
                                         declared placement says what a flow is,
                                         so no size threshold has to be tuned.

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

from collections import defaultdict

import numpy as np
import pandas as pd

from . import roles
from .fabric import Bottleneck, Topology
from .roles import Placement


def annotate(fct: pd.DataFrame, topo: Topology, placement: Placement,
             mtu: int) -> pd.DataFrame:
    """Attach hops / path / flow_class.

    Both arguments are required. Without the topology there are no hop counts,
    so a TP collective on a dedicated host-to-host link is indistinguishable
    from a congested KV transfer -- and mixing them is not a rounding error, it
    is a bimodal mixture whose mean and CV describe the mixing ratio rather than
    the fabric. On the reference T1 run that is 2880 direct flows against 82
    fabric ones: the statistics could not move no matter what the buffer did.
    Without the placement there is no class. Neither has a default."""
    f = fct.copy()
    f["hops"] = [topo.dist.get(s, {}).get(d, np.nan) for s, d in zip(f["src"], f["dst"])]
    f["path"] = [topo.path(s, d) for s, d in zip(f["src"], f["dst"])]
    f["flow_class"] = roles.classify(f, placement, mtu)
    return f


def crosses(flows: pd.DataFrame, bn: Bottleneck) -> pd.Series:
    """Whether each flow's path traverses the bottleneck's directed link."""
    return pd.Series([bool(p) and (bn.switch, bn.peer) in p for p in flows["path"]],
                     index=flows.index)


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


def _ingress_ports(topo: Topology, sw: int, peer: int,
                   congesting: pd.DataFrame) -> tuple[int, ...]:
    """Port indices on `sw` that feed traffic into the (sw, peer) egress, per
    the paths of `congesting`. Shared by find_bottleneck (the single deepest
    queue) and candidate_links (every link at least one flow crosses) so both
    derive F_ports the same way -- from the flow paths, never guessed."""
    ingress: set[int] = set()
    for path in congesting["path"]:
        if not path:
            continue
        for i, (x, y) in enumerate(path):
            if (x, y) == (sw, peer) and i > 0:
                if (q := topo.port_facing(sw, path[i - 1][0])) is not None:
                    ingress.add(q)
    return tuple(sorted(ingress))


def find_bottleneck(topo: Topology, port_max: dict[tuple[int, int], int],
                    congesting: pd.DataFrame) -> Bottleneck:
    """The congested directed link, plus the ingress ports feeding it.

    Ground truth is qlen.txt: the port that actually built the deepest queue.
    There is no fallback -- an empty qlen.txt is a broken run, not a reason to
    guess from flow volumes.

    `congesting` is the flow population whose paths define the ingress set: pass
    the class that is meant to congest this link (kv), not everything, or an
    unrelated flow that happens to share the egress inflates F_ports.

    The ingress ports are returned as a tuple of PORT indices, which is what the
    PFC threshold is compared against: SwitchMmu keys ingress accounting on
    (port, qIndex), not on flows. Dozens of flows entering through two host ports
    load two ingress counters, not dozens."""
    if not port_max:
        raise ValueError("qlen.txt has no port samples: cannot locate the "
                         "bottleneck. Check QLEN_MON_START against the run.")
    (sw, port) = max(port_max.items(), key=lambda kv: kv[1])[0]
    if port not in topo.ports.get(sw, {}):
        raise ValueError(
            f"qlen.txt reports the deepest queue on port {port} of switch {sw}, "
            f"which the topology says does not exist (it has ports "
            f"{sorted(topo.ports.get(sw, {}))}). physical_topology.txt and this "
            f"run's ns-3 output do not describe the same fabric.")
    peer = topo.ports[sw][port].peer
    ingress = _ingress_ports(topo, sw, peer, congesting)
    return Bottleneck(sw, port, peer, topo.ports[sw][port].rate, ingress)


def candidate_links(topo: Topology, port_max: dict[tuple[int, int], int],
                    congesting: pd.DataFrame) -> list[Bottleneck]:
    """Every directed (switch, egress_port) at least one row of `congesting`
    crosses, as a Bottleneck, sorted by that port's peak queue (deepest first
    -- so element 0 is find_bottleneck's link whenever qlen.txt and the flow
    paths agree on which one that is).

    Where find_bottleneck starts from qlen.txt's argmax and asks which flows
    feed it, this starts from the flows and asks which links they touch: it
    walks every path once, collecting the (sw, peer) edges seen and, for each,
    the ingress ports the SAME way _ingress_ports does. That is the only
    difference -- one deepest link vs. every link the KV population could
    possibly have congested, which is what a single global bottleneck cannot
    show on a topology with several independently-congestible uplinks (e.g.
    one oversubscribed ToR->core link per ToR)."""
    ingress_by_link: dict[tuple[int, int], set[int]] = defaultdict(set)
    for path in congesting["path"]:
        if not path:
            continue
        for i, (x, y) in enumerate(path):
            if not topo.is_switch(x):
                continue
            if i > 0 and (q := topo.port_facing(x, path[i - 1][0])) is not None:
                ingress_by_link[(x, y)].add(q)
            else:
                ingress_by_link.setdefault((x, y), set())

    out = []
    for (sw, peer), ingress in ingress_by_link.items():
        port = topo.port_facing(sw, peer)
        if port is None:
            continue
        out.append(Bottleneck(sw, port, peer, topo.ports[sw][port].rate,
                              tuple(sorted(ingress))))
    out.sort(key=lambda bn: port_max.get((bn.switch, bn.egress_port), 0),
             reverse=True)
    return out


def concurrency_series(intervals: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Step series (times, concurrent-flow count) built from the same
    sweep-line event list concurrency_stats collapses to peak/mean -- for
    plotting the count over time instead of summarising it. One point per
    open/close event; the count is the value AFTER that event, so plotting
    with drawstyle='steps-post' reproduces the true step function. Empty
    arrays on empty input."""
    if not intervals:
        return np.array([]), np.array([])
    ev = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
               key=lambda x: (x[0], x[1]))       # closes before opens at a tie
    times, counts, cur = [], [], 0
    for t, delta in ev:
        cur += delta
        times.append(t)
        counts.append(cur)
    return np.asarray(times, dtype=float), np.asarray(counts, dtype=float)


def intervals(flows: pd.DataFrame) -> list[tuple[int, int]]:
    """(start, arrival) per flow, for concurrency_stats. The caller decides the
    population; this does not filter."""
    return list(zip(flows["start"].astype(int), flows["arrival"].astype(int)))