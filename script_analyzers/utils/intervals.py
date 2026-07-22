#!/usr/bin/env python3
"""
utils.intervals — interval-set algebra on [start, end) pairs.

One home for the four questions the analyzers keep asking of a set of intervals:
how much time does their union cover, how much of A is masked by B, what is A
minus B, and how many overlap at once. Every function takes an iterable of
(start, end) pairs and drops the empty ones (end <= start).

Summing raw durations double-counts concurrency; these merge first, which is the
honest busy time. `overlap_len` likewise merges each side before intersecting, so
self-overlapping inputs (e.g. one rank's concurrent transfers) are not counted
twice -- the reason it lives here rather than being re-derived per caller.

Kept apart from utils.astra (a CSV reader) because it is pure geometry, and from
utils.flows.concurrency_stats (which additionally weights concurrency by the time
experienced -- a different question); utils.astra, bandwidth_sweep and
astra_analyzer all reach for the plain algebra here.
"""

from __future__ import annotations


def merge(intervals) -> list[tuple[float, float]]:
    """Sorted, with overlapping OR touching intervals fused into disjoint ones."""
    out: list[list[float]] = []
    for s, e in sorted((s, e) for s, e in intervals if e > s):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def total(intervals) -> float:
    """Raw sum of lengths -- no merge. Use on an already-disjoint set (e.g. the
    output of merge / subtract) where summing is exactly the covered time."""
    return float(sum(e - s for s, e in intervals))


def union_len(intervals) -> float:
    """Total time covered by the union: merge first, then sum."""
    return total(merge(intervals))


def subtract(a, b) -> list[tuple[float, float]]:
    """merge(a) minus merge(b): the parts of A not covered by B, as disjoint
    intervals."""
    a = merge(a)
    if not a:
        return []
    b = merge(b)
    res: list[tuple[float, float]] = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                res.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            res.append((cur, e))
    return res


def overlap_len(a, b) -> float:
    """Time covered by BOTH sets -- how much of A is masked by B. Two-pointer
    sweep over each side already merged, so neither side self-double-counts."""
    a, b = merge(a), merge(b)
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def max_concurrency(intervals) -> int:
    """Greatest number of intervals in flight at any instant (sweep line). Ends
    are processed before starts at a shared tick, i.e. intervals that merely touch
    ([a, t) and [t, b)) do not count as concurrent -- consistent with merge()."""
    ev: list[tuple[float, int]] = []
    for s, e in intervals:
        if e > s:
            ev.append((s, 1))
            ev.append((e, -1))
    ev.sort()
    cur = peak = 0
    for _, d in ev:
        cur += d
        if cur > peak:
            peak = cur
    return peak
