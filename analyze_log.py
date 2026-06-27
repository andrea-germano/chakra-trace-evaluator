#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

_EVENT_RE = re.compile(
    r"(?P<verb>issue|callback),"
    r"sys->id=(?P<sys>\d+),\s*"
    r"tick=(?P<tick>\d+),\s*"
    r"node->id=(?P<nid>\d+),\s*"
    r"node->name=(?P<name>.+?),\s*"
    r"node->type=(?P<ntype>\d+)"
)
_METRIC_RE = re.compile(r"(\w+)=([-+0-9.eE]+)")
_STAT_RE = re.compile(
    r"sys\[(?P<sys>\d+)\],\s*(?P<key>[A-Za-z \-]+?):\s*(?P<val>[-0-9.]+)%?"
)

# Chakra node->type integer -> coarse kind.
#   4 = COMP, 5 = SEND, 6 = RECV, 7 = COLL (all-reduce here), 3/1 = COLL (legacy)
NTYPE = {4: "COMP", 5: "SEND", 6: "RECV", 7: "COLL", 3: "COLL", 1: "COLL"}
COMM_NTYPES = (5, 6, 7, 3, 1)        # anything that is communication
COST_NTYPES = (5, 7, 3, 1)           # producer-side cost (SEND + collective), not blocking RECV


@dataclass
class Event:
    sys: int
    node_id: int
    name: str
    ntype: int
    issue_tick: Optional[int] = None
    callback_tick: Optional[int] = None
    op_intensity: Optional[float] = None
    perf: Optional[float] = None
    compute_util: Optional[float] = None
    memory_util: Optional[float] = None

    @property
    def duration(self) -> int:
        if self.issue_tick is None or self.callback_tick is None:
            return 0
        return self.callback_tick - self.issue_tick


def parse_trace(path: Path) -> List[Event]:
    events: Dict[Tuple[int, int], Event] = {}
    last_issued: Dict[int, Tuple[int, int]] = {}
    for raw in path.read_text().splitlines():
        m = _EVENT_RE.search(raw)
        if m:
            key = (int(m["sys"]), int(m["nid"]))
            ev = events.get(key)
            if ev is None:
                ev = Event(sys=int(m["sys"]), node_id=int(m["nid"]),
                           name=m["name"], ntype=int(m["ntype"]))
                events[key] = ev
            tick = int(m["tick"])
            if m["verb"] == "issue":
                ev.issue_tick = tick
                last_issued[int(m["sys"])] = key
            else:
                ev.callback_tick = tick
            continue
        if "operation_intensity=" in raw:
            kv = dict(_METRIC_RE.findall(raw))
            target = None
            for sid, key in last_issued.items():
                e = events.get(key)
                if e and e.op_intensity is None and e.ntype == 4:
                    target = e
            if target is not None:
                target.op_intensity = float(kv.get("operation_intensity", "nan"))
                target.perf = float(kv.get("perf", "nan"))
                target.compute_util = float(kv.get("compute_utilization", "nan"))
                target.memory_util = float(kv.get("memory_utilization", "nan"))
    return list(events.values())


def parse_statistics(path: Path) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = defaultdict(dict)
    for raw in path.read_text().splitlines():
        m = _STAT_RE.search(raw)
        if not m:
            continue
        key = m["key"].strip().lower().replace(" ", "_").replace("-", "_")
        try:
            stats[int(m["sys"])][key] = float(m["val"])
        except ValueError:
            pass
    return dict(stats)


# --------------------------------------------------------------------------- #
# Name parsing & classification (MLSynth inference convention)
# --------------------------------------------------------------------------- #

def parse_fields(name: str) -> Dict[str, str]:
    """key=value fields from a `..._k=v_k=v_...` node name."""
    return dict(tok.split("=", 1) for tok in name.split("_") if "=" in tok)


def parse_op(name: str) -> Optional[str]:
    # node names end in _op=attn / _op=ffw (naming.comp_name / coll_name)
    return parse_fields(name).get("op")


def class_prefix(name: str) -> str:
    """Leading class token, or '' for compute/all-reduce nodes (which start `pl=`)."""
    first = name.split("_", 1)[0]
    return "" if "=" in first else first


# category -> (color, legend label)
CATEGORIES = [
    ("prefill_compute",   "#1f77b4", "Prefill compute"),
    ("prefill_allreduce", "#ff7f0e", "Prefill all-reduce (TP)"),
    ("decode_compute",    "#2ca02c", "Decode compute"),
    ("decode_allreduce",  "#98df8a", "Decode all-reduce (TP)"),
    ("pp",                "#9467bd", "PP send/recv (pipeline)"),
    ("kv_transfer",       "#d62728", "KV-cache transfer"),
    ("first_token",       "#8c564b", "First-token handoff"),
    ("other",             "#bcbd22", "Other"),
]
CAT_COLOR = {c: col for c, col, _ in CATEGORIES}
CAT_LABEL = {c: lab for c, _, lab in CATEGORIES}


def classify(ev: Event) -> str:
    cls = class_prefix(ev.name)
    pl = parse_fields(ev.name).get("pl")
    if ev.ntype == 4:                       # compute
        return "prefill_compute" if pl == "p" else "decode_compute"
    if ev.ntype == 7:                       # all-reduce collective
        return "prefill_allreduce" if pl == "p" else "decode_allreduce"
    if cls == "PP":
        return "pp"
    if cls == "KV":
        return "kv_transfer"
    if cls == "FIRSTTOK":
        return "first_token"
    return "other"


# --------------------------------------------------------------------------- #
# Interval helpers
# --------------------------------------------------------------------------- #

def merge(ivs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ivs = sorted((a, b) for a, b in ivs if b > a)
    out: List[Tuple[int, int]] = []
    for lo, hi in ivs:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def union(ivs: List[Tuple[int, int]]) -> int:
    return sum(hi - lo for lo, hi in merge(ivs))


def intersection(a, b) -> List[Tuple[int, int]]:
    A, B = merge(a), merge(b)
    res, i, j = [], 0, 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if hi > lo:
            res.append((lo, hi))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return res


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #

@dataclass
class Topo:
    pool: Dict[int, str]                 # sys -> 'prefill' / 'decode' / '?'
    stage: Dict[int, int]                # sys -> pipeline stage (from COMP ss)
    shard: Dict[int, int]                # sys -> tensor-parallel shard (from COMP sh)
    prefill_stages: Dict[int, List[int]] # stage -> [sys ...]
    decode_sys: List[int]


def infer_topology(events: List[Event]) -> Topo:
    pool: Dict[int, str] = {}
    stage: Dict[int, int] = {}
    shard: Dict[int, int] = {}
    for ev in events:
        if ev.ntype != 4:
            continue
        f = parse_fields(ev.name)
        pl = f.get("pl")
        if pl == "p":
            pool[ev.sys] = "prefill"
        elif pl == "d":
            pool.setdefault(ev.sys, "decode")
        if "ss" in f:
            stage[ev.sys] = int(f["ss"])
        if "sh" in f:
            shard[ev.sys] = int(f["sh"])
    prefill_stages: Dict[int, List[int]] = defaultdict(list)
    for s, p in pool.items():
        if p == "prefill":
            prefill_stages[stage.get(s, 0)].append(s)
    for st in prefill_stages.values():
        st.sort()
    decode_sys = sorted(s for s, p in pool.items() if p == "decode")
    return Topo(pool, stage, shard, dict(sorted(prefill_stages.items())), decode_sys)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@dataclass
class NodeAgg:
    sys: int
    pool: str = "?"
    cat_intervals: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    cat_cost_intervals: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    compute_intervals: List[Tuple[int, int]] = field(default_factory=list)
    comm_intervals: List[Tuple[int, int]] = field(default_factory=list)
    first_tick: Optional[int] = None
    last_tick: Optional[int] = None
    op_intensity_sum: float = 0.0
    op_intensity_n: int = 0
    comp_bound: int = 0
    mem_bound: int = 0


def aggregate(events: List[Event], topo: Topo) -> Dict[int, NodeAgg]:
    aggs: Dict[int, NodeAgg] = {}
    for ev in events:
        a = aggs.get(ev.sys)
        if a is None:
            a = NodeAgg(sys=ev.sys, pool=topo.pool.get(ev.sys, "?"))
            aggs[ev.sys] = a
        cat = classify(ev)
        if ev.issue_tick is not None and ev.callback_tick is not None and ev.callback_tick > ev.issue_tick:
            span = (ev.issue_tick, ev.callback_tick)
            a.cat_intervals[cat].append(span)
            if ev.ntype == 4:
                a.compute_intervals.append(span)
            if ev.ntype in COMM_NTYPES:
                a.comm_intervals.append(span)
            if ev.ntype in COST_NTYPES:
                a.cat_cost_intervals[cat].append(span)
        for t in (ev.issue_tick, ev.callback_tick):
            if t is None:
                continue
            a.first_tick = t if a.first_tick is None else min(a.first_tick, t)
            a.last_tick = t if a.last_tick is None else max(a.last_tick, t)
        if ev.ntype == 4 and ev.op_intensity is not None:
            a.op_intensity_sum += ev.op_intensity
            a.op_intensity_n += 1
            if (ev.memory_util or 0) >= (ev.compute_util or 0):
                a.mem_bound += 1
            else:
                a.comp_bound += 1
    return aggs


# --------------------------------------------------------------------------- #
# Latency metrics: TTFT, TPOT, and a critical-path compute/comm split
# --------------------------------------------------------------------------- #

def latency_metrics(events: List[Event], topo: Topo) -> Dict[str, object]:
    prefill_end = None
    for ev in events:
        if ev.ntype == 4 and parse_fields(ev.name).get("pl") == "p" and ev.callback_tick is not None:
            prefill_end = ev.callback_tick if prefill_end is None else max(prefill_end, ev.callback_tick)

    # token ready = completion of the last layer's ffw on the decode pool, per iteration
    dec = [ev for ev in events
           if ev.ntype == 4 and parse_fields(ev.name).get("pl") == "d" and ev.callback_tick is not None]
    max_L = max((int(parse_fields(e.name).get("L", 0)) for e in dec), default=0)
    token_end: Dict[int, int] = {}
    for ev in dec:
        f = parse_fields(ev.name)
        if int(f.get("L", 0)) == max_L and parse_op(ev.name) == "ffw":
            it = int(f.get("it", 0))
            token_end[it] = max(token_end.get(it, 0), ev.callback_tick)

    its = sorted(token_end)
    ends = [token_end[i] for i in its]
    tbt = [ends[i] - ends[i - 1] for i in range(1, len(ends))]
    first_gap = (ends[0] - prefill_end) if (ends and prefill_end is not None) else None

    return {
        "ttft_ns": prefill_end,
        "decode_iterations": its,
        "token_end_ns": ends,
        "first_token_to_decode_gap_ns": first_gap,
        "tbt_ns": tbt,
        "tpot_mean_ns": (sum(tbt) / len(tbt)) if tbt else None,
        "tpot_steady_ns": (sum(tbt[1:]) / len(tbt[1:])) if len(tbt) > 1 else (tbt[0] if tbt else None),
    }


def ttft_decomposition(aggs: Dict[int, NodeAgg], topo: Topo) -> Dict[str, object]:
    """Compute-vs-communication split of the TTFT critical path.

    Stages run strictly serially (single batch, no micro-batch pipelining), so
    the critical path is the sum over stages of that stage's busy time.  Within a
    stage the TP shards are concurrent and identical, so a stage's compute / AR
    is the max over its shards of the per-shard interval union.  PP transfer is
    reported separately; its big 'receive' span is a pipeline *bubble* (idle wait
    for the previous stage), already accounted for as that stage's busy time, so
    it is not double-counted into the path total.
    """
    stages = sorted(topo.prefill_stages)
    per_stage = []
    comp_path = ar_path = kv_exposed = 0
    for st in stages:
        syss = topo.prefill_stages[st]
        comp = max((union(aggs[s].cat_intervals.get("prefill_compute", [])) for s in syss), default=0)
        ar = max((union(aggs[s].cat_intervals.get("prefill_allreduce", [])) for s in syss), default=0)
        kv = max((union(aggs[s].cat_cost_intervals.get("kv_transfer", [])) for s in syss), default=0)
        per_stage.append({"stage": st, "compute_ns": comp, "allreduce_ns": ar, "kv_send_ns": kv})
        comp_path += comp
        ar_path += ar
        kv_exposed = max(kv_exposed, kv)
    total = comp_path + ar_path
    return {
        "compute_ns": comp_path,
        "allreduce_ns": ar_path,
        "critical_path_ns": total,
        "compute_frac": (comp_path / total) if total else None,
        "allreduce_frac": (ar_path / total) if total else None,
        "per_stage": per_stage,
        "kv_send_overlapped_ns": kv_exposed,
    }


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #

def reconcile(aggs: Dict[int, NodeAgg], stats: Dict[int, Dict[str, float]]) -> List[str]:
    lines = []
    for s in sorted(aggs):
        a = aggs[s]
        st = stats.get(s, {})
        gpu, comm = union(a.compute_intervals), union(a.comm_intervals)
        wall = (a.last_tick - a.first_tick) if a.first_tick is not None else None
        fmt = lambda v: f"{int(v):,}" if v is not None else "n/a"
        lines.append(
            f"sys[{s}] ({a.pool}): compute parsed={fmt(gpu)} stats={fmt(st.get('gpu_time'))} | "
            f"comm(union) parsed={fmt(comm)} stats={fmt(st.get('comm_time'))} | "
            f"wall parsed={fmt(wall)} stats={fmt(st.get('wall_time'))}"
        )
    return lines


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _ms(t: float) -> float:
    return t / 1e6


def _gantt(ax, events, sys_ids, topo, tick_ns, xlim=None, title=""):
    """Shared Gantt renderer: three lanes per sys (compute / send+coll / recv)."""
    LANES = {"compute": -0.24, "sendcoll": 0.0, "recv": 0.24}
    LH = 0.18
    yticks, ylabels = [], []
    sel = [e for e in events if e.sys in sys_ids and e.issue_tick is not None and e.callback_tick is not None]
    by_sys = defaultdict(list)
    for e in sel:
        by_sys[e.sys].append(e)
    for row, sid in enumerate(sys_ids):
        y = row
        yticks.append(y)
        pool = topo.pool.get(sid, "?")
        tag = f"st{topo.stage.get(sid,'?')}·sh{topo.shard.get(sid,'?')}"
        ylabels.append(f"sys[{sid}]\n{pool} {tag}")
        comp_iv = [(e.issue_tick, e.callback_tick) for e in by_sys[sid] if e.ntype == 4]
        comm_iv = [(e.issue_tick, e.callback_tick) for e in by_sys[sid] if e.ntype in COMM_NTYPES]
        for lo, hi in intersection(comp_iv, comm_iv):
            ax.add_patch(mpatches.Rectangle(
                (_ms(lo * tick_ns), y - 0.36), _ms((hi - lo) * tick_ns), 0.72,
                facecolor="0.6", alpha=0.28, edgecolor="none", zorder=0))
        for e in by_sys[sid]:
            cat = classify(e)
            lane = "compute" if e.ntype == 4 else "recv" if e.ntype == 6 else "sendcoll"
            x0 = _ms(e.issue_tick * tick_ns)
            w = _ms(max(e.duration, 1) * tick_ns)
            ax.barh(y + LANES[lane], w, left=x0, height=LH,
                    color=CAT_COLOR.get(cat, "#bcbd22"), edgecolor="black",
                    linewidth=0.15, hatch="///" if e.ntype == 6 else None,
                    alpha=0.95, zorder=2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.6, len(sys_ids) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("time (ms)")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)


def _gantt_legend(ax):
    handles = [mpatches.Patch(color=col, label=lab) for c, col, lab in CATEGORIES]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="(RECV / blocking wait)"))
    handles.append(mpatches.Patch(facecolor="0.6", alpha=0.28, label="compute & comm overlap"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, fontsize=8, frameon=False)


def fig_phase_overview(lat, topo, aggs, tick_ns, out):
    """One bar per pool showing the prefill window and the decode window
    (decode starts at end-of-prefill), with TTFT and TPOT annotated."""
    ttft = lat["ttft_ns"]
    ends = lat["token_end_ns"] or []
    g_end = max((a.last_tick for a in aggs.values() if a.last_tick is not None), default=0)
    fig, ax = plt.subplots(figsize=(13, 3.4))

    # prefill bar
    ax.barh(1, _ms(ttft * tick_ns), left=0, height=0.5,
            color=CAT_COLOR["prefill_compute"], alpha=0.85, label="Prefill phase")
    ax.text(_ms(ttft * tick_ns) / 2, 1, f"PREFILL\nTTFT = {_ms(ttft*tick_ns):.2f} ms",
            ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    # decode bar starting at end of prefill
    if ends:
        ax.barh(0, _ms((g_end - ttft) * tick_ns), left=_ms(ttft * tick_ns), height=0.5,
                color=CAT_COLOR["decode_compute"], alpha=0.85, label="Decode phase")
        for k, e in enumerate(ends):
            ax.axvline(_ms(e * tick_ns), 0.02, 0.34, color="white", lw=0.8, alpha=0.8)
        tpot = lat["tpot_steady_ns"]
        # decode window is tiny next to prefill, so label it in the empty left part
        # of the decode row with an arrow pointing at the bar.
        ax.annotate(
            f"DECODE: {len(ends)} tokens · TPOT = {_ms(tpot*tick_ns):.3f} ms/token",
            xy=(_ms(ttft * tick_ns), 0), xycoords="data",
            xytext=(_ms(ttft * tick_ns) * 0.45, 0), textcoords="data",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=CAT_COLOR["decode_compute"],
            arrowprops=dict(arrowstyle="->", color=CAT_COLOR["decode_compute"], lw=1.5))
    ax.axvline(_ms(ttft * tick_ns), color="black", lw=1.2, ls="--")
    ax.text(_ms(ttft * tick_ns), 1.7, "first token", ha="center", fontsize=8)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["decode pool", "prefill pool"])
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel("time (ms)")
    ax.set_title("Inference phase overview — prefill (→ TTFT) then decode (→ TPOT per token)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_ttft_breakdown(decomp, lat, tick_ns, out, ref_ttft_ms=None):
    """LEFT: TTFT critical path as alternating compute / all-reduce segments per
            stage (this is literally where the milliseconds go).
       RIGHT: compute-vs-communication share of TTFT as a donut."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [2.3, 1]})
    bottom = 0.0
    for ps in decomp["per_stage"]:
        for key, label, color in [
            ("compute_ns",   f"stage {ps['stage']} compute",   CAT_COLOR["prefill_compute"]),
            ("allreduce_ns", f"stage {ps['stage']} all-reduce", CAT_COLOR["prefill_allreduce"]),
        ]:
            h = _ms(ps[key] * tick_ns)
            if h <= 0:
                continue
            ax1.bar(0, h, 0.5, bottom=bottom, color=color, edgecolor="white")
            ax1.text(0, bottom + h / 2, f"{label}\n{h:.2f} ms", ha="center", va="center",
                     color="white", fontsize=8, fontweight="bold")
            bottom += h
    ax1.text(0, bottom * 1.01, f"TTFT = {bottom:.2f} ms", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
    if ref_ttft_ms:
        ax1.axhline(ref_ttft_ms, color="crimson", ls="--", lw=1.5)
        ax1.text(0.38, ref_ttft_ms, f"reference\n{ref_ttft_ms:.0f} ms", color="crimson",
                 fontsize=8, va="center", ha="left")
        ax1.annotate("", xy=(0.32, bottom), xytext=(0.32, ref_ttft_ms),
                     arrowprops=dict(arrowstyle="<->", color="crimson"))
        ax1.text(0.34, (bottom + ref_ttft_ms) / 2, f"gap {bottom-ref_ttft_ms:+.1f} ms",
                 color="crimson", fontsize=8, va="center")
    ax1.set_xlim(-0.55, 0.7)
    ax1.set_xticks([])
    ax1.set_ylabel("latency (ms)")
    ax1.set_ylim(0, bottom * 1.15)
    ax1.set_title("TTFT critical path: compute vs TP all-reduce, per pipeline stage")

    comp = _ms(decomp["compute_ns"] * tick_ns)
    ar = _ms(decomp["allreduce_ns"] * tick_ns)
    wed, _, _ = ax2.pie(
        [comp, ar], colors=[CAT_COLOR["prefill_compute"], CAT_COLOR["prefill_allreduce"]],
        startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"),
        autopct=lambda p: f"{p:.0f}%", pctdistance=0.78,
        textprops=dict(color="white", fontweight="bold", fontsize=10))
    ax2.legend(wed, [f"compute  {comp:.1f} ms", f"all-reduce  {ar:.1f} ms"],
               loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False)
    ax2.set_title("TTFT composition")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_prefill_timeline(events, topo, tick_ns, ttft_ns, out):
    sys_ids = sorted(s for s, p in topo.pool.items() if p == "prefill")
    fig, ax = plt.subplots(figsize=(14, 1.4 + 0.9 * len(sys_ids)))
    _gantt(ax, events, sys_ids, topo, tick_ns, xlim=(0, _ms(ttft_ns * tick_ns) * 1.04),
           title="Prefill timeline — lanes: compute (top) / send+all-reduce (mid) / recv (bottom, hatched)")
    ax.axvline(_ms(ttft_ns * tick_ns), color="black", ls="--", lw=1)
    ax.text(_ms(ttft_ns * tick_ns), -0.55, f"TTFT {_ms(ttft_ns*tick_ns):.1f} ms",
            ha="right", fontsize=8)
    _gantt_legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_stage_detail(events, topo, tick_ns, stage, out):
    sys_ids = topo.prefill_stages[stage]
    win_events = [e for e in events if e.sys in sys_ids and e.issue_tick is not None]
    lo = min((e.issue_tick for e in win_events), default=0)
    hi = max((e.callback_tick for e in win_events if e.callback_tick is not None), default=1)
    fig, ax = plt.subplots(figsize=(14, 1.2 + 1.1 * len(sys_ids)))
    _gantt(ax, events, sys_ids, topo, tick_ns,
           xlim=(_ms(lo * tick_ns) - 1, _ms(hi * tick_ns) + 1),
           title=f"Prefill PP stage {stage} — detailed view (sys {sys_ids}, TP={len(sys_ids)})")
    _gantt_legend(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interactive (zoomable) Gantt — single self-contained HTML, replaces the
# per-stage detail PNGs.  Rendered with WebGL (one `scattergl` polyline per
# category, one task = one line segment) so it stays smooth for traces with
# 10^5+ nodes, where one-SVG-rect-per-task would choke the browser.  Navigate
# by drag-to-pan + scroll-to-zoom + Prefill/Decode/Full buttons; hover any task
# for its details.  Covers every sys: prefill stages (ordered by stage then TP
# shard) followed by the decode pool.
# --------------------------------------------------------------------------- #

# CDN fallback used only when the `plotly` python package is not installed.
_PLOTLY_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Inference Gantt</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>html,body{{margin:0;height:100%;background:#fff;
font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif}}
#gantt{{width:100%;height:100vh}}</style></head>
<body><div id="gantt"></div>
<script>
const data={data};
const layout={layout};
const config={config};
Plotly.newPlot("gantt", data, layout, config);
window.addEventListener("resize", () => Plotly.Plots.resize("gantt"));
</script></body></html>
"""


def _interactive_gantt_spec(events, topo, tick_ns, lat):
    """Build Plotly (data, layout, config) dicts for the zoomable Gantt.

    Each task is rendered as a single WebGL line segment (one `scattergl`
    trace per category, segments separated by None) rather than one SVG bar
    per task.  This keeps the DOM tiny and pushes drawing onto the GPU, so the
    page stays smooth for traces with 10^5+ nodes.  The range slider (which
    renders a full second copy of every trace) is intentionally omitted;
    navigation is drag-to-pan + scroll-to-zoom + the Prefill/Decode/Full
    buttons.  Returns plain JSON-serialisable structures so the same spec
    drives both the `plotly` python package and the CDN HTML fallback.
    """
    # Row order: prefill by (stage, shard, sys), then any decode-only sys.
    prefill_sys = sorted(
        (s for s, p in topo.pool.items() if p == "prefill"),
        key=lambda s: (topo.stage.get(s, 0), topo.shard.get(s, 0), s),
    )
    ordered = prefill_sys + [s for s in topo.decode_sys if s not in prefill_sys]
    row_of = {s: i for i, s in enumerate(ordered)}
    n = len(ordered)

    # Three lanes per sys row, mirroring the static _gantt layout.
    LANES = {"compute": -0.26, "sendcoll": 0.0, "recv": 0.26}
    LINE_W = 6                                  # bar thickness, in pixels
    SEP_CD = ["", "", 0.0, 0.0, 0.0, "", ""]    # placeholder for None separators

    # A SEND and its matching RECV share the SAME node name (MLSynth naming
    # contract), so we can pair them.  RECV nodes are emitted with no real
    # dependency (parents=None) and are therefore *issued at tick 0*; drawn
    # from their issue tick they produce a blocking-wait bar stretching from the
    # start of the run to arrival, which is misleading — the device is simply
    # idle until the message lands, and the actual transfer only happens once
    # the producer's SEND has gone out.  We anchor each RECV's drawn start to
    # its matching SEND (when data can begin arriving) and leave the idle gap
    # before it blank.  Display-only: the raw intervals used by metrics are kept.
    send_cb: Dict[str, int] = {}            # name -> earliest SEND completion
    send_is: Dict[str, int] = {}            # name -> earliest SEND issue
    active_cb_by_sys: Dict[int, list] = defaultdict(list)   # sys -> sorted real-work completions
    for e in events:
        if e.ntype == 5:
            if e.issue_tick is not None and (e.name not in send_is or e.issue_tick < send_is[e.name]):
                send_is[e.name] = e.issue_tick
            if e.callback_tick is not None and (e.name not in send_cb or e.callback_tick < send_cb[e.name]):
                send_cb[e.name] = e.callback_tick
        if e.ntype in (4, 5, 7) and e.callback_tick is not None:   # compute/send/coll = real work
            active_cb_by_sys[e.sys].append(e.callback_tick)
    for s in active_cb_by_sys:
        active_cb_by_sys[s].sort()
    prefill_end_tick = lat.get("ttft_ns")   # first token / KV / decode cannot land before this

    def _recv_start(e: "Event") -> Tuple[int, str]:
        """Drawn start for a RECV, never the artificial tick-0 issue.

        Priority: matching SEND (tightest, real transfer window) -> the device's
        own last real-work completion (exposed idle starts there) -> for decode-
        pool receivers, the end of prefill (data can't arrive earlier) -> a short
        sliver ending at arrival.  Whatever wins, it is < callback.
        """
        cb = e.callback_tick
        cands = [v for v in (send_cb.get(e.name), send_is.get(e.name)) if v is not None and v < cb]
        if cands:
            return max(cands), "transfer window (anchored to SEND)"
        lst = active_cb_by_sys.get(e.sys)
        if lst:
            idx = bisect.bisect_right(lst, cb) - 1
            if idx >= 0 and lst[idx] < cb:
                return lst[idx], "exposed wait (from this device's last op)"
        if topo.pool.get(e.sys) == "decode" and prefill_end_tick is not None and prefill_end_tick < cb:
            return prefill_end_tick, "exposed wait (from end of prefill)"
        return max(e.issue_tick, cb - 1), "blocking RECV (no matching SEND found)"

    # Per-category polyline buffers: x/y get [x0, x1, None] per task.
    seg: Dict[str, Dict[str, list]] = {}
    end_ms = 0.0
    drawn = 0
    for e in events:
        if e.sys not in row_of or e.issue_tick is None or e.callback_tick is None:
            continue
        cat = classify(e)
        lane = "compute" if e.ntype == 4 else "recv" if e.ntype == 6 else "sendcoll"
        start_tick = e.issue_tick
        note = ""
        if e.ntype == 6:                        # blocking RECV: never draw from tick 0
            start_tick, why = _recv_start(e)
            note = f"{why} · posted@{_ms(e.issue_tick * tick_ns):.3f} ms, no dep"
        x0 = _ms(start_tick * tick_ns)
        x1 = _ms(e.callback_tick * tick_ns)
        if x1 <= x0:                            # keep zero/1-tick ops findable when zoomed in
            x1 = x0 + _ms(max(e.callback_tick - start_tick, 1) * tick_ns)
        end_ms = max(end_ms, x1)
        y = row_of[e.sys] + LANES[lane]
        extra = note
        if e.ntype == 4 and e.op_intensity is not None and not math.isnan(e.op_intensity):
            extra = (f"opInt={e.op_intensity:.1f} · perf={e.perf:.1f} · "
                     f"compUtil={e.compute_util:.2f} · memUtil={e.memory_util:.2f}")
        cd = [e.name, CAT_LABEL.get(cat, cat), round(x0, 6), round(x1, 6),
              round(x1 - x0, 6), NTYPE.get(e.ntype, str(e.ntype)), extra]
        d = seg.setdefault(cat, {"x": [], "y": [], "cd": []})
        d["x"] += [x0, x1, None]
        d["y"] += [y, y, None]
        d["cd"] += [cd, cd, SEP_CD]
        drawn += 1

    data = []
    for cat, _col, label in CATEGORIES:
        if cat not in seg:
            continue
        d = seg[cat]
        data.append({
            "type": "scattergl", "mode": "lines", "name": label,
            "x": d["x"], "y": d["y"], "customdata": d["cd"],
            "line": {"color": CAT_COLOR[cat], "width": LINE_W},
            "connectgaps": False, "hoverlabel": {"namelength": -1},
            "hovertemplate": (
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} · %{customdata[5]}<br>"
                "start %{customdata[2]:.4f} ms · end %{customdata[3]:.4f} ms · "
                "dur %{customdata[4]:.4f} ms<br>%{customdata[6]}<extra></extra>"
            ),
        })

    end_of_prefill_ms = _ms((lat.get("ttft_ns") or 0) * tick_ns)
    end_ms = end_ms or end_of_prefill_ms or 1.0

    # First DECODE token completion = end of decode iteration it=0 (last layer ffw).
    # Two naming schools coexist: under Splitwise/DistServe the prefill already emits
    # the first token (so this is the 2nd token); under the "decode pool emits every
    # output token" convention THIS is the user-visible TTFT. We mark both boundaries
    # and annotate the gap, which is precisely the disaggregation tax on getting that
    # next token out: KV-cache transfer + first-token handoff + the first (cold) decode
    # step, including its first all-reduce barrier.
    token_ends = lat.get("token_end_ns") or []
    first_decode_ms = _ms(token_ends[0] * tick_ns) if token_ends else None

    # Subsequent decode token-completion markers as ONE webgl trace (cheap), not N
    # shapes. it=0 is skipped here: it gets its own labelled "TTFT" line below.
    tail_ends = token_ends[1:5000]
    if tail_ends:
        tx, ty = [], []
        for tk in tail_ends:
            xm = _ms(tk * tick_ns)
            tx += [xm, xm, None]
            ty += [-0.6, n - 0.4, None]
        data.append({
            "type": "scattergl", "mode": "lines", "name": "token done",
            "x": tx, "y": ty, "connectgaps": False, "hoverinfo": "skip",
            "line": {"color": "rgba(0,128,0,0.45)", "width": 1, "dash": "dot"},
        })

    ticktext = [f"sys[{s}] · {topo.pool.get(s,'?')} "
                f"st{topo.stage.get(s,'?')}·sh{topo.shard.get(s,'?')}" for s in ordered]

    shapes = []
    annotations = []
    if end_of_prefill_ms > 0:                  # boundary 1: end of prefill (KV ready)
        shapes.append({"type": "line", "x0": end_of_prefill_ms, "x1": end_of_prefill_ms,
                       "y0": -0.6, "y1": n - 0.4, "yref": "y",
                       "line": {"color": "black", "width": 1.2, "dash": "dash"}})
        annotations.append({"x": end_of_prefill_ms, "y": 1.0, "yref": "paper",
                            "text": f"End of prefill {end_of_prefill_ms:.2f} ms",
                            "showarrow": False, "xanchor": "left", "font": {"size": 11}})
    if first_decode_ms is not None and first_decode_ms > 0:   # boundary 2: 1st decode token
        shapes.append({"type": "line", "x0": first_decode_ms, "x1": first_decode_ms,
                       "y0": -0.6, "y1": n - 0.4, "yref": "y",
                       "line": {"color": "#006400", "width": 1.6}})
        annotations.append({"x": first_decode_ms, "y": 1.0, "yref": "paper",
                            "text": f"TTFT {first_decode_ms:.2f} ms", "showarrow": False,
                            "xanchor": "left", "font": {"size": 11, "color": "#006400"}})
        # Gap = delay introduced by KV-cache transfer + handoff + first decode step.
        if end_of_prefill_ms > 0 and first_decode_ms > end_of_prefill_ms:
            gap = first_decode_ms - end_of_prefill_ms
            mid = 0.5 * (end_of_prefill_ms + first_decode_ms)
            shapes.append({"type": "line", "x0": end_of_prefill_ms, "x1": first_decode_ms,
                           "y0": -0.5, "y1": -0.5, "xref": "x", "yref": "y",
                           "line": {"color": "#d62728", "width": 1.2}})
            annotations.append({"x": mid, "y": -0.5, "xref": "x", "yref": "y",
                                "text": f"KV transfer + 1st decode = {gap:.2f} ms",
                                "showarrow": False, "xanchor": "center", "yanchor": "bottom",
                                "font": {"size": 10, "color": "#d62728"}})

    layout = {
        "title": {"text": f"Interactive inference Gantt ({drawn:,} tasks, WebGL) — "
                          "drag to pan · scroll to zoom · prefill stages then decode"
                          "<br><sub>RECV bars show the transfer window (anchored to the "
                          "matching SEND); the idle wait before arrival is left blank</sub>"},
        "height": max(420, 80 + 34 * n), "hovermode": "closest", "dragmode": "pan",
        "xaxis": {"title": {"text": "time (ms)"}, "range": [0, end_ms * 1.01],
                  "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)", "zeroline": False},
        "yaxis": {"tickvals": list(range(n)), "ticktext": ticktext,
                  "range": [n - 0.4, -0.6], "autorange": False, "fixedrange": False},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        "shapes": shapes, "annotations": annotations,
        "margin": {"l": 175, "r": 30, "t": 110, "b": 40},
        "plot_bgcolor": "white", "paper_bgcolor": "white",
        "updatemenus": [{
            "type": "buttons", "direction": "right", "x": 0, "y": 1.16,
            "xanchor": "left", "yanchor": "bottom", "showactive": False, "pad": {"r": 6},
            "buttons": [
                {"label": "Full", "method": "relayout", "args": [{"xaxis.range": [0, end_ms * 1.01]}]},
                {"label": "Prefill", "method": "relayout",
                 "args": [{"xaxis.range": [0, (end_of_prefill_ms * 1.02) if end_of_prefill_ms else end_ms]}]},
                {"label": "Decode", "method": "relayout",
                 "args": [{"xaxis.range": [(end_of_prefill_ms * 0.999) if end_of_prefill_ms else 0, end_ms * 1.01]}]},
            ],
        }],
    }

    config = {"scrollZoom": True, "displaylogo": False, "responsive": True,
              "toImageButtonOptions": {"format": "png", "filename": "inference_gantt"}}
    return data, layout, config


def fig_interactive_gantt(events, topo, tick_ns, lat, out_html):
    """Write the self-contained, zoomable Gantt to `out_html`.

    Uses the `plotly` python package when available (it can inline the JS for
    fully offline use); otherwise falls back to a hand-written page that loads
    plotly.js from the CDN, so no extra install is required.
    """
    data, layout, config = _interactive_gantt_spec(events, topo, tick_ns, lat)
    try:
        import plotly.graph_objects as go            # noqa: WPS433 (lazy, optional)
    except ImportError:
        go = None
    if go is not None:
        go.Figure(data=data, layout=layout).write_html(
            str(out_html), include_plotlyjs="cdn", full_html=True, config=config)
    else:
        Path(out_html).write_text(_PLOTLY_HTML_TEMPLATE.format(
            data=json.dumps(data), layout=json.dumps(layout), config=json.dumps(config)))


def fig_decode(events, topo, lat, tick_ns, out):
    sys_ids = topo.decode_sys
    ttft = lat["ttft_ns"]
    ends = lat["token_end_ns"] or []
    g_end = max((e.callback_tick for e in events
                 if e.sys in sys_ids and e.callback_tick is not None), default=ttft)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 2.0 + 0.55 * len(sys_ids)),
                                   gridspec_kw={"height_ratios": [len(sys_ids), 4]})
    _gantt(ax1, events, sys_ids, topo, tick_ns,
           xlim=(_ms(ttft * tick_ns) - 1, _ms(g_end * tick_ns) + 1),
           title="Decode timeline (starts at end of prefill) — token completions marked")
    for e in ends:
        ax1.axvline(_ms(e * tick_ns), color="green", lw=0.8, ls=":", alpha=0.7)

    tbt = lat["tbt_ns"]
    if tbt:
        xs = lat["decode_iterations"][1:]
        ys = [_ms(t * tick_ns) for t in tbt]
        span = (max(ys) - min(ys)) or 0.0
        dec = 4 if span == 0 else max(4, int(math.ceil(-math.log10(span))) + 2)
        ax2.plot(xs, ys, "o-", color=CAT_COLOR["decode_compute"])
        for xv, yv in zip(xs, ys):
            ax2.annotate(f"{yv:.{dec}f}", (xv, yv), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7)
        mean = sum(ys) / len(ys)
        ax2.axhline(mean, color="gray", ls="--", lw=1)
        ax2.text(xs[-1], mean, f" TPOT≈{mean:.3f} ms", va="bottom", ha="right", fontsize=8, color="gray")
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) or (hi * 0.05 or 1)
        ax2.set_ylim(lo - pad * 0.6, hi + pad * 1.1)
    else:
        ax2.text(0.5, 0.5, "no decode TBT series", ha="center", va="center")
    ax2.set_xlabel("decode iteration (output token index)")
    ax2.set_ylabel("time between\ntokens (ms)")
    ax2.set_title("TPOT — inter-token latency per decode step", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_comm_breakdown(aggs, topo, tick_ns, out):
    comm_cats = ["prefill_allreduce", "decode_allreduce", "pp", "kv_transfer", "first_token"]
    sys_ids = sorted(aggs)
    x = np.arange(len(sys_ids))
    width = 0.8 / len(comm_cats)
    fig, ax = plt.subplots(figsize=(2 + 1.0 * len(sys_ids), 5.2))
    for i, cat in enumerate(comm_cats):
        vals = [_ms(union(aggs[s].cat_cost_intervals.get(cat, [])) * tick_ns) for s in sys_ids]
        if not any(vals):
            continue
        ax.bar(x + i * width, vals, width, label=CAT_LABEL[cat], color=CAT_COLOR[cat])
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([f"sys[{s}]\n{aggs[s].pool}" for s in sys_ids], fontsize=7)
    ax.set_ylabel("send/collective time (ms, wall-clock union)")
    ax.set_title("Communication cost by category, per node (producer/collective side;\n"
                 "blocking RECV waits excluded — they are bubbles, shown in the Gantt)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dumps
# --------------------------------------------------------------------------- #

def dump_events_csv(events, out):
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sys", "node_id", "name", "category", "ntype_kind", "op",
                    "issue_tick", "callback_tick", "duration", "op_intensity",
                    "compute_util", "memory_util"])
        for e in sorted(events, key=lambda e: (e.sys, e.issue_tick or 0)):
            w.writerow([e.sys, e.node_id, e.name, classify(e), NTYPE.get(e.ntype, e.ntype),
                        parse_op(e.name), e.issue_tick, e.callback_tick, e.duration,
                        e.op_intensity, e.compute_util, e.memory_util])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name", nargs="?", default=None,
                    help="run subfolder under output/astra_logs/ (legacy layout)")
    ap.add_argument("--trace", type=Path, default=None, help="path to workload_trace.log")
    ap.add_argument("--stats", type=Path, default=None, help="path to statistics.log")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--tick-ns", type=float, default=1.0, help="ns per ASTRA-sim tick (default 1.0)")
    ap.add_argument("--ref-ttft-ms", type=float, default=None,
                    help="reference TTFT (e.g. aiconfigurator) drawn for comparison")
    args = ap.parse_args(argv)

    if args.trace:
        trace_path, stats_path = args.trace, args.stats
        out_dir = args.out or Path("result_graphs")
    else:
        run = args.run_name or "."
        log_dir = Path("output/astra_logs") / run
        trace_path = log_dir / "workload_trace.log"
        stats_path = log_dir / "statistics.log"
        out_dir = args.out or (Path("result_graphs") / run)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = parse_trace(trace_path)
    stats = parse_statistics(stats_path) if stats_path and Path(stats_path).exists() else {}
    topo = infer_topology(events)
    aggs = aggregate(events, topo)
    lat = latency_metrics(events, topo)
    decomp = ttft_decomposition(aggs, topo)
    tn = args.tick_ns

    # figures
    fig_phase_overview(lat, topo, aggs, tn, out_dir / "fig_phase_overview.png")
    fig_ttft_breakdown(decomp, lat, tn, out_dir / "fig_ttft_breakdown.png", args.ref_ttft_ms)
    fig_prefill_timeline(events, topo, tn, lat["ttft_ns"], out_dir / "fig_prefill_timeline.png")
    # Per-stage detail PNGs are superseded by one zoomable HTML Gantt: open it
    # and box-/scroll-zoom into any stage (prefill or decode) at any scale.
    fig_interactive_gantt(events, topo, tn, lat, out_dir / "gantt_interactive.html")
    fig_decode(events, topo, lat, tn, out_dir / "fig_decode_tpot.png")
    fig_comm_breakdown(aggs, topo, tn, out_dir / "fig_comm_breakdown.png")

    dump_events_csv(events, out_dir / "events.csv")
    summary = {
        "tick_ns": tn,
        "topology": {
            "prefill_stages": topo.prefill_stages,
            "decode_sys": topo.decode_sys,
            "tp_per_prefill_stage": {st: len(s) for st, s in topo.prefill_stages.items()},
            "tp_decode": len(topo.decode_sys),
        },
        "latency_ms": {
            "ttft": _ms(lat["ttft_ns"] * tn) if lat["ttft_ns"] else None,
            "tpot_steady": _ms(lat["tpot_steady_ns"] * tn) if lat["tpot_steady_ns"] else None,
            "tpot_mean": _ms(lat["tpot_mean_ns"] * tn) if lat["tpot_mean_ns"] else None,
            "first_token_to_decode_gap": _ms(lat["first_token_to_decode_gap_ns"] * tn)
            if lat["first_token_to_decode_gap_ns"] else None,
            "tbt_series": [_ms(t * tn) for t in lat["tbt_ns"]],
        },
        "ttft_decomposition_ms": {
            "compute": _ms(decomp["compute_ns"] * tn),
            "allreduce": _ms(decomp["allreduce_ns"] * tn),
            "compute_frac": decomp["compute_frac"],
            "allreduce_frac": decomp["allreduce_frac"],
            "per_stage": [{k: (_ms(v * tn) if k.endswith("_ns") else v) for k, v in ps.items()}
                          for ps in decomp["per_stage"]],
        },
        "astrasim_stats": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # console report
    p = print
    p("=" * 78)
    p(f"Parsed {len(events)} nodes across {len(aggs)} sys.   1 tick = {tn} ns")
    p(f"Topology: prefill PP={len(topo.prefill_stages)} stages x TP="
      f"{len(next(iter(topo.prefill_stages.values()))) if topo.prefill_stages else 0}, "
      f"decode TP={len(topo.decode_sys)}")
    p("-" * 78)
    p("Reconciliation parsed-trace vs statistics.log (comm now INCLUDES type-7 all-reduce):")
    for line in reconcile(aggs, stats):
        p("  " + line)
    p("-" * 78)
    p(f"TTFT (end of prefill)            : {_ms(lat['ttft_ns']*tn):.3f} ms")
    p(f"   ├─ compute on critical path   : {_ms(decomp['compute_ns']*tn):.3f} ms "
      f"({decomp['compute_frac']*100:.1f}%)")
    p(f"   └─ TP all-reduce on path      : {_ms(decomp['allreduce_ns']*tn):.3f} ms "
      f"({decomp['allreduce_frac']*100:.1f}%)")
    for ps in decomp["per_stage"]:
        p(f"      stage {ps['stage']}: compute {_ms(ps['compute_ns']*tn):.2f} ms + "
          f"all-reduce {_ms(ps['allreduce_ns']*tn):.2f} ms")
    if args.ref_ttft_ms:
        p(f"   reference TTFT                 : {args.ref_ttft_ms:.3f} ms  "
          f"(gap {_ms(lat['ttft_ns']*tn)-args.ref_ttft_ms:+.2f} ms)")
    p("-" * 78)
    if lat["tpot_steady_ns"]:
        p(f"TPOT (steady inter-token)        : {_ms(lat['tpot_steady_ns']*tn):.4f} ms / token  "
          f"({len(lat['token_end_ns'])} tokens)")
        p(f"first-token -> 1st decode gap     : {_ms(lat['first_token_to_decode_gap_ns']*tn):.3f} ms")
    p("-" * 78)
    p(f"Figures + CSV/JSON written to: {out_dir}/")
    p(f"Interactive Gantt (zoomable)  : {out_dir}/gantt_interactive.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())