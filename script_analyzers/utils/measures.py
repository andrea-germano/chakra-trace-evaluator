#!/usr/bin/env python3
"""
utils.measures — the per-run measures more than one analyzer reports.

buffer_sweep used to own these and incast_sweep / cc_sweep imported them from
it, which kept "one definition per metric" but made a sweep script the library
of its siblings. They live here now: everything below is a MEASUREMENT of one
run — a number read out of the ASTRA frame or the ns-3 logs — with no opinion
about what it means. Interpretation (which figure, which axis, what counts as
a finding) stays in the analyzers, per utils/__init__.py.

Two sources, deliberately split:

    ASTRA stats CSV   ttft_from, barrier, kv_rank_series, kv_skew_stats,
                      decode_ar_stats, decode_stall_stats — per-op ticks,
                      already labelled by op/stage/iteration, identical to the
                      ns-3 instants to the nanosecond but with none of the
                      flow-classification heuristics.
    ns-3 logs         pause_stats, victim_pause_intervals, link_metrics —
                      queues, PFC and per-physical-link stats, which the CSVs
                      cannot express (they carry no path information).

Warnings go through utils.cli.warn — the same single stream the analyzers
drain at the end of their main().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import astra, intervals, ns3
from . import flows as flowlib
from .cli import need, warn
from .fabric import Bottleneck, Topology
from .roles import Placement

NAN = float("nan")


# --------------------------------------------------------------------------- #
# ASTRA-side measures
# --------------------------------------------------------------------------- #
def ttft_from(adf: pd.DataFrame, tag: str) -> float:
    """TTFT = the first token, produced at the END OF PREFILL (FIRSTTOK send,
    NOT DECFB -- DECFB is the second token, one decode pipeline late).

    The instant and its fallback chain live in astra.end_of_prefill (one
    definition for every sweep); this only adds the warning voice. Takes the
    ALREADY-READ frame -- the caller reads the run once and shares it."""
    inst, src = astra.end_of_prefill(adf)
    if src == "prefill_comp":
        warn(f"{tag}: no FIRSTTOK in the ASTRA trace; using the last prefill "
             f"compute end as end-of-prefill TTFT.")
    elif inst is None:
        warn(f"{tag}: no FIRSTTOK and no prefill COMP in the ASTRA trace; TTFT "
             f"unavailable.")
        return NAN
    return float(inst)


def barrier(kv: pd.DataFrame, placement: Placement) -> dict:
    """The first decode step cannot start until every KV flow feeding a decode
    rank has arrived."""
    out = {"decode_ranks": ",".join(map(str, placement.decode_ranks))}
    ready, dur = {}, {}
    for d in placement.decode_ranks:
        arr = kv.loc[kv["dst"] == d, "arrival"]
        if len(arr):
            ready[d] = float(arr.max())
            dur[d] = float(arr.max() - arr.min())
    need(ready, f"no KV arrival for any declared decode rank "
                f"{placement.decode_ranks} in the ASTRA trace: either "
                f"--placement is wrong, or the trace's KV recv roles could not "
                f"be resolved (astra.kv_arrivals came back empty).")
    if len(ready) < len(placement.decode_ranks):
        warn(f"only {len(ready)}/{len(placement.decode_ranks)} decode ranks "
             f"receive KV; the barrier is over {sorted(ready)}.")
    out["kv_gate_ns"] = max(ready.values())
    out["kv_ready_min_ns"] = min(ready.values())
    out["cross_rank_skew_ns"] = max(ready.values()) - min(ready.values())
    out["kv_stream_duration_ns"] = max(dur.values())
    return out


def kv_rank_series(kv: pd.DataFrame, placement: Placement) -> dict:
    """rank -> (arrival_times_ns, cumulative_bytes), sorted by arrival. The
    raw material for the cumulative-KV-arrival figures: skew is the horizontal
    spread between ranks' curves, smoothness is whether each curve ramps or
    stair-steps with flats."""
    out = {}
    for d in placement.decode_ranks:
        sub = kv.loc[kv["dst"] == d].sort_values("arrival")
        if not len(sub):
            continue
        out[int(d)] = (sub["arrival"].to_numpy(dtype=float),
                       np.cumsum(sub["size"].to_numpy(dtype=float)))
    return out


def kv_skew_stats(kv_arr: pd.DataFrame) -> tuple[dict, dict, dict]:
    """The KV skew figures' raw material, all from the ASTRA KV recv rows
    (utils.astra.kv_arrivals, which carries stage/shard/layer from the name).

    Returns (scalars, rank_span, rank_layer_delta):

        scalars           kv_tp_skew_{min,mean,p99}_ns and kv_tp_skew_n over the
                          per-(stage, layer) TP-GROUP skew population: for each
                          (decode stage, layer), max-min arrival across the KV
                          shards feeding that TP group. That spread is exactly
                          the wait the layer's first decode all-reduce inherits.
                          Groups with fewer than 2 shard arrivals carry no skew
                          and are excluded.
        rank_span         rank -> last-first KV arrival at that rank (ns): the
                          total transfer time each rank observes.
        rank_layer_delta  rank -> completion(highest layer) - completion(lowest
                          layer) at that rank (ns), SIGNED: layers arrive out of
                          order, so this is not the span -- a negative value
                          means the last layer's KV landed before the first's.
                          A layer delivered in several chunks is complete at its
                          last chunk.
    """
    scal = {"kv_tp_skew_min_ns": NAN, "kv_tp_skew_mean_ns": NAN,
            "kv_tp_skew_p99_ns": NAN, "kv_tp_skew_n": 0}
    rank_span: dict[int, float] = {}
    rank_delta: dict[int, float] = {}
    if kv_arr is None or kv_arr.empty:
        return scal, rank_span, rank_delta

    for d, g in kv_arr.groupby("dst"):
        arr = g["arrival"].astype(float)
        rank_span[int(d)] = float(arr.max() - arr.min())
        gl = g.dropna(subset=["layer"])
        if len(gl):
            per_layer = gl.groupby("layer")["arrival"].max()
            rank_delta[int(d)] = float(per_layer[per_layer.index.max()]
                                       - per_layer[per_layer.index.min()])

    grouped = kv_arr.dropna(subset=["stage", "layer"]) \
                    .groupby(["stage", "layer"])["arrival"] \
                    .agg(["min", "max", "count"])
    sk = (grouped["max"] - grouped["min"])[grouped["count"] >= 2]
    if len(sk):
        scal.update(kv_tp_skew_min_ns=float(sk.min()),
                    kv_tp_skew_mean_ns=float(sk.mean()),
                    kv_tp_skew_p99_ns=float(sk.quantile(0.99)),
                    kv_tp_skew_n=int(len(sk)))
    return scal, rank_span, rank_delta


def decode_ar_stats(adf: pd.DataFrame | None) -> dict[int, dict]:
    """First vs steady-state TP all-reduce of each DECODE PP stage, from the
    ASTRA stats CSV: decode stage -> {first_skew_ns, first_dur_ns,
    rest_skew_mean_ns, rest_dur_mean_ns, n}.

    Per collective (ss, L, it, op): ENTRY SKEW is the spread of the shards'
    start ticks -- how staggered the TP ranks entered the collective -- and
    duration is the slowest shard's. The FIRST collective of a stage (earliest
    entry) is the one gated by that stage's own KV transfer, so its entry skew
    is the KV skew the decode pipeline actually inherits; the rest of the
    stage's collectives are the steady-state control. Empty dict with no ASTRA
    run or no decode TP (TP=1)."""
    out: dict[int, dict] = {}
    if adf is None or adf.empty:
        return out
    tp = adf[(adf["op_class"] == "TP") & (adf["phase"] == "decode")]
    if not len(tp) or "ss" not in tp.columns:
        return out
    keys = [c for c in ("ss", "L", "it", "op") if c in tp.columns]
    g = (tp.groupby(keys, dropna=False)
           .agg(start=("start_tick", "min"), start_max=("start_tick", "max"),
                dur=("duration", "max")).reset_index())
    g["skew"] = g["start_max"] - g["start"]
    g["ss"] = pd.to_numeric(g["ss"], errors="coerce")
    for st, gg in g.dropna(subset=["ss"]).groupby("ss"):
        gg = gg.sort_values("start")
        first, rest = gg.iloc[0], gg.iloc[1:]
        out[int(st)] = {
            "first_skew_ns": float(first["skew"]),
            "first_dur_ns": float(first["dur"]),
            "rest_skew_mean_ns": float(rest["skew"].mean()) if len(rest) else NAN,
            "rest_dur_mean_ns": float(rest["dur"].mean()) if len(rest) else NAN,
            "n": int(len(gg)),
        }
    return out


def decode_stall_stats(adf: pd.DataFrame | None,
                       kv_arr: pd.DataFrame) -> tuple[dict, dict[int, dict]]:
    """How much the decode is stalled waiting for its KV, from the ASTRA CSV.

    The decode does NOT gate on the KV barrier: stage 0 wakes when the FIRSTTOK
    message ARRIVES (its send instant is the TTFT, but the message itself queues
    behind the KV bulk), and each later stage wakes on the it=0 PP-decode
    activation. A stage then consumes its KV layer by layer, so it stalls only
    where it outruns the transfer. Two views of that stall:

    scalars
        dec_start_ns      first decode COMP start (= stage 0's wake)
        tok2_ns           the second token: the first DECFB send, max over
                          shards (the feedback is ready once the slowest shard
                          sent -- same barrier logic as firsttok_send_instant)
        tok2_latency_ns   tok2 - dec_start: wall-clock of the first decode pass
        itl_steady_ns     mean gap between the remaining DECFB iterations, the
                          steady-state inter-token control: the first pass's
                          excess over it is the KV-induced stall
    per stage
        input_arrival_ns  when the stage COULD start (FIRSTTOK arrival for
                          stage 0, it=0 PP-decode activation arrival after)
        kv_ready_ns       last KV arrival for the stage
        kv_lateness_ns    kv_ready - input_arrival: >0 = the stage outruns its
                          KV and must stall inside the first pass; <=0 = the KV
                          was already resident (the transfer is fully masked)
    """
    scal = {"dec_start_ns": NAN, "tok2_ns": NAN, "tok2_latency_ns": NAN,
            "itl_steady_ns": NAN}
    stages: dict[int, dict] = {}
    if adf is None or adf.empty:
        return scal, stages

    comp = adf[(adf["op_class"] == "COMP") & (adf["phase"] == "decode")]
    if len(comp):
        scal["dec_start_ns"] = float(comp["start_tick"].min())

    fb = adf[(adf["op_class"] == "DECFB") & (adf["comm_role"] == "send")]
    if len(fb) and "it" in fb.columns:
        inst = (fb.assign(it=pd.to_numeric(fb["it"], errors="coerce"))
                  .dropna(subset=["it"])
                  .groupby("it")["start_tick"].max().sort_index())
        if len(inst):
            scal["tok2_ns"] = float(inst.iloc[0])
            if pd.notna(scal["dec_start_ns"]):
                scal["tok2_latency_ns"] = scal["tok2_ns"] - scal["dec_start_ns"]
            gaps = inst.diff().dropna()
            if len(gaps):
                scal["itl_steady_ns"] = float(gaps.mean())

    kv_ready = {}
    if kv_arr is not None and len(kv_arr) and kv_arr["stage"].notna().any():
        kv_ready = kv_arr.dropna(subset=["stage"]).groupby("stage")["arrival"] \
                         .max().to_dict()

    inputs: dict[int, float] = {}
    ft = adf[(adf["op_class"] == "FIRSTTOK") & (adf["comm_role"] == "recv")]
    if len(ft):
        inputs[0] = float(ft["end_tick"].max())
    ppd = adf[(adf["op_class"] == "PP") & (adf["phase"] == "decode")
              & (adf["comm_role"] == "recv")]
    if len(ppd):
        w0 = ppd.assign(it=pd.to_numeric(ppd["it"], errors="coerce"),
                        ds=pd.to_numeric(ppd["ds"], errors="coerce")) \
                .dropna(subset=["it", "ds"])
        w0 = w0[w0["it"] == 0]
        for ds, g in w0.groupby("ds"):
            inputs[int(ds)] = float(g["end_tick"].max())

    for st in sorted(set(kv_ready) | set(inputs)):
        ready = float(kv_ready.get(st, NAN))
        arr = float(inputs.get(st, NAN))
        stages[int(st)] = {"input_arrival_ns": arr, "kv_ready_ns": ready,
                           "kv_lateness_ns": ready - arr}
    return scal, stages


# --------------------------------------------------------------------------- #
# ns-3-side measures
# --------------------------------------------------------------------------- #
def union_len(spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Covered length of the union of `spans` clipped to [lo, hi]. The union
    algebra is utils.intervals; only the clip window is this function's own."""
    clipped = [(max(s, lo), min(e, hi)) for s, e in spans if min(e, hi) > max(s, lo)]
    return int(intervals.union_len(clipped))


def pause_stats(pfc: ns3.PfcLog, bn: Bottleneck, topo: Topology,
                lo: int, hi: int) -> dict:
    """Backpressure on the ingress side of `bn`, over [lo, hi]. Kept
    link-generic so it works for every candidate link, not only one."""
    span = max(hi - lo, 1)
    victims = set(bn.pause_victims(topo))

    frames_bn = frames_total = 0
    devices_paused: set[tuple[int, int]] = set()
    for (node, _nt, ifidx, _q), events in pfc.events.items():
        n_in_win = sum(1 for t, typ in events if typ == 1 and lo <= t <= hi)
        frames_total += n_in_win
        if (node, ifidx) in victims:
            frames_bn += n_in_win
            if n_in_win:
                devices_paused.add((node, ifidx))

    iv = pfc.pause_intervals(clamp_to=hi)
    per_dev: dict[tuple[int, int], list] = {}
    for (node, _nt, ifidx, _q), spans in iv.items():
        if (node, ifidx) in victims:
            per_dev.setdefault((node, ifidx), []).extend(spans)
    pct = 0.0
    if per_dev:
        best = max(per_dev, key=lambda k: union_len(per_dev[k], lo, hi))
        pct = 100.0 * union_len(per_dev[best], lo, hi) / span

    return {"pause_frames_bn": float(frames_bn),
            "pause_frames_total": float(frames_total),
            "paused_devices": float(len(devices_paused)),
            "pause_pct_of_window": pct}


def victim_pause_intervals(pfc: ns3.PfcLog, bn: Bottleneck, topo: Topology,
                           clamp_to: int) -> list[tuple[int, int]]:
    """Raw PAUSE intervals (not unioned) on `bn`'s ingress victims, for
    shading a timeline. pause_stats reduces the same population to one %
    number; this keeps the intervals themselves."""
    victims = set(bn.pause_victims(topo))
    out: list[tuple[int, int]] = []
    for (node, _nt, ifidx, _q), spans in pfc.pause_intervals(clamp_to=clamp_to).items():
        if (node, ifidx) in victims:
            out.extend(spans)
    return out


@dataclass
class LinkStat:
    """One candidate KV-crossed link's stats for one run -- scored the way a
    single bottleneck is (window/floor/efficiency, queue occupancy, PFC pauses),
    plus concurrency, which a single-link analysis never needed with only one
    link to look at."""
    label: str = ""
    switch: int = -1
    egress_port: int = -1
    peer: int = -1
    rate_gbps: float = NAN
    f_ports: int = 0
    kv_bytes: float = NAN
    window_ns: float = NAN
    floor_ns: float = NAN
    delivered_gbps: float = NAN
    eff_pct: float = NAN                  # floor/window, %: delivered vs the hard floor
    qpeak_bytes: float = NAN
    qmean_bytes: float = NAN
    qpeak_pct: float = NAN                # qpeak_bytes / this run's buffer_bytes, %
    conc_peak: float = NAN                # most concurrent KV flows at once
    conc_mean: float = NAN                # mean concurrency each flow actually saw
    pause_frames: float = NAN
    pause_pct_of_window: float = NAN


def link_metrics(kv: pd.DataFrame, bn: Bottleneck, topo: Topology,
                 pfc: ns3.PfcLog, qlen: ns3.QlenLog, buffer_bytes: float) -> LinkStat:
    ls = LinkStat(label=str(bn), switch=bn.switch, egress_port=bn.egress_port,
                 peer=bn.peer, rate_gbps=bn.rate / 1e9, f_ports=bn.f_ports)
    kv_bn = kv[flowlib.crosses(kv, bn)]
    if not len(kv_bn):
        return ls
    lo, hi = int(kv_bn["start"].min()), int(kv_bn["arrival"].max())
    ls.window_ns = hi - lo
    ls.kv_bytes = float(kv_bn["size"].sum())
    ls.floor_ns = ls.kv_bytes * 8e9 / bn.rate
    if ls.window_ns > 0:
        ls.delivered_gbps = ls.kv_bytes * 8.0 / ls.window_ns
        ls.eff_pct = 100 * ls.floor_ns / ls.window_ns
    ls.qpeak_bytes = float(qlen.port_max.get((bn.switch, bn.egress_port), NAN))
    ls.qmean_bytes = float(qlen.port_mean.get((bn.switch, bn.egress_port), NAN))
    if buffer_bytes and pd.notna(ls.qpeak_bytes):
        ls.qpeak_pct = 100 * ls.qpeak_bytes / buffer_bytes
    ls.conc_peak, ls.conc_mean = flowlib.concurrency_stats(flowlib.flow_spans(kv_bn))
    pstats = pause_stats(pfc, bn, topo, lo, hi)
    ls.pause_frames = pstats["pause_frames_bn"]
    ls.pause_pct_of_window = pstats["pause_pct_of_window"]
    return ls
