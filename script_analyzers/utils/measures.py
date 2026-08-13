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
                      kv_layer_skew, decode_ar_stats, decode_stall_stats —
                      per-op ticks, already labelled by op/stage/iteration,
                      identical to the ns-3 instants to the nanosecond but with
                      none of the flow-classification heuristics.
    ns-3 logs         pause_stats, victim_pause_intervals, link_metrics —
                      queues, PFC and per-physical-link stats, which the CSVs
                      cannot express (they carry no path information).

Plus one SWEEP-wide reading, knee_scalars: not a per-run measure but a reading
of the assembled per-run curve ("where does the buffer stop mattering"), shared
because buffer_sweep and incast_sweep ask it of the same columns — each sweep
declares its own pause column and saturation metric set and gets the same
knee definitions.

Warnings go through utils.cli.warn — the same single stream the analyzers
drain at the end of their main().
"""

from __future__ import annotations

import re
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


def kv_layer_skew(kv_arr: pd.DataFrame) -> pd.DataFrame:
    """KV arrival skew BETWEEN THE TP SHARDS OF ONE PP STAGE, per layer.

    One row per (decode PP stage, layer); columns stage, layer, skew_ns,
    signed_ns, n_shards.

    What is being compared, exactly. A decode rank is identified by the pair
    (PP stage, TP shard) -- on a d-tp2pp2 placement that is sys4=(0,0),
    sys5=(0,1), sys6=(1,0), sys7=(1,1). One layer's KV is split across the TP
    shards of ONE stage, and those shards are fed by different ranks over
    different paths, so they do not land together. Fixing the stage and the
    layer and varying ONLY the TP shard isolates that:

        skew(stage, layer) = max_shard arrival - min_shard arrival

    which at TP=2 is |arrival(shard 1) - arrival(shard 0)|: max-min over two
    values IS their absolute difference. Non-negative by construction at any TP
    width, so no abs() is needed. Nothing is compared across PP stages here (a
    stage's layers are its own, and the offset between stages is pipeline
    structure, not skew) and nothing is compared across layers.

    A shard's layer is complete at its LAST chunk, so the per-shard instant is a
    max over that shard's rows before the across-shard spread is taken. On the
    traces the sweeps read each (stage, layer, shard) is a single row and the
    reduction is a no-op -- it is here so that a chunked delivery would still
    measure shard-vs-shard rather than silently folding one shard's internal
    spread into the skew.

    signed_ns is defined only at TP=2, where the pair has an order:
    arrival(higher shard index) - arrival(lower). Its SIGN says which shard is
    late, and a population sitting on one side of zero is a systematic path
    asymmetry rather than random congestion. NaN at other TP widths, where "which
    one is late" has no single answer.

    Groups with fewer than 2 shards carry no skew and are dropped. This is the
    same population kv_skew_stats reduces to min/mean/p99, so the figures and
    the CSV scalars cannot disagree."""
    cols = ["stage", "layer", "skew_ns", "signed_ns", "n_shards"]
    if kv_arr is None or kv_arr.empty:
        return pd.DataFrame(columns=cols)
    k = kv_arr.dropna(subset=["stage", "layer", "shard"])
    if k.empty:
        return pd.DataFrame(columns=cols)
    # per (stage, layer, shard): that shard's layer is complete at its last chunk
    per_shard = k.groupby(["stage", "layer", "shard"])["arrival"].max()
    g = per_shard.groupby(level=["stage", "layer"]).agg(["min", "max", "count"])
    g = g[g["count"] >= 2]
    if g.empty:
        return pd.DataFrame(columns=cols)

    wide = per_shard.unstack("shard")           # columns = TP shard index
    shards = sorted(wide.columns)
    signed = (wide[shards[1]] - wide[shards[0]]
              if len(shards) == 2 else pd.Series(NAN, index=wide.index))

    out = g.reset_index()
    return pd.DataFrame({"stage": out["stage"].astype(int),
                         "layer": out["layer"].astype(int),
                         "skew_ns": (out["max"] - out["min"]).astype(float),
                         "signed_ns": signed.reindex(
                             pd.MultiIndex.from_arrays([out["stage"], out["layer"]])
                         ).to_numpy(dtype=float),
                         "n_shards": out["count"].astype(int)})


# Per-packet wire overhead of the ns-3 RDMA model (RdmaHw::GetNxtPacket):
# SeqTsHeader(6 + IntHeader) + UdpHeader(8) + Ipv4Header(20) + PppHeader(2).
# The INT header rides on every data packet and its size is the CC's, so two
# algorithms moving the same payload do NOT put the same number of bytes on the
# wire -- ignoring it charges HPCC ~1.6 ms of its own telemetry as if it were
# idle time. Sizes from src/network/utils/int-header.h: NORMAL is hop[5]*8 + 2,
# TS is one uint64, PINT is a uint16.
BASE_HEADER_BYTES = 6 + 8 + 20 + 2
INT_HEADER_BYTES = {1: 0,      # DCQCN   -> IntHeader::NONE
                    3: 5 * 8 + 2,   # HPCC      -> NORMAL, the 5-hop stack
                    7: 8,      # TIMELY  -> TS
                    8: 0,      # DCTCP   -> NONE
                    10: 2}     # HPCC-PINT -> PINT


def wire_header_bytes(cc_mode: int | None) -> float:
    """Bytes of header every data packet carries, for this run's CC_MODE. NaN on
    an unknown mode -- the caller then reports the payload-only floor rather
    than silently assuming a header size that would bias the measure."""
    if cc_mode is None or int(cc_mode) not in INT_HEADER_BYTES:
        return NAN
    return BASE_HEADER_BYTES + INT_HEADER_BYTES[int(cc_mode)]


def kv_handover_idle(kv_arr: pd.DataFrame, kv_sends: pd.DataFrame,
                     placement: Placement, rate_bps: float, payload: int,
                     header_bytes: float,
                     other_sends: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """How much of the KV handover a decode rank's link spent NOT SENDING.

    The transfer moves a fixed number of bytes over a link of fixed rate, so its
    DURATION carries no information about congestion: it is bytes/rate whatever
    the fan-in. What congestion produces is the gap between the window the
    transfer took and that hard floor, which is delay the decode inherits:

        window = last KV arrival at the rank - first send feeding it
        floor  = wire bytes / rate,  wire = payload + ceil(payload/MTU) * header
        idle   = window - floor

    Read entirely from the ASTRA CSV (arrival ticks and comm_size, already
    labelled by destination stage/shard) plus two declared constants, the link
    rate and the per-packet header. Verified against the ns-3 path: for the
    bottleneck link of T2.1/T3/T4 the window and the byte count agree with
    fct.txt + candidate_links to the NANOSECOND and to the BYTE, with none of
    the flow classification, path reconstruction or bottleneck-ranking the ns-3
    route needs -- and it reports every receiver instead of the one link a
    ranking happened to pick.

    PRECONDITION, which the caller must check: the receiver's own link is the
    narrowest on the path (rate_bps = min rate from every sender to it). That
    holds when the fan-in converges on the last hop, as it does here. If the
    pinch moves to a shared uplink, this measure no longer describes it and the
    ns-3 per-link route (link_metrics) is the one to use.

    THE IDLE IS NOT ALL INCAST, and `other_sends` is what splits it. A receiving
    link can only be fed by its own senders, so it is starved whenever EVERY
    sender feeding it is busy putting something else on its NIC. On a
    pipeline-parallel prefill that "something else" is the PP activation handed
    to the next stage, and the senders of one stage emit it simultaneously, so
    the receiver loses 100% of its supply for the duration however wide the
    fan-in is. Measured as the intersection, over the senders of a rank, of
    their non-KV send spans, clipped to the handover window:

        starved_ns = |intersection over senders of (their other transfers)|
        incast_ns  = idle_ns - starved_ns

    `other_sends` must already exclude control-sized rows: a FIRSTTOK send is a
    handful of bytes but its row spans tens of milliseconds of waiting, and
    counting that as "the sender is busy" would wipe the whole measure out.
    Without `other_sends` the split is not attempted and incast_ns is NaN.

    Returns (scalars, per_rank). Scalars are the WORST receiver by incast_ns --
    the term the fan-in is responsible for -- plus the means. floor is identical
    for every rank when the KV is evenly sharded, which is what makes the
    numbers comparable across topologies."""
    scal = {"kv_floor_ns": NAN, "kv_window_ns": NAN, "kv_idle_ns": NAN,
            "kv_idle_mean_ns": NAN, "kv_link_busy_pct": NAN,
            "kv_starved_ns": NAN, "kv_incast_ns": NAN,
            "kv_incast_mean_ns": NAN, "kv_starved_mean_ns": NAN}
    per_rank: dict[int, dict] = {}
    if (kv_arr is None or kv_arr.empty or kv_sends is None or kv_sends.empty
            or not rate_bps):
        return scal, per_rank

    # (stage, shard) -> rank, from the placement: it is what a KV name's ds/dsh
    # address, and what lets a send row be attributed to a receiver without
    # touching the topology.
    rank_of = {(si, sh): r
               for si, ranks in enumerate(placement.decode)
               for sh, r in enumerate(ranks)}
    s = kv_sends.assign(ds=pd.to_numeric(kv_sends.get("ds"), errors="coerce"),
                        dsh=pd.to_numeric(kv_sends.get("dsh"), errors="coerce"))
    s = s.dropna(subset=["ds", "dsh"])
    s["rank"] = [rank_of.get((int(a), int(b))) for a, b in zip(s["ds"], s["dsh"])]
    s = s.dropna(subset=["rank"])
    first_send = s.groupby("rank")["start_tick"].min()
    senders_of = {int(r): sorted(g["sys_id"].unique())
                  for r, g in s.groupby("rank")}
    busy_of: dict[int, list] = {}
    if other_sends is not None and len(other_sends):
        for sy, g in other_sends.groupby("sys_id"):
            busy_of[int(sy)] = intervals.merge(
                [(int(a), int(b)) for a, b in zip(g["start_tick"], g["end_tick"])])

    for d, g in kv_arr.groupby("dst"):
        d = int(d)
        if d not in first_send.index:
            continue
        payload_bytes = float(g["size"].sum())
        npkt = float(np.ceil(g["size"].to_numpy() / max(payload, 1)).sum())
        wire = payload_bytes + (npkt * header_bytes
                                if pd.notna(header_bytes) else 0.0)
        floor = wire * 8e9 / rate_bps
        lo, hi = float(first_send.loc[d]), float(g["arrival"].max())
        window = hi - lo
        starved = NAN
        if other_sends is not None:
            # every sender of d busy elsewhere at once = d's link has no supply
            cur = busy_of.get(senders_of.get(d, [None])[0], [])
            for sy in senders_of.get(d, [])[1:]:
                cur = _intersect(cur, busy_of.get(int(sy), []))
            cur = [(max(a, lo), min(b, hi)) for a, b in cur
                   if min(b, hi) > max(a, lo)]
            starved = float(intervals.union_len(cur))
        per_rank[d] = {"floor_ns": floor, "window_ns": window,
                       "idle_ns": window - floor, "starved_ns": starved,
                       "incast_ns": (window - floor - starved
                                     if pd.notna(starved) else NAN),
                       "payload_bytes": payload_bytes, "packets": npkt,
                       # the window's two ENDS, not only its length: when a
                       # receiver's handover starts is a property of the
                       # pipeline (its layers must be produced first), not of
                       # the fabric, and separating the two is the only way to
                       # read a completion time. start = first send feeding it,
                       # end = its last arrival, i.e. its own KV gate.
                       "start_ns": lo, "end_ns": hi}
    if not per_rank:
        return scal, per_rank
    key = ("incast_ns" if any(pd.notna(m["incast_ns"]) for m in per_rank.values())
           else "idle_ns")
    worst = max(per_rank.values(),
                key=lambda m: m[key] if pd.notna(m[key]) else -np.inf)
    scal["kv_floor_ns"] = worst["floor_ns"]
    scal["kv_window_ns"] = worst["window_ns"]
    scal["kv_idle_ns"] = worst["idle_ns"]
    scal["kv_starved_ns"] = worst["starved_ns"]
    scal["kv_incast_ns"] = worst["incast_ns"]
    scal["kv_idle_mean_ns"] = float(np.mean([m["idle_ns"]
                                             for m in per_rank.values()]))
    inc = [m["incast_ns"] for m in per_rank.values() if pd.notna(m["incast_ns"])]
    scal["kv_incast_mean_ns"] = float(np.mean(inc)) if inc else NAN
    stv = [m["starved_ns"] for m in per_rank.values() if pd.notna(m["starved_ns"])]
    scal["kv_starved_mean_ns"] = float(np.mean(stv)) if stv else NAN
    if worst["window_ns"] > 0:
        scal["kv_link_busy_pct"] = 100.0 * worst["floor_ns"] / worst["window_ns"]
    return scal, per_rank


def _intersect(a: list, b: list) -> list:
    """Intersection of two interval sets, as A minus (A minus B). utils.intervals
    has union, subtract and overlap length but no set intersection; this is the
    one line that composes it from what is there."""
    return intervals.subtract(a, intervals.subtract(a, b))


def decode_ar_stats(adf: pd.DataFrame | None) -> dict[int, dict]:
    """First vs steady-state TP all-reduce of each DECODE PP stage, from the
    ASTRA stats CSV: decode stage -> {first_skew_ns, first_dur_ns,
    rest_skew_mean_ns, rest_dur_mean_ns, first_bw, rest_bw_mean, n}.

    Per collective (ss, L, it, op): ENTRY SKEW is the spread of the shards'
    start ticks -- how staggered the TP ranks entered the collective -- and
    duration is the slowest shard's. The FIRST collective of a stage (earliest
    entry) is the one gated by that stage's own KV transfer, so its entry skew
    is the KV skew the decode pipeline actually inherits; the rest of the
    stage's collectives are the steady-state control. Empty dict with no ASTRA
    run or no decode TP (TP=1).

    EFFECTIVE BANDWIDTH (comm_size / duration, bytes/ns = GB/s) is reported
    beside the durations so the decode side can be read on the SAME axis the
    prefill all-reduce is read on (buffer_sweep's rs_allreduce_stats). The
    duration ratio first/steady is dominated by the idle skew wait rather than
    by a slower transfer, which is why the bandwidth framing exists at all: the
    stall shows as the effective bandwidth collapsing, with the ungated wire
    rate as the ceiling. Both are returned; the analyzer picks. NaN when the
    collective carries no comm_size."""
    out: dict[int, dict] = {}
    if adf is None or adf.empty:
        return out
    tp = adf[(adf["op_class"] == "TP") & (adf["phase"] == "decode")]
    if not len(tp) or "ss" not in tp.columns:
        return out
    keys = [c for c in ("ss", "L", "it", "op") if c in tp.columns]
    agg = {"start": ("start_tick", "min"), "start_max": ("start_tick", "max"),
           "dur": ("duration", "max")}
    if "comm_size" in tp.columns:
        agg["cs"] = ("comm_size", "first")
    g = tp.groupby(keys, dropna=False).agg(**agg).reset_index()
    g["skew"] = g["start_max"] - g["start"]
    g["ss"] = pd.to_numeric(g["ss"], errors="coerce")
    # bytes / wall-clock of the collective; only defined where it took time.
    g["bw"] = (g["cs"] / g["dur"].where(g["dur"] > 0)
               if "cs" in g.columns else NAN)
    for st, gg in g.dropna(subset=["ss"]).groupby("ss"):
        gg = gg.sort_values("start")
        first, rest = gg.iloc[0], gg.iloc[1:]
        out[int(st)] = {
            "first_skew_ns": float(first["skew"]),
            "first_dur_ns": float(first["dur"]),
            "rest_skew_mean_ns": float(rest["skew"].mean()) if len(rest) else NAN,
            "rest_dur_mean_ns": float(rest["dur"].mean()) if len(rest) else NAN,
            "first_bw": float(first["bw"]) if "cs" in gg.columns else NAN,
            "rest_bw_mean": (float(rest["bw"].mean())
                             if "cs" in gg.columns and len(rest) else NAN),
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


# A gap is "closed by" a KV delivery when the rank resumes within this much of
# the arrival. The traces put the two in the same nanosecond (the resumed op's
# start_tick IS the arrival's end_tick), so the window only has to absorb the
# scheduler, not a modelling difference.
KV_RESUME_TOL_NS = 10_000


def first_pass_stall(adf: pd.DataFrame | None, kv_arr: pd.DataFrame,
                     tok2_ns: float) -> tuple[dict, dict[int, dict]]:
    """What the first decode pass actually WAITS for -- measured as idle inside
    the pass, not inferred from the KV envelope.

    kv_gate - dec_start ("the KV still in flight when the pipeline wakes") is an
    UPPER BOUND on the stall and routinely a loose one, because the decode does
    not need all of its KV at once: it consumes layer by layer and stalls only
    where it outruns the transfer. On T1/16 requests that bound reads 0.25-4.1 ms
    while every decode rank runs its first pass strictly back to back -- zero
    idle, no stall at all. Reporting the bound as the stall would be reporting a
    cost nobody paid.

    So the stall is measured where it would have to appear: the rank is BUSY for
    the union of its own first-iteration decode COMP and TP intervals, and every
    hole between them is time it spent waiting. A hole is attributed to the KV
    when a KV arrival at that rank lands in it (within KV_RESUME_TOL_NS of the
    resume) -- that arrival is what let the rank continue. Holes no arrival
    explains are waits on something else (a slower TP shard, the PP hop) and are
    counted separately rather than folded into the KV bill.

    Busy = COMP and TP only. PP/DECFB recv rows are pre-posted at the run origin
    and carry no occupancy; the sends are instantaneous at this scale.

    The pipeline stall is the UNION of the ranks' blocked intervals, not the
    worst rank's total. TP shards re-synchronise at every layer's all-reduce, so
    the group advances only when NO shard is blocked, and two shards blocked at
    different layers hold the pipeline for the sum of both waits. Taking the
    worst rank instead understates it by 2x on a 64-request run (28.996 ms
    against a true 36.257 ms), while the union reproduces the makespan
    arithmetic: the union and the pass's excess over its own steady inter-token
    gap -- two independent measurements of the same wait, one from the ops' idle
    and one from the token timestamps -- agree within 0.2% on the buffer sweeps
    (36.495/36.429, 47.079/47.013, 5.357/5.333 ms, and 0/0 where the pass never
    waits) and within 3% on the incast levels (36.21/37.33 ms at T3). The
    residual is one-signed -- the measured value is never the larger -- because
    a shard that reaches an all-reduce and waits there for a KV-blocked peer
    spends that wait INSIDE the collective's duration, where this function reads
    it as busy. It is a floor on the stall, in other words, and a tight one.

    Returns (scalars, detail):
        scalars
            dec_kv_block_ns     union of the KV-blocked idle over all decode
                                ranks: the measured first-pass KV stall
            dec_other_idle_ns   idle the union leaves unexplained by any KV
                                arrival (a PP hop, a shard behind its peer)
            dec_kv_block_max_ns the WORST SINGLE RANK's KV-blocked idle. Kept
                                because it answers a different question -- how
                                much one rank personally waited -- and its
                                distance from the union says whether the shards
                                stall together or in turn.
            dec_crit_rank       the rank whose first DECFB send is tok2
        detail
            {"per_rank": rank -> {busy_ns, idle_ns, kv_block_ns, other_idle_ns,
                                  spans, gaps},
             "kv_blocked_spans": merged [(start, end)] of the KV-blocked idle,
             "idle_spans":       merged [(start, end)] of ALL idle}
            The spans are the raw material for a timeline figure, which needs
            the POSITIONS and not only the totals; gaps = [(start, end, is_kv)].
    """
    scal = {"dec_kv_block_ns": NAN, "dec_other_idle_ns": NAN,
            "dec_kv_block_max_ns": NAN, "dec_crit_rank": NAN}
    detail: dict = {"per_rank": {}, "kv_blocked_spans": [], "idle_spans": []}
    per_rank: dict[int, dict] = detail["per_rank"]
    if adf is None or adf.empty:
        return scal, detail

    ops = adf[(adf["phase"] == "decode")
              & (adf["op_class"].isin(("COMP", "TP")))]
    if ops.empty or "it" not in ops.columns:
        return scal, detail
    ops = ops.assign(it=pd.to_numeric(ops["it"], errors="coerce"))
    ops = ops[ops["it"] == 0]
    if ops.empty:
        return scal, detail

    arrivals = {}
    if kv_arr is not None and len(kv_arr):
        arrivals = {int(r): np.sort(g["arrival"].to_numpy(dtype=float))
                    for r, g in kv_arr.groupby("dst")}

    for rank, sub in ops.groupby("sys_id"):
        iv = sorted(zip(sub["start_tick"].astype(float),
                        sub["end_tick"].astype(float)))
        spans: list[list[float]] = []
        for a, b in iv:
            if spans and a <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], b)
            else:
                spans.append([a, b])
        arr = arrivals.get(int(rank), np.empty(0))
        gaps = []
        for (_, end), (start, _) in zip(spans, spans[1:]):
            if start - end <= KV_RESUME_TOL_NS:
                continue
            hit = arr[(arr >= end) & (arr <= start + KV_RESUME_TOL_NS)]
            gaps.append((end, start, bool(len(hit))))
        kv_block = sum(b - a for a, b, is_kv in gaps if is_kv)
        other = sum(b - a for a, b, is_kv in gaps if not is_kv)
        per_rank[int(rank)] = {
            "busy_ns": float(sum(b - a for a, b in spans)),
            "idle_ns": float(kv_block + other),
            "kv_block_ns": float(kv_block),
            "other_idle_ns": float(other),
            "spans": [(float(a), float(b)) for a, b in spans],
            "gaps": gaps,
        }
    if not per_rank:
        return scal, detail

    def merge(iv: list) -> list[tuple[float, float]]:
        out: list[list[float]] = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    blocked = merge([(a, b) for m in per_rank.values()
                     for a, b, is_kv in m["gaps"] if is_kv])
    idle_all = merge([(a, b) for m in per_rank.values() for a, b, _ in m["gaps"]])
    detail["kv_blocked_spans"] = blocked
    detail["idle_spans"] = idle_all
    scal["dec_kv_block_ns"] = float(sum(b - a for a, b in blocked))
    scal["dec_other_idle_ns"] = float(sum(b - a for a, b in idle_all)
                                      - scal["dec_kv_block_ns"])
    scal["dec_kv_block_max_ns"] = max(m["kv_block_ns"] for m in per_rank.values())
    # The critical rank: the one that emitted token 2. tok2 is the MAX over
    # shards of the first DECFB send, so the rank whose own first send is that
    # instant is the one everything else waited for. Reported, not used as the
    # stall -- see the union above.
    if pd.notna(tok2_ns):
        fb = adf[(adf["op_class"] == "DECFB") & (adf["comm_role"] == "send")]
        if len(fb) and "it" in fb.columns:
            fb = fb.assign(it=pd.to_numeric(fb["it"], errors="coerce"))
            fb = fb[(fb["it"] == 0) & (abs(fb["start_tick"] - tok2_ns) <= 1)]
            cand = [int(r) for r in fb["sys_id"].unique() if int(r) in per_rank]
            if cand:
                scal["dec_crit_rank"] = float(max(
                    cand, key=lambda r: per_rank[r]["kv_block_ns"]))
    return scal, detail


# --------------------------------------------------------------------------- #
# Sweep-wide readings (of the assembled per-run table, not of one run)
# --------------------------------------------------------------------------- #
# A knee is a READING of a curve, so its threshold is a named constant rather
# than a literal buried in a function: change it here and every sweep's figures,
# CSV and printed report move together.
PFC_KNEE_FRAC = 0.01        # PAUSE count <= 1% of the smallest-buffer value
STALL_ONSET_FRAC = 0.10     # first decode pass > steady ITL by more than 10%


def decode_worst_stage(s: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-stage decN_* columns an analyzer flattened (decode_ar_stats
    and decode_stall_stats, one column block per decode stage) to the WORST
    decode stage, plus the run-wide decode-stall scalar, so every cross-model /
    cross-topology reader has one stage-count-independent number per run:

        dec_ar_first_skew_ns    max over stages of the entry skew into the
                                stage's FIRST (KV-gated) TP all-reduce.
        dec_ar_first_bw         MIN over stages of that collective's effective
                                bandwidth -- the worst case is the slowest, and
                                this is the decode twin of rs_ar_first_bw, so the
                                two sides of the run are read on one axis.
        dec_ar_first_over_rest  max over stages of first duration / that same
                                stage's steady-state mean. Kept in the CSV for
                                continuity but read alongside dec_ar_first_bw:
                                the ratio is dominated by the idle wait at the
                                barrier, not by a slower wire.
        dec_kv_lateness_ns      max over stages of (KV ready - first-input
                                arrival): how late the KV is at the stage that
                                waits most for it.
        decode_kv_stall_ns      tok2_latency - itl_steady: the first decode
                                pass's excess over the steady inter-token gap,
                                i.e. the wall-clock the pipeline actually spent
                                stalled on KV in the first pass.

    Stage numbering differs between models and between incast topologies
    (different PP/TP), which is why the reduction happens here, once, and not in
    each caller. Absent column blocks give NaN columns rather than an error: a
    sweep with no decode TP all-reduce (TP=1) or no DECFB (no second token) still
    gets the schema, filled with NaN."""
    def _cols(pattern: str) -> list[str]:
        return [c for c in s.columns if re.fullmatch(pattern, c)]

    fsk = _cols(r"dec\d+_ar_first_skew_ns")
    s["dec_ar_first_skew_ns"] = (s[fsk].max(axis=1) if fsk else NAN)

    fbw = _cols(r"dec\d+_ar_first_bw")
    s["dec_ar_first_bw"] = (s[fbw].min(axis=1) if fbw else NAN)
    rbw = _cols(r"dec\d+_ar_rest_bw")
    s["dec_ar_rest_bw"] = (s[rbw].mean(axis=1) if rbw else NAN)

    stages = sorted(int(m.group(1)) for c in s.columns
                    if (m := re.fullmatch(r"dec(\d+)_ar_first_skew_ns", c)))
    ratios = [s[f"dec{st}_ar_first_dur_ns"] / s[f"dec{st}_ar_rest_dur_mean_ns"]
              for st in stages]
    s["dec_ar_first_over_rest"] = (pd.concat(ratios, axis=1).max(axis=1)
                                   if ratios else NAN)

    lat = _cols(r"dec\d+_kv_lateness_ns")
    s["dec_kv_lateness_ns"] = (s[lat].max(axis=1) if lat else NAN)

    if {"tok2_latency_ns", "itl_steady_ns"} <= set(s.columns):
        s["decode_kv_stall_ns"] = s["tok2_latency_ns"] - s["itl_steady_ns"]
    else:
        s["decode_kv_stall_ns"] = NAN
    return s


def knee_scalars(s: pd.DataFrame, pause_col: str,
                 saturation_metrics: tuple, x_col: str = "buffer_mb") -> dict:
    """The three x values at which something stops (or starts) happening, read
    off ONE sweep's assembled summary (one buffer axis: the caller slices per
    topology/model before calling).

        knee_pfc_mb         backpressure is gone: `pause_col` has fallen to
                            <= PFC_KNEE_FRAC of its value at the smallest x.
                            NaN if the sweep never pauses.
        knee_stall_mb       the decode-side cost has appeared: the first decode
                            pass exceeds the steady inter-token gap by more than
                            STALL_ONSET_FRAC. NaN if it never does.
        knee_saturation_mb  nothing changes any more: from here up, every
                            `saturation_metrics` column matches the largest-x run
                            within tolerance. Runs past this point cost
                            simulation time and add no information.

    `pause_col` and `saturation_metrics` are the caller's, because what counts
    as "the" PAUSE count and "the run is unchanged" is a per-sweep declaration:
    buffer_sweep watches its bottleneck link, incast_sweep the whole fabric.
    Each entry of saturation_metrics is (column, rtol, atol); columns absent
    from `s` are skipped rather than failing the match.

    All three read the curve, none of them fits it."""
    out = {"knee_pfc_mb": NAN, "knee_stall_mb": NAN, "knee_saturation_mb": NAN}
    d = s.sort_values(x_col)
    if d.empty:
        return out

    pf = d[pause_col] if pause_col in d.columns else None
    if pf is not None and pf.notna().any():
        base = float(pf.dropna().iloc[0])
        if base > 0:
            hit = d.loc[pf <= PFC_KNEE_FRAC * base, x_col]
            if len(hit):
                out["knee_pfc_mb"] = float(hit.iloc[0])

    if {"decode_kv_stall_ns", "itl_steady_ns"} <= set(d.columns):
        rel = d["decode_kv_stall_ns"] / d["itl_steady_ns"]
        hit = d.loc[rel > STALL_ONSET_FRAC, x_col]
        if len(hit):
            out["knee_stall_mb"] = float(hit.iloc[0])

    # Saturation: walk up from the smallest buffer and take the first value from
    # which the WHOLE tail is indistinguishable from the largest-buffer run. The
    # last row is skipped: a one-row tail matches the reference by construction
    # (it IS the reference), so returning the largest buffer would report a knee
    # for a sweep that never actually flattened. NaN there means "not reached
    # within the swept range" -- extend the sweep to find out.
    ref = d.iloc[-1]
    for i in range(len(d) - 1):
        tail = d.iloc[i:]
        if all(_matches_ref(tail, ref, col, rtol, atol)
               for col, rtol, atol in saturation_metrics if col in d.columns):
            out["knee_saturation_mb"] = float(tail[x_col].iloc[0])
            break
    return out


def _matches_ref(tail: pd.DataFrame, ref: pd.Series, col: str,
                 rtol: float, atol: float) -> bool:
    """Every value of `col` in `tail` within tolerance of the reference run's.
    A NaN reference is only matched by an all-NaN tail: 'both unmeasured' is
    agreement, 'one unmeasured' is not."""
    r = float(ref[col]) if pd.notna(ref[col]) else NAN
    v = pd.to_numeric(tail[col], errors="coerce")
    if pd.isna(r):
        return not v.notna().any()
    return bool(np.all(np.abs(v - r) <= atol + rtol * abs(r)))


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
