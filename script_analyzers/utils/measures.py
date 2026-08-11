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
