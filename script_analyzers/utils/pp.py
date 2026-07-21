#!/usr/bin/env python3
"""
utils.pp — pipeline-parallel activation arrival skew, measured on the fabric.

Why this module exists
--------------------------------------------------------------------------------
buffer_sweep measures the *effect* of the buffer (KV-arrival gate, cross-rank KV
skew) but not its *cause*. The causal driver, established by hand on the T1 run,
is upstream of the KV: the skew with which a prefill stage's PP activation
arrives at the ranks of the NEXT stage. That skew Delta feeds the ring all-reduce
of the receiving stage (RS gated on the local wake, AG on max(Delta, W)) and only
then propagates to KV readiness. The buffer moves Delta via the PFC/DCQCN regime
on the oversubscribed uplink; Delta moves everything else. So Delta(buffer) is the
curve that explains the sweep, and it is not drawn today.

What a PP activation is, on T1, and how it is told apart from TP
--------------------------------------------------------------------------------
Between prefill ranks fct.txt holds two very different populations:

    41 943 040 B, 80 per ordered pair, hops == 1   -> TP all-reduce chunks
                                                      (INTRA-stage, dedicated link)
    83 886 080 B,  1 per ordered pair, hops  > 1   -> PP activation handoff
                                                      (INTER-stage, over the fabric)

The PP activation is the INTER-stage flow: source rank in stage i, destination
rank in stage i+1. utils.flows already labels these 'pp_prefill'; on top of that
we keep only the ones that actually cross stages (src_stage != dst_stage), which
drops any same-stage flow that a coarse role check would let through. The 40 MiB
TP chunks are hops==1 and classed 'tp', so they never enter here — but the
inter-stage test is the belt-and-braces that makes this robust if a future
topology puts a TP group behind a switch.

The skew, precisely
--------------------------------------------------------------------------------
A wave is the set of PP activations that feed the SAME destination stage in the
SAME iteration. For that wave:

    Delta = max_r arrival[r] - min_r arrival[r]     over the destination ranks

the synchronisation cost: how long the earliest-fed rank of the receiving stage
waits for the latest before its all-reduce can progress. On T1 there is one wave
per (dst_stage, it) — e.g. {0->2, 1->3} feeding stage 1 — so Delta is exactly the
rank-2-vs-rank-3 arrival gap we traced by hand (44 us at 16 MB, 163 us at 8 MB).
Multiple waves (deeper pipelines, more iterations) are reported per wave and the
worst is the headline.

Measured on the fabric, never in the schedule: arrival = start + fct from
fct.txt. The ASTRA schedule is not used (pre-posted RECVs sit at tick 0). Nothing
is fitted; empty input yields empty output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .roles import Placement

NAN = float("nan")


@dataclass
class PPResult:
    """PP arrival-skew summary for one run. Scalars go on a buffer_sweep Row;
    the two frames are kept for the figures and never flattened into the CSV."""
    available: bool = False
    skew_ns: float = NAN            # worst wave cross-rank arrival skew (headline)
    skew_mean_ns: float = NAN       # mean skew over waves
    first_ns: float = NAN           # earliest arrival of the worst wave
    last_ns: float = NAN            # latest arrival of the worst wave
    stage: object = None            # destination stage of the worst wave
    n_waves: int = 0
    waves: object = field(default=None)     # per-wave DataFrame
    arrivals: object = field(default=None)  # per-flow DataFrame


def _stage_of_rank(placement: Placement) -> dict[int, int]:
    """rank -> prefill stage index (only prefill ranks appear)."""
    m: dict[int, int] = {}
    for stage_idx, ranks in enumerate(placement.prefill):
        for r in ranks:
            m[int(r)] = stage_idx
    return m


def pp_prefill_arrivals(flows: pd.DataFrame, placement: Placement) -> pd.DataFrame:
    """One row per INTER-stage PP-prefill activation, fabric-timed.

    `flows` is utils.flows.annotate output. We keep flow_class == 'pp_prefill'
    (drops tp, kv, ctrl), then require src_stage != dst_stage so only true
    inter-stage activations remain. A per-wave index is attached: the flows into
    one (dst_stage) are ordered by start and numbered, so the k-th activation
    into each destination rank of a stage forms wave k. On T1 (one activation
    per stage) this makes exactly one wave; deeper pipelines get one per handoff.

    Columns: src, dst (ranks), src_stage, dst_stage, wave, start, arrival (ns).
    Empty frame with those columns when there is no inter-stage pp_prefill flow.
    """
    cols = ["src", "dst", "src_stage", "dst_stage", "wave", "start", "arrival"]
    if flows is None or flows.empty or "flow_class" not in flows.columns:
        return pd.DataFrame(columns=cols)
    pp = flows[flows["flow_class"] == "pp_prefill"].copy()
    if pp.empty:
        return pd.DataFrame(columns=cols)

    stage = _stage_of_rank(placement)
    pp["src_stage"] = pp["src"].map(stage)
    pp["dst_stage"] = pp["dst"].map(stage)
    pp = pp.dropna(subset=["src_stage", "dst_stage"])
    pp = pp[pp["src_stage"] != pp["dst_stage"]]        # inter-stage only
    if pp.empty:
        return pd.DataFrame(columns=cols)
    pp["src_stage"] = pp["src_stage"].astype(int)
    pp["dst_stage"] = pp["dst_stage"].astype(int)

    # Wave index: within one destination stage, the k-th activation each of its
    # ranks receives (ordered by send start) belongs to wave k. Ranks of a stage
    # get the same number of activations, so this aligns them for the skew.
    pp = pp.sort_values("start")
    pp["wave"] = pp.groupby(["dst_stage", "dst"]).cumcount()
    return pp[cols].reset_index(drop=True)


def wave_skew(pp_arr: pd.DataFrame) -> pd.DataFrame:
    """Cross-rank arrival skew per (dst_stage, wave).

    Columns: dst_stage, wave, skew_ns, first_ns, last_ns, n_ranks. One row per
    activation wave; this is the finest true skew fct.txt supports. Empty frame
    on empty input.
    """
    cols = ["dst_stage", "wave", "skew_ns", "first_ns", "last_ns", "n_ranks"]
    if pp_arr is None or pp_arr.empty:
        return pd.DataFrame(columns=cols)
    recs = []
    for (s, w), g in pp_arr.groupby(["dst_stage", "wave"]):
        arr = g["arrival"].astype(float)
        recs.append(dict(dst_stage=int(s), wave=int(w),
                         skew_ns=float(arr.max() - arr.min()),
                         first_ns=float(arr.min()), last_ns=float(arr.max()),
                         n_ranks=int(g["dst"].nunique())))
    return pd.DataFrame(recs, columns=cols).sort_values(["dst_stage", "wave"]) \
             .reset_index(drop=True)


def measure(flows: pd.DataFrame, placement: Placement) -> "PPResult":
    """PP skew summary for one run, as a typed result.

    The headline number is `skew_ns`: the worst wave's cross-rank PP arrival
    skew — the Delta that gates the receiving stage's all-reduce. Frames
    (`waves`, `arrivals`) ride along for plotting. `available` is False when no
    inter-stage PP flow could be timed (PP=1, or a single prefill stage); the
    caller warns and the PP figures are skipped.

    Returning a dataclass rather than a dict is deliberate: every field a Row
    stores is named here once, so a new field cannot leak onto Row through a
    blind setattr loop, and mypy/readers see the shape.
    """
    pp_arr = pp_prefill_arrivals(flows, placement)
    waves = wave_skew(pp_arr)
    if waves.empty:
        return PPResult(available=False, waves=waves, arrivals=pp_arr)
    top = waves.loc[waves["skew_ns"].idxmax()]
    return PPResult(
        available=True,
        skew_ns=float(top["skew_ns"]),
        skew_mean_ns=float(waves["skew_ns"].mean()),
        first_ns=float(top["first_ns"]),
        last_ns=float(top["last_ns"]),
        stage=int(top["dst_stage"]),
        n_waves=int(len(waves)),
        waves=waves,
        arrivals=pp_arr)