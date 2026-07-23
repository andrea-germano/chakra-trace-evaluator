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

What a PP activation is, and how it is identified
--------------------------------------------------------------------------------
The PP activation is the inter-stage handoff: a prefill rank in stage i sends its
output to a rank in stage i+1. In the ASTRA stats CSV it is an op with op_class
'PP' and phase 'prefill'; the op name carries `ss` (source stage), `ds` (dest
stage) and `it` (iteration) directly, so no size/hops discrimination against the
intra-stage TP chunks and no src!=dst guard are needed -- the classification the
fct.txt path had to reconstruct is already in the name.

The skew, precisely
--------------------------------------------------------------------------------
A wave is the set of PP activations that feed the SAME destination stage in the
SAME iteration `it`. For that wave:

    Delta = max_r arrival[r] - min_r arrival[r]     over the destination ranks

the synchronisation cost: how long the earliest-fed rank of the receiving stage
waits for the latest before its all-reduce can progress. On T1 there is one wave
per (dst_stage, it) — e.g. {0->2, 1->3} feeding stage 1. Multiple waves (deeper
pipelines, more iterations) are reported per wave and the worst is the headline.

Source: the ASTRA stats CSV (utils.astra.pp_arrivals), recv side -- sys_id is the
receiving rank and end_tick its arrival, which equals the ns-3 fct arrival
(start+fct) to the nanosecond but comes pre-classified by op/stage/iteration, so
the wave index is `it` itself rather than a sort-by-start cumcount heuristic. The
recv start_tick/duration are wait-dominated (pre-posted at the origin) and unused;
only end_tick is read. Nothing is fitted; empty input yields empty output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import astra

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


def measure(adf: pd.DataFrame) -> "PPResult":
    """PP skew summary for one run, as a typed result, from the ASTRA stats CSV.

    The headline number is `skew_ns`: the worst wave's cross-rank PP arrival
    skew — the Delta that gates the receiving stage's all-reduce. Frames
    (`waves`, `arrivals`) ride along for plotting. `available` is False when no
    inter-stage PP activation could be timed (PP=1, or a single prefill stage);
    the caller warns and the PP figures are skipped.

    Returning a dataclass rather than a dict is deliberate: every field a Row
    stores is named here once, so a new field cannot leak onto Row through a
    blind setattr loop, and mypy/readers see the shape.
    """
    pp_arr = astra.pp_arrivals(adf)
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