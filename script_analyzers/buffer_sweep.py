#!/usr/bin/env python3
"""
buffer_sweep — what does the per-switch buffer actually buy, and where is the
bill paid?

One sweep at a time, one workload at a time: the buffer is the only knob that
moves, and every figure below asks what that knob reaches. The cross-model
overlay is buffer_compare.py, which calls analyse_sweep from here so a model is
scored identically whether it is analysed alone or beside others.

The chain this sweep follows
--------------------------------------------------------------------------------
The buffer does not add bandwidth; it changes the congestion REGIME on the
oversubscribed uplink. From there two causal chains run in opposite directions,
and keeping them apart is the whole point of the figure set:

    PREFILL   buffer -> PFC/PAUSE -> PP arrival skew -> the receiving stage's
              FIRST (skew-gated) all-reduce -> TTFT.                     (fig 01)

    DECODE    buffer -> when PFC stops pacing the senders -> the decode pipeline
              is released earlier while the KV stream stays as long -> the first
              decode pass stalls on KV that is still in flight.     (figs 02, 05)

More buffer helps the first chain and hurts the second. Every "does it reach the
user?" panel is drawn so that inversion is visible rather than inferred.

Four things this file measures that the previous version did not
--------------------------------------------------------------------------------
1.  THE PER-LAYER SKEW AS A DISTRIBUTION, NOT THREE SUMMARY LINES (fig 03). For
    each (decode stage, layer) the KV shards of that layer land apart by
    max-min = |arrival(shard 0) - arrival(shard 1)| at TP=2 -- the wait that
    layer's first all-reduce inherits. Plotting min/mean/p99 vs buffer hid that
    the population is wide and heavy-tailed at EVERY buffer (T1: a median near
    10-17 ms with layers from 0.03 to 85 ms), so a mean that moves 20% is not a
    distribution that moves. Boxplots per (buffer, stage), fliers kept.

    Two other spreads exist and are NOT this one; both stay in summary.csv and
    neither is plotted, because they were read as "the skew" and are not:
      cross_rank_skew_ns  max-min over decode RANKS of each rank's LAST KV
                          arrival -- i.e. the gap between the rank that finished
                          first and the rank that finished last. On T1 that is
                          ~250 ms and nearly buffer-independent, because it is
                          dominated by the pipeline offset between decode stages
                          (stage 0 completes ~230 ms before stage 1), not by
                          congestion. That offset is visible directly in fig 02,
                          where it belongs; as a scalar labelled "skew" it
                          invited the reading that the buffer fails to fix a
                          250 ms misalignment, when there is nothing to fix.
      xrank_over_tp_skew  the ratio of the two. Kept for continuity only.

2.  THE KV TAIL PAST THE DECODE START (kv_tail_after_dec_start_ns, figs 02/05).
    kv_gate - dec_start: how much KV is still on the wire when the decode
    pipeline wakes. This is the CAUSE of decode_kv_stall_ns, which the old
    figure set reported as an effect with no cause column beside it. It is the
    same quantity astra_analyzer reports as kv_tail_vs_decode_start_ns, so the
    time-domain view and the sweep agree.

3.  THE STEADY-STATE INTER-TOKEN GAP AS AN EXPLICIT CONTROL (fig 05). itl_steady
    was only ever a denominator. It is invariant across the sweep (T1: 0.08% over
    2..64 MiB), and showing that invariance is what licenses the claim that the
    buffer touches the transient and nothing else. A control that is never drawn
    is a control the reader has to take on faith.

4.  BUFFER BLOAT (fig 10). qpeak alone always grows -- it is bounded by the knob
    being swept. Against qmean it says whether the added buffer absorbs SUSTAINED
    load or only rare spikes: on T1 qmean grows 1.75x while qpeak grows 6.7x, so
    the occupancy/peak ratio collapses 0.43 -> 0.11 and the extra megabytes are
    standing idle between excursions.

And one premise corrected
--------------------------------------------------------------------------------
The gap between TTFT and the start of decode is NOT the cost of PFC (fig 04).
T2, which emits zero PAUSE frames in its entire sweep, has a LARGER gap (304 ms)
than T1 at its most congested (292 ms at 2 MiB); and within T1 the gap GROWS
(300 -> 328 ms) while the PAUSE count collapses (57311 -> 0). The gap is KV bytes
divided by the bandwidth available to move them -- a serialisation cost that PFC
modulates by a few percent. Figure 04 plots it against the PAUSE count precisely
so the non-correlation is on the page instead of being assumed away.

Three knees, and they do not coincide
--------------------------------------------------------------------------------
Reported as sweep-wide scalars (constant columns in summary.csv, vertical rules
on the figures) because "where does it stop mattering" is a different question
from "how much does it move":

    knee_pfc_mb          smallest buffer at which the bottleneck's PAUSE count
                         has fallen to <=1% of its smallest-buffer value.
    knee_stall_mb        smallest buffer at which the first decode pass exceeds
                         the steady inter-token gap by >10% -- the ONSET of the
                         decode-side cost, which on T1 arrives while thousands
                         of PAUSE frames are still being emitted.
    knee_saturation_mb   smallest buffer beyond which the run stops changing at
                         all (every metric in SATURATION_METRICS within
                         tolerance of the largest-buffer run). Past it the sweep
                         is measuring nothing.

Figures
--------------------------------------------------------------------------------
    01  CAUSAL CHAIN TO TTFT        PP arrival skew -> the receiving stage's
                                    first (gated) all-reduce -> its steady-state
                                    all-reduces and the first prefill stage's
                                    (the two ungated references) -> TTFT.
    02  KV CUMULATIVE ARRIVAL       one panel per decode STAGE, every buffer
                                    overlaid and colour-graded, each run's decode
                                    start marked on the baseline: KV drawn right
                                    of a run's own marker is KV the pipeline is
                                    already waiting on.
    03  KV TP-SHARD SKEW            per (PP stage, layer): |arrival(shard 1) -
                                    arrival(shard 0)| between the two TP shards
                                    of that stage. Boxplots, one box per
                                    (buffer, stage) over the stage's layers.
    04  TTFT -> TOKEN 2             a waterfall per buffer: handoff in flight |
                                    decode awake with its KV still arriving |
                                    first pass completing, with the PAUSE count
                                    beside each row.
    05  DECODE GATED BY KV          the first decode pass against the steady
                                    control, and beside it the stall overlaid
                                    with its cause (KV tail past decode start,
                                    worst-stage KV lateness): the curves
                                    coincide, which is the finding.
    06  DECODE ALL-REDUCE           figure 01's analysis, on the decode side:
                                    EFFECTIVE BANDWIDTH of each stage's first
                                    (KV-gated) collective against its own steady
                                    state. Same framing as the prefill panels on
                                    purpose -- the duration ratio measures the
                                    idle wait, not a slower transfer.
    07  BOTTLENECK BUFFER(t)        occupancy of the bottleneck switch as % of
                                    BUFFER_SIZE, PFC PAUSE spans shaded.
    08  QUEUE(t) PER SWITCH         the same, as a grid (rows = switch, columns =
                                    buffer): which switches queue at all.
    09  PER-LINK CONGESTION         efficiency, PAUSE count and peak queue for
                                    EVERY KV-crossed link, not just the measured
                                    bottleneck -- on T1 the two ToR->core uplinks
                                    both saturate and the downlinks never pause,
                                    which one link0 line cannot show.
    10  BUFFER BLOAT                peak vs mean occupancy in MB, and the ratio.

Everything is measured, nothing fitted: fct.txt / pfc.txt / qlen.txt for the
fabric, this run's ASTRA-sim stats CSV for every compute-side instant.

Declared, never inferred:
    --sweep       the one path input; every other path is derived (utils.paths).
    --placement   the rank->role map (utils.roles).
    --bottleneck  optional 'sw->peer' to force which link is ground truth; it
                  must be among the links this sweep's KV flows cross.
    --top-links   how many KV-crossed links figures 09 and summary.csv carry
                  (default 6). The link SET is topology-derived and identical at
                  every buffer value of one sweep -- only its congestion ranking
                  can shift, so the set and its display order are fixed once,
                  from the smallest-buffer run, never re-ranked per row.

Usage
-----
    python3 buffer_sweep.py --sweep buffer_sweep_T1
    python3 buffer_sweep.py --sweep buffer_sweep_T2 --top-links 4 -o /tmp/x
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import astra
from utils import flows as flowlib
from utils import ns3, paths, pp, roles
from utils.cli import Abort, drain_warnings, need, warn
from utils.fabric import parse_ns3_config, parse_topology
from utils.measures import (LinkStat, barrier, decode_ar_stats,
                            decode_stall_stats, decode_worst_stage,
                            knee_scalars, kv_layer_skew, kv_skew_stats,
                            link_metrics, ttft_from, victim_pause_intervals)
from utils.plots import (BLUE, CORAL, GREEN, KNEE_STYLE, MS, MUTED, VIOLET,
                         buf_colour, downsample_max, logx_pow2, mark_knees,
                         save_fig, zoom_y)
from utils.roles import Placement
from utils.paths import BUFFER_AXIS, fresh_dir

NAN = float("nan")
KIND = "buffer"
US = 1e-3                       # ns -> µs (MS, ns -> ms, comes from utils.plots)

# The knee READINGS (and their thresholds) live in utils.measures.knee_scalars,
# shared with incast_sweep; what stays here is this sweep's declaration of WHICH
# columns they are read from -- the bottleneck link's PAUSE count, and the set
# below.
PAUSE_KNEE_COL = "link0_pause_frames"

# Saturation: (column, relative tolerance, absolute tolerance). A run is
# "identical to the largest-buffer run" when every one of these agrees. PAUSE is
# a COUNT, so it gets an absolute tolerance (half a frame) and no relative one --
# a relative tolerance is meaningless once the count reaches zero.
SATURATION_METRICS = (
    ("ttft_ns", 1e-3, 0.0),
    ("kv_gate_ns", 1e-3, 0.0),
    ("tok2_latency_ns", 1e-3, 0.0),
    ("cross_rank_skew_ns", 1e-3, 0.0),
    ("link0_qpeak_bytes", 1e-3, 0.0),
    ("link0_pause_frames", 0.0, 0.5),
)


# --------------------------------------------------------------------------- #
# One row per run. (LinkStat, the per-link score, lives in utils.measures.)
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    tag: str = ""
    buffer_mb: float = NAN
    buffer_bytes: float = NAN
    bottleneck: str = ""
    bn_rate_gbps: float = NAN

    # -- 01 prefill causal chain ---------------------------------------------- #
    ttft_ns: float = NAN                  # token 1 = END OF PREFILL (FIRSTTOK send)
    pp_skew_ns: float = NAN               # worst-wave cross-rank PP arrival skew
    pp_skew_mean_ns: float = NAN
    pp_first_ns: float = NAN
    pp_last_ns: float = NAN
    pp_stage: object = None               # destination stage of the worst wave
    pp_n_waves: int = 0
    # All-reduce metrics come from the ASTRA stats CSV (authoritative per-collective
    # duration/bytes), for the receiving stage's PREFILL TP all-reduce.
    rs_ar_first_ns: float = NAN           # duration of the GATED all-reduce
    rs_ar_rest_mean_ns: float = NAN       # mean duration of the steady-state ones
    rs_ar_first_bw: float = NAN           # gated all-reduce eff. bw (bytes/ns = GB/s)
    rs_ar_rest_bw: float = NAN            # steady-state eff. bw (mean)
    rs_ar_first_stage_bw: float = NAN     # first prefill stage's AR: ungated reference
    rs_ar_n: int = 0                      # receiving-stage prefill TP collectives

    # -- 02/03 KV delivery: when it lands, and how unevenly -------------------- #
    kv_gate_ns: float = NAN               # last KV arrival over all decode ranks
    kv_ready_min_ns: float = NAN          # first rank to be fully fed
    kv_gate_over_ttft: float = NAN        # decode-ready instant as a multiple of TTFT
    cross_rank_skew_ns: float = NAN       # kv_gate - kv_ready_min: the WHOLE spread
    kv_stream_duration_ns: float = NAN
    decode_ranks: str = ""
    # Population: one skew per (decode stage, layer) = the spread between the
    # arrivals of the KV shards feeding the same TP group -- the wait that layer's
    # first decode all-reduce inherits. Groups with a single arrival are excluded.
    kv_tp_skew_min_ns: float = NAN
    kv_tp_skew_mean_ns: float = NAN
    kv_tp_skew_p99_ns: float = NAN
    kv_tp_skew_n: int = 0
    # SIGNED median of the same population (TP=2 only): arrival(shard 1) -
    # arrival(shard 0), median over the stage's layers. Near zero = the two
    # shards are equally (un)lucky and the skew is congestion noise; away from
    # zero = one shard is systematically late, i.e. a path asymmetry.
    kv_shard_bias_ns: float = NAN
    # The scale gap between the two skews above, as one number. >>1 means the
    # buffer-sensitive within-group skew is a small part of the total spread.
    xrank_over_tp_skew: float = NAN

    # -- 04 from the first token to the decode -------------------------------- #
    dec_start_ns: float = NAN             # first decode COMP start (stage 0 wakes
                                          # on the FIRSTTOK ARRIVAL, not its send)
    dec_start_after_ttft_ns: float = NAN  # dec_start - ttft: the exposed handoff
    kv_gate_after_ttft_ns: float = NAN    # kv_gate - ttft: until the LAST KV lands

    # -- 05 decode stalled by KV reception ------------------------------------ #
    kv_tail_after_dec_start_ns: float = NAN  # kv_gate - dec_start: KV still in
                                          # flight when the pipeline wakes -- the
                                          # cause of the first-pass stall
    tok2_ns: float = NAN                  # second token: first DECFB send, max
                                          # over shards (slowest shard's send)
    tok2_latency_ns: float = NAN          # tok2 - dec_start: the first decode pass
    tok2_after_tok1_ns: float = NAN       # tok2 - TTFT: the user-visible gap
    itl_steady_ns: float = NAN            # mean gap of the remaining DECFB
                                          # iterations: the steady-state CONTROL

    # -- 10 buffer bloat ------------------------------------------------------ #
    q_bloat_ratio: float = NAN            # link0 qmean/qpeak: sustained vs spiky

    # -- not flattened: per-figure raw data ----------------------------------- #
    links: list = field(default_factory=list)            # list[LinkStat]
    kv_stage_series: dict = field(default_factory=dict)  # stage -> (times, cumbytes)
    kv_layer_skew: object = None                         # per (stage, layer) skew
    kv_rank_span: dict = field(default_factory=dict)     # rank -> last-first arrival
    dec_ar: dict = field(default_factory=dict)           # decode stage -> first/steady
                                                         # all-reduce skew/dur/bw
    dec_stall: dict = field(default_factory=dict)        # decode stage -> KV readiness
                                                         # vs first-input arrival
    bn_pause_intervals: list = field(default_factory=list)
    qseries: dict = field(default_factory=dict)          # sw -> (ts_ns, bytes)
    qswitch_peak: dict = field(default_factory=dict)
    qswitch_mean: dict = field(default_factory=dict)
    pause_intervals: dict = field(default_factory=dict)  # sw -> [(start, end)]

    def flat(self) -> dict:
        d = asdict(self)
        for k in ("links", "kv_stage_series", "kv_layer_skew", "kv_rank_span",
                  "dec_ar", "dec_stall", "bn_pause_intervals", "qseries",
                  "qswitch_peak", "qswitch_mean", "pause_intervals"):
            d.pop(k, None)
        for rank in sorted(self.kv_rank_span):
            d[f"kv_rank{rank}_span_ns"] = self.kv_rank_span[rank]
        for st in sorted(self.dec_ar):
            m = self.dec_ar[st]
            d[f"dec{st}_ar_first_skew_ns"] = m["first_skew_ns"]
            d[f"dec{st}_ar_first_dur_ns"] = m["first_dur_ns"]
            d[f"dec{st}_ar_rest_skew_mean_ns"] = m["rest_skew_mean_ns"]
            d[f"dec{st}_ar_rest_dur_mean_ns"] = m["rest_dur_mean_ns"]
            d[f"dec{st}_ar_first_bw"] = m["first_bw"]
            d[f"dec{st}_ar_rest_bw"] = m["rest_bw_mean"]
        for st in sorted(self.dec_stall):
            m = self.dec_stall[st]
            d[f"dec{st}_input_arrival_ns"] = m["input_arrival_ns"]
            d[f"dec{st}_kv_ready_ns"] = m["kv_ready_ns"]
            d[f"dec{st}_kv_lateness_ns"] = m["kv_lateness_ns"]
        for i, ls in enumerate(self.links):
            d[f"link{i}_label"] = ls.label
            d[f"link{i}_window_ns"] = ls.window_ns      # KV window: the normaliser
            d[f"link{i}_eff_pct"] = ls.eff_pct          # for the raw PAUSE count
            d[f"link{i}_delivered_gbps"] = ls.delivered_gbps
            d[f"link{i}_conc_peak"] = ls.conc_peak
            d[f"link{i}_conc_mean"] = ls.conc_mean
            d[f"link{i}_qpeak_pct"] = ls.qpeak_pct
            d[f"link{i}_qpeak_bytes"] = ls.qpeak_bytes    # absolute occupancy:
            d[f"link{i}_qmean_bytes"] = ls.qmean_bytes    # comparable in MB across
                                                          # runs, unlike qpeak_pct
                                                          # (% of the swept buffer)
            d[f"link{i}_pause_frames"] = ls.pause_frames
            d[f"link{i}_pause_pct_of_window"] = ls.pause_pct_of_window
        return d


# --------------------------------------------------------------------------- #
# Measurement helpers. The ones shared with incast_sweep/cc_sweep (barrier,
# kv_skew_stats, decode_ar_stats, decode_stall_stats, ttft_from, link_metrics,
# victim_pause_intervals) live in utils.measures; what stays here is this
# sweep's own question.
# --------------------------------------------------------------------------- #
def rs_allreduce_stats(adf: pd.DataFrame | None, placement: Placement, ppr) -> dict:
    """The prefill TP all-reduce, read from the ASTRA stats CSV (the authoritative
    per-collective duration and bytes -- the ns-3 fct.txt only sees the on-wire
    bursts, which under-count the collective's wall-clock ~10x).

    Reported as EFFECTIVE BANDWIDTH, comm_size / duration in bytes/ns (= GB/s,
    the CSV's own bw_bytes_per_ns), for three all-reduces:

        rs_ar_first_bw        the receiving stage's FIRST prefill all-reduce --
                              the one gated by the PP wave, so its duration is
                              stretched by the skew stall and its effective bw is
                              depressed. rs_ar_first_ns keeps its raw duration.
        rs_ar_rest_bw         the mean over that stage's remaining (steady-state)
                              all-reduces -- flat, buffer-independent.
        rs_ar_first_stage_bw  the FIRST prefill stage's all-reduce (stage 0), which
                              starts immediately and is never gated: the reference
                              the steady receiving-stage bw is compared against.

    rs_ar_first_bw is a SINGLE COLLECTIVE (n=1) while rs_ar_rest_bw averages the
    other rs_ar_n-1 of the run: the gated one is noisy by construction and is
    read as such -- it is the only sample of the quantity the sweep is about.

    Stages are the ASTRA `ss` field. With PP=1 there is no wave and no receiving
    stage, so the ONLY prefill stage reports instead: its first all-reduce is
    ungated (nothing to inherit a skew from), but its bw-vs-buffer curve is still
    the quantity the cross-model compare overlays -- expected flat, the control.
    All NaN/0 with no ASTRA run, or no prefill TP (TP=1)."""
    out = {"rs_ar_first_ns": NAN, "rs_ar_rest_mean_ns": NAN, "rs_ar_n": 0,
           "rs_ar_first_bw": NAN, "rs_ar_rest_bw": NAN, "rs_ar_first_stage_bw": NAN}
    if adf is None:
        return out
    tp = adf[(adf["op_class"] == "TP") & (adf["phase"] == "prefill")]
    if not len(tp) or "ss" not in tp.columns:
        return out
    keys = [c for c in ("pl", "ss", "L", "it", "op") if c in tp.columns]
    # One row per collective: slowest shard sets the wall-clock duration, comm_size
    # is the identical per-rank payload. bw is the CSV's effective rate for that
    # collective (bytes moved / how long it took).
    g = (tp.groupby(keys, dropna=False)
           .agg(start=("start_tick", "min"), dur=("duration", "max"),
                cs=("comm_size", "first")).reset_index())
    g = g[g["dur"] > 0].copy()
    if g.empty:
        return out
    g["bw"] = g["cs"] / g["dur"]                       # bytes/ns = GB/s
    g["ss"] = pd.to_numeric(g["ss"], errors="coerce")

    if ppr.available and ppr.stage is not None:
        stage = ppr.stage
    else:
        # PP=1: no wave, no receiving stage. The only prefill stage reports;
        # several stages with no measurable wave stays NaN (ambiguous).
        stages = g["ss"].dropna().unique()
        if len(stages) != 1:
            return out
        stage = stages[0]
    recv = g[g["ss"] == stage].sort_values("start")
    if recv.empty:
        return out
    first = recv.iloc[0]
    out["rs_ar_first_bw"] = float(first["bw"])
    out["rs_ar_first_ns"] = float(first["dur"])
    out["rs_ar_n"] = int(len(recv))
    rest = recv.iloc[1:]
    if len(rest):
        out["rs_ar_rest_bw"] = float(rest["bw"].mean())
        out["rs_ar_rest_mean_ns"] = float(rest["dur"].mean())
    first_stage = g[g["ss"] == 0]
    if len(first_stage):
        out["rs_ar_first_stage_bw"] = float(first_stage["bw"].mean())
    return out


def kv_stage_series(kv_arr: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """decode STAGE -> (arrival_times_ns, cumulative_bytes), sorted by arrival.

    The raw material for figure 02. Grouped by stage (the KV name's `ds`) and not
    by receiving rank on purpose: a stage is what stalls -- its ranks enter the
    same TP collective, so the pipeline waits for the LAST shard of the stage
    either way, and the within-stage spread is already reported as a scalar
    (kv_tp_skew_*). One curve per stage keeps the panel readable at any TP width.

    Rows whose name carries no `ds` are dropped, never guessed; empty dict when
    the trace has no KV recv at all."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if kv_arr is None or kv_arr.empty:
        return out
    for st, sub in kv_arr.dropna(subset=["stage"]).groupby("stage"):
        sub = sub.sort_values("arrival")
        out[int(st)] = (sub["arrival"].to_numpy(dtype=float),
                        np.cumsum(sub["size"].to_numpy(dtype=float)))
    return out


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyse(tag: str, p: paths.SweepPaths, placement: Placement,
            chosen_labels: list[str], want_series: bool = True) -> Row:
    """One run. analyse_sweep has already guaranteed the config inputs exist
    (Abort otherwise) and every output file is on disk (runs missing one are
    skipped before this is called), so nothing here re-checks file presence."""
    buf = BUFFER_AXIS.value(tag)
    ns3_dir = p.ns3_run(tag)
    topo = parse_topology(p.topology(tag))
    cfg = parse_ns3_config(p.config(tag))
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    need(cfg.buffer_mb is not None,
         f"{tag}: no BUFFER_SIZE in {p.config(tag)}.")
    need(abs(cfg.buffer_mb - buf) < 1e-6,
         f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but 'buf{buf:g}' "
         f"in the directory name. One of the two is lying.")

    row = Row(tag=tag, buffer_mb=float(buf), buffer_bytes=float(buf) * 1024 * 1024)

    # The one read of this run's ASTRA trace: TTFT, KV arrivals, PP skew,
    # all-reduce and decode-stall metrics all share this frame.
    adir = p.astra_run(tag)
    adf = astra.read_run(adir)
    need(adf is not None, f"{tag}: no readable stats_sys*.csv under {adir}.")
    row.ttft_ns = ttft_from(adf, tag)

    raw = ns3.read_fct(ns3_dir / "fct.txt")
    need(raw is not None and len(raw), f"{tag}: fct.txt has no parsable rows.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)
    for w in roles.check(f, placement):
        warn(f"{tag}: {w}")
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{tag}: no KV flow after classification.")

    # series (the per-sample queue timeline) feed only the per-tag plots; a
    # cross-model compare (want_series=False) needs just the scalars, so it skips
    # building them -- the big saving on qlen.txt reads across many workloads.
    qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=want_series)
    need(qlen is not None and qlen.port_max, f"{tag}: qlen.txt has no samples.")

    pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
    need(pfc is not None, f"{tag}: pfc.txt unreadable.")
    if pfc.qidx_state == "MISSING":
        warn(f"{tag}: pfc.txt has no qIndex; pause_pct_of_window is "
             f"approximate (see ns3.PFC_QIDX_PATCH). The pause frame COUNT "
             f"is unaffected.")

    run_end = int(f["arrival"].max())

    # -- every link this run's KV flows cross, indexed by the SWEEP-WIDE fixed
    #    label order (chosen_labels), not this run's own congestion ranking --
    #    see the module docstring: the link SET is topology-derived and
    #    invariant across buffer values, only the ranking can shift. ------ #
    links_here = {str(bn): bn for bn in
                  flowlib.candidate_links(topo, qlen.port_max, kv)}
    row.links = []
    per_switch: dict[int, list] = defaultdict(list)
    for label in chosen_labels:
        bn_i = links_here.get(label)
        if bn_i is None:
            warn(f"{tag}: link {label} (present in another run of this sweep) "
                 f"is not crossed by any KV flow here; recorded as NaN.")
            row.links.append(LinkStat(label=label))
            continue
        row.links.append(link_metrics(kv, bn_i, topo, pfc, qlen, row.buffer_bytes))
        # PAUSE on this link's ingress victims is what throttles the inflow
        # into bn_i.switch -- the signal figures 07/08 overlay on that switch's
        # queue. A switch is never itself the PFC "victim" of ITS OWN egress
        # queue; the upstream neighbour feeding it is (see PfcLog docstring).
        per_switch[bn_i.switch].extend(
            victim_pause_intervals(pfc, bn_i, topo, clamp_to=run_end))
    # two candidate links on one switch can share PAUSE victims; dedupe so the
    # figure-08 shading does not stack the same interval twice.
    row.pause_intervals = {sw: sorted(set(iv)) for sw, iv in per_switch.items()}
    need(row.links, f"{tag}: no candidate link to report.")

    bn = links_here.get(chosen_labels[0])
    need(bn is not None, f"{tag}: the top-ranked link {chosen_labels[0]} is "
                         f"not crossed by any KV flow in THIS run -- it cannot "
                         f"be treated as the bottleneck here.")
    row.bottleneck, row.bn_rate_gbps = str(bn), bn.rate / 1e9

    kv_bn = kv[flowlib.crosses(kv, bn)]
    need(len(kv_bn), f"{tag}: no KV flow crosses the top-ranked link {bn}.")
    row.bn_pause_intervals = victim_pause_intervals(pfc, bn, topo, clamp_to=run_end)

    # qmean/qpeak at the bottleneck: is the extra buffer holding SUSTAINED load
    # or only rare excursions? Both are already measured by link_metrics; the
    # ratio is the reading (figure 10).
    ls0 = row.links[0]
    if pd.notna(ls0.qpeak_bytes) and ls0.qpeak_bytes > 0:
        row.q_bloat_ratio = float(ls0.qmean_bytes / ls0.qpeak_bytes)

    # KV arrival, PP skew and all-reduce metrics all read the ASTRA stats CSV
    # (per-op end_tick = arrival, cleanly labelled by op/stage/iteration) rather
    # than reconstructing them from fct.txt flows -- identical nanosecond values,
    # none of the flow classification / send-recv dedup / wave-grouping heuristics.
    # (Queue occupancy, PFC and per-physical-link stats above stay on ns-3: ASTRA
    # has no equivalent.) `adf` was read once at the top of this function.
    kv_arr = astra.kv_arrivals(adf)

    for k, v in barrier(kv_arr, placement).items():
        setattr(row, k, v)
    row.kv_gate_over_ttft = (row.kv_gate_ns / row.ttft_ns
                             if pd.notna(row.ttft_ns) and row.ttft_ns > 0 else NAN)

    skew_scal, row.kv_rank_span, _layer_delta = kv_skew_stats(kv_arr)
    for k, v in skew_scal.items():
        setattr(row, k, v)
    if row.kv_tp_skew_n == 0:
        warn(f"{tag}: no (stage, layer) KV group has >=2 shard arrivals; the "
             f"TP-group skew figure will be empty for this run. (TP=1, or the "
             f"KV names carry no ds=/L= fields.)")
    if pd.notna(row.kv_tp_skew_mean_ns) and row.kv_tp_skew_mean_ns > 0:
        row.xrank_over_tp_skew = float(row.cross_rank_skew_ns
                                       / row.kv_tp_skew_mean_ns)

    row.dec_ar = decode_ar_stats(adf)
    if not row.dec_ar:
        warn(f"{tag}: no decode TP all-reduce in the ASTRA stats; the decode "
             f"all-reduce figure will be empty for this run.")

    stall_scal, row.dec_stall = decode_stall_stats(adf, kv_arr)
    for k, v in stall_scal.items():
        setattr(row, k, v)
    if pd.notna(row.tok2_ns) and pd.notna(row.ttft_ns):
        row.tok2_after_tok1_ns = row.tok2_ns - row.ttft_ns
    if pd.isna(row.tok2_ns):
        warn(f"{tag}: no DECFB send in the ASTRA stats; the decode KV-stall "
             f"figure will be empty for this run.")

    # The three instants of figure 04/05, all differences of quantities already
    # measured above -- named here so the CSV carries them and no reader has to
    # subtract two columns to get the quantity the figures are about.
    if pd.notna(row.dec_start_ns) and pd.notna(row.ttft_ns):
        row.dec_start_after_ttft_ns = row.dec_start_ns - row.ttft_ns
    if pd.notna(row.kv_gate_ns) and pd.notna(row.ttft_ns):
        row.kv_gate_after_ttft_ns = row.kv_gate_ns - row.ttft_ns
    if pd.notna(row.kv_gate_ns) and pd.notna(row.dec_start_ns):
        row.kv_tail_after_dec_start_ns = row.kv_gate_ns - row.dec_start_ns

    ppr = pp.measure(adf)
    if not ppr.available:
        warn(f"{tag}: no inter-stage PP-prefill activation found; the causal-chain "
             f"figure will be empty for this run. (PP=1, or placement has "
             f"one prefill stage.)")
    row.pp_skew_ns = ppr.skew_ns
    row.pp_skew_mean_ns = ppr.skew_mean_ns
    row.pp_first_ns = ppr.first_ns
    row.pp_last_ns = ppr.last_ns
    row.pp_stage = ppr.stage
    row.pp_n_waves = ppr.n_waves
    for k, v in rs_allreduce_stats(adf, placement, ppr).items():
        setattr(row, k, v)
    if ppr.available and pd.isna(row.rs_ar_first_bw):
        warn(f"{tag}: no prefill TP all-reduce in the ASTRA stats for the "
             f"receiving stage {ppr.stage}; all-reduce bandwidths unavailable.")

    row.kv_stage_series = kv_stage_series(kv_arr)
    row.kv_layer_skew = kv_layer_skew(kv_arr)
    if len(row.kv_layer_skew) and row.kv_layer_skew["signed_ns"].notna().any():
        row.kv_shard_bias_ns = float(row.kv_layer_skew["signed_ns"].median())

    for sw, (ts, ys) in qlen.switch_series.items():
        if len(ts) == 0:
            continue
        row.qseries[sw] = downsample_max(ts, ys, 2000)
        row.qswitch_peak[sw] = float(qlen.switch_total_max.get(sw, max(ys)))
        row.qswitch_mean[sw] = float(np.mean(ys))

    return row


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_plots(rows: list[Row], s: pd.DataFrame, outdir: Path,
               chosen_labels: list[str]) -> list[Path]:
    written: list[Path] = []
    x = s["buffer_mb"]
    runs = sorted(rows, key=lambda r: r.buffer_mb)

    # 01 CAUSAL CHAIN TO TTFT ------------------------------------------------ #
    # One stacked panel per link in the chain, sharing the buffer x-axis. A
    # single twin-axis plot cannot hold these together: TTFT is in ms and moves
    # little, PP skew and the all-reduce span are in µs (~1000x smaller) and
    # move a lot, so on shared axes the coupling is unreadable. Stacked, the
    # chain reads top-to-bottom and the correlation is vertical peak alignment;
    # each panel keeps its own autoscaled scale. Order = causal order:
    # buffer -> PFC/PAUSE -> PP skew -> receiving-stage all-reduce -> TTFT.
    if s["ttft_ns"].notna().any():
        panels = []      # (ylabel, [(series, colour, style, label), ...])
        if s["pp_skew_ns"].notna().any():
            panels.append(("PP arrival skew (µs)",
                           [(s["pp_skew_ns"] * US, CORAL, "s--", None)]))
        # All-reduce panels are EFFECTIVE BANDWIDTH (bytes/ns = GB/s) from the
        # ASTRA stats CSV, not duration. The gated one's bw is depressed because
        # its duration carries the skew stall; the steady one is compared against
        # the first prefill stage's all-reduce, which starts immediately (ungated).
        if s["rs_ar_first_bw"].notna().any():
            n = int(s["rs_ar_n"].max()) if s["rs_ar_n"].notna().any() else 0
            panels.append((f"Gated all-reduce\neff. bw (GB/s)\n(n=1 of {n})",
                           [(s["rs_ar_first_bw"], VIOLET, "^:", None)]))
        steady = []
        if s["rs_ar_rest_bw"].notna().any():
            steady.append((s["rs_ar_rest_bw"], GREEN, "D-.", "receiving stage (steady)"))
        if s["rs_ar_first_stage_bw"].notna().any():
            steady.append((s["rs_ar_first_stage_bw"], BLUE, "o--", "first prefill stage"))
        if steady:
            panels.append(("Steady all-reduce\neff. bw (GB/s)", steady))
        panels.append(("TTFT (ms)", [(s["ttft_ns"] * MS, BLUE, "o-", None)]))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8.5, 2.0 * n + 1.0))
        axes = np.atleast_1d(axes)
        for i, (ylabel, curves) in enumerate(panels):
            a = axes[i]
            for series, colour, style, label in curves:
                a.plot(x, series, style, color=colour, label=label)
            zoom_y(a, pd.concat([c[0] for c in curves]))
            mark_knees(a, s, label=(i == 0))
            a.set_ylabel(ylabel, fontsize=9)
            a.grid(True, alpha=0.3, which="both")
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=7, loc="best")
            if i != n - 1:
                a.set_xlabel("")
        fig.suptitle("Does PP skew propagate into TTFT?", y=0.99)
        save_fig(fig, outdir, "01_causal_chain_to_ttft.png", written)

    # 02 CUMULATIVE KV ARRIVAL PER DECODE STAGE ------------------------------ #
    # ONE panel per decode stage, EVERY buffer overlaid and colour-graded, so the
    # comparison the sweep is about is a comparison between curves in the same
    # axes instead of between neighbouring small panels. Y is % of what that
    # stage eventually receives, so stages of different size share the axis.
    # The triangle on the baseline is that run's decode START: KV drawn to the
    # RIGHT of a run's own triangle is KV the pipeline is already waiting on.
    stages = sorted({st for r in runs for st in r.kv_stage_series})
    if stages:
        bufs = [r.buffer_mb for r in runs]
        n = len(stages)
        # sharex: both stages live on the SAME absolute clock, and the decode
        # start is one run-wide instant. Independent x ranges would hide the
        # offset between the stages, which is the largest spread in the run.
        fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 6.8),
                                 sharey=True, sharex=True, squeeze=False)
        for k, st in enumerate(stages):
            a = axes[0][k]
            for r in runs:
                if st not in r.kv_stage_series:
                    continue
                t, cum = r.kv_stage_series[st]
                total = cum[-1] if len(cum) else 1.0
                c = buf_colour(r.buffer_mb, bufs)
                a.step(t * MS, 100 * cum / total, where="post", color=c,
                       lw=2.0, label=f"{r.buffer_mb:g} MiB")
                if pd.notna(r.dec_start_ns):
                    a.plot(r.dec_start_ns * MS, 0, marker="^", color=c, ms=11,
                           clip_on=False, zorder=5)
            a.set_title(f"decode stage {st}", fontsize=11)
            a.set_xlabel("Time (ms)")
            a.grid(True, alpha=0.3)
        axes[0][0].set_ylabel("KV arrived (%)")
        h, lab = axes[0][0].get_legend_handles_labels()
        h.append(Line2D([], [], color="k", marker="^", ls="none", ms=9))
        lab.append("decode start")
        axes[0][0].legend(h, lab, fontsize=10, loc="upper left")
        save_fig(fig, outdir, "02_kv_cumulative_arrival_per_stage.png", written)

    # 03 KV SKEW BETWEEN THE TP SHARDS OF ONE PP STAGE ----------------------- #
    # The comparison is: same PP stage, same layer, DIFFERENT TP shard --
    # |arrival(shard 1) - arrival(shard 0)| (kv_layer_skew). Nothing is compared
    # across stages (that offset is pipeline structure, fig 02) or across
    # layers. One box per (buffer, stage), over that stage's layers: median,
    # IQR, 1.5-IQR whiskers, fliers kept -- the tail layers are a finding, not
    # noise. The SIGNED twin of this population is not plotted but its median
    # lands in summary.csv (kv_shard_bias_ns): away from zero = one shard is
    # systematically late (path asymmetry), which no buffer can fix.
    pops = [(r, r.kv_layer_skew) for r in runs
            if r.kv_layer_skew is not None and len(r.kv_layer_skew)]
    if pops:
        st_all = sorted({int(v) for _, d in pops for v in d["stage"].unique()})
        st_colour = {st: (BLUE, CORAL, GREEN, VIOLET)[i % 4]
                     for i, st in enumerate(st_all)}
        allv = np.concatenate([d["skew_ns"].to_numpy() for _, d in pops]) * MS
        pos = allv[allv > 0]

        fig, ax = plt.subplots(figsize=(12, 6.5))
        # one box per (buffer, stage): the stage pair sits around the buffer
        # tick, offset in log2 because the axis is log2. Whiskers at 1.5 IQR,
        # fliers kept -- the tail layers ARE a finding, not noise.
        half = 0.16                                    # half-offset, in octaves
        for r, d in pops:
            for i, st in enumerate(st_all):
                y = d.loc[d["stage"] == st, "skew_ns"].to_numpy(dtype=float) * MS
                if not len(y):
                    continue
                p = r.buffer_mb * 2.0 ** ((i - (len(st_all) - 1) / 2) * 2 * half)
                bp = ax.boxplot([y], positions=[p], widths=p * 0.22,
                                patch_artist=True, manage_ticks=False,
                                medianprops=dict(color="k", lw=1.6),
                                flierprops=dict(marker="o", ms=3.5,
                                                mfc=st_colour[st], mec="none",
                                                alpha=0.6))
                bp["boxes"][0].set(facecolor=st_colour[st], alpha=0.55,
                                   edgecolor=st_colour[st])
        # symlog, not log: two shards can land in the same nanosecond (skew 0,
        # the whole T2 sweep), and a log axis would drop those boxes silently.
        ax.set_yscale("symlog", linthresh=max(float(pos.min()), 1e-4)
                      if len(pos) else 1e-4)
        logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel("|arrival(shard 1) − arrival(shard 0)| (ms)")
        n_lay = int(max(len(d[d["stage"] == st])
                        for _, d in pops for st in st_all))
        ax.set_title(f"KV skew between the two TP shards of a PP stage "
                     f"({n_lay} layers per box)", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.legend(handles=[Patch(facecolor=st_colour[st], alpha=0.55,
                                 label=f"PP stage {st}") for st in st_all],
                  fontsize=9)
        save_fig(fig, outdir, "03_kv_tp_shard_skew.png", written)

    # 04 FROM THE FIRST TOKEN TO THE SECOND ---------------------------------- #
    # The interval TTFT -> token 2 as a waterfall, one bar per buffer, time
    # measured from the first token. Three consecutive segments, which is the
    # whole story of this sweep in one row:
    #   BLUE   TTFT -> decode start: the first-token handoff still in flight,
    #          queued behind the KV bulk. The decode does not exist yet.
    #   CORAL  decode start -> KV gate: the pipeline is awake and consuming, and
    #          its KV is STILL ARRIVING. This is the stall, and it is the segment
    #          that grows with the buffer.
    #   GREEN  KV gate -> token 2: KV complete, the first pass finishing.
    # The PAUSE count is printed at the end of each row: it collapses to zero
    # from top to bottom while the coral segment grows, which is the whole reason
    # this figure is a waterfall and not a line plot of the total.
    wf = [r for r in runs if all(pd.notna(v) for v in
                                 (r.ttft_ns, r.dec_start_ns, r.kv_gate_ns, r.tok2_ns))]
    if wf:
        fig, ax = plt.subplots(figsize=(13, 0.8 * len(wf) + 2.4))
        end = 0.0
        for i, r in enumerate(wf):
            t0 = r.ttft_ns
            ds = (r.dec_start_ns - t0) * MS
            kg = (r.kv_gate_ns - t0) * MS
            t2 = (r.tok2_ns - t0) * MS
            end = max(end, t2)
            ax.barh(i, ds, left=0, height=0.62, color=BLUE)
            ax.barh(i, max(kg - ds, 0.0), left=ds, height=0.62, color=CORAL)
            ax.barh(i, max(t2 - kg, 0.0), left=max(kg, ds), height=0.62, color=GREEN)
            pf = r.links[0].pause_frames
            if pd.notna(pf):
                ax.text(t2 * 1.012, i, f"{pf:,.0f} PAUSE", va="center",
                        fontsize=9, color=MUTED)
        ax.set_yticks(range(len(wf)))
        ax.set_yticklabels([f"{r.buffer_mb:g} MiB" for r in wf])
        ax.invert_yaxis()
        ax.set_xlim(0, end * 1.20)
        ax.set_xlabel("ms after the first token")
        ax.grid(True, axis="x", alpha=0.3)
        # outside, above: the bars run the full width and any in-axes corner
        # would sit on top of a row's PAUSE label.
        ax.legend(handles=[Patch(color=BLUE, label="handoff in flight"),
                           Patch(color=CORAL, label="decode awake, KV still arriving"),
                           Patch(color=GREEN, label="first pass completing")],
                  fontsize=9, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), frameon=False)
        save_fig(fig, outdir, "04_first_token_to_second.png", written)

    # 05 HOW MUCH THE DECODE IS STALLED BY ITS KV ---------------------------- #
    # Two panels. (The old "token 2 after token 1" panel duplicated figure 04's
    # bar totals and is gone; the per-stage lateness panel is folded into the
    # right panel as its worst stage -- stage 0's -200 ms of fully-masked slack
    # was compressing the axis that matters. Per-stage values stay in the CSV.)
    #   LEFT   the effect: the first decode pass against the steady inter-token
    #          gap. The control is drawn, not just used as a denominator: its
    #          flatness is what says the buffer touches only the transient.
    #   RIGHT  the same excess (first pass - control) OVERLAID with its cause,
    #          the KV tail past the decode start, and with the worst stage's KV
    #          lateness. Three separately-measured quantities; that the three
    #          curves lie on top of each other IS the finding -- the first-pass
    #          stall is the KV still in flight, all of it, nothing else.
    if s["tok2_latency_ns"].notna().any():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

        axL.plot(x, s["tok2_latency_ns"] * MS, "o-", color=CORAL,
                 label="first decode pass")
        if s["itl_steady_ns"].notna().any():
            axL.plot(x, s["itl_steady_ns"] * MS, "v--", color=MUTED,
                     label="steady inter-token gap (control)")
        axL.set_title("First decode pass vs steady state", fontsize=11)
        axL.set_ylabel("Decode start → token 2 (ms)")

        for col, colour, style, label in (
                ("decode_kv_stall_ns", CORAL, "o-",
                 "first-pass stall (pass − control)"),
                ("kv_tail_after_dec_start_ns", VIOLET, "s--",
                 "KV tail past decode start"),
                ("dec_kv_lateness_ns", GREEN, "^:",
                 "KV lateness, worst stage")):
            if col in s.columns and s[col].notna().any():
                axR.plot(x, s[col] * MS, style, color=colour, label=label)
        axR.axhline(0.0, color="k", linestyle=":", alpha=0.5)
        axR.set_title("The stall and its cause coincide", fontsize=11)
        axR.set_ylabel("ms")

        for a in (axL, axR):
            mark_knees(a, s, label=(a is axL))
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            a.grid(True, alpha=0.3, which="both")
            a.legend(fontsize=8)
        save_fig(fig, outdir, "05_decode_kv_stall.png", written)

    # 06 THE DECODE ALL-REDUCE, ANALYSED LIKE THE PREFILL ONE ---------------- #
    # Figure 01 asks of the prefill: does the arrival skew stretch the FIRST
    # (gated) collective while the steady-state ones stay flat? This asks it of
    # the decode, where the gate is the stage's own KV instead of a PP wave.
    # Same framing as figure 01 -- EFFECTIVE BANDWIDTH, not the duration ratio:
    # the transfer is not slower, the early shard sits idle at the barrier, so
    # the honest reading is bytes-per-wall-time collapsing toward zero with the
    # ungated steady-state rate as the ceiling. One panel: the old entry-skew
    # companion was zero everywhere but one 12 µs point -- the same information
    # (it is µs-scale, i.e. the collective is NOT where the decode pays) is
    # visible here as first ~= steady, and the raw skews stay in the CSV.
    dec_stages = sorted({st for st_r in (r.dec_ar for r in runs) for st in st_r})
    if dec_stages:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        cmap = plt.get_cmap("tab10")
        any_bw = False
        for i, st in enumerate(dec_stages):
            c = cmap(i % 10)
            have = [r for r in runs if st in r.dec_ar]
            xs = [r.buffer_mb for r in have]
            fb = [r.dec_ar[st]["first_bw"] for r in have]
            rb = [r.dec_ar[st]["rest_bw_mean"] for r in have]
            if any(pd.notna(v) for v in fb):
                any_bw = True
                ax.plot(xs, fb, "o-", color=c, label=f"stage {st} — first (KV-gated)")
                ax.plot(xs, rb, "v--", color=c, alpha=0.5,
                        label=f"stage {st} — steady mean")
        if not any_bw:
            ax.text(0.5, 0.5, "no comm_size on the decode collectives",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color=MUTED)
        mark_knees(ax, s)
        logx_pow2(ax, s, "buffer_mb", "Per-switch buffer (MiB)")
        ax.set_ylabel("Effective bandwidth (GB/s)")
        ax.set_title("Decode first TP all-reduce vs its steady state", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
        save_fig(fig, outdir, "06_decode_allreduce.png", written)

    # 07 BOTTLENECK BUFFER OCCUPANCY(t), WITH PFC PAUSES --------------------- #
    # One column per buffer value: how full the bottleneck switch's buffer is, as
    # a % of BUFFER_SIZE, with PFC PAUSE spans shaded on top -- the pauses line up
    # with a full buffer, which is the mechanism the whole sweep rests on.
    bn_sw = int(rows[0].bottleneck.split("->")[0])
    occ_runs = [r for r in runs if bn_sw in r.qseries and r.buffer_bytes]
    if occ_runs:
        ncols = len(occ_runs)
        fig, axes = plt.subplots(1, ncols, squeeze=False, sharex=True, sharey=True,
                                 figsize=(max(3.0 * ncols, 6), 3.6))
        for j, r in enumerate(occ_runs):
            a = axes[0][j]
            ts, ys = r.qseries[bn_sw]
            occ = np.asarray(ys) / r.buffer_bytes * 100.0
            a.fill_between(np.asarray(ts) * MS, occ, color=VIOLET, alpha=0.5, step="mid")
            a.plot(np.asarray(ts) * MS, occ, color=VIOLET, lw=0.6, alpha=0.8)
            for s0, e0 in r.bn_pause_intervals:
                a.axvspan(s0 * MS, e0 * MS, color=CORAL, alpha=0.25, lw=0)
            a.set_ylim(0, 100)
            a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
            a.set_xlabel("Time (ms)", fontsize=8)
            a.grid(True, alpha=0.3)
        axes[0][0].set_ylabel(f"Buffer occupancy\nswitch {bn_sw} (% of buffer)",
                              fontsize=9)
        fig.suptitle(f"Bottleneck buffer occupancy ({rows[0].bottleneck})"
                     "  —  shaded = PFC PAUSE", y=1.02)
        save_fig(fig, outdir, "07_bottleneck_buffer_and_pauses.png", written)

    # 08 QUEUE OCCUPANCY(t) PER SWITCH, WITH PFC PAUSES ---------------------- #
    # The same picture for every switch that queues at all: which parts of the
    # fabric are actually involved. On an oversubscribed tree only the uplink
    # side fills, and a single bottleneck line cannot say that.
    switches = sorted({sw for r in rows for sw in r.qseries})
    if switches:
        cmap = plt.get_cmap("viridis")
        bufs = [r.buffer_mb for r in runs]
        cnorm = (matplotlib.colors.LogNorm(vmin=min(bufs), vmax=max(bufs))
                 if len(set(bufs)) > 1 else None)
        nrows, ncols = len(switches), len(runs)
        fig, axes = plt.subplots(
            nrows, ncols, squeeze=False, sharex=True, sharey="row",
            figsize=(max(2.1 * ncols + 1.6, 6), max(1.7 * nrows + 1.0, 4)))
        for i, sw in enumerate(switches):
            for j, r in enumerate(runs):
                a = axes[i][j]
                if sw in r.qseries:
                    ts, ys = r.qseries[sw]
                    col = cmap(cnorm(r.buffer_mb)) if cnorm else BLUE
                    t, y = np.asarray(ts) * MS, np.asarray(ys) / 1e3
                    a.fill_between(t, y, color=col, alpha=0.85, lw=0)
                    a.plot(t, y, color="#222222", lw=0.5, alpha=0.6)
                # a full-height translucent span gets visually swallowed by an
                # opaque, tall queue fill exactly where it matters most (pauses
                # correlate with high occupancy) -- a top ribbon in axes-
                # fraction y stays visible regardless of the fill underneath.
                for s0, e0 in r.pause_intervals.get(sw, []):
                    a.axvspan(s0 * MS, e0 * MS, ymin=0.88, ymax=1.0,
                              transform=a.get_xaxis_transform(),
                              color=CORAL, alpha=0.9, lw=0)
                a.grid(True, alpha=0.2)
                if i == 0:
                    a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9)
                if j == 0:
                    mark = "\n(bottleneck)" if sw == bn_sw else ""
                    a.set_ylabel(f"switch {sw}{mark}\n(kB)", fontsize=8)
                if i == nrows - 1:
                    a.set_xlabel("Time (ms)", fontsize=8)
                    a.locator_params(axis="x", nbins=4)
        fig.suptitle("Queue occupancy over time — rows = switch, columns = buffer",
                     y=1.01)
        save_fig(fig, outdir, "08_queue_occupancy_per_switch.png", written)

    # 09 CONGESTION ACROSS EVERY KV-CROSSED LINK ----------------------------- #
    # Three panels, one line per link: delivered efficiency, PAUSE count and peak
    # queue. Reporting only the measured bottleneck (link0) hides two things that
    # matter on an oversubscribed tree -- that a SECOND uplink is carrying a
    # comparable amount of backpressure, and that the downlinks never pause at
    # all, which localises the congestion to the ingress side of the core switch.
    if chosen_labels:
        fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5))
        cmap = plt.get_cmap("tab10")
        for i, label in enumerate(chosen_labels):
            lw = 2.6 if i == 0 else 1.2
            lbl = label + ("  (measured bottleneck)" if i == 0 else "")
            c = cmap(i % 10)
            for a, col, scale in ((axA, f"link{i}_eff_pct", 1.0),
                                  (axB, f"link{i}_pause_frames", 1.0),
                                  (axC, f"link{i}_qpeak_bytes", 1 / 2**20)):
                if col in s.columns and s[col].notna().any():
                    a.plot(x, s[col] * scale, marker="o", lw=lw, color=c,
                           label=lbl if a is axA else label)
        axB.set_yscale("symlog", linthresh=1)
        for a, ttl, yl in (
                (axA, "Delivered KV bandwidth per link", "KV bandwidth (% of nominal)"),
                (axB, "PFC PAUSE frames per link", "PAUSE frames (symlog)"),
                (axC, "Peak queue occupancy per link", "Peak occupancy (MB)")):
            mark_knees(a, s, label=(a is axB))
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            a.set_title(ttl)
            a.set_ylabel(yl)
            a.grid(True, alpha=0.3, which="both")
            a.legend(fontsize=7)
        fig.suptitle("Congestion across every KV-crossed link, not just the "
                     "bottleneck", y=1.02)
        save_fig(fig, outdir, "09_per_link_congestion.png", written)

    # 10 IS THE ADDED BUFFER ABSORBING LOAD, OR STANDING IDLE? --------------- #
    # LEFT, absolute MB: peak against mean occupancy at the bottleneck. Peak
    # always grows -- it is bounded by the knob being swept, so on its own it
    # says nothing. RIGHT, their ratio: a flat ratio means the extra buffer is
    # carrying sustained load; a collapsing one means it only absorbs rare
    # excursions and is idle the rest of the time.
    qp, qm = "link0_qpeak_bytes", "link0_qmean_bytes"
    if qp in s.columns and s[qp].notna().any():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        axL.plot(x, s[qp] / 2**20, "o-", color=CORAL, label="peak occupancy")
        if qm in s.columns and s[qm].notna().any():
            axL.plot(x, s[qm] / 2**20, "v--", color=BLUE, label="mean occupancy")
        axL.plot(x, x, ":", color=MUTED, lw=1.0, label="the buffer itself")
        axL.set_yscale("log")
        mark_knees(axL, s)
        logx_pow2(axL, s, "buffer_mb", "Per-switch buffer (MiB)")
        axL.set_ylabel("Occupancy at the bottleneck (MiB, log)")
        axL.set_title("Peak and mean occupancy against the buffer")
        axL.grid(True, alpha=0.3, which="both")
        axL.legend(fontsize=8)

        if "q_bloat_ratio" in s.columns and s["q_bloat_ratio"].notna().any():
            axR.plot(x, s["q_bloat_ratio"], "o-", color=VIOLET)
            axR.set_ylim(0, max(1.0, float(s["q_bloat_ratio"].max()) * 1.15))
            mark_knees(axR, s)
        logx_pow2(axR, s, "buffer_mb", "Per-switch buffer (MiB)")
        axR.set_ylabel("mean ÷ peak occupancy")
        axR.set_title("Sustained load, or rare excursions?")
        axR.grid(True, alpha=0.3, which="both")
        if axR.get_legend_handles_labels()[0]:
            axR.legend(fontsize=8)
        fig.suptitle("Buffer bloat at the bottleneck", y=1.02)
        save_fig(fig, outdir, "10_buffer_bloat.png", written)

    return written


# --------------------------------------------------------------------------- #
# The printed table: the chain, then the two things the buffer trades off.
REPORT = ["buffer_mb", "ttft_ns", "pp_skew_ns", "rs_ar_first_bw",
          "kv_gate_after_ttft_ns", "cross_rank_skew_ns", "kv_tp_skew_mean_ns",
          "xrank_over_tp_skew", "kv_tail_after_dec_start_ns",
          "decode_kv_stall_ns", "itl_steady_ns",
          "link0_eff_pct", "link0_pause_frames", "q_bloat_ratio"]


def analyse_sweep(p: paths.SweepPaths, placement: Placement,
                  top_links: int = 6, bn_force: str | None = None,
                  verbose: bool = True,
                  want_series: bool = True,
                  tags: list[str] | None = None) -> tuple[list[Row], pd.DataFrame, list[str]]:
    """Score one workload's whole buffer sweep, exactly as buffer_sweep.main
    does -- factored out so buffer_compare gets identical numbers (one
    definition of the metrics, two tools). Returns (rows sorted by buffer, the flat
    summary DataFrame, chosen_labels). Raises Abort on any condition main would.
    Does no I/O beyond reading the sweep; writing figures/CSV stays in the
    caller. `verbose` gates the progress prints so a multi-workload caller stays
    quiet.

    `tags` restricts the analysis to a subset of the sweep's run dirs (default:
    all of p.tags('ns3')). A 2D sweep (e.g. oversubscription x buffer) shares one
    ns3 dir but moves two knobs, so oversub2d_sweep calls this once per
    oversubscription level with that level's buffer tags -- keeping the
    single-variant (one buffer axis) contract intact within each call."""
    need(not p.missing_roots(),
         "derived root(s) do not exist:\n    "
         + "\n    ".join(p.missing_roots())
         + f"\n  --sweep {p.sweep!r} is probably wrong.")
    tags = tags if tags is not None else p.tags("ns3")
    need(tags, f"no run sub-directory under {p.ns3_root}")

    # Config INPUTS and the swept-axis token are what the analysis reasons FROM:
    # any of them missing on any run is an Abort, before touching a single file.
    miss_cfg = [f"{t}: missing {f}" for t in tags
                for f in (p.topology(t), p.config(t)) if not f.is_file()]
    need(not miss_cfg, "config input(s) missing -- fix the configs (or prune "
                       "the stale ns-3 output):\n    " + "\n    ".join(miss_cfg))
    for t in tags:
        need(BUFFER_AXIS.value(t) is not None,
             f"{t}: no 'buf<num>' token in the directory name; the swept axis "
             f"is unreadable.")

    # OUTPUTS are what the simulations write: a run still missing one has just
    # not finished (or died) -- skip it with a name, and analyse the rest.
    usable = []
    for t in tags:
        nd = p.ns3_run(t)
        missing = [n for n, ok in (
            ("fct.txt", (nd / "fct.txt").is_file()),
            ("pfc.txt", (nd / "pfc.txt").is_file()),
            ("qlen.txt", (nd / "qlen.txt").is_file()),
            ("stats_sys*.csv", any(p.astra_run(t).glob("stats_sys*.csv")))) if not ok]
        if missing:
            warn(f"{t}: output(s) missing ({', '.join(missing)}) -- run skipped.")
        else:
            usable.append(t)
    need(usable, "no run has all its outputs on disk yet.")
    tags = usable

    if verbose:
        print(p.describe())
        print(f"  placement\n{placement.describe()}\n")
    if msg := roles.cross_check(placement, p.astra_run(tags[0])):
        warn(msg)

    variants = {BUFFER_AXIS.variant(t) for t in tags}
    need(len(variants) == 1,
         f"this sweep moves more than one knob: variants {sorted(variants)}. "
         f"Split into one sweep per variant.")

    # -- fix the link SET and its display order once, from the smallest
    #    buffer (most congested, least likely to be a near-tie). Every run
    #    then reports THESE labels, in THIS order -- see the module docstring
    #    on why the set itself cannot vary within one sweep. ------------- #
    ref_tag = min(tags, key=lambda t: BUFFER_AXIS.value(t))
    ref_topo = parse_topology(p.topology(ref_tag))
    ref_cfg = parse_ns3_config(p.config(ref_tag))
    ref_raw = ns3.read_fct(p.ns3_run(ref_tag) / "fct.txt")
    need(ref_raw is not None and len(ref_raw),
         f"{ref_tag}: fct.txt has no parsable rows; cannot fix the link set.")
    ref_f = flowlib.annotate(ref_raw, ref_topo, placement, ref_cfg.payload)
    ref_kv = ref_f[ref_f["flow_class"] == "kv"]
    need(len(ref_kv), f"{ref_tag}: no KV flow after classification.")
    ref_qlen = ns3.read_qlen(p.ns3_run(ref_tag) / "qlen.txt", series=False)
    need(ref_qlen is not None and ref_qlen.port_max,
         f"{ref_tag}: qlen.txt has no samples.")
    canonical = flowlib.candidate_links(ref_topo, ref_qlen.port_max, ref_kv)
    need(canonical, f"{ref_tag}: no link is crossed by any KV flow -- "
                    f"classification or topology is wrong.")

    if bn_force:
        sw, peer = (int(v) for v in bn_force.split("->"))
        idx = next((i for i, l in enumerate(canonical)
                    if l.switch == sw and l.peer == peer), None)
        need(idx is not None,
             f"--bottleneck {bn_force}: not among the links this sweep's KV "
             f"flows cross ({[str(l) for l in canonical]}).")
        canonical.insert(0, canonical.pop(idx))

    chosen_labels = [str(l) for l in canonical[:top_links]]
    if verbose:
        print(f"link set ({len(canonical)} total, top {len(chosen_labels)} kept):")
        for lab in chosen_labels:
            print(f"  - {lab}")
        print(f"\nAnalysing {len(tags)} runs:")

    rows = [analyse(t, p, placement, chosen_labels, want_series) for t in tags]
    if verbose:
        for r in sorted(rows, key=lambda r: r.buffer_mb):
            skew = r.pp_skew_ns * US if pd.notna(r.pp_skew_ns) else float("nan")
            print(f"  + buf{r.buffer_mb:<5g} bn={r.bottleneck:<8} "
                  f"ttft={r.ttft_ns * MS:6.2f}ms gate={r.kv_gate_ns * MS:6.2f}ms  "
                  f"pp_skew={skew:7.2f}us  eff={r.links[0].eff_pct:5.1f}%")

    bns = {r.bottleneck for r in rows}
    need(len(bns) == 1,
         f"the top-ranked link is not the same on every run: {sorted(bns)}. "
         f"Pass --bottleneck to fix one explicitly.")

    rows = sorted(rows, key=lambda r: r.buffer_mb)
    s = (pd.DataFrame([r.flat() for r in rows])
         .sort_values("buffer_mb").reset_index(drop=True))
    s = decode_worst_stage(s)
    # Sweep-wide readings, written back as constant columns so summary.csv and
    # every cross-model consumer carry them without re-deriving them. They need
    # decode_kv_stall_ns, so they come last.
    for k, v in knee_scalars(s, PAUSE_KNEE_COL, SATURATION_METRICS).items():
        s[k] = v
    return rows, s, chosen_labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_arguments(ap, KIND)
    for act in ap._actions:
        if act.dest == "out":
            act.help = "output dir (default: results/sweep_analysis/" \
                       f"{KIND}/<workload>/<sweep>)"
    roles.add_argument(ap)
    ap.add_argument("--bottleneck", default=None,
                    help="'sw->peer', e.g. '8->12'. Must be among the links "
                         "this sweep's KV flows cross. Default: the deepest "
                         "queue among them, measured on the smallest-buffer run.")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links figure 09 and summary.csv "
                         "carry (default: 6)")
    a = ap.parse_args(argv)

    try:
        p = paths.SweepPaths(sweep=a.sweep, workload=a.workload, root=Path(a.root))
        outdir = (Path(a.out) if a.out else
                  p.root / "results" / "sweep_analysis" / KIND / p.workload / p.sweep)
        placement = Placement.parse(a.placement)
        print(f"  out      {outdir}")
        rows, s, chosen_labels = analyse_sweep(
            p, placement, top_links=a.top_links, bn_force=a.bottleneck, verbose=True)

        fresh_dir(outdir)
        s.to_csv(outdir / "summary.csv", index=False)
        plots = make_plots(rows, s, outdir, chosen_labels)

        pd.set_option("display.width", 260)
        print("\n================ BUFFER SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))
        print("\n---- knees (MiB) ----")
        for col, (_c, name) in KNEE_STYLE.items():
            v = s[col].dropna()
            print(f"  {name:<12} {f'{v.iloc[0]:g}' if len(v) else 'never reached'}")
        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", *[q.name for q in plots]]:
            print(f"  {fpath}")
        return drain_warnings(" — the numbers above are conditional on them")
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
