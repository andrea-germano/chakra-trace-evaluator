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

2.  THE KV TAIL PAST THE DECODE START (kv_tail_after_dec_start_ns, figs 02/05),
    AND WHAT IT IS NOT. kv_gate - dec_start: how much KV is still on the wire
    when the decode pipeline wakes. Same quantity astra_analyzer reports as
    kv_tail_vs_decode_start_ns, so the time-domain view and the sweep agree.

    It was introduced here as the CAUSE of decode_kv_stall_ns. It is not: it is
    an UPPER BOUND on it, and on this model a loose one. The decode does not
    need all of its KV at once -- it walks the layers and waits only where it
    outruns the transfer -- so the tail counts KV the pipeline was never going
    to ask for yet. On T1/16 requests the tail reads 0.25-4.1 ms while every
    decode rank runs its first pass strictly back to back: the true stall is
    zero at every buffer. The measured quantity is dec_kv_block_ns
    (utils.measures.first_pass_stall): the idle INSIDE the pass that a KV
    arrival ends, unioned over the decode ranks. Where a stall does exist it
    reproduces the makespan arithmetic to 0.2% on this sweep (T1/64 requests at
    8 MiB: 47.079 ms measured against a 47.013 ms excess over the steady
    inter-token gap, while the tail claims 53.039). Both are kept and figure 05 draws them
    together, because the DISTANCE between them is itself the reading: it is
    the KV that arrived late and cost nothing.

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

Figure 04's left panel says what that serialisation is a piece OF, which nothing
in the previous figure set showed: the transfer does not begin at TTFT. The first
KV send leaves ~1 ms into the run (streaming: a layer's KV goes as soon as that
layer's prefill is done) and the last lands ~206 ms later, so on T1/16 requests
the prefill hides ~63% of a ~205 ms transfer and only the remaining ~37% is
exposed -- of which 0.2-4 ms finds the decode already awake, and (point 2 above)
none of it actually stalls that pass. Three nested quantities, one bar:
kv_transfer_ns, kv_gate_after_ttft_ns (kv_exposed_frac of it) and
kv_tail_after_dec_start_ns, with the measured stall in the row label as the
fourth number the geometry cannot carry. Reading the exposed tail
without the masked bulk beside it invites the conclusion that the fabric moves
KV for 77 ms of user-visible time, when what it moves in that window is the
last third of a transfer that has been running since the prefill started.

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
                                    start drawn as a vertical rule in that run's
                                    own colour: the curve's height AT the rule is
                                    how much of the stage's KV had landed when the
                                    pipeline woke, and everything to the right of
                                    it is KV the pipeline is already waiting on.
    03  KV TP-SHARD SKEW            per (PP stage, layer): |arrival(shard 1) -
                                    arrival(shard 0)| between the two TP shards
                                    of that stage. Boxplots, one box per
                                    (buffer, stage) over the stage's layers.
    04  THE KV TRANSFER             LEFT the whole transfer per buffer, first KV
                                    send -> last KV arrival, split into the part
                                    running behind the prefill | the part exposed
                                    past TTFT with the decode not yet awake | the
                                    part that finds the decode already awake (the
                                    stall), with the three totals spelled out.
                                    RIGHT the same exposed part zoomed, TTFT ->
                                    token 2, PAUSE count beside each row.
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
                            first_pass_stall, knee_scalars, kv_layer_skew,
                            kv_skew_stats, link_metrics, ttft_from,
                            victim_pause_intervals)
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
    kv_start_ns: float = NAN              # first KV SEND issue: the transfer begins
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

    # -- 04 the KV transfer, and the part of it the user pays for -------------- #
    dec_start_ns: float = NAN             # first decode COMP start (stage 0 wakes
                                          # on the FIRSTTOK ARRIVAL, not its send)
    dec_start_after_ttft_ns: float = NAN  # dec_start - ttft: the exposed handoff
    kv_gate_after_ttft_ns: float = NAN    # kv_gate - ttft: until the LAST KV lands
    # The transfer's own window and its three-way split. The three parts PARTITION
    # kv_transfer_ns (masked + exposed = total; the stall is the tail of the
    # exposed part, not a fourth part), so a reader can add them and a figure can
    # stack them without double counting. It is the run's ENVELOPE -- first send
    # of any request to last arrival of any request -- not one request's own
    # transfer time, which a 16-request run does not have a single value of:
    kv_transfer_ns: float = NAN           # kv_gate - kv_start: the WHOLE transfer
    kv_masked_ns: float = NAN             # kv_start -> TTFT: moved behind the
                                          # prefill, invisible to the user
    kv_exposed_frac: float = NAN          # kv_gate_after_ttft / kv_transfer: how
                                          # much of the transfer the prefill fails
                                          # to hide (1.0 = nothing is hidden)

    # -- 05 decode stalled by KV reception ------------------------------------ #
    kv_tail_after_dec_start_ns: float = NAN  # kv_gate - dec_start: KV still in
                                          # flight when the pipeline wakes. An
                                          # UPPER BOUND on the stall, not the
                                          # stall: the decode consumes layer by
                                          # layer and waits only where it outruns
                                          # the transfer (see dec_kv_block_ns).
    # The stall as measured inside the pass (utils.measures.first_pass_stall):
    # idle between the decode ranks' own intervals, attributed to the KV arrival
    # that ends it, unioned over the ranks. On T1/16 requests this is 0 at every
    # buffer while the bound above reads 0.25-4.1 ms -- the pipeline never
    # actually waits there.
    dec_kv_block_ns: float = NAN          # union of the KV-blocked idle
    dec_other_idle_ns: float = NAN        # idle no KV arrival explains
    dec_kv_block_max_ns: float = NAN      # worst SINGLE rank's KV-blocked idle
    dec_crit_rank: float = NAN            # the rank that emitted token 2
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
    dec_idle: dict = field(default_factory=dict)         # first-pass busy/idle
                                                         # spans, per rank and
                                                         # unioned (figure 04)
    bn_pause_intervals: list = field(default_factory=list)
    qseries: dict = field(default_factory=dict)          # sw -> (ts_ns, bytes)
    qswitch_peak: dict = field(default_factory=dict)
    qswitch_mean: dict = field(default_factory=dict)
    pause_intervals: dict = field(default_factory=dict)  # sw -> [(start, end)]

    def flat(self) -> dict:
        d = asdict(self)
        for k in ("links", "kv_stage_series", "kv_layer_skew", "kv_rank_span",
                  "dec_ar", "dec_stall", "dec_idle", "bn_pause_intervals",
                  "qseries", "qswitch_peak", "qswitch_mean", "pause_intervals"):
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
            # The handover framing (incast_sweep's headline) on the LINK route:
            # the transfer cannot beat bytes/rate, so its duration measures
            # nothing -- what congestion produces is idle = window - floor, the
            # time the link spent NOT sending. eff_pct is the same reading as a
            # percentage; the ms are what a reader can add to a latency budget.
            #
            # Deliberately the link route and not measures.kv_handover_idle:
            # that one is per RECEIVING RANK and its stated precondition is that
            # the receiver's own link is the narrowest on the path. On a tree
            # the pinch is a SHARED uplink instead (T1/T7: 200 Gb/s uplink, 800
            # Gb/s access link busy 11.8%, two decode ranks behind it), so a
            # per-rank floor taken at the uplink rate is ~2x understated and the
            # whole error lands in idle. Note that kv_handover_idle's caller-side
            # guard -- "all sender->receiver pairs share one narrowest rate" --
            # PASSES here, because every path pinches on an identical 200 Gb/s
            # uplink: the guard cannot tell a shared uplink from a private
            # access link. The `starved`/`incast` split needs the per-rank route
            # and is therefore NOT available on a tree; only the total is.
            d[f"link{i}_rate_gbps"] = ls.rate_gbps
            d[f"link{i}_kv_bytes"] = ls.kv_bytes
            d[f"link{i}_floor_ns"] = ls.floor_ns
            d[f"link{i}_idle_ns"] = (ls.window_ns - ls.floor_ns
                                     if pd.notna(ls.window_ns)
                                     and pd.notna(ls.floor_ns) else NAN)
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
    # The other end of the transfer window. barrier() only sees arrivals (recv
    # rows are pre-posted at tick 0), so the instant the first KV byte was ISSUED
    # comes from the send side -- figure 04's left edge.
    kv_start = astra.kv_send_start(adf)
    if kv_start is None:
        warn(f"{tag}: no KV send row in the ASTRA stats; the transfer's start "
             f"instant is unavailable and figure 04's masked segment is empty.")
    else:
        row.kv_start_ns = float(kv_start)

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
    # The same stall measured directly, inside the pass, instead of bounded by
    # the KV envelope -- the two disagree whenever the decode consumes its KV
    # faster than the transfer delivers the tail it does not need yet.
    idle_scal, row.dec_idle = first_pass_stall(adf, kv_arr, row.tok2_ns)
    for k, v in idle_scal.items():
        setattr(row, k, v)
    if not row.dec_idle:
        warn(f"{tag}: no it=0 decode COMP/TP op in the ASTRA stats; the "
             f"measured first-pass stall is unavailable (figure 04's right "
             f"panel falls back to the KV envelope).")
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
    # The transfer as a whole, and how much of it the prefill hides. Clamped at
    # zero, never negative: a run whose KV lands before TTFT is fully masked
    # (exposed = 0), not one with a negative exposure.
    if pd.notna(row.kv_start_ns) and pd.notna(row.kv_gate_ns):
        row.kv_transfer_ns = row.kv_gate_ns - row.kv_start_ns
        if pd.notna(row.ttft_ns):
            row.kv_masked_ns = max(0.0, min(row.ttft_ns, row.kv_gate_ns)
                                   - row.kv_start_ns)
        if row.kv_transfer_ns > 0 and pd.notna(row.kv_gate_after_ttft_ns):
            row.kv_exposed_frac = max(0.0, row.kv_gate_after_ttft_ns) / row.kv_transfer_ns

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
        fig.suptitle("PP arrival skew, the gated all-reduce and TTFT "
                     "vs buffer", y=0.99)
        save_fig(fig, outdir, "01_causal_chain_to_ttft.png", written)

    # 02 CUMULATIVE KV ARRIVAL PER DECODE STAGE ------------------------------ #
    # ONE panel per decode stage, EVERY buffer overlaid and colour-graded, so the
    # comparison the sweep is about is a comparison between curves in the same
    # axes instead of between neighbouring small panels. Y is % of what that
    # stage eventually receives, so stages of different size share the axis.
    # The dashed rule is that run's decode START, in that run's own colour: KV
    # drawn to the RIGHT of a run's own rule is KV the pipeline is already
    # waiting on. A full-height line and not the old baseline triangle -- the
    # marker sat below the curves and the reader had to project it upwards to
    # read off HOW MUCH of the stage's KV was still outstanding at that instant,
    # which is the whole quantity the panel exists to show.
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
                    a.axvline(r.dec_start_ns * MS, color=c, ls="--", lw=1.5,
                              alpha=0.85, zorder=5)
            a.set_title(f"decode stage {st}", fontsize=11)
            a.set_xlabel("Time (ms)")
            a.grid(True, alpha=0.3)
        axes[0][0].set_ylabel("KV arrived (%)")
        h, lab = axes[0][0].get_legend_handles_labels()
        h.append(Line2D([], [], color="k", ls="--", lw=1.5))
        lab.append("decode start (per run)")
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

    # 04 THE KV TRANSFER, AND THE PART OF IT THE USER PAYS FOR --------------- #
    # Both panels share one clock -- ms measured from that run's OWN first token,
    # so x=0 is TTFT on every row -- and one colour code, because the segments
    # are literally the same instants read twice:
    #   MUTED  KV start -> TTFT: the transfer running behind the prefill. It is
    #          the bulk of the transfer and it costs the user nothing; without it
    #          on the page the exposed tail looks like the whole handover.
    #   BLUE   TTFT -> decode start: the first-token handoff still in flight,
    #          queued behind the KV bulk. Exposed, but the decode does not exist
    #          yet, so the KV arriving here is still free.
    #   CORAL  decode start -> KV gate: the pipeline is awake and its KV is STILL
    #          ARRIVING. This is the WINDOW IN WHICH A STALL IS POSSIBLE, and it
    #          is emphatically not the stall: the decode consumes layer by layer
    #          and waits only where it outruns the transfer. The measured stall
    #          (dec_kv_block_ns) is in the row label and in the right panel.
    #
    # LEFT   the whole transfer, kv_start -> kv_gate, split MUTED | BLUE | CORAL,
    #        with the totals in each row's label -- transfer, exposed share, and
    #        the MEASURED stall, which on T1 is zero at every buffer while the
    #        coral window reads 0.25-4.1 ms.
    # RIGHT  what the first pass actually did, on the rank that emitted token 2:
    #        computing (green), idle with a KV arrival ending the wait (coral),
    #        idle for another reason (violet). PAUSE count beside each row.
    wf = [r for r in runs if all(pd.notna(v) for v in
                                 (r.ttft_ns, r.dec_start_ns, r.kv_gate_ns, r.tok2_ns))]
    if wf:
        fig, (axT, axW) = plt.subplots(
            1, 2, figsize=(16.5, 0.8 * len(wf) + 2.8), sharey=True,
            gridspec_kw=dict(width_ratios=(1.25, 1.0)))

        # -- LEFT: the transfer as a whole ---------------------------------- #
        # Each part is clipped to the transfer's own window before it is drawn,
        # so a fully-masked run (KV complete before TTFT) or one whose decode
        # wakes after the gate (no stall) loses that segment instead of drawing
        # a negative one.
        def seg(lo: float, hi: float, a0: float, gate: float) -> tuple[float, float]:
            lo, hi = min(max(lo, a0), gate), min(max(hi, a0), gate)
            return lo, hi - lo

        spans = [(i, (r.kv_start_ns - r.ttft_ns) * MS,   # a0: before the token
                  (r.dec_start_ns - r.ttft_ns) * MS,     # ds
                  (r.kv_gate_ns - r.ttft_ns) * MS)       # gate
                 for i, r in enumerate(wf) if pd.notna(r.kv_start_ns)]
        left_edge = min([a0 for _, a0, _, _ in spans] + [0.0])
        right_edge = max([g for _, _, _, g in spans] + [0.0])
        span = max(right_edge - left_edge, 1e-9)
        # The three quantities as text, in the ROW LABEL rather than beside the
        # bar: they are nested (stalling < exposed < total) and the stalling one
        # is a 0.1-2% slice, so the geometry alone reads as zero -- but a margin
        # wide enough for ~50 characters is a margin whose width depends on the
        # figure's dpi and font, and every fixed guess collides with the bars on
        # some sweep. The tick column cannot collide with anything.
        detail: dict[int, str] = {}
        for i, a0, ds, gate in spans:
            for lo, hi, colour in ((a0, 0.0, MUTED), (0.0, ds, BLUE),
                                   (ds, gate, CORAL)):
                x0, w = seg(lo, hi, a0, gate)
                if w > 0:
                    axT.barh(i, w, left=x0, height=0.62, color=colour)
            # the third number is the MEASURED stall, not the coral segment:
            # the segment is the window in which a stall COULD happen, and on
            # most of this sweep none does.
            blocked = wf[i].dec_kv_block_ns * MS
            detail[i] = (f"{gate - a0:,.0f} ms transfer\n"
                         f"{100 * max(gate, 0.0) / (gate - a0):.0f}% exposed · "
                         + ("stall not measured" if pd.isna(blocked) else
                            ("no stall" if blocked <= 0 else
                             (f"{blocked * 1e3:,.0f} µs stall" if blocked < 1.0
                              else f"{blocked:,.1f} ms stall"))))
        axT.axvline(0.0, color="k", lw=1.4, zorder=4)
        axT.annotate("first token", xy=(0.0, 1.0),
                     xycoords=("data", "axes fraction"),
                     xytext=(3, -3), textcoords="offset points",
                     ha="left", va="top", fontsize=8.5, color="k")
        axT.set_xlim(left_edge - 0.03 * span, right_edge + 0.06 * span)
        axT.set_xlabel("ms relative to the first token")
        axT.set_title("The whole KV transfer: hidden behind the prefill vs "
                      "exposed", fontsize=10, loc="left", pad=30)
        axT.legend(handles=[Patch(color=MUTED, label="behind the prefill"),
                            Patch(color=BLUE, label="exposed, decode not awake yet"),
                            Patch(color=CORAL, label="exposed, decode awake "
                                                     "(where a stall is possible)")],
                   fontsize=8.5, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, 1.005), frameon=False)
        if not spans:
            axT.text(0.5, 0.5, "no KV send instant in this sweep's traces",
                     transform=axT.transAxes, ha="center", va="center",
                     fontsize=10, color=MUTED)

        # -- RIGHT: where the first pass actually spends the time ----------- #
        # NOT the [decode start, KV gate] split the old figure drew: that split
        # calls every millisecond between the two a stall, and the trace says
        # otherwise -- the decode consumes its KV layer by layer and on most of
        # this sweep runs the whole first pass back to back while the tail of
        # the transfer, which it does not need yet, is still landing.
        #
        # The bar is the PIPELINE: green wherever some decode rank is making
        # progress, coral over the intervals in which a rank sat idle and a KV
        # arrival is what let it resume (first_pass_stall's union -- the TP
        # shards re-synchronise every layer, so any blocked shard holds the
        # group). Green + coral + violet = the pass, and the coral total is the
        # same number as the pass's excess over its steady inter-token gap.
        end = 0.0
        for i, r in enumerate(wf):
            t0 = r.ttft_ns
            ds = (r.dec_start_ns - t0) * MS
            t2 = (r.tok2_ns - t0) * MS
            end = max(end, t2)
            axW.barh(i, ds, left=0, height=0.62, color=BLUE)
            if not r.dec_idle:
                # no measured timeline: the whole pass as one undifferentiated
                # block, so the row is visibly NOT making the finer claim.
                axW.barh(i, max(t2 - ds, 0.0), left=ds, height=0.62,
                         color=GREEN, alpha=0.45, hatch="//")
            else:
                axW.barh(i, max(t2 - ds, 0.0), left=ds, height=0.62, color=GREEN)
                for a, b in r.dec_idle["idle_spans"]:
                    axW.barh(i, (b - a) * MS, left=(a - t0) * MS, height=0.62,
                             color=VIOLET)
                for a, b in r.dec_idle["kv_blocked_spans"]:
                    axW.barh(i, (b - a) * MS, left=(a - t0) * MS, height=0.62,
                             color=CORAL)
            pf = r.links[0].pause_frames
            if pd.notna(pf):
                axW.text(t2 * 1.012, i, f"{pf:,.0f} PAUSE", va="center",
                         fontsize=9, color=MUTED)
        axW.set_xlim(0, end * 1.28)
        axW.set_xlabel("ms after the first token")
        axW.set_title("Zoom: the first decode pass, as measured",
                      fontsize=10, loc="left", pad=32)
        # outside, above: the bars run the full width and any in-axes corner
        # would sit on top of a row's PAUSE label.
        axW.legend(handles=[Patch(color=BLUE, label="handoff in flight"),
                            Patch(color=GREEN, label="decode progressing"),
                            Patch(color=CORAL, label="idle, resumed by a KV arrival"),
                            Patch(color=VIOLET, label="idle, other")],
                   fontsize=8.5, ncol=4, loc="lower center",
                   bbox_to_anchor=(0.5, 1.005), frameon=False)

        axT.set_yticks(range(len(wf)))
        axT.set_yticklabels([f"{r.buffer_mb:g} MiB" + (f"\n{detail[i]}"
                                                       if i in detail else "")
                             for i, r in enumerate(wf)], fontsize=8.5)
        axT.invert_yaxis()
        for a in (axT, axW):
            a.grid(True, axis="x", alpha=0.3)
        save_fig(fig, outdir, "04_kv_transfer_and_handoff.png", written)

    # 05 FIRST DECODE PASS, ITS STALL, AND THE KV TAIL ----------------------- #
    # Two panels. (The old "token 2 after token 1" panel duplicated figure 04's
    # bar totals and is gone; the per-stage lateness panel is folded into the
    # right panel as its worst stage -- stage 0's -200 ms of fully-masked slack
    # was compressing the axis that matters. Per-stage values stay in the CSV.)
    #   LEFT   the effect: the first decode pass against the steady inter-token
    #          gap. The control is drawn, not just used as a denominator: its
    #          flatness is what says the buffer touches only the transient.
    #   RIGHT  the same excess (first pass - control), the KV tail past the
    #          decode start, the worst stage's KV lateness -- and the STALL AS
    #          MEASURED inside the pass (idle a KV arrival ends, on the critical
    #          rank). The last curve is the one to read: it agrees with the
    #          excess (T1/32k at 64 MiB: 5.357 vs 5.333 ms, two independent
    #          measurements) while the KV tail sits 2.5x above both, or 4 ms
    #          above a pass that never waited at all. The tail is an upper
    #          bound; the old caption called the three curves coincident, and
    #          on this sweep they are not.
    if s["tok2_latency_ns"].notna().any():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

        axL.plot(x, s["tok2_latency_ns"] * MS, "o-", color=CORAL,
                 label="first decode pass")
        if s["itl_steady_ns"].notna().any():
            axL.plot(x, s["itl_steady_ns"] * MS, "v--", color=MUTED,
                     label="steady inter-token gap (control)")
        axL.set_title("First decode pass vs steady state", fontsize=11)
        axL.set_ylabel("Decode start → token 2 (ms)")

        # The measured stall is drawn as a WIDE PALE BAND under the excess curve,
        # not as another line: the two are independent measurements of the same
        # wait and they land on top of each other, so two thin lines would just
        # hide one another and the agreement would read as a missing series.
        for col, colour, style, width, alpha, label in (
                ("dec_kv_block_ns", BLUE, "-", 6.0, 0.35,
                 "measured stall (idle a KV arrival ends)"),
                ("decode_kv_stall_ns", CORAL, "o-", 1.6, 1.0,
                 "first-pass stall (pass − control)"),
                ("kv_tail_after_dec_start_ns", VIOLET, "s--", 1.6, 1.0,
                 "KV tail past decode start (upper bound)"),
                ("dec_kv_lateness_ns", GREEN, "^:", 1.6, 1.0,
                 "KV lateness, worst stage")):
            if col in s.columns and s[col].notna().any():
                axR.plot(x, s[col] * MS, style, color=colour, lw=width,
                         alpha=alpha, label=label)
        axR.axhline(0.0, color="k", linestyle=":", alpha=0.5)
        axR.set_title("First-pass stall, KV tail and worst-stage lateness",
                      fontsize=11)
        axR.set_ylabel("ms")

        for a in (axL, axR):
            mark_knees(a, s, label=(a is axL))
            logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
            a.grid(True, alpha=0.3, which="both")
            a.legend(fontsize=8)
        save_fig(fig, outdir, "05_decode_kv_stall.png", written)

    # 06 DECODE FIRST TP ALL-REDUCE vs ITS STEADY STATE ---------------------- #
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
        fig.suptitle("Congestion per KV-crossed link", y=1.02)
        save_fig(fig, outdir, "09_per_link_congestion.png", written)

    # 10 PEAK AND MEAN QUEUE OCCUPANCY AT THE BOTTLENECK --------------------- #
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
        axR.set_title("Mean ÷ peak occupancy")
        axR.grid(True, alpha=0.3, which="both")
        if axR.get_legend_handles_labels()[0]:
            axR.legend(fontsize=8)
        fig.suptitle("Peak and mean queue occupancy at the bottleneck",
                     y=1.02)
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
