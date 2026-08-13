#!/usr/bin/env python3
"""
incast_sweep — what a KV-cache incast costs the inference, and what the
per-switch buffer can do about it.

The question, stated so it can be answered
--------------------------------------------------------------------------------
At the prefill/decode handover, several prefill ranks push their shard of one
layer's KV cache into the SAME decode rank at once. The obvious thing to measure
is how long the handover takes — and that is exactly the thing that carries no
information: the transfer moves a fixed number of bytes over a link of fixed
rate, so its duration is bytes/rate whatever the fan-in and whatever the buffer.
On this data the floor is 134.2 ms per decode rank and every run lands within
1–8% of it.

What a fan-in actually produces is the time the bottleneck link spends NOT
SENDING inside that window: ramp-up, backpressure, retransmission, an unlucky
sharing order. So the measured quantity here is

    kv_idle_ns = window − floor        (measures.LinkStat, per run)

the gap between the window the transfer took and the serialisation floor. It is
delay the decode inherits — the first token to second token interval is the
exposed handover — and it is the only part of the handover any buffer, CC or
placement decision can remove.

The axis, and its control
--------------------------------------------------------------------------------
The incast degree is the RESHARDING RATIO tp_prefill / tp_decode, not the prefill
TP width: a decode rank owns 1/tp_decode of each layer's KV and needs only that
fraction of the prefill shards. So the three levels are fan-in 1, 2 and 4 — and
T2.1, at fan-in 1, is the NO-INCAST CONTROL: identical KV volume, identical link
rate, identical model, one sender per layer instead of several. Everything on the
degree axis is read against it (utils.incast.fan_in).

Two things are deliberately NOT put on the degree axis AS NETWORK RESULTS,
because a wider fan-in here also means a wider prefill pool and therefore more
compute: TTFT and the makespan. They fall along that axis for a reason that has
nothing to do with the network, and they stay in summary.csv as columns.

Figure 11 is the one place they are read along that axis, and it is the
opposite of that mistake rather than an instance of it: it exists to MEASURE
how little of their movement is network. A wider prefill buys TTFT, but the
handover it feeds is anchored to the instant the prefill starts PRODUCING the
KV, not to the first token, and its length is fixed by bytes/rate -- so the
prefill stops covering it and the same transfer moves from hidden to exposed.
The accounting there (Δmakespan = Δ(KV stream start) + Δfloor + Δstarved +
Δincast + Δlag + Δrest) attributes the payback term by term, and the network
term is one small bar among the structural ones.

What was cut, and why
--------------------------------------------------------------------------------
This file used to carry eighteen figures. The ones removed measured quantities
that, on this data, sit below the resolution of a single run per configuration:
the intra-stage KV skew (0.2–3.9 ms against a 134 ms floor, non-monotone), TTFT
(flat by construction — it is upstream of the KV), the makespan (±0.4%,
non-monotone), and the decode's first all-reduce entry skew (µs-scale). The
stall-versus-cause overlay went too, on the argument that the first-pass stall
IS the KV tail -- an argument that turned out to be false (the tail is only an
upper bound; see kv_tail_after_dec_start_ns) but a figure that is not missed:
figure 02 now draws the stall itself, measured, at its true positions inside the
pass. Those columns all survive in summary.csv;
what is gone is the suggestion that they are findings.

Saturation is reported rather than hidden: from a level's saturation knee upward
the runs are byte-identical (T3 from 24 MiB, T4 from 48), so those points are not
independent samples and the per-level report says so.

Prefill/decode split, per topology
--------------------------------------------------------------------------------
The classification that names a flow 'kv' depends on the rank->role placement,
and the placement differs per topology. It is recovered per level from that
level's ASTRA trace (roles.from_astra) and then VERIFIED against the fabric
traffic (roles.check): the summary reports the KV / 'other' flow counts per
topology so a split that silently failed on a wider-TP topology is visible
rather than assumed. The link set and its order are fixed per LEVEL from its
smallest-buffer run (canonical_links), so a bn_* curve is one physical link and
not a walk from one link to another as the ranking shifts.

Reused verbatim (one definition of a metric): the ns-3 / ASTRA readers, the flow
classification and bottleneck search (utils.flows/fabric), the PP-skew measure
(utils.pp), and the shared measures of utils.measures — TTFT, the KV barrier,
per-link stats (floor/window/idle come from LinkStat), the decode-side measures,
their worst-stage reduction and the knee readings. Only the incast-specific
orchestration and the figures are this file's own.

One measure of measures.barrier is deliberately dropped here:
kv_stream_duration_ns (last arrival − first arrival) FALLS with the fan-in while
the transfer gets longer, because a wider fan-in delays the first arrival more
than the last. buffer_sweep keeps it because it holds the fan-in fixed.

Output
--------------------------------------------------------------------------------
    <out>/01_incast_cost.png                 THE HEADLINE: bottleneck link idle
                                             time (window − floor) vs buffer and
                                             vs fan-in, against the fan-in-1 control
    <out>/02_first_token_to_second.png       where the first->second token interval
                                             goes, AS A SHARE of it: handoff |
                                             decode progressing | the MEASURED
                                             idle it spends waiting on KV, one bar
                                             per buffer, panel/topo (the absolute
                                             interval and its dilation over the
                                             steady ITL are written on each bar)
    <out>/03_fabric_cost.png                 what the buffer DOES change: PFC frames,
                                             dropped packets, peak queue
    <out>/04_kv_shard_skew_distribution.png  the per-(stage, layer) shard-skew
                                             population, boxplots per (buffer, stage)
    <out>/05_kv_fct_cdf.png                  KV flow-completion-time CDF per buffer,
                                             panel/topo (tick = p99)
    <out>/06_buffer_bloat.png                peak vs mean occupancy at each topology's
                                             bottleneck, and their ratio
    <out>/07_per_link_congestion.png         efficiency, PAUSE and peak queue for every
                                             KV-crossed link, one row per topology
    <out>/08_<level>_queue_fill_busy_switches.png  queue(t), busiest switches only
    <out>/09_<level>_kv_cumulative_arrival.png     cumulative KV per decode rank; a flat
                                             stretch in a curve IS idle link time
    <out>/10_insensitivity.png               fabric state (PAUSE, drops; orders of
                                             magnitude) next to application times
                                             (normalised; a fixed ±2% window), same
                                             buffer axis
    <out>/11_prefill_vs_handover_tradeoff.png  THE TRADE-OFF: left, the handover on
                                             ABSOLUTE time per topology, cut in three
                                             — hidden under the prefill, exposed while
                                             the first token is still in flight to the
                                             decode (nothing to hold up), and landing
                                             with the decode awake (the only part that
                                             can stall it, with the measured stall
                                             written on the row); right, the TTFT gain
                                             of the widest prefill and every term that
                                             takes it back, down to the net makespan
    <out>/12_per_stage_handover.png         PER DECODE STAGE: left, when each
                                             stage's handover runs and completes on
                                             the absolute clock (same bytes, shifted
                                             start); right, its window as a % of the
                                             floor over the buffer sweep -- how much
                                             longer the same payload actually took
    <out>/summary.csv     one row per run: the measures above plus the controls and
                          references kept out of the figures (kv_skew_ms, ttft_ms,
                          total_exec_ms, pp_skew_us, kv_skew_global_ms), the
                          per-stage decN_* blocks, the per-link link{i}_* blocks
                          and the three knees as per-topology constant columns.

Usage
-----
    python3 incast_sweep.py
    python3 incast_sweep.py --levels T3 T4 --top-switches 3 --top-links 6
    python3 incast_sweep.py -o /tmp/incast
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils import astra, incast, ns3, pp, roles
from utils import flows as flowlib
from utils.cli import Abort, drain_warnings, need, warn
from utils.fabric import parse_ns3_config, parse_topology
from utils.measures import (LinkStat, barrier, decode_ar_stats,
                            decode_stall_stats, decode_worst_stage,
                            first_pass_stall, knee_scalars, kv_handover_idle,
                            kv_layer_skew, kv_rank_series, kv_skew_stats,
                            link_metrics, ttft_from, victim_pause_intervals,
                            wire_header_bytes)
from utils.paths import BUFFER_AXIS, fresh_dir
from utils.plots import (AMBER, BLUE, CORAL, GREEN, KNEE_STYLE, LOSS_RED, MS, MUTED,
                         VIOLET, buf_colour, downsample_max, logx_pow2,
                         loss_proxies, mark_knees, mark_lossy, save_fig,
                         zoom_y)
from utils.roles import Placement

NAN = float("nan")

# Which columns this sweep's knees are read from (utils.measures.knee_scalars
# defines the readings; each sweep declares its own inputs). buffer_sweep watches
# its single bottleneck link; here the PAUSE count that matters is the WHOLE
# fabric's -- an incast topology backpressures on several switches at once, and
# the per-switch census is figure 11's business, not the knee's.
PAUSE_KNEE_COL = "total_pause_frames"

# Saturation: (column, relative tolerance, absolute tolerance) -- "this run is
# indistinguishable from the largest-buffer run". Same idea as buffer_sweep's
# set, with the incast headline (intra-stage KV skew) in place of the global
# cross-rank spread, and the fabric-wide PAUSE count as a COUNT (absolute
# tolerance of half a frame; a relative one is meaningless at zero).
SATURATION_METRICS = (
    ("ttft_ns", 1e-3, 0.0),
    ("kv_gate_ns", 1e-3, 0.0),
    ("tok2_latency_ns", 1e-3, 0.0),
    ("kv_skew_ns", 1e-3, 0.0),
    ("bn_qpeak_mb", 1e-3, 0.0),
    ("total_pause_frames", 0.0, 0.5),
)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    tag: str = ""
    level: str = ""
    incast_degree: int = 0                # fan-in: prefill senders per decode
                                          # rank per layer = tp_p/tp_d (1/2/4).
                                          # NOT the prefill TP width.
    prefill_tp: int = 0                   # the width that names the topology
    decode_tp: int = 0
    buffer_mb: float = NAN
    buffer_bytes: float = NAN
    bottleneck: str = ""

    # -- THE HEADLINE: what the incast actually costs, in time ---------------- #
    # The handover cannot be faster than bytes/rate, so its DURATION is not a
    # measure of anything an incast does. What the incast produces is the time
    # the receiving link spends NOT SENDING during the transfer -- the gap
    # between the window it took and the hard serialisation floor. That gap is
    # the delay the decode inherits, and it is the only part any buffer, CC or
    # placement decision can remove.
    #
    # Measured from the ASTRA CSV (measures.kv_handover_idle), not from fct.txt
    # plus the topology: the two agree to the NANOSECOND and to the BYTE on this
    # fabric, but the CSV route needs no flow classification, no path
    # reconstruction and no bottleneck ranking, and it reports EVERY decode rank
    # instead of the one link a ranking happened to pick. Its floor counts WIRE
    # bytes (payload plus per-packet headers, the INT header sized by this run's
    # CC_MODE), so the residue is idle time and not encapsulation.
    kv_floor_ns: float = NAN              # wire bytes * 8 / receiver link rate
    kv_window_ns: float = NAN             # first send -> last arrival, measured
    kv_idle_ns: float = NAN               # window - floor, at the WORST receiver
    kv_idle_mean_ns: float = NAN          # mean over the decode ranks
    kv_link_busy_pct: float = NAN         # floor / window, %
    kv_link_rate_gbps: float = NAN        # narrowest rate on the KV path
    kv_header_bytes: float = NAN          # per-packet wire overhead assumed
    # The idle SPLIT (measures.kv_handover_idle). Not all of it is incast: a
    # receiving link is also starved whenever every sender feeding it is busy
    # putting something else on its NIC, which on a PP prefill is the activation
    # handed to the next stage -- emitted by all the senders of a stage at once,
    # so the receiver loses its whole supply for ~6.8 ms whatever the fan-in.
    # That term is identical in every topology here, so it is the residue, not
    # the total, that measures the incast.
    kv_starved_ns: float = NAN            # all senders busy elsewhere
    kv_incast_ns: float = NAN             # idle - starved: THE incast cost
    kv_incast_mean_ns: float = NAN
    kv_starved_mean_ns: float = NAN
    kv_rank_idle_ns: dict = field(default_factory=dict)     # rank -> idle ns
    kv_rank_incast_ns: dict = field(default_factory=dict)   # rank -> incast ns
    # Per DECODE STAGE, mean over that stage's ranks. Aggregating over ALL the
    # decode ranks hides the one structural difference between them: only the
    # ranks fed by a NON-LAST prefill stage see the starvation, because only a
    # non-last stage has a PP activation to forward. Their residue therefore
    # also contains the recovery from that 6.8 ms outage, and mixing them with
    # the clean ranks is what made the headline non-monotone in the fan-in.
    kv_stage_incast_ns: dict = field(default_factory=dict)   # decode stage -> mean
    kv_stage_starved_ns: dict = field(default_factory=dict)  # decode stage -> mean

    # -- the two execution times, absolute ns ------------------------------- #
    ttft_ns: float = NAN                  # end of prefill (FIRSTTOK send)
    total_exec_ns: float = NAN            # last op end: whole-workload wall clock

    # -- KV arrival skew WITHIN a decode stage ------------------------------ #
    # Kept as a COLUMN, not as a figure: on this data it is 0.2-3.9 ms against a
    # 134 ms floor, i.e. 0.2% of the makespan, and it moves non-monotonically
    # with one run per point -- below the resolution of the experiment. The
    # skew that does scale with the fan-in is the per-(stage, layer) one below,
    # which figure 04 shows as a distribution.
    kv_skew_ns: float = NAN               # worst intra-stage skew: max over decode
                                          # stages of (max-min KV arrival among that
                                          # stage's OWN TP ranks)
    kv_skew_mean_ns: float = NAN          # mean intra-stage skew over decode stages
    kv_skew_global_ns: float = NAN        # max-min over ALL decode ranks (reference)
    kv_skew_stage_ns: dict = field(default_factory=dict)  # stage idx -> intra skew ns
    kv_gate_ns: float = NAN               # decode start = last KV arrival
    kv_ready_min_ns: float = NAN
    decode_ranks: str = ""
    # kv_stream_duration_ns (last arrival - first arrival, from measures.barrier)
    # is deliberately NOT recorded here. Across fan-in values it is a trap: it
    # FALLS 128 -> 121 -> 71 ms from fan-in 1 to 4 while the transfer actually
    # gets LONGER (144 ms measured on ns-3 at fan-in 4), because a wider fan-in
    # delays the FIRST arrival more than the last. buffer_sweep keeps it because
    # it holds the fan-in fixed; here it would only invite a wrong reading.

    # -- KV skew within each TP group (measures.kv_skew_stats), figure 04 ---- #
    # One skew per (decode stage, layer): the spread between the arrivals of
    # the KV shards feeding the same TP group -- the wait that layer's first
    # decode all-reduce inherits. Finer grain than kv_skew_ns (per stage only).
    kv_tp_skew_min_ns: float = NAN
    kv_tp_skew_mean_ns: float = NAN
    kv_tp_skew_p99_ns: float = NAN
    kv_tp_skew_n: int = 0

    # -- what the user sees (measures.decode_stall_stats) -------------------- #
    # tok2_after_tok1_ns is the headline's twin: the gap between the first and
    # the second token IS the exposed KV handover, so every millisecond the
    # incast wastes on the link (kv_idle_ns) lands here.
    dec_start_ns: float = NAN             # first decode COMP start
    tok2_ns: float = NAN                  # second token: first DECFB send
    tok2_latency_ns: float = NAN          # tok2 - dec_start: first decode pass
    tok2_after_tok1_ns: float = NAN       # tok2 - TTFT: user-visible gap
    itl_steady_ns: float = NAN            # steady inter-token gap (control)

    # -- the three instants the first-token -> second-token interval is cut at
    # (figure 02). Differences of quantities measured above, named here so the
    # CSV carries them and no reader has to subtract two columns.
    dec_start_after_ttft_ns: float = NAN  # dec_start - ttft: the exposed handoff
                                          # (the FIRSTTOK message queued behind
                                          # the KV bulk)
    kv_gate_after_ttft_ns: float = NAN    # kv_gate - ttft: until the LAST KV lands
    kv_tail_after_dec_start_ns: float = NAN  # kv_gate - dec_start: KV still in
                                          # flight when the pipeline wakes. An
                                          # UPPER BOUND on the stall, not the
                                          # stall: the decode consumes its KV
                                          # layer by layer and waits only where
                                          # it outruns the transfer.
    # The stall as MEASURED inside the pass (utils.measures.first_pass_stall):
    # idle between the decode ranks' own it=0 intervals, attributed to the KV
    # arrival that ends it, unioned over the ranks. Figure 02 draws these spans;
    # the bound above is what the figure used to draw in their place.
    dec_kv_block_ns: float = NAN          # union of the KV-blocked idle
    dec_other_idle_ns: float = NAN        # idle no KV arrival explains
    dec_kv_block_max_ns: float = NAN      # worst SINGLE rank's KV-blocked idle
    dec_crit_rank: float = NAN            # the rank that emitted token 2

    # -- the SIGNED twin of the (stage, layer) skew population, at TP=2 only:
    # arrival(shard 1) - arrival(shard 0), median over layers. Near zero = the
    # shards are equally (un)lucky and the skew is congestion noise; away from
    # zero = one shard is systematically late, i.e. a path asymmetry no buffer
    # can fix. NaN at wider TP, where "which one is late" has no single answer.
    kv_shard_bias_ns: float = NAN

    # -- buffer bloat at the (fixed) bottleneck link (figure 06) ------------- #
    q_bloat_ratio: float = NAN            # qmean/qpeak: sustained load or spikes

    # -- PP skew: ~0 by construction for these placements, kept as a column
    # only so its absence is on record rather than assumed.
    pp_skew_ns: float = NAN
    pp_skew_mean_ns: float = NAN
    pp_available: bool = False

    # -- prefill/decode split health, per topology -------------------------- #
    kv_flows: int = 0
    other_flows: int = 0
    split_ok: bool = True

    # -- the KV-crossed links + fabric totals -------------------------------- #
    # links[0] is the bottleneck; the list follows the LEVEL-WIDE fixed label
    # order (see canonical_links), so link i is the same physical link at every
    # buffer of one topology -- without that, a "bottleneck occupancy vs buffer"
    # curve silently walks from one link to another (T3's does: 16->26 at the
    # small buffers, 16->25 at the large ones).
    links: list = field(default_factory=list)            # list[LinkStat]
    total_pause_frames: float = NAN
    bn_pause_intervals: list = field(default_factory=list)

    # -- packet loss ("Headroom full" drops = lossless-fabric violation) ----- #
    dropped_packets: float = NAN          # NaN => UNKNOWN (no drops.txt captured)
    loss_captured: bool = False           # True iff drops.txt existed for this run
    packets_delivered: int = 0            # sum ceil(size/payload) over flows
    dropped_per_switch: dict = field(default_factory=dict)  # switch id -> drops

    # -- KV FCT raw material (per-flow, for cc_sweep's distribution/CDF) ------ #
    # kv_fct_ns are the durations of the ASTRA CSV KV send rows -- bit-identical
    # to the ns-3 fct of the same flows (verified to the nanosecond), but already
    # classified by name, so no downstream re-read of fct.txt is ever needed.
    # kv_slowdown alone stays ns-3-sourced: its denominator standalone_fct
    # (base_rtt + bytes/pairBw) exists only in fct.txt.
    kv_fct_ns: object = None              # np.ndarray | None
    kv_slowdown: object = None            # np.ndarray | None
    kv_bytes: float = NAN                 # total bulk-KV payload bytes
    kv_goodput_gbps: float = NAN          # kv_bytes*8 / (last arrival - first send)
    # The tail of that FCT population as scalars (figure 05 draws the CDF). An
    # incast is a TAIL story -- the mean KV flow is fine and the last one gates
    # the stage -- so p99/max are the columns, with p50 as the reference.
    kv_fct_p50_ns: float = NAN
    kv_fct_p99_ns: float = NAN
    kv_fct_max_ns: float = NAN
    kv_slowdown_p99: float = NAN          # fct / (base_rtt + bytes/pairBw)

    # -- per-switch, not flattened ------------------------------------------ #
    qseries: dict = field(default_factory=dict)          # sw -> (ts_ns, bytes) downsampled
    qswitch_peak: dict = field(default_factory=dict)     # sw -> peak total bytes
    qswitch_mean: dict = field(default_factory=dict)     # sw -> mean total bytes
    pfc_per_switch: dict = field(default_factory=dict)   # sw -> PAUSE frame count
    pause_intervals: dict = field(default_factory=dict)  # sw -> [(start,end)]

    # -- decode-side raw data (buffer_sweep's measures), not flattened as-is - #
    # per RECEIVER, the whole kv_handover_idle record (start/end/floor/idle/
    # starved/incast/bytes) and the decode stage each receiver belongs to.
    # The scalars above reduce this to the worst receiver and to per-stage
    # means; figure 12 needs the receivers themselves, with the two ends of
    # each window, to put a completion time on an absolute clock.
    kv_rank_stat: dict = field(default_factory=dict)     # rank -> metrics dict
    kv_stage_ranks: dict = field(default_factory=dict)   # decode stage -> [ranks]
    kv_rank_series: dict = field(default_factory=dict)   # rank -> (times_ns, cumbytes)
    kv_layer_skew: object = None                         # per (stage, layer) skew
    dec_ar: dict = field(default_factory=dict)           # stage -> first/steady AR
    dec_stall: dict = field(default_factory=dict)        # stage -> KV lateness
    dec_idle: dict = field(default_factory=dict)         # first-pass busy/idle
                                                         # spans, per rank and
                                                         # unioned (figure 02)

    # -- derived ------------------------------------------------------------ #
    @property
    def total_over_ttft(self) -> float:
        return (self.total_exec_ns / self.ttft_ns
                if pd.notna(self.total_exec_ns) and pd.notna(self.ttft_ns)
                and self.ttft_ns > 0 else NAN)

    @property
    def lossy(self) -> bool:
        """This run dropped packets (loss is known AND non-zero)."""
        return pd.notna(self.dropped_packets) and self.dropped_packets > 0

    @property
    def drop_rate(self) -> float:
        return (self.dropped_packets / self.packets_delivered
                if pd.notna(self.dropped_packets) and self.packets_delivered > 0
                else NAN)

    def flat(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("qseries", "qswitch_peak", "qswitch_mean",
                          "pfc_per_switch", "pause_intervals", "links",
                          "bn_pause_intervals", "dropped_per_switch",
                          "kv_skew_stage_ns", "kv_rank_series", "kv_layer_skew",
                          "kv_rank_idle_ns", "kv_rank_incast_ns",
                          "kv_rank_stat", "kv_stage_ranks",
                          "kv_stage_incast_ns", "kv_stage_starved_ns",
                          "dec_ar", "dec_stall", "dec_idle",
                          "kv_fct_ns", "kv_slowdown")}
        d["total_over_ttft"] = self.total_over_ttft
        d["lossy"] = self.lossy
        d["drop_rate_pct"] = self.drop_rate * 100 if pd.notna(self.drop_rate) else NAN
        # the headline, in the unit the figures use
        d["kv_floor_ms"] = self.kv_floor_ns * MS
        d["kv_window_ms"] = self.kv_window_ns * MS
        d["kv_idle_ms"] = self.kv_idle_ns * MS
        d["kv_idle_pct_of_floor"] = (100 * self.kv_idle_ns / self.kv_floor_ns
                                     if pd.notna(self.kv_idle_ns)
                                     and self.kv_floor_ns else NAN)
        d["kv_idle_mean_ms"] = self.kv_idle_mean_ns * MS
        d["kv_starved_ms"] = self.kv_starved_ns * MS
        d["kv_incast_ms"] = self.kv_incast_ns * MS
        d["kv_incast_mean_ms"] = self.kv_incast_mean_ns * MS
        d["kv_starved_mean_ms"] = self.kv_starved_mean_ns * MS
        for r in sorted(self.kv_rank_idle_ns):
            d[f"kv_idle_rank{r}_ms"] = self.kv_rank_idle_ns[r] * MS
            d[f"kv_incast_rank{r}_ms"] = self.kv_rank_incast_ns.get(r, NAN) * MS
        for si in sorted(self.kv_stage_incast_ns):
            d[f"kv_incast_dec{si}_ms"] = self.kv_stage_incast_ns[si] * MS
            d[f"kv_starved_dec{si}_ms"] = self.kv_stage_starved_ns.get(si, NAN) * MS
        d["kv_skew_ms"] = self.kv_skew_ns * MS                  # worst intra-stage
        d["kv_skew_mean_ms"] = self.kv_skew_mean_ns * MS
        d["kv_skew_global_ms"] = self.kv_skew_global_ns * MS    # inter-stage, reference
        for si, v in sorted(self.kv_skew_stage_ns.items()):
            d[f"kv_skew_d{si}_ms"] = v * MS                     # per decode stage
        d["pp_skew_us"] = self.pp_skew_ns / 1e3
        d["ttft_ms"] = self.ttft_ns * MS
        d["total_exec_ms"] = self.total_exec_ns * MS
        d["total_minus_ttft_ms"] = (self.total_exec_ns - self.ttft_ns) * MS
        d["kv_tp_skew_mean_ms"] = self.kv_tp_skew_mean_ns * MS  # fig 04, ms twins
        d["kv_tp_skew_p99_ms"] = self.kv_tp_skew_p99_ns * MS
        d["tok2_latency_ms"] = self.tok2_latency_ns * MS        # fig 02, ms twins
        d["tok2_after_tok1_ms"] = self.tok2_after_tok1_ns * MS
        d["itl_steady_ms"] = self.itl_steady_ns * MS
        # The model-clock LEDGER: one accounting identity per run,
        #     tok1→tok2 = head + floor + starved + incast + lag ,
        # exact by construction because head is computed as the residual.
        #   lag  = tok2 − last KV arrival: the release lag. ≪ ITL means the
        #          final KV byte releases the second token, and that the ~one
        #          ITL of genuine first-pass compute was interleaved with the
        #          arrivals. It does NOT by itself make the rest of the
        #          decode-awake stretch a stall -- that is dec_kv_block_ns,
        #          measured as idle inside the pass (figure 02);
        #   head = where the KV stream's first send sits relative to tok1
        #          (negative: during prefill, hidden under it). Everything
        #          else — floor + starved + incast + lag — is EXPOSED on the
        #          second token's clock.
        lag_ns = (self.tok2_ns - self.kv_gate_ns
                  if pd.notna(self.tok2_ns) and pd.notna(self.kv_gate_ns)
                  else NAN)
        d["lag_ms"] = lag_ns * MS
        d["lag_over_itl"] = (lag_ns / self.itl_steady_ns
                             if pd.notna(lag_ns) and pd.notna(self.itl_steady_ns)
                             and self.itl_steady_ns > 0 else NAN)
        d["stream_head_ms"] = (
            (self.tok2_after_tok1_ns - lag_ns - self.kv_window_ns) * MS
            if all(pd.notna(v) for v in (self.tok2_after_tok1_ns, lag_ns,
                                         self.kv_window_ns)) else NAN)
        # The same window, cut at the FIRST TOKEN instead of at its own start:
        # how much of the transfer ran before tok1 (hidden under the prefill,
        # free on the model's clock) and how much after it (exposed: the user
        # is already waiting for the second token). The two always sum to the
        # window, so a prefill that gets faster does not shorten the transfer --
        # it moves it from one column to the other. kv_stream_start_ms is the
        # absolute instant the whole handover is anchored to.
        d["kv_stream_start_ms"] = (self.ttft_ns * MS + d["stream_head_ms"]
                                   if pd.notna(d["stream_head_ms"]) else NAN)
        if pd.notna(d["stream_head_ms"]) and pd.notna(self.kv_window_ns):
            hidden = min(max(-d["stream_head_ms"], 0.0), self.kv_window_ns * MS)
            d["kv_hidden_ms"] = hidden
            d["kv_exposed_ms"] = self.kv_window_ns * MS - hidden
        else:
            d["kv_hidden_ms"] = d["kv_exposed_ms"] = NAN
        # per-stage decode columns, same names buffer_sweep flattens to, so the
        # cross-tool readers see one schema
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
        d["kv_fct_p99_ms"] = self.kv_fct_p99_ns * MS
        d["kv_fct_max_ms"] = self.kv_fct_max_ns * MS
        # the bottleneck (links[0]) under short names, then EVERY level-fixed
        # link under its index -- figure 07 and any cross-link reading come from
        # the indexed block, which is comparable across the buffers of one level
        # because the label order is fixed for the level (canonical_links).
        if self.links:
            ls: LinkStat = self.links[0]
            d["bn_eff_pct"] = ls.eff_pct
            d["bn_kv_gb"] = ls.kv_bytes / 1e9 if pd.notna(ls.kv_bytes) else NAN
            d["bn_rate_gbps"] = ls.rate_gbps
            d["bn_delivered_gbps"] = ls.delivered_gbps
            d["bn_conc_peak"] = ls.conc_peak
            d["bn_pause_frames"] = ls.pause_frames
            d["bn_pause_pct_of_window"] = ls.pause_pct_of_window
            d["bn_qpeak_mb"] = ls.qpeak_bytes / 2**20 if pd.notna(ls.qpeak_bytes) else NAN
            d["bn_qmean_mb"] = ls.qmean_bytes / 2**20 if pd.notna(ls.qmean_bytes) else NAN
        for i, ls in enumerate(self.links):
            d[f"link{i}_label"] = ls.label
            d[f"link{i}_eff_pct"] = ls.eff_pct
            d[f"link{i}_delivered_gbps"] = ls.delivered_gbps
            d[f"link{i}_pause_frames"] = ls.pause_frames
            d[f"link{i}_qpeak_bytes"] = ls.qpeak_bytes
            d[f"link{i}_qmean_bytes"] = ls.qmean_bytes
        return d


# --------------------------------------------------------------------------- #
# Measurement helpers that are incast's own. (The shared per-run measures come
# from utils.measures; the per-switch PFC census is a PfcLog method.)
# --------------------------------------------------------------------------- #
def kv_stage_skew(kv: pd.DataFrame, placement: Placement) -> dict:
    """KV-cache arrival skew computed WITHIN each decode stage (a TP group),
    not across the whole decode pool.

    The KV cache of a decode stage is tensor-parallel sharded over the ranks of
    THAT stage; the stage's attention / all-reduce cannot progress until its
    SLOWEST rank has its shard. So the sync cost that matters per stage is, over
    the ranks of one stage,

        skew_stage = max_r ready_r - min_r ready_r ,

    with ready_r = the last KV arrival at rank r (the max over the incast fan-in
    of prefill senders onto r). This is deliberately intra-stage: the large
    spread the global barrier reports is mostly BETWEEN stages (an early decode
    stage receives its KV long before a later one) -- a pipeline effect, not the
    per-stage skew the receiving TP group actually pays.

    Returns per_stage {stage_idx: skew_ns} over the stages that had >=2 ranks
    fed, the scalar worst_ns / mean_ns over those stages, the global_ns spread
    (kept only for reference), and short_stages: stages that declare >=2 ranks
    but had fewer than 2 receive any KV (their intra-stage skew is undefined)."""
    ready: dict[int, float] = {}
    for d in placement.decode_ranks:
        arr = kv.loc[kv["dst"] == d, "arrival"]
        if len(arr):
            ready[int(d)] = float(arr.max())

    per_stage: dict[int, float] = {}
    short: list[int] = []
    for si, ranks in enumerate(placement.decode):
        fed = [ready[r] for r in ranks if r in ready]
        if len(fed) >= 2:
            per_stage[si] = max(fed) - min(fed)
        elif len(ranks) >= 2:
            short.append(si)
    worst = max(per_stage.values()) if per_stage else NAN
    mean = float(np.mean(list(per_stage.values()))) if per_stage else NAN
    glob = (max(ready.values()) - min(ready.values())) if ready else NAN
    return {"per_stage": per_stage, "worst_ns": worst, "mean_ns": mean,
            "global_ns": glob, "short_stages": short}


def analyse(tag: str, p: incast.IncastPaths, placement: Placement,
            chosen_labels: list[str]) -> Row:
    """One run. analyse_level has already aborted on missing config inputs and
    skipped runs with missing outputs (usable_tags), so nothing here re-checks
    file presence.

    `chosen_labels` is the level's FIXED link set and display order (see
    canonical_links): this run reports those links, in that order, whether or
    not its own congestion ranking agrees -- a link no KV flow crosses here is
    recorded as an empty LinkStat rather than silently swapped for another."""
    buf = BUFFER_AXIS.value(tag)
    need(buf is not None, f"{tag}: no 'buf<num>' token in the name.")
    ns3_dir = p.ns3_run(tag)
    topo = parse_topology(p.topology(tag))
    cfg = parse_ns3_config(p.config(tag))
    for w in cfg.warnings():
        warn(f"{tag}: {w}")
    need(cfg.buffer_mb is None or abs(cfg.buffer_mb - buf) < 1e-6,
         f"{tag}: BUFFER_SIZE={cfg.buffer_mb} MiB in config.txt but "
         f"'buf{buf:g}' in the name. One of the two is lying.")

    row = Row(tag=tag, level=p.level, incast_degree=incast.fan_in(placement),
              prefill_tp=incast.prefill_tp(placement),
              decode_tp=incast.decode_tp(placement),
              buffer_mb=float(buf), buffer_bytes=float(buf) * 1024 * 1024)

    # The one read of this run's ASTRA trace: TTFT/total, KV arrivals & FCTs,
    # PP skew and the decode-side measures all share this frame.
    adir = p.astra_run(tag)
    adf = astra.read_run(adir)
    need(adf is not None, f"{tag}: no readable stats_sys*.csv under {adir}.")
    row.total_exec_ns = float(adf["end_tick"].max())
    row.ttft_ns = ttft_from(adf, tag)

    raw = ns3.read_fct(ns3_dir / "fct.txt")
    need(raw is not None and len(raw), f"{tag}: fct.txt has no parsable rows.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)

    # packet loss ("Headroom full"). Read OPTIONALLY (not via need()): a run
    # recorded before stdout capture has no drops.txt and stays analysable, but
    # its loss is UNKNOWN and must not be shown as lossless.
    ds = ns3.read_drops(ns3_dir / "drops.txt")
    row.loss_captured = ds.captured
    row.dropped_packets = float(ds.total) if ds.captured else NAN
    row.dropped_per_switch = dict(ds.per_switch)
    row.packets_delivered = int(np.ceil(
        f["size"].to_numpy() / max(cfg.payload, 1)).sum())
    if not ds.captured:
        warn(f"{tag}: no drops.txt — packet loss UNKNOWN; re-run with the updated "
             f"generate_log_ns3.sh to record it.")
    elif ds.total:
        warn(f"{tag}: NOT lossless — {ds.total} packet(s) dropped "
             f"('Headroom full').")
    split_warnings = roles.check(f, placement)
    for w in split_warnings:
        warn(f"{tag}: {w}")
    vc = f["flow_class"].value_counts()
    row.kv_flows = int(vc.get("kv", 0) + vc.get("kv_ctrl", 0))
    row.other_flows = int(vc.get("other", 0))
    row.split_ok = not split_warnings
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{tag}: no KV flow after classification -- the prefill/decode "
                  f"split does not match this topology's traffic.")

    # KV FCT raw material (see the Row fields): per-flow durations from the CSV
    # KV send rows, filtered to bulk (> one packet) to mirror the 'kv' class
    # exactly; the slowdown array is the only fct.txt-sourced piece.
    kv_send = astra.sends(adf, adf["op_class"] == "KV")
    kv_send = kv_send[kv_send["comm_size"] > cfg.payload]
    if len(kv_send) != len(kv):
        warn(f"{tag}: {len(kv_send)} CSV KV sends vs {len(kv)} classified fct "
             f"kv flows; the FCT stats use the CSV rows.")
    row.kv_fct_ns = kv_send["duration"].to_numpy(dtype=float)
    row.kv_slowdown = kv["slowdown"].dropna().to_numpy(dtype=float)
    if len(row.kv_fct_ns):
        row.kv_fct_p50_ns = float(np.percentile(row.kv_fct_ns, 50))
        row.kv_fct_p99_ns = float(np.percentile(row.kv_fct_ns, 99))
        row.kv_fct_max_ns = float(row.kv_fct_ns.max())
    if len(row.kv_slowdown):
        row.kv_slowdown_p99 = float(np.percentile(row.kv_slowdown, 99))
    row.kv_bytes = float(kv_send["comm_size"].sum()) if len(kv_send) else NAN
    span = (float(kv_send["end_tick"].max() - kv_send["start_tick"].min())
            if len(kv_send) else 0.0)
    # Aggregate KV goodput over the KV phase: total KV bytes over first-send ->
    # last-arrival wall time. Not a link utilisation (the shards cross different
    # links) -- "how fast the whole KV handover completed".
    row.kv_goodput_gbps = row.kv_bytes * 8.0 / span if span > 0 else NAN

    qlen = ns3.read_qlen(ns3_dir / "qlen.txt", series=True)
    need(qlen is not None and qlen.port_max, f"{tag}: qlen.txt has no samples.")
    pfc = ns3.read_pfc(ns3_dir / "pfc.txt")
    need(pfc is not None, f"{tag}: pfc.txt unreadable.")

    run_end = int(f["arrival"].max())

    # The level's fixed link set, scored here. Ranking the links per RUN (as this
    # analyzer used to) makes "the bottleneck" a different physical link at
    # different buffers -- on T3 the deepest queue moves 16->26 to 16->25 halfway
    # up the sweep -- so every bottleneck curve would be a comparison between two
    # links. The set and its order come from the level's smallest-buffer run.
    links_here = {str(l): l for l in
                  flowlib.candidate_links(topo, qlen.port_max, kv)}
    need(links_here, f"{tag}: no link is crossed by any KV flow.")
    for label in chosen_labels:
        bn_i = links_here.get(label)
        if bn_i is None:
            warn(f"{tag}: link {label} (crossed by KV in another run of this "
                 f"level) is not crossed by any KV flow here; recorded as NaN.")
            row.links.append(LinkStat(label=label))
            continue
        row.links.append(link_metrics(kv, bn_i, topo, pfc, qlen, row.buffer_bytes))
    bn = links_here.get(chosen_labels[0])
    need(bn is not None,
         f"{tag}: the level's bottleneck link {chosen_labels[0]} is not crossed "
         f"by any KV flow in THIS run -- it cannot be the bottleneck here.")
    row.bottleneck = str(bn)
    row.bn_pause_intervals = victim_pause_intervals(pfc, bn, topo,
                                                    clamp_to=run_end)
    ls0 = row.links[0]
    # qmean against qpeak at the bottleneck: is the added buffer holding
    # SUSTAINED load, or only rare excursions? (figure 06)
    if pd.notna(ls0.qpeak_bytes) and ls0.qpeak_bytes > 0:
        row.q_bloat_ratio = float(ls0.qmean_bytes / ls0.qpeak_bytes)

    # KV arrival timing and PP skew read the ASTRA stats CSV (per-op end_tick =
    # arrival, cleanly labelled) instead of reconstructing them from fct.txt --
    # same nanosecond values, none of the flow-classification / incast fan-in /
    # wave-grouping heuristics. Fabric metrics above (link, PFC, drops, queues)
    # stay on ns-3. `adf` was read once at the top of this function.
    kv_arr = astra.kv_arrivals(adf)

    # barrier gives the decode-start gate and the per-rank ready times; the
    # headline SKEW is intra-stage (computed separately, NOT barrier's global
    # cross_rank_skew, which is dominated by the inter-stage gap).
    b = barrier(kv_arr, placement)
    row.kv_gate_ns = b["kv_gate_ns"]
    row.kv_ready_min_ns = b["kv_ready_min_ns"]
    row.decode_ranks = b["decode_ranks"]      # b['kv_stream_duration_ns'] is
                                              # deliberately dropped -- see Row

    # THE HEADLINE, from the CSV. The rate is the only thing the ASTRA frame
    # cannot supply, and it is not a free parameter: it is the NARROWEST rate on
    # the KV paths (fabric.pair_bw). Requiring it to be the same for every
    # sender->receiver pair is the precondition of kv_handover_idle -- with one
    # rate the receiver's own link is the pinch and the floor means what it
    # says; with several the fan-in converges somewhere else and this measure
    # would be describing the wrong link.
    rates = {topo.pair_bw.get((int(s_), int(d_)))
             for s_, d_ in zip(kv["src"], kv["dst"])}
    rates.discard(None)
    need(len(rates) == 1,
         f"{tag}: the KV paths do not share one narrowest rate "
         f"({sorted(r / 1e9 for r in rates)} Gb/s), so the receiving link is "
         f"not the single pinch and kv_handover_idle would misdescribe it.")
    row.kv_link_rate_gbps = rates.pop() / 1e9
    row.kv_header_bytes = wire_header_bytes(cfg.cc_mode)
    if pd.isna(row.kv_header_bytes):
        warn(f"{tag}: unknown CC_MODE {cfg.cc_mode}; the handover floor counts "
             f"payload only, so kv_idle_ns absorbs the per-packet headers.")
    # Everything a prefill rank puts on its NIC that is NOT the KV bulk, bulk
    # only: a control-sized FIRSTTOK row spans tens of ms of waiting and would
    # be read as "the sender is busy" for the whole handover.
    other = astra.sends(adf, adf["op_class"] != "KV")
    other = other[other["comm_size"] > astra.CONTROL_MAX_BYTES]
    idle_scal, per_rank = kv_handover_idle(
        kv_arr, kv_send, placement, row.kv_link_rate_gbps * 1e9,
        cfg.payload, row.kv_header_bytes, other_sends=other)
    for k, v in idle_scal.items():
        setattr(row, k, v)
    row.kv_rank_idle_ns = {r: m["idle_ns"] for r, m in per_rank.items()}
    row.kv_rank_incast_ns = {r: m["incast_ns"] for r, m in per_rank.items()}
    row.kv_rank_stat = per_rank
    for si, ranks in enumerate(placement.decode):
        got = [per_rank[r] for r in ranks if r in per_rank]
        if got:
            row.kv_stage_ranks[si] = [r for r in ranks if r in per_rank]
            row.kv_stage_incast_ns[si] = float(np.mean([m["incast_ns"] for m in got]))
            row.kv_stage_starved_ns[si] = float(np.mean([m["starved_ns"] for m in got]))
    sk = kv_stage_skew(kv_arr, placement)
    row.kv_skew_ns = sk["worst_ns"]
    row.kv_skew_mean_ns = sk["mean_ns"]
    row.kv_skew_global_ns = sk["global_ns"]
    row.kv_skew_stage_ns = sk["per_stage"]
    if sk["short_stages"]:
        warn(f"{tag}: decode stage(s) {sk['short_stages']} declare >=2 ranks but "
             f"<2 received KV; their intra-stage skew is omitted.")

    # decode-side measures, from utils.measures (one definition each):
    # the per-(stage, layer) TP-group shard skew (fig 04), the per-rank
    # cumulative KV arrival (fig 09) and the first-pass KV stall (fig 02). The
    # rank span / layer delta kv_skew_stats also returns feed buffer_sweep's
    # figures only; unused here. decode_ar_stats is kept for summary.csv: its
    # entry skews are µs-scale on this data, so they get no figure.
    skew_scal, _, _ = kv_skew_stats(kv_arr)
    for k, v in skew_scal.items():
        setattr(row, k, v)
    if row.kv_tp_skew_n == 0:
        warn(f"{tag}: no (stage, layer) KV group has >=2 shard arrivals; the "
             f"TP-group skew figure (04) will be empty for this run.")
    row.kv_rank_series = kv_rank_series(kv_arr, placement)
    # the (stage, layer) skew population, kept whole for figure 04: on this data
    # it is heavy-tailed at every buffer, so the min/mean/p99 scalars beside it
    # cannot describe it.
    row.kv_layer_skew = kv_layer_skew(kv_arr)
    if len(row.kv_layer_skew) and row.kv_layer_skew["signed_ns"].notna().any():
        row.kv_shard_bias_ns = float(row.kv_layer_skew["signed_ns"].median())
    row.dec_ar = decode_ar_stats(adf)
    if not row.dec_ar:
        warn(f"{tag}: no decode TP all-reduce in the ASTRA stats; the decode "
             f"first-all-reduce columns will be NaN for this run.")
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
             f"measured first-pass stall is unavailable (figure 02 falls back "
             f"to the KV envelope).")
    if pd.notna(row.tok2_ns) and pd.notna(row.ttft_ns):
        row.tok2_after_tok1_ns = row.tok2_ns - row.ttft_ns
    if pd.isna(row.tok2_ns):
        warn(f"{tag}: no DECFB send in the ASTRA stats; the decode KV-stall "
             f"figure (02) will be empty for this run.")

    # the cuts of the first-token -> second-token interval (figures 12 and 14),
    # each a difference of two instants already measured above.
    if pd.notna(row.dec_start_ns) and pd.notna(row.ttft_ns):
        row.dec_start_after_ttft_ns = row.dec_start_ns - row.ttft_ns
    if pd.notna(row.kv_gate_ns) and pd.notna(row.ttft_ns):
        row.kv_gate_after_ttft_ns = row.kv_gate_ns - row.ttft_ns
    if pd.notna(row.kv_gate_ns) and pd.notna(row.dec_start_ns):
        row.kv_tail_after_dec_start_ns = row.kv_gate_ns - row.dec_start_ns

    ppr = pp.measure(adf)
    row.pp_available = ppr.available
    row.pp_skew_ns = ppr.skew_ns
    row.pp_skew_mean_ns = ppr.skew_mean_ns
    if not ppr.available:
        warn(f"{tag}: no inter-stage PP-prefill flow (expected for these "
             f"placements); PP skew recorded as NaN/0.")

    # per-switch fabric census
    row.pfc_per_switch, row.total_pause_frames = pfc.switch_pause_census()
    row.pause_intervals = pfc.switch_pause_intervals(run_end)
    for sw, (ts, ys) in qlen.switch_series.items():
        if len(ts) == 0:
            continue
        row.qseries[sw] = downsample_max(ts, ys, 2000)
        row.qswitch_peak[sw] = float(qlen.switch_total_max.get(sw, max(ys)))
        row.qswitch_mean[sw] = float(np.mean(ys))
    return row


# --------------------------------------------------------------------------- #
# Busiest-switch selection (per topology, fixed across its buffers)
# --------------------------------------------------------------------------- #
def busy_switches(rows: list[Row], k_per_metric: int, cap: int) -> list[int]:
    """The switches worth showing for one topology: the union of the top
    `k_per_metric` by PFC PAUSE frames and the top `k_per_metric` by peak buffer
    occupancy (the user's 'most PFC AND/OR most buffer' -- a union, not a single
    blended score, so a switch that dominates EITHER axis is kept). Occupancy
    order first, so the incast point (the deepest-queue core switch) heads the
    list; capped at `cap` for legible grids. Aggregated as the MAX over the
    topology's buffers so the set is fixed across the sweep, not re-picked per
    column."""
    agg_pfc: dict[int, float] = defaultdict(float)
    agg_occ: dict[int, float] = defaultdict(float)
    for r in rows:
        for sw, c in r.pfc_per_switch.items():
            agg_pfc[sw] = max(agg_pfc[sw], c)
        for sw, v in r.qswitch_peak.items():
            agg_occ[sw] = max(agg_occ[sw], v)
    top_occ = [sw for sw, v in sorted(agg_occ.items(), key=lambda x: -x[1])
               if v > 0][:k_per_metric]
    top_pfc = [sw for sw, v in sorted(agg_pfc.items(), key=lambda x: -x[1])
               if v > 0][:k_per_metric]
    ordered: list[int] = []
    for sw in top_occ + top_pfc:
        if sw not in ordered:
            ordered.append(sw)
    return ordered[:cap]


# --------------------------------------------------------------------------- #
# Per-topology analysis
# --------------------------------------------------------------------------- #
@dataclass
class Level:
    level: str
    degree: int          # the FAN-IN (tp_p/tp_d), not the prefill TP width
    rows: list           # list[Row], sorted by buffer
    busy: list           # list[int] switch ids
    label: str           # "T3 · fan-in 2"
    links: list          # list[str], the level's fixed KV-crossed link labels
    summary: object = None   # pd.DataFrame: this level's flat rows + its knees


def canonical_links(p: incast.IncastPaths, tags: list[str],
                    placement: Placement, top_links: int) -> list[str]:
    """The KV-crossed links this LEVEL reports, and the order it reports them in,
    decided once from its SMALLEST-buffer run (the most congested, so the least
    likely to rank two links by a near-tie).

    The link SET is a property of the topology and the placement, so it does not
    vary across the buffers of one level; only its congestion RANKING can, and
    letting it would turn every per-link curve into a comparison between
    different links. buffer_sweep fixes the set for the same reason; here it also
    fixes which link the bn_* columns describe."""
    ref = min(tags, key=lambda t: BUFFER_AXIS.value(t))
    topo = parse_topology(p.topology(ref))
    cfg = parse_ns3_config(p.config(ref))
    raw = ns3.read_fct(p.ns3_run(ref) / "fct.txt")
    need(raw is not None and len(raw),
         f"{ref}: fct.txt has no parsable rows; cannot fix the link set.")
    f = flowlib.annotate(raw, topo, placement, cfg.payload)
    kv = f[f["flow_class"] == "kv"]
    need(len(kv), f"{ref}: no KV flow after classification -- the prefill/decode "
                  f"split does not match this topology's traffic.")
    qlen = ns3.read_qlen(p.ns3_run(ref) / "qlen.txt", series=False)
    need(qlen is not None and qlen.port_max, f"{ref}: qlen.txt has no samples.")
    cands = flowlib.candidate_links(topo, qlen.port_max, kv)
    need(cands, f"{ref}: no link is crossed by any KV flow -- classification or "
                f"topology is wrong.")
    return [str(l) for l in cands[:top_links]]


def analyse_level(level: str, root: Path, out_workload: str, config_sweep: str,
                  k_switches: int, top_links: int) -> Level | None:
    p = incast.IncastPaths(level=level, out_workload=out_workload,
                           config_sweep=config_sweep, root=root)
    missing_roots = p.missing_roots()
    # the config root is an INPUT: without it nothing can be analysed -- Abort.
    # Output roots merely mean the level has not produced anything yet -- skip.
    need(not any(m.startswith("config_root") for m in missing_roots),
         f"{level}: " + "; ".join(m for m in missing_roots
                                  if m.startswith("config_root")))
    if missing_roots:
        warn(f"{level}: skipped, output root(s) missing:\n    "
             + "\n    ".join(missing_roots))
        return None
    miss_cfg = p.missing_configs()
    need(not miss_cfg,
         f"{level}: config input(s) missing -- fix the configs (or prune the "
         f"stale ns-3 output):\n    " + "\n    ".join(miss_cfg))
    tags, skipped = p.usable_tags()
    for s in skipped:
        warn(f"{level}: {s} (run not finished yet?) -- skipped.")
    if not tags:
        warn(f"{level}: no run has all its outputs on disk yet; skipped.")
        return None

    placement = recover_placement(p, tags)
    tp_p, tp_d = incast.prefill_tp(placement), incast.decode_tp(placement)
    degree = incast.fan_in(placement)
    links = canonical_links(p, tags, placement, top_links)
    print(f"\n===== {level}  (prefill TP{tp_p} -> decode TP{tp_d}: "
          f"FAN-IN {degree}{' — the no-incast control' if degree <= 1 else ''}) "
          f"=====")
    print(f"  placement {roles.spec_of(placement)}")
    print(f"  buffers   {[BUFFER_AXIS.value(t) for t in tags]}")
    print(f"  links     {links[0]} (bottleneck)"
          + (f" + {len(links) - 1} more: {', '.join(links[1:])}"
             if len(links) > 1 else ""))

    rows = []
    for tag in tags:
        try:
            rows.append(analyse(tag, p, placement, links))
        except Abort as e:
            warn(f"{level}: run {tag} dropped -- {e}")
    if not rows:
        warn(f"{level}: every run failed to analyse; skipped.")
        return None
    rows.sort(key=lambda r: r.buffer_mb)
    busy = busy_switches(rows, k_switches, cap=2 * k_switches)

    for r in rows:
        flag = "" if r.split_ok else "  ! split check FAILED"
        if r.lossy:
            flag += f"  ** LOSS: {r.dropped_packets:.0f} pkt ({r.drop_rate*100:.2g}%) **"
        elif not r.loss_captured:
            flag += "  (loss unknown: no drops.txt)"
        print(f"  + buf{r.buffer_mb:<4g} bn={r.bottleneck:<8} "
              f"floor={r.kv_floor_ns*MS:6.1f}ms  idle={r.kv_idle_ns*MS:5.1f}ms  "
              f"tok2-tok1={r.tok2_after_tok1_ns*MS:6.1f}ms  "
              f"pfc={r.total_pause_frames:.0f}{flag}")
    print(f"  busiest switches (top {k_switches} by PFC ∪ by occupancy): {busy}")

    # This level's summary frame, assembled here rather than in main because the
    # knees are read PER TOPOLOGY: each level has its own buffer axis, and a knee
    # of the pooled three-topology table would be meaningless. They are written
    # back as constant columns of the level's rows, so the concatenated
    # summary.csv carries one knee value per topology.
    sl = (pd.DataFrame([r.flat() for r in rows])
          .sort_values("buffer_mb").reset_index(drop=True))
    sl = decode_worst_stage(sl)
    # ms twins of the three decode-stall columns decode_worst_stage produces in
    # ns, so the printed table reads in one unit (the ns columns stay: the
    # cross-tool readers key on those names).
    for col in ("decode_kv_stall", "kv_tail_after_dec_start", "dec_kv_lateness",
                "dec_kv_block"):
        if f"{col}_ns" in sl.columns:
            sl[f"{col}_ms"] = sl[f"{col}_ns"] * MS
    for k, v in knee_scalars(sl, PAUSE_KNEE_COL, SATURATION_METRICS).items():
        sl[k] = v
    print("  knees (MiB): " + ", ".join(
        f"{name}={f'{sl[col].iloc[0]:g}' if pd.notna(sl[col].iloc[0]) else '—'}"
        for col, (_c, name) in KNEE_STYLE.items()))
    sat = sl["knee_saturation_mb"].iloc[0]
    if pd.notna(sat):
        dup = sorted(sl.loc[sl["buffer_mb"] > sat, "buffer_mb"])
        if dup:
            print(f"  note: {', '.join(f'{b:g}' for b in dup)} MiB are "
                  f"indistinguishable from {sat:g} MiB — not independent points.")
    return Level(level=level, degree=degree, rows=rows, busy=busy,
                 label=f"{level} · fan-in {degree}", links=links, summary=sl)


def recover_placement(p: incast.IncastPaths, tags: list[str]) -> Placement:
    """The level's rank->role map, recovered from its ASTRA trace (prefill TP
    width differs per topology, so it is not a single CLI placement). Tries each
    run until one trace is readable."""
    last_err = None
    for tag in tags:
        adir = p.astra_run(tag)
        if not adir.is_dir():
            continue
        try:
            return roles.from_astra(adir)
        except Exception as e:                                  # noqa: BLE001
            last_err = e
    raise Abort(f"{p.level}: no readable ASTRA trace to recover the placement "
                f"from. Last error: {last_err}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# Per-level colour overrides: viridis maps the last level to its pale-yellow
# endpoint, which is nearly invisible on a white background. Override those
# levels with readable, mutually distinct hues (kept CVD-safe against the
# viridis purple/teal of the other levels, and away from the reserved LOSS_RED).
_LEVEL_COLOR_OVERRIDE = {
    "T4": "#e8590c",          # dark orange -- readable, distinct from T2.1/T3
}


def _level_colors(levels: list[Level]) -> dict[str, tuple]:
    cmap = plt.get_cmap("viridis")
    n = max(len(levels) - 1, 1)
    return {lv.level: _LEVEL_COLOR_OVERRIDE.get(lv.level, cmap(i / n))
            for i, lv in enumerate(levels)}


def fig_incast_cost(levels: list[Level], s: pd.DataFrame, outdir: Path,
                    written: list[Path]) -> None:
    """01 -- the receiving link's idle time during the KV handover, SPLIT into
    the part the fan-in is responsible for and the part it is not.

    The handover moves a fixed number of bytes over a link of fixed rate, so its
    duration measures nothing: it is bytes/rate at every degree and every
    buffer. What congestion produces is the time the receiving link spends NOT
    SENDING inside that window, window - floor. But that idle has two causes and
    only one of them is the incast:

      STARVED  every sender feeding this rank is busy putting something else on
               its own NIC. On a pipeline-parallel prefill that is the
               activation handed to the next stage, and the senders of a stage
               emit it simultaneously, so the receiver loses 100% of its supply
               for the duration -- ~6.8 ms here, the same in every topology
               because the activation does not depend on the fan-in.
      INCAST   what is left: senders that had data and the wire, and still did
               not fill the link. Rate-control transients and the synchronised
               back-off of the converging flows.

    The figure asks the three questions in order, one per panel, because the
    answer to the first is meaningless without the other two.

    A -- HOW LONG DID THE HANDOVER TAKE, AND WHY? The bar is the transfer, laid
    out in absolute ms: the floor first (the part physics owes, identical at
    every fan-in), then the idle, split into its two causes. The dashed rule at
    the floor is the shortest the transfer could possibly be, so the coral tail
    beyond it IS the cost, read against the right yardstick instead of floating
    on its own axis. The label at the end gives the same fact as a utilisation.
    The bars are MEANS over the decode ranks -- only the ranks of the first
    decode stage pay the starvation, so a "worst" pick silently selects a stage
    and stops being comparable across topologies -- and the caret marks the
    WORST receiver's window, so the spread between ranks is visible rather than
    averaged away (it is what summary.csv and the report quote).

    B -- DOES MORE BUFFER FIX IT? The incast term against the swept buffer. Flat
    lines are a result, not an empty panel: they say the queues never reached
    the smaller buffer's limit and the cost is a property of the rate control,
    not of the switch memory.

    C -- IS IT QUEUEING AT ALL? The incast term next to the delay the deepest
    queue could possibly explain (peak occupancy / link rate), in the SAME unit,
    on a log axis. A fabric that is congested in the ordinary sense pays its
    delay in the queues, so the two bars stay comparable; when the coral bar
    towers over the grey one the link is simply not being filled -- the senders
    are backing off, and no amount of buffer is going to help. The starvation is
    deliberately NOT in this comparison: it is a sender-side outage and was
    never going to be in a switch queue.

    The two idle causes, defined once:
      STARVED  every sender feeding this rank is busy putting something else on
               its own NIC. On a pipeline-parallel prefill that is the
               activation handed to the next stage, and the senders of a stage
               emit it simultaneously, so the receiver loses 100% of its supply
               for the duration -- the same in every topology, because the
               activation does not depend on the fan-in.
      INCAST   what is left: senders that had data and the wire, and still did
               not fill the link. Rate-control transients and the synchronised
               back-off of the converging flows.

    Fan-in 1 is the CONTROL -- same bytes, same link, one sender per layer --
    and its incast term is the floor of the phenomenon."""
    if ("kv_incast_mean_ms" not in s.columns
            or not s["kv_incast_mean_ms"].notna().any()):
        return
    colors = _level_colors(levels)
    big = (s[s["buffer_mb"] == s["buffer_mb"].max()]
           .sort_values("incast_degree"))
    if big.empty:
        return
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(17.0, 5.0), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})
    ref = f"{s['buffer_mb'].max():g} MiB"

    # ---- A: where the handover time went -------------------------------- #
    floor = big["kv_floor_ms"].to_numpy(dtype=float)
    starved = big["kv_starved_mean_ms"].fillna(0).to_numpy(dtype=float)
    inc = big["kv_incast_mean_ms"].fillna(0).to_numpy(dtype=float)
    y = np.arange(len(big), dtype=float)
    axA.barh(y, floor, height=0.55, color="#d4d7de",
             label="floor: bytes / link rate (unavoidable)")
    axA.barh(y, starved, left=floor, height=0.55, color=MUTED,
             label="starved: every sender busy with the PP activation")
    axA.barh(y, inc, left=floor + starved, height=0.55, color=CORAL,
             label="incast: senders had data and did not fill the link")
    # The per-rank SPREAD, as a whisker over the mean bar. Not "the worst
    # receiver": the worst by incast term can have a SHORTER window than the
    # mean (on a starved stage the starvation dominates the idle), and a marker
    # that lands left of its own bar reads as an error. min-max over the decode
    # ranks says what it means and always brackets the bar.
    rank_cols = [c for c in big.columns
                 if re.fullmatch(r"kv_idle_rank\d+_ms", c)]
    hi_end = (floor + starved + inc).copy()
    if rank_cols:
        for i, (_, r) in enumerate(big.iterrows()):
            v = r[rank_cols].dropna().astype(float)
            if len(v) < 2:
                continue
            a_, b_ = floor[i] + float(v.min()), floor[i] + float(v.max())
            axA.plot([a_, b_], [y[i], y[i]], color="#111111", lw=1.3, zorder=5,
                     solid_capstyle="butt",
                     label=("spread across the decode ranks (min–max)"
                            if i == 0 else None))
            axA.plot([a_, b_], [y[i], y[i]], marker="|", ms=9, mew=1.5,
                     color="#111111", ls="none", zorder=5)
            hi_end[i] = max(hi_end[i], b_)
    if len(set(np.round(floor, 3))) == 1:
        axA.axvline(floor[0], color="#555555", ls="--", lw=1.2)
        axA.annotate(f"floor {floor[0]:.1f} ms", (floor[0], -0.62),
                     fontsize=8, ha="center", va="bottom", color="#555555")
    for i in range(len(big)):
        tot = floor[i] + starved[i] + inc[i]
        axA.annotate(f"{100 * floor[i] / tot:.1f}% busy   +{starved[i] + inc[i]:.1f} ms idle",
                     (hi_end[i] * 1.015, y[i]), fontsize=8.5,
                     va="center", color="0.30")
    lab = {L.level: L.label for L in levels}
    axA.set_yticks(y)
    axA.set_yticklabels([lab.get(lv, lv) + ("\n(control)" if d <= 1 else "")
                         for lv, d in zip(big["level"], big["incast_degree"])],
                        fontsize=8.5)
    axA.invert_yaxis()
    axA.set_ylim(len(big) - 0.4, -0.95)
    axA.set_xlim(0, float(hi_end.max()) * 1.42)
    axA.set_xlabel("Duration of the KV handover (ms)")
    axA.set_title(f"A · Where the handover time went  (@ {ref})", fontsize=10)
    axA.grid(True, axis="x", alpha=0.3)
    # legends go BELOW their panel: every inside corner of A and B holds either
    # a bar, a line or the annotation that explains it.
    axA.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
               ncol=2, frameon=False)

    # ---- B: does the buffer change it? ---------------------------------- #
    anyl = anyu = False
    for lv in levels:
        g = (s[s["level"] == lv.level].dropna(subset=["kv_incast_mean_ms"])
             .sort_values("buffer_mb"))
        if g.empty:
            continue
        axB.plot(g["buffer_mb"], g["kv_incast_mean_ms"], "o-",
                 color=colors[lv.level],
                 label=f"{lv.label}" + ("  (control)" if lv.degree <= 1 else ""))
        ml, mu = mark_lossy(axB, g, "buffer_mb", "kv_incast_mean_ms")
        anyl |= ml
        anyu |= mu
    logx_pow2(axB, s, "buffer_mb", "Per-switch buffer (MiB)")
    axB.set_ylabel("Incast term, mean over the decode ranks (ms)")
    axB.set_ylim(bottom=0)
    # A flat set of lines is the finding, not an empty panel -- say so.
    spread = s.groupby("level", observed=True)["kv_incast_mean_ms"].apply(
        lambda v: (v.max() - v.min()) if v.notna().any() else NAN)
    if spread.notna().any() and float(spread.max()) < 0.05:
        axB.annotate("identical at every buffer:\nthe cost is the rate control,\n"
                     "not the switch memory", (0.5, 0.46),
                     xycoords="axes fraction", ha="center", fontsize=8.5,
                     color="0.35", style="italic")
    axB.set_title("B · Does more buffer fix it?", fontsize=10)
    axB.grid(True, alpha=0.3, which="both")
    h, _ = axB.get_legend_handles_labels()
    axB.legend(handles=h + loss_proxies(anyl, anyu), fontsize=7.5,
               loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
               frameon=False)

    # ---- C: is it queueing at all? -------------------------------------- #
    # The deepest queue, expressed as the delay it could account for at the same
    # link rate -- the only way to put "MiB of buffer" and "ms of idle" on one
    # axis honestly.
    qdelay = np.full(len(big), NAN)
    if {"bn_qpeak_mb", "kv_link_rate_gbps"} <= set(big.columns):
        qdelay = (big["bn_qpeak_mb"].to_numpy(dtype=float) * 2**20 * 8e3
                  / (big["kv_link_rate_gbps"].to_numpy(dtype=float) * 1e9))
    x = np.arange(len(big), dtype=float)
    w = 0.36
    # The INCAST term, not the whole idle: the starvation is a sender-side
    # outage and was never going to be in a switch queue, so including it would
    # inflate the ratio exactly where the answer is least interesting.
    lo = min([v for v in np.concatenate([inc, qdelay]) if v > 0] or [1e-3])
    axC.bar(x - w / 2, np.maximum(inc, lo * 0.35), width=w, color=CORAL,
            label="incast term left on the receiving link")
    axC.bar(x + w / 2, np.maximum(np.nan_to_num(qdelay), lo * 0.35), width=w,
            color="#9aa0a6", label="delay the deepest queue could explain")
    for i in range(len(big)):
        if inc[i] > 0 and pd.notna(qdelay[i]) and qdelay[i] > 0:
            axC.annotate(f"×{inc[i] / qdelay[i]:.0f}",
                         (x[i], max(inc[i], qdelay[i]) * 1.5),
                         ha="center", fontsize=8.5, fontweight="bold",
                         color=CORAL)
    axC.set_yscale("log")
    axC.set_ylim(lo * 0.1, float(np.nanmax(np.concatenate([inc, qdelay]))) * 12)
    axC.set_xticks(x)
    # the PAUSE census rides in the tick label: inside the panel it would sit
    # under a bar (log axis: the bars run down to the floor).
    axC.set_xticklabels(
        [f"fan-in {int(d)}\n{pf:,.0f} PAUSE" if pd.notna(pf) else f"fan-in {int(d)}"
         for d, pf in zip(big["incast_degree"], big["total_pause_frames"])],
        fontsize=8.5)
    axC.set_ylabel("Time (ms, log)")
    axC.set_title(f"C · Is it queueing?  (@ {ref})", fontsize=10)
    axC.grid(True, axis="y", alpha=0.3, which="both")
    axC.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("What the KV handover cost the receiving link, what the buffer "
                 "does about it, and whether the delay is in the queues", y=1.02)
    save_fig(fig, outdir, "01_incast_cost.png", written)



def fig_fabric_cost(levels: list[Level], s: pd.DataFrame, outdir: Path,
                    written: list[Path]) -> None:
    """03 -- the fabric's state vs the buffer: PFC PAUSE frames (symlog, so a
    pause-free run at the right-hand end still shows), dropped packets
    ('Headroom full', a lossless-fabric violation) and the peak queue at the
    bottleneck. These move by orders of magnitude where figure 01 barely does.

    Read together with figure 01 this is the finding of the sweep: buffering
    decides the fabric's FAILURE MODE -- loss, backpressure, or standing queue --
    and barely touches the time the application waits. The loss panel plots only
    runs with a captured drops.txt, so a flat line at zero there is a certified
    lossless sweep rather than an unmeasured one."""
    panels = [("total_pause_frames", "PFC PAUSE frames, whole fabric (symlog)",
               "Backpressure", True),
              ("dropped_packets", "Dropped packets ('Headroom full')",
               "Loss", False),
              ("bn_qpeak_mb", "Peak queue at the bottleneck (MiB)",
               "Standing queue", False)]
    panels = [p for p in panels if p[0] in s.columns and s[p[0]].notna().any()]
    if not panels:
        return
    colors = _level_colors(levels)
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(max(5.0 * len(panels), 6), 4.8))
    anyl = anyu = False
    for j, (col, ylab, title, symlog) in enumerate(panels):
        a = axes[0][j]
        for lv in levels:
            g = (s[s["level"] == lv.level].dropna(subset=[col])
                 .sort_values("buffer_mb"))
            if g.empty:
                continue
            a.plot(g["buffer_mb"], g[col], "o-", color=colors[lv.level],
                   label=lv.label)
            ml, mu = mark_lossy(a, g, "buffer_mb", col)
            if j == 0:
                anyl |= ml
                anyu |= mu
            if col == "dropped_packets":
                for _, rr in g[g[col] > 0].iterrows():
                    a.annotate(f"{int(rr[col])}", (rr["buffer_mb"], rr[col]),
                               textcoords="offset points", xytext=(6, 6),
                               fontsize=8, color=LOSS_RED, fontweight="bold")
        if symlog:
            a.set_yscale("symlog", linthresh=10)
        else:
            a.set_ylim(bottom=0)
        logx_pow2(a, s, "buffer_mb", "Per-switch buffer (MiB)")
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=10)
        a.grid(True, alpha=0.3, which="both")
        if j == 0:
            h, _ = a.get_legend_handles_labels()
            a.legend(handles=h + loss_proxies(anyl, anyu), fontsize=8,
                     title="topology")
    fig.suptitle("PFC frames, dropped packets and peak queue vs per-switch "
                 "buffer", y=1.02)
    save_fig(fig, outdir, "03_fabric_cost.png", written)


def fig_kv_cumulative(lv: Level, outdir: Path, written: list[Path]) -> None:
    """09 -- (per topology) cumulative KV bytes arrived per decode rank over
    time, one panel per buffer (buffer_sweep's 02). The horizontal spread
    between the ranks' curves IS the KV arrival skew; a staircase with flat
    stretches IS a PFC stall. Dotted vline = the KV gate (last arrival, decode
    barrier)."""
    runs = [r for r in lv.rows if r.kv_rank_series]
    if not runs:
        return
    ranks = sorted({d for r in runs for d in r.kv_rank_series})
    cmap = plt.get_cmap("tab10")
    ncols = len(runs)
    fig, axes = plt.subplots(1, ncols, figsize=(max(3.0 * ncols, 6), 4.6),
                             sharey=True, squeeze=False)
    for j, r in enumerate(runs):
        a = axes[0][j]
        for i, d in enumerate(ranks):
            if d not in r.kv_rank_series:
                continue
            t, cum = r.kv_rank_series[d]
            total = cum[-1] if len(cum) else 1.0
            a.step(t * MS, 100 * cum / total, where="post",
                   color=cmap(i % 10), label=f"rank {d}")
        if pd.notna(r.kv_gate_ns):
            a.axvline(r.kv_gate_ns * MS, color="k", linestyle=":", alpha=0.5)
        tc = (LOSS_RED if r.lossy
              else MUTED if not r.loss_captured else "black")
        a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9, color=tc)
        a.set_xlabel("Time (ms)", fontsize=8)
        a.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("KV arrived (% of total)")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{lv.label}: cumulative KV arrival per decode rank "
                 f"(red title = dropped packets)", y=1.02)
    save_fig(fig, outdir, f"09_{lv.level}_kv_cumulative_arrival.png", written)


def fig_queue_fill(lv: Level, outdir: Path, written: list[Path]) -> None:
    """08 -- (per topology) how the BUSIEST switches' buffers fill over time,
    rows = busy switch, cols = buffer, with PFC PAUSE spans shaded as a top
    ribbon. The pared-down descendant of buffer_sweep's per-switch grid: same
    picture, but only the switches that actually matter here."""
    if not lv.busy:
        return
    runs = lv.rows
    nrows, ncols = len(lv.busy), len(runs)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                             sharey="row",
                             figsize=(max(2.1 * ncols + 1.8, 6),
                                      max(1.7 * nrows + 1.0, 4)))
    cmap = plt.get_cmap("viridis")
    bufs = [r.buffer_mb for r in runs]
    cnorm = (matplotlib.colors.LogNorm(vmin=min(bufs), vmax=max(bufs))
             if len(set(bufs)) > 1 else None)
    for i, sw in enumerate(lv.busy):
        for j, r in enumerate(runs):
            a = axes[i][j]
            if sw in r.qseries:
                ts, ys = r.qseries[sw]
                col = cmap(cnorm(r.buffer_mb)) if cnorm else BLUE
                a.fill_between(np.asarray(ts) * MS, np.asarray(ys) / 1e3,
                               color=col, alpha=0.85, lw=0)
                a.plot(np.asarray(ts) * MS, np.asarray(ys) / 1e3, color="#222222",
                       lw=0.5, alpha=0.6)
            for s0, e0 in r.pause_intervals.get(sw, []):
                a.axvspan(s0 * MS, e0 * MS, ymin=0.88, ymax=1.0,
                          transform=a.get_xaxis_transform(), color=CORAL,
                          alpha=0.9, lw=0)
            a.grid(True, alpha=0.2)
            if i == 0:
                tc = (LOSS_RED if r.lossy
                      else MUTED if not r.loss_captured else "black")
                a.set_title(f"{r.buffer_mb:g} MiB", fontsize=9, color=tc)
            if j == 0:
                a.set_ylabel(f"sw {sw}\n(kB)", fontsize=8)
            if i == nrows - 1:
                a.set_xlabel("Time (ms)", fontsize=8)
                a.locator_params(axis="x", nbins=4)
    lossy = [f"{r.buffer_mb:g}" for r in runs if r.lossy]
    note = f"  [red title = dropped packets: {', '.join(lossy)} MiB]" if lossy else ""
    fig.suptitle(f"{lv.label}: busiest switches' buffer fill over time "
                 f"(PFC PAUSE shaded){note}", y=1.01)
    save_fig(fig, outdir, f"08_{lv.level}_queue_fill_busy_switches.png", written)


def fig_first_to_second(levels: list[Level], outdir: Path,
                        written: list[Path]) -> None:
    """02 -- where the interval between the FIRST and the SECOND token actually
    goes, as a waterfall (buffer_sweep's 04), one panel per topology, one bar per
    buffer, time measured from the first token:

      BLUE   TTFT -> decode start: the first-token handoff still in flight,
             queued behind the KV bulk. The decode does not exist yet.
      GREEN  decode start -> token 2: the first pass, wherever it is making
             progress.
      CORAL  the intervals INSIDE that pass in which a decode rank sat idle and
             a KV arrival is what let it resume (first_pass_stall's union over
             the ranks). VIOLET: idle no arrival explains.

    The coral used to be the whole [decode start, KV gate] stretch, on the
    argument that the pass finishes within a fraction of an ITL of the gate so
    the rest of that stretch must be waiting. It does not follow, and on
    buffer_sweep's workloads it is false: the decode consumes its KV layer by
    layer, and a run whose bar was 4 ms of "stall" turns out to run the whole
    pass back to back while the tail of the transfer -- KV it has not reached
    yet -- is still landing. What is drawn now is the idle itself, and its
    total matches the pass's excess over the steady ITL within 3% on these
    levels (T3: 36.2 against 37.3 ms) -- see first_pass_stall on why the
    measured value is the smaller of the two.

    THE AXIS IS A SHARE, NOT A DURATION. Every bar is normalised to its own
    tok1->tok2 interval, so the three segments read as the COMPOSITION of that
    interval and the topologies are comparable at a glance -- the ms figure had
    the composition encoded as the difference between two bar lengths, which is
    exactly the reading a stacked bar is bad at. What normalising away would
    lose is that the interval itself grows with the fan-in, so it is written
    back on each bar: the absolute total and its dilation over the steady ITL,
    at the right-hand end.

    The dotted rule per bar is the steady ITL as a SHARE of that interval (what
    tok2 would cost if the KV were already local, over what it cost): the
    sliver to its left is the second token, everything to its right is handover
    exposure. Its reciprocal is the x ITL dilation printed beside the bar, so
    the two readings check each other.

    The fabric-wide PAUSE count is printed at the end of each row, so the trade
    the sweep is about -- backpressure collapsing while the coral segment does
    or does not grow -- is read off one picture instead of correlated across
    figures 04 and 08."""
    usable = [(lv, [r for r in lv.rows
                    if all(pd.notna(v) for v in (r.ttft_ns, r.dec_start_ns,
                                                 r.kv_gate_ns, r.tok2_ns))
                    and r.tok2_ns > r.ttft_ns])
              for lv in levels]
    usable = [(lv, runs) for lv, runs in usable if runs]
    if not usable:
        return
    n = len(usable)
    rows_max = max(len(runs) for _, runs in usable)
    fig, axes = plt.subplots(1, n, squeeze=False, sharex=True,
                             figsize=(max(6.5 * n, 8), 0.62 * rows_max + 2.6))
    for j, (lv, runs) in enumerate(usable):
        a = axes[0][j]
        for i, r in enumerate(runs):
            t0 = r.ttft_ns
            t2 = (r.tok2_ns - t0) * MS               # the interval = 100%
            ds = 100 * (r.dec_start_ns - t0) * MS / t2
            a.barh(i, ds, left=0, height=0.62, color=BLUE)
            a.barh(i, max(100.0 - ds, 0.0), left=ds, height=0.62,
                   color=GREEN, **({} if r.dec_idle else
                                   dict(alpha=0.45, hatch="//")))
            for spans, colour in ((r.dec_idle.get("idle_spans", ()), VIOLET),
                                  (r.dec_idle.get("kv_blocked_spans", ()), CORAL)):
                for lo, hi in spans:
                    a.barh(i, 100 * (hi - lo) * MS / t2,
                           left=100 * (lo - t0) * MS / t2, height=0.62,
                           color=colour)
            # the counterfactual, as a share of THIS bar: one ITL of the
            # interval it actually took.
            if pd.notna(r.itl_steady_ns) and r.itl_steady_ns > 0:
                share = 100 * r.itl_steady_ns * MS / t2
                a.plot([share, share], [i - 0.34, i + 0.34], color="#444444",
                       ls=":", lw=1.3, zorder=4)
            note = f"{t2:.0f} ms"
            if pd.notna(r.itl_steady_ns) and r.itl_steady_ns > 0:
                note += f" · {t2 / (r.itl_steady_ns * MS):.0f}× ITL"
            if pd.notna(r.total_pause_frames):
                note += f" · {r.total_pause_frames:,.0f} PAUSE"
            a.text(102.5, i, note, va="center", fontsize=8,
                   color=LOSS_RED if r.lossy else MUTED)
        a.set_yticks(range(len(runs)))
        a.set_yticklabels([f"{r.buffer_mb:g} MiB" for r in runs], fontsize=8)
        a.invert_yaxis()
        a.set_title(lv.label, fontsize=10)
        a.set_xlabel("% of the first-token → second-token interval")
        a.set_xticks([0, 25, 50, 75, 100])
        a.grid(True, axis="x", alpha=0.3)
    # room on the right of every panel for the total / dilation / PAUSE label
    axes[0][0].set_xlim(0, 162)
    fig.legend(handles=[Patch(color=BLUE, label="handoff in flight"),
                        Patch(color=GREEN, label="decode progressing"),
                        Patch(color=CORAL, label="idle, resumed by a KV arrival"),
                        Patch(color=VIOLET, label="idle, other"),
                        plt.Line2D([], [], color="#444444", ls=":", lw=1.3,
                                   label="one steady ITL — tok2 if the KV were free")],
               fontsize=9, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    save_fig(fig, outdir, "02_first_token_to_second.png", written)


def fig_stage_handover(levels: list[Level], outdir: Path,
                       written: list[Path]) -> None:
    """12 -- WHEN each decode stage's KV handover completes, and why the stages
    do not complete together although they carry the SAME number of bytes.

    Every decode rank of these runs receives exactly the same payload (the KV
    is evenly sharded and each receiver owns a dedicated port), so the
    serialisation floor bytes/rate is one number for all of them. Their
    completion times are nevertheless tens of milliseconds apart, and the two
    panels separate the two candidate explanations:

      LEFT (absolute clock, at the largest buffer). One bar per receiving rank,
      grouped by decode stage: it starts at the FIRST SEND feeding that rank
      and ends at its LAST ARRIVAL -- that end IS the stage's KV gate, the
      instant its first decode pass can complete. The bar is cut into the floor
      (bytes/rate), the starvation and the incast term, exactly as figure 01
      cuts the worst receiver. What the panel shows is that the stage-to-stage
      gap is almost entirely a SHIFT of the bar, not a stretch of it: a later
      stage's layers are produced later by the pipelined prefill, so its
      transfer STARTS later. The dotted rule per group is that topology's first
      token; the handover ending to the right of it is the exposed part.

      RIGHT (the stretch, per stage, over the whole buffer sweep). The same
      window as a percentage of the floor, mean over the ranks of the stage --
      100% would be a transfer that never leaves the wire idle. This is the
      'does the same size take longer' question, answered per stage and per
      buffer: on this data the answer is a few percent, one to two orders of
      magnitude less than the shift the left panel shows.

    Read together: the completion time of a stage is set by the prefill's
    production order first and by the fabric second."""
    usable = [lv for lv in levels
              if any(r.kv_rank_stat and r.kv_stage_ranks for r in lv.rows)]
    if not usable:
        return
    colors = _level_colors(levels)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(16.0, 5.6), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ---- A: the absolute clock, at the largest buffer -------------------- #
    y = 0.0
    yticks, ylabels, tok1, xmax = [], [], [], 0.0
    for lv in usable:
        rows = [r for r in lv.rows if r.kv_rank_stat and r.kv_stage_ranks]
        r = max(rows, key=lambda q: q.buffer_mb)
        first_row_of_level = y
        for si in sorted(r.kv_stage_ranks):
            for rk in r.kv_stage_ranks[si]:
                m = r.kv_rank_stat.get(rk)
                if not m or pd.isna(m.get("start_ns")):
                    continue
                x0 = m["start_ns"] * MS
                fl = m["floor_ns"] * MS
                st = (m["starved_ns"] if pd.notna(m["starved_ns"]) else 0.0) * MS
                ic = (m["incast_ns"] if pd.notna(m["incast_ns"]) else 0.0) * MS
                axA.barh(y, fl, left=x0, height=0.6, color="#d4d7de",
                         label="floor: bytes / link rate (unavoidable)"
                               if not yticks else None)
                axA.barh(y, st, left=x0 + fl, height=0.6, color=MUTED,
                         label="starved: every sender busy with the PP activation"
                               if not yticks else None)
                axA.barh(y, ic, left=x0 + fl + st, height=0.6, color=CORAL,
                         label="incast: senders had data and did not fill the link"
                               if not yticks else None)
                end = m["end_ns"] * MS
                # offset in POINTS, not in ms: these workloads span three
                # orders of magnitude of burst length (a 64-token run finishes
                # in ~4 ms, the 8192-token one in ~200), and a fixed
                # data-coordinate offset that reads well on one puts the label
                # off the axis on the other.
                axA.annotate(f"{end:.3g} ms", (end, y), textcoords="offset points",
                             xytext=(6, 0), va="center", fontsize=8,
                             color="0.30")
                xmax = max(xmax, end)
                yticks.append(y)
                ylabels.append(f"dec{si} · rank {rk}")
                y += 1
            y += 0.35                      # a gap between the decode stages
        # that topology's first token, over its own rows only. The label is
        # deferred: on a short burst tok1 can fall to the RIGHT of every
        # completion, so which side of the rule it fits on is only known once
        # the axis is.
        if pd.notna(r.ttft_ns) and yticks:
            axA.plot([r.ttft_ns * MS] * 2, [first_row_of_level - 0.5, y - 0.85],
                     ls=":", lw=1.4, color=colors[lv.level], zorder=4)
            tok1.append((r.ttft_ns * MS, first_row_of_level - 0.62,
                         f"{lv.label}  ·  token 1", colors[lv.level]))
            xmax = max(xmax, r.ttft_ns * MS)
        y += 1.15                          # a wider gap between the topologies
    if not yticks:
        plt.close(fig)
        return
    for x_, y_, txt, colr in tok1:
        axA.text(x_, y_, txt, fontsize=8, va="bottom", color=colr,
                 ha="right" if x_ > 0.62 * xmax else "left")
    axA.set_yticks(yticks)
    axA.set_yticklabels(ylabels, fontsize=8)
    axA.set_ylim(y - 1.0, -1.5)
    axA.set_xlim(0, xmax * 1.14)
    axA.set_xlabel("Time from the start of the run (ms)")
    axA.set_title("A · When each stage's KV handover runs, and when it completes "
                  "(@ largest buffer)", fontsize=10)
    axA.grid(True, axis="x", alpha=0.3)
    axA.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.11),
               ncol=2, frameon=False)

    # ---- B: the stretch, per stage, across the buffer sweep -------------- #
    styles = ["-", "--", ":", "-."]
    marks = ["o", "s", "^", "D"]
    frames = []
    for lv in usable:
        stages = sorted({si for r in lv.rows for si in r.kv_stage_ranks})
        for k, si in enumerate(stages):
            xs, ys = [], []
            for r in sorted(lv.rows, key=lambda q: q.buffer_mb):
                v = [r.kv_rank_stat[rk]["window_ns"] / r.kv_rank_stat[rk]["floor_ns"]
                     for rk in r.kv_stage_ranks.get(si, [])
                     if rk in r.kv_rank_stat and r.kv_rank_stat[rk]["floor_ns"]]
                if v:
                    xs.append(r.buffer_mb)
                    ys.append(100 * float(np.mean(v)))
            if xs:
                axB.plot(xs, ys, styles[k % 4], marker=marks[k % 4], ms=5,
                         color=colors[lv.level],
                         label=f"{lv.label} · dec{si}")
                frames.append(pd.DataFrame({"buffer_mb": xs}))
    axB.axhline(100, color="#555555", ls="--", lw=1.2)
    axB.annotate("100% = the bytes at wire speed", (1.0, 100),
                 xycoords=("axes fraction", "data"), textcoords="offset points",
                 xytext=(-3, 3), ha="right", va="bottom", fontsize=8,
                 color="#555555")
    # The stages need not start from the same baseline and the panel must not
    # be read as if they did: a stage fed by a NON-LAST prefill stage loses its
    # supply while those senders hand the PP activation forward, which inflates
    # its window by the same amount at every fan-in. Whether that happens is a
    # property of the placement and of the transfer mode (a bulk handover runs
    # after the prefill, so nothing competes with it), so the note is measured
    # here rather than asserted -- and the reading it prescribes, comparing a
    # stage to the control curve of the SAME stage, holds either way.
    starved_share: dict[int, float] = {}
    for lv in usable:
        rows = [r for r in lv.rows if r.kv_rank_stat and r.kv_stage_ranks]
        if not rows:
            continue
        r = max(rows, key=lambda q: q.buffer_mb)
        for si, ranks in r.kv_stage_ranks.items():
            v = [100 * m["starved_ns"] / m["floor_ns"]
                 for rk in ranks if (m := r.kv_rank_stat.get(rk))
                 and m["floor_ns"] and pd.notna(m["starved_ns"])]
            if v:
                starved_share[si] = max(starved_share.get(si, 0.0),
                                        float(np.mean(v)))
    hit = {si: v for si, v in starved_share.items() if v >= 1.0}
    note = ""
    if hit:
        who = ", ".join(f"dec{si}" for si in sorted(hit))
        note = (f"{who} also carries the PP-activation starvation\n"
                f"(~{max(hit.values()):.0f}%, fan-in independent).\n")
    if frames:
        logx_pow2(axB, pd.concat(frames, ignore_index=True), "buffer_mb",
                  "Per-switch buffer (MiB)")
    axB.set_ylabel("Handover window, % of the serialisation floor")
    axB.set_title("B · Does the same payload take longer to move?", fontsize=10)
    axB.grid(True, alpha=0.3, which="both")
    curves = axB.legend(fontsize=7.5, ncol=2, loc="upper center",
                        bbox_to_anchor=(0.5, -0.11), frameon=False)
    axB.add_artist(curves)
    # The note rides in a second, handle-less legend at loc='best' rather than
    # at a hand-picked corner: the curves sit anywhere between 100% and 200%
    # depending on the burst length, and every fixed anchor tried collided with
    # them on one workload or another.
    # one invisible handle, not an empty list: legend() with two empty lists
    # falls back to auto-detection and draws the curve legend a second time.
    tag = axB.legend(handles=[plt.Line2D([], [], ls="none")], labels=[""],
                     loc="best", frameon=True, handlelength=0,
                     handletextpad=0, framealpha=0.85, edgecolor="0.85",
                     title=note + "Read each stage against the fan-in-1\n"
                                  "control curve of the SAME line style.",
                     title_fontsize=7.5)
    tag.get_title().set_color("0.35")

    fig.suptitle("Per decode stage: the same KV payload, moved at different "
                 "times — and barely at a different speed", y=1.02)
    save_fig(fig, outdir, "12_per_stage_handover.png", written)


def fig_shard_skew_dist(levels: list[Level], outdir: Path,
                        written: list[Path]) -> None:
    """04 -- the one skew that does scale with the fan-in, as a DISTRIBUTION
    (buffer_sweep's 03). One box per (buffer, decode stage) over that stage's
    layers, of skew(stage, layer) = max-min arrival across the KV shards feeding
    that TP group -- 0 ms at fan-in 1 by construction (a single sender cannot be
    skewed against itself), ~7 ms at fan-in 2 and ~10 at fan-in 4.

    It is drawn as a population and not as three summary lines because the
    population is heavy-tailed at every buffer: a mean that moves 20% is not a
    distribution that moves, and only the boxes say whether the buffer shifts
    the median, squeezes the IQR or merely clips the tail. Fliers are kept --
    the tail layers are the ones that gate a decode stage.

    What it does NOT do on this data is reach the user: the decode waits for the
    whole cache anyway, so this skew is absorbed by the barrier rather than added
    to it. That is why it is figure 04 and not figure 01.

    Symlog y (linear near zero): at a wide incast degree two shards can land in
    the same nanosecond, and a log axis would silently drop those boxes."""
    pops = [(lv, [(r, r.kv_layer_skew) for r in lv.rows
                  if r.kv_layer_skew is not None and len(r.kv_layer_skew)])
            for lv in levels]
    pops = [(lv, ps) for lv, ps in pops if ps]
    if not pops:
        return
    n = len(pops)
    fig, axes = plt.subplots(1, n, squeeze=False, figsize=(max(6.0 * n, 7), 5.4))
    half = 0.16                                    # half-offset, in octaves
    for j, (lv, ps) in enumerate(pops):
        a = axes[0][j]
        st_all = sorted({int(v) for _, d in ps for v in d["stage"].unique()})
        st_colour = {st: (BLUE, CORAL, GREEN, VIOLET)[i % 4]
                     for i, st in enumerate(st_all)}
        allv = np.concatenate([d["skew_ns"].to_numpy() for _, d in ps]) * MS
        pos = allv[allv > 0]
        for r, d in ps:
            for i, st in enumerate(st_all):
                y = d.loc[d["stage"] == st, "skew_ns"].to_numpy(dtype=float) * MS
                if not len(y):
                    continue
                x = r.buffer_mb * 2.0 ** ((i - (len(st_all) - 1) / 2) * 2 * half)
                bp = a.boxplot([y], positions=[x], widths=x * 0.22,
                               patch_artist=True, manage_ticks=False,
                               medianprops=dict(color="k", lw=1.6),
                               flierprops=dict(marker="o", ms=3.0,
                                               mfc=st_colour[st], mec="none",
                                               alpha=0.6))
                bp["boxes"][0].set(facecolor=st_colour[st], alpha=0.55,
                                   edgecolor=st_colour[st])
        if len(pos):
            a.set_yscale("symlog", linthresh=float(pos.min()))
        else:
            # every shard pair landed in the same nanosecond: a symlog axis of
            # an all-zero population is an empty frame around a flat line, so
            # say it instead of drawing it.
            a.set_ylim(-0.05, 1.0)
            a.text(0.5, 0.55, "every (stage, layer) skew is exactly 0\n"
                              "— the TP shards land together",
                   ha="center", va="center", transform=a.transAxes,
                   fontsize=9, color=MUTED)
        gx = pd.DataFrame({"buffer_mb": [r.buffer_mb for r, _ in ps]})
        logx_pow2(a, gx, "buffer_mb", "Per-switch buffer (MiB)")
        mark_knees(a, lv.summary, label=(j == 0))
        n_lay = max(len(d[d["stage"] == st]) for _, d in ps for st in st_all)
        # the SIGNED median of the same population, read at the LARGEST buffer:
        # there the queueing that flips the sign run to run is gone, so a value
        # still away from zero is the one shard being systematically late -- a
        # path/placement asymmetry no amount of buffer removes. (The per-run
        # values are in summary.csv as kv_shard_bias_ns.)
        rmax = max(ps, key=lambda rd: rd[0].buffer_mb)[0]
        btxt = (f", shard bias {rmax.kv_shard_bias_ns * MS:+.3f} ms at "
                f"{rmax.buffer_mb:g} MiB" if pd.notna(rmax.kv_shard_bias_ns)
                else "")
        a.set_title(f"{lv.label}  ({n_lay} layers per box{btxt})", fontsize=10)
        a.grid(True, axis="y", alpha=0.3, which="both")
        h, _ = a.get_legend_handles_labels()
        a.legend(handles=h + [Patch(facecolor=st_colour[st], alpha=0.55,
                                    label=f"decode stage d{st}")
                              for st in st_all], fontsize=8)
        if j == 0:
            a.set_ylabel("Cross-shard arrival skew within a TP group (ms)")
    fig.suptitle("KV arrival skew across the shards of a TP group, "
                 "per (stage, layer)", y=1.02)
    save_fig(fig, outdir, "04_kv_shard_skew_distribution.png", written)


def fig_buffer_bloat(levels: list[Level], outdir: Path,
                     written: list[Path]) -> None:
    """06 -- peak and mean queue occupancy at each topology's fixed bottleneck
    link, and their ratio (buffer_sweep's 10). TOP: both in MiB, with the buffer
    itself as the reference line -- peak alone always grows, since it is bounded
    by the knob being swept, so on its own it says nothing. BOTTOM: the ratio,
    which is the reading: does the added buffer hold sustained load or only rare
    excursions? A flat ratio means the extra megabytes
    carry sustained load; a collapsing one means they only absorb rare
    excursions and idle in between -- and the incast question is whether a wider
    fan-in makes the excursions frequent enough to keep them busy."""
    usable = [lv for lv in levels
              if lv.summary is not None
              and "bn_qpeak_mb" in lv.summary.columns
              and lv.summary["bn_qpeak_mb"].notna().any()]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(2, n, squeeze=False, figsize=(max(5.0 * n, 6), 8.2))
    for j, lv in enumerate(usable):
        axT, axB = axes[0][j], axes[1][j]
        g = lv.summary.sort_values("buffer_mb")
        axT.plot(g["buffer_mb"], g["bn_qpeak_mb"], "o-", color=CORAL,
                 label="peak occupancy")
        if "bn_qmean_mb" in g.columns and g["bn_qmean_mb"].notna().any():
            axT.plot(g["buffer_mb"], g["bn_qmean_mb"], "v--", color=BLUE,
                     label="mean occupancy")
        axT.plot(g["buffer_mb"], g["buffer_mb"], ":", color=MUTED, lw=1.0,
                 label="the buffer itself")
        axT.set_yscale("log")
        axT.set_title(f"{lv.label} — {g['bottleneck'].iloc[0]}", fontsize=10)
        if "q_bloat_ratio" in g.columns and g["q_bloat_ratio"].notna().any():
            axB.plot(g["buffer_mb"], g["q_bloat_ratio"], "o-", color=VIOLET)
            axB.set_ylim(0, max(1.0, float(g["q_bloat_ratio"].max()) * 1.15))
        for a in (axT, axB):
            mark_knees(a, g, label=(a is axT and j == 0))
            logx_pow2(a, g, "buffer_mb", "Per-switch buffer (MiB)")
            a.grid(True, alpha=0.3, which="both")
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        if j == 0:
            axT.set_ylabel("Occupancy at the bottleneck (MiB, log)")
            axB.set_ylabel("mean ÷ peak occupancy")
    fig.suptitle("Peak and mean queue occupancy at the bottleneck", y=1.0)
    save_fig(fig, outdir, "06_buffer_bloat.png", written)


def fig_kv_fct(levels: list[Level], outdir: Path, written: list[Path]) -> None:
    """05 -- the KV flow-completion-time DISTRIBUTION, one panel per topology,
    one CDF per buffer (colour-graded along the swept axis), with the p99 marked.

    An incast is a tail story: the stage waits for its LAST shard, so the mean
    KV flow says almost nothing and the shape of the upper tail says everything.
    A CDF that stands up straight and then flattens far to the right is a few
    badly-delayed shards -- exactly what a deep buffer produces when it converts
    loss and backpressure into queueing delay -- while a CDF that shifts bodily
    right is every flow being slowed. The two look identical in the p99 column of
    summary.csv and completely different here."""
    usable = [lv for lv in levels
              if any(r.kv_fct_ns is not None and len(r.kv_fct_ns)
                     for r in lv.rows)]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(1, n, squeeze=False, sharey=True,
                             figsize=(max(5.0 * n, 6), 4.6))
    for j, lv in enumerate(usable):
        a = axes[0][j]
        bufs = [r.buffer_mb for r in lv.rows]
        nflows = 0
        for r in lv.rows:
            if r.kv_fct_ns is None or not len(r.kv_fct_ns):
                continue
            nflows = max(nflows, len(r.kv_fct_ns))
            x = np.sort(np.asarray(r.kv_fct_ns, dtype=float)) * MS
            y = 100.0 * np.arange(1, len(x) + 1) / len(x)
            c = LOSS_RED if r.lossy else buf_colour(r.buffer_mb, bufs)
            a.step(x, y, where="post", color=c, lw=1.8,
                   label=f"{r.buffer_mb:g} MiB" + (" (lossy)" if r.lossy else ""))
            if pd.notna(r.kv_fct_p99_ns):
                a.plot(r.kv_fct_p99_ns * MS, 99.0, marker="|", ms=10, color=c)
        a.axhline(99.0, color="k", ls=":", lw=0.8, alpha=0.5)
        a.set_title(f"{lv.label}  ({nflows} KV flows)", fontsize=10)
        a.set_xlabel("KV flow completion time (ms)")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7, loc="lower right")
        if j == 0:
            a.set_ylabel("Flows completed (%)")
    fig.suptitle("KV flow completion time, one CDF per buffer (tick = p99)",
                 y=1.02)
    save_fig(fig, outdir, "05_kv_fct_cdf.png", written)


def fig_per_link(levels: list[Level], outdir: Path, written: list[Path]) -> None:
    """07 -- congestion across EVERY KV-crossed link of each topology, not only
    the bottleneck (buffer_sweep's 09): delivered KV efficiency, PAUSE frames and
    peak queue, one row of panels per topology, one line per link, the bottleneck
    drawn thick.

    On a fan-in this is what says whether the incast has ONE congestion point or
    several: a single saturated uplink beside idle siblings is a placement
    artefact, while several uplinks pausing together is the fan-in itself. The
    link set is the level's fixed one, so a line is the same physical link at
    every buffer."""
    usable = [lv for lv in levels if lv.links and lv.summary is not None]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(n, 3, squeeze=False, figsize=(16, 4.4 * n))
    cmap = plt.get_cmap("tab10")
    for i, lv in enumerate(usable):
        axA, axB, axC = axes[i]
        g = lv.summary.sort_values("buffer_mb")
        for k, label in enumerate(lv.links):
            lw = 2.6 if k == 0 else 1.2
            c = cmap(k % 10)
            lbl = label + ("  (bottleneck)" if k == 0 else "")
            for a, col, scale in ((axA, f"link{k}_eff_pct", 1.0),
                                  (axB, f"link{k}_pause_frames", 1.0),
                                  (axC, f"link{k}_qpeak_bytes", 1 / 2**20)):
                if col in g.columns and g[col].notna().any():
                    a.plot(g["buffer_mb"], g[col] * scale, marker="o", lw=lw,
                           color=c, label=lbl if a is axA else label)
        axB.set_yscale("symlog", linthresh=1)
        for a, ttl, yl in (
                (axA, "delivered KV bandwidth", "KV bandwidth (% of nominal)"),
                (axB, "PFC PAUSE frames", "PAUSE frames (symlog)"),
                (axC, "peak queue occupancy", "Peak occupancy (MiB)")):
            mark_knees(a, g, label=(a is axB))
            logx_pow2(a, g, "buffer_mb", "Per-switch buffer (MiB)")
            a.set_title(f"{lv.label}: {ttl}", fontsize=10)
            a.set_ylabel(yl)
            a.grid(True, alpha=0.3, which="both")
            a.legend(fontsize=7)
    fig.suptitle("Congestion per KV-crossed link, one row per topology", y=1.0)
    save_fig(fig, outdir, "07_per_link_congestion.png", written)


# --------------------------------------------------------------------------- #
# The printed table, in reading order: what the transfer costs (floor, and the
# idle time the incast adds on top of it), what the user sees for it, what the
# fabric paid, and the health checks. Everything else -- the intra-stage skew,
# TTFT, the makespan, pp_skew, the per-stage decN_* and per-link link{i}_*
# blocks -- stays in summary.csv: those are references and controls, and on this
# data they either do not move or move below the resolution of a single run.
REPORT = ["level", "incast_degree", "buffer_mb", "bottleneck",
          "kv_floor_ms", "kv_idle_ms", "kv_starved_ms", "kv_incast_ms",
          "tok2_after_tok1_ms", "itl_steady_ms",
          "kv_tp_skew_mean_ms", "kv_fct_p99_ms",
          "total_pause_frames", "dropped_packets", "bn_qpeak_mb",
          "q_bloat_ratio", "loss_captured", "split_ok"]


def fig_insensitivity(levels: list[Level], s: pd.DataFrame, outdir: Path,
                      written: list[Path]) -> None:
    """10 -- the sweep's null result stated as one juxtaposition: on the SAME
    buffer axis, the fabric's state (left) moves by orders of magnitude while
    every time the application can observe (right) stays inside a fraction of
    a percent.

    The right panel normalises each series to its own mean on a y-window of AT
    LEAST ±2%, widened only if an excursion would leave it -- never narrowed to
    magnify one: the flatness is a measurement, not an axis choice, and the
    largest excursion found anywhere is printed inside the panel so the reader
    does not have to squint for it. Colour = topology (fan-in), linestyle =
    metric, both panels."""
    fabric = [("total_pause_frames", "PFC PAUSE frames", "-"),
              ("dropped_packets", "dropped packets", "--")]
    app = [("tok2_after_tok1_ms", "token2 − token1", "-"),
           ("total_exec_ms", "makespan", "--"),
           ("kv_window_ms", "KV handover window", ":")]
    fabric = [m for m in fabric if m[0] in s.columns and s[m[0]].notna().any()]
    app = [m for m in app if m[0] in s.columns and s[m[0]].notna().any()]
    if not fabric or not app:
        return
    colors = _level_colors(levels)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for lv in levels:
        g = s[s["level"] == lv.level].sort_values("buffer_mb")
        for col, _lab, ls in fabric:
            axL.plot(g["buffer_mb"], g[col], ls, marker="o", ms=3.5,
                     color=colors[lv.level])
    axL.set_yscale("symlog", linthresh=1)
    axL.set_ylim(bottom=0)
    logx_pow2(axL, s, "buffer_mb", "Per-switch buffer (MiB)")
    axL.set_ylabel("Count (symlog)")
    axL.set_title("Fabric state", fontsize=10)
    axL.grid(True, alpha=0.3, which="both")

    worst = 0.0
    for lv in levels:
        g = s[s["level"] == lv.level].sort_values("buffer_mb")
        for col, _lab, ls in app:
            v = g[col].dropna()
            if len(v) < 2 or not v.mean():
                continue
            norm = g[col] / v.mean()
            worst = max(worst, float((norm - 1).abs().max()))
            axR.plot(g["buffer_mb"], norm, ls, marker="o", ms=3.5,
                     color=colors[lv.level])
    half = max(0.02, worst * 1.25)
    axR.set_ylim(1 - half, 1 + half)
    logx_pow2(axR, s, "buffer_mb", "Per-switch buffer (MiB)")
    axR.set_ylabel("Value / its own mean over the sweep")
    axR.set_title("Application time", fontsize=10)
    axR.grid(True, alpha=0.3)
    axR.text(0.03, 0.05, f"largest excursion anywhere: "
             f"±{100 * worst:.2f}%  (window is ±{100 * half:.0f}%)",
             transform=axR.transAxes, fontsize=8, color="0.35")

    from matplotlib.lines import Line2D
    topo = [Line2D([], [], color=colors[lv.level], label=lv.label)
            for lv in levels]
    axL.legend(handles=topo + [Line2D([], [], color="k", ls=ls, label=lab)
                               for _c, lab, ls in fabric], fontsize=8)
    axR.legend(handles=[Line2D([], [], color="k", ls=ls, label=lab)
                        for _c, lab, ls in app], fontsize=8, loc="upper right")
    fig.suptitle("Fabric state and application-visible time, across the same "
                 "buffer sweep", y=1.02)
    save_fig(fig, outdir, "10_insensitivity.png", written)


def tradeoff_frame(s: pd.DataFrame, levels: list[Level]) -> pd.DataFrame:
    """One row per topology at the largest buffer, with the instants the
    prefill/handover trade-off is made of. Shared by figure 11 and the report so
    the picture and the sentences cannot drift apart.

    The columns are absolute instants (ms from the run's start), not durations:
    the whole point is WHERE the transfer sits on the clock, and a table of
    durations is exactly what hides it.

    The transfer window [kv_start, gate] is cut at TWO instants, not one, because
    'exposed' is not one thing:

        kv_start .. ttft   HIDDEN     the prefill is still running, the transfer
                                      is free on the model's clock
        ttft .. wake       EXPOSED,   the first token has been SENT but the
                           IDLE      decode has not received it yet, so no
                                      decode op exists that this KV could hold
                                      up: it is exposed to the user's clock and
                                      to nothing else
        wake .. gate       EXPOSED,   the decode is awake and consuming layer by
                           BLOCKING   layer while the tail of the same transfer
                                      is still landing -- the only stretch where
                                      the KV can actually stop the pipeline

    `wake` is the FIRSTTOK ARRIVAL at the decode (dec0_input_arrival_ns, the max
    over stage 0's TP shards: the group advances at its slowest shard), which is
    the instant stage 0 could first run. It is NOT the TTFT: the TTFT is when the
    prefill SENT that token, and the message then queues behind the KV bulk it
    shares the fabric with -- at fan-in 1 it arrives 76 ms later, exactly at the
    gate.

    wake..gate is still an ENVELOPE, not the stall: the decode needs its layers
    in order, not all at once, so it only waits where it outruns the transfer.
    `blocked_ms` (dec_kv_block_ms, measured idle attributed to a KV arrival) is
    carried alongside so the figure and the report can quote the bound and the
    realised cost together instead of passing the bound off as the cost."""
    if s.empty:
        return pd.DataFrame()
    need_cols = {"ttft_ms", "stream_head_ms", "kv_window_ms", "lag_ms",
                 "tok2_after_tok1_ms", "total_exec_ms"}
    if not need_cols <= set(s.columns):
        return pd.DataFrame()
    big = s[s["buffer_mb"] == s["buffer_mb"].max()].dropna(subset=list(need_cols))
    if big.empty:
        return pd.DataFrame()
    lab = {L.level: L.label for L in levels}
    out = big.copy()
    out["label"] = out["level"].map(lab).fillna(out["level"])
    out["kv_start_ms"] = out["ttft_ms"] + out["stream_head_ms"]
    out["gate_ms"] = out["kv_start_ms"] + out["kv_window_ms"]
    out["tok2_ms"] = out["ttft_ms"] + out["tok2_after_tok1_ms"]
    out["rest_ms"] = out["total_exec_ms"] - out["tok2_ms"]

    # The two cuts, both clamped INTO the window and kept ordered, so the three
    # parts always sum to kv_window_ms whatever order the instants come out in
    # (a stream that starts after the first token, a wake past the gate).
    cut_ttft = out["ttft_ms"].clip(lower=out["kv_start_ms"],
                                   upper=out["gate_ms"])
    wake = pd.Series(NAN, index=out.index, dtype=float)
    for col in ("dec0_input_arrival_ns", "dec_start_ns"):   # measured, then bound
        if col in out.columns:
            wake = wake.fillna(out[col] * MS)
    out["wake_ms"] = wake
    cut_wake = wake.clip(lower=cut_ttft, upper=out["gate_ms"]).fillna(
        out["gate_ms"])            # no wake instant -> nothing is attributable
    out["hidden_ms"] = cut_ttft - out["kv_start_ms"]
    out["exposed_idle_ms"] = cut_wake - cut_ttft
    out["exposed_blocking_ms"] = out["gate_ms"] - cut_wake
    out["exposed_ms"] = out["exposed_idle_ms"] + out["exposed_blocking_ms"]
    out["blocked_ms"] = (out["dec_kv_block_ms"] if "dec_kv_block_ms"
                         in out.columns else NAN)
    return out.sort_values("incast_degree").reset_index(drop=True)


# The payback terms of the trade-off, in the order the waterfall stacks them:
# (column, label, short label for the figure's x axis, is_network).
# Δ(kv_start) is not here -- it is the part of the TTFT gain that SURVIVES, and
# the waterfall's opening bar already carries it.
PAYBACK = [("stream_head_ms", "lost prefill/KV overlap", "lost\noverlap", False),
           ("kv_starved_ms", "PP-activation starvation", "PP\nstarvation", False),
           ("kv_incast_ms", "incast (congestion)", "INCAST\n(network)", True),
           ("lag_ms", "release lag", "release\nlag", False),
           ("rest_ms", "rest of decode", "rest of\ndecode", False)]


def fig_prefill_tradeoff(t: pd.DataFrame, outdir: Path,
                         written: list[Path]) -> None:
    """11 -- what widening the prefill actually buys, and what the handover
    takes back.

    A wider prefill pool is bought for TTFT, and it delivers: the first token
    lands much earlier. But the KV handover does NOT get faster with it -- it
    moves a fixed number of bytes into each decode rank at a fixed rate, so its
    length is the same ~136 ms floor at every width. What changes is WHERE that
    fixed block sits relative to the first token.

    LEFT, absolute time, one row per topology, the transfer cut at the two
    instants that change what it costs (see tradeoff_frame):

      BLUE   prefill running before the KV stream starts
      GREEN  KV streaming under the prefill -- HIDDEN, free on the model's clock
      AMBER  KV after the first token was sent but before the decode received
             it -- EXPOSED to the user's clock, but no decode op exists yet for
             it to hold up
      CORAL  KV still landing after the decode woke -- the ONLY stretch in which
             the transfer can stall the pipeline; the measured stall inside it
             is written on the row
      VIOLET release lag (gate -> token 2), MUTED the rest of the decode

    Widening the prefill slides the tok1 marker left much faster than it slides
    the stream's start, so green shrinks and amber+coral grow by the same amount:
    the transfer is not removed, it is UNCOVERED. What the split then shows is
    that being uncovered is not by itself a cost -- at fan-in 1 the whole exposed
    stretch is amber (the first token is still queued behind the KV bulk and
    arrives AT the gate, so the decode never waits), and only from fan-in 2 does
    a coral stretch appear at all.

    RIGHT, the same statement as an accounting, for the widest fan-in against
    the narrowest: the TTFT gain, then every term that takes part of it back,
    then what is left in the makespan. This is the figure that answers "we gain
    in TTFT but the makespan barely moves -- where does it go", and it also
    shows how little of the payback is congestion (one coral bar among four
    structural ones)."""
    if t.empty or len(t) < 2:
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 5.2),
                                   gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ---- LEFT: the absolute timeline ------------------------------------- #
    # (width, left edge, colour, legend). Every segment is anchored to the
    # instant it STARTS at, never to the running sum of the ones before it, so a
    # clamped or missing cut cannot slide the rest of the row.
    segs = [("kv_start_ms", None, BLUE, "prefill (before the KV stream starts)"),
            ("hidden_ms", "kv_start_ms", GREEN,
             "KV under the prefill — HIDDEN"),
            ("exposed_idle_ms", "ttft_ms", AMBER,
             "KV exposed before the decode has its first token —\n"
             "nothing to hold up"),
            ("exposed_blocking_ms", "wake_ms", CORAL,
             "KV still landing with the decode awake —\ncan stall the pipeline"),
            ("lag_ms", "gate_ms", VIOLET, "release lag (gate → token 2)"),
            ("rest_ms", "tok2_ms", MUTED, "rest of the decode")]
    marks = [("ttft_ms", "token 1 sent"), ("wake_ms", "token 1 at decode"),
             ("gate_ms", "last KV layer"), ("tok2_ms", "token 2")]
    for i, r in t.iterrows():
        for col, leftcol, colour, _lab in segs:
            width = max(float(r[col]), 0.0)
            left = 0.0 if leftcol is None else float(r[leftcol])
            if not (width > 0 and np.isfinite(left)):
                continue
            axL.barh(i, width, left=left, height=0.6, color=colour,
                     edgecolor="white", linewidth=0.4)
        for col, name in marks:
            if pd.notna(r[col]):
                axL.plot(r[col], i, marker="|", ms=20, mew=2.0, color="#111111",
                         ls="none", zorder=5)
        blocked = (f" ({r['blocked_ms']:.0f} stalling)"
                   if pd.notna(r["blocked_ms"]) else "")
        axL.annotate(f"hidden {r['hidden_ms']:.0f}  |  exposed "
                     f"{r['exposed_idle_ms']:.0f} + "
                     f"{r['exposed_blocking_ms']:.0f} ms{blocked}",
                     (float(r["total_exec_ms"]) * 1.02, i), fontsize=7.5,
                     va="center", color="0.35")

    # The four instants are named once, over the top row. Two of them COINCIDE
    # there (at fan-in 1 the first token reaches the decode exactly at the gate,
    # having queued behind the KV it shares the link with) -- so labels closer
    # than TIE_MS are merged into one rather than drawn on top of each other,
    # and what is left alternates between two heights.
    TIE_MS = float(t["total_exec_ms"].max()) * 0.01
    top = t.iloc[0]
    named: list[tuple[float, list[str]]] = []
    for col, name in sorted(marks, key=lambda m: float(top[m[0]])
                            if pd.notna(top[m[0]]) else float("inf")):
        if not pd.notna(top[col]):
            continue
        x = float(top[col])
        if named and x - named[-1][0] <= TIE_MS:
            named[-1][1].append(name)
        else:
            named.append((x, [name]))
    for k, (x, names) in enumerate(named):
        axL.annotate(" = ".join(n.replace("\n", " ") for n in names),
                     (x, -0.36 - 0.30 * (k % 2)), fontsize=7,
                     ha="center", va="bottom", color="#111111")
    axL.set_yticks(range(len(t)))
    axL.set_yticklabels([r["label"] for _, r in t.iterrows()], fontsize=9)
    axL.invert_yaxis()
    axL.set_ylim(len(t) - 0.4, -1.15)     # headroom for the instant labels
    axL.set_xlim(0, float(t["total_exec_ms"].max()) * 1.32)
    axL.set_xlabel("Absolute time (ms from the start of the run)")
    axL.set_title("Where the KV handover sits on the clock, and which part of "
                  "it the decode can feel", fontsize=10)
    axL.grid(True, axis="x", alpha=0.3)
    axL.legend(handles=[Patch(color=c, label=l) for _c, _lc, c, l in segs],
               fontsize=7.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), frameon=False)

    # ---- RIGHT: the payback waterfall ------------------------------------ #
    base, wide = t.iloc[0], t.iloc[-1]
    # GREEN what the wider prefill gives, CORAL the one term that is the fabric,
    # grey the structural ones, BLUE what survives -- the same reading of the
    # three colours as the left panel, so the two can be read together.
    bars = [("TTFT gain\n(prefill)", wide["ttft_ms"] - base["ttft_ms"], GREEN)]
    for col, _lab, short, is_net in PAYBACK:
        bars.append((short, float(wide[col] - base[col]),
                     CORAL if is_net else MUTED))
    cum = 0.0
    for k, (lab, val, colour) in enumerate(bars):
        axR.bar(k, val, bottom=cum, width=0.62, color=colour,
                edgecolor="white", linewidth=0.5)
        axR.annotate(f"{val:+.1f}", (k, cum + val + (0.9 if val >= 0 else -0.9)),
                     ha="center", va="bottom" if val >= 0 else "top",
                     fontsize=8, fontweight="bold")
        nxt = cum + val
        axR.plot([k - 0.31, k + 0.31 + 0.38], [nxt, nxt], color="0.55", lw=0.9,
                 ls=":", zorder=1)
        cum = nxt
    axR.bar(len(bars), cum, width=0.62, color=BLUE, edgecolor="white",
            linewidth=0.5)
    axR.annotate(f"{cum:+.1f}", (len(bars), cum - 0.9), ha="center", va="top",
                 fontsize=8, fontweight="bold")
    axR.axhline(0, color="#333333", lw=1.0)
    axR.set_xticks(range(len(bars) + 1))
    axR.set_xticklabels([b[0] for b in bars] + ["net\nΔmakespan"], fontsize=8)
    axR.set_ylabel("ms vs the narrowest prefill")
    kept = 100 * cum / (wide["ttft_ms"] - base["ttft_ms"]) \
        if wide["ttft_ms"] != base["ttft_ms"] else NAN
    axR.set_title(f"{wide['label']} vs {base['label']}: {kept:.0f}% of the "
                  f"TTFT gain reaches the makespan", fontsize=10)
    axR.grid(True, axis="y", alpha=0.3)
    fig.suptitle("The prefill/handover trade-off: a wider prefill moves the "
                 "transfer out from under itself", y=1.02)
    save_fig(fig, outdir, "11_prefill_vs_handover_tradeoff.png", written)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Same --sweep/--workload/--root vocabulary as the other analyzers, with the
    # incast twist documented in utils.incast: the generator wrote the configs
    # (--sweep) and the outputs (--workload) under DIFFERENT names, and the run
    # tags sit directly under output/<domain>/<workload> with no <sweep> nesting.
    ap.add_argument("--sweep", default=incast.CONFIG_SWEEP,
                    help=f"config sub-dir under configs/astra_sim/ns3 (default: "
                         f"{incast.CONFIG_SWEEP})")
    ap.add_argument("--workload", default=incast.OUT_WORKLOAD,
                    help=f"workload dir under output/<domain>; the run tags sit "
                         f"directly under it (default: {incast.OUT_WORKLOAD})")
    ap.add_argument("--root", default=str(incast.ROOT), type=Path,
                    help=f"project root (default: {incast.ROOT})")
    ap.add_argument("--levels", nargs="+", default=None,
                    help="incast levels to analyse, e.g. --levels T3 T4 "
                         "(default: every level found)")
    ap.add_argument("--top-switches", type=int, default=3,
                    help="how many switches to keep PER metric (PFC, occupancy); "
                         "the busy set is their union (default: 3)")
    ap.add_argument("--top-links", type=int, default=4,
                    help="how many KV-crossed links per topology figure 07 and "
                         "summary.csv carry; the first is the bottleneck every "
                         "bn_* column describes (default: 4)")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/incast/"
                         "<workload>)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    outdir = (Path(a.out) if a.out else
              root / "results" / "sweep_analysis" / "incast" / a.workload)

    try:
        levels_found = incast.discover_levels(a.workload, root, "ns3")
        need(levels_found,
             f"no incast level under {root / 'output' / 'ns3' / a.workload}. "
             f"Is --workload right?")
        if a.levels:
            want = set(a.levels)
            missing = want - set(levels_found)
            need(not missing, f"--levels {sorted(missing)} not present; "
                              f"found {levels_found}")
            levels_found = [l for l in levels_found if l in want]

        print(f"  root      {root}")
        print(f"  workload  {a.workload}")
        print(f"  out       {outdir}")
        print(f"  levels    {levels_found}")

        levels = []
        for lv in levels_found:
            L = analyse_level(lv, root, a.workload, a.sweep,
                              a.top_switches, a.top_links)
            if L is not None:
                levels.append(L)
        need(levels, "no incast level produced any analysable run.")
        levels.sort(key=lambda L: L.degree)

        # each level brought its own summary (per-topology knees and worst-stage
        # reductions already applied); the sweep-wide table is their concatenation
        s = pd.concat([L.summary for L in levels], ignore_index=True)
        s = s.sort_values(["incast_degree", "buffer_mb"]).reset_index(drop=True)

        fresh_dir(outdir)
        front = [c for c in REPORT if c in s.columns]
        s[front + [c for c in s.columns if c not in front]].to_csv(
            outdir / "summary.csv", index=False)

        # numeric order == write order, so the "Wrote" listing reads 01..09
        written: list[Path] = []
        fig_incast_cost(levels, s, outdir, written)         # 01 the headline
        fig_first_to_second(levels, outdir, written)        # 02 what the user sees
        fig_fabric_cost(levels, s, outdir, written)         # 03 what the buffer buys
        fig_shard_skew_dist(levels, outdir, written)        # 04
        fig_kv_fct(levels, outdir, written)                 # 05
        fig_buffer_bloat(levels, outdir, written)           # 06
        fig_per_link(levels, outdir, written)               # 07
        for L in levels:                                    # 08 per topology
            fig_queue_fill(L, outdir, written)
        for L in levels:                                    # 09 per topology
            fig_kv_cumulative(L, outdir, written)
        fig_insensitivity(levels, s, outdir, written)       # 10 the null result
        tframe = tradeoff_frame(s, levels)                  # 11 the trade-off
        fig_prefill_tradeoff(tframe, outdir, written)
        fig_stage_handover(levels, outdir, written)         # 12 per stage

        pd.set_option("display.width", 240)
        print("\n================ INCAST SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))

        # the knees, one row per topology: "where does the buffer stop
        # mattering" is a different question from "how much does it move", and
        # the answer is per incast degree.
        print("\n---- knees per topology (MiB) ----")
        print(f"  {'topology':<16} {'fan-in':>6}  "
              + "  ".join(f"{name:<12}" for _c, name in KNEE_STYLE.values()))
        for L in levels:
            vals = []
            for col in KNEE_STYLE:
                v = L.summary[col].iloc[0] if col in L.summary.columns else NAN
                vals.append(f"{v:g}" if pd.notna(v) else "never")
            print(f"  {L.label:<16} {L.degree:>6}  "
                  + "  ".join(f"{v:<12}" for v in vals))

        # The headline, at the LARGEST buffer -- the end of the sweep where no run
        # is lossy or paused, so what is left is attributable to the fan-in alone.
        #
        # PER DECODE STAGE, not pooled over the ranks. The floor is the same for
        # every rank (the KV is evenly sharded and each receiver owns a dedicated
        # 100 Gb/s port), but the ranks are NOT interchangeable: those fed by a
        # non-last prefill stage lose 6.8 ms to that stage's PP activation and
        # then have to recover from it, so their residue is not a clean reading
        # of the fan-in. dec0 is the contaminated stage, dec1 the clean one.
        big = s[s["buffer_mb"] == s["buffer_mb"].max()].sort_values("incast_degree")
        stage_cols = sorted(c for c in s.columns
                            if re.fullmatch(r"kv_incast_dec\d+_ms", c))
        if len(big) > 1 and stage_cols:
            print(f"\n---- the incast cost, at the largest buffer "
                  f"({s['buffer_mb'].max():g} MiB) ----")
            head = "".join(f"{c[10:-3]:>20}" for c in stage_cols)
            print(f"  {'fan-in':>6}  {'floor':>9}{head}   (starved, INCAST) "
                  f"mean over the stage's ranks")
            for _, rr in big.iterrows():
                cells = ""
                for c in stage_cols:
                    sc = c.replace("kv_incast_", "kv_starved_")
                    cells += f"{rr.get(sc, NAN):8.2f} +{rr[c]:7.2f} ms"
                print(f"  {int(rr['incast_degree']):>6}  "
                      f"{rr['kv_floor_ms']:6.1f} ms{cells}")
            ctrl = big.iloc[0]
            res = max((ctrl[c] for c in stage_cols), default=NAN)
            print("  idle = window − floor; starved = every sender of that rank "
                  "busy with the PP activation;")
            print("  INCAST = the residue, i.e. senders that had data and the "
                  "wire and still did not fill the link.")
            print(f"  RESOLUTION: the fan-in-1 control leaves {res:.3f} ms "
                  f"({100 * res / ctrl['kv_floor_ms']:.2f} % of the floor), so the "
                  f"floor is attainable and the")
            print("  measure is sound -- but across the buffers of ONE topology "
                  "the residue itself wanders by ~1 ms")
            print("  with no loss and no PAUSE, which is the scatter this "
                  "one-run-per-point sweep can resolve. Read")
            print("  fan-in 1 vs fan-in >1 as the finding; do NOT read an "
                  "ordering between fan-in 2 and 4 off it.")

        # WHEN each stage's handover runs (figure 12), as numbers. Same payload
        # on every receiver, so the completion times separate two effects that
        # a single duration cannot: WHEN the transfer could start (the prefill
        # must have produced those layers -- a pipeline property) and HOW MUCH
        # LONGER than the floor it then took (the fabric's part). Mean over the
        # ranks of the stage; the ranks of a stage agree to well under a ms.
        print(f"\n---- per decode stage, at the largest buffer "
              f"({s['buffer_mb'].max():g} MiB) ----")
        print(f"  {'topology':<16} {'stage':>6} {'starts':>9} {'completes':>10} "
              f"{'window':>9} {'% of floor':>11} {'after tok1':>11}")
        for L in levels:
            rows = [r for r in L.rows if r.kv_rank_stat and r.kv_stage_ranks]
            if not rows:
                continue
            r = max(rows, key=lambda q: q.buffer_mb)
            for si in sorted(r.kv_stage_ranks):
                m = [r.kv_rank_stat[rk] for rk in r.kv_stage_ranks[si]
                     if rk in r.kv_rank_stat]
                if not m:
                    continue
                st = float(np.mean([q["start_ns"] for q in m])) * MS
                en = float(np.mean([q["end_ns"] for q in m])) * MS
                wn = float(np.mean([q["window_ns"] for q in m])) * MS
                pc = 100 * float(np.mean([q["window_ns"] / q["floor_ns"]
                                          for q in m if q["floor_ns"]]))
                aft = en - r.ttft_ns * MS if pd.notna(r.ttft_ns) else NAN
                print(f"  {L.label:<16} {'dec' + str(si):>6} {st:6.1f} ms "
                      f"{en:7.1f} ms {wn:6.1f} ms {pc:10.1f} % "
                      f"{aft:+8.1f} ms")
        print("  starts = first send feeding the stage; completes = its last "
              "arrival, i.e. that stage's KV gate.")
        print("  The stages carry the SAME bytes and see the same floor: the "
              "gap between their completion")
        print("  times is a shifted START (the pipelined prefill produces a "
              "later stage's layers later),")
        print("  not a slower transfer -- the window itself is within a few "
              "percent of the floor everywhere.")

        # Does the incast reach the MODEL's clock? The release lag tok2 − gate
        # names the binding constraint: ~0 means the last KV byte of the gate
        # stage releases the second token (network-bound), so that stage's idle
        # lands on tok2 1:1; ~ITL means decode compute was pacing anyway and
        # the idle is absorbed. On dec1 (the gate stage here) starved is 0, so
        # the idle IS the incast term.
        if len(big) > 1 and {"tok2_ns", "kv_gate_ns"} <= set(big.columns):
            print("\n---- pass-through to the second token, at the largest "
                  f"buffer ({s['buffer_mb'].max():g} MiB) ----")
            print(f"  {'fan-in':>6}  {'tok2 - gate':>12}  {'/ ITL':>6}  "
                  f"{'gate stage':>10}  {'its idle':>9}  {'lands on tok2':>14}")
            for _, rr in big.iterrows():
                lag = (rr["tok2_ns"] - rr["kv_gate_ns"]) * MS
                itl = rr.get("itl_steady_ms", NAN)
                gst = 0
                for si in range(8):
                    if f"dec{si}_kv_ready_ns" in big.columns and pd.notna(
                            rr.get(f"dec{si}_kv_ready_ns")):
                        if rr[f"dec{si}_kv_ready_ns"] >= rr.get(
                                f"dec{gst}_kv_ready_ns", -1):
                            gst = si
                idle = (rr.get(f"kv_starved_dec{gst}_ms", NAN)
                        + rr.get(f"kv_incast_dec{gst}_ms", NAN))
                frac = lag / itl if itl else NAN
                verdict = ("1:1 (network-bound)" if frac < 0.2 else
                           "absorbed (compute-bound)" if frac > 0.8 else
                           "partial")
                print(f"  {int(rr['incast_degree']):>6}  {lag:9.2f} ms  "
                      f"{frac:6.2f}  {'dec' + str(gst):>10}  {idle:6.2f} ms  "
                      f"{verdict:>24}")

        # The model's clock, exactly: the ledger tok1→tok2 = head + floor +
        # starved + incast + lag (see Row.flat), read against the fan-in-1
        # control. It answers the two questions the pass-through table only
        # names: how much the handover DILATES the second token (exposure vs
        # the steady ITL — at this burst length the handover is not hidden at
        # all, it IS tok2's latency), and how much of that dilation the fan-in
        # actually ADDS (floor and head are fan-in-independent, so the control
        # pays almost all of it too — which is exactly what hides the incast
        # increment inside every end-to-end metric).
        ledger_cols = {"stream_head_ms", "kv_floor_ms", "kv_starved_ms",
                       "kv_incast_ms", "lag_ms", "tok2_after_tok1_ms"}
        if len(big) > 1 and ledger_cols <= set(big.columns):
            ctrl = big.iloc[0]
            has_ctrl = int(ctrl["incast_degree"]) == 1
            print(f"\n---- the model's clock, exactly, at the largest buffer "
                  f"({s['buffer_mb'].max():g} MiB) ----")
            print("  tok1→tok2 = head + floor + starved + incast + lag   "
                  "(worst receiver; head < 0: the KV stream is already running "
                  "during prefill)")
            print(f"  {'fan-in':>6} {'head':>8} {'floor':>8} {'starved':>8} "
                  f"{'incast':>8} {'lag':>6}  {'= tok1→tok2':>12} {'x ITL':>6}"
                  f" {'vs control':>12}")
            for _, rr in big.iterrows():
                itl = rr.get("itl_steady_ms", NAN)
                dil = rr["tok2_after_tok1_ms"] / itl if itl else NAN
                dvs = (rr["tok2_after_tok1_ms"] - ctrl["tok2_after_tok1_ms"]
                       if has_ctrl else NAN)
                print(f"  {int(rr['incast_degree']):>6} "
                      f"{rr['stream_head_ms']:8.2f} {rr['kv_floor_ms']:8.2f} "
                      f"{rr['kv_starved_ms']:8.2f} {rr['kv_incast_ms']:8.2f} "
                      f"{rr['lag_ms']:6.2f}  {rr['tok2_after_tok1_ms']:12.2f} "
                      f"{dil:5.1f}x"
                      + (f" {dvs:+9.2f} ms" if has_ctrl and pd.notna(dvs)
                         else f" {'—':>12}"))
            worst = big.iloc[-1]
            itl = worst.get("itl_steady_ms", NAN)
            stall = worst.get("tok2_latency_ms", NAN)
            if pd.notna(itl) and itl > 0:
                print(f"  EXPOSURE: tok1→tok2 is "
                      f"~{worst['tok2_after_tok1_ms'] / itl:.0f}x the steady ITL "
                      f"({itl:.1f} ms) — at this burst length the handover is "
                      f"NOT hidden: it IS the second token's latency.")
            blocked = worst.get("dec_kv_block_ms", NAN)
            if pd.notna(stall) and pd.notna(blocked):
                print(f"  STALL, not compute: of the {stall:.1f} ms the pass "
                      f"takes (dec start → tok2), {blocked:.1f} ms "
                      f"({100 * blocked / stall:.0f}%) is measured idle that a "
                      f"KV arrival ends — not inferred from the KV tail, which "
                      f"only bounds it "
                      f"({worst.get('kv_tail_after_dec_start_ms', NAN):.1f} ms).")
            if has_ctrl:
                # Attribution, not just the total: Δtok1→tok2 vs the control is
                # NOT the incast's cost. Δhead is the prefill getting faster
                # with the wider pool (tok1 moves, the KV stream does not — a
                # compute effect, the same trap as putting TTFT on the degree
                # axis); Δlag is the control's compute-paced regime (lag≈ITL)
                # vanishing once the gate is late. Only Δincast (and the
                # per-stage starvation recovery) is congestion.
                terms = [("stream_head_ms", "Δhead"), ("kv_floor_ms", "Δfloor"),
                         ("kv_starved_ms", "Δstarved"),
                         ("kv_incast_ms", "Δincast"), ("lag_ms", "Δlag")]
                print("  Δ vs the fan-in-1 control, term by term (the worst "
                      "receiver may sit in a different decode stage per run):")
                print("  " + f"{'fan-in':>6} "
                      + " ".join(f"{n:>9}" for _c, n in terms)
                      + f"  {'= Δtok1→tok2':>13}")
                for _, rr in big.iloc[1:].iterrows():
                    cells = " ".join(f"{rr[c] - ctrl[c]:+9.2f}"
                                     for c, _n in terms)
                    print(f"  {int(rr['incast_degree']):>6} {cells}  "
                          f"{rr['tok2_after_tok1_ms'] - ctrl['tok2_after_tok1_ms']:+13.2f}")
                dinc = big["kv_incast_ms"].iloc[-1] - ctrl["kv_incast_ms"]
                print(f"  MASKING: the control already pays "
                      f"{ctrl['tok2_after_tok1_ms']:.1f} ms of the exposure "
                      f"(floor does not depend on the fan-in), and of the "
                      f"widest fan-in's Δtok1→tok2 only Δincast "
                      f"({dinc:+.2f} ms) is the incast itself — Δhead is "
                      f"prefill speed (compute), Δlag a regime change. That is "
                      f"why end-to-end metrics cannot isolate the incast: its "
                      f"increment is millisecond-scale inside a ~100 ms "
                      f"transfer exposure the control pays too, and the other "
                      f"terms of the ledger move more than it does for "
                      f"non-network reasons.")

        # THE TRADE-OFF (figure 11). The degree axis is also the prefill-width
        # axis, so it buys TTFT -- and the question this answers is where that
        # gain goes, since the makespan does not fall by nearly as much.
        #
        # This is the one place TTFT and the makespan are read along the degree
        # axis, and it is not the reading the rest of the file refuses (see the
        # module docstring): nothing here attributes them to the network. The
        # opposite -- it MEASURES how little of the difference is network, by
        # anchoring both to the instant the KV stream starts.
        if len(tframe) > 1:
            base, wide = tframe.iloc[0], tframe.iloc[-1]
            print(f"\n---- what the wider prefill buys, and what the handover "
                  f"takes back ({s['buffer_mb'].max():g} MiB) ----")
            print(f"  {'fan-in':>6} {'prefill':>8} {'TTFT':>9} "
                  f"{'KV starts':>10} {'KV hidden':>10} {'exp/idle':>10} "
                  f"{'exp/blocking':>13} {'of it stalls':>13} "
                  f"{'token 2':>9} {'makespan':>10}")
            for _, r in tframe.iterrows():
                print(f"  {int(r['incast_degree']):>6} {'TP' + str(int(r['prefill_tp'])):>8} "
                      f"{r['ttft_ms']:6.1f} ms {r['kv_start_ms']:7.1f} ms "
                      f"{r['hidden_ms']:7.1f} ms {r['exposed_idle_ms']:7.1f} ms "
                      f"{r['exposed_blocking_ms']:10.1f} ms "
                      f"{r['blocked_ms']:10.1f} ms "
                      f"{r['tok2_ms']:6.1f} ms {r['total_exec_ms']:7.1f} ms")
            d_ttft = wide["ttft_ms"] - base["ttft_ms"]
            d_mk = wide["total_exec_ms"] - base["total_exec_ms"]
            d_start = wide["kv_start_ms"] - base["kv_start_ms"]
            if d_ttft:
                print(f"  RETENTION: {wide['label']} cuts TTFT by "
                      f"{abs(d_ttft):.1f} ms against {base['label']} but the "
                      f"makespan by only {abs(d_mk):.1f} ms — "
                      f"{100 * d_mk / d_ttft:.0f}% of the gain survives, "
                      f"{abs(d_ttft - d_mk):.1f} ms is paid back.")
            print(f"  WHY: the handover is anchored to the instant the prefill "
                  f"starts PRODUCING the KV ({base['kv_start_ms']:.1f} → "
                  f"{wide['kv_start_ms']:.1f} ms, only {d_start:+.1f} ms), not "
                  f"to the first token ({d_ttft:+.1f} ms), and its length is "
                  f"fixed by bytes/rate. So the prefill stops covering it:")
            print(f"  hidden {base['hidden_ms']:.1f} → {wide['hidden_ms']:.1f} ms, "
                  f"exposed {base['exposed_ms']:.1f} → {wide['exposed_ms']:.1f} ms, "
                  f"window {base['kv_window_ms']:.1f} → "
                  f"{wide['kv_window_ms']:.1f} ms — the transfer is not "
                  f"removed, it is UNCOVERED.")
            # ... and being uncovered is not yet a cost: the first token has to
            # reach the decode before any of the KV can hold an op up, and that
            # message queues behind the KV bulk itself.
            print(f"  EXPOSED TO WHAT: of the exposed stretch, "
                  f"{base['exposed_idle_ms']:.1f} → {wide['exposed_idle_ms']:.1f} ms "
                  f"runs before the decode even receives the first token "
                  f"(nothing to hold up) and only "
                  f"{base['exposed_blocking_ms']:.1f} → "
                  f"{wide['exposed_blocking_ms']:.1f} ms lands with the decode "
                  f"awake — of which the MEASURED stall is "
                  f"{base['blocked_ms']:.1f} → {wide['blocked_ms']:.1f} ms "
                  f"(the rest is KV that arrived late and free, ahead of the "
                  f"layer that needed it).")
            terms = [(lab, float(wide[c] - base[c]), net)
                     for c, lab, _short, net in PAYBACK]
            print("  Payback, term by term: "
                  + ", ".join(f"{lab} {v:+.1f}" for lab, v, _n in terms) + " ms.")
            net_sum = sum(v for _l, v, n in terms if n)
            print(f"  Of that, {net_sum:+.1f} ms is congestion (the incast "
                  f"term); the rest is structural — the overlap the prefill no "
                  f"longer provides. A faster prefill does not make the fabric "
                  f"worse, it stops hiding it.")

        # What DOES separate the fan-ins monotonically is not a time but an
        # occupancy: the queue that builds at the convergence point, and hence
        # the buffer the switch needs before it starts pausing and then dropping.
        thr = [("dropped_packets", "loss-free from"),
               (PAUSE_KNEE_COL, "PAUSE-free from")]
        print(f"\n---- what the fan-in costs in BUFFER, not in time ----")
        print(f"  {'fan-in':>6}  {'peak queue':>12}  {'= delay of':>11}  "
              + "  ".join(f"{n:>16}" for _c, n in thr))
        for L in levels:
            g = L.summary.sort_values("buffer_mb")
            qp = g["bn_qpeak_mb"].iloc[-1] if "bn_qpeak_mb" in g else NAN
            cells = []
            for col, _n in thr:
                ok = g[g[col].fillna(1) <= 0]["buffer_mb"]
                cells.append(f"{ok.min():g} MiB" if len(ok) else "never")
            # peak queue / link rate: the standing queue in the unit the
            # handover is measured in, i.e. what it adds to a packet's transit.
            dly = (qp * 2**20 * 8e3 / (g["kv_link_rate_gbps"].iloc[-1] * 1e9)
                   if pd.notna(qp) else NAN)
            print(f"  {L.degree:>6}  {qp:9.3f} MiB  {dly:8.3f} ms  "
                  + "  ".join(f"{c:>16}" for c in cells))
        print("  Peak queue is at each topology's own bottleneck link, at the "
              "largest buffer (no PAUSE, no loss),")
        print("  so it is the queue the incast BUILDS, not the queue a small "
              "buffer truncated.")

        # The same fact as figure 10, as numbers: what the buffer moves by
        # orders of magnitude next to what it moves by fractions of a percent.
        # span % = (max - min) / mean over the topology's whole buffer sweep.
        print("\n---- fabric state vs application time, over each topology's "
              "buffer sweep ----")
        print(f"  {'topology':<16} {'PAUSE frames':>14} {'dropped pkts':>14}"
              f" {'tok2-tok1 span':>15} {'makespan span':>14}"
              f" {'KV window span':>15}")
        for L in levels:
            g = L.summary
            def rng(col):
                v = g[col].dropna()
                return (f"{int(v.max())} → {int(v.min())}" if len(v) else "?")
            def span(col):
                v = g[col].dropna()
                return (f"{100 * (v.max() - v.min()) / v.mean():.2f} %"
                        if len(v) > 1 and v.mean() else "?")
            print(f"  {L.label:<16} {rng('total_pause_frames'):>14} "
                  f"{rng('dropped_packets'):>14} "
                  f"{span('tok2_after_tok1_ms'):>15} "
                  f"{span('total_exec_ms'):>14} {span('kv_window_ms'):>15}")

        # prefill/decode split verdict, per the user's explicit ask
        bad = s[~s["split_ok"]] if "split_ok" in s.columns else s.iloc[0:0]
        if len(bad):
            print(f"\n! prefill/decode split check FAILED on "
                  f"{sorted(set(bad['tag']))} — see warnings.")
        else:
            print("\nprefill/decode split: OK on every analysed run "
                  "(KV flows classified, no 'other', across all topologies).")

        # packet-loss verdict: which runs stopped being lossless, and by how much
        if "lossy" in s.columns:
            lossy = s[s["lossy"] == True]
            unknown = (s[~s["loss_captured"]] if "loss_captured" in s.columns
                       else s.iloc[0:0])
            if len(lossy):
                tot = int(lossy["dropped_packets"].sum())
                print(f"\n! PACKET LOSS (NOT lossless) on {len(lossy)} run(s), "
                      f"{tot} dropped pkt total — flagged RED in the figures:")
                for _, rr in lossy.sort_values("dropped_packets",
                                               ascending=False).iterrows():
                    print(f"    {rr['level']} buf{rr['buffer_mb']:g}: "
                          f"{int(rr['dropped_packets'])} pkt "
                          f"({rr['drop_rate_pct']:.3g}% of delivered)")
            else:
                print("\npacket loss: none on any run with a captured drops.txt "
                      "(fabric lossless).")
            if len(unknown):
                print(f"  (loss UNKNOWN — no drops.txt — on "
                      f"{sorted(set(unknown['tag']))}; re-run to record.)")
        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", *[q.name for q in written]]:
            print(f"  {fpath}")
        return drain_warnings()
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
