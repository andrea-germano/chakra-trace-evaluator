#!/usr/bin/env python3
"""
cc_sweep — the congestion-control comparison on the disaggregated-inference
fabric: same topologies and KV-cache incast as incast_sweep (T3/T4 = prefill
TP4/TP8 converging on TP2 decode pools), but the swept knob is the CC ALGORITHM
(dcqcn / hpcc / hpcc-pint / timely), with the per-switch buffer as a
small secondary axis (8/16/32 MiB) that separates the PFC-storm regime from the
comfortable one.

Per-CC parameters are the HPCC SIGCOMM'19 artifact set at 100G (vwin variants);
see configs/astra_sim/ns3/documentation/cc_sweep_parameters.md. The
manifest.json of each run carries param_set/cc, and this analyzer cross-checks
the manifest's cc against the tag so a stale config cannot silently masquerade
as another algorithm.

Reused verbatim from incast_sweep (one definition of a metric): the whole
per-run measurement (`analyse` -> Row: TTFT, makespan, intra-stage KV skew,
PFC census, drops, busiest switches), the placement recovery and the loss
marking.

What this analyzer adds on top of that measurement
--------------------------------------------------------------------------------
A per-run table answers "what happened"; a CC comparison has to answer "which
algorithm, on what, by how much, and at what cost" -- so the numbers are turned
into an argument here rather than left to the reader:

  * the KV-flow FCT DISTRIBUTION (mean/p50/p90/p99/max, tail ratio p99/p50,
    p99 slowdown) -- the metric CC papers compare on. The FCTs come from the
    ASTRA CSV KV send rows the per-run measurement already read (bit-identical
    to the ns-3 fct of the same flows); only the slowdown stays fct.txt-sourced,
    because standalone_fct exists nowhere else;
  * the QUEUEING a CC leaves behind: peak and time-mean switch occupancy, and
    the peak as a % of the configured buffer (i.e. whether the buffer is even
    a binding constraint), plus PFC frames, how many switches paused and for
    how long. A CC that wins on time while parking megabytes in the switches
    has not won the same thing;
  * a makespan DECOMPOSITION into TTFT (prefill, incl. its TP collectives) and
    the post-TTFT tail, because on this workload the CC spread is largely paid
    in prefill -- a fact invisible in a makespan-only table;
  * per-level RANKINGS with the delta to the best CC, per-CC BUFFER SENSITIVITY
    (including runs that are bit-identical across buffers, i.e. buffer-inert),
    and auto-derived FINDINGS (spread, who wins what, where the two orderings
    disagree, the smallest PFC-free/lossless buffer, the queue-vs-time cost);
  * a COVERAGE table over the full (cc x buffer) grid taken from the configs,
    so a cell that is missing, incomplete, lossy or known-unobtainable is
    reported as such instead of quietly leaving the summary one row short;
  * an ECN-CONFOUND check: could the spread be an artifact of the ECN marking
    being too restrictive, rather than of the algorithms? Answered structurally
    (which CCs even read the ECN maps) and empirically (goodput, queue depth,
    prefill neutrality) -- see ecn_confound_findings.

Output
--------------------------------------------------------------------------------
    <out>/01_pfc_frames_vs_buffer.png      PFC PAUSE frames, panel/topo, line/CC
    <out>/02_kv_fct_p99_vs_buffer.png      p99 KV-flow FCT, panel/topo, line/CC
    <out>/03_kv_arrival_skew_vs_buffer.png worst intra-stage KV skew, line/CC
    <out>/04_makespan_vs_buffer.png        makespan (y fitted), panel/topo, line/CC
    <out>/05_ttft_vs_buffer.png            TTFT, panel/topo, line/CC
    <out>/06_kv_fct_cdf.png                KV-flow FCT distribution, line/CC
    <out>/07_queue_occupancy_vs_buffer.png peak & mean switch occupancy, line/CC
    <out>/08_tradeoff_queue_vs_time.png    queueing vs time, point/(CC,buffer)
    <out>/09_cc_profile.png                every metric normalised to the best CC
    <out>/summary.csv                      one row per analysed run
    <out>/coverage.csv                     one row per EXPECTED (cc, buffer) cell
    <out>/report.md                        the console report, verbatim

Usage
-----
    python3 cc_sweep.py                    # analyse what is on disk, skip the rest
    python3 cc_sweep.py --levels T4
    python3 cc_sweep.py --ref-buffer 16    # rank/compare at 16 MiB instead of the
                                           # largest buffer common to every CC
    python3 cc_sweep.py --workload llama2_13b_16reqs_512prompt_incast_sweep \\
                        --sweep incast_sweep      # smoke-test on the incast runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import incast, intervals, roles
from utils.cli import Abort, drain_warnings, need, warn
from utils.paths import BUFFER_AXIS, fresh_dir
from utils.plots import (MS, logx_pow2, loss_proxies, mark_lossy,
                         save_fig, zoom_y)

from incast_sweep import Row, analyse, canonical_links, recover_placement

NAN = float("nan")
MIB = 1 / 2**20
KB = 1 / 1e3

CONFIG_SWEEP = "cc_sweep"
OUT_WORKLOAD = "llama2_13b_16reqs_512prompt_cc_sweep"

# tag = "<level>_bx<rate>_<cc>_buf<n>"; the cc token is everything between the
# bx and buf tokens (may contain '-', e.g. hpcc-pint, but never '_').
_CC = re.compile(r"_bx[^_]+_(.+?)_buf\d", re.IGNORECASE)

# Fixed CC identity: same colour in every figure. hpcc-pint is orange and
# dashed, NOT a second green: it sits within a few percent of hpcc on most
# metrics, and two adjacent greens made the pair unreadable in every line plot.
CC_ORDER = ["dcqcn", "hpcc", "timely", "hpcc-pint"]
CC_STYLE = {
    "dcqcn":     dict(color="#1f77b4", ls="-"),
    "hpcc":      dict(color="#2b8a3e", ls="-"),
    "timely":    dict(color="#6a4c93", ls="-"),
    "hpcc-pint": dict(color="#e69f00", ls="--"),
}
BUFFER_MARKER = {8: "o", 16: "s", 32: "^"}

# The feedback signal each algorithm's rate control consumes. This is what makes
# the ECN-confound question answerable structurally: only the ECN-driven CCs can
# be affected by the KMIN/KMAX/PMAX maps at all.
CC_SIGNAL = {
    "dcqcn": "ECN (RED ramp)", "hpcc": "INT",
    "hpcc-pint": "PINT", "timely": "RTT", "none": "window only",
}
ECN_DRIVEN = {"dcqcn"}

# Cells that are known NOT to be obtainable, and why. Without this an absent
# output directory is indistinguishable from a run that simply has not been
# launched yet, and the coverage table would report "not run" for a point that
# will never exist. Verified, reproducible facts only -- each entry names its
# evidence, and each is itself a RESULT of the sweep, not a defect of the setup.
KNOWN_UNOBTAINABLE = {
    "T4_bx100_dcqcn_buf8":
        "ns-3 livelock. DCQCN at 8 MiB on T4 drops ~418k packets, the stalled "
        "ranks' collectives never complete, ASTRA-sim therefore never calls "
        "Simulator::Stop(), and monitor_buffer reschedules unconditionally -- the "
        "process spins at 100% CPU forever. Reproduced twice with byte-identical "
        "output (freezes at tick 202,067,077, 7/20 ranks done, 417,967 drop "
        "lines). The other three CCs complete at buf8 on T4 with ZERO drops, so "
        "this is DCQCN collapsing, i.e. a result of the sweep.",
}

# The headline metrics, in report order: (column, label, lower_is_better, format).
# One definition, used by the ranking tables, the sensitivity table, the profile
# figure and the findings, so they cannot disagree about what "better" means.
@dataclass(frozen=True)
class Metric:
    col: str
    label: str
    lower_better: bool = True
    fmt: str = "{:.2f}"
    unit: str = ""


METRICS = [
    Metric("ttft_ms", "TTFT", True, "{:.2f}", "ms"),
    Metric("tail_ms", "post-TTFT tail", True, "{:.2f}", "ms"),
    Metric("total_exec_ms", "makespan", True, "{:.2f}", "ms"),
    Metric("kv_fct_p50_ms", "KV FCT p50", True, "{:.2f}", "ms"),
    Metric("kv_fct_p99_ms", "KV FCT p99", True, "{:.2f}", "ms"),
    Metric("kv_fct_tail_ratio", "KV FCT p99/p50", True, "{:.3f}", ""),
    Metric("kv_slowdown_p99", "KV slowdown p99", True, "{:.1f}", "x"),
    Metric("q_peak_mb", "peak switch queue", True, "{:.2f}", "MiB"),
    Metric("q_mean_kb", "mean switch queue", True, "{:.1f}", "kB"),
    Metric("total_pause_frames", "PFC frames", True, "{:.0f}", ""),
    Metric("dropped_packets", "dropped packets", True, "{:.0f}", ""),
]
# What the normalised profile figure shows, split into its two families. Ratios
# only make sense for metrics that are strictly positive for every CC (PFC frames
# and drops are zero almost everywhere -- they get figure 01 and the coverage
# table instead), and the two families are drawn on separate axes because they
# differ in dynamic range by an order of magnitude: the completion times land
# within ~15% of each other, the queue occupancies within a factor of 30. One
# shared axis makes whichever family is flatter unreadable.
PROFILE_GROUPS = [
    ("Time (linear, zoomed)",
     ["ttft_ms", "tail_ms", "total_exec_ms", "kv_fct_p50_ms", "kv_fct_p99_ms",
      "kv_fct_tail_ratio"], "linear"),
    ("Queueing (log)", ["q_peak_mb", "q_mean_kb"], "log"),
]


def cc_of(tag: str) -> str | None:
    m = _CC.search(tag)
    return m.group(1).lower() if m else None


def _style(cc: str) -> dict:
    return CC_STYLE.get(cc, dict(color="#444444", ls="-"))


def _cc_sort_key(cc: str) -> tuple:
    return (CC_ORDER.index(cc) if cc in CC_ORDER else len(CC_ORDER), cc)


# --------------------------------------------------------------------------- #
# The report: one text, printed to the console and written as report.md
# --------------------------------------------------------------------------- #
class Report:
    """Everything the analyzer concludes, built once and rendered twice.

    The console rendering is what the user reads while it runs; report.md is the
    same content, kept next to the figures so a result can be quoted later
    without re-running the sweep. Building the two from one block list is the
    point: a finding that exists in only one of them is a finding that will
    eventually disagree with itself. Tables go into fenced blocks rather than
    Markdown pipe tables -- pandas already aligns them, and the alignment is
    what makes a 5x3 sweep grid readable."""

    def __init__(self) -> None:
        self.blocks: list[tuple] = []

    def h(self, level: int, text: str) -> None:
        self.blocks.append(("h", level, text))

    def p(self, text: str = "") -> None:
        self.blocks.append(("p", 0, text))

    def bullet(self, text: str) -> None:
        self.blocks.append(("b", 0, text))

    def table(self, df: pd.DataFrame, index: bool = False) -> None:
        self.blocks.append(("t", 0, df.to_string(index=index)))

    def pre(self, text: str) -> None:
        self.blocks.append(("t", 0, text))

    def console(self) -> str:
        out: list[str] = []
        for kind, lvl, text in self.blocks:
            if kind == "h":
                bar = {1: "=", 2: "-", 3: " "}.get(lvl, " ")
                if lvl == 1:
                    out += ["", f"{bar * 16} {text} {bar * 16}"]
                elif lvl == 2:
                    out += ["", f"--- {text} ---"]
                else:
                    out += ["", f"  {text}"]
            elif kind == "p":
                out.append(text)
            elif kind == "b":
                out.append(f"  * {text}")
            else:
                out += ["", *(f"  {l}" for l in text.splitlines()), ""]
        return "\n".join(out)

    def markdown(self, title: str) -> str:
        out = [f"# {title}", ""]
        for kind, lvl, text in self.blocks:
            if kind == "h":
                out += ["", f"{'#' * (lvl + 1)} {text}", ""]
            elif kind == "p":
                out += [text, ""]
            elif kind == "b":
                out.append(f"- {text}")
            else:
                out += ["", "```text", text, "```", ""]
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CC-specific per-run extras
# --------------------------------------------------------------------------- #
def read_manifest(p: incast.IncastPaths, tag: str) -> dict:
    mpath = p.config_root / tag / "manifest.json"
    if not mpath.is_file():
        warn(f"{tag}: no manifest.json next to config.txt; cc taken from the "
             f"tag alone.")
        return {}
    try:
        return json.loads(mpath.read_text())
    except Exception as e:                                        # noqa: BLE001
        warn(f"{tag}: unreadable manifest.json ({e}); cc taken from the tag.")
        return {}


def kv_fct_stats(row: Row) -> dict:
    """The FCT distribution of the KV flows -- the metric CC papers compare on --
    computed from the raw material incast_sweep.analyse already keeps on the Row:
    kv_fct_ns are the ASTRA CSV KV send durations (bit-identical to the ns-3 fct
    of the same flows, verified to the nanosecond, and already classified by
    name), kv_slowdown is fct/standalone_fct from fct.txt (the standalone
    reference exists only there). No file is re-read here.

    slowdown = fct / standalone_fct (see utils.ns3.read_fct): ~concurrency when
    the bottleneck is shared fairly, worse when the CC lets queues build. The
    tail ratio p99/p50 is the same statement without the standalone reference:
    how much longer the unluckiest KV shard takes than the median one, which is
    what a decode stage's TP barrier actually waits for.

    The tail ratio is here INSTEAD of a Jain fairness index over per-flow
    goodput, which was tried and is not usable on this workload: every KV flow is
    the same size (40 MiB), but ~5% of them run after the incast has drained and
    finish at 3-6x the contended rate (visible as the low tail of figure 06's
    CDF). Jain reads that as gross unfairness -- it scored the CC with the BEST
    p50 and p99 the worst -- because it cannot tell "starved some senders" from
    "finished early enough that the stragglers got the fabric to themselves".
    A percentile ratio is immune to those few points."""
    fct = np.asarray(row.kv_fct_ns if row.kv_fct_ns is not None else [],
                     dtype=float)
    if not len(fct):
        warn(f"{row.tag}: no KV send in the ASTRA trace; FCT stats unavailable.")
        return {}
    p50, p99 = np.percentile(fct, 50), np.percentile(fct, 99)
    out = {
        "kv_fct_mean_ns": float(fct.mean()),
        "kv_fct_p50_ns": float(p50),
        "kv_fct_p90_ns": float(np.percentile(fct, 90)),
        "kv_fct_p99_ns": float(p99),
        "kv_fct_max_ns": float(fct.max()),
        "kv_fct_tail_ratio": float(p99 / p50) if p50 > 0 else NAN,
    }
    sl = np.asarray(row.kv_slowdown if row.kv_slowdown is not None else [],
                    dtype=float)
    if len(sl):
        out["kv_slowdown_mean"] = float(sl.mean())
        out["kv_slowdown_p50"] = float(np.percentile(sl, 50))
        out["kv_slowdown_p99"] = float(np.percentile(sl, 99))
    return out


def fabric_extras(row: Row) -> dict:
    """The queueing a CC left in the fabric, reduced from the per-switch censuses
    incast_sweep.analyse already collected (no extra file is read).

    Peak occupancy is the max over switches of that switch's peak TOTAL egress
    bytes -- comparable to BUFFER_SIZE, which ns-3 applies per switch, so
    q_peak_pct says whether the buffer was a binding constraint at all or just
    headroom nobody touched. The mean is the time-average over the whole
    monitored run (qlen.txt samples every 100 ns, idle switches included), i.e.
    the STANDING queue the CC keeps: the latency a small flow crossing the fabric
    would have paid at a random instant. Peak answers "did it fit", mean answers
    "what did it cost everyone else".

    Paused time is the union of a switch's PAUSE spans (never their sum: spans of
    different queues on one switch overlap), taken for the worst switch."""
    peaks = [v for v in row.qswitch_peak.values() if pd.notna(v)]
    means = [v for v in row.qswitch_mean.values() if pd.notna(v)]
    out = {
        "q_peak_mb": max(peaks) * MIB if peaks else NAN,
        "q_mean_kb": float(np.mean(means)) * KB if means else NAN,
        "q_peak_switch": max(row.qswitch_peak, key=row.qswitch_peak.get)
                         if row.qswitch_peak else -1,
        "pfc_switches": sum(1 for v in row.pfc_per_switch.values() if v > 0),
        "drop_switches": sum(1 for v in row.dropped_per_switch.values() if v > 0),
    }
    if peaks and row.buffer_bytes:
        out["q_peak_pct_of_buffer"] = 100.0 * max(peaks) / row.buffer_bytes
    pausetime = [intervals.union_len(sp) for sp in row.pause_intervals.values()]
    out["pause_ms_worst_switch"] = max(pausetime) * MS if pausetime else 0.0
    if pd.notna(row.total_exec_ns) and row.total_exec_ns > 0:
        out["pause_pct_of_run"] = 100.0 * (max(pausetime) if pausetime else 0.0) \
                                  / row.total_exec_ns
    return out


@dataclass
class CcRun:
    cc: str
    param_set: str
    row: Row                       # incast_sweep's full per-run measurement
    manifest: dict = field(default_factory=dict)
    # -- KV flow-completion distribution (kv_fct_stats, from row.kv_fct_ns) --- #
    kv_fct_mean_ns: float = NAN
    kv_fct_p50_ns: float = NAN
    kv_fct_p90_ns: float = NAN
    kv_fct_p99_ns: float = NAN
    kv_fct_max_ns: float = NAN
    kv_fct_tail_ratio: float = NAN
    kv_slowdown_mean: float = NAN
    kv_slowdown_p50: float = NAN
    kv_slowdown_p99: float = NAN
    # -- queueing / backpressure --------------------------------------------- #
    q_peak_mb: float = NAN
    q_mean_kb: float = NAN
    q_peak_pct_of_buffer: float = NAN
    q_peak_switch: int = -1
    pfc_switches: int = 0
    drop_switches: int = 0
    pause_ms_worst_switch: float = NAN
    pause_pct_of_run: float = NAN
    # -- coverage ------------------------------------------------------------ #
    complete: bool = True          # False = flow count differs from sibling runs

    @property
    def kv_fct_ns(self) -> np.ndarray:
        """Per-flow KV FCTs for the CDF figure, straight from the Row's raw
        material (no copy kept here)."""
        return (self.row.kv_fct_ns if self.row.kv_fct_ns is not None
                else np.array([]))

    def flat(self) -> dict:
        d = self.row.flat()
        d["cc"] = self.cc
        d["param_set"] = self.param_set
        d["kv_fct_mean_ms"] = self.kv_fct_mean_ns * MS
        d["kv_fct_p50_ms"] = self.kv_fct_p50_ns * MS
        d["kv_fct_p90_ms"] = self.kv_fct_p90_ns * MS
        d["kv_fct_p99_ms"] = self.kv_fct_p99_ns * MS
        d["kv_fct_max_ms"] = self.kv_fct_max_ns * MS
        d["kv_fct_tail_ratio"] = self.kv_fct_tail_ratio
        d["kv_slowdown_mean"] = self.kv_slowdown_mean
        d["kv_slowdown_p50"] = self.kv_slowdown_p50
        d["kv_slowdown_p99"] = self.kv_slowdown_p99
        d["kv_bytes_mb"] = self.row.kv_bytes * MIB   # kv_goodput_gbps already
                                                     # rides in from row.flat()
        d["tail_ms"] = d.get("total_minus_ttft_ms", NAN)
        d["q_peak_mb"] = self.q_peak_mb
        d["q_mean_kb"] = self.q_mean_kb
        d["q_peak_pct_of_buffer"] = self.q_peak_pct_of_buffer
        d["q_peak_switch"] = self.q_peak_switch
        d["pfc_switches"] = self.pfc_switches
        d["drop_switches"] = self.drop_switches
        d["pause_ms_worst_switch"] = self.pause_ms_worst_switch
        d["pause_pct_of_run"] = self.pause_pct_of_run
        d["complete"] = self.complete
        return d


# --------------------------------------------------------------------------- #
# Per-topology analysis
# --------------------------------------------------------------------------- #
@dataclass
class Level:
    level: str
    degree: int
    runs: list                     # list[CcRun], sorted (cc, buffer)
    label: str
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


def _expected_grid(p: incast.IncastPaths) -> list[tuple[str, str, float]]:
    """(tag, cc, buffer) for every cell the CONFIGS declare for this level.

    The expected grid comes from configs/, not from output/: a cell whose output
    is missing is exactly the case this exists to report, and deriving the grid
    from the outputs would define the missing cell out of existence."""
    out = []
    for tag in p.tags("config"):
        cc, buf = cc_of(tag), BUFFER_AXIS.value(tag)
        if cc is not None and buf is not None:
            out.append((tag, cc, float(buf)))
    return sorted(out, key=lambda x: (_cc_sort_key(x[1]), x[2]))


def _coverage(p: incast.IncastPaths, runs: list[CcRun], skipped: dict[str, str],
              failed: dict[str, str]) -> pd.DataFrame:
    """One row per EXPECTED (cc, buffer) cell with what became of it. Status is
    the single vocabulary the coverage matrix, coverage.csv and the findings all
    speak: ok / loss:<n> / short / incomplete / failed / UNOBTAINABLE / not run."""
    by_tag = {r.row.tag: r for r in runs}
    rows = []
    for tag, cc, buf in _expected_grid(p):
        r = by_tag.get(tag)
        if r is not None:
            if not r.complete:
                status, note = "short", "flow count differs from sibling runs"
            elif r.row.lossy:
                status = f"loss:{r.row.dropped_packets:.0f}"
                note = (f"{r.row.dropped_packets:.0f} pkt dropped on "
                        f"{r.drop_switches} switch(es)")
            else:
                status, note = "ok", ""
        elif tag in failed:
            status, note = "failed", failed[tag]
        elif tag in skipped:
            status, note = "incomplete", skipped[tag]
        elif tag in KNOWN_UNOBTAINABLE:
            status, note = "UNOBTAINABLE", KNOWN_UNOBTAINABLE[tag]
        else:
            status, note = "not run", "no ns-3 output directory"
        rows.append({"level": p.level, "cc": cc, "buffer_mb": buf, "tag": tag,
                     "status": status, "note": note})
    return pd.DataFrame(rows)


def coverage_matrix(cov: pd.DataFrame) -> pd.DataFrame:
    """The coverage rows pivoted to the shape the sweep was launched in: one row
    per CC, one column per buffer. A grid is read as a grid."""
    m = cov.pivot_table(index="cc", columns="buffer_mb", values="status",
                        aggfunc="first")
    m = m.reindex(sorted(m.index, key=_cc_sort_key))
    m.columns = [f"buf{c:g}" for c in m.columns]
    return m.fillna("-").reset_index()


def analyse_level(level: str, root: Path, out_workload: str,
                  config_sweep: str) -> Level | None:
    p = incast.IncastPaths(level=level, out_workload=out_workload,
                           config_sweep=config_sweep, root=root)
    missing_roots = p.missing_roots()
    # config root/inputs are what the analysis reasons FROM: Abort when missing.
    # Output roots/files mean the runs have not finished: skip and name them.
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
    tags, skipped_msgs = p.usable_tags()
    skipped = {m.split(":", 1)[0]: m.split(": ", 1)[-1] for m in skipped_msgs}
    for s in skipped_msgs:
        warn(f"{level}: {s} (run not finished yet?) -- skipped.")
    if not tags:
        warn(f"{level}: no run has all its outputs on disk yet; skipped.")
        return None

    placement = recover_placement(p, tags)
    degree = incast.prefill_tp(placement)
    # The KV-crossed link set, fixed once for the level (incast_sweep's rule):
    # re-ranking it per run would make the bn_* columns describe a different
    # physical link at different CCs, which is exactly the comparison this sweep
    # must not make.
    links = canonical_links(p, tags, placement, top_links=1)
    print(f"\n===== {level}  (prefill TP{degree}, bottleneck {links[0]}) =====")
    print(f"  placement {roles.spec_of(placement)}")

    runs: list[CcRun] = []
    failed: dict[str, str] = {}
    for tag in tags:
        cc = cc_of(tag)
        if cc is None:
            warn(f"{level}: {tag} has no cc token; skipped.")
            continue
        man = read_manifest(p, tag)
        if man.get("cc") and man["cc"].lower() != cc:
            warn(f"{tag}: manifest says cc={man['cc']} but the tag says {cc}; "
                 f"STALE CONFIG? Trusting the manifest.")
            cc = man["cc"].lower()
        try:
            row = analyse(tag, p, placement, links)
        except Abort as e:
            warn(f"{level}: run {tag} dropped -- {e}")
            failed[tag] = str(e)
            continue
        run = CcRun(cc=cc, param_set=man.get("param_set", ""), row=row,
                    manifest=man)
        for k, v in kv_fct_stats(row).items():
            setattr(run, k, v)
        for k, v in fabric_extras(row).items():
            setattr(run, k, v)
        runs.append(run)
    if not runs:
        warn(f"{level}: every run failed to analyse; skipped.")
        return None
    runs.sort(key=lambda r: (_cc_sort_key(r.cc), r.row.buffer_mb))

    # Completeness check: every run of a level replays the same workload, so
    # the flow count must match across CCs/buffers. A short count means the
    # simulation was interrupted (fct.txt truncated); a LONGER one means the
    # modal group itself is suspect. Either way the flow sets differ and the
    # metrics are not comparable, so any deviation from the modal count flags.
    totals = [r.row.kv_flows + r.row.other_flows for r in runs]
    modal = max(set(totals), key=totals.count)
    for r in runs:
        t = r.row.kv_flows + r.row.other_flows
        if t != modal:
            r.complete = False
            warn(f"{r.row.tag}: {t} flows vs {modal} in sibling runs — "
                 f"run looks {'INCOMPLETE (interrupted?)' if t < modal else 'anomalous (extra flows?)'}; "
                 f"its metrics are not trustworthy, re-run it.")

    for r in runs:
        flag = "" if r.row.split_ok else "  ! split check FAILED"
        if r.row.lossy:
            flag += f"  ** LOSS: {r.row.dropped_packets:.0f} pkt **"
        print(f"  + {r.cc:<9} buf{r.row.buffer_mb:<4g} "
              f"ttft={r.row.ttft_ns * MS:6.1f}ms  "
              f"total={r.row.total_exec_ns * MS:6.1f}ms  "
              f"kv_p99={r.kv_fct_p99_ns * MS:6.2f}ms  "
              f"qpeak={r.q_peak_mb:5.2f}MiB  "
              f"pfc={r.row.total_pause_frames:6.0f}{flag}")
    cov = _coverage(p, runs, skipped, failed)
    return Level(level=level, degree=degree, runs=runs,
                 label=f"{level} (tp{degree})", coverage=cov)


# --------------------------------------------------------------------------- #
# Comparison tables
# --------------------------------------------------------------------------- #
def _fmt_delta(v: float, best: float, m: Metric) -> str:
    """"<value> (<delta vs best>)", the two numbers a CC comparison is made of.
    The best CC is starred rather than shown as "+0.0%", so the eye finds it
    without reading every row."""
    if pd.isna(v):
        return "—"
    txt = m.fmt.format(v)
    if pd.isna(best):
        return txt
    if v == best:
        return f"{txt} *"
    if best == 0:                 # % of zero is undefined; PFC frames and drops
        return f"{txt} (+{m.fmt.format(v - best)})"   # compare in absolute terms
    d = 100.0 * (v - best) / abs(best)
    return f"{txt} ({d:+.1f}%)"


def ref_buffer(g: pd.DataFrame, forced: float | None) -> float | None:
    """The buffer at which the CCs are compared head-to-head: by default the
    LARGEST buffer every CC of this level actually has a run at.

    Largest, not smallest, on purpose. The small-buffer end is where a CC either
    survives or collapses -- that comparison is the coverage table and figure 01,
    and it is not a like-for-like ranking (a CC missing there is missing BECAUSE
    it collapsed). The large-buffer end is where every CC is in its comfortable
    regime, so a difference there is the algorithm's own cost, not the buffer's."""
    if g.empty:
        return None
    if forced is not None:
        if forced in set(g["buffer_mb"]):
            return float(forced)
        warn(f"--ref-buffer {forced:g} has no run on {g['level'].iloc[0]}; "
             f"falling back to the largest common buffer.")
    per_cc = g.groupby("cc", observed=True)["buffer_mb"].apply(set)
    common = set.intersection(*per_cc) if len(per_cc) else set()
    if not common:                      # no buffer common to all: take the widest row
        return float(max(g["buffer_mb"]))
    return float(max(common))


def ranking_table(g: pd.DataFrame, buf: float) -> pd.DataFrame:
    """Every headline metric for every CC at one buffer, each cell carrying its
    delta to the best CC on that metric. This is the table the whole sweep is
    for: it reads across (what does this CC cost) and down (who wins this
    metric)."""
    gb = g[g["buffer_mb"] == buf]
    rows = []
    for cc in sorted(gb["cc"].unique(), key=_cc_sort_key):
        r = gb[gb["cc"] == cc].iloc[0]
        out = {"cc": cc}
        for m in METRICS:
            if m.col not in gb.columns:
                continue
            col = gb[m.col].dropna()
            best = (col.min() if m.lower_better else col.max()) if len(col) else NAN
            head = f"{m.label} ({m.unit})" if m.unit else m.label
            out[head] = _fmt_delta(r.get(m.col, NAN), best, m)
        rows.append(out)
    return pd.DataFrame(rows)


def sensitivity_table(g: pd.DataFrame) -> pd.DataFrame:
    """How much the per-switch buffer moves each CC, and where it stops mattering.

    'inert' means the runs at different buffers came out BIT-IDENTICAL on every
    headline metric: the simulation is deterministic, so that is not noise -- it
    is proof that the queues never reached the smaller buffer's limit and the
    buffer axis is doing nothing for that CC. Reporting it as "0.0% variation"
    would leave the reader wondering whether the effect was merely small.

    The middle verdict matters as much: a CC whose TIMES move by less than a
    percent while its PFC census does not is being changed by the buffer in the
    fabric and not in the workload -- worth knowing before quoting a makespan
    from one buffer as if it were the CC's number."""
    watch = ["ttft_ms", "total_exec_ms", "kv_fct_p99_ms", "q_peak_mb"]
    rows = []
    for cc in sorted(g["cc"].unique(), key=_cc_sort_key):
        gc = g[g["cc"] == cc].sort_values("buffer_mb")
        out = {"cc": cc, "buffers": ",".join(f"{b:g}" for b in gc["buffer_mb"])}
        inert, worst_time = True, 0.0
        for col in watch:
            v = gc[col].dropna()
            name = f"Δ{col.replace('_ms', '').replace('_mb', '')}"
            if len(v) < 2:
                out[name] = "—"
                inert = False
                continue
            rng = (v.max() - v.min()) / abs(v.min()) * 100 if v.min() else NAN
            out[name] = f"{rng:.2f}%"
            if col != "q_peak_mb" and pd.notna(rng):
                worst_time = max(worst_time, rng)
            if v.nunique() > 1:
                inert = False
        clean = gc[(gc["total_pause_frames"].fillna(0) == 0)
                   & (gc["dropped_packets"].fillna(0) == 0)]
        out["min PFC-free buf"] = (f"{clean['buffer_mb'].min():g}"
                                   if len(clean) else "none of the swept buffers")
        if inert:
            out["verdict"] = "buffer-inert (bit-identical)"
        elif worst_time < 1.0:
            out["verdict"] = f"times flat (<1%), fabric moves"
        else:
            out["verdict"] = f"buffer-sensitive (up to {worst_time:.1f}%)"
        rows.append(out)
    return pd.DataFrame(rows)


def worst_cell_table(s: pd.DataFrame) -> pd.DataFrame:
    """The runs where the fabric actually hurt: any run with PFC, loss, or a peak
    queue above a quarter of its buffer. A sweep whose interesting rows are 4 out
    of 29 should say which 4 rather than print 29 and hope."""
    if s.empty:
        return pd.DataFrame()
    hot = s[(s["total_pause_frames"].fillna(0) > 0)
            | (s["dropped_packets"].fillna(0) > 0)
            | (s["q_peak_pct_of_buffer"].fillna(0) > 25)]
    cols = ["level", "cc", "buffer_mb", "total_pause_frames", "pfc_switches",
            "pause_ms_worst_switch", "dropped_packets", "q_peak_mb",
            "q_peak_pct_of_buffer", "kv_fct_p99_ms", "total_exec_ms"]
    return hot[[c for c in cols if c in hot.columns]].sort_values(
        ["level", "total_pause_frames"], ascending=[True, False])


def _ecn_at_fabric(man: dict, key: str) -> float | None:
    """The KMIN/KMAX/PMAX value that applies to the FABRIC links, pulled out of
    the manifest's per-rate map ("<n> <rate_bps> <val> <rate_bps> <val> ...").
    The scale-up entries (1024G/4800G, effectively non-marking) are exactly the
    ones this must not return, hence the lookup by the fabric's own rate."""
    ov = man.get("cc_overrides") or {}
    m = ov.get(key)
    if not m:
        return None
    try:
        rate = float(man.get("bx_gbps")) * 1e9
    except (TypeError, ValueError):
        return None
    toks = m.split()[1:]
    for r, v in zip(toks[::2], toks[1::2]):
        if float(r) == rate:
            return float(v)
    return None


def ecn_marking_of(man: dict, cc: str) -> str:
    """One human-readable cell: the ECN marking this run's fabric links carry,
    and whether the CC even reads it."""
    kmin = _ecn_at_fabric(man, "KMIN_MAP")
    kmax = _ecn_at_fabric(man, "KMAX_MAP")
    pmax = _ecn_at_fabric(man, "PMAX_MAP")
    if kmin is None or kmax is None:
        return "template maps"
    txt = (f"step@{kmin:g}KB" if kmin == kmax
           else f"{kmin:g}/{kmax:g}KB p={pmax:g}")
    return txt if cc in ECN_DRIVEN else f"{txt} (ignored)"


def cc_parameter_table(levels: list[Level]) -> pd.DataFrame:
    """What each CC actually ran with, taken from the manifests -- so the report
    is self-contained and a parameter set cannot be mis-remembered later. The
    full ECN/threshold maps are per-rate tables and stay in the manifest; the
    one value that matters for the comparison -- the marking on the FABRIC links,
    and whether the CC reads it at all -- is extracted into its own column."""
    seen: dict[str, dict] = {}
    for L in levels:
        for r in L.runs:
            if r.cc not in seen and r.manifest:
                seen[r.cc] = r.manifest
    rows = []
    for cc in sorted(seen, key=_cc_sort_key):
        man = seen[cc]
        ov = {k: v for k, v in (man.get("cc_overrides") or {}).items()
              if not k.endswith("_MAP")}
        rows.append({"cc": cc,
                     "signal": CC_SIGNAL.get(cc, "?"),
                     "ECN @ fabric": ecn_marking_of(man, cc),
                     "cc_mode": man.get("cc_mode", "?"),
                     "param_set": man.get("param_set", "?"),
                     "overrides (maps omitted)":
                         ", ".join(f"{k}={v}" for k, v in sorted(ov.items())) or "—"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Findings: the sentences the tables above imply
# --------------------------------------------------------------------------- #
def _best_worst(g: pd.DataFrame, col: str, lower_better: bool = True):
    v = g.dropna(subset=[col])
    if v.empty:
        return None
    i_best = v[col].idxmin() if lower_better else v[col].idxmax()
    i_worst = v[col].idxmax() if lower_better else v[col].idxmin()
    return v.loc[i_best], v.loc[i_worst]


def _spread_pct(a: float, b: float) -> float:
    return 100.0 * (b - a) / a if a else NAN


def level_findings(g: pd.DataFrame, buf: float, label: str) -> list[str]:
    """The comparison, in sentences. Everything here is derived from the row set
    it is handed -- no CC is named in the source -- so the text stays true when
    the sweep gains or loses an algorithm."""
    out: list[str] = []
    gb = g[g["buffer_mb"] == buf]
    if gb.empty:
        return out
    if gb["cc"].nunique() < 2:
        # e.g. the incast sweep's single-CC data, which this analyzer can also
        # read: every "best vs worst" sentence below would compare a CC to
        # itself and read as a 0.0% spread, which is not a finding.
        out.append(f"{label} @ buf{buf:g}: only one CC on disk "
                   f"({gb['cc'].iloc[0]}) — no head-to-head comparison; the "
                   f"buffer axis below is all this level can say.")
        return out + _binding_finding(gb, buf, label)

    bw = _best_worst(gb, "total_exec_ms")
    if bw:
        b, w = bw
        out.append(f"{label} @ buf{buf:g}: makespan spread "
                   f"{_spread_pct(b['total_exec_ms'], w['total_exec_ms']):.1f}% — "
                   f"best {b['cc']} {b['total_exec_ms']:.1f} ms, worst {w['cc']} "
                   f"{w['total_exec_ms']:.1f} ms "
                   f"(+{w['total_exec_ms'] - b['total_exec_ms']:.1f} ms).")
        # Where the spread is paid: prefill (TTFT) or the post-TTFT tail. On a
        # disaggregated run these are different customers -- TTFT is the user's
        # latency, the tail is the pool's throughput -- so a makespan-only
        # ranking hides which one the CC is charging.
        if {"ttft_ms", "tail_ms"} <= set(gb.columns):
            t_rng = gb["ttft_ms"].max() - gb["ttft_ms"].min()
            k_rng = gb["tail_ms"].max() - gb["tail_ms"].min()
            if pd.notna(t_rng) and pd.notna(k_rng):
                who = ("prefill/TTFT" if t_rng > k_rng else "the post-TTFT tail")
                out.append(f"{label}: that spread is carried by {who} — TTFT "
                           f"varies by {t_rng:.1f} ms across CCs vs {k_rng:.1f} ms "
                           f"for the tail. The CC is charged mostly on the "
                           f"{'prefill TP collectives' if t_rng > k_rng else 'KV handover and decode'}.")

    bw_fct = _best_worst(gb, "kv_fct_p99_ms")
    if bw_fct:
        b, w = bw_fct
        out.append(f"{label} @ buf{buf:g}: KV FCT p99 best {b['cc']} "
                   f"{b['kv_fct_p99_ms']:.1f} ms, worst {w['cc']} "
                   f"{w['kv_fct_p99_ms']:.1f} ms "
                   f"({_spread_pct(b['kv_fct_p99_ms'], w['kv_fct_p99_ms']):.1f}% "
                   f"spread).")
    # The two rankings disagreeing is the single most useful thing this sweep can
    # say: it means "fastest fabric" and "fastest workload" are not one choice.
    if bw and bw_fct and bw[0]["cc"] != bw_fct[0]["cc"]:
        out.append(f"{label}: the makespan winner ({bw[0]['cc']}) is NOT the KV-tail "
                   f"winner ({bw_fct[0]['cc']}) — the fabric metric and the "
                   f"end-to-end metric rank differently here, so the choice "
                   f"depends on which one the deployment is serving.")

    # Queueing cost of the time winner.
    bw_q = _best_worst(gb, "q_mean_kb")
    if bw_q and bw:
        qb = bw_q[0]
        tb = bw[0]
        if qb["cc"] != tb["cc"] and pd.notna(qb["total_exec_ms"]):
            out.append(f"{label}: {qb['cc']} holds the shortest standing queues "
                       f"({qb['q_mean_kb']:.1f} kB mean vs {tb['q_mean_kb']:.1f} kB "
                       f"for {tb['cc']}) but finishes "
                       f"{_spread_pct(tb['total_exec_ms'], qb['total_exec_ms']):+.1f}% "
                       f"later — the classic queue-vs-completion trade-off.")
        elif qb["cc"] == tb["cc"]:
            out.append(f"{label}: {tb['cc']} wins on both time and standing queue "
                       f"({tb['q_mean_kb']:.1f} kB mean) — no trade-off to make "
                       f"at this buffer.")

    return out + _binding_finding(gb, buf, label)


def _binding_finding(gb: pd.DataFrame, buf: float, label: str) -> list[str]:
    """Is the swept buffer a constraint at this point, or just headroom? The
    question every "more buffer did nothing" row implicitly asks."""
    if "q_peak_pct_of_buffer" not in gb.columns or \
            not gb["q_peak_pct_of_buffer"].notna().any():
        return []
    mx = gb["q_peak_pct_of_buffer"].max()
    return [f"{label} @ buf{buf:g}: the deepest switch queue reaches {mx:.1f}% of "
            f"the configured buffer (worst CC), so at this buffer the limit is "
            f"{'binding' if mx > 60 else 'not a constraint — the buffer axis only bites lower down'}."]


def buffer_findings(g: pd.DataFrame, label: str) -> list[str]:
    """What the secondary axis did: where PFC/loss appear, and the smallest
    buffer at which the whole level is quiet."""
    out: list[str] = []
    if "total_pause_frames" not in g.columns:
        return out
    quiet = []
    for buf, gb in g.groupby("buffer_mb"):
        if (gb["total_pause_frames"].fillna(0) == 0).all() and \
           (gb["dropped_packets"].fillna(0) == 0).all():
            quiet.append(buf)
    if quiet:
        out.append(f"{label}: every analysed CC is PFC-free and lossless from "
                   f"buf{min(quiet):g} upwards.")
    noisy = g[g["total_pause_frames"].fillna(0) > 0]
    if len(noisy):
        who = ", ".join(f"{r['cc']}@buf{r['buffer_mb']:g} "
                        f"({r['total_pause_frames']:.0f} frames, "
                        f"{r['pfc_switches']:.0f} switch(es))"
                        for _, r in noisy.sort_values("total_pause_frames",
                                                      ascending=False).iterrows())
        out.append(f"{label}: PFC backpressure only on {who}.")
    else:
        out.append(f"{label}: no PFC PAUSE anywhere in the level.")
    return out


def cross_level_findings(s: pd.DataFrame, refs: dict[str, float]) -> list[str]:
    """Does the CC ordering survive a change of topology? A ranking that flips
    between T3 and T4 is a ranking about the incast degree, not about the CC."""
    out: list[str] = []
    orders: dict[str, list[str]] = {}
    for lvl, g in s.groupby("level", observed=True):
        buf = refs.get(lvl)
        gb = g[g["buffer_mb"] == buf].dropna(subset=["total_exec_ms"])
        if len(gb) >= 2:
            orders[lvl] = list(gb.sort_values("total_exec_ms")["cc"])
    if len(orders) < 2:
        return out
    for lvl, o in orders.items():
        out.append(f"{lvl} makespan order (best→worst) @ buf{refs[lvl]:g}: "
                   f"{' < '.join(map(str, o))}")
    uniq = {tuple(o) for o in orders.values()}
    if len(uniq) == 1:
        out.append("The makespan ordering is IDENTICAL across topologies — the "
                   "ranking is a property of the CC, not of the incast degree.")
    else:
        tops = {o[0] for o in orders.values()}
        out.append(f"The ordering CHANGES across topologies"
                   f"{' (though the winner ' + tops.pop() + ' is the same everywhere)' if len(tops) == 1 else ''}"
                   f" — read each level's ranking on its own, and treat any "
                   f"single-topology conclusion as topology-specific.")
    return out


def ecn_confound_findings(s: pd.DataFrame, refs: dict[str, float],
                          levels: list[Level]) -> list[str]:
    """Could the CC spread be an artifact of the ECN marking being too
    restrictive (too-early / too-aggressive), rather than of the algorithms?
    The one methodological objection this sweep invites, answered in the report
    itself so it cannot be raised against a number quoted from it.

    Over-restrictive marking has ONE empirical signature: the marked senders are
    throttled below what the fabric could carry -- shallow queues AND
    under-delivery, on the CCs that read the marking. So the check is
    structural first (which CCs consume ECN at all: the INT/PINT/RTT algorithms
    cannot be touched by KMIN/KMAX/PMAX) and then empirical on the ECN-driven
    ones (their goodput and queue depth relative to the rest, and whether the
    prefill phase -- where the spread is paid -- differentiates them at all).
    Every bullet is computed from the rows it is handed; a bullet whose evidence
    is not on disk is not emitted."""
    out: list[str] = []
    if s.empty:
        return out

    # The thresholds actually applied to the fabric links, from any ECN manifest.
    man = next((r.manifest for L in levels for r in L.runs
                if r.cc in ECN_DRIVEN and r.manifest), None)
    if man:
        kmin = _ecn_at_fabric(man, "KMIN_MAP")
        kmax = _ecn_at_fabric(man, "KMAX_MAP")
        if kmin is not None and kmax is not None:
            out.append(
                f"Thresholds on paper: the fabric links mark at "
                f"KMIN={kmin:g}KB / KMAX={kmax:g}KB (DCQCN RED ramp) — the HPCC "
                f"SIGCOMM'19 artifact values at "
                f"{man.get('bx_gbps', '?')}G, i.e. LATER-marking than the NVIDIA "
                f"100GbE RoCE reference (150/1500KB). 'Too restrictive' is not "
                f"true of the configuration itself; see "
                f"documentation/cc_sweep_parameters.md.")

    for lvl, g in s.groupby("level", observed=True):
        buf = refs.get(lvl)
        gb = g[g["buffer_mb"] == buf]
        ecn = gb[gb["cc"].isin(ECN_DRIVEN)]
        non = gb[~gb["cc"].isin(ECN_DRIVEN)]
        if ecn.empty or non.empty:
            continue
        label = f"{lvl} @ buf{buf:g}"

        # 1) Structural: is the spread carried by CCs that never read ECN?
        bw = _best_worst(gb, "ttft_ms")
        if bw and bw[1]["cc"] not in ECN_DRIVEN:
            b, w = bw
            out.append(
                f"{label}: the TTFT spread is carried by {w['cc']} "
                f"(+{_spread_pct(b['ttft_ms'], w['ttft_ms']):.1f}% vs best), an "
                f"algorithm that reacts to {CC_SIGNAL.get(w['cc'], '?')} and "
                f"never reads the ECN maps — no marking threshold can be its "
                f"cause.")

        # 2) Delivery: over-restrictive marking would throttle the marked CCs'
        #    goodput below the unmarked ones'. The opposite holding is decisive.
        if "kv_goodput_gbps" in gb.columns and gb["kv_goodput_gbps"].notna().all():
            e_hi, n_hi = ecn["kv_goodput_gbps"].max(), non["kv_goodput_gbps"].max()
            e_cc = ecn.loc[ecn["kv_goodput_gbps"].idxmax(), "cc"]
            if e_hi >= n_hi:
                out.append(
                    f"{label}: the highest KV goodput of the level belongs to an "
                    f"ECN-driven CC ({e_cc}, {e_hi:.0f} Gb/s vs {n_hi:.0f} Gb/s "
                    f"best non-ECN) — the marking is not throttling delivery.")

        # 3) Queues: throttled-by-marking means pinned-shallow queues. The
        #    ECN-driven CCs holding the deepest standing queues is the opposite
        #    signature (a slow control loop overshooting the ramp).
        if gb["q_mean_kb"].notna().all():
            deep = gb.loc[gb["q_mean_kb"].idxmax()]
            if deep["cc"] in ECN_DRIVEN:
                out.append(
                    f"{label}: the deepest standing queues are {deep['cc']}'s "
                    f"({deep['q_mean_kb']:.0f} kB mean vs "
                    f"{non['q_mean_kb'].min():.0f} kB for the best non-ECN CC) — "
                    f"with over-restrictive marking its queues would be pinned "
                    f"shallow, not the deepest of the sweep.")

        # 4) Prefill neutrality: the spread is paid in TTFT, so the marking
        #    would have to act during the prefill collectives to explain it.
        #    The ECN CCs and the RTT one agreeing to a fraction of a percent
        #    there says it does not.
        probe = gb[gb["cc"].isin(ECN_DRIVEN | {"timely"})]["ttft_ms"].dropna()
        if len(probe) >= 2:
            rel = _spread_pct(probe.min(), probe.max())
            if pd.notna(rel) and rel < 0.5:
                out.append(
                    f"{label}: TTFT agrees to {rel:.2f}% across the ECN-driven "
                    f"CCs and timely — during prefill the marking never "
                    f"differentiates them, so whatever TTFT gap the level shows "
                    f"belongs to CCs reacting to other signals, not to ECN.")
    return out


def coverage_findings(cov: pd.DataFrame) -> tuple[str, list[str]]:
    """(headline, one line per cell that is not plainly 'ok'). A cell that is
    missing, lossy, truncated or known-unobtainable is a caveat on every number
    derived from that level, so it is stated before the rankings, not after."""
    if cov.empty:
        return "Coverage: nothing declared.", []
    bad = cov[cov["status"] != "ok"]
    if bad.empty:
        return (f"Coverage: all {len(cov)} declared (cc, buffer) cells analysed "
                f"and lossless."), []
    head = (f"Coverage: {len(cov) - len(bad)}/{len(cov)} declared cells are "
            f"clean; {len(bad)} cell{'s' if len(bad) > 1 else ''} need"
            f"{'' if len(bad) > 1 else 's'} a word:")
    return head, [f"{r['level']} {r['cc']} buf{r['buffer_mb']:g} → "
                  f"{r['status']}: {r['note']}" for _, r in bad.iterrows()]


# --------------------------------------------------------------------------- #
# Figures: panel per topology, one line per CC, buffer on x
# --------------------------------------------------------------------------- #
def _cc_lines(levels: list[Level], s: pd.DataFrame, ycol: str, ylabel: str,
              title: str, fname: str, outdir: Path, written: list[Path],
              yscale: str | None = None, zoom: bool = False) -> None:
    usable = [lv for lv in levels
              if not s[(s["level"] == lv.level)].dropna(subset=[ycol]).empty]
    if not usable:
        return
    n = len(usable)
    anyl = bool(s["lossy"].any()) if "lossy" in s.columns else False
    anyu = bool((~s["loss_captured"]).any()) if "loss_captured" in s.columns else False
    fig, axes = plt.subplots(1, n, figsize=(max(5.2 * n, 6), 4.8), squeeze=False)
    for j, lv in enumerate(usable):
        a = axes[0][j]
        g0 = s[s["level"] == lv.level]
        ally = []
        for cc in sorted(g0["cc"].unique(), key=_cc_sort_key):
            g = (g0[g0["cc"] == cc].dropna(subset=[ycol])
                 .sort_values("buffer_mb"))
            if g.empty:
                continue
            a.plot(g["buffer_mb"], g[ycol], marker="o", label=cc, **_style(cc))
            mark_lossy(a, g, "buffer_mb", ycol)
            ally.append(g[ycol])
        logx_pow2(a, g0, "buffer_mb", "Per-switch buffer (MiB)")
        if yscale == "symlog":
            a.set_yscale("symlog", linthresh=10)
            a.set_ylim(bottom=0)          # a frame count has no negative decades
        elif zoom and ally:
            zoom_y(a, pd.concat(ally))
        else:
            a.set_ylim(bottom=0)
        a.set_title(lv.label, fontsize=10)
        a.grid(True, alpha=0.3, which="both")
        if j == 0:
            # one legend for the figure: every panel draws the same CC set, and
            # a repeated legend box was covering data in half the panels.
            h, _ = a.get_legend_handles_labels()
            a.legend(handles=h + loss_proxies(anyl, anyu), fontsize=8,
                     title="cc")
            a.set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    save_fig(fig, outdir, fname, written)


def fig_fct_cdf(levels: list[Level], refs: dict[str, float], outdir: Path,
                written: list[Path]) -> None:
    """06 -- the whole KV-flow FCT distribution per CC, not the single p99 point
    of figure 02. One panel per topology, at the head-to-head buffer.

    The empirical CDF is the honest form for this: the incast delivers a few
    hundred KV shards per run, so a percentile is a rank statistic over a small
    sample and two CCs whose p99 differ by 2% can have visibly different bodies.
    The right-hand tail — the last few percent — is what a decode stage's TP
    barrier waits for, and it is the part a mean would erase; the dotted ticks
    mark each CC's p99 so the figure-02 numbers can be located inside it."""
    usable = [lv for lv in levels
              if any(len(r.kv_fct_ns) for r in lv.runs)]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(1, n, figsize=(max(5.4 * n, 6), 4.8), squeeze=False)
    for j, lv in enumerate(usable):
        a = axes[0][j]
        buf = refs.get(lv.level)
        drew = False
        for r in sorted(lv.runs, key=lambda r: _cc_sort_key(r.cc)):
            if r.row.buffer_mb != buf or not len(r.kv_fct_ns):
                continue
            v = np.sort(r.kv_fct_ns) * MS
            y = (np.arange(len(v)) + 0.5) / len(v)
            a.plot(v, y, label=f"{r.cc} (n={len(v)})", lw=1.6, **_style(r.cc))
            # p99 as a short tick in the tail region, not a full-height rule:
            # five full verticals on top of five CDFs made the tail — the part
            # the figure exists for — the least readable part of the panel.
            a.vlines(np.percentile(v, 99), 0.90, 1.02, ls=":", lw=1.4,
                     alpha=0.9, color=_style(r.cc)["color"])
            drew = True
        if not drew:
            continue
        a.set_xlabel("KV flow-completion time (ms)")
        a.set_ylim(0, 1.02)
        a.set_title(f"{lv.label} @ buf{buf:g} MiB", fontsize=10)
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8, title="cc", loc="upper left")
        if j == 0:
            a.set_ylabel("Empirical CDF over KV flows")
    fig.suptitle("KV-flow FCT distribution per CC (dotted tick = that CC's p99)",
                 y=1.02)
    save_fig(fig, outdir, "06_kv_fct_cdf.png", written)


def fig_queue(levels: list[Level], s: pd.DataFrame, outdir: Path,
              written: list[Path]) -> None:
    """07 -- what the CC leaves in the switches. Top row: PEAK occupancy against
    the configured buffer (dashed = the buffer itself, i.e. the ceiling the run
    was given); bottom row: the TIME-MEAN occupancy, log-scaled because CCs that
    target a shallow queue and CCs that fill one differ by orders of magnitude,
    not by percents.

    This is the axis a completion-time table cannot show. Peak says whether the
    buffer was ever a constraint — every point far below the dashed line means
    the buffer axis of this sweep is inert for that CC — and the mean is the
    standing queue every other flow in the fabric pays latency for."""
    usable = [lv for lv in levels
              if not s[s["level"] == lv.level].dropna(subset=["q_peak_mb"]).empty]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(2, n, figsize=(max(5.2 * n, 6), 8.4), squeeze=False)
    for j, lv in enumerate(usable):
        g0 = s[s["level"] == lv.level]
        for row, (ycol, ylab) in enumerate((("q_peak_mb", "Peak switch queue (MiB)"),
                                            ("q_mean_kb", "Mean switch queue (kB, log)"))):
            a = axes[row][j]
            for cc in sorted(g0["cc"].unique(), key=_cc_sort_key):
                g = g0[g0["cc"] == cc].dropna(subset=[ycol]).sort_values("buffer_mb")
                if g.empty:
                    continue
                a.plot(g["buffer_mb"], g[ycol], marker="o", label=cc, **_style(cc))
                mark_lossy(a, g, "buffer_mb", ycol)
            if row == 0:
                xs = sorted(g0["buffer_mb"].unique())
                a.plot(xs, xs, ls="--", color="#9aa0a6", lw=1.2,
                       label="buffer (ceiling)")
                a.set_yscale("log", base=2)
            else:
                a.set_yscale("log")
            logx_pow2(a, g0, "buffer_mb", "Per-switch buffer (MiB)")
            a.set_title(lv.label if row == 0 else "", fontsize=10)
            a.grid(True, alpha=0.3, which="both")
            if j == 0:                    # one legend per row, not per panel
                a.legend(fontsize=8, title="cc")
                a.set_ylabel(ylab)
    fig.suptitle("Switch-buffer occupancy per CC: peak vs the configured ceiling "
                 "(top), standing queue (bottom)", y=1.01)
    save_fig(fig, outdir, "07_queue_occupancy_vs_buffer.png", written)


def fig_tradeoff(levels: list[Level], s: pd.DataFrame, outdir: Path,
                 written: list[Path]) -> None:
    """08 -- the trade-off plane: queueing on x, time on y, one point per
    (CC, buffer). Top row against the KV tail (the fabric's own metric), bottom
    row against the makespan (the workload's).

    A CC comparison collapses to one number only if the metrics agree; they do
    not, and this is where that becomes visible. Down-and-left is strictly
    better; a point that is left-but-up bought short queues with completion time.
    Marker shape carries the buffer, so a CC whose three buffers sit on top of
    each other is showing buffer-inertness directly."""
    usable = [lv for lv in levels
              if not s[s["level"] == lv.level].dropna(
                  subset=["q_mean_kb", "total_exec_ms"]).empty]
    if not usable:
        return
    n = len(usable)
    fig, axes = plt.subplots(2, n, figsize=(max(5.2 * n, 6), 8.4), squeeze=False)
    for j, lv in enumerate(usable):
        g0 = s[s["level"] == lv.level]
        for row, (ycol, ylab) in enumerate(
                (("kv_fct_p99_ms", "KV FCT p99 (ms)"),
                 ("total_exec_ms", "Makespan (ms)"))):
            a = axes[row][j]
            for cc in sorted(g0["cc"].unique(), key=_cc_sort_key):
                g = g0[g0["cc"] == cc].dropna(subset=["q_mean_kb", ycol])
                for _, r in g.iterrows():
                    a.scatter(r["q_mean_kb"], r[ycol], s=70,
                              marker=BUFFER_MARKER.get(int(r["buffer_mb"]), "D"),
                              color=_style(cc)["color"], zorder=3,
                              edgecolors="white", linewidths=0.6)
                if not g.empty:                        # one label per CC
                    a.scatter([], [], color=_style(cc)["color"], label=cc, s=70)
            a.set_xscale("log")
            a.set_xlabel("Mean switch queue (kB, log)")
            a.set_title(lv.label if row == 0 else "", fontsize=10)
            a.grid(True, alpha=0.3, which="both")
            if j == 0:                    # one legend per row, not per panel
                a.set_ylabel(ylab)
                h, _ = a.get_legend_handles_labels()
                shapes = [Line2D([], [], marker=m, ls="none", color="#444444",
                                 label=f"buf{b:g}")
                          for b, m in sorted(BUFFER_MARKER.items())
                          if b in set(g0["buffer_mb"])]
                a.legend(handles=h + shapes, fontsize=7, ncol=2)
    fig.suptitle("Standing queue against KV FCT p99 (top) and makespan "
                 "(bottom), per CC (marker = buffer)", y=1.01)
    save_fig(fig, outdir, "08_tradeoff_queue_vs_time.png", written)


def fig_profile(levels: list[Level], s: pd.DataFrame, refs: dict[str, float],
                outdir: Path, written: list[Path]) -> None:
    """09 -- every metric normalised to the best CC on that metric (1.0 = winner),
    grouped bars, one panel per topology, at the head-to-head buffer.

    The one figure that answers "so which one" at a glance: absolute units differ
    by six orders of magnitude between a makespan in ms and a queue in kB, so the
    only way to put them on one axis is as a ratio to the best. A bar at 1.15
    means 15% worse than the winner of THAT metric — the bars are comparable
    down a group, not across groups. Top row is the time family on a zoomed
    linear axis, bottom row the queueing family on a log one (see
    PROFILE_GROUPS)."""
    usable = [lv for lv in levels if not s[s["level"] == lv.level].empty]
    if not usable:
        return
    n, ng = len(usable), len(PROFILE_GROUPS)
    fig, axes = plt.subplots(ng, n, figsize=(max(6.0 * n, 7), 3.6 * ng),
                             squeeze=False)
    for j, lv in enumerate(usable):
        buf = refs.get(lv.level)
        g = s[(s["level"] == lv.level) & (s["buffer_mb"] == buf)]
        ccs = sorted(g["cc"].unique(), key=_cc_sort_key)
        if not ccs:
            continue
        for i, (gname, gcols, scale) in enumerate(PROFILE_GROUPS):
            a = axes[i][j]
            cols = [m for m in METRICS if m.col in gcols and m.col in g.columns
                    and g[m.col].notna().all() and (g[m.col] > 0).all()]
            if not cols:
                continue
            x = np.arange(len(cols), dtype=float)
            w = 0.8 / len(ccs)
            top = 1.0
            for k, cc in enumerate(ccs):
                r = g[g["cc"] == cc].iloc[0]
                # every profile metric is lower-is-better, so the ratio is always
                # value/best and always >= 1: "how many times the winner" reads
                # the same way down every group.
                vals = [r[m.col] / g[m.col].min() for m in cols]
                top = max(top, max(vals))
                bars = a.bar(x + k * w - 0.4 + w / 2, vals, width=w, label=cc,
                             color=_style(cc)["color"], alpha=0.9)
                a.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
            a.axhline(1.0, color="#444444", lw=1.0, ls="--")
            a.set_xticks(x)
            a.set_xticklabels([m.label for m in cols], rotation=20, ha="right",
                              fontsize=8)
            if scale == "log":
                a.set_yscale("log")
                a.set_ylim(0.9, top * 1.8)
            else:
                a.set_ylim(0.98, 1 + (top - 1) * 1.35)
            a.set_title(f"{lv.label} @ buf{buf:g} MiB — {gname}", fontsize=9)
            a.grid(True, axis="y", alpha=0.3)
            a.legend(fontsize=7, title="cc", ncol=2)
            if j == 0:
                a.set_ylabel("Ratio to the best CC\n(1.0 = best)")
    fig.suptitle("Per-metric distance from the best CC, normalised", y=1.01)
    save_fig(fig, outdir, "09_cc_profile.png", written)


# --------------------------------------------------------------------------- #
REPORT = ["level", "cc", "buffer_mb", "ttft_ms", "tail_ms", "total_exec_ms",
          "kv_fct_p50_ms", "kv_fct_p99_ms", "kv_fct_tail_ratio",
          "kv_slowdown_p99", "kv_skew_ms", "q_peak_mb",
          "q_peak_pct_of_buffer", "q_mean_kb", "total_pause_frames",
          "pfc_switches", "dropped_packets", "split_ok"]


def build_report(levels: list[Level], s: pd.DataFrame, cov: pd.DataFrame,
                 refs: dict[str, float], meta: dict) -> Report:
    """The whole analysis as one document. Kept separate from main() so the order
    of the argument is explicit and editable: context -> coverage -> head-to-head
    ranking -> buffer axis -> the runs that hurt -> findings."""
    rep = Report()
    rep.h(1, "CC SWEEP")
    rep.p(f"workload : {meta['workload']}")
    rep.p(f"configs  : {meta['sweep']}")
    rep.p(f"levels   : {', '.join(L.label for L in levels)}")
    rep.p(f"buffers  : {', '.join(f'{b:g}' for b in sorted(s['buffer_mb'].unique()))} MiB")
    rep.p(f"runs     : {len(s)} analysed of {len(cov)} declared")
    rep.p("Head-to-head buffer per level (largest buffer common to every CC): "
          + ", ".join(f"{lv}={b:g} MiB" for lv, b in sorted(refs.items())))

    rep.h(2, "What each CC ran with")
    rep.table(cc_parameter_table(levels))
    rep.p("Full ECN/threshold maps (KMIN/KMAX/PMAX) are in each run's "
          "configs/.../manifest.json.")

    rep.h(2, "Coverage of the declared (cc x buffer) grid")
    for L in levels:
        rep.h(3, L.label)
        rep.table(coverage_matrix(L.coverage))
    head, cells = coverage_findings(cov)
    rep.p(head)
    for line in cells:
        rep.bullet(line)

    rep.h(2, "Head-to-head ranking (value, and delta to the best CC; * = best)")
    for L in levels:
        buf = refs.get(L.level)
        g = s[s["level"] == L.level]
        if buf is None or g.empty:
            continue
        rep.h(3, f"{L.label} @ buf{buf:g} MiB")
        rep.table(ranking_table(g, buf))

    rep.h(2, "Does the buffer matter? (per CC, across the swept buffers)")
    for L in levels:
        g = s[s["level"] == L.level]
        if g.empty:
            continue
        rep.h(3, L.label)
        rep.table(sensitivity_table(g))

    rep.h(2, "Is the ECN marking the confound?")
    rep.p("Whether the CC spread could be an artifact of too-restrictive ECN "
          "marking rather than of the algorithms. Structurally, only "
          + " and ".join(sorted(ECN_DRIVEN))
          + (" reads" if len(ECN_DRIVEN) == 1 else " read")
          + " the KMIN/KMAX/PMAX maps (see the 'signal' column above); "
          "empirically, over-restrictive marking would leave the marked CCs "
          "with shallow queues and under-delivered goodput:")
    ecn_lines = ecn_confound_findings(s, refs, levels)
    if ecn_lines:
        for line in ecn_lines:
            rep.bullet(line)
    else:
        rep.p("No check could be computed on this row set (single-CC data, or "
              "missing goodput/queue columns).")

    hot = worst_cell_table(s)
    rep.h(2, "Runs where the fabric actually hurt (PFC, loss, or peak queue > 25% of buffer)")
    if hot.empty:
        rep.p("None: no PFC, no loss, and every peak queue below a quarter of "
              "its buffer on every analysed run.")
    else:
        rep.table(hot)

    rep.h(2, "Findings")
    for L in levels:
        g = s[s["level"] == L.level]
        buf = refs.get(L.level)
        if g.empty or buf is None:
            continue
        rep.h(3, L.label)
        for line in level_findings(g, buf, L.label):
            rep.bullet(line)
        for line in buffer_findings(g, L.label):
            rep.bullet(line)
    cross = cross_level_findings(s, refs)
    if cross:
        rep.h(3, "Across topologies")
        for line in cross:
            rep.bullet(line)

    rep.h(2, "Full per-run table")
    rep.table(s[[c for c in REPORT if c in s.columns]])
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default=CONFIG_SWEEP,
                    help=f"config sub-dir under configs/astra_sim/ns3 "
                         f"(default: {CONFIG_SWEEP})")
    ap.add_argument("--workload", default=OUT_WORKLOAD,
                    help=f"workload dir under output/<domain> (default: "
                         f"{OUT_WORKLOAD})")
    ap.add_argument("--root", default=str(incast.ROOT), type=Path)
    ap.add_argument("--levels", nargs="+", default=None,
                    help="topologies to analyse, e.g. --levels T3 T4")
    ap.add_argument("--ref-buffer", type=float, default=None,
                    help="buffer (MiB) at which the CCs are compared head-to-head "
                         "(default: the largest buffer every CC has a run at)")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/cc/"
                         "<workload>)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    outdir = (Path(a.out) if a.out else
              root / "results" / "sweep_analysis" / "cc" / a.workload)

    try:
        levels_found = incast.discover_levels(a.workload, root, "ns3")
        need(levels_found,
             f"no run under {root / 'output' / 'ns3' / a.workload}. "
             f"Is --workload right (has the sweep been launched)?")
        if a.levels:
            missing = set(a.levels) - set(levels_found)
            need(not missing, f"--levels {sorted(missing)} not present; "
                              f"found {levels_found}")
            levels_found = [l for l in levels_found if l in set(a.levels)]

        print(f"  root      {root}")
        print(f"  workload  {a.workload}")
        print(f"  out       {outdir}")
        print(f"  levels    {levels_found}")

        levels = []
        for lv in levels_found:
            L = analyse_level(lv, root, a.workload, a.sweep)
            if L is not None:
                levels.append(L)
        need(levels, "no level produced any analysable run.")
        levels.sort(key=lambda L: L.degree)

        s = pd.DataFrame([r.flat() for L in levels for r in L.runs])
        s["cc"] = pd.Categorical(
            s["cc"],
            categories=sorted(s["cc"].unique(), key=_cc_sort_key), ordered=True)
        s = s.sort_values(["incast_degree", "cc", "buffer_mb"]).reset_index(drop=True)
        cov = pd.concat([L.coverage for L in levels], ignore_index=True)
        refs = {L.level: ref_buffer(s[s["level"] == L.level], a.ref_buffer)
                for L in levels}
        refs = {k: v for k, v in refs.items() if v is not None}

        fresh_dir(outdir)
        front = [c for c in REPORT if c in s.columns]
        s[front + [c for c in s.columns if c not in front]].to_csv(
            outdir / "summary.csv", index=False)
        cov.to_csv(outdir / "coverage.csv", index=False)

        written: list[Path] = []
        _cc_lines(levels, s, "total_pause_frames",
                  "PFC PAUSE frames, whole fabric (symlog)",
                  "PFC backpressure vs buffer, per CC",
                  "01_pfc_frames_vs_buffer.png", outdir, written,
                  yscale="symlog")
        _cc_lines(levels, s, "kv_fct_p99_ms", "p99 KV-flow FCT (ms)",
                  "p99 KV flow-completion time vs buffer, per CC (y fitted to data)",
                  "02_kv_fct_p99_vs_buffer.png", outdir, written, zoom=True)
        _cc_lines(levels, s, "kv_skew_ms", "Intra-stage KV arrival skew (ms)",
                  "Worst intra-stage KV arrival skew vs buffer, per CC",
                  "03_kv_arrival_skew_vs_buffer.png", outdir, written)
        _cc_lines(levels, s, "total_exec_ms", "Makespan (ms)",
                  "Makespan vs buffer, per CC (y fitted to data)",
                  "04_makespan_vs_buffer.png", outdir, written, zoom=True)
        _cc_lines(levels, s, "ttft_ms", "TTFT (ms)",
                  "TTFT vs buffer, per CC (y fitted to data)",
                  "05_ttft_vs_buffer.png", outdir, written, zoom=True)
        fig_fct_cdf(levels, refs, outdir, written)
        fig_queue(levels, s, outdir, written)
        fig_tradeoff(levels, s, outdir, written)
        fig_profile(levels, s, refs, outdir, written)

        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        rep = build_report(levels, s, cov, refs,
                           {"workload": a.workload, "sweep": a.sweep})
        print(rep.console())
        (outdir / "report.md").write_text(rep.markdown(a.workload))

        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", "coverage.csv", "report.md",
                      *[q.name for q in written]]:
            print(f"  {fpath}")
        return drain_warnings()
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
