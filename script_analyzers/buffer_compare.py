#!/usr/bin/env python3
"""
buffer_compare — same buffer sweep, different MODELS: whose causal chain
(PP skew -> receiving-stage all-reduce -> TTFT) responds the same way to the
per-switch buffer?

The cross-model companion to buffer_sweep. Auto-discovers every workload
directory under output/ns3 that ran the given sweep (buffer_sweep_T1 by default)
and overlays them on one figure per metric. Reuses buffer_sweep.analyse_sweep so
a model is scored here identically to the single-model analysis -- one definition
of the metrics, two tools.

What goes on the axes, and why
--------------------------------------------------------------------------------
Two kinds of quantity, in two blocks.

Block A -- FABRIC-domain magnitudes, plotted RAW (absolute units). A PP arrival
skew, a queue depth, a PAUSE count, a link utilisation all live in the network
domain: they are caused by the fabric + buffer, not by how many parameters or
tokens a model has, so they do NOT carry the model's compute scale the way
ttft_ns or a tensor collective does. Their raw value is already the physical
number to compare across models -- no normalisation:

    pp_skew_ms           pp_skew_ns / 1e6: cross-rank arrival misalignment on the
                         receiving stage. A delta (idle ms), not a workload size,
                         so directly comparable across models.
    kv_tp_skew_mean_ms   mean per-(decode stage, layer) spread between the
                         arrivals of the KV shards feeding the SAME TP group
                         (buffer_sweep fig 08) -- the decode-side analog of
                         pp_skew_ms. A congestion-caused delta, so raw ms.
    dec_ar_first_skew_ms worst decode stage's entry skew into its FIRST TP
                         all-reduce (buffer_sweep fig 10): how staggered the
                         shards entered the one collective gated by that
                         stage's own KV. Also a delta, so raw ms.
    kv_tp_skew_p99_ms    the p99 of the same per-(stage, layer) shard-skew
                         population whose mean is plotted above: the tail a
                         layer's first all-reduce can actually inherit, which
                         a mean smooths away. A delta, so raw ms.
    decode_kv_stall_ms   buffer_sweep fig 11 (middle) as one number: the first
                         decode pass's excess over the steady inter-token gap
                         (tok2_latency - itl_steady) -- the wall-clock the
                         pipeline actually spent stalled on KV. Waiting time,
                         not compute, so raw ms.
    dec_kv_lateness_ms   fig 11 (right) reduced to the worst stage: KV ready
                         minus that stage's first-input arrival. >0 = the
                         stage outruns its KV and stalls; <=0 = the transfer
                         is fully masked. A signed fabric delta, so raw ms.
    qpeak_mb             link0_qpeak_bytes / 2^20: peak occupancy at the
                         bottleneck port, in MB. Absolute bytes -- deliberately
                         NOT qpeak_pct, whose denominator is the swept buffer
                         (dividing a queue by the buffer is circular on a buffer
                         sweep).
    pause_frames         link0_pause_frames: PFC PAUSE event count at the
                         bottleneck link. Raw count -- also grows with run
                         duration, so read alongside the KV window.
    line_rate_pct        already a % (link0_eff_pct): KV delivered vs the
                         bottleneck's nominal rate. Absolute bytes cancel.

Block B -- NORMALISED "does the fabric effect reach the user?" quantities. Here
the raw ns WOULD carry compute scale (a 70B's TTFT and collectives dwarf a
13B's), so these are made dimensionless -- self-normalised (divided by another
quantity of the SAME run) or normalised to that model's OWN largest-buffer run:

    rs_ar_first_bw       the first (skew-gated) all-reduce's EFFECTIVE BANDWIDTH
                         (comm_size / duration, GB/s), from the ASTRA CSV. The
                         skew stall shows as the bandwidth collapsing -- the early
                         rank waits idle at the barrier, so bytes-per-wall-time
                         drops -- with the ungated wire rate as the ceiling. A
                         rate, not a duration, so it is comparable across models
                         without normalising. Preferred over the old
                         ar_first_over_rest, whose "N x slower" framing was really
                         the idle skew wait (the transfer itself runs at W).
    ttft_slowdown        ttft_ns / ttft_ns at THIS model's largest buffer: how
                         much the buffer moves TTFT, relative to the most-relaxed
                         (largest-buffer) configuration. Flat ~1 across the sweep
                         means the buffer -> skew -> all-reduce chain does NOT
                         reach TTFT for that model.
    dec_ar_first_over_rest
                         worst decode stage's FIRST all-reduce duration in units
                         of that same stage's steady-state mean (buffer_sweep
                         fig 10): does the KV skew the decode pipeline inherits
                         actually stretch its first collective? Self-normalised
                         per stage, so dimensionless and comparable.
    tok2_over_itl        the first decode pass in units of THIS model's steady
                         inter-token gap (tok2_latency / itl_steady): does the
                         KV stall reach the user-visible second token? Flat ~1
                         means the transfer is fully hidden; the excess over 1
                         is decode_kv_stall_ms made dimensionless, so models of
                         different compute scale share the axis.

Kept in summary.csv but no longer plotted: ar_first_over_rest (rs_ar_first_ns /
rs_ar_rest_mean_ns -- the same stall as a duration multiple, dominated by the
idle skew wait rather than a slower transfer), skew_over_ar_rest (redundant now
that pp_skew_ms is shown raw -- it was the same skew in collective-units) and
kv_gate_over_ttft (decode-start timing, orthogonal to the buffer chain).

Discovery
---------
    <ROOT>/output/ns3/<workload>/<sweep>/<tag>/{fct,pfc,qlen}.txt

Every sub-directory of output/ns3 that contains a `<sweep>` directory is a
model; nothing is hard-coded. Run with --list to see what would be picked up
without analysing anything.

PP=1 models are first-class citizens: they have no PP wave, so pp_skew_ms (and
the skew-gating story) is NaN and their line drops from that one figure only.
The all-reduce bandwidths still report (buffer_sweep falls back to the single
prefill stage -- ungated, so expected flat: the control), and every fabric-
domain, KV and TTFT metric is measured exactly as for PP>1.

Bottleneck consistency is checked WITHIN each model's sweep (as buffer_sweep
does), never ACROSS models: different topologies number their switches
differently, so one 'sw->peer' string cannot be required to match across
models. --bottleneck is therefore not exposed here; pass it to
buffer_sweep.py directly if one model's auto-detected bottleneck needs
overriding.

Usage
-----
    python3 buffer_compare.py
    python3 buffer_compare.py --sweep buffer_sweep_T2 --workloads 'llama2_13b_*'
    python3 buffer_compare.py --list
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import pandas as pd

from utils import paths, roles
from utils.plots import lines_by_group
from utils.roles import Placement
from buffer_sweep import analyse_sweep
from utils.cli import Abort, need

NAN = float("nan")


def load_workload(root: Path, workload: str, sweep: str,
                  placement: Placement, top_links: int) -> pd.DataFrame:
    """One row per (workload, buffer) run -- mirrors buffer_sweep.main's call
    to analyse_sweep so a model is scored identically whether analysed alone or
    here, then adds only the comparable/normalised columns (see module docstring
    for why the absolute-ns columns already in `s` are not plotted across
    models)."""
    p = paths.SweepPaths(sweep=sweep, workload=workload, root=root)
    need(not p.missing_roots(),
         f"{workload}: derived root(s) do not exist:\n    "
         + "\n    ".join(p.missing_roots()))
    # want_series=False: this cross-model compare plots only scalars, so the
    # per-tag queue timelines are never built -- the big saving on qlen.txt reads.
    _, s, _ = analyse_sweep(p, placement, top_links=top_links,
                            bn_force=None, verbose=False, want_series=False)

    s = s.copy()
    # --- Block A: fabric-domain quantities, comparable in ABSOLUTE units ------
    # These live in the network domain (a delay, a byte count, an event count, a
    # %), not the compute domain, so they do NOT carry the model's parameter/
    # token scale the way ttft_ns or a tensor collective does -- their raw value
    # is already the physical number to compare across runs. No normalisation.
    s["pp_skew_ms"] = s["pp_skew_ns"] / 1e6          # arrival misalignment (delta)
    s["line_rate_pct"] = s.get("link0_eff_pct")      # already a %
    s["pause_frames"] = s.get("link0_pause_frames")  # raw PFC PAUSE event count
    # normalise the count by the KV window: a raw count also grows with run
    # duration, so frames/ms is the count made comparable across runs.
    win = s.get("link0_window_ns")
    s["pause_rate"] = (s["pause_frames"] / (win / 1e6)
                       if win is not None else NAN)  # PAUSE frames per ms of window
    qb = s.get("link0_qpeak_bytes")                  # absolute peak occupancy, MB
    qm = s.get("link0_qmean_bytes")                  # -- NOT qpeak_pct: dividing by
    s["qpeak_mb"] = qb / 2**20 if qb is not None else NAN   # the swept buffer is
    s["qmean_mb"] = qm / 2**20 if qm is not None else NAN   # circular on a buffer
    # decode-side skews (buffer_sweep figs 08/10): congestion-caused deltas like
    # pp_skew_ms, so raw ms. The per-stage decN_* columns are stage-numbered per
    # model (different PP splits); analyse_sweep already reduced them to the
    # WORST stage (buffer_sweep.decode_worst_stage) -- only units change here.
    s["kv_tp_skew_mean_ms"] = s["kv_tp_skew_mean_ns"] / 1e6
    s["kv_tp_skew_p99_ms"] = s["kv_tp_skew_p99_ns"] / 1e6
    s["dec_ar_first_skew_ms"] = s["dec_ar_first_skew_ns"] / 1e6
    # decode-stall family (buffer_sweep fig 11, reduced by decode_worst_stage):
    # waiting-time deltas, so raw ms like the skews above.
    s["decode_kv_stall_ms"] = s["decode_kv_stall_ns"] / 1e6
    s["dec_kv_lateness_ms"] = s["dec_kv_lateness_ns"] / 1e6
    # kept in the CSV for continuity, no longer plotted:                  # sweep.
    s["pause_pct_of_window"] = s.get("link0_pause_pct_of_window")
    s["qpeak_pct"] = s.get("link0_qpeak_pct")
    s["skew_over_ar_rest"] = s["pp_skew_ns"] / s["rs_ar_rest_mean_ns"]
    # --- Block B: normalised "does it propagate to TTFT?" quantities ----------
    # ar_first_over_rest is self-normalised (first gated all-reduce in units of
    # this model's own steady-state collective); ttft_slowdown is normalised to
    # this model's largest-buffer run. These answer the payoff question, not the
    # fabric magnitude one, so they stay dimensionless.
    s["ar_first_over_rest"] = s["rs_ar_first_ns"] / s["rs_ar_rest_mean_ns"]
    # first decode pass in units of this model's own steady inter-token gap:
    # the dimensionless twin of decode_kv_stall_ms (see module docstring).
    s["tok2_over_itl"] = s["tok2_latency_ns"] / s["itl_steady_ns"]
    # normalised to THIS model's largest-buffer (most relaxed) run.
    tt = s.dropna(subset=["ttft_ns"]).sort_values("buffer_mb")
    ref = float(tt["ttft_ns"].iloc[-1]) if len(tt) else NAN
    s["ttft_slowdown"] = (s["ttft_ns"] / ref
                          if pd.notna(ref) and ref > 0 else NAN)
    s.insert(0, "workload", workload)
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_compare_arguments(ap, "buffer_compare",
                                default_sweep="buffer_sweep_T1")
    ap.add_argument("--top-links", type=int, default=6,
                    help="how many KV-crossed links analyse_sweep scores "
                         "(only link0 is compared here; default: 6)")
    roles.add_argument(ap)
    a = ap.parse_args(argv)

    root = Path(a.root)
    workloads = paths.discover_workloads(root, a.sweep, "ns3")
    if a.workloads:
        workloads = [w for w in workloads
                    if any(fnmatch.fnmatch(w, pat) for pat in a.workloads)]
    if a.exclude:
        workloads = [w for w in workloads
                    if not any(fnmatch.fnmatch(w, pat) for pat in a.exclude)]

    print(f"sweep    {a.sweep}")
    print(f"root     {root}")
    print(f"found    {len(workloads)} workload(s):")
    for w in workloads:
        print(f"  - {w}")
    if a.list:
        return 0

    try:
        need(workloads,
             f"no workload under {root / 'output' / 'ns3'} has a "
             f"{a.sweep!r} sub-directory (or --workloads/--exclude filtered "
             f"all of them out)")

        placement = Placement.parse(a.placement)
        outdir = (Path(a.out) if a.out else
                  root / "results" / "sweep_analysis" / "buffer_compare" / a.sweep)

        frames = []
        print(f"\nScanning {len(workloads)} workload(s):")
        for w in workloads:
            summ = load_workload(root, w, a.sweep, placement, a.top_links)
            frames.append(summ)
            sk = summ.sort_values("buffer_mb")["pp_skew_ms"].iloc[0]
            print(f"  + {w:<55} buf={summ['buffer_mb'].min():g}.."
                  f"{summ['buffer_mb'].max():g}  "
                  f"bn={summ['bottleneck'].iloc[0]}  "
                  f"skew@min_buf="
                  f"{f'{sk:.2f}ms' if pd.notna(sk) else 'n/a (PP=1)'}")

        combined = pd.concat(frames, ignore_index=True)
        front = ["workload", "tag", "bottleneck", "buffer_mb",
                 "pp_skew_ms", "kv_tp_skew_mean_ms", "kv_tp_skew_p99_ms",
                 "dec_ar_first_skew_ms", "decode_kv_stall_ms",
                 "dec_kv_lateness_ms",
                 "qpeak_mb", "qmean_mb", "pause_rate", "pause_frames",
                 "line_rate_pct", "rs_ar_first_bw", "rs_ar_rest_bw",
                 "rs_ar_first_stage_bw", "ttft_slowdown", "dec_ar_first_over_rest",
                 "tok2_over_itl", "ar_first_over_rest",
                 "pause_pct_of_window", "qpeak_pct", "skew_over_ar_rest",
                 "kv_gate_over_ttft"]
        combined = combined[[c for c in front if c in combined.columns]
                            + [c for c in combined.columns if c not in front]]
        outdir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(outdir / "summary.csv", index=False)

        written: list[Path] = []

        def line_by_workload(ycol: str, ylabel: str, title: str, fname: str,
                             hline: float | None = None) -> None:
            # one line per model, x = buffer, log-2 axis (utils.plots.lines_by_group)
            lines_by_group(combined, "workload", "buffer_mb", ycol,
                           "Per-switch buffer (MiB)", ylabel, title, fname,
                           outdir, written, hline=hline, logx2=True)

        # === Block A: fabric-domain magnitudes, absolute & comparable ========
        line_by_workload(
            "pp_skew_ms", "PP arrival skew (ms)",
            "PP arrival skew",
            "pp_skew_ms_by_workload.png")

        line_by_workload(
            "kv_tp_skew_mean_ms", "KV TP-group skew, mean (ms)",
            "KV shard arrival skew within decode TP groups",
            "kv_tp_skew_by_workload.png")

        line_by_workload(
            "dec_ar_first_skew_ms", "Decode first all-reduce entry skew (ms)",
            "KV skew inherited by the decode pipeline (worst stage)",
            "decode_ar_first_skew_by_workload.png")

        line_by_workload(
            "kv_tp_skew_p99_ms", "KV TP-group skew, p99 (ms)",
            "KV shard arrival skew within decode TP groups — the tail",
            "kv_tp_skew_p99_by_workload.png")

        line_by_workload(
            "decode_kv_stall_ms", "First-pass KV stall (ms)",
            "Decode first pass: excess over the steady inter-token gap",
            "decode_kv_stall_by_workload.png", hline=0.0)

        line_by_workload(
            "dec_kv_lateness_ms", "KV ready − first input (ms)",
            "KV lateness at the decode stages' first input (worst stage)",
            "dec_kv_lateness_by_workload.png", hline=0.0)

        line_by_workload(
            "qpeak_mb", "Peak queue occupancy (MB)",
            "Bottleneck buffer occupancy",
            "qpeak_occupancy_mb_by_workload.png")

        line_by_workload(
            "pause_rate", "PFC PAUSE (frames/ms)",
            "Backpressure intensity",
            "pause_rate_by_workload.png")

        line_by_workload(
            "line_rate_pct", "KV bandwidth (% of line-rate)",
            "KV bandwidth utilisation",
            "line_rate_efficiency_by_workload.png")

        # === Block B: does the fabric effect propagate to the user? ==========
        # First (skew-gated) all-reduce as EFFECTIVE BANDWIDTH (bytes/duration),
        # like buffer_sweep fig 01: the stall shows as the bandwidth COLLAPSING
        # (the early rank sits idle at the barrier, so bytes/wall-time drops),
        # bounded above by the ungated wire rate. Honest about what happens --
        # the transfer is not slower, it waits -- and comparable across models
        # because it is a rate, not a compute-scaled duration.
        line_by_workload(
            "rs_ar_first_bw", "First all-reduce eff. bw (GB/s)",
            "Skew stall on the first all-reduce (effective bandwidth collapses)",
            "allreduce_first_bandwidth_by_workload.png")

        line_by_workload(
            "ttft_slowdown", "TTFT (×largest-buffer)",
            "TTFT sensitivity to buffer",
            "ttft_slowdown_by_workload.png", hline=1.0)

        line_by_workload(
            "dec_ar_first_over_rest", "Decode first all-reduce (×steady-state)",
            "Does the inherited KV skew stretch the decode first all-reduce?",
            "decode_ar_first_over_rest_by_workload.png", hline=1.0)

        line_by_workload(
            "tok2_over_itl", "First decode pass (×steady ITL)",
            "Does the KV stall reach the second token?",
            "tok2_over_itl_by_workload.png", hline=1.0)

        print(f"\nWrote {outdir}:")
        print("  summary.csv")
        for pth in written:
            print(f"  {pth.name}")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
