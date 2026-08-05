#!/usr/bin/env python3
"""
cc_sweep — the congestion-control comparison on the disaggregated-inference
fabric: same topologies and KV-cache incast as incast_sweep (T3/T4 = prefill
TP4/TP8 converging on TP2 decode pools), but the swept knob is the CC ALGORITHM
(dcqcn / hpcc / timely / dctcp / none), with the per-switch buffer as a small
secondary axis (8/16/32 MiB) that separates the PFC-storm regime from the
comfortable one.

`none` (CC_MODE 12) is the window-only ideal baseline: injection capped at the
per-pair BDP, PFC as the only backpressure. Every other CC pays its control
loop on top of that floor, so the summary reports each CC's makespan and TTFT
RELATIVE to `none` at the same (topology, buffer) — the price (or gain) of the
congestion control itself.

Per-CC parameters are the HPCC SIGCOMM'19 artifact set at 100G (vwin variants);
see configs/astra_sim/ns3/documentation/cc_parameter_provenance.md. The
manifest.json of each run carries param_set/cc, and this analyzer cross-checks
the manifest's cc against the tag so a stale config cannot silently masquerade
as another algorithm.

Reused verbatim from incast_sweep (one definition of a metric): the whole
per-run measurement (`analyse` -> Row: TTFT, makespan, intra-stage KV skew,
PFC census, drops, busiest switches), the placement recovery and the loss
marking. New here: the CC axis, the KV-flow FCT distribution (mean/p50/p99 and
p99 slowdown — the classic CC-comparison metric, read from the same fct.txt),
and the vs-`none` normalisation.

Output
--------------------------------------------------------------------------------
    <out>/01_pfc_frames_vs_buffer.png      PFC PAUSE frames, panel/topo, line/CC
    <out>/02_kv_fct_p99_vs_buffer.png      p99 KV-flow FCT, panel/topo, line/CC
    <out>/03_kv_arrival_skew_vs_buffer.png worst intra-stage KV skew, line/CC
    <out>/04_makespan_vs_buffer.png        makespan (y fitted), panel/topo, line/CC
    <out>/05_ttft_vs_buffer.png            TTFT, panel/topo, line/CC
    <out>/06_makespan_vs_none.png          makespan normalised to `none`, line/CC
    <out>/summary.csv                      one row per run
Usage
-----
    python3 cc_sweep.py                    # analyse what is on disk, skip the rest
    python3 cc_sweep.py --levels T4
    python3 cc_sweep.py --workload llama2_13b_16reqs_512prompt_incast_sweep \\
                        --sweep incast_sweep      # smoke-test on the incast runs
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import incast, ns3, roles
from utils import flows as flowlib
from utils.cli import Abort, need
from utils.fabric import parse_ns3_config, parse_topology
from utils.paths import BUFFER_AXIS
from utils.plots import logx_pow2, save_fig
from utils.roles import Placement

from buffer_sweep import MS, WARNINGS, warn, _zoom_y
from incast_sweep import Row, _loss_proxies, _mark_lossy, analyse, recover_placement

NAN = float("nan")

CONFIG_SWEEP = "cc_sweep"
OUT_WORKLOAD = "llama2_13b_16reqs_512prompt_cc_sweep"

# tag = "<level>_bx<rate>_<cc>_buf<n>"; the cc token is everything between the
# bx and buf tokens (may contain '-', e.g. hpcc-pint, but never '_').
_CC = re.compile(r"_bx[^_]+_(.+?)_buf\d", re.IGNORECASE)

# Fixed CC identity: same colour in every figure. `none` is the ideal baseline
# and is drawn dashed grey so it reads as a floor, not as a competitor.
CC_ORDER = ["none", "dcqcn", "hpcc", "timely", "dctcp", "hpcc-pint"]
CC_STYLE = {
    "none":      dict(color="#9aa0a6", ls="--"),
    "dcqcn":     dict(color="#1f77b4", ls="-"),
    "hpcc":      dict(color="#2b8a3e", ls="-"),
    "timely":    dict(color="#6a4c93", ls="-"),
    "dctcp":     dict(color="#d1495b", ls="-"),
    "hpcc-pint": dict(color="#66a61e", ls="-"),
}


def cc_of(tag: str) -> str | None:
    m = _CC.search(tag)
    return m.group(1).lower() if m else None


def _style(cc: str) -> dict:
    return CC_STYLE.get(cc, dict(color="#444444", ls="-"))


def _cc_sort_key(cc: str) -> tuple:
    return (CC_ORDER.index(cc) if cc in CC_ORDER else len(CC_ORDER), cc)


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


def kv_fct_stats(tag: str, p: incast.IncastPaths, placement: Placement) -> dict:
    """The FCT distribution of the KV flows — the metric CC papers compare on.
    Re-reads fct.txt (cheap) because incast_sweep.Row keeps arrivals, not FCTs.
    slowdown = fct / standalone_fct (see utils.ns3.read_fct): ~concurrency when
    the bottleneck is shared fairly, worse when the CC lets queues build."""
    topo = parse_topology(p.topology(tag))
    cfg = parse_ns3_config(p.config(tag))
    raw = ns3.read_fct(p.ns3_run(tag) / "fct.txt")
    if raw is None or not len(raw):
        warn(f"{tag}: fct.txt unreadable for the FCT stats.")
        return {}
    f = flowlib.annotate(raw, topo, placement, cfg.payload)
    kv = f[f["flow_class"] == "kv"]
    if not len(kv):
        warn(f"{tag}: no KV flow classified; FCT stats unavailable.")
        return {}
    fct = kv["fct"].to_numpy(dtype=float)
    out = {
        "kv_fct_mean_ns": float(fct.mean()),
        "kv_fct_p50_ns": float(np.percentile(fct, 50)),
        "kv_fct_p99_ns": float(np.percentile(fct, 99)),
    }
    sl = kv["slowdown"].dropna()
    if len(sl):
        out["kv_slowdown_p99"] = float(np.percentile(sl, 99))
    return out


@dataclass
class CcRun:
    cc: str
    param_set: str
    row: Row                       # incast_sweep's full per-run measurement
    kv_fct_mean_ns: float = NAN
    kv_fct_p50_ns: float = NAN
    kv_fct_p99_ns: float = NAN
    kv_slowdown_p99: float = NAN

    def flat(self) -> dict:
        d = self.row.flat()
        d["cc"] = self.cc
        d["param_set"] = self.param_set
        d["kv_fct_mean_ms"] = self.kv_fct_mean_ns * MS
        d["kv_fct_p50_ms"] = self.kv_fct_p50_ns * MS
        d["kv_fct_p99_ms"] = self.kv_fct_p99_ns * MS
        d["kv_slowdown_p99"] = self.kv_slowdown_p99
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


def analyse_level(level: str, root: Path, out_workload: str,
                  config_sweep: str) -> Level | None:
    p = incast.IncastPaths(level=level, out_workload=out_workload,
                           config_sweep=config_sweep, root=root)
    if p.missing_roots():
        warn(f"{level}: skipped, derived root(s) missing:\n    "
             + "\n    ".join(p.missing_roots()))
        return None
    tags, skipped = p.usable_tags()
    for s in skipped:
        warn(f"{level}: {s} (run not finished yet?) -- skipped.")
    if not tags:
        warn(f"{level}: no run has all inputs on disk yet; skipped.")
        return None

    placement = recover_placement(p, tags)
    degree = incast.prefill_tp(placement)
    print(f"\n===== {level}  (prefill TP{degree}) =====")
    print(f"  placement {roles.spec_of(placement)}")

    runs: list[CcRun] = []
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
            row = analyse(tag, p, placement)
        except Abort as e:
            warn(f"{level}: run {tag} dropped -- {e}")
            continue
        run = CcRun(cc=cc, param_set=man.get("param_set", ""), row=row)
        for k, v in kv_fct_stats(tag, p, placement).items():
            setattr(run, k, v)
        runs.append(run)
    if not runs:
        warn(f"{level}: every run failed to analyse; skipped.")
        return None
    runs.sort(key=lambda r: (_cc_sort_key(r.cc), r.row.buffer_mb))

    # Completeness check: every run of a level replays the same workload, so
    # the flow count must match across CCs/buffers. A short count means the
    # simulation was interrupted (fct.txt truncated) and its metrics lie.
    totals = [r.row.kv_flows + r.row.other_flows for r in runs]
    modal = max(set(totals), key=totals.count)
    for r in runs:
        t = r.row.kv_flows + r.row.other_flows
        if t < modal:
            warn(f"{r.row.tag}: only {t} flows vs {modal} in sibling runs — "
                 f"run looks INCOMPLETE (interrupted?); its metrics are not "
                 f"trustworthy, re-run it.")

    for r in runs:
        flag = "" if r.row.split_ok else "  ! split check FAILED"
        if r.row.lossy:
            flag += f"  ** LOSS: {r.row.dropped_packets:.0f} pkt **"
        print(f"  + {r.cc:<9} buf{r.row.buffer_mb:<4g} "
              f"ttft={r.row.ttft_ns * MS:6.1f}ms  "
              f"total={r.row.total_exec_ns * MS:6.1f}ms  "
              f"kv_p99={r.kv_fct_p99_ns * MS:6.2f}ms  "
              f"pfc={r.row.total_pause_frames:6.0f}{flag}")
    return Level(level=level, degree=degree, runs=runs,
                 label=f"{level} (tp{degree})")


def add_vs_none(s: pd.DataFrame) -> pd.DataFrame:
    """makespan/TTFT normalised to the `none` run of the SAME (level, buffer):
    the cost of the congestion control itself. NaN when `none` is not on disk
    (e.g. smoke-testing on the dcqcn-only incast sweep)."""
    base = (s[s["cc"] == "none"]
            .set_index(["level", "buffer_mb"])[["total_exec_ms", "ttft_ms"]]
            .rename(columns={"total_exec_ms": "_none_total",
                             "ttft_ms": "_none_ttft"}))
    s = s.join(base, on=["level", "buffer_mb"])
    s["makespan_vs_none"] = s["total_exec_ms"] / s["_none_total"]
    s["ttft_vs_none"] = s["ttft_ms"] / s["_none_ttft"]
    return s.drop(columns=["_none_total", "_none_ttft"])


# --------------------------------------------------------------------------- #
# Figures: panel per topology, one line per CC, buffer on x
# --------------------------------------------------------------------------- #
def _cc_lines(levels: list[Level], s: pd.DataFrame, ycol: str, ylabel: str,
              title: str, fname: str, outdir: Path, written: list[Path],
              yscale: str | None = None, zoom: bool = False,
              hline1: bool = False) -> None:
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
            _mark_lossy(a, g, "buffer_mb", ycol)
            ally.append(g[ycol])
        if hline1:
            a.axhline(1.0, color="#9aa0a6", lw=0.8, ls=":")
        logx_pow2(a, g0, "buffer_mb", "Per-switch buffer (MiB)")
        if yscale == "symlog":
            a.set_yscale("symlog", linthresh=10)
        elif zoom and ally:
            _zoom_y(a, pd.concat(ally))
        else:
            a.set_ylim(bottom=0)
        a.set_title(lv.label, fontsize=10)
        a.grid(True, alpha=0.3, which="both")
        h, _ = a.get_legend_handles_labels()
        proxies = _loss_proxies(anyl, anyu) if j == 0 else []
        a.legend(handles=h + proxies, fontsize=8, title="cc")
        if j == 0:
            a.set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    save_fig(fig, outdir, fname, written)


# --------------------------------------------------------------------------- #
REPORT = ["level", "cc", "buffer_mb", "ttft_ms", "total_exec_ms",
          "makespan_vs_none", "kv_fct_p50_ms", "kv_fct_p99_ms",
          "kv_slowdown_p99", "kv_skew_ms", "total_pause_frames",
          "dropped_packets", "split_ok"]


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
        s = add_vs_none(s)
        s["cc"] = pd.Categorical(
            s["cc"],
            categories=sorted(s["cc"].unique(), key=_cc_sort_key), ordered=True)
        s = s.sort_values(["incast_degree", "cc", "buffer_mb"]).reset_index(drop=True)

        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        front = [c for c in REPORT if c in s.columns]
        s[front + [c for c in s.columns if c not in front]].to_csv(
            outdir / "summary.csv", index=False)

        written: list[Path] = []
        _cc_lines(levels, s, "total_pause_frames",
                  "PFC PAUSE frames, whole fabric (symlog)",
                  "PFC backpressure vs buffer, per CC",
                  "01_pfc_frames_vs_buffer.png", outdir, written,
                  yscale="symlog")
        _cc_lines(levels, s, "kv_fct_p99_ms", "p99 KV-flow FCT (ms)",
                  "p99 KV flow-completion time vs buffer, per CC",
                  "02_kv_fct_p99_vs_buffer.png", outdir, written)
        _cc_lines(levels, s, "kv_skew_ms", "Intra-stage KV arrival skew (ms)",
                  "Worst intra-stage KV arrival skew vs buffer, per CC",
                  "03_kv_arrival_skew_vs_buffer.png", outdir, written)
        _cc_lines(levels, s, "total_exec_ms", "Makespan (ms)",
                  "Makespan vs buffer, per CC (y fitted to data)",
                  "04_makespan_vs_buffer.png", outdir, written, zoom=True)
        _cc_lines(levels, s, "ttft_ms", "TTFT (ms)",
                  "TTFT vs buffer, per CC (y fitted to data)",
                  "05_ttft_vs_buffer.png", outdir, written, zoom=True)
        if s["makespan_vs_none"].notna().any():
            _cc_lines(levels, s, "makespan_vs_none",
                      "Makespan / makespan(none)",
                      "Cost of the CC: makespan normalised to the window-only "
                      "baseline", "06_makespan_vs_none.png", outdir, written,
                      zoom=True, hline1=True)

        pd.set_option("display.width", 240)
        print("\n================ CC SWEEP ================")
        print(s[[c for c in REPORT if c in s.columns]].to_string(index=False))

        if "lossy" in s.columns and s["lossy"].any():
            lossy = s[s["lossy"] == True]
            print(f"\n! PACKET LOSS on {len(lossy)} run(s) — flagged RED in "
                  f"the figures:")
            for _, rr in lossy.iterrows():
                print(f"    {rr['level']} {rr['cc']} buf{rr['buffer_mb']:g}: "
                      f"{int(rr['dropped_packets'])} pkt")

        print(f"\nWrote {outdir}:")
        for fpath in ["summary.csv", *[q.name for q in written]]:
            print(f"  {fpath}")
        if WARNINGS:
            print(f"\n{len(WARNINGS)} WARNING(S):")
            for w in WARNINGS:
                print(f"  ! {w}")
            return 1
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
