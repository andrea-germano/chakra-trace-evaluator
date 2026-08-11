#!/usr/bin/env python3
"""
roofline_compare — the bandwidth sweep with the units divided out: are the
eight workloads eight results, or one?

bandwidth_compare.py puts every workload on a shared *speedup* axis, which
answers "who scales better with bandwidth". It cannot answer the question one
step up: whether the eight curves are the same curve seen at different offsets.
They look different because each workload is normalised to its OWN slowest run,
so a workload that is already compute-bound at bx25 and one that is still
fabric-bound at bx400 both start at 1.0 and neither position says where it sits
relative to the other.

Here both axes are dimensionless and shared, so a point's position means the
same thing for every workload:

    beta = T_net / T_comp        x: how much fabric work per unit of compute
    y    = makespan / T_comp     y: how much worse than the compute-only floor

with the two closed-form bounds drawn as reference curves:

    y = max(1, beta)             perfect overlap  -- transfer entirely hidden
    y = 1 + beta                 zero overlap     -- transfer fully serialised

Every run must land between them, and WHERE it lands is the measurement. The
distance is reported as the overlap fraction

    phi = (1 + beta - y) / beta

= the share of the KV transfer that compute actually managed to hide. phi = 1 is
the perfect-overlap curve, phi = 0 the serialisation curve, and phi < 0 means the
run did worse than serialising, i.e. the transfer itself ran below line rate.

Only the bandwidth sweep gets this treatment. beta moves because BANDWIDTH moves;
the cc, buffer, incast and oversub sweeps all hold bandwidth and bytes fixed, so
every one of their runs has the same beta and the whole sweep collapses to a
single x -- a vertical stripe, not a curve. Their effects are second-order
*around* a point of this plot, which is the right way to read them and the reason
they are analysed separately.

The two ingredients
-------------------
T_comp, the floor, is MEASURED, not modelled: `nonfabric_union_ns`, the wall-clock
time some rank is inside an op that does not ride the swept links -- compute, the
TP all-reduce (intra-node, fixed at 4800 Gbps) and the decode feedback. Every gap
left in that union is a wait on a link the sweep moves, so the union is exactly
the makespan the run would have with that fabric free. It comes out bit-identical
across the whole 16x sweep, and the tool refuses to normalise if it does not.

Getting this denominator right is most of the work here, and two plausible
choices are both wrong: the makespan of the run that hides its transfer (makes
y = 1 and phi = 1 true by definition) and the compute union alone (charges the
bandwidth-independent TP stalls to the KV transfer). compute_floor has the full
account and the size of each error.

T_net, the ideal transfer time, is kv_total_bytes / (kv_senders x bandwidth).
`kv_senders` (bandwidth_sweep.summarise_run) is the number of distinct ranks that
send KV, and each has one uplink, so that product is the aggregate rate the
transfer could reach if every sender ran at line rate the whole time. Not
kv_count: streaming splits the same bytes over 20x more flows without adding a
single link.

phi is derived from makespan, beta and T_comp alone. It deliberately does not go
through `kv_exposed_ns`, whose notion of "hidden" is overlap with compute running
ANYWHERE in the system -- a test almost nothing fails once compute covers most of
the makespan.

One approximation is left standing: beta counts only the KV bytes, while PP and
FIRSTTOK ride the same swept links. They are ~2.5% of the KV volume here, so they
show up as part of the residual above the perfect-overlap bound rather than as
their own term.

Usage
-----
    python3 roofline_compare.py
    python3 roofline_compare.py --workloads 'llama2_13b_p-tp2pp2*'
    python3 roofline_compare.py --list
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import paths
from utils.cli import Abort, need
from utils.plots import MS, BLUE, CORAL, MUTED, save_fig, zoom_y
from bandwidth_compare import load_workload

# T_comp is supposed to be a property of the workload, so the same number must
# come out of every run of the sweep. This is how far apart they are allowed to
# be before the quantity is not that: in practice the spread is 0.00%.
TCOMP_SPREAD_MAX = 0.01


def compute_floor(summ: pd.DataFrame) -> float:
    """T_comp: the makespan this workload would have on an infinitely fast fabric.

    `nonfabric_union_ns` (bandwidth_sweep.summarise_run) -- the wall-clock time
    some rank is inside an op that does NOT ride the swept links (compute, the
    intra-node TP all-reduce, the decode feedback). Every gap left in it is a
    wait on a link the sweep moves, so with that fabric free the makespan
    collapses onto exactly this.

    Two wrong floors were tried first and both are instructive.

    The makespan of the run whose `kv_exposed_ns` hit zero is circular twice
    over. Once because y = makespan / T_comp is then 1.0 by construction at that
    run and phi = 1.0 with it -- the end of the curve was a definition, not a
    measurement. Twice because `kv_exposed_ns` scores a transfer as hidden if it
    overlaps compute ANYWHERE in the system, a test almost nothing fails once
    compute covers most of the makespan.

    `comp_union_ns` fixes the circularity and introduces a misattribution: it
    leaves the TP stalls outside the floor, so they are charged to the KV
    transfer. They are 89% of the residual idle at the top of the sweep and they
    do not move with bandwidth at all -- TP is on the 4800 Gbps intra-node link.
    That understated phi at bx400 by roughly half and made it non-monotone.

    The gap between this floor and the measured makespan at bx400 is ~1.2%, i.e.
    at 400 Gbps the swept fabric has essentially left the critical path. That is
    the sanity check the first two floors both failed."""
    v = summ["nonfabric_union_ns"].dropna()
    need(not v.empty,
         f"{summ['workload'].iloc[0]}: no COMP rows, so there is no compute "
         f"floor to normalise against")
    spread = (v.max() - v.min()) / v.max()
    need(spread <= TCOMP_SPREAD_MAX,
         f"{summ['workload'].iloc[0]}: the compute union moves {spread:.1%} "
         f"across the sweep ({v.min():.0f}..{v.max():.0f} ns). It is supposed to "
         f"be bandwidth-independent; if it is not, these runs differ in more "
         f"than bandwidth and beta would mix two knobs.")
    return float(v.median())


def normalise(summ: pd.DataFrame) -> pd.DataFrame:
    """Add beta / y / phi and the reference bounds to one workload's sweep."""
    need(summ["kv_senders"].gt(0).all(),
         f"{summ['workload'].iloc[0]}: a run has no KV senders, so the ideal "
         f"transfer time is undefined -- is this a disaggregated workload?")
    tcomp = compute_floor(summ)
    need(tcomp > 0, f"{summ['workload'].iloc[0]}: non-positive compute floor")

    s = summ.copy()
    s["Tcomp_ns"] = tcomp
    # bandwidth is per link in Gbps; bytes -> bits, Gbit/s -> bit/ns, so the
    # 1e9 (giga) and the 1e9 (s -> ns) cancel and the 8 is all that survives.
    s["Tnet_ideal_ns"] = s["kv_total_bytes"] * 8 / (s["kv_senders"] * s["bandwidth"])
    s["beta"] = s["Tnet_ideal_ns"] / tcomp
    s["y_norm"] = s["makespan_ns"] / tcomp
    s["y_overlap"] = np.maximum(1.0, s["beta"])          # perfect overlap bound
    s["y_serial"] = 1.0 + s["beta"]                      # zero overlap bound
    s["phi_overlap"] = (s["y_serial"] - s["y_norm"]) / s["beta"]
    # Achieved aggregate KV rate as a share of what the uplinks could carry --
    # the term that separates "the transfer was slow" from "the transfer was not
    # hidden", which phi alone conflates.
    s["kv_agg_efficiency"] = s["Tnet_ideal_ns"] / s["kv_busy_union_ns"]
    return s


def _style(workload: str) -> dict:
    """Bulk is the control, not a competitor: it forgoes overlap by construction,
    so it is drawn dashed and grey and must land ON the serialisation line. It
    doing so is what says the normalisation is not fitting itself to the data."""
    if "_bulk_" in workload:
        return dict(linestyle="--", color=MUTED, marker="s", zorder=3)
    return dict(linestyle="-", marker="o", zorder=4)


def _short(workload: str) -> str:
    """Drop the parts every workload in a sweep shares -- the parallelism spec is
    constant here and eats half the legend."""
    return (workload.replace("llama2_13b_", "")
                    .replace("p-tp2pp2_d-tp2pp2_", "")
                    .replace("prompt", ""))


def fig_roofline(df: pd.DataFrame, outdir: Path, written: list[Path]) -> None:
    """The collapse plot: every run of every workload on one pair of axes."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    b = np.logspace(np.log10(df["beta"].min() * 0.7),
                    np.log10(df["beta"].max() * 1.4), 200)
    ax.plot(b, np.maximum(1.0, b), color="k", lw=1.4,
            label=r"perfect overlap  $y=\max(1,\beta)$")
    ax.plot(b, 1.0 + b, color="k", lw=1.4, linestyle=":",
            label=r"zero overlap  $y=1+\beta$")
    ax.axvline(1.0, color=MUTED, lw=1.0, alpha=0.5)

    for w, g in df.groupby("workload"):
        g = g.sort_values("beta")
        ax.plot(g["beta"], g["y_norm"], label=_short(w), **_style(w))

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Annotate after the data is in, so the label sits inside the final y range
    # instead of wherever the reference curves alone happened to end.
    ax.text(1.03, ax.get_ylim()[0] * 1.05,
            r"$\beta=1$: fabric = compute", rotation=90, va="bottom",
            fontsize=8, color=MUTED)
    ax.set_xlabel(r"$\beta$ = ideal KV transfer time / compute-only makespan")
    ax.set_ylabel(r"$y$ = makespan / compute-only makespan")
    ax.set_title(f"One curve, {df['workload'].nunique()} workloads: "
                 f"the bandwidth sweep with units divided out")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "01_roofline_collapse.png", written)


def fig_overlap(df: pd.DataFrame, outdir: Path, written: list[Path]) -> None:
    """phi against beta: how much of the transfer compute actually hid.

    The reading this figure exists for is the SLOPE. If streaming bought a fixed
    amount of overlap, phi would be flat in beta."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for w, g in df.groupby("workload"):
        g = g.sort_values("beta")
        ax.plot(g["beta"], g["phi_overlap"], label=_short(w), **_style(w))
    ax.axhline(1.0, color="k", lw=1.2, label="fully hidden")
    ax.axhline(0.0, color="k", lw=1.2, linestyle=":", label="fully serialised")
    ax.axvline(1.0, color=MUTED, lw=1.0, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ = ideal KV transfer time / compute-only makespan")
    ax.set_ylabel(r"$\varphi$ = fraction of KV transfer hidden by compute")
    ax.set_title("Streaming KV hides the transfer only when the fabric is already fast")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "02_overlap_fraction.png", written)


def fig_residual(df: pd.DataFrame, outdir: Path, written: list[Path]) -> None:
    """What the collapse does NOT explain, as a share of the compute floor.

    y - max(1, beta) is the makespan the two-term model fails to account for. On
    a shared axis it says whether the leftover is a workload effect (curves apart)
    or a beta effect (curves together)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for w, g in df.groupby("workload"):
        g = g.sort_values("beta")
        ax.plot(g["beta"], g["y_norm"] - g["y_overlap"], label=_short(w), **_style(w))
    ax.axhline(0.0, color="k", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ = ideal KV transfer time / compute-only makespan")
    ax.set_ylabel(r"$y - \max(1,\beta)$   (units of compute-only makespan)")
    ax.set_title("Residual above the perfect-overlap bound")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "03_residual_vs_bound.png", written)


def fig_efficiency(df: pd.DataFrame, outdir: Path, written: list[Path]) -> None:
    """Aggregate KV rate as a share of the uplinks' capacity.

    Separates the two ways a run can sit above the perfect-overlap bound: the
    transfer was not hidden (phi), or the transfer was slow (here). A streaming
    run that falls off at high beta is losing wire time, not overlap."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for w, g in df.groupby("workload"):
        g = g.sort_values("beta")
        ax.plot(g["beta"], g["kv_agg_efficiency"], label=_short(w), **_style(w))
    ax.axhline(1.0, color="k", lw=1.2, label="senders x line rate")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ = ideal KV transfer time / compute-only makespan")
    ax.set_ylabel("achieved / ideal aggregate KV rate")
    ax.set_title("Is the KV transfer slow, or just not hidden?")
    ax.grid(alpha=0.25)
    # Include the 1.0 rule in the range: it is the reference the whole panel is
    # read against, and zooming to the data alone pushes it off-figure while
    # leaving its legend entry behind.
    zoom_y(ax, pd.concat([df["kv_agg_efficiency"], pd.Series([1.0])]))
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "04_kv_aggregate_efficiency.png", written)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_compare_arguments(ap, "roofline_compare",
                                default_sweep="bandwidth_sweep")
    ap.add_argument("--pattern", default="*.csv",
                    help="glob for the per-rank CSVs inside each run dir "
                         "(default: *.csv)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    workloads = paths.discover_workloads(root, a.sweep, "astra")
    if a.workloads:
        workloads = [w for w in workloads
                     if any(fnmatch.fnmatch(w, p) for p in a.workloads)]
    if a.exclude:
        workloads = [w for w in workloads
                     if not any(fnmatch.fnmatch(w, p) for p in a.exclude)]

    print(f"sweep    {a.sweep}")
    print(f"root     {root}")
    print(f"found    {len(workloads)} workload(s):")
    for w in workloads:
        print(f"  - {w}")
    if a.list:
        return 0

    try:
        need(workloads,
             f"no workload under {root / 'output' / 'astra_logs'} has a "
             f"{a.sweep!r} sub-directory (or the filters removed them all)")
        # The x axis IS bandwidth divided out. A sweep that moves anything else
        # would put two different knobs on one beta and the collapse would be
        # meaningless rather than absent.
        need("bandwidth" in a.sweep,
             f"--sweep {a.sweep!r}: beta only varies through bandwidth. In a "
             f"cc/buffer/incast/oversub sweep every run has the same beta and "
             f"this plot is a vertical stripe -- use the sweep's own analyzer.")

        outdir = (Path(a.out) if a.out else
                  root / "results" / "sweep_analysis" / "roofline_compare" / a.sweep)

        frames = []
        print(f"\nNormalising {len(workloads)} workload(s):")
        for w in workloads:
            s = normalise(load_workload(root, w, a.sweep, a.pattern))
            frames.append(s)
            print(f"  + {_short(w):<26} T_comp={s['Tcomp_ns'].iloc[0] * MS:8.1f} ms"
                  f"  beta {s['beta'].min():.2f}..{s['beta'].max():.2f}"
                  f"  phi {s['phi_overlap'].min():+.2f}..{s['phi_overlap'].max():+.2f}")

        df = pd.concat(frames, ignore_index=True)
        front = ["workload", "run_dir", "bandwidth", "Tcomp_ns",
                 "Tnet_ideal_ns", "beta", "y_norm", "y_overlap", "y_serial",
                 "phi_overlap", "kv_agg_efficiency", "kv_senders", "makespan_ms"]
        df = df[[c for c in front if c in df.columns]
                + [c for c in df.columns if c not in front]]
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "summary.csv", index=False)

        written: list[Path] = []
        fig_roofline(df, outdir, written)
        fig_overlap(df, outdir, written)
        fig_residual(df, outdir, written)
        fig_efficiency(df, outdir, written)

        # The collapse, as a number rather than a picture: at a given bandwidth
        # the workloads' makespans differ by an order of magnitude, and after
        # normalisation they should not.
        print("\nCollapse quality -- spread of y across workloads at equal bandwidth:")
        print(f"  {'bx':>6}  {'makespan max/min':>17}  {'y max/min':>10}  {'mean phi':>9}")
        for bw, g in df.groupby("bandwidth"):
            m, y = g["makespan_ns"], g["y_norm"]
            print(f"  {bw:>6.0f}  {m.max() / m.min():>17.2f}x  "
                  f"{y.max() / y.min():>9.2f}x  {g['phi_overlap'].mean():>+9.2f}")

        print(f"\nWrote {outdir}:")
        print("  summary.csv")
        for p in written:
            print(f"  {p.name}")
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
