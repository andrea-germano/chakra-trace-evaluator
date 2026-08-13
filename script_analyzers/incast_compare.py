#!/usr/bin/env python3
"""
incast_compare -- same fabric, same fan-ins, different burst lengths: the
amortisation of the incast transient.

incast_sweep answers "what does the fan-in cost on ONE workload". On the
16reqs/512prompt run the answer is ~2 ms of receiving-link idle against a
135.6 ms serialisation floor -- invisible. This script puts that run side by
side with the 1reqs/{512,128,32}prompt reruns of the SAME topologies, buffers
and congestion control, where the floor shrinks 16x/64x/256x while the
transient does not, and locates the regime where the incast stops being
amortised.

Reads the summary.csv files incast_sweep already wrote (one per workload);
nothing is re-analysed here, so run incast_sweep on each workload first. One
row per (workload, topology), read at the LARGEST buffer of that pair -- the
converged, loss-free end of each sweep -- so what differs between rows is the
burst length alone.

The x axis is the serialisation floor (KV bytes per receiver / link rate):
the burst length expressed in wire-time. It is identical across the
topologies of one workload by construction (same decode placement, evenly
sharded KV), which is what makes the overlay meaningful; a need() enforces it.

RESOLUTION GUARD. The fan-in-1 control measures the harness's own noise floor
(per-flow setup, scheduling grain, latency), and that noise is amortised
exactly like the transient, so it too grows as a share of a shrinking floor.
Where the control's residue exceeds CONTROL_LIMIT of the floor, nothing at
that burst length is attributable to congestion any more: those points are
drawn hollow inside a shaded band and must not be quoted.

Output
------
    <out>/01_transient_amortization.png  the incast term absolute, as a share
                                         of the floor, and the bottleneck link
                                         efficiency, vs the floor
    <out>/02_buffer_cost.png             what the fan-in costs in BUFFER vs the
                                         floor: peak queue at the bottleneck,
                                         smallest PAUSE-free and smallest
                                         loss-free buffer (whole sweep per row)
    <out>/03_model_clock.png             does it reach the MODEL? release lag
                                         (tok2 − gate)/ITL, the gate stage's
                                         incast, and the decode KV stall
    <out>/summary.csv                    one row per (workload, topology)

Usage
-----
    python3 incast_compare.py
    python3 incast_compare.py --summaries path/a/summary.csv path/b/summary.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import paths
from utils.cli import Abort, need, warn
from utils.paths import fresh_dir
from utils.plots import save_fig

NAN = float("nan")

# Above this, the fan-in-1 control's residue is no longer small against the
# floor and the burst length is below the harness's resolution (see module
# docstring). 20% is far above every valid point measured (<=5%) and far below
# the broken one (284%), so the verdict does not sit on the threshold.
CONTROL_LIMIT = 0.20

# Same per-level colour policy as incast_sweep: viridis order by fan-in, with
# T4 pulled off the invisible pale-yellow endpoint.
_LEVEL_COLOR_OVERRIDE = {"T4": "#e8590c"}

# How the workload moved the KV: layer by layer as the prefill produced it
# ('stream'), or as one per-request transfer after the prefill finished
# ('bulk', which is what vLLM/NIXL does). Two runs of the same burst length in
# different modes are DIFFERENT experiments and sit on the same x, so they get
# different curves -- drawn as one they read as a 6x instability of the
# measurement at 8192 tokens, which is the opposite of what they show.
# (linestyle, marker): the marker carries its share because a mode present at a
# single burst length draws no line to be dashed.
_MODE_STYLE = {"stream": ("-", "o"), "bulk": ("--", "s")}


def _mode_of(name: str) -> str:
    return "bulk" if re.search(r"(?:^|_)bulk(?:_|$)", name) else "stream"


def _gate_stage_incast(row: pd.Series, cols) -> float:
    """Idle of the decode stage whose last KV arrival IS the gate (highest
    decN_kv_ready), starved + incast. On the gate stage of these runs starved
    is 0 -- its prefill feeder is the last stage, which forwards no PP
    activation -- so this is the incast term that can land on tok2."""
    gst, best = None, -1.0
    for si in range(8):
        c = f"dec{si}_kv_ready_ns"
        if c in cols and pd.notna(row.get(c)) and float(row[c]) >= best:
            gst, best = si, float(row[c])
    if gst is None:
        return NAN
    return (float(row.get(f"kv_starved_dec{gst}_ms", NAN))
            + float(row.get(f"kv_incast_dec{gst}_ms", NAN)))


def _tokens_of(name: str) -> int | None:
    """Total prompt tokens of a workload dir name: <n>reqs x <p>prompt."""
    m = re.search(r"(\d+)reqs_(\d+)prompt", name)
    return int(m.group(1)) * int(m.group(2)) if m else None


def load_summaries(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        name = f.parent.name
        tok = _tokens_of(name)
        need(tok is not None,
             f"{f}: cannot read <n>reqs_<p>prompt from the directory name "
             f"{name!r}; the token count is the comparison axis.")
        d = pd.read_csv(f)
        need({"level", "incast_degree", "buffer_mb", "kv_floor_ms",
              "kv_incast_mean_ms"} <= set(d.columns),
             f"{f}: missing incast_sweep columns -- regenerate it with the "
             f"current incast_sweep.py.")
        # one workload = one floor: enforced, not assumed. The tolerance is
        # loose enough for per-packet ceil() rounding (sub-us on ms floors),
        # tight enough to catch an actually uneven shard.
        fl = d["kv_floor_ms"].dropna()
        need(len(fl) and float(fl.max() - fl.min()) < 1e-3 * float(fl.max()),
             f"{f}: kv_floor_ms differs across its own rows "
             f"({fl.min():g}..{fl.max():g} ms); the workload does not shard "
             f"the KV evenly and cannot sit on one x position.")
        for lv, g in d.groupby("level"):
            big = g.loc[g["buffer_mb"].idxmax()]

            def free_from(col):
                """Smallest buffer from which the metric stays zero for every
                larger buffer of this (workload, level) -- a knee read off the
                tail, so a mid-sweep zero next to a lossy larger buffer does
                not count. NaN = never within the swept range."""
                gg = g.sort_values("buffer_mb")
                v = gg[col].fillna(0).to_numpy()
                b = gg["buffer_mb"].to_numpy()
                out = NAN
                for i in range(len(v) - 1, -1, -1):
                    if v[i] == 0:
                        out = float(b[i])
                    else:
                        break
                return out

            rows.append({
                "workload": name, "tokens": tok, "mode": _mode_of(name),
                "level": lv, "fan_in": int(big["incast_degree"]),
                "buffer_mb": float(big["buffer_mb"]),
                "floor_ms": float(fl.mean()),   # one x per workload (see need)
                "incast_ms": float(big["kv_incast_mean_ms"]),
                "starved_ms": float(big.get("kv_starved_mean_ms", NAN)),
                "eff_pct": float(big.get("bn_eff_pct", NAN)),
                "qpeak_mb": float(big.get("bn_qpeak_mb", NAN)),
                "pause_free_mb": free_from("total_pause_frames"),
                "loss_free_mb": free_from("dropped_packets"),
                "n_buffers": int(g["buffer_mb"].nunique()),
                # -- the model-clock view (figure 03) --------------------- #
                # release lag: tok2 − last KV arrival. ~0 = the KV tail
                # releases the token (network-bound, the gate stage's idle
                # lands on tok2 1:1); ~ITL = compute was pacing anyway.
                "lag_ms": (float(big["tok2_ns"] - big["kv_gate_ns"]) / 1e6
                           if {"tok2_ns", "kv_gate_ns"} <= set(d.columns)
                           else NAN),
                "itl_ms": float(big.get("itl_steady_ms", NAN)),
                "gate_incast_ms": _gate_stage_incast(big, d.columns),
                "stall_ms": float(big.get("decode_kv_stall_ms", NAN)),
            })
    s = pd.DataFrame(rows).sort_values(["tokens", "fan_in"])
    s["incast_of_floor_pct"] = 100 * s["incast_ms"] / s["floor_ms"]
    return s.reset_index(drop=True)


def mark_resolution(s: pd.DataFrame) -> pd.DataFrame:
    """A burst length is `resolved` iff its fan-in-1 control keeps its residue
    under CONTROL_LIMIT of the floor. No control at that length -> unresolved,
    conservatively."""
    ok = {}
    for tok, g in s.groupby("tokens"):
        ctrl = g[g["fan_in"] == 1]
        ok[tok] = bool(len(ctrl)) and bool(
            (ctrl["incast_of_floor_pct"] <= 100 * CONTROL_LIMIT).all())
    s["resolved"] = s["tokens"].map(ok)
    return s


def _series(s: pd.DataFrame) -> list[tuple]:
    """(label, rows, colour, linestyle, marker) for every curve the figures
    draw: one per (topology, transfer mode). Colour keeps carrying the
    topology, linestyle and marker carry the mode.

    With a single mode loaded -- the usual case -- this returns exactly one
    solid curve per topology, i.e. the behaviour that predates the bulk runs,
    label included. The three figures share it because they had three copies
    of the same colour map and the same loop."""
    levels = sorted(s["level"].unique(),
                    key=lambda l: s.loc[s["level"] == l, "fan_in"].iloc[0])
    cmap = plt.get_cmap("viridis")
    n = max(len(levels) - 1, 1)
    colors = {l: _LEVEL_COLOR_OVERRIDE.get(l, cmap(i / n))
              for i, l in enumerate(levels)}
    modes = [m for m in _MODE_STYLE if (s["mode"] == m).any()]
    out = []
    for lv in levels:
        for m in modes:
            g = s[(s["level"] == lv) & (s["mode"] == m)].sort_values("floor_ms")
            if g.empty:
                continue
            lab = f"{lv} · fan-in {g['fan_in'].iloc[0]}"
            if len(modes) > 1:
                lab += f" · {m}"
            out.append((lab, g, colors[lv], *_MODE_STYLE[m]))
    return out


def fig_amortization(s: pd.DataFrame, outdir: Path,
                     written: list[Path]) -> None:
    """01 -- three views of the same rows, shared log-x = the floor. Left: the
    incast term in ms (near-horizontal = a fixed transient). Middle: the same
    term as a share of the floor (what the fixed transient does to a shrinking
    burst). Right: bottleneck link efficiency during the handover. Hollow
    markers + shaded band = below the control's resolution; the numbers there
    are harness noise, not congestion."""
    series = _series(s)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("incast_ms", "Incast term (ms)", None),
              ("incast_of_floor_pct", "Incast term (% of the floor)", "log"),
              ("eff_pct", "Bottleneck link efficiency (%)", None)]
    bad = s.loc[~s["resolved"], "floor_ms"]
    for ax, (col, ylab, yscale) in zip(axes, panels):
        for lab, g, colr, ls, mk in series:
            ax.plot(g["floor_ms"], g[col], ls, color=colr,
                    label=lab, zorder=2)
            r = g[g["resolved"]]
            ax.plot(r["floor_ms"], r[col], mk, ms=6, color=colr, zorder=3)
            u = g[~g["resolved"]]
            ax.plot(u["floor_ms"], u[col], mk, ms=6, mfc="white",
                    color=colr, zorder=3)
        if len(bad):
            ax.axvspan(bad.min() * 0.55, bad.max() * 1.8, color="0.88",
                       zorder=0)
        ax.set_xscale("log")
        if yscale:
            ax.set_yscale(yscale)
        xs = sorted(s["floor_ms"].unique())
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{x:.3g}\n{int(s.loc[s['floor_ms'] == x, 'tokens'].iloc[0])} tok"
             for x in xs], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("Serialisation floor (ms) / prompt tokens")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, which="major")
    if s["incast_of_floor_pct"].max() > 100:
        axes[1].axhline(100, color="0.4", lw=0.8, ls=":")
        axes[1].annotate("= floor", (axes[1].get_xlim()[0], 100),
                         fontsize=7, color="0.4", va="bottom")
    axes[0].legend(fontsize=8, title="topology")
    if len(bad):
        axes[1].text(0.03, 0.96, "shaded: below the control's\nresolution",
                     transform=axes[1].transAxes, fontsize=8, va="top",
                     color="0.35")
    fig.suptitle("The incast term against the burst length, at each sweep's "
                 "largest buffer", y=1.03)
    save_fig(fig, outdir, "01_transient_amortization.png", written)


def fig_buffer_cost(s: pd.DataFrame, outdir: Path,
                    written: list[Path]) -> None:
    """02 -- the buffer axis of the same comparison: the standing queue the
    fan-in builds at the convergence point (largest buffer, so nothing was
    truncated) and the smallest buffer at which the fabric stops pausing /
    stops dropping, vs the burst length. These are read over each row's WHOLE
    buffer family, which is what the congruent grid buys. The resolution band
    of figure 01 does not apply here: queue occupancy and PAUSE counts are
    fabric-side counters, meaningful at every burst length."""
    series = _series(s)
    bufs = [8, 12, 16, 24, 32, 48, 64]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("qpeak_mb", "Peak queue at the bottleneck (MiB)", "log"),
              ("pause_free_mb", "Smallest PAUSE-free buffer (MiB)", None),
              ("loss_free_mb", "Smallest loss-free buffer (MiB)", None)]
    for ax, (col, ylab, yscale) in zip(axes, panels):
        for lab, g, colr, ls, mk in series:
            ax.plot(g["floor_ms"], g[col], ls, marker=mk, ms=6, color=colr,
                    label=lab)
            miss = g[g[col].isna()]
            if len(miss) and col != "qpeak_mb":
                ax.plot(miss["floor_ms"], [max(bufs)] * len(miss), "^",
                        ms=8, mfc="white", color=colr)
        ax.set_xscale("log")
        if yscale:
            ax.set_yscale(yscale)
        else:
            ax.set_yticks(bufs)
            ax.set_ylim(0, max(bufs) * 1.1)
        xs = sorted(s["floor_ms"].unique())
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{x:.3g}\n{int(s.loc[s['floor_ms'] == x, 'tokens'].iloc[0])} tok"
             for x in xs], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("Serialisation floor (ms) / prompt tokens")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, which="major")
    axes[0].legend(fontsize=8, title="topology")
    if s[["pause_free_mb", "loss_free_mb"]].isna().any().any():
        axes[1].text(0.03, 0.96, "hollow ▲ at the top: never within\n"
                     "the swept 8–64 MiB range",
                     transform=axes[1].transAxes, fontsize=8, va="top",
                     color="0.35")
    fig.suptitle("The fan-in's buffer cost against the burst length", y=1.03)
    save_fig(fig, outdir, "02_buffer_cost.png", written)


def fig_model_clock(s: pd.DataFrame, outdir: Path,
                    written: list[Path]) -> None:
    """03 -- whether the fan-in's cost reaches the model's clock, and where.
    Left: the release lag (tok2 − last KV arrival) over the steady ITL -- 0
    means the KV tail releases the second token (network-bound), 1 means
    decode compute was pacing anyway. Middle: the gate stage's incast, i.e.
    the term that lands on tok2 1:1 exactly where the left panel sits near 0.
    Right: how long decode sits waiting for KV in total (floor + idle minus
    everything masked) -- the fan-in's term is the middle panel, the rest of
    this is serialisation. Resolution band as in figure 01, middle panel only:
    the lag and the stall are direct clock readings, valid at every length."""
    s = s.assign(lag_over_itl=s["lag_ms"] / s["itl_ms"])
    series = _series(s)
    bad = s.loc[~s["resolved"], "floor_ms"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("lag_over_itl", "(tok2 − last KV arrival) / steady ITL", None),
              ("gate_incast_ms", "Incast on the gate stage (ms)", None),
              ("stall_ms", "Decode KV stall (ms, symlog)", "symlog")]
    for j, (ax, (col, ylab, yscale)) in enumerate(zip(axes, panels)):
        for lab, g, colr, ls, mk in series:
            ax.plot(g["floor_ms"], g[col], ls, color=colr, label=lab)
            r = g[g["resolved"]] if j == 1 else g
            u = g[~g["resolved"]] if j == 1 else g.iloc[0:0]
            ax.plot(r["floor_ms"], r[col], mk, ms=6, color=colr)
            ax.plot(u["floor_ms"], u[col], mk, ms=6, mfc="white", color=colr)
        if j == 1 and len(bad):
            ax.axvspan(bad.min() * 0.55, bad.max() * 1.8, color="0.88",
                       zorder=0)
        ax.set_xscale("log")
        if yscale:
            ax.set_yscale(yscale, linthresh=1)
        xs = sorted(s["floor_ms"].unique())
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{x:.3g}\n{int(s.loc[s['floor_ms'] == x, 'tokens'].iloc[0])} tok"
             for x in xs], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("Serialisation floor (ms) / prompt tokens")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, which="major")
    axes[0].axhline(0, color="0.4", lw=0.8, ls=":")
    axes[0].axhline(1, color="0.4", lw=0.8, ls=":")
    axes[0].text(0.02, 0.06, "0 = KV tail releases tok2 (network-bound)\n"
                 "1 = decode compute paces (incast absorbed)",
                 transform=axes[0].transAxes, fontsize=8, color="0.35",
                 va="bottom")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].legend(fontsize=8, title="topology", loc="center right")
    fig.suptitle("What the model's clock sees of the fan-in, against the "
                 "burst length", y=1.03)
    save_fig(fig, outdir, "03_model_clock.png", written)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(paths.ROOT), type=Path,
                    help=f"project root (default: {paths.ROOT})")
    ap.add_argument("--summaries", nargs="+", default=None, type=Path,
                    help="explicit summary.csv paths (default: every "
                         "results/sweep_analysis/incast*/<workload>/summary.csv)")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/sweep_analysis/"
                         "incast_compare)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    files = (a.summaries if a.summaries else
             sorted(root.glob("results/sweep_analysis/incast/*/summary.csv"))
             + sorted(root.glob(
                 "results/sweep_analysis/incast_short/*/summary.csv")))
    outdir = Path(a.out) if a.out else \
        root / "results" / "sweep_analysis" / "incast_compare"

    try:
        need(files, "no summary.csv found under results/sweep_analysis/"
                    "incast{,_short}/ -- run incast_sweep first.")
        for f in files:
            print(f"  + {f.relative_to(root) if f.is_relative_to(root) else f}")
        s = mark_resolution(load_summaries(files))
        need(s["tokens"].nunique() > 1,
             "only one burst length found; there is nothing to compare. "
             "Run the 1reqs workloads (experiments_incast_short.txt).")

        fresh_dir(outdir)
        s.to_csv(outdir / "summary.csv", index=False)
        written: list[Path] = []
        fig_amortization(s, outdir, written)
        fig_buffer_cost(s, outdir, written)
        fig_model_clock(s, outdir, written)

        pd.set_option("display.width", 200)
        print("\n============= INCAST vs BURST LENGTH =============")
        print(s[["tokens", "level", "fan_in", "buffer_mb", "floor_ms",
                 "incast_ms", "incast_of_floor_pct", "eff_pct", "qpeak_mb",
                 "pause_free_mb", "loss_free_mb", "n_buffers",
                 "resolved"]].to_string(index=False))
        part = s[s["n_buffers"] < s["n_buffers"].max()]
        if len(part):
            print(f"\n! partial buffer families (excluded/unobtainable cells): "
                  + ", ".join(f"{r['level']}@{int(r['tokens'])}tok "
                              f"({int(r['n_buffers'])} buffers)"
                              for _, r in part.iterrows()))
        u = s[~s["resolved"]]
        if len(u):
            print(f"\n! {sorted(u['tokens'].unique())} tokens: the fan-in-1 "
                  f"control exceeds {100 * CONTROL_LIMIT:.0f}% of the floor "
                  f"there -- below resolution, drawn hollow, not quotable.")
        inc = s[(s["resolved"]) & (s["fan_in"] > 1)]
        if len(inc):
            w = inc.loc[inc["incast_of_floor_pct"].idxmax()]
            print(f"\nlargest resolved incast share: {w['incast_of_floor_pct']:.1f}% "
                  f"of the floor ({w['incast_ms']:.2f} of {w['floor_ms']:.2f} ms) "
                  f"at {int(w['tokens'])} tokens, {w['level']} fan-in "
                  f"{int(w['fan_in'])}.")
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
