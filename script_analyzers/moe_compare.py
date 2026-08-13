#!/usr/bin/env python3
"""
moe_compare — one fabric, several MODELS: what changes when the FFN becomes a
mixture of experts.

The other cross-model companions (bandwidth_compare, buffer_compare) hold the
model fixed and sweep the fabric. This one does the opposite: ONE fabric config,
one request blob, and the model as the only moving part -- dense vs MoE -- so the
difference between two runs is attributable to the expert layer and nothing else.

The comparison it is built for
--------------------------------------------------------------------------------
The reference arms live in configs/mlsynth/ and are iso-FLOPs by construction:
the dense FFN (intermediate 13824, sharded over tp) becomes 8 WHOLE experts of
6912 with top_k=2, so 2 x 6912 = 13824 and a token costs the same FLOPs either
way. What moves is the shape of the block, and it depends on WHERE the expert
group lives (both shapes are the verified vLLM-main behaviour):

    dense           : attention -> all-reduce -> FFN -> all-reduce
    MoE, dp=1       : attention -> all-reduce -> gate -> masked local experts
                      -> all-reduce            (NO a2a: vLLM activates no
                      all-to-all kernel without dp>1 -- use_all2all_kernels)
    MoE, tp>1 dp>1  : attention -> reduce-scatter -> gate -> A2A dispatch ->
                      experts -> A2A combine -> weighted sum -> all-gather
                      (vLLM EP + use_sequence_parallel_moe, DeepEP backends)

In the wide-EP arm two things move at once. The second all-reduce disappears:
with whole experts (etp = 1) a token comes back complete from the rank that owns
its expert, so there is no partial sum left to reconcile. And the first one is
SPLIT around the expert phase: each tensor rank owns and dispatches 1/tp of the
tokens. Reduce-scatter + all-gather move together exactly what one all-reduce
moves, so a wide-EP block still puts half a dense block's collective volume on
the wire; half of it just sits after the experts instead of all of it before.
Wire counts are one copy per selected expert in BOTH phases (the DeepEP
low-latency shape, per-expert buffers). vLLM's prefill would use the
high-throughput layout instead, where a token travels once per destination
rank, so the prefill a2a here is an overestimate, never an underestimate.

The dp=1 arm is therefore the CONTROL: its network trace is dense-shaped by
construction (two all-reduces, op=attn/ffw, zero A2A), so any dense-vs-MoE gap
it shows is pure compute -- whole-expert weights against a handful of decode
tokens -- and any gap the wide-EP arm adds on top of it is pure fabric.

Iso-FLOPs is NOT iso-time, and the runs say so: experts are kept whole (etp = 1),
so a device stores num_experts / ep of them -- whole experts where the dense rank
held a 1/tp shard of one FFN. The cost model charges only the weights of experts
actually HIT by a token (grouped-GEMM semantics), but at these batch sizes every
local expert is hit, so decode reads several times the dense per-rank weight
bytes at the same FLOPs. Prefill does not care (thousands of tokens per rank,
compute-bound either way). Decode does: a handful of tokens against whole weight
matrices is memory-bound, the arithmetic intensity drops by the same factor, and
the expert FFN takes correspondingly longer. That is the real MoE serving cost
and it is not a fabric cost -- which is why the compute profile is in the summary
next to the communication mix, and why reading the latency figure alone would
attribute it to the wrong thing.

The arms, in the order the report ranks them:

    dense tp2/pp2      the reference every delta is taken against.
    MoE   tp2/pp2      dp=1 -> masked path, zero a2a: the compute-only control.
    MoE   tp2/dp2      EP = tp*dp = 4 per pool: attention replicas on the dp
                       axis, experts spread over the whole stage. Two scale-up
                       domains, so 2 of every 3 a2a edges leave the domain and
                       cross the scale-out plane. This is vLLM's wide-EP
                       deployment, and the regime where an all-to-all becomes a
                       fabric problem.
    routing            balanced Dirichlet(alpha=100) -- statistically flat with
                       realistic expert pairing; NOT the round-robin `uniform`,
                       whose consecutive-expert picks always land both copies of
                       a token on one rank and halve the dispatch artificially --
                       against Dirichlet(alpha=0.5) (hot experts): same total
                       volume, different distribution over the EP ranks.

Run it once per fabric: --tag T2_bx100_dcqcn_buf32 (modest flat 100G fabric)
and --tag T2.3_bx400_dcqcn_buf32 (MoE-optimized dual-plane, TOPOLOGIES.md
§T2.3). The pair of reports answers how much of the wide-EP penalty a
deployable expert-plane upgrade buys back, with the dense and masked arms as
controls that should barely move.

Why the second fabric is DUAL-plane, and what makes the pair comparable at all.
A cross-fabric reading is only meaningful if the fabric change touches the
expert traffic and nothing else, and the thing that ruins it is the KV cache:
it is the single largest transfer in the run (6.25 GiB, seconds on the wire
against milliseconds of collectives), it sits inside makespan and decode start,
and it is a POOL-CROSSING flow -- so any topology that fattens a pool's access
links to help the all-to-all also hands the KV stream a wider pipe, and the
dense reference moves with it. Every single-plane attempt was measured and
fails (TOPOLOGIES.md §T2.3 lists them with their dense controls): one fat
uplink gives the right inter-pool aggregate but loses T2's per-source 100G cap
(dense -26%), one thin uplink keeps a 100G cap but cuts the aggregate to 1/4
and buries every arm under a ~537 ms serialization floor, and a LAG splits one
source across members and loses the cap anyway. T2.3 instead gives the experts
their own 400 Gbps intra-pool plane and leaves the KV cache on a SECOND 100 Gbps
NIC per GPU whose leaf/spine plane reproduces T2's KV path exactly -- same
per-source cap (it is enforced at the NIC, where T2 enforces it), same 4x100G
aggregate, same 10 us of latency. The planes cannot mix: ns-3 routes by hop
count, intra-pool is 2 hops through the ToR against 4 across the KV plane, and
the two pool ToRs have no uplink between them.

That leaves ONE residual, worth quoting next to the numbers: PP activations are
intra-pool too, so they ride the fat plane and the dense TTFT improves ~4%. It
is a real pool-fattening benefit rather than a KV artefact, and it is the whole
of the dense movement. A second thing changes for the MoE arms only: KV no
longer contends with the all-to-all on a shared NIC as it does in T2. That is
traffic segregation, which is part of what the topology is buying.

Figures (results/sweep_analysis/moe_compare/<tag>/ by default)
--------------------------------------------------------------------------------
  01  LATENCY          TTFT | decode start | second token | steady ITL | makespan,
                       one bar group per arm, dense first as the reference.
  02  COMMUNICATION    bytes and on-wire time per class, counted once each
                       (astra_analyzer.aggregate_comm_time), stacked per arm --
                       where the TP all-reduce goes and what A2A costs instead.
  03  LOAD BALANCE     per-rank compute busy time, one panel per arm. A skewed
                       router shows up here as a straggler: the expert group runs
                       at the speed of its hottest rank, and the gap is the
                       imbalance the fabric cannot fix.
  04  EXPOSURE         per rank, communication hidden behind compute vs exposed
                       on the critical path. The MoE question in one figure: the
                       dispatch gates the expert FFN that follows it, so unlike a
                       KV stream it has very little to hide behind.

summary.json / summary.txt carry every number in the figures, plus the deltas
against the dense arm, so the table can be pasted without re-deriving anything.

    python3 moe_compare.py --sweep moe_compare --tag T2_bx100_dcqcn_buf32
    python3 moe_compare.py --sweep moe_compare --tag T2.3_bx400_dcqcn_buf32

Nothing here re-implements a metric: load_run / compute_metrics /
aggregate_comm_time / per_sys_utilisation are astra_analyzer's, exactly as
buffer_compare borrows buffer_sweep's analyse_sweep. What this file owns is the
arm ordering, the deltas and the four cross-arm figures.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import astra_analyzer as aa
from utils import astra, paths
from utils.cli import Abort, drain_warnings, need, warn
from utils.plots import save_fig

KIND = "moe_compare"

# Class colours are astra_analyzer's, so a class is the same hue in the
# single-run figures and here.
CLASS_ORDER = ("TP", "A2A", "KV", "PP", "DECFB", "KVREQ")


# --------------------------------------------------------------------------- #
#  Arms
# --------------------------------------------------------------------------- #
def arm_label(workload: str) -> str:
    """Short label for a workload directory name.

    The generator's names already encode everything (model, EP layout, routing),
    they are just too long for an axis: llama2_13b_moe8e2k-skew_p-tp2dp2_... ->
    'MoE skew tp2dp2'. Unknown shapes fall back to the raw name rather than being
    guessed into a wrong bucket."""
    moe = "moe8e2k" in workload
    skew = "-skew" in workload
    shape = ""
    for token in workload.split("_"):
        if token.startswith("p-"):
            shape = token[2:]
    if not moe:
        return f"dense {shape}".strip()
    return f"MoE {'skew' if skew else 'bal'} {shape}".strip()


def collective_profile(df) -> list[dict]:
    """The TP-group collectives of one arm, split by the operation that emitted them.

    Needed because the `TP` class stopped meaning one thing. A dense block -- and
    the masked dp=1 MoE arm, whose collectives are dense-shaped by construction --
    all-reduces twice per layer (`op=attn`, `op=ffw`); a wide-EP MoE block
    reduce-scatters once and all-gathers once (`op=rs`, `op=ag`). The class total
    alone would compare four payloads against two and call them equal.

    Read the bytes as what utils.astra reports for every collective -- the payload
    DECLARED on the node, not its volume on the wire. Declared is not the same
    quantity for every type, and the difference is not cosmetic (ASTRA-sim's
    Ring.cc is the authority):

        all-reduce      declares the tensor S,  moves 2(n-1)/n * S
        reduce-scatter  declares the tensor S,  moves   (n-1)/n * S
        all-gather      declares the SHARD S/n, moves   (n-1)   * S/n

    So a dense block declares 2S per layer and moves 2 * 2(n-1)/n * S; a MoE block
    declares S + S/n and moves 2 * (n-1)/n * S -- half the dense wire volume, which
    is the point. Comparing the declared columns alone would say something else."""
    out = []
    tp = df[(df["op_class"] == "TP")]
    if tp.empty:
        return out
    for op, g in astra.collapse_collectives(tp).groupby("op", dropna=False):
        n = g["n_ranks"].astype(float)
        declared = g["comm_size"].astype(float)
        out.append({"op": str(op), "count": int(len(g)),
                    "declared_bytes": float(declared.sum()),
                    "wire_bytes": float((declared * _ring_wire_factor(str(op), n)).sum()),
                    "total_dur": float(g["duration"].sum())})
    out.sort(key=lambda r: -r["wire_bytes"])
    return out


def _ring_wire_factor(op: str, n):
    """Bytes the WHOLE group injects per unit of declared size, for one ring
    collective -- steps x message x participants:

        reduce-scatter   (n-1)  steps x S/n   x n  =  (n-1)
        all-gather       (n-1)  steps x shard x n  =  (n-1) * n
        all-reduce      2(n-1)  steps x S/n   x n  = 2(n-1)

    `op` is MLSynth's name for the operation that emitted the node: 'rs' and 'ag'
    are the sequence-parallel pair of a MoE block, anything else is an all-reduce."""
    if op == "rs":
        return n - 1
    if op == "ag":
        return (n - 1) * n
    return 2 * (n - 1)


def compute_profile(df) -> list[dict]:
    """Per (pool, op) compute: total time, arithmetic intensity, memory-bound share.

    This is where the iso-FLOPs design gets checked, and where the MoE's real
    decode cost shows up. Same FLOPs does NOT mean same time: a device holds
    num_experts/ep WHOLE experts, so in decode -- a handful of tokens against the
    full weight matrices -- the expert FFN reads several times the dense per-rank
    weights at the same FLOPs. The roofline turns that into time, and the run
    labels it: is_memory_bound with the operation intensity to say by how much."""
    out = []
    comp = df[df["is_compute"]]
    for (pl, op), g in comp.groupby(["pl", "op"], dropna=False):
        out.append({
            "pool": "prefill" if pl == "p" else "decode" if pl == "d" else str(pl),
            "op": str(op),
            "total_dur": float(g["duration"].sum()),
            "count": int(len(g)),
            "mem_bound_frac": float(g["is_memory_bound"].mean()),
            "op_intensity": float(g["operation_intensity"].mean()),
        })
    out.sort(key=lambda r: (r["pool"], -r["total_dur"]))
    return out


def sort_key(workload: str) -> tuple:
    """Dense first (it is the reference every delta is taken against), then MoE
    by EP width, uniform routing before skewed."""
    moe = "moe8e2k" in workload
    wide = "dp2" in workload
    skew = "-skew" in workload
    return (moe, wide, skew, workload)


def collect(root: Path, sweep: str, tag: str, keep, drop) -> list[dict]:
    """One entry per workload that ran (sweep, tag), read through
    astra_analyzer's own reader."""
    found = paths.discover_workloads(root, sweep, "astra")
    if keep:
        found = [w for w in found if any(fnmatch.fnmatch(w, p) for p in keep)]
    if drop:
        found = [w for w in found if not any(fnmatch.fnmatch(w, p) for p in drop)]
    need(found, f"no workload under {root}/output/astra_logs has a '{sweep}' directory")

    arms = []
    for workload in sorted(found, key=sort_key):
        sp = paths.SweepPaths(sweep=sweep, workload=workload, root=root)
        run_dir = sp.astra_run(tag)
        if not run_dir.is_dir():
            warn(f"{workload}: no run at {run_dir} — arm skipped")
            continue
        df = aa.load_run(run_dir)
        if df is None or df.empty:
            warn(f"{workload}: no readable stats_sys*.csv in {run_dir} — arm skipped")
            continue
        roles = aa.sys_roles(df)
        arms.append({
            "workload": workload,
            "label": arm_label(workload),
            "run_dir": str(run_dir),
            "metrics": aa.compute_metrics(df, roles),
            "comm": aa.aggregate_comm_time(df),
            "collectives": collective_profile(df),
            "compute": compute_profile(df),
            "util": aa.per_sys_utilisation(df, roles),
            "n_ranks": int(df["sys_id"].nunique()),
        })
    need(arms, f"none of the discovered workloads has a run at tag {tag!r}")
    return arms


def deltas(arms: list[dict]) -> None:
    """Add each arm's latency ratios against the first (dense) arm, in place.
    Ratios, not nanoseconds: the arms are iso-FLOPs, so what is being asked is
    'how much slower', and that survives being quoted next to another fabric."""
    ref = arms[0]["metrics"]
    for a in arms:
        m = a["metrics"]
        a["delta"] = {
            k: (m[k] / ref[k] if ref.get(k) and m.get(k) else None)
            for k in ("ttft_ns", "decode_start_ns", "second_token_ns",
                      "tpot_steady_ns", "makespan_ns")
        }


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def chart_latency(arms, out, written):
    """01 — the five latency milestones, one bar group per arm."""
    keys = [("ttft_ns", "TTFT"), ("decode_start_ns", "decode start"),
            ("second_token_ns", "second token"), ("tpot_steady_ns", "steady ITL"),
            ("makespan_ns", "makespan")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.6))
    labels = [a["label"] for a in arms]
    x = np.arange(len(arms))
    colors = ["#5b8def"] + ["#00b8d4", "#0f766e", "#7c5cff", "#b45309"][:len(arms) - 1]
    for ax, (key, title) in zip(axes, keys):
        vals = np.array([a["metrics"].get(key) or np.nan for a in arms], dtype=float)
        ax.bar(x, vals / 1e6, color=colors[:len(arms)], edgecolor="white")
        ref = vals[0]
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            plain = i == 0 or not np.isfinite(ref) or not ref
            txt = f"{v/1e6:.1f} ms" if plain else f"{v/1e6:.1f} ms\n(x{v/ref:.2f})"
            ax.text(i, v / 1e6, txt, ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("ms")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.margins(y=0.2)
    fig.suptitle("Latency milestones — dense vs MoE, same fabric, same requests")
    save_fig(fig, out, "01_latency", written)


def chart_communication(arms, out, written):
    """02 — bytes and on-wire time per class, each transfer counted once.

    Bytes stack (they are one budget, and the point is that the total barely
    moves at EP = TP); time does NOT. The classes' aggregate durations span three
    orders of magnitude -- a KV stream of seconds next to an all-reduce of
    milliseconds -- so stacking them would draw one class and hide the rest.
    Grouped bars on a log axis instead: the comparison there is A2A against
    itself across arms, which is what the EP width does to the wire."""
    def value(a, cls, field):
        return next((r[field] for r in a["comm"] if r["cls"] == cls), 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [a["label"] for a in arms]
    x = np.arange(len(arms))
    present = [c for c in CLASS_ORDER
               if any(value(a, c, "total_bytes") for a in arms)]

    bottom = np.zeros(len(arms))
    for cls in present:
        vals = np.array([value(a, cls, "total_bytes") for a in arms]) / 1024 ** 3
        ax1.bar(x, vals, bottom=bottom, label=cls,
                color=aa.CAT_COLORS.get(cls, "#888"), edgecolor="white")
        bottom += vals
    for i, tot in enumerate(bottom):
        ax1.text(i, tot, f"{tot:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_title("bytes moved (each transfer counted once)")
    ax1.set_ylabel("GiB")

    width = 0.8 / max(len(present), 1)
    for j, cls in enumerate(present):
        vals = np.array([value(a, cls, "total_dur") for a in arms]) / 1e6
        # a class an arm does not have is left OUT, not floored: on a log axis a
        # floor would draw a bar where there is no traffic at all (dense has no
        # A2A, the pp=1 arms have no PP)
        vals[vals <= 0] = np.nan
        ax2.bar(x + (j - (len(present) - 1) / 2) * width, vals,
                width=width, label=cls, color=aa.CAT_COLORS.get(cls, "#888"),
                edgecolor="white")
    ax2.set_yscale("log")
    ax2.set_title("time on the wire, summed over transfers (send side, log)")
    ax2.set_ylabel("ms")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.margins(y=0.15)
    fig.suptitle("Communication mix — bytes moved and time on the wire, per class")
    save_fig(fig, out, "02_communication_mix", written)


def chart_balance(arms, out, written):
    """03 — per-rank compute busy time: the straggler a skewed router creates."""
    fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arms):
        rows = a["util"]
        busy = np.array([r["compute_busy"] for r in rows], dtype=float) / 1e6
        cols = ["#4f9bff" if r["role"] == "prefill" else "#c77dff" for r in rows]
        ax.bar(np.arange(len(rows)), busy, color=cols, edgecolor="white")
        pre = busy[[i for i, r in enumerate(rows) if r["role"] == "prefill"]]
        if len(pre):
            spread = pre.max() / pre.mean() if pre.mean() else np.nan
            ax.axhline(pre.mean(), color="#d1495b", ls="--", lw=1.2)
            ax.set_title(f"{a['label']}\nprefill max/mean = {spread:.3f}", fontsize=9)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([r["sys_id"] for r in rows], fontsize=7)
        ax.set_xlabel("rank")
    axes[0].set_ylabel("compute busy (ms)")
    fig.suptitle("Load balance — blue = prefill, purple = decode; dashed line = prefill mean")
    save_fig(fig, out, "03_load_balance", written)


def chart_exposure(arms, out, written):
    """04 — communication hidden behind compute vs exposed on the critical path."""
    fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, arms):
        rows = a["util"]
        idx = np.arange(len(rows))
        hidden = np.array([r["comm_hidden"] for r in rows], dtype=float) / 1e6
        exposed = np.array([r["comm_exposed"] for r in rows], dtype=float) / 1e6
        ax.bar(idx, hidden, color="#22c08a", edgecolor="white", label="hidden")
        ax.bar(idx, exposed, bottom=hidden, color="#ff9f43", edgecolor="white",
               label="exposed")
        tot = hidden + exposed
        frac = exposed.sum() / tot.sum() if tot.sum() else np.nan
        ax.set_title(f"{a['label']}\nexposed = {frac:.0%} of comm", fontsize=9)
        ax.set_xticks(idx)
        ax.set_xticklabels([r["sys_id"] for r in rows], fontsize=7)
        ax.set_xlabel("rank")
    axes[0].set_ylabel("communication time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Communication exposure — hidden vs exposed time, per sys")
    save_fig(fig, out, "04_exposure", written)


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def write_summary(arms, out: Path, sweep: str, tag: str, written: list[Path]) -> None:
    payload = {"sweep": sweep, "tag": tag,
               "reference_arm": arms[0]["workload"], "arms": []}
    lines = [f"moe_compare — sweep {sweep}, fabric {tag}",
             f"reference (all ratios are against it): {arms[0]['label']}", ""]
    head = (f"{'arm':<22}{'TTFT':>12}{'dec.start':>12}{'2nd tok':>12}"
            f"{'ITL':>12}{'makespan':>12}{'x ref':>8}")
    lines += [head, "-" * len(head)]
    for a in arms:
        m, d = a["metrics"], a["delta"]
        ratio = d.get("makespan_ns")
        ratio_txt = f"{ratio:.2f}" if ratio else "n/a"
        lines.append(
            f"{a['label']:<22}"
            f"{aa.fmt_time(m.get('ttft_ns')):>12}"
            f"{aa.fmt_time(m.get('decode_start_ns')):>12}"
            f"{aa.fmt_time(m.get('second_token_ns')):>12}"
            f"{aa.fmt_time(m.get('tpot_steady_ns')):>12}"
            f"{aa.fmt_time(m.get('makespan_ns')):>12}"
            f"{ratio_txt:>8}")
    lines.append("")
    for a in arms:
        lines.append(f"[{a['label']}]  {a['workload']}")
        for r in sorted(a["comm"], key=lambda r: -r["total_bytes"]):
            lines.append(f"    {r['cls']:<8}{r['total_bytes']/1024**3:>8.3f} GiB"
                         f"{aa.fmt_time(r['total_dur']):>12} on wire"
                         f"   n={r['count']}")
        for r in a["collectives"]:
            lines.append(f"    TP/{r['op']:<8}{r['declared_bytes']/1024**3:>8.3f} GiB dichiarati ->"
                         f"{r['wire_bytes']/1024**3:>8.3f} GiB sul filo"
                         f"{aa.fmt_time(r['total_dur']):>12}   n={r['count']}")
        wire = sum(r["wire_bytes"] for r in a["collectives"])
        if wire:
            lines.append(f"    TP totale sul filo: {wire/1024**3:.3f} GiB")
        for r in a["compute"]:
            lines.append(f"    {r['pool']:<8}{r['op']:<6}"
                         f"{aa.fmt_time(r['total_dur']):>12}"
                         f"   intensity {r['op_intensity']:>8.1f}"
                         f"   {'memory-bound' if r['mem_bound_frac'] > 0.5 else 'compute-bound'}")
        util = a["util"]
        pre = [r["compute_busy"] for r in util if r["role"] == "prefill"]
        if pre:
            lines.append(f"    prefill compute busy: mean {aa.fmt_time(np.mean(pre))}, "
                         f"max {aa.fmt_time(max(pre))}, max/mean {max(pre)/np.mean(pre):.3f}")
        comm_tot = sum(r["comm_union"] for r in util)
        comm_exp = sum(r["comm_exposed"] for r in util)
        if comm_tot:
            lines.append(f"    communication exposed on the critical path: "
                         f"{comm_exp/comm_tot:.1%} ({aa.fmt_time(comm_exp)} of "
                         f"{aa.fmt_time(comm_tot)})")
        lines.append("")
        payload["arms"].append({k: a[k] for k in
                                ("workload", "label", "run_dir", "metrics", "comm", "collectives", "compute",
                                 "util", "delta", "n_ranks")})

    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    written += [out / "summary.json", out / "summary.txt"]
    print(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_compare_arguments(ap, KIND, default_sweep="moe_compare")
    ap.add_argument("--tag", default="T2_bx100_dcqcn_buf32",
                    help="fabric config every arm ran (default: %(default)s)")
    a = ap.parse_args(argv)

    try:
        root = Path(a.root)
        arms = collect(root, a.sweep, a.tag, a.workloads, a.exclude)
        deltas(arms)
        out = paths.fresh_dir(Path(a.out) if a.out
                              else root / "results" / "sweep_analysis" / KIND / a.tag)
        written: list[Path] = []
        chart_latency(arms, out, written)
        chart_communication(arms, out, written)
        chart_balance(arms, out, written)
        chart_exposure(arms, out, written)
        write_summary(arms, out, a.sweep, a.tag, written)
        print(f"\n{len(written)} file(s) written to {out}")
    except Abort as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 2
    return drain_warnings(" — the numbers above are conditional on them")


if __name__ == "__main__":
    raise SystemExit(main())
