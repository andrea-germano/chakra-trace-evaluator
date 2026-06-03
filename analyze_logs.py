"""
parse_disagg_inference.py
=========================
Parse an ASTRA-sim (analytical backend) log produced from an MLSynth
**disaggregated prefill/decode** inference workload, and produce plots and
statistics that highlight the impact of KV-cache communication on
Time-To-First-Token (TTFT) and Inter-Token Latency (ITL).

The script is CONFIG-DRIVEN: pass the same MLSynth YAML you used to generate
the traces (`-c config.yaml`). From it the script derives the prefill/decode
NPU partition for ANY tensor-parallel (tp) / pipeline-parallel (pp) layout,
using MLSynth's deterministic NPU-id assignment:

    prefill pool : NPU ids [0, prefill_tp*prefill_pp)
                   id = pp_stage*prefill_tp + tp_rank
    decode  pool : NPU ids [prefill_npus, prefill_npus + decode_tp*decode_pp)
                   id = prefill_npus + pp_stage*decode_tp + tp_rank

ASTRA-sim's sys[i] maps 1:1 to NPU i (trace files are name.{npu_id}.et), so the
config alone tells us which sys is prefill and which is decode.

All metrics/plots are derived ONLY from the log; the YAML is used solely to
label NPUs and to know how many decode steps / layers-per-stage to expect.
No external assumptions (e.g. alternative bandwidths) are introduced.

Usage
-----
    python3 parse_disagg_inference.py LOG -c CONFIG.yaml [-o OUTDIR]

If no config is given the script falls back to a behavioural heuristic
(compute-bound NPUs = prefill, memory-bound = decode) and warns.

Semantics note
--------------
ASTRA-sim reports, per NPU: Wall = GPU(compute) + Comm. "Comm" lumps together
real network transfer AND any blocking/exposed waiting on dependencies. For a
single-NPU pool (tp=pp=1) "Comm" of the prefill NPU is exactly the KV-cache
push. With pp>1 the split per NPU still holds, but the per-NPU "Comm" of a
downstream stage also contains the wait for upstream stages; the per-NPU
stacked chart (chart 02) is the fully faithful view in that case.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


# --------------------------------------------------------------------------- #
# Regex for the log                                                            #
# --------------------------------------------------------------------------- #
RE_DEBUG = re.compile(
    r"\[workload\] \[debug\] "
    r"operation_intensity=(?P<oi>[\d.eE+-]+), "
    r"perf=(?P<perf>[\d.eE+-]+), "
    r"elapsed_time=(?P<elapsed>[\d.eE+-]+) "
    r"compute_utilization=(?P<cu>[\d.eE+-]+) "
    r"memory_utilization=(?P<mu>[\d.eE+-]+) "
    r"tensor_size=(?P<ts>\d+) "
    r"num_ops=(?P<nops>\d+)"
)
RE_FINISHED = re.compile(
    r"sys\[(?P<sid>\d+)\] finished, (?P<cycles>\d+) cycles, "
    r"exposed communication (?P<exposed>\d+) cycles"
)
RE_STAT = re.compile(
    r"\[statistics\] \[info\] sys\[(?P<sid>\d+)\], (?P<key>[^:]+):\s*(?P<val>[-\d.]+)%?"
)


# --------------------------------------------------------------------------- #
# Config (mirrors MLSynth config.py just enough for our needs)                 #
# --------------------------------------------------------------------------- #
def load_config(path):
    """Return a dict: model, prefill{tp,pp}, decode{tp,pp}, decode_steps,
    requests, kv_transfer, serialize. Mirrors MLSynth's RunConfig.from_yaml."""
    if not _HAVE_YAML:
        raise RuntimeError("PyYAML not available; cannot read the config.")
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    model = data.get("model", {})

    def one(block):
        return {"tp": int(block.get("tp_size", 1)), "pp": int(block.get("pp_size", 1))}

    if "prefill_parallelism" in data or "decode_parallelism" in data:
        prefill = one(data.get("prefill_parallelism", {}))
        decode = one(data.get("decode_parallelism", {}))
    else:
        block = data.get("parallelism", {})
        prefill = decode = one(block)

    inf = data.get("inference", data)  # tolerate flattened layouts
    if "requests" in inf:
        reqs = [{"prompt_len": int(r["prompt_len"]), "gen_len": int(r["gen_len"])}
                for r in inf["requests"]]
    elif "num_requests" in inf:
        reqs = [{"prompt_len": int(inf["prompt_len"]), "gen_len": int(inf["gen_len"])}
                for _ in range(int(inf["num_requests"]))]
    else:
        reqs = []

    kv = inf.get("kv_transfer", {})
    return {
        "model": {
            "name": model.get("name", "model"),
            "num_layers": int(model.get("num_layers", 0)),
            "hidden_size": int(model.get("hidden_size", 0)),
            "bytes_per_val": int(model.get("bytes_per_val", 2)),
        },
        "prefill": prefill,
        "decode": decode,
        "requests": reqs,
        "decode_steps": max((r["gen_len"] for r in reqs), default=0),
        "kv_transfer": {"mode": kv.get("mode"), "direction": kv.get("direction")},
        "serialize": bool(inf.get("serialize_decode_iterations", True)),
    }


def build_partition(cfg):
    """Map every NPU id -> (role, pp_stage, tp_rank) using MLSynth's layout."""
    pf, dc = cfg["prefill"], cfg["decode"]
    prefill_npus = pf["tp"] * pf["pp"]
    decode_npus = dc["tp"] * dc["pp"]
    part = {}
    for stage in range(pf["pp"]):
        for rank in range(pf["tp"]):
            part[stage * pf["tp"] + rank] = ("prefill", stage, rank)
    for stage in range(dc["pp"]):
        for rank in range(dc["tp"]):
            part[prefill_npus + stage * dc["tp"] + rank] = ("decode", stage, rank)
    return part, prefill_npus, decode_npus


# --------------------------------------------------------------------------- #
# Log parsing                                                                  #
# --------------------------------------------------------------------------- #
def parse_log(path):
    """ops[sid] -> list of op dicts (in log order); finished[sid] -> {cycles,
    exposed_comm}; stats[sid] -> {metric: value}.

    Debug records carry no sid; we attribute the records that appear up to and
    including each 'sys[N] finished' marker to sys[N] (matches ASTRA-sim's
    per-system logging order)."""
    ops = defaultdict(list)
    finished, stats = {}, defaultdict(dict)
    pending = []  # debug ops not yet attributed
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_DEBUG.search(line)
            if m:
                d = m.groupdict()
                pending.append({
                    "operation_intensity": float(d["oi"]),
                    "elapsed_time": float(d["elapsed"]),
                    "compute_utilization": float(d["cu"]),
                    "memory_utilization": float(d["mu"]),
                    "tensor_size": int(d["ts"]),
                    "num_ops": int(d["nops"]),
                })
                continue
            mf = RE_FINISHED.search(line)
            if mf:
                sid = int(mf.group("sid"))
                finished[sid] = {"cycles": int(mf.group("cycles")),
                                 "exposed_comm": int(mf.group("exposed"))}
                ops[sid].extend(pending)
                pending = []
                continue
            ms = RE_STAT.search(line)
            if ms:
                sid = int(ms.group("sid"))
                stats[sid][ms.group("key").strip()] = float(ms.group("val"))
    return dict(ops), finished, dict(stats)


def derive_ms_per_cycle(ops, stats):
    """1 cycle in ms, derived from the data: for NPUs that have debug ops and a
    GPU-time stat, sum(elapsed_time)[s]*1e3 should equal GPU cycles. Take the
    median ratio across such NPUs (robust). Fallback: 1 cycle = 1 ns."""
    ratios = []
    for sid, recs in ops.items():
        gpu = stats.get(sid, {}).get("GPU time")
        es = sum(r["elapsed_time"] for r in recs)
        if gpu and gpu > 0 and es > 0:
            ratios.append((es * 1e3) / gpu)
    return float(np.median(ratios)) if ratios else 1e-6


# --------------------------------------------------------------------------- #
# Per-token splitting for the decode NPUs                                      #
# --------------------------------------------------------------------------- #
def split_tokens(recs, decode_steps):
    """Return a list of per-token compute times (seconds) for one decode NPU.

    Preferred: split the NPU's ops into `decode_steps` equal contiguous chunks
    (each token runs the same #ops on a given NPU). Fallback: group by the
    'growing' tensor_size (one distinct value per token). Last resort: spread
    the total uniformly."""
    if not recs:
        return []
    n = len(recs)
    if decode_steps and n % decode_steps == 0:
        per = n // decode_steps
        return [sum(r["elapsed_time"] for r in recs[i * per:(i + 1) * per])
                for i in range(decode_steps)]
    # fallback: growing-tensor heuristic
    ts = [r["tensor_size"] for r in recs]
    const_val = Counter(ts).most_common(1)[0][0]
    tokens, cur, last = [], [], None
    for r in recs:
        grow = r["tensor_size"] != const_val
        if grow and last is not None and r["tensor_size"] != last:
            tokens.append(cur); cur = []
        if grow:
            last = r["tensor_size"]
        cur.append(r)
    if cur:
        tokens.append(cur)
    if tokens:
        return [sum(r["elapsed_time"] for r in t) for t in tokens]
    total = sum(r["elapsed_time"] for r in recs)
    k = decode_steps or 1
    return [total / k] * k


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #
def analyse(ops, finished, stats, ms, part, cfg, warns):
    # ---- per-NPU aggregates (from stats; convert cycles->ms) ----
    npus = {}
    for sid in sorted(set(list(stats) + list(finished) + list(ops))):
        s = stats.get(sid, {})
        role, stage, rank = part.get(sid, ("unknown", -1, -1))
        npus[sid] = {
            "sid": sid, "role": role, "pp_stage": stage, "tp_rank": rank,
            "wall_ms": s.get("Wall time", 0) * ms,
            "compute_ms": s.get("GPU time", 0) * ms,
            "comm_ms": s.get("Comm time", 0) * ms,
            "exposed_ms": finished.get(sid, {}).get("exposed_comm", 0) * ms,
            "compute_bound_pct": s.get("Compute bound percentage"),
            "compute_util": s.get("Average compute utilization"),
            "mem_util": s.get("Average memory utilization"),
            "op_intensity": s.get("Average operation intensity"),
            "n_ops": len(ops.get(sid, [])),
        }

    prefill_ids = [i for i, v in npus.items() if v["role"] == "prefill"]
    decode_ids = [i for i, v in npus.items() if v["role"] == "decode"]
    if not prefill_ids or not decode_ids:
        warns.append("Could not identify both a prefill and a decode pool from "
                     "the config/log. Check the YAML matches this run.")

    # ---- pool-level critical path ----
    # KV is ready (and decode may start) when the last prefill NPU finishes its
    # compute + push. Use the critical (max-wall) prefill NPU for a clean,
    # internally-consistent compute/transfer split.
    crit_sid = max(prefill_ids, key=lambda i: npus[i]["wall_ms"]) if prefill_ids else None
    prefill_compute_ms = npus[crit_sid]["compute_ms"] if crit_sid is not None else 0.0
    kv_transfer_ms = npus[crit_sid]["comm_ms"] if crit_sid is not None else 0.0
    prefill_ready_ms = (prefill_compute_ms + kv_transfer_ms)

    # ---- decode per-token latency (ITL) aggregated over the pipeline ----
    steps = cfg["decode_steps"] if cfg else 0
    # per decode NPU -> per-token compute (seconds)
    per_npu_tok = {i: split_tokens(ops.get(i, []), steps) for i in decode_ids}
    # cross-check op counts when config known
    if cfg and cfg["model"]["num_layers"] and cfg["decode"]["pp"]:
        lps = cfg["model"]["num_layers"] // cfg["decode"]["pp"]
        for i in decode_ids:
            n = npus[i]["n_ops"]
            if steps and lps and n and n % (steps * lps) != 0:
                warns.append(f"sys[{i}]: {n} decode ops not divisible by "
                             f"decode_steps*layers_per_stage ({steps}*{lps}); "
                             "per-token split used a fallback.")

    # group decode NPUs by pipeline stage; a token traverses all stages
    # sequentially, ranks within a stage run in parallel (take max).
    stages = defaultdict(list)
    for i in decode_ids:
        stages[npus[i]["pp_stage"]].append(i)
    n_tok = min((len(per_npu_tok[i]) for i in decode_ids), default=0)
    itl_ms = []
    for t in range(n_tok):
        total = 0.0
        for st in sorted(stages):
            ranks = stages[st]
            total += max(per_npu_tok[i][t] for i in ranks)  # parallel ranks
        itl_ms.append(total * 1e3)  # s -> ms

    avg_itl = float(np.mean(itl_ms)) if itl_ms else 0.0
    decode_total_ms = float(np.sum(itl_ms)) if itl_ms else 0.0

    # ---- TTFT ----
    first_step_ms = itl_ms[0] if itl_ms else 0.0
    ttft_ms = prefill_ready_ms + first_step_ms
    kv_share = (kv_transfer_ms / ttft_ms * 100) if ttft_ms else 0.0
    kv_in_tokens = (kv_transfer_ms / avg_itl) if avg_itl else 0.0
    kv_amortised = (kv_transfer_ms / len(itl_ms)) if itl_ms else 0.0

    # ---- end-to-end makespan = max wall over all NPUs ----
    makespan_ms = max((v["wall_ms"] for v in npus.values()), default=0.0)

    # ---- pool-mean roofline context ----
    def pool_mean(ids, key):
        vals = [npus[i][key] for i in ids if npus[i][key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "ms_per_cycle": ms,
        "config": cfg,
        "n_prefill_npus": len(prefill_ids),
        "n_decode_npus": len(decode_ids),
        "critical_prefill_sid": crit_sid,
        "prefill_compute_ms": prefill_compute_ms,
        "kv_transfer_ms": kv_transfer_ms,
        "prefill_ready_ms": prefill_ready_ms,
        "itl_ms": itl_ms,
        "avg_itl_ms": avg_itl,
        "decode_total_ms": decode_total_ms,
        "ttft_ms": ttft_ms,
        "ttft_first_step_ms": first_step_ms,
        "kv_share_of_ttft_pct": kv_share,
        "kv_in_tokens": kv_in_tokens,
        "kv_amortised_per_tok_ms": kv_amortised,
        "makespan_ms": makespan_ms,
        "npus": npus,
        "pool_means": {
            "prefill": {k: pool_mean(prefill_ids, k) for k in
                        ("compute_bound_pct", "compute_util", "mem_util", "op_intensity")},
            "decode": {k: pool_mean(decode_ids, k) for k in
                       ("compute_bound_pct", "compute_util", "mem_util", "op_intensity")},
        },
        "warnings": warns,
    }


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
C_COMPUTE = "#3B7DD8"
C_KV = "#E8743B"
C_T1 = "#2E8B57"
C_TN = "#6FB98F"
C_ACCENT = "#1F2937"


def _save(fig, outdir, name):
    p = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_critical_path(a, outdir):
    fig, ax = plt.subplots(figsize=(11, 3.0))
    t = 0.0
    for label, dur, col in [("Prefill compute", a["prefill_compute_ms"], C_COMPUTE),
                            ("KV-cache transfer (push)", a["kv_transfer_ms"], C_KV)]:
        ax.barh(0, dur, left=t, color=col, edgecolor="white")
        if dur > 0:
            ax.text(t + dur / 2, 0, f"{dur:.2f} ms", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        t += dur
    for i, d in enumerate(a["itl_ms"]):
        ax.barh(0, d, left=t, color=C_T1 if i == 0 else C_TN, edgecolor="white")
        t += d
    ax.axvline(a["ttft_ms"], color=C_ACCENT, ls="--", lw=1.4)
    ax.text(a["ttft_ms"], -0.5, f"TTFT = {a['ttft_ms']:.2f} ms ",
            color=C_ACCENT, fontsize=9, fontweight="bold", va="top", ha="right")
    ax.set_yticks([]); ax.set_ylim(-0.6, 0.7)
    ax.set_xlabel("Time along the critical path (ms)")
    crit = a["critical_prefill_sid"]
    ax.set_title(f"Critical path: prefill compute → KV push → decode tokens "
                 f"(critical prefill NPU: sys[{crit}])",
                 fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_COMPUTE, label="Prefill compute"),
                       Patch(color=C_KV, label="KV-cache transfer"),
                       Patch(color=C_T1, label="1st token"),
                       Patch(color=C_TN, label="later tokens")],
              loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    return _save(fig, outdir, "01_critical_path_ttft.png")


def plot_per_npu(a, outdir):
    """Stacked compute vs comm for every NPU — faithful for any tp/pp."""
    npus = a["npus"]
    ids = sorted(npus)
    labels, comp, comm, edge = [], [], [], []
    for i in ids:
        v = npus[i]
        tag = "P" if v["role"] == "prefill" else ("D" if v["role"] == "decode" else "?")
        labels.append(f"{tag} sys{i}\ns{v['pp_stage']}r{v['tp_rank']}")
        comp.append(v["compute_ms"]); comm.append(v["comm_ms"])
        edge.append(C_T1 if v["role"] == "prefill" else C_ACCENT)
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(ids) + 3), 5))
    ax.bar(x, comp, 0.6, color=C_COMPUTE, label="Compute")
    ax.bar(x, comm, 0.6, bottom=comp, color=C_KV, label="Comm (KV push / wait)")
    for xi, c1, c2 in zip(x, comp, comm):
        if c1 + c2 > 0:
            ax.text(xi, c1 + c2, f"{c1 + c2:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=C_ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-NPU wall-time breakdown (P = prefill, D = decode)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9); ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, outdir, "02_per_npu_breakdown.png")


def plot_itl(a, outdir):
    itl = a["itl_ms"]
    if not itl:
        return None
    spread_us = (max(itl) - min(itl)) * 1e3
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(1, len(itl) + 1)
    ax.plot(x, itl, "-o", color=C_COMPUTE, lw=2, ms=7, zorder=3, label="ITL per token")
    ax.axhline(a["avg_itl_ms"], color=C_KV, ls="--", lw=1.6, zorder=2,
               label=f"mean ITL = {a['avg_itl_ms']:.4f} ms")
    if len(itl) <= 16:
        for xi, v in zip(x, itl):
            ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xlabel("Generated token index"); ax.set_ylabel("Inter-token latency (ms)")
    ax.margins(y=0.30)
    ax.set_title("Inter-token latency (ITL) — decode compute on the critical path\n"
                 f"zoomed axis: total growth {spread_us:.3f} µs over {len(itl)} tokens "
                 "(KV-cache lengthening)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    return _save(fig, outdir, "03_itl_per_token.png")


def plot_cost(a, outdir):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = ["KV-cache\ntransfer", "TTFT", "mean ITL\n(1 token)", "decode total\n(all tokens)"]
    vals = [a["kv_transfer_ms"], a["ttft_ms"], a["avg_itl_ms"], a["decode_total_ms"]]
    cols = [C_KV, C_ACCENT, C_COMPUTE, C_TN]
    for b, v in zip(ax.bar(names, vals, color=cols, edgecolor="white"), vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} ms",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_title("KV-cache communication cost in perspective\n"
                 f"one KV push ≈ {a['kv_in_tokens']:.1f} decode tokens • "
                 f"{a['kv_share_of_ttft_pct']:.0f}% of TTFT",
                 fontsize=12, fontweight="bold")
    ax.margins(y=0.18); ax.grid(axis="y", alpha=0.3)
    return _save(fig, outdir, "04_kv_cost_perspective.png")


def plot_roofline(a, outdir):
    pm = a["pool_means"]
    rows = ["Compute bound (%)", "Avg compute util (%)", "Avg memory util (%)",
            "Avg operation intensity"]
    keys = ["compute_bound_pct", "compute_util", "mem_util", "op_intensity"]
    def fmt(x):
        return "—" if x is None else f"{x:.2f}"
    cell = [[fmt(pm["prefill"][k]), fmt(pm["decode"][k])] for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 3.0)); ax.axis("off")
    tbl = ax.table(cellText=cell, rowLabels=rows,
                   colLabels=[f"Prefill pool\n({a['n_prefill_npus']} NPU)",
                              f"Decode pool\n({a['n_decode_npus']} NPU)"],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for (r, c), cell_obj in tbl.get_celld().items():
        if r == 0:
            cell_obj.set_facecolor(C_ACCENT)
            cell_obj.set_text_props(color="white", fontweight="bold")
    ax.set_title("Roofline profile per pool (mean over NPUs, from log stats)",
                 fontsize=12, fontweight="bold", pad=16)
    return _save(fig, outdir, "05_roofline_profile.png")


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def report(a):
    cfg = a["config"]
    L = ["=" * 66, "  DISAGGREGATED INFERENCE — KV-cache communication analysis", "=" * 66]
    if cfg:
        L.append(f"  Model: {cfg['model']['name']} | layers={cfg['model']['num_layers']} "
                 f"hidden={cfg['model']['hidden_size']}")
        L.append(f"  Prefill pool: tp={cfg['prefill']['tp']} pp={cfg['prefill']['pp']} "
                 f"({a['n_prefill_npus']} NPU)   |   "
                 f"Decode pool: tp={cfg['decode']['tp']} pp={cfg['decode']['pp']} "
                 f"({a['n_decode_npus']} NPU)")
        L.append(f"  KV transfer: {cfg['kv_transfer']['mode']}/{cfg['kv_transfer']['direction']} "
                 f"| decode steps: {cfg['decode_steps']}")
    L.append(f"  Time factor derived from log: 1 cycle ≈ {a['ms_per_cycle']*1e6:.3f} ns")
    L.append("")
    L.append("  --- Time To First Token (TTFT) ---")
    L.append(f"   prefill compute            : {a['prefill_compute_ms']:.3f} ms")
    L.append(f"   KV-cache transfer (push)   : {a['kv_transfer_ms']:.3f} ms")
    L.append(f"   1st decode step            : {a['ttft_first_step_ms']:.3f} ms")
    L.append(f"   TTFT TOTAL                 : {a['ttft_ms']:.3f} ms")
    L.append(f"   KV share of TTFT           : {a['kv_share_of_ttft_pct']:.1f} %")
    L.append("")
    L.append("  --- Inter-Token Latency (ITL, decode compute) ---")
    L.append(f"   mean ITL                   : {a['avg_itl_ms']:.4f} ms")
    if a["itl_ms"]:
        L.append(f"   ITL min / max              : {min(a['itl_ms']):.4f} / {max(a['itl_ms']):.4f} ms")
    L.append(f"   decode total (all tokens)  : {a['decode_total_ms']:.3f} ms")
    L.append("")
    L.append("  --- KV-cache communication impact ---")
    L.append(f"   one KV push ≈ {a['kv_in_tokens']:.1f} decode tokens")
    L.append(f"   KV amortised over {len(a['itl_ms'])} tokens : {a['kv_amortised_per_tok_ms']:.3f} ms/token")
    L.append(f"   end-to-end makespan        : {a['makespan_ms']:.3f} ms")
    L.append("")
    L.append("  --- Per-NPU (role | wall = compute + comm) ---")
    for i in sorted(a["npus"]):
        v = a["npus"][i]
        L.append(f"   sys[{i}] {v['role']:<7} s{v['pp_stage']}r{v['tp_rank']} | "
                 f"wall {v['wall_ms']:.3f} = comp {v['compute_ms']:.3f} + comm {v['comm_ms']:.3f} ms "
                 f"| op-int {v['op_intensity']}")
    if a["warnings"]:
        L.append(""); L.append("  --- Warnings ---")
        for w in a["warnings"]:
            L.append(f"   ! {w}")
    L.append("=" * 66)
    out = "\n".join(L)
    print(out)
    return out


def write_npu_csv(a, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "role", "pp_stage", "tp_rank", "wall_ms", "compute_ms",
                    "comm_ms", "exposed_ms", "compute_bound_pct", "compute_util",
                    "mem_util", "op_intensity", "n_ops"])
        for i in sorted(a["npus"]):
            v = a["npus"][i]
            w.writerow([v["sid"], v["role"], v["pp_stage"], v["tp_rank"],
                        f"{v['wall_ms']:.6f}", f"{v['compute_ms']:.6f}",
                        f"{v['comm_ms']:.6f}", f"{v['exposed_ms']:.6f}",
                        v["compute_bound_pct"], v["compute_util"], v["mem_util"],
                        v["op_intensity"], v["n_ops"]])


# --------------------------------------------------------------------------- #
# Fallback partition (no config): behavioural heuristic                        #
# --------------------------------------------------------------------------- #
def heuristic_partition(stats, warns):
    warns.append("No config provided: prefill/decode roles inferred from the log "
                 "(compute-bound→prefill, memory-bound→decode). Pass -c CONFIG.yaml "
                 "for tp/pp-aware labelling.")
    part = {}
    for sid, s in stats.items():
        cb = s.get("Compute bound percentage", 0)
        role = "prefill" if cb is not None and cb >= 50 else "decode"
        part[sid] = (role, 0, 0)
    # assign pp_stage/tp_rank sequentially within each role for display
    for role in ("prefill", "decode"):
        k = 0
        for sid in sorted(p for p, (r, *_ ) in part.items() if r == role):
            part[sid] = (role, k, 0); k += 1
    npf = sum(1 for v in part.values() if v[0] == "prefill")
    ndc = sum(1 for v in part.values() if v[0] == "decode")
    return part, npf, ndc


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Parse ASTRA-sim disaggregated KV-cache log.")
    ap.add_argument("log", nargs="?", default="log.log", help="ASTRA-sim log file")
    ap.add_argument("-c", "--config", help="MLSynth YAML used to generate the traces")
    ap.add_argument("-o", "--outdir", default="kv_analysis_output", help="output directory")
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        sys.exit(f"Log file not found: {args.log}")
    os.makedirs(args.outdir, exist_ok=True)

    ops, finished, stats = parse_log(args.log)
    if not stats:
        sys.exit("No statistics found — does the log have the expected format?")

    warns = []
    cfg = None
    if args.config:
        if not os.path.isfile(args.config):
            sys.exit(f"Config file not found: {args.config}")
        cfg = load_config(args.config)
        part, npf, ndc = build_partition(cfg)
        # sanity: config NPU count vs sys ids in the log
        n_log = len(set(list(stats) + list(finished)))
        if (npf + ndc) != n_log:
            warns.append(f"Config implies {npf + ndc} NPUs but the log has {n_log} "
                         "sys[] entries — is this the right config for this run?")
    else:
        part, npf, ndc = heuristic_partition(stats, warns)

    ms = derive_ms_per_cycle(ops, stats)
    a = analyse(ops, finished, stats, ms, part, cfg, warns)

    figs = [plot_critical_path(a, args.outdir),
            plot_per_npu(a, args.outdir),
            plot_itl(a, args.outdir),
            plot_cost(a, args.outdir),
            plot_roofline(a, args.outdir)]
    figs = [f for f in figs if f]

    text = report(a)
    with open(os.path.join(args.outdir, "statistics.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.outdir, "statistics.json"), "w") as f:
        json.dump(a, f, indent=2)
    write_npu_csv(a, os.path.join(args.outdir, "per_npu.csv"))

    print("\nGenerated files:")
    for p in figs + [os.path.join(args.outdir, n) for n in
                     ("statistics.txt", "statistics.json", "per_npu.csv")]:
        print("  ", p)


if __name__ == "__main__":
    main()