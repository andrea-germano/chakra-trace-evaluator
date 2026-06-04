import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import yaml
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


# --------------------------------------------------------------------------- #
# Log regexes                                                                  #
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
# Config (mirrors config.py:RunConfig just enough to label NPUs)               #
# --------------------------------------------------------------------------- #
def load_config(path):
    if not _HAVE_YAML:
        raise RuntimeError("PyYAML not available; cannot read the config.")
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    model = data.get("model", {})

    def one(block):
        return {"tp": int(block.get("tp_size", 1)), "pp": int(block.get("pp_size", 1))}

    # Two-pool form (disaggregated) or a single shared parallelism block.
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
        # gen_len drives the number of decode steps; with mixed requests this is
        # the longest one (the per-token split is then approximate -- we warn).
        "decode_steps": max((r["gen_len"] for r in reqs), default=0),
        "total_prompt_tokens": sum(r["prompt_len"] for r in reqs),
        "kv_transfer": {"mode": kv.get("mode"), "direction": kv.get("direction")},
        "serialize": bool(inf.get("serialize_decode_iterations", True)),
    }


def build_partition(cfg):
    """Map every NPU id -> (role, pp_stage, tp_rank) using MLSynth's layout."""
    pf, dc = cfg["prefill"], cfg["decode"]
    prefill_npus = pf["tp"] * pf["pp"]
    part = {}
    for stage in range(pf["pp"]):
        for rank in range(pf["tp"]):
            part[stage * pf["tp"] + rank] = ("prefill", stage, rank)
    for stage in range(dc["pp"]):
        for rank in range(dc["tp"]):
            part[prefill_npus + stage * dc["tp"] + rank] = ("decode", stage, rank)
    return part, prefill_npus, dc["tp"] * dc["pp"]


def heuristic_partition(stats, warns):
    """Fallback when no config is given: classify by compute-bound percentage.
    pp_stage/tp_rank are unknown, so every NPU is placed in stage 0 -- the
    pp-aware critical path then degenerates to a single stage (still correct for
    tp-only layouts, approximate for pp)."""
    warns.append("No config provided: prefill/decode roles inferred from the log "
                 "(compute-bound -> prefill, memory-bound -> decode). Pass "
                 "-c CONFIG.yaml for tp/pp-aware labelling.")
    part = {}
    for sid, s in stats.items():
        cb = s.get("Compute bound percentage", 0)
        role = "prefill" if cb is not None and cb >= 50 else "decode"
        part[sid] = (role, 0, 0)
    npf = sum(1 for v in part.values() if v[0] == "prefill")
    ndc = sum(1 for v in part.values() if v[0] == "decode")
    return part, npf, ndc


# --------------------------------------------------------------------------- #
# Log parsing                                                                  #
# --------------------------------------------------------------------------- #
def parse_log(path):
    """Return (ops, finished, stats):
        ops[sid]      -> list of COMP-node dicts attributed to that NPU,
        finished[sid] -> {cycles, exposed_comm},
        stats[sid]    -> {metric: value}.
    COMP debug records carry no sid, so they are attributed to the next
    'sys[N] finished' marker (see the module docstring for the limitation)."""
    ops = defaultdict(list)
    finished, stats = {}, defaultdict(dict)
    pending = []
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
    """Recover the wall-clock value of one cycle, in milliseconds, from the data
    itself: an NPU's summed COMP elapsed_time (seconds) equals its GPU-time cycle
    count, so ms_per_cycle = sum(elapsed)*1e3 / GPU_cycles. Median over NPUs is
    robust to interleaving; fall back to 1 cycle = 1 ns."""
    ratios = []
    for sid, recs in ops.items():
        gpu = stats.get(sid, {}).get("GPU time")
        es = sum(r["elapsed_time"] for r in recs)
        if gpu and gpu > 0 and es > 0:
            ratios.append((es * 1e3) / gpu)
    return float(np.median(ratios)) if ratios else 1e-6


# --------------------------------------------------------------------------- #
# Per-token splitting for decode NPUs                                          #
# --------------------------------------------------------------------------- #
def split_tokens(recs, decode_steps):
    """Per-token compute time (seconds) for one decode NPU.

    A decode NPU runs the SAME number of ops per generated token (one pass over
    the layers it owns), so the principled split is `decode_steps` equal,
    contiguous chunks by op count. Because we sum elapsed_time inside each chunk,
    the growing attention cost (KV cache lengthening) is captured automatically
    -- no special growth heuristic needed. If the op count is not divisible by
    decode_steps (usually a sign of mis-attributed COMP records), fall back to a
    uniform spread; the caller warns when this happens."""
    if not recs:
        return []
    n = len(recs)
    if decode_steps and n % decode_steps == 0:
        per = n // decode_steps
        return [sum(r["elapsed_time"] for r in recs[i * per:(i + 1) * per])
                for i in range(decode_steps)]
    total = sum(r["elapsed_time"] for r in recs)
    k = decode_steps or 1
    return [total / k] * k


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #
def analyse(ops, finished, stats, ms_per_cycle, part, cfg, warns):
    # ---- pass 1: raw per-NPU timings (compute / exposed / wall) ----
    npus = {}
    for sid in sorted(set(list(stats) + list(finished) + list(ops))):
        s = stats.get(sid, {})
        role, stage, rank = part.get(sid, ("unknown", -1, -1))
        wall = s.get("Wall time", finished.get(sid, {}).get("cycles", 0)) * ms_per_cycle
        compute = s.get("GPU time", 0) * ms_per_cycle
        comm_total = s.get("Comm time", 0) * ms_per_cycle
        # exposed comm is the comm that actually blocks the NPU; fall back to
        # total comm when the 'finished' marker is missing.
        exposed = (finished.get(sid, {}).get("exposed_comm", 0) * ms_per_cycle
                   if sid in finished else comm_total)
        npus[sid] = {
            "sid": sid, "role": role, "pp_stage": stage, "tp_rank": rank,
            "wall_ms": wall, "compute_ms": compute, "comm_ms": comm_total,
            "exposed_ms": exposed,
            "compute_bound_pct": s.get("Compute bound percentage"),
            "compute_util": s.get("Average compute utilization"),
            "mem_util": s.get("Average memory utilization"),
            "op_intensity": s.get("Average operation intensity"),
            "n_ops": len(ops.get(sid, [])),
        }

    prefill_ids = [i for i, v in npus.items() if v["role"] == "prefill"]
    decode_ids = [i for i, v in npus.items() if v["role"] == "decode"]
    if not prefill_ids or not decode_ids:
        warns.append("Could not identify both a prefill and a decode pool. "
                     "Check the YAML matches this run.")

    # End of the whole run (latest wall), reported as the end-to-end makespan.
    makespan_ms = max((v["wall_ms"] for v in npus.values()), default=0.0)

    # ---- prefill critical path (pp-aware) ----
    # `prefill_ready` is when the first-token handoff completes: the latest wall
    # in the prefill pool, which already includes pipeline fill/drain and the KV
    # push. Its compute share is the sum over stages of the per-stage compute
    # (max over the ranks in a stage, since ranks run in parallel); the rest is
    # exposed comm on the critical path (KV push, and PP sends when pp>1).
    prefill_ready_ms = max((npus[i]["wall_ms"] for i in prefill_ids), default=0.0)
    crit_sid = (max(prefill_ids, key=lambda i: npus[i]["wall_ms"])
                if prefill_ids else None)

    pf_stages = defaultdict(list)
    for i in prefill_ids:
        pf_stages[npus[i]["pp_stage"]].append(i)
    prefill_compute_ms = sum(max(npus[i]["compute_ms"] for i in ids)
                             for ids in pf_stages.values())
    prefill_comm_ms = max(0.0, prefill_ready_ms - prefill_compute_ms)
    if cfg and cfg["prefill"]["pp"] > 1:
        warns.append("pp_prefill > 1: 'prefill comm' on the critical path folds "
                     "the KV push together with PP sends. Separate them with "
                     "per-node comm tags if you need KV in isolation.")

    # ---- pass 2: split each NPU's wall into compute / handoff-wait / comm+idle ----
    # wall == compute + exposed_comm in the analytical backend, so the three
    # buckets always reconstruct the wall. We split the exposed comm into the
    # initial handoff stall (decode only) and a single "comm + idle" bucket for
    # everything else: the prefill KV push, and on the decode side the recurring
    # PP recv waits and inter-token/inter-stage bubbles. Those last two cannot be
    # told apart from the lumped statistic without per-node comm tags, hence the
    # joint label.
    for sid, v in npus.items():
        if v["role"] == "decode":
            # Blocked on the handoff RECV until prefill_ready; that share of the
            # exposed comm is the startup stall, the rest is recurring comm+idle.
            wait = min(v["exposed_ms"], prefill_ready_ms)
        else:
            wait = 0.0  # prefill has no upstream handoff to wait on
        v["wait_ms"] = wait
        v["comm_idle_ms"] = v["exposed_ms"] - wait

    # ---- decode per-token latency (ITL), aggregated over the pipeline ----
    steps = cfg["decode_steps"] if cfg else 0
    per_npu_tok = {i: split_tokens(ops.get(i, []), steps) for i in decode_ids}
    if cfg and cfg["model"]["num_layers"] and cfg["decode"]["pp"]:
        lps = cfg["model"]["num_layers"] // cfg["decode"]["pp"]
        for i in decode_ids:
            n = npus[i]["n_ops"]
            if steps and lps and n and n % (steps * lps) != 0:
                warns.append(f"sys[{i}]: {n} decode ops not divisible by "
                             f"decode_steps*layers_per_stage ({steps}*{lps}); "
                             "per-token split fell back to a uniform spread "
                             "(likely COMP records mis-attributed across NPUs).")

    # A token traverses pipeline stages sequentially; the ranks within a stage
    # run in parallel (take the max). Summing the stage maxima gives the
    # per-token latency. This is the latency of one token regardless of regime;
    # `serialize` only controls whether successive tokens overlap in the pipeline
    # (throughput), which does not change single-token latency.
    dc_stages = defaultdict(list)
    for i in decode_ids:
        dc_stages[npus[i]["pp_stage"]].append(i)
    n_tok = min((len(per_npu_tok[i]) for i in decode_ids), default=0)
    itl_ms = []
    for t in range(n_tok):
        total = sum(max(per_npu_tok[i][t] for i in dc_stages[st])
                    for st in sorted(dc_stages))
        itl_ms.append(total * 1e3)  # s -> ms

    avg_itl = float(np.mean(itl_ms)) if itl_ms else 0.0
    decode_total_ms = float(np.sum(itl_ms)) if itl_ms else 0.0

    # Steady-state pipeline bound (slowest stage) -- only meaningful for a
    # saturated, non-serialized pipeline; reported for context.
    decode_pipeline_bound_ms = 0.0
    if itl_ms and len(dc_stages) > 1 and cfg and not cfg["serialize"]:
        per_stage_max = [max(per_npu_tok[i][0] for i in dc_stages[st])
                         for st in sorted(dc_stages)]
        decode_pipeline_bound_ms = max(per_stage_max) * 1e3

    # ---- decode-pool startup wait (~TTFT seen from the decode side) ----
    decode_wait_ms = (float(np.mean([npus[i]["wait_ms"] for i in decode_ids]))
                      if decode_ids else 0.0)

    # ---- TTFT and comm-cost framing ----
    first_step_ms = itl_ms[0] if itl_ms else 0.0
    ttft_ms = prefill_ready_ms + first_step_ms
    comm_share = (prefill_comm_ms / ttft_ms * 100) if ttft_ms else 0.0
    comm_in_tokens = (prefill_comm_ms / avg_itl) if avg_itl else 0.0
    comm_amortised = (prefill_comm_ms / len(itl_ms)) if itl_ms else 0.0

    def pool_mean(ids, key):
        vals = [npus[i][key] for i in ids if npus[i].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def pool_sum(ids, key):
        return float(sum(npus[i][key] for i in ids))

    def pool_buckets(ids):
        # FIX: Changed keys to exactly match the refactored keys created in Pass 2
        return {k: pool_sum(ids, k) for k in
                ("compute_ms", "comm_idle_ms", "wait_ms")}

    return {
        "ms_per_cycle": ms_per_cycle,
        "config": cfg,
        "n_prefill_npus": len(prefill_ids),
        "n_decode_npus": len(decode_ids),
        "critical_prefill_sid": crit_sid,
        "prefill_ready_ms": prefill_ready_ms,
        "prefill_compute_ms": prefill_compute_ms,
        "prefill_comm_ms": prefill_comm_ms,
        "decode_wait_ms": decode_wait_ms,
        "itl_ms": itl_ms,
        "avg_itl_ms": avg_itl,
        "decode_total_ms": decode_total_ms,
        "decode_pipeline_bound_ms": decode_pipeline_bound_ms,
        "ttft_ms": ttft_ms,
        "ttft_first_step_ms": first_step_ms,
        "comm_share_of_ttft_pct": comm_share,
        "prefill_comm_in_tokens": comm_in_tokens,
        "prefill_comm_amortised_per_tok_ms": comm_amortised,
        "makespan_ms": makespan_ms,
        "npus": npus,
        "pool_aggregate": {
            "prefill": pool_buckets(prefill_ids),
            "decode": pool_buckets(decode_ids),
        },
        "pool_means": {
            "prefill": {k: pool_mean(prefill_ids, k) for k in
                        ("compute_bound_pct", "compute_util", "mem_util", "op_intensity")},
            "decode": {k: pool_mean(decode_ids, k) for k in
                       ("compute_bound_pct", "compute_util", "mem_util", "op_intensity")},
        },
        "warnings": warns,
    }


# --------------------------------------------------------------------------- #
# Palette + helpers                                                            #
# --------------------------------------------------------------------------- #
C_PREFILL = "#3B7DD8"   # compute
C_TRANSFER = "#E8743B"  # transfer (this NPU's own exposed comm)
C_HANDOFF = "#9B59B6"   # handoff wait (decode startup stall)
C_IDLE = "#B0B7C3"      # residual idle
C_DECODE = "#2E8B57"    # decode compute (1st token)
C_DECODE2 = "#6FB98F"   # later decode tokens
C_INK = "#1F2937"

# Order used by every stacked bar so colours mean the same thing everywhere.
BUCKETS = [("compute_ms", "Compute", C_PREFILL),
           ("comm_idle_ms", "Comm + idle", C_TRANSFER),
           ("wait_ms", "Handoff wait", C_HANDOFF)]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _stacked(ax, x, width, segments):
    """Draw stacked bars from a list of (values, label, colour); return the top
    of each bar so callers can annotate totals."""
    bottom = np.zeros(len(x))
    for vals, label, col in segments:
        ax.bar(x, vals, width, bottom=bottom, color=col, label=label)
        bottom = bottom + np.asarray(vals, dtype=float)
    return bottom


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_critical_path(a, outdir):
    """Wall-clock critical path: prefill compute -> prefill comm -> decode."""
    fig, ax = plt.subplots(figsize=(11, 3.0))
    t = 0.0
    for label, dur, col in [("Prefill compute", a["prefill_compute_ms"], C_PREFILL),
                            ("Prefill comm", a["prefill_comm_ms"], C_TRANSFER)]:
        ax.barh(0, dur, left=t, color=col, edgecolor="white")
        if dur > 0:
            ax.text(t + dur / 2, 0, f"{dur:.2f} ms", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        t += dur
    for i, d in enumerate(a["itl_ms"]):
        ax.barh(0, d, left=t, color=C_DECODE if i == 0 else C_DECODE2,
                edgecolor="white")
        t += d
    ax.axvline(a["ttft_ms"], color=C_INK, ls="--", lw=1.4)
    ax.text(a["ttft_ms"], -0.5, f"TTFT = {a['ttft_ms']:.2f} ms ",
            color=C_INK, fontsize=9, fontweight="bold", va="top", ha="right")
    ax.set_yticks([]); ax.set_ylim(-0.6, 0.7)
    ax.set_xlabel("Time along the critical path (ms)")
    ax.set_title(f"Critical path: prefill -> comm -> decode tokens "
                 f"(critical prefill NPU sys[{a['critical_prefill_sid']}])",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=[Patch(color=C_PREFILL, label="Prefill compute"),
                       Patch(color=C_TRANSFER, label="Prefill comm (KV push / PP)"),
                       Patch(color=C_DECODE, label="1st token"),
                       Patch(color=C_DECODE2, label="later tokens")],
              loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    return _save(fig, outdir, "01_critical_path.png")


def plot_per_npu(a, outdir):
    """Per-NPU wall split into compute / transfer / handoff-wait / idle."""
    npus = a["npus"]
    ids = sorted(npus)
    labels = []
    for i in ids:
        v = npus[i]
        tag = "P" if v["role"] == "prefill" else ("D" if v["role"] == "decode" else "?")
        labels.append(f"{tag} sys{i}\ns{v['pp_stage']}r{v['tp_rank']}")
    x = np.arange(len(ids))
    segments = [([npus[i][k] for i in ids], label, col) for k, label, col in BUCKETS]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(ids) + 3), 5))
    tops = _stacked(ax, x, 0.6, segments)
    for xi, tot in zip(x, tops):
        if tot > 0:
            ax.text(xi, tot, f"{tot:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-NPU wall time (P=prefill, D=decode)",
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
    ax.plot(x, itl, "-o", color=C_PREFILL, lw=2, ms=7, zorder=3, label="ITL per token")
    ax.axhline(a["avg_itl_ms"], color=C_TRANSFER, ls="--", lw=1.6, zorder=2,
               label=f"mean ITL = {a['avg_itl_ms']:.4f} ms")
    if len(itl) <= 16:
        for xi, v in zip(x, itl):
            ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xlabel("Generated token index"); ax.set_ylabel("Inter-token latency (ms)")
    ax.margins(y=0.30)
    ax.set_title("Inter-token latency (decode compute on the critical path)\n"
                 f"growth {spread_us:.3f} us over {len(itl)} tokens (KV lengthening)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    return _save(fig, outdir, "03_itl_per_token.png")


def plot_ttft_composition(a, outdir):
    """Donut of what makes up the Time-To-First-Token for this run."""
    parts = [("Prefill compute", a["prefill_compute_ms"], C_PREFILL),
             ("Prefill comm", a["prefill_comm_ms"], C_TRANSFER),
             ("1st decode step", a["ttft_first_step_ms"], C_DECODE)]
    parts = [(l, v, c) for (l, v, c) in parts if v > 0]
    if not parts or a["ttft_ms"] <= 0:
        return None
    vals = [v for _, v, _ in parts]
    cols = [c for _, _, c in parts]
    labels = [f"{l}\n{v:.2f} ms ({v/a['ttft_ms']*100:.0f}%)" for l, v, _ in parts]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.pie(vals, colors=cols, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white"),
           labels=labels, labeldistance=1.12,
           textprops=dict(fontsize=9, fontweight="bold"))
    ax.text(0, 0, f"TTFT\n{a['ttft_ms']:.2f} ms", ha="center", va="center",
            fontsize=13, fontweight="bold", color=C_INK)
    ax.set_title("What makes up the Time-To-First-Token", fontsize=12.5,
                 fontweight="bold")
    return _save(fig, outdir, "04_ttft_composition.png")


def plot_comm_cost(a, outdir):
    """Prefill comm cost put in perspective against the other latencies."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = ["Prefill\ncomm", "TTFT", "mean ITL\n(1 token)", "decode total\n(all tokens)"]
    vals = [a["prefill_comm_ms"], a["ttft_ms"], a["avg_itl_ms"], a["decode_total_ms"]]
    cols = [C_TRANSFER, C_INK, C_PREFILL, C_DECODE]
    for b, v in zip(ax.bar(names, vals, color=cols, edgecolor="white"), vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f} ms",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Prefill comm in perspective\n"
                 f"~ {a['prefill_comm_in_tokens']:.1f} decode tokens "
                 f"• {a['comm_share_of_ttft_pct']:.0f}% of TTFT",
                 fontsize=12, fontweight="bold")
    ax.margins(y=0.18); ax.grid(axis="y", alpha=0.3)
    return _save(fig, outdir, "05_comm_cost_perspective.png")


def plot_pool_summary(a, outdir):
    """Pool-level aggregate of the four buckets across all NPUs in each pool."""
    agg = a["pool_aggregate"]
    x = np.arange(2)
    segments = [([agg["prefill"][k], agg["decode"][k]], label, col)
                for k, label, col in BUCKETS]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    tops = _stacked(ax, x, 0.55, segments)
    for xi, tot in zip(x, tops):
        if tot > 0:
            ax.text(xi, tot, f"{tot:.1f} ms", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels(
        [f"Prefill pool\n({a['n_prefill_npus']} NPU)",
         f"Decode pool\n({a['n_decode_npus']} NPU)"], fontsize=10)
    ax.set_ylabel("Summed NPU time (ms)")
    ax.set_title("Time budget per pool (summed over NPUs)", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.margins(y=0.14)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, outdir, "06_pool_time_budget.png")


def plot_roofline(a, outdir):
    """Roofline profile per pool (means over NPUs) -- prefill compute-bound,
    decode memory-bound, straight from the log stats."""
    pm = a["pool_means"]
    rows = ["Compute bound (%)", "Avg compute util (%)", "Avg memory util (%)",
            "Avg operation intensity"]
    keys = ["compute_bound_pct", "compute_util", "mem_util", "op_intensity"]
    def fmt(x):
        return "-" if x is None else f"{x:.2f}"
    cell = [[fmt(pm["prefill"][k]), fmt(pm["decode"][k])] for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 3.0)); ax.axis("off")
    tbl = ax.table(cellText=cell, rowLabels=rows,
                   colLabels=[f"Prefill pool\n({a['n_prefill_npus']} NPU)",
                              f"Decode pool\n({a['n_decode_npus']} NPU)"],
                   cellLoc="center", rowLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for (r, c), cell_obj in tbl.get_celld().items():
        if r == 0:
            cell_obj.set_facecolor(C_INK)
            cell_obj.set_text_props(color="white", fontweight="bold")
    ax.set_title("Roofline profile per pool (mean over NPUs, from log stats)",
                 fontsize=12, fontweight="bold", pad=16)
    return _save(fig, outdir, "07_roofline_profile.png")


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def report(a):
    cfg = a["config"]
    L = ["=" * 70, "  DISAGGREGATED INFERENCE — single-run analysis", "=" * 70]
    if cfg:
        L.append(f"  Model: {cfg['model']['name']} | layers={cfg['model']['num_layers']} "
                 f"hidden={cfg['model']['hidden_size']}")
        L.append(f"  Prefill: tp={cfg['prefill']['tp']} pp={cfg['prefill']['pp']} "
                 f"({a['n_prefill_npus']} NPU)  |  "
                 f"Decode: tp={cfg['decode']['tp']} pp={cfg['decode']['pp']} "
                 f"({a['n_decode_npus']} NPU)")
        L.append(f"  KV transfer: {cfg['kv_transfer']['mode']}/{cfg['kv_transfer']['direction']} "
                 f"| decode steps: {cfg['decode_steps']} | serialize: {cfg['serialize']}")
    L.append(f"  Time factor from log: 1 cycle ~ {a['ms_per_cycle']*1e6:.3f} ns")
    L.append("")
    L.append("  --- Time To First Token (TTFT) ---")
    L.append(f"   prefill compute (path)     : {a['prefill_compute_ms']:.3f} ms")
    L.append(f"   prefill comm (KV push / PP): {a['prefill_comm_ms']:.3f} ms")
    L.append(f"   prefill ready              : {a['prefill_ready_ms']:.3f} ms")
    L.append(f"   1st decode step            : {a['ttft_first_step_ms']:.3f} ms")
    L.append(f"   TTFT TOTAL                 : {a['ttft_ms']:.3f} ms "
             f"(prefill-comm share {a['comm_share_of_ttft_pct']:.1f}%)")
    L.append("")
    L.append("  --- Decode ---")
    L.append(f"   mean ITL                   : {a['avg_itl_ms']:.4f} ms")
    if a["itl_ms"]:
        L.append(f"   ITL min / max              : {min(a['itl_ms']):.4f} / {max(a['itl_ms']):.4f} ms")
    L.append(f"   decode total (all tokens)  : {a['decode_total_ms']:.3f} ms")
    if a["decode_pipeline_bound_ms"] > 0:
        L.append(f"   pipeline bound (slowest st): {a['decode_pipeline_bound_ms']:.4f} ms/token")
    L.append(f"   decode pool handoff wait   : {a['decode_wait_ms']:.3f} ms")
    L.append("")
    L.append("  --- Prefill-comm impact ---")
    L.append(f"   prefill comm ~ {a['prefill_comm_in_tokens']:.1f} decode tokens")
    L.append(f"   amortised over {len(a['itl_ms'])} tokens : {a['prefill_comm_amortised_per_tok_ms']:.4f} ms/token")
    L.append(f"   end-to-end makespan        : {a['makespan_ms']:.3f} ms")
    L.append("")
    # FIX: Updated formatting string to reflect the 'comm_idle_ms' key instead of 'transfer_ms' and 'idle_ms'
    L.append("  --- Per-NPU (wall = compute + comm+idle + wait) ---")
    for i in sorted(a["npus"]):
        v = a["npus"][i]
        L.append(f"   sys[{i}] {v['role']:<7} s{v['pp_stage']}r{v['tp_rank']} | "
                 f"wall {v['wall_ms']:.3f} = comp {v['compute_ms']:.3f} + "
                 f"comm+idle {v['comm_idle_ms']:.3f} + wait {v['wait_ms']:.3f} ms "
                 f"| op-int {v['op_intensity']}")
    if a["warnings"]:
        L.append(""); L.append("  --- Warnings ---")
        for w in a["warnings"]:
            L.append(f"   ! {w}")
    L.append("=" * 70)
    out = "\n".join(L)
    print(out)
    return out


def write_npu_csv(a, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # FIX: Replaced old headers and dictionary references
        w.writerow(["sid", "role", "pp_stage", "tp_rank", "wall_ms", "compute_ms",
                    "comm_idle_ms", "wait_ms", "exposed_ms", "comm_total_ms",
                    "compute_bound_pct", "compute_util", "mem_util", "op_intensity",
                    "n_ops"])
        for i in sorted(a["npus"]):
            v = a["npus"][i]
            w.writerow([v["sid"], v["role"], v["pp_stage"], v["tp_rank"],
                        f"{v['wall_ms']:.6f}", f"{v['compute_ms']:.6f}",
                        f"{v['comm_idle_ms']:.6f}", f"{v['wait_ms']:.6f}",
                        f"{v['exposed_ms']:.6f}", f"{v['comm_ms']:.6f}",
                        v["compute_bound_pct"], v["compute_util"], v["mem_util"],
                        v["op_intensity"], v["n_ops"]])


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Deep single-run analysis of one ASTRA-sim disaggregated "
                    "inference log.")
    ap.add_argument("log", nargs="?", default="log.log", help="ASTRA-sim log file")
    ap.add_argument("-c", "--config", help="MLSynth YAML used to generate the traces")
    ap.add_argument("-o", "--outdir", default="run_analysis", help="output directory")
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        sys.exit(f"Log file not found: {args.log}")
    os.makedirs(args.outdir, exist_ok=True)

    ops, finished, stats = parse_log(args.log)
    if not stats and not finished:
        sys.exit("No statistics found — does the log have the expected format?")

    warns = []
    cfg = None
    if args.config:
        if not os.path.isfile(args.config):
            sys.exit(f"Config file not found: {args.config}")
        cfg = load_config(args.config)
        part, npf, ndc = build_partition(cfg)
        n_log = len(set(list(stats) + list(finished)))
        if (npf + ndc) != n_log:
            warns.append(f"Config implies {npf + ndc} NPUs but the log has {n_log} "
                         "sys[] entries — is this the right config for this run?")
    else:
        part, npf, ndc = heuristic_partition(stats, warns)

    ms_per_cycle = derive_ms_per_cycle(ops, stats)
    a = analyse(ops, finished, stats, ms_per_cycle, part, cfg, warns)

    figs = [plot_critical_path(a, args.outdir),
            plot_per_npu(a, args.outdir),
            plot_itl(a, args.outdir),
            plot_ttft_composition(a, args.outdir),
            plot_comm_cost(a, args.outdir),
            plot_pool_summary(a, args.outdir),
            plot_roofline(a, args.outdir)]
    figs = [f for f in figs if f]

    text = report(a)
    with open(os.path.join(args.outdir, "statistics.txt"), "w") as f:
        f.write(text + "\n")
    write_npu_csv(a, os.path.join(args.outdir, "per_npu.csv"))
    # summary.json is the hand-off point for any separate comparison script.
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({k: v for k, v in a.items() if k != "npus"}, f, indent=2)

    print("\nGenerated files:")
    for p in figs + [os.path.join(args.outdir, n) for n in
                     ("statistics.txt", "summary.json", "per_npu.csv")]:
        print("  ", p)


if __name__ == "__main__":
    main()