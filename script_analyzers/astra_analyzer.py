#!/usr/bin/env python3
"""
astra_analyzer — one ASTRA-sim run, in the time domain.

The sibling of ns3_analyzer.py on the compute side: where the sweeps collapse a
run to scalars, this keeps the whole timeline and answers the "when" questions --
when the first token is ready, how much of the KV-cache transfer is exposed on
the critical path, how skewed KV readiness is across decode ranks.

The reader half is shared, not re-implemented: utils.astra does the glob +
concat of stats_sys*.csv, the MLSynth name parsing, the send/recv tagging and
the counted-once de-dup (sends / collapse_collectives / unique_transfers);
utils.intervals does the union / overlap / subtract / concurrency algebra;
utils.paths resolves the run directory. The interactive shell lives in
astra_timeline_template.html. What stays here is the interpretation: the latency
metrics, the six charts and the timeline assembly. See utils/__init__.py for why
readers are shared and conclusions are not.

This grew out of the standalone csv_analyzer.py and reproduces its analysis; the
one intentional difference is the control-traffic threshold, now split into an
analysis value (aligned with utils.astra) and a lower visualisation value -- see
CONTROL_MAX_BYTES / TIMELINE_CONTROL_MAX_BYTES below.

    python3 astra_analyzer.py <model>/<tag>       # path under output/astra_logs
    python3 astra_analyzer.py --sweep buffer_sweep_T1 --tag T1_bx200_dcqcn_buf4

Outputs (results/astra_analysis/... by default):
    01_latency_breakdown.png  02_communication_mix.png  03_effective_bandwidth.png
    04_sys_utilisation.png    05_kv_transfer_impact.png 06_sync_skew.png
    timeline.html  summary.json  summary.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils import astra, intervals, paths
from utils.cli import Abort, need


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

# Root from which model/topology sub-paths are resolved when a relative --input
# is given, and the root under which per-run result folders are written.
DATA_DIR = paths.ROOT / "output" / "astra_logs"
RESULT_DIR = paths.ROOT / "results" / "astra_analysis"

# Two thresholds, on purpose, for the two different jobs this file does.
#
# ANALYSIS -- every number and chart. Payloads <= this are control / signalling
# (the 8-byte FIRSTTOK / DECFB handoffs), excluded from bandwidth, the comm mix,
# per-sys utilisation and the KV-exposure hide set. Matches utils.astra so the
# time-domain view and the sweep analyzers agree on what "a transfer" is. At 128
# a blocked 8-byte control message no longer counts as "hiding" KV, so the
# headline KV-exposure is higher (and more honest) than csv_analyzer's old value.
CONTROL_MAX_BYTES = 128

# VISUALISATION -- the interactive timeline only. Kept low so the tiny handoff
# messages stay drawn as their own bars on the canvas (useful to SEE), even though
# the analysis above folds them into control. Purely a display choice; it changes
# no metric. Raise it toward CONTROL_MAX_BYTES to declutter the timeline instead.
TIMELINE_CONTROL_MAX_BYTES = 1

# Colours (kept consistent between the HTML timeline and the PNG charts).
CAT_COLORS = {
    "COMPUTE_attn": "#4f9bff",
    "COMPUTE_ffw":  "#7c5cff",
    "COMPUTE":      "#5b8def",
    "TP":           "#22c08a",
    "KV":           "#ff6b6b",
    "KVREQ":        "#ff9f43",
    "PP":           "#f7c948",
    "DECFB":        "#c77dff",
    "FIRSTTOK":     "#9aa5b1",
    "CONTROL":      "#9aa5b1",
    "OTHER":        "#9aa5b1",
}

EXPOSED_COLOR = "#ff9f43"


# --------------------------------------------------------------------------- #
#  Categorisation  (name parsing / roles / send-recv tagging live in utils.astra)
# --------------------------------------------------------------------------- #

def categorise(row) -> str:
    """Return the analysis category for a row, based on type + name class.

    Finer-grained than utils.astra.op_class: GPU compute is split into attn / ffw
    so the shared colour scheme (and the timeline lanes) can tell them apart."""
    if row["type"] == "GPU":
        op = row.get("op")
        if op in ("attn", "ffw"):
            return f"COMPUTE_{op}"
        return "COMPUTE"
    cls = row.get("cls", "OTHER")
    if cls == "FIRSTTOK":
        return "FIRSTTOK"
    if cls in ("TP", "KV", "KVREQ", "PP", "DECFB"):
        return cls
    return "OTHER"


# --------------------------------------------------------------------------- #
#  Loading  (utils.astra.read_run + the derived columns this analyzer needs)
# --------------------------------------------------------------------------- #

def load_run(run_dir: Path, pattern: str = "*.csv") -> pd.DataFrame | None:
    """Read one run via the shared reader, then augment for the time-domain view."""
    df = astra.read_run(Path(run_dir), pattern)
    if df is None:
        return None
    return _augment(df)


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the time-domain analysis needs on top of the shared
    reader. read_run already provides everything error-prone -- cls / the naming
    fields / is_compute / is_comm / is_control / comm_role / wait_dominated -- so
    none of that is duplicated here. What is added is purely presentational: a
    usable integer time interval, a fine-grained `category` (compute split into
    attn/ffw for the colour scheme), and the per-row `eff_bw` / `bw_used` the
    bandwidth chart plots."""
    # drop rows without a usable time interval; recompute duration from the ticks
    # (robust to a stale duration column, which read_run passes through verbatim).
    df = df.dropna(subset=["start_tick", "end_tick"]).copy()
    df["start_tick"] = df["start_tick"].astype("int64")
    df["end_tick"] = df["end_tick"].astype("int64")
    df["duration"] = (df["end_tick"] - df["start_tick"]).clip(lower=0)

    df["category"] = df.apply(categorise, axis=1)
    # `is_control` is the ANALYSIS flag (CONTROL_MAX_BYTES); it drives every metric
    # and chart. read_run's send/recv tagging does not depend on it, so overriding
    # read_run's own is_control here is safe. The timeline uses its own, lower
    # threshold (see build_timeline_html / TIMELINE_CONTROL_MAX_BYTES).
    df["is_control"] = df["is_comm"] & (df["comm_size"] <= CONTROL_MAX_BYTES)

    # per-row effective bandwidth and the bandwidth actually used downstream
    # (reported achieved bw when present and positive, else size / duration).
    df["eff_bw"] = np.where(df["duration"] > 0, df["comm_size"] / df["duration"], np.nan)
    rep = pd.to_numeric(df.get("bw_bytes_per_ns"), errors="coerce")
    df["bw_used"] = np.where(np.isfinite(rep) & (rep > 0), rep, df["eff_bw"])
    return df


# --------------------------------------------------------------------------- #
#  Per-sys role + metrics
# --------------------------------------------------------------------------- #

def sys_roles(df: pd.DataFrame) -> dict:
    """sys_id -> {role, pl, ss, sh}. Derived by utils.astra.sys_roles (the shared
    rule: a rank's pool comes from its own COMP rows) and normalised to
    csv_analyzer's dict shape (role '?' rather than None, plus the pl short form)."""
    out = {}
    for sid, info in astra.sys_roles(df).items():
        role = info.get("role")
        out[int(sid)] = {
            "role": role or "?",
            "pl": astra.pool_of_role(role),
            "ss": info.get("ss"),
            "sh": info.get("sh"),
        }
    return out


def kv_ready_per_decode_npu(df: pd.DataFrame, roles: dict) -> list:
    """For each decode sys that receives KV, the instant its last KV chunk
    arrives.  KV rows on a decode sys are RECVs (the SEND side lives on the
    prefill pool), so the *arrival* is their end_tick; the spread of these
    instants across ranks is the synchronisation skew the first decode
    all-reduce has to absorb."""
    out = []
    kv = df[df["cls"] == "KV"]
    for sid, grp in kv.groupby("sys_id"):
        sid = int(sid)
        info = roles.get(sid, {})
        if info.get("role") != "decode":
            continue  # keep the receive side only
        out.append({
            "sys_id": sid,
            "stage": info.get("ss"),
            "shard": info.get("sh"),
            "kv_ready_tick": int(grp["end_tick"].max()),
        })
    out.sort(key=lambda r: r["kv_ready_tick"])
    return out


def compute_metrics(df: pd.DataFrame, roles: dict) -> dict:
    """Derive TTFT, decode-start, second-token latency, TPOT and sync-skew from
    the combined timeline."""
    m = {}
    t0 = int(df["start_tick"].min())
    t_end = int(df["end_tick"].max())
    m["t0"] = t0
    m["t_end"] = t_end
    m["makespan_ns"] = t_end - t0

    # A token is ready only when that step's forward pass has fully completed,
    # which for a tensor-parallel layer means AFTER the final TP all-reduce that
    # reassembles the partial sums -- not at the last ffw compute. The forward
    # set therefore includes both compute nodes and the pool's TP all-reduces
    # (cls == "TP"). PP / KV / DECFB transfers are excluded on purpose: they move
    # data between stages/pools or feed the next step, i.e. after the token has
    # already been produced. When tp_size == 1 there are no TP nodes and this
    # collapses to the compute-only behaviour.
    def _fwd_ops(pool: str) -> pd.DataFrame:
        sel = df[(df["pl"] == pool) & (df["is_compute"] | (df["cls"] == "TP"))].copy()
        if len(sel):
            sel["it_num"] = pd.to_numeric(sel["it"], errors="coerce")
        return sel

    prefill_fwd = _fwd_ops("p")
    decode_fwd = _fwd_ops("d")

    # --- prefill "first token" instant ---------------------------------------
    # The first token leaves the LAST pipeline stage, after its final layer's
    # forward (incl. the TP all-reduce that reassembles the partial sums).
    #
    # The simulator reports the TP shards of that final all-reduce finishing a
    # little apart (per-rank reporting skew of a collective that is logically a
    # barrier). A barrier cannot complete for ANY participant before the SLOWEST
    # one reaches it, so the token is only genuinely ready at the LATEST shard's
    # reported completion -- first_token_ready is therefore MAX-based (over the
    # final-stage final-layer forward ends / FIRSTTOK sends across TP shards).
    # This is what TTFT and the KV-exposure / 2nd-token waterfall are built on.
    # prefill_all_ready is kept as an alias of the same max-based instant for
    # backward-compatible callers/diagnostics.
    m["prefill_compute_end"] = int(prefill_fwd["end_tick"].max()) if len(prefill_fwd) else None
    m["prefill_all_ready_tick"] = m["prefill_compute_end"]
    m["prefill_all_ready_ns"] = ((m["prefill_compute_end"] - t0)
                                 if m["prefill_compute_end"] is not None else None)

    def _first_token_ready() -> int | None:
        # Prefer the FIRSTTOK send instant (shared with the sweeps' end-of-prefill
        # TTFT): send start == the moment a TP shard produced the first token for
        # its decode peer, taken as the max over shards since the final all-reduce
        # is a barrier. See utils.astra.firsttok_send_instant.
        inst = astra.firsttok_send_instant(df)
        if inst is not None:
            return int(inst)
        # Fallback: final-stage, final-layer forward end (slowest shard).
        if len(prefill_fwd):
            ss_num = pd.to_numeric(prefill_fwd["ss"], errors="coerce")
            fin = prefill_fwd[ss_num == ss_num.max()] if ss_num.notna().any() else prefill_fwd
            L_num = pd.to_numeric(fin["L"], errors="coerce")
            if L_num.notna().any():
                fin = fin[L_num == L_num.max()]
            return int(fin["end_tick"].max())
        return None

    # --- decode-start instant (decode it=0 begins) ---------------------------
    # This is gated by the KV-cache transfer + the first-token handoff, i.e. it
    # is network-sensitive. It is NOT the TTFT. (The first op of it=0 is the attn
    # compute, so the min start is unaffected by including TP nodes.)
    decode_start_tick = None
    if len(decode_fwd):
        it0 = decode_fwd[decode_fwd["it_num"] == 0]
        if len(it0):
            decode_start_tick = int(it0["start_tick"].min())
    m["decode_start_tick"] = decode_start_tick
    m["decode_start_ns"] = (decode_start_tick - t0) if decode_start_tick is not None else None

    # --- TTFT = first token produced (network-insensitive) -------------------
    # Built on first_token_ready (max over TP shards, since the final all-reduce
    # is a barrier that only completes once its slowest participant does).
    ttft_tick = _first_token_ready()
    if ttft_tick is None:
        ttft_tick = decode_start_tick
    m["ttft_tick"] = ttft_tick
    m["ttft_ns"] = (ttft_tick - t0) if ttft_tick is not None else None

    # --- first-token handoff exposed on the path to the 2nd token ------------
    # decode_start - first_token_ready: the prefill->decode FIRST-TOKEN handoff
    # latency that is not hidden behind prefill compute. This is a small,
    # specific quantity -- NOT the cost of moving the KV cache itself (that is
    # kv_transfer_exposed_ns below). Both instants live on the same causal chain,
    # so the gap is >= 0 (clamped for robustness).
    if decode_start_tick is not None and ttft_tick is not None:
        m["kv_handoff_exposed_ns"] = max(0, decode_start_tick - ttft_tick)
    else:
        m["kv_handoff_exposed_ns"] = None

    # --- KV-cache transfer exposed (the headline "KV transfer exposed") -------
    # Time the KV-cache transfer is actually on the wire with nothing hiding it
    # (no compute, no other real transfer). See kv_exposed_time(). In a run where
    # the KV transfer overlaps prefill compute it is small; when it serialises
    # into the pipeline (slow links / large cache) it dominates TTFT.
    m.update(kv_exposed_time(df, roles))
    # Back-compat alias: the headline metric is the exposed KV-cache transfer.
    m["kv_exposure_ns"] = m["kv_transfer_exposed_ns"]

    # --- second-token instant + TPOT -----------------------------------------
    # Each decode iteration's token is ready at the end of that iteration's last
    # TP all-reduce, hence decode_fwd (compute + TP) rather than compute-only.
    second_tick = None
    tpot_ns = None
    tpot_steady_ns = None
    it_end = {}
    if len(decode_fwd):
        for it_num, g in decode_fwd.dropna(subset=["it_num"]).groupby("it_num"):
            it_end[int(it_num)] = int(g["end_tick"].max())
        if 0 in it_end:
            second_tick = it_end[0]
        its = sorted(it_end)
        comp_times = [it_end[i] for i in its]
        if len(comp_times) >= 2:
            diffs = np.diff(comp_times)
            tpot_ns = float(np.mean(diffs))
            # Steady-state TPOT excludes the first gap (it0 -> it1), which carries
            # the first-decode-all-reduce / eager-RECV warm-up artefact.
            tpot_steady_ns = float(np.mean(diffs[1:])) if len(diffs) >= 2 else float(diffs[-1])
    m["second_token_tick"] = second_tick
    m["second_token_ns"] = (second_tick - t0) if second_tick is not None else None
    m["tpot_ns"] = tpot_ns
    m["tpot_steady_ns"] = tpot_steady_ns
    m["decode_it_end"] = it_end
    m["n_decode_iters"] = len(it_end)

    # --- synchronisation skew of KV readiness across decode ranks -------------
    # The first decode all-reduce is a TP collective *within one pipeline stage*
    # (it never spans stages), so the skew it must absorb is the spread of
    # per-rank KV-ready instants WITHIN a stage -- not the global min..max, which
    # is dominated by the inter-stage pipeline offset (stage 1 receives its KV
    # cache much later than stage 0) and is therefore not a synchronisation tax.
    kv_ready = kv_ready_per_decode_npu(df, roles)
    m["kv_ready_per_npu"] = kv_ready

    def _stage_spreads(rows):
        by_stage = {}
        for r in rows:
            by_stage.setdefault(str(r["stage"]), []).append(r["kv_ready_tick"])
        return {s: (max(v) - min(v)) for s, v in by_stage.items() if len(v) >= 2}

    spreads = _stage_spreads(kv_ready)
    m["sync_skew_per_stage_ns"] = spreads
    # the binding skew = the worst per-stage spread (what the slowest stage's
    # first all-reduce has to absorb).
    m["sync_skew_ns"] = max(spreads.values()) if spreads else None
    # spread within the head decode stage (stage 0), which gates the very first
    # decode all-reduce; falls back to the binding skew if stage info is absent.
    m["sync_skew_head_ns"] = spreads.get("0", m["sync_skew_ns"])

    # --- KV-cache "tail": how late the last KV chunk *arrives on the decode
    #     side* vs the instant decode actually needs it (= decode start).
    #     Use the receive (decode-pool) arrivals only -- a KV send completing on
    #     the prefill side is not what decode waits for. Positive => KV exposed
    #     (decode stalled on a late chunk); negative => KV fully prefetched. -----
    if kv_ready and decode_start_tick is not None:
        last_kv_arrival = max(r["kv_ready_tick"] for r in kv_ready)
        m["kv_last_end"] = int(last_kv_arrival)
        m["kv_tail_vs_decode_start_ns"] = int(last_kv_arrival) - decode_start_tick
    else:
        m["kv_last_end"] = None
        m["kv_tail_vs_decode_start_ns"] = None

    return m


# --------------------------------------------------------------------------- #
#  Exposed / overlapped communication per sys
# --------------------------------------------------------------------------- #
# Interval-set algebra (union / overlap / subtract / concurrency) is shared in
# utils.intervals; this module only decides which intervals to feed it.

def kv_exposed_time(df: pd.DataFrame, roles: dict) -> dict:
    """Time the KV-cache transfer is genuinely EXPOSED -- i.e. on the wire while
    no other useful work hides it.

    A KV chunk only adds to end-to-end latency for the part of its transfer that
    is NOT overlapped by something else making forward progress: GPU compute, or
    another *real* transfer. So `exposed = union(KV on-wire) - union(compute U
    real-other-comm)`.

    Two subtleties make this correct rather than degenerate:
      * On-wire KV = the SEND side only. A receive sits pre-posted and blocked
        until the data is produced upstream; counting its long blocked span as
        "KV in flight" would massively over-count.
      * The hiding set excludes pre-posted / blocked receives for the same
        reason -- a peer sitting blocked on a recv is NOT doing work that hides
        our KV transfer. Including every recv is what collapses the metric to 0
        (the blocked PP recvs span almost the whole run). We therefore subtract
        only compute and SEND-side non-KV communication.

    The headline number is computed on the prefill pool (the KV source): it is a
    subset of [t0, TTFT] by construction, so it composes cleanly with the TTFT /
    2nd-token waterfall. A per-sys breakdown (send-side exposure for prefill
    ranks; blocked recv-wait for decode ranks) is returned for transparency."""
    has_role = "comm_role" in df.columns

    def send_mask(frame):
        if has_role:
            return frame["comm_role"] == "send"
        return ~frame["wait_dominated"] if "wait_dominated" in frame.columns \
            else pd.Series(True, index=frame.index)

    # --- headline: prefill-pool exposed KV ---
    # KV/PP node names carry no pl= field, so the prefill pool is identified by
    # each sys's role (from roles), not by a name field.
    prefill_sids = {sid for sid, info in roles.items() if info.get("role") == "prefill"}
    pf = df[df["sys_id"].isin(prefill_sids)]
    kv = pf[(pf["cls"] == "KV") & ~pf["is_control"]]
    kv = kv[send_mask(kv)]
    kv_iv = intervals.merge(zip(kv["start_tick"], kv["end_tick"]))

    comp = list(zip(pf[pf["is_compute"]]["start_tick"], pf[pf["is_compute"]]["end_tick"]))
    othc = pf[pf["is_comm"] & ~pf["is_control"] & (pf["cls"] != "KV")]
    othc = othc[send_mask(othc)]
    oth = list(zip(othc["start_tick"], othc["end_tick"]))
    hide = intervals.merge(comp + oth)

    exposed_iv = intervals.subtract(kv_iv, hide)
    exposed_ns = int(intervals.total(exposed_iv))
    onwire_ns = int(intervals.total(kv_iv))

    # --- per-sys breakdown (uses each node's own compute + other-comm) ---
    per = []
    for sid, g in df.groupby("sys_id"):
        k = g[(g["cls"] == "KV") & ~g["is_control"]]
        if not len(k):
            continue
        sid = int(sid)
        role = roles.get(sid, {}).get("role", "?")
        # prefill ranks send; decode ranks only have the blocked recv-wait
        kiv = intervals.merge(zip(k["start_tick"], k["end_tick"]))
        other = g[g["is_comm"] & ~g["is_control"] & (g["cls"] != "KV")]
        b = intervals.merge(
            list(zip(g[g["is_compute"]]["start_tick"], g[g["is_compute"]]["end_tick"])) +
            list(zip(other["start_tick"], other["end_tick"])))
        per.append({
            "sys_id": sid,
            "role": role,
            "kv_onwire_ns": int(intervals.total(kiv)),
            "kv_exposed_ns": int(intervals.total(intervals.subtract(kiv, b))),
            "kind": "send" if role == "prefill" else "recv-wait",
        })
    per.sort(key=lambda r: (r["role"], r["sys_id"]))

    return {
        "kv_transfer_exposed_ns": exposed_ns,
        "kv_onwire_ns": onwire_ns,
        "kv_exposed_intervals": exposed_iv,
        "kv_exposed_per_sys": per,
    }


def per_sys_utilisation(df: pd.DataFrame, roles: dict) -> list:
    """For each sys: busy compute time, comm time hidden behind compute, and the
    idle span (pipeline bubble = waiting for peers / network)."""
    rows = []
    for sid, grp in df.groupby("sys_id"):
        sid = int(sid)
        span0 = int(grp["start_tick"].min())
        span1 = int(grp["end_tick"].max())
        span = span1 - span0
        comp = list(zip(grp[grp["is_compute"]]["start_tick"],
                        grp[grp["is_compute"]]["end_tick"]))
        # genuine transfers only: drop tiny control traffic AND the pre-posted
        # recvs that merely sit blocked (their span is waiting, not transfer).
        commg = grp[grp["is_comm"] & ~grp["is_control"] & ~grp["wait_dominated"]]
        comm = list(zip(commg["start_tick"], commg["end_tick"]))
        comp_busy = int(intervals.union_len(comp))
        comm_union = int(intervals.union_len(comm))
        comm_hidden = int(intervals.overlap_len(comp, comm))
        comm_exposed = comm_union - comm_hidden
        # idle = span not covered by compute (pipeline bubble; includes the
        # initial wait for prefill/KV on decode shards)
        idle = span - comp_busy
        rows.append({
            "sys_id": sid,
            "role": roles.get(sid, {}).get("role", "?"),
            "span": span,
            "compute_busy": comp_busy,
            "comm_union": comm_union,
            "comm_hidden": comm_hidden,
            "comm_exposed": comm_exposed,
            "idle": idle,
        })
    rows.sort(key=lambda r: (r["role"], r["sys_id"]))
    return rows


# --------------------------------------------------------------------------- #
#  Time formatting helpers
# --------------------------------------------------------------------------- #

def fmt_time(ns):
    """ns -> human-readable string with adaptive unit."""
    if ns is None or (isinstance(ns, float) and not np.isfinite(ns)):
        return "n/a"
    ns = float(ns)
    a = abs(ns)
    if a >= 1e9:
        return f"{ns/1e9:.3f} s"
    if a >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if a >= 1e3:
        return f"{ns/1e3:.3f} µs"
    return f"{ns:.0f} ns"


def _ms_axis(ax, axis="x"):
    fmt = FuncFormatter(lambda v, _: f"{v/1e6:.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------- #
#  PNG charts (communication impact)
# --------------------------------------------------------------------------- #

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafbfc",
        "axes.edgecolor": "#cfd6dd",
        "axes.grid": True,
        "grid.color": "#e6eaef",
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 130,
    })


def chart_latency_breakdown(df, metrics, roles, out):
    """Waterfall of the critical path to the first and second token, plus the
    per-iteration decode latency (TPOT).

    The path to the second token is split into three faithful segments:
      prefill compute (-> first token / TTFT),
      KV transfer + first-token handoff exposed (network-sensitive),
      decode step 1 compute (-> second token).
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    t0 = metrics["t0"]
    ttft = metrics["ttft_ns"]
    dstart = metrics["decode_start_ns"]
    second = metrics["second_token_ns"]
    kv_tx = metrics.get("kv_transfer_exposed_ns") or 0   # exposed KV-cache transfer
    handoff = metrics.get("kv_handoff_exposed_ns") or 0  # first-token handoff gap
    all_ready = metrics.get("prefill_all_ready_ns")

    # --- left: critical-path composition, tiling [0, second] --------------- #
    #   prefill region [0, ttft] is split into the part hidden/served by
    #     compute+pipeline and the part that is EXPOSED KV-cache transfer;
    #   then the first-token handoff gap [ttft, decode_start];
    #   then decode step 1 -> 2nd token.
    segs = []  # (label, start_ns, len_ns, color)
    if ttft is not None:
        kv_part = min(kv_tx, ttft)
        base_part = max(0, ttft - kv_part)
        segs.append(("Prefill compute\n+ pipeline", 0, base_part, "#5b8def"))
        if kv_part > 0:
            segs.append(("Exposed KV-cache\ntransfer", base_part, kv_part, "#ff6b6b"))
    exp_len = max(0, (dstart or 0) - (ttft or 0))
    if exp_len > 0:
        segs.append(("First-token\nhandoff", ttft, exp_len, EXPOSED_COLOR))
    base_decode = max(ttft or 0, dstart or 0) if (ttft is not None or dstart is not None) else None
    if base_decode is not None and second is not None and second > base_decode:
        segs.append(("Decode step 1\n(→ 2nd token)", base_decode,
                     second - base_decode, "#7c5cff"))

    scale_ref = (second or dstart or ttft or 1)
    for label, s, l, c in segs:
        axL.barh(0, l, left=s, color=c, edgecolor="white", height=0.5)
        if l > scale_ref * 0.04:
            axL.text(s + l / 2, 0, f"{label}\n{fmt_time(l)}", ha="center",
                     va="center", fontsize=8.5, color="white", fontweight="bold")

    if ttft is not None:
        axL.axvline(ttft, color="#ff6b6b", lw=2, ls="--")
        axL.text(ttft, 0.42, f"TTFT\n{fmt_time(ttft)}", color="#ff6b6b",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    # the slowest TP shard's reported finish (only annotate if meaningfully later)
    if all_ready is not None and ttft is not None and (all_ready - ttft) > scale_ref * 0.01:
        axL.axvline(all_ready, color="#ff6b6b", lw=1, ls=":", alpha=0.6)
    if dstart is not None and exp_len > 0:
        axL.axvline(dstart, color="#d97706", lw=2, ls="--")
        axL.text(dstart, 0.42, f"decode start\n{fmt_time(dstart)}", color="#b45309",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    if second is not None:
        axL.axvline(second, color="#22c08a", lw=2, ls="--")
        axL.text(second, -0.42, f"2nd token\n{fmt_time(second)}", color="#159b6e",
                 ha="center", va="top", fontsize=9.5, fontweight="bold")

    axL.set_ylim(-0.9, 0.9)
    axL.set_yticks([])
    axL.set_xlabel("time since request start (ms)")
    _ms_axis(axL)
    parts = []
    if ttft:
        parts.append(f"exposed KV-cache transfer {fmt_time(kv_tx)} "
                     f"({100.0 * kv_tx / ttft:.0f}% of TTFT)")
    parts.append(f"first-token handoff {fmt_time(handoff)}")
    axL.set_title("Critical path to first & second token\n" + "  ·  ".join(parts),
                  fontsize=10.5)
    axL.margins(x=0.02)

    # --- right: per-iteration decode latency (TPOT) ------------------------ #
    it_end = metrics["decode_it_end"]
    its = sorted(it_end)
    if len(its) >= 2:
        comp = [it_end[i] for i in its]
        gaps = np.diff(comp) / 1e6  # ms
        colors = ["#c77dff"] + ["#7c5cff"] * (len(gaps) - 1)  # mark the warm-up gap
        axR.bar(its[1:], gaps, color=colors, edgecolor="white")
        tpot = metrics["tpot_ns"]
        tpot_s = metrics.get("tpot_steady_ns")
        if tpot:
            axR.axhline(tpot / 1e6, color="#ff6b6b", ls="--", lw=2,
                        label=f"mean TPOT = {fmt_time(tpot)}")
        if tpot_s:
            axR.axhline(tpot_s / 1e6, color="#159b6e", ls=":", lw=2,
                        label=f"steady-state TPOT = {fmt_time(tpot_s)}")
        axR.legend(loc="upper right", fontsize=8.5)
        axR.set_xlabel("decode iteration (output token #)")
        axR.set_ylabel("time per token (ms)")
        axR.set_title("Decode latency per token (TPOT)\n(first gap = warm-up)")
        axR.set_xticks(its[1:])
    else:
        axR.text(0.5, 0.5, "Not enough decode iterations\nto compute TPOT",
                 ha="center", va="center", transform=axR.transAxes, color="#888")
        axR.set_xticks([]); axR.set_yticks([])
        axR.set_title("Decode latency per token (TPOT)")

    fig.suptitle("Serving latency breakdown", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def aggregate_comm_time(df: pd.DataFrame) -> list:
    """Aggregate on-wire time and bytes per communication class, counting each
    transfer exactly once.

    The de-dup itself lives in utils.astra.unique_transfers: point-to-point
    classes (KV / PP / DECFB / KVREQ) are taken on their SEND side (the blocked
    RECV's duration is upstream wait, not time on the wire), and the TP all-reduce
    is collapsed to one representative per logical collective (keyed by
    pl/ss/L/it/op, slowest rank for the duration) rather than summed once per
    participating rank. Here we only sum what it returns.

    Returns a list of dicts {cls, total_dur, total_bytes, count} sorted by
    total_dur ascending."""
    comm = df[df["is_comm"] & ~df["is_control"]]
    rows = []
    for cls in ("KV", "PP", "DECFB", "KVREQ", "TP"):
        t = astra.unique_transfers(comm, cls)
        if t is None or t.empty:
            continue
        rows.append({
            "cls": cls,
            "total_dur": float(t["duration"].sum()),
            "total_bytes": float(t["comm_size"].sum()),
            "count": int(len(t)),
        })
    rows.sort(key=lambda r: r["total_dur"])
    return rows


def chart_comm_breakdown(df, out):
    """Total time-on-wire and total bytes moved, per communication class.

    Uses the send side for point-to-point traffic and one representative per TP
    collective, so the KV total is not inflated by blocked receives and the TP
    total is not multiplied by the number of participating ranks."""
    rows = aggregate_comm_time(df)
    if not rows:
        return False

    labels = [r["cls"] for r in rows]
    durs = np.array([r["total_dur"] for r in rows])
    payload = np.array([r["total_bytes"] for r in rows])
    colors = [CAT_COLORS.get(c, "#888") for c in labels]
    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.barh(y, durs / 1e6, color=colors, edgecolor="white")
    for i, r in enumerate(rows):
        ax1.text(r["total_dur"] / 1e6, i,
                 f"  {fmt_time(r['total_dur'])}  (n={r['count']})",
                 va="center", fontsize=8.5)
    ax1.set_yticks(y); ax1.set_yticklabels(labels)
    ax1.set_xlabel("aggregate time on wire (ms)")
    ax1.set_title("Time spent per communication type\n"
                  "(send side; TP collapsed per collective)", fontsize=11)
    ax1.margins(x=0.18)

    ax2.barh(y, payload / 1e9, color=colors, edgecolor="white")
    for i, r in enumerate(rows):
        ax2.text(r["total_bytes"] / 1e9, i, f"  {r['total_bytes']/1e9:.2f} GB",
                 va="center", fontsize=8.5)
    ax2.set_yticks(y); ax2.set_yticklabels(labels)
    ax2.set_xlabel("total payload moved (GB)")
    ax2.set_title("Data volume per communication type")
    ax2.margins(x=0.18)

    fig.suptitle("Communication mix (control/serialize traffic excluded)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_bandwidth(df, out):
    """Effective-bandwidth distribution per class, computed on the transmit
    (SEND) side and using the simulator-reported bandwidth when available.

    Using the SEND side avoids attributing the blocked-recv duration to the link
    (a scheduling artefact), and the reported per-node bandwidth, when present,
    sidesteps size/duration entirely.  Receives / wait-dominated samples are
    still drawn, faintly, so the artefact is visible rather than hidden."""
    comm = df[df["is_comm"] & ~df["is_control"]].copy()
    comm = comm[np.isfinite(comm["bw_used"]) & (comm["bw_used"] > 0)]
    if comm.empty:
        return False

    classes = [c for c in ["KV", "PP", "TP", "KVREQ", "DECFB"]
               if c in comm["cls"].unique()]
    fig, ax = plt.subplots(figsize=(11, 5.2))

    has_role = "comm_role" in comm.columns
    positions, labels = [], []
    pos = 0
    for cls in classes:
        sub = comm[comm["cls"] == cls]
        # reliable = SEND side (or, if no role info, the non-wait-dominated rows)
        if has_role and (sub["comm_role"] == "send").any():
            rel = sub[sub["comm_role"] == "send"]["bw_used"].values
            wt = sub[sub["comm_role"] != "send"]["bw_used"].values
        else:
            rel = sub[~sub["wait_dominated"]]["bw_used"].values
            wt = sub[sub["wait_dominated"]]["bw_used"].values
        c = CAT_COLORS.get(cls, "#888")
        if len(rel):
            jitter = (np.random.RandomState(pos).rand(len(rel)) - 0.5) * 0.28
            ax.scatter(np.full(len(rel), pos) + jitter, rel, s=18, color=c,
                       alpha=0.55, edgecolor="none", zorder=3)
            ax.scatter([pos], [np.median(rel)], marker="_", s=900, color="#222",
                       linewidths=2.5, zorder=4)
            ax.text(pos, np.median(rel), f"  median\n  {np.median(rel):.0f} GB/s",
                    fontsize=8, va="center", ha="left", color="#222")
        if len(wt):
            jitter = (np.random.RandomState(pos + 99).rand(len(wt)) - 0.5) * 0.28
            ax.scatter(np.full(len(wt), pos) + jitter, wt, s=18, color="#bbb",
                       alpha=0.6, marker="x", zorder=2)
        positions.append(pos)
        labels.append(cls)
        pos += 1

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("achieved bandwidth  (GB/s, log)")
    ax.set_title("Per-transfer achieved bandwidth (send side)  —  congestion indicator")
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               markersize=8, label="send side (reliable)"),
        Line2D([0], [0], marker="x", color="#bbb", linestyle="None",
               markersize=8, label="recv / wait-dominated (unreliable)"),
        Line2D([0], [0], marker="_", color="#222", markersize=14, lw=2.5,
               label="median (reliable)"),
    ]
    ax.legend(handles=proxies, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_utilisation(util_rows, out):
    """Per-sys breakdown: compute busy vs idle (pipeline bubble) and how much
    communication was hidden behind compute vs exposed."""
    if not util_rows:
        return False
    labels = [f"sys{r['sys_id']}\n({r['role']})" for r in util_rows]
    busy = np.array([r["compute_busy"] for r in util_rows]) / 1e6
    idle = np.array([r["idle"] for r in util_rows]) / 1e6
    hidden = np.array([r["comm_hidden"] for r in util_rows]) / 1e6
    exposed = np.array([r["comm_exposed"] for r in util_rows]) / 1e6

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, len(labels) * 0.9), 8.5),
                                   sharex=True)

    ax1.bar(x, busy, color="#5b8def", label="GPU compute busy")
    ax1.bar(x, idle, bottom=busy, color="#e8edf3", edgecolor="#cfd6dd",
            label="idle (pipeline bubble / waiting)")
    ax1.set_ylabel("time (ms)")
    ax1.set_title("GPU occupancy per sys  —  idle = stalled on peers / network")
    ax1.legend(loc="upper right")

    ax2.bar(x, hidden, color="#22c08a", label="comm hidden behind compute")
    ax2.bar(x, exposed, bottom=hidden, color="#ff6b6b",
            label="comm exposed (on critical path)")
    ax2.set_ylabel("time (ms)")
    ax2.set_title("Communication: hidden vs exposed per sys")
    ax2.legend(loc="upper right")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_kv_timeline_impact(df, metrics, out):
    """How the KV-cache transfers line up against the instant decode actually
    starts: shows the 'tail' of KV traffic that decode must wait for before it
    can produce tokens."""
    kv = df[(df["cls"] == "KV") & ~df["is_control"]].copy()
    if kv.empty:
        return False
    t0 = metrics["t0"]
    ref = metrics.get("decode_start_tick") or metrics.get("ttft_tick")

    fig, ax = plt.subplots(figsize=(11, 5))
    # one horizontal lollipop per KV transfer, ordered by completion. Only the
    # send side carries a meaningful on-wire interval.
    if "comm_role" in kv.columns and (kv["comm_role"] == "send").any():
        kv = kv[kv["comm_role"] == "send"]
    kv = kv.sort_values("end_tick").reset_index(drop=True)
    y = np.arange(len(kv))
    wait = kv["wait_dominated"].values
    colors = np.where(wait, "#ff9f43", "#ff6b6b")
    ax.hlines(y, (kv["start_tick"] - t0) / 1e6, (kv["end_tick"] - t0) / 1e6,
              color=colors, lw=1.2, alpha=0.8)
    ax.scatter((kv["end_tick"] - t0) / 1e6, y, s=10, color=colors, zorder=3)

    if ref is not None:
        ax.axvline((ref - t0) / 1e6, color="#d97706", lw=2, ls="--",
                   label=f"decode start ({fmt_time(metrics.get('decode_start_ns'))})")
        ax.legend(loc="lower right")
    ax.set_xlabel("time since request start (ms)")
    ax.set_ylabel("KV-cache transfer (sorted by completion)")
    tail = metrics.get("kv_tail_vs_decode_start_ns")
    tail_txt = ("last KV completes %s decode start"
                % (f"{fmt_time(abs(tail))} after" if (tail or 0) > 0
                   else f"{fmt_time(abs(tail or 0))} before")) if tail is not None else ""
    ax.set_title("KV-cache transfers vs. decode-start instant\n" + tail_txt,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def chart_sync_skew(df, metrics, roles, out):
    """Synchronisation skew: the spread of per-rank KV-ready instants on the
    decode side.  The first decode all-reduce is gated by the *slowest* KV
    receiver, so this spread is a data-dependency synchronisation tax that is
    distinct from raw bandwidth contention."""
    ready = metrics.get("kv_ready_per_npu") or []
    if len(ready) < 2:
        return False
    t0 = metrics["t0"]

    # group by decode pp stage for colour; head stage (== '0') gates the first AR
    stages = sorted({str(r["stage"]) for r in ready})
    cmap = plt.get_cmap("viridis")
    stage_color = {s: cmap(i / max(1, len(stages) - 1)) for i, s in enumerate(stages)}

    ready_sorted = sorted(ready, key=lambda r: r["kv_ready_tick"])
    y = np.arange(len(ready_sorted))
    xs = [(r["kv_ready_tick"] - t0) / 1e6 for r in ready_sorted]
    cols = [stage_color[str(r["stage"])] for r in ready_sorted]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # stem from the earliest ready instant to each rank's ready instant
    x_min = min(xs)
    ax.hlines(y, x_min, xs, color="#cbd5e1", lw=1.2, zorder=1)
    ax.scatter(xs, y, s=70, color=cols, zorder=3, edgecolor="white", linewidth=0.8)

    # shade each stage's within-stage skew band (this is the spread that stage's
    # first decode all-reduce must absorb); highlight the head stage (== '0').
    for st in stages:
        sx = [(r["kv_ready_tick"] - t0) / 1e6 for r in ready_sorted
              if str(r["stage"]) == st]
        if len(sx) < 2:
            continue
        is_head = (st == "0")
        ax.axvspan(min(sx), max(sx), color="#ff6b6b" if is_head else "#9aa5b1",
                   alpha=0.12 if is_head else 0.07, zorder=0)
    head = [r for r in ready_sorted if str(r["stage"]) == "0"]
    if len(head) >= 2:
        hx = [(r["kv_ready_tick"] - t0) / 1e6 for r in head]
        ax.axvline(min(hx), color="#ff6b6b", lw=1, ls=":", alpha=0.7)
        ax.axvline(max(hx), color="#ff6b6b", lw=1.4, ls="--",
                   label=f"head-stage skew = {fmt_time(metrics.get('sync_skew_head_ns'))}")

    dstart = metrics.get("decode_start_ns")
    if dstart is not None:
        # x-axis is in ms (data points are /1e6); convert the marker too.
        ax.axvline(dstart / 1e6, color="#d97706", lw=2, ls="--",
                   label=f"decode start ({fmt_time(dstart)})")

    ax.set_yticks(y)
    ax.set_yticklabels([f"sys{r['sys_id']} (stage {r['stage']}, shard {r['shard']})"
                        for r in ready_sorted], fontsize=8)
    ax.set_xlabel("KV-ready instant since request start (ms)")
    skew = metrics.get("sync_skew_ns")
    ax.set_title("Per-rank KV-ready skew on the decode side\n"
                 f"worst within-stage skew = {fmt_time(skew)}  "
                 f"(absorbed by that stage's first decode all-reduce)", fontsize=11)

    # stage legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="None", color="w",
                      markerfacecolor=stage_color[s], markersize=9,
                      label=f"decode stage {s}") for s in stages]
    leg1 = ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=8)
    ax.add_artist(leg1)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
#  Interactive HTML timeline (canvas, raster)
# --------------------------------------------------------------------------- #

_TEMPLATE_PATH = Path(__file__).with_name("utils/astra_timeline_template.html")


@lru_cache(maxsize=1)
def _timeline_template() -> str:
    """The static HTML/canvas/JS shell, read from astra_timeline_template.html.
    build_timeline_html injects the run into its __PLACEHOLDER__ slots."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def build_timeline_html(df, metrics, roles, title):
    """Assemble lane layout + bar list and inject into the HTML template."""
    # Keep compute + comm above the VISUALISATION control threshold, which is
    # lower than the analysis one: the tiny FIRSTTOK / DECFB handoffs stay drawn
    # as their own bars here even though every metric folds them into control.
    viz_control = df["is_comm"] & (df["comm_size"] <= TIMELINE_CONTROL_MAX_BYTES)
    vis = df[df["is_compute"] | (df["is_comm"] & ~viz_control)].copy()

    # ---- lane layout: per sys -> [compute, TP, KV, PP, DECFB, OTHER] ----
    lane_order = ["compute", "TP", "KV", "PP", "KVREQ", "DECFB", "OTHER"]

    def lane_key_of(row):
        if row["is_compute"]:
            return "compute"
        return row["cls"] if row["cls"] in lane_order else "OTHER"

    vis["lane_key"] = vis.apply(lane_key_of, axis=1)

    lanes = []          # list of dicts {label, role}
    lane_index = {}     # (sys, lane_key) -> idx
    lane_key_by_idx = []  # idx -> lane_key ("compute"/"TP"/"KV"/...)
    for sid in sorted(vis["sys_id"].unique()):
        sid = int(sid)
        role = roles.get(sid, {}).get("role", "?")
        present = vis[vis["sys_id"] == sid]["lane_key"].unique()
        for lk in lane_order:
            if lk in present:
                idx = len(lanes)
                lane_index[(sid, lk)] = idx
                pretty = "compute" if lk == "compute" else lk
                lanes.append({"label": f"sys{sid} · {pretty}", "role": role})
                lane_key_by_idx.append(lk)

    # ---- category index table ----
    cats = sorted(vis["category"].unique())
    cat_idx = {c: i for i, c in enumerate(cats)}

    # ---- bars ----
    # `flow_key` identifies the *physical* flow a comm event belongs to (its
    # src/dst stage + shard pair, from the node name), independent of
    # layer/iteration -- used below only to order rows so a connection's
    # repeated transfers land on adjacent lines; it never causes two bars to
    # share a row.
    def flow_key_of(r):
        if r.is_compute:
            return None
        parts = [f"{k}={v}" for k in ("ss", "ds", "ssh", "dsh", "sh")
                 if (v := getattr(r, k, None)) is not None and not (isinstance(v, float) and np.isnan(v))]
        return "|".join(parts) or None

    # In "Hide concurrent comms" (flat) mode every bar in a lane is drawn on sub-row
    # 0, so overlapping transfers stack on the canvas and only the last-drawn one is
    # visible -- painter's algorithm. That draw order is just `vis`'s row order
    # unless we fix it, which is CSV read order and carries no meaning. The one
    # class where transfers carry a destination layer is KV (`L=` in the name; PP
    # and DECFB don't -- they move a whole stage boundary, not a per-layer chunk),
    # so sort by L: whichever recv is hiding the others in the flat view is then
    # always the highest-layer one in flight, not whichever happened to be read
    # from disk last. Rows without an L (compute, PP, DECFB, TP, ...) keep their
    # relative order -- `stable` only reorders the KV rows that actually have one.
    l_num = pd.to_numeric(vis["L"], errors="coerce") if "L" in vis.columns else np.nan
    vis = vis.assign(_lsort=l_num).sort_values("_lsort", kind="stable", na_position="last")

    bars = []
    flow_keys = []
    for r in vis.itertuples(index=False):
        li = lane_index.get((int(r.sys_id), r.lane_key))
        if li is None:
            continue
        bar = {
            "l": li,
            "a": int(r.start_tick),
            "b": int(r.end_tick),
            "c": cat_idx[r.category],
            "n": r.name,
        }
        if r.is_comm:
            bar["s"] = float(r.comm_size)
            bar["d"] = (float(r.eff_bw) if np.isfinite(r.eff_bw) else None)
            bar["w"] = bool(r.wait_dominated)
        bars.append(bar)
        flow_keys.append(flow_key_of(r))

    # ---- sub-row layout ------------------------------------------------------ #
    # Most lanes (compute, TP, PP, ...) rarely have genuinely overlapping bars,
    # so they use the original greedy minimal-track packing: reuse a track as
    # soon as it frees up, so a non-overlapping lane stays a single flat row.
    #
    # The KV lane is different: this is exactly where concurrent transfers to
    # distinct destinations matter, and reusing tracks there hides that a row
    # can splice together segments from unrelated flows.  So for KV only,
    # every transfer gets its OWN permanent row -- never shared with any other
    # bar, overlapping or not.  Rows are ordered by flow identity (src/dst
    # stage + shard, from the node name), then start time, purely so a
    # connection's repeated transfers land on adjacent lines.
    #
    # `depth` = rows needed for layout; `peak` = true simultaneous-overlap
    # count, kept separately for the concurrency badge/tooltip.
    from collections import defaultdict
    lane_bar_idx = defaultdict(list)
    for bi, b in enumerate(bars):
        lane_bar_idx[b["l"]].append(bi)

    for li in range(len(lanes)):
        idxs = lane_bar_idx.get(li, [])

        if lane_key_by_idx[li] == "KV":
            groups = defaultdict(list)
            for i in idxs:
                groups[flow_keys[i]].append(i)
            ordered_keys = sorted(groups, key=lambda fk: min(bars[i]["a"] for i in groups[fk]))

            row = 0
            for fk in ordered_keys:
                for i in sorted(groups[fk], key=lambda i: bars[i]["a"]):
                    bars[i]["k"] = row
                    row += 1
            lanes[li]["depth"] = max(1, row)
        else:
            idxs_sorted = sorted(idxs, key=lambda i: (bars[i]["a"], bars[i]["b"]))
            track_end = []                       # last end tick per open track
            for i in idxs_sorted:
                a, bnd = bars[i]["a"], bars[i]["b"]
                slot = next((k for k in range(len(track_end)) if track_end[k] <= a), None)
                if slot is None:
                    slot = len(track_end)
                    track_end.append(bnd)
                else:
                    track_end[slot] = bnd
                bars[i]["k"] = slot
            lanes[li]["depth"] = max(1, len(track_end))

        # true simultaneous-overlap count, for the concurrency badge/tooltip
        lanes[li]["peak"] = max(1, intervals.max_concurrency(
            (bars[i]["a"], bars[i]["b"]) for i in idxs))

    # ---- peak concurrent KV transfers (header metric) ---------------------- #
    # How many KV-cache transfers are simultaneously on the wire at the busiest
    # instant.  Prefer the SEND side (real transmit-side transfers) so paired
    # send/recv rows are not double counted.
    kv_vis = vis[vis["category"] == "KV"]
    if "comm_role" in kv_vis.columns and (kv_vis["comm_role"] == "send").any():
        kv_prof_rows = kv_vis[kv_vis["comm_role"] == "send"]
    else:
        kv_prof_rows = kv_vis
    kv_peak = intervals.max_concurrency(
        zip(kv_prof_rows["start_tick"], kv_prof_rows["end_tick"]))

    data = {
        "t0": metrics["t0"],
        "tEnd": metrics["t_end"],
        "ttft": metrics["ttft_tick"],
        "decodeStart": metrics.get("decode_start_tick"),
        "second": metrics["second_token_tick"],
        "lanes": lanes,
        "cats": cats,
        "colors": {c: CAT_COLORS.get(c, "#888") for c in cats},
        "bars": bars,
        "kvPeak": int(kv_peak),
    }

    html = (_timeline_template()
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__TITLE__", title)
            .replace("__TTFT__", fmt_time(metrics["ttft_ns"]))
            .replace("__KVEXP__", fmt_time(metrics.get("kv_exposure_ns")))
            .replace("__SECOND__", fmt_time(metrics["second_token_ns"]))
            .replace("__TPOT__", fmt_time(metrics["tpot_ns"]))
            .replace("__KVPEAK__", f"{data['kvPeak']}×")
            .replace("__MAKESPAN__", fmt_time(metrics["makespan_ns"])))
    return html


# --------------------------------------------------------------------------- #
#  Pipeline + CLI
# --------------------------------------------------------------------------- #

def analyse(run_dir: Path, out_dir: Path, title: str, pattern: str = "*.csv") -> None:
    """Load one run, compute metrics, and write the six charts, the interactive
    timeline and the summary into out_dir."""
    print(f"Loading CSVs from {run_dir} ...")
    df = load_run(run_dir, pattern)
    need(df is not None and not df.empty, f"no readable CSV rows in {run_dir}")

    roles = sys_roles(df)
    metrics = compute_metrics(df, roles)
    util = per_sys_utilisation(df, roles)

    print(f"  {len(df):,} timeline events across {df['sys_id'].nunique()} sys instances")
    print("  roles: " + ", ".join(
        f"sys{s}={r['role']}" for s, r in sorted(roles.items())))
    print(f"  TTFT={fmt_time(metrics['ttft_ns'])}  "
          f"decode-start={fmt_time(metrics['decode_start_ns'])}")
    print(f"  KV-cache transfer exposed={fmt_time(metrics['kv_transfer_exposed_ns'])}"
          f" (of {fmt_time(metrics['kv_onwire_ns'])} on-wire)  "
          f"first-token handoff={fmt_time(metrics['kv_handoff_exposed_ns'])}")
    print(f"  2nd-token={fmt_time(metrics['second_token_ns'])}  "
          f"TPOT={fmt_time(metrics['tpot_ns'])}  "
          f"TPOT(steady)={fmt_time(metrics['tpot_steady_ns'])}  "
          f"sync-skew={fmt_time(metrics['sync_skew_ns'])}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- charts ----
    _style()
    print("Rendering charts ...")
    chart_latency_breakdown(df, metrics, roles, out_dir / "01_latency_breakdown.png")
    chart_comm_breakdown(df, out_dir / "02_communication_mix.png")
    chart_bandwidth(df, out_dir / "03_effective_bandwidth.png")
    chart_utilisation(util, out_dir / "04_sys_utilisation.png")
    chart_kv_timeline_impact(df, metrics, out_dir / "05_kv_transfer_impact.png")
    chart_sync_skew(df, metrics, roles, out_dir / "06_sync_skew.png")

    # ---- interactive timeline ----
    print("Building interactive timeline ...")
    html = build_timeline_html(df, metrics, roles, title)
    (out_dir / "timeline.html").write_text(html)

    # ---- summary ----
    summary = {
        "title": title,
        "n_events": int(len(df)),
        "n_sys": int(df["sys_id"].nunique()),
        "roles": {str(k): v for k, v in roles.items()},
        "ttft_ns": metrics["ttft_ns"],
        "ttft_human": fmt_time(metrics["ttft_ns"]),
        "prefill_all_ready_ns": metrics.get("prefill_all_ready_ns"),
        "prefill_all_ready_human": fmt_time(metrics.get("prefill_all_ready_ns")),
        "decode_start_ns": metrics["decode_start_ns"],
        "decode_start_human": fmt_time(metrics["decode_start_ns"]),
        "kv_exposure_ns": metrics["kv_exposure_ns"],
        "kv_exposure_human": fmt_time(metrics["kv_exposure_ns"]),
        "kv_transfer_exposed_ns": metrics["kv_transfer_exposed_ns"],
        "kv_transfer_exposed_human": fmt_time(metrics["kv_transfer_exposed_ns"]),
        "kv_onwire_ns": metrics["kv_onwire_ns"],
        "kv_onwire_human": fmt_time(metrics["kv_onwire_ns"]),
        "kv_handoff_exposed_ns": metrics["kv_handoff_exposed_ns"],
        "kv_handoff_exposed_human": fmt_time(metrics["kv_handoff_exposed_ns"]),
        "kv_exposed_per_sys": metrics["kv_exposed_per_sys"],
        "second_token_ns": metrics["second_token_ns"],
        "second_token_human": fmt_time(metrics["second_token_ns"]),
        "tpot_ns": metrics["tpot_ns"],
        "tpot_human": fmt_time(metrics["tpot_ns"]),
        "tpot_steady_ns": metrics["tpot_steady_ns"],
        "tpot_steady_human": fmt_time(metrics["tpot_steady_ns"]),
        "sync_skew_ns": metrics["sync_skew_ns"],
        "sync_skew_human": fmt_time(metrics["sync_skew_ns"]),
        "sync_skew_head_ns": metrics["sync_skew_head_ns"],
        "sync_skew_head_human": fmt_time(metrics["sync_skew_head_ns"]),
        "sync_skew_per_stage_ns": metrics.get("sync_skew_per_stage_ns"),
        "makespan_ns": metrics["makespan_ns"],
        "makespan_human": fmt_time(metrics["makespan_ns"]),
        "n_decode_iters": metrics["n_decode_iters"],
        "kv_tail_vs_decode_start_ns": metrics["kv_tail_vs_decode_start_ns"],
        "comm_time_per_class": aggregate_comm_time(df),
        "per_sys_utilisation": util,
        "kv_ready_per_npu": metrics["kv_ready_per_npu"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Run: {title}\n")
        f.write(f"Events: {len(df):,}   sys instances: {df['sys_id'].nunique()}\n\n")
        f.write(f"TTFT (first token produced)   : {fmt_time(metrics['ttft_ns'])}\n")
        if metrics.get("prefill_all_ready_ns") is not None:
            f.write(f"  (all TP shards materialised) : {fmt_time(metrics['prefill_all_ready_ns'])}\n")
        f.write(f"Decode start (KV-gated)       : {fmt_time(metrics['decode_start_ns'])}\n")
        f.write(f"KV-cache transfer exposed     : {fmt_time(metrics['kv_transfer_exposed_ns'])}"
                f"  (of {fmt_time(metrics['kv_onwire_ns'])} on-wire)\n")
        f.write(f"First-token handoff exposed   : {fmt_time(metrics['kv_handoff_exposed_ns'])}\n")
        f.write(f"2nd token (prefill+KV+decode1): {fmt_time(metrics['second_token_ns'])}\n")
        f.write(f"TPOT (per output token)       : {fmt_time(metrics['tpot_ns'])}\n")
        f.write(f"TPOT steady-state             : {fmt_time(metrics['tpot_steady_ns'])}\n")
        f.write(f"Sync skew (worst per-stage)   : {fmt_time(metrics['sync_skew_ns'])}\n")
        f.write(f"Sync skew (head stage)        : {fmt_time(metrics['sync_skew_head_ns'])}\n")
        f.write(f"Total makespan                : {fmt_time(metrics['makespan_ns'])}\n")
        f.write(f"Decode iterations             : {metrics['n_decode_iters']}\n")
        if metrics["kv_tail_vs_decode_start_ns"] is not None:
            f.write(f"Last KV chunk vs decode start : {fmt_time(metrics['kv_tail_vs_decode_start_ns'])}\n")

    print(f"\nDone. Outputs in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print("  -", p.name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=None,
                    help="path to a run dir with stats_sys*.csv (absolute, or "
                         "relative to output/astra_logs). Alternative to --sweep/--tag.")
    ap.add_argument("--sweep", default=None,
                    help="sweep dir name (package convention); with --tag resolves "
                         "the run via utils.paths")
    ap.add_argument("--tag", default=None, help="run tag under the sweep")
    ap.add_argument("--workload", default=paths.WORKLOAD,
                    help=f"workload dir under output/astra_logs (default: {paths.WORKLOAD})")
    ap.add_argument("--root", default=str(paths.ROOT), type=Path,
                    help=f"project root (default: {paths.ROOT})")
    ap.add_argument("-o", "--out", default=None, type=Path,
                    help="output dir (default: results/astra_analysis/<...>)")
    ap.add_argument("--title", default=None, help="label shown in the timeline header")
    ap.add_argument("--pattern", default="*.csv",
                    help="glob for the per-rank CSVs (default: *.csv)")
    args = ap.parse_args(argv)

    try:
        root = Path(args.root)
        if args.input:
            p = Path(args.input)
            run_dir = p if (p.is_absolute() or p.exists()) else (root / "output" / "astra_logs" / args.input)
            default_out = root / "results" / "astra_analysis" / args.input
        elif args.sweep and args.tag:
            sp = paths.SweepPaths(args.sweep, args.workload, root)
            run_dir = sp.astra_run(args.tag)
            default_out = root / "results" / "astra_analysis" / args.workload / args.sweep / args.tag
        else:
            need(False, "give an input run dir (positional), or both --sweep and --tag")

        need(run_dir.is_dir(), f"run dir not found: {run_dir}")
        out_dir = args.out or default_out
        title = args.title or run_dir.name or "run"
        analyse(run_dir, out_dir, title, args.pattern)
        return 0
    except Abort as e:
        print(f"\nABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
