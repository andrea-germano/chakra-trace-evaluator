#!/usr/bin/env python3
"""
analyze_sim.py  --  Disaggregated prefill/decode LLM-serving timeline & network analyzer.

Reads every *.csv produced by the simulator inside an input folder (one CSV per
"sys" instance = one GPU/TP-shard), reconstructs the combined execution timeline of
the whole distributed system for a single request, and produces:

  1. timeline.html      -> interactive, canvas-based (raster) Gantt chart of ALL
                           sys instances in one view. Pan / zoom / hover-for-name.
                           TTFT, decode-start and 2nd-token latency drawn as marker
                           lines + big numbers in the header.
  2. *.png              -> static charts characterising the impact of the different
                           communication types (KV-cache transfer, TP, PP, ...) on
                           the serving metrics (TTFT / TPOT), plus network-bandwidth
                           / congestion / synchronisation-skew views.
  3. summary.json/.txt  -> the headline numbers.

Node types are characterised using the naming convention in `naming.py`
(re-implemented here so the script is self-contained and does not need to import it).

Usage:
    python analyze_sim.py <input_folder> <output_folder>

------------------------------------------------------------------------------
Changes vs the previous revision (see CHANGELOG block):
  * TTFT is now the PREFILL-side first-token instant (network-insensitive by
    construction). The instant decode actually begins -- gated by the KV-cache
    transfer and the first-token handoff -- is reported separately as
    `decode_start`, and the gap `kv_exposure = decode_start - ttft` is surfaced
    as a first-class metric. The latency waterfall labels that gap as KV-transfer
    exposure rather than "sampling".
  * New `sync_skew` metric and chart: the spread of per-rank KV-ready instants on
    the decode side, which the first decode all-reduce must absorb.
  * Communication rows are tagged SEND vs RECV by pool role, so the effective-
    bandwidth / congestion view uses the transmit side and the simulator-reported
    bandwidth when available, instead of size/duration on blocked receives.
  * TPOT is reported both as a full mean and as a steady-state mean that excludes
    the first decode gap (the one carrying the first-all-reduce warm-up artefact).
------------------------------------------------------------------------------
"""

import sys
import os
import re
import json
import glob
import math
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

# Root from which model/topology sub-paths are resolved when --input is omitted.
DATA_DIR = "./output/astra_logs"
# Root under which per-run result folders (charts, timeline, summary) are written.
RESULT_DIR = "./result_graphs"

# Communications whose payload is <= this many bytes are treated as control /
# signalling traffic (the "serialize" / 8-byte FIRSTTOK messages).  They are
# excluded from the bandwidth / impact analysis and from the timeline clutter,
# but FIRSTTOK is still used as a marker for the first-token instant.
CONTROL_MAX_BYTES = 1024

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

# Accent used for the network-exposed (KV transfer + handoff) part of the path.
EXPOSED_COLOR = "#ff9f43"

CSV_COLUMNS = [
    "sys_id", "node_id", "name", "type", "comm_size", "start_tick", "end_tick",
    "duration", "bw_bytes_per_ns", "operation_intensity", "compute_utilization",
    "memory_utilization", "is_memory_bound",
]


# --------------------------------------------------------------------------- #
#  Name parsing  (mirrors naming.py)
# --------------------------------------------------------------------------- #

# Field keys used in the names, in their canonical order.
_FIELD_KEYS = ("pl", "ss", "ds", "sh", "ssh", "dsh", "L", "seg", "op", "it")
_FIELD_RE = re.compile(r"([a-zA-Z]+)=([^_]+)")


def parse_name(name: str) -> dict:
    """Split a node name into its class (prefix) and its key=value fields.

    e.g. 'KV_ss=0_ds=0_ssh=0_dsh=1_L=22_it=0'
         -> {'cls': 'KV', 'ss': '0', 'ds': '0', 'ssh': '0', 'dsh': '1',
             'L': '22', 'it': '0'}
    """
    if not isinstance(name, str) or "_" not in name:
        return {"cls": name if isinstance(name, str) else "?"}
    cls = name.split("_", 1)[0]
    fields = {"cls": cls}
    for k, v in _FIELD_RE.findall(name):
        if k in _FIELD_KEYS:
            fields[k] = v
    return fields


def categorise(row) -> str:
    """Return the analysis category for a row, based on type + name class."""
    if row["type"] == "GPU":
        op = row.get("op")
        if op in ("attn", "ffw"):
            return f"COMPUTE_{op}"
        return "COMPUTE"
    cls = row.get("cls", "OTHER")
    # tiny control / signalling traffic
    if cls == "FIRSTTOK":
        return "FIRSTTOK"
    if cls in ("TP", "KV", "KVREQ", "PP", "DECFB"):
        return cls
    return "OTHER"


def role_of_pool(pl: str) -> str:
    return {"p": "prefill", "d": "decode"}.get(pl, "?")


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #

def load_folder(folder: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files found in {folder!r}")

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:                                  # pragma: no cover
            print(f"  ! skipping {os.path.basename(p)}: {e}")
            continue
        # tolerate header drift: keep only known columns that exist
        for c in CSV_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df["__file"] = os.path.basename(p)
        frames.append(df)

    if not frames:
        raise SystemExit("No readable CSV files.")

    df = pd.concat(frames, ignore_index=True)

    # numeric coercion
    for c in ("sys_id", "node_id", "comm_size", "start_tick", "end_tick",
              "duration", "bw_bytes_per_ns"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows without a usable time interval
    df = df.dropna(subset=["start_tick", "end_tick"]).copy()
    df["start_tick"] = df["start_tick"].astype("int64")
    df["end_tick"] = df["end_tick"].astype("int64")
    df["duration"] = (df["end_tick"] - df["start_tick"]).clip(lower=0)

    # parse names -> fields
    parsed = df["name"].apply(parse_name)
    df["cls"] = parsed.apply(lambda d: d.get("cls"))
    for k in _FIELD_KEYS:
        df[k] = parsed.apply(lambda d: d.get(k))

    df["category"] = df.apply(categorise, axis=1)
    df["is_compute"] = df["type"] == "GPU"
    df["is_comm"] = df["type"] == "COMM"
    df["comm_size"] = df["comm_size"].fillna(0)
    # control / signalling = tiny comm payloads
    df["is_control"] = df["is_comm"] & (df["comm_size"] <= CONTROL_MAX_BYTES)
    return df


# --------------------------------------------------------------------------- #
#  SEND / RECV disambiguation
# --------------------------------------------------------------------------- #

def tag_comm_role(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """Tag each communication row as 'send' or 'recv'.

    A SEND and its matching RECV carry the *same* node name (so that ASTRA-sim
    can pair them by tag); they therefore appear as two rows on two different
    sys instances.  We recover which side a row is by comparing the owning sys's
    pool/stage against the source coordinates encoded in the name:

      * KV, FIRSTTOK : source is the prefill pool        -> prefill side = send
      * KVREQ        : the pull request goes decode->prefill -> decode side = send
      * PP, DECFB    : source is the stage whose id == name's `ss`

    The distinction matters because a pre-posted RECV sits blocked until its
    data is produced upstream, so its duration (and any size/duration bandwidth)
    is a scheduling artefact, not a link property.  The SEND side reflects the
    real time on the wire."""
    if df.empty:
        df["comm_role"] = ""
        return df

    def role_of(r) -> str:
        if not r.is_comm:
            return ""
        cls = r.cls
        sys_role = roles.get(int(r.sys_id), {}).get("role")
        if cls in ("KV", "FIRSTTOK"):
            return "send" if sys_role == "prefill" else "recv"
        if cls == "KVREQ":
            return "send" if sys_role == "decode" else "recv"
        if cls in ("PP", "DECFB"):
            sys_stage = roles.get(int(r.sys_id), {}).get("ss")
            if r.ss is not None and sys_stage is not None and str(sys_stage) == str(r.ss):
                return "send"
            return "recv"
        return ""

    df["comm_role"] = [role_of(r) for r in df.itertuples(index=False)]
    return df


# --------------------------------------------------------------------------- #
#  Wait-dominated flag  (unreliable reported bandwidth)
# --------------------------------------------------------------------------- #

def flag_wait_dominated(df: pd.DataFrame) -> pd.DataFrame:
    """Flag transfers whose reported duration (and therefore effective
    bandwidth = size / duration) is an artefact of *scheduling* rather than of
    the link speed.

    In this simulator the receiving side pre-posts a recv at the very start of
    the run; it then sits blocked until the matching data is actually produced
    upstream, so its duration = (long wait) + (real transfer) and its reported
    bandwidth collapses to a tiny, meaningless number.  These pre-posted recvs
    are exactly the ones whose ``start_tick`` equals the global simulation
    origin.  That is a clean, payload-size-independent and topology-independent
    signal -- unlike a bandwidth threshold, which would wrongly punish
    legitimately slower transfers (e.g. a small decode all-reduce vs a huge
    prefill one).

    A transfer is therefore wait-dominated when it is pre-posted at the origin
    AND it is not, in fact, a fast transfer that genuinely ran at t0; OR it is a
    RECV (whose duration spans an upstream wait) for which a paired SEND exists.
    """
    df["eff_bw"] = np.where(df["duration"] > 0,
                            df["comm_size"] / df["duration"], np.nan)
    df["wait_dominated"] = False
    if df.empty:
        return df

    origin = int(df["start_tick"].min())

    # peak (uncongested) bandwidth per class, from transfers that were NOT
    # pre-posted at the origin -- used only to spare a hypothetical fast t0 send.
    comm = df[df["is_comm"] & ~df["is_control"]]
    peak = {}
    for cls, grp in comm.groupby("cls"):
        ref = grp[grp["start_tick"] > origin]["eff_bw"].dropna()
        peak[cls] = ref.max() if len(ref) else np.nan

    has_role = "comm_role" in df.columns

    def is_wait(r):
        if not r.is_comm or r.is_control:
            return False
        # a RECV's duration spans the upstream wait -> never a reliable bw sample
        if has_role and getattr(r, "comm_role", "") == "recv":
            return True
        if r.start_tick != origin:
            return False
        p = peak.get(r.cls, np.nan)
        # genuinely fast transfer that ran at the origin -> keep as reliable
        if np.isfinite(p) and np.isfinite(r.eff_bw) and r.eff_bw >= 0.5 * p:
            return False
        return True

    df["wait_dominated"] = df.apply(is_wait, axis=1)

    # bandwidth actually used downstream: prefer the simulator-reported achieved
    # bandwidth when present and positive, otherwise fall back to size/duration.
    rep = pd.to_numeric(df.get("bw_bytes_per_ns"), errors="coerce")
    df["bw_used"] = np.where(np.isfinite(rep) & (rep > 0), rep, df["eff_bw"])
    return df


# --------------------------------------------------------------------------- #
#  Per-sys role + metrics
# --------------------------------------------------------------------------- #

def sys_roles(df: pd.DataFrame) -> dict:
    """Map sys_id -> dict(role, pl, ss, sh)."""
    roles = {}
    for sid, grp in df.groupby("sys_id"):
        comp = grp[grp["is_compute"]]
        pls = comp["pl"].dropna()
        pl = pls.mode().iloc[0] if len(pls) else None
        ss = comp["ss"].dropna()
        sh = comp["sh"].dropna()
        roles[int(sid)] = {
            "role": role_of_pool(pl),
            "pl": pl,
            "ss": ss.mode().iloc[0] if len(ss) else None,
            "sh": sh.mode().iloc[0] if len(sh) else None,
        }
    return roles


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

    # --- prefill end = first token produced by prefill (== TTFT) --------------
    # Max end over the prefill forward set = end of the final TP all-reduce on
    # the last pipeline stage (the latest forward op), i.e. the instant the first
    # token is actually ready.
    m["prefill_compute_end"] = int(prefill_fwd["end_tick"].max()) if len(prefill_fwd) else None

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

    # --- TTFT = first token = prefill end (network-insensitive) --------------
    firsttok = df[df["cls"] == "FIRSTTOK"]
    ttft_tick = m["prefill_compute_end"]
    if ttft_tick is None and len(firsttok):
        # FIRSTTOK send completes right after the prefill tail; .min() picks the
        # send side rather than a (possibly eager) recv.
        ttft_tick = int(firsttok["end_tick"].min())
    if ttft_tick is None:
        ttft_tick = decode_start_tick
    m["ttft_tick"] = ttft_tick
    m["ttft_ns"] = (ttft_tick - t0) if ttft_tick is not None else None

    # --- KV transfer + handoff exposed on the path to the 2nd token ----------
    if decode_start_tick is not None and ttft_tick is not None:
        m["kv_exposure_ns"] = decode_start_tick - ttft_tick
    else:
        m["kv_exposure_ns"] = None

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
    kv_ready = kv_ready_per_decode_npu(df, roles)
    m["kv_ready_per_npu"] = kv_ready
    if len(kv_ready) >= 2:
        readies = [r["kv_ready_tick"] for r in kv_ready]
        m["sync_skew_ns"] = max(readies) - min(readies)
    else:
        m["sync_skew_ns"] = None
    # skew within the head decode stage (the group that gates the FIRST decode
    # all-reduce); falls back to global skew if stage info is unavailable.
    head = [r for r in kv_ready if str(r["stage"]) == "0"]
    if len(head) >= 2:
        hr = [r["kv_ready_tick"] for r in head]
        m["sync_skew_head_ns"] = max(hr) - min(hr)
    else:
        m["sync_skew_head_ns"] = m["sync_skew_ns"]

    # --- KV-cache "tail": how late the last KV chunk completes vs the instant
    #     decode actually needs it (= decode start). Positive => KV exposed. ---
    kv = df[(df["cls"] == "KV") & ~df["is_control"]]
    ref_tick = decode_start_tick if decode_start_tick is not None else ttft_tick
    if len(kv) and ref_tick is not None:
        m["kv_last_end"] = int(kv["end_tick"].max())
        m["kv_tail_vs_decode_start_ns"] = int(kv["end_tick"].max()) - ref_tick
    else:
        m["kv_last_end"] = None
        m["kv_tail_vs_decode_start_ns"] = None

    return m


# --------------------------------------------------------------------------- #
#  Exposed / overlapped communication per sys
# --------------------------------------------------------------------------- #

def interval_union_len(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def interval_overlap_len(a, b):
    """Length of overlap between two unions of intervals a and b."""
    if not a or not b:
        return 0
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


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
        comp_busy = interval_union_len(comp)
        comm_union = interval_union_len(comm)
        comm_hidden = interval_overlap_len(comp, comm)
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
        return f"{ns/1e3:.3f} \u00b5s"
    return f"{ns:.0f} ns"


def fmt_bw(bw_bytes_per_ns):
    """bytes/ns == GB/s. Return GB/s string."""
    if bw_bytes_per_ns is None or not np.isfinite(bw_bytes_per_ns):
        return "n/a"
    return f"{bw_bytes_per_ns:.2f} GB/s"


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
    kv_exp = metrics.get("kv_exposure_ns")

    # --- left: TTFT / decode-start / 2nd-token composition ----------------- #
    segs = []  # (label, start_ns, len_ns, color)
    if ttft is not None:
        segs.append(("Prefill compute\n(\u2192 first token)", 0, ttft, "#5b8def"))
    if ttft is not None and dstart is not None and dstart > ttft:
        segs.append(("KV transfer +\nhandoff (exposed)", ttft, dstart - ttft, EXPOSED_COLOR))
    base_decode = dstart if (dstart is not None) else ttft
    if base_decode is not None and second is not None and second > base_decode:
        segs.append(("Decode step 1\n(\u2192 2nd token)", base_decode,
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
    if dstart is not None and (ttft is None or dstart > ttft):
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
    sub = ""
    if kv_exp is not None:
        sub = f"\nKV transfer exposed on 2nd-token path: {fmt_time(kv_exp)}"
    axL.set_title("Critical path to first & second token" + sub, fontsize=11)
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
    """Aggregate on-wire time and bytes per communication class, WITHOUT the
    double counting that a naive per-row sum introduces:

      * point-to-point transfers (KV / PP / DECFB / KVREQ) are counted once, on
        their SEND side. The matching RECV sits blocked until the data is
        produced upstream, so its duration spans that wait and is not time on the
        wire; summing it in is exactly what inflates the KV total.
      * the TP all-reduce is one logical collective spread over the tp
        participating ranks (one node per rank, identical apart from the shard
        id `sh`). It is collapsed to a single representative per collective --
        keyed by (pl, ss, L, it, op), with the slowest rank giving the wall-clock
        duration -- instead of being summed tp times.

    Returns a list of dicts {cls, total_dur, total_bytes, count} sorted by
    total_dur ascending."""
    comm = df[df["is_comm"] & ~df["is_control"]].copy()
    rows = []
    if comm.empty:
        return rows
    has_role = "comm_role" in comm.columns

    # --- point-to-point: SEND side only ---
    for cls in ("KV", "PP", "DECFB", "KVREQ"):
        sub = comm[comm["cls"] == cls]
        if sub.empty:
            continue
        if has_role and (sub["comm_role"] == "send").any():
            sub = sub[sub["comm_role"] == "send"]
        else:
            # no role info for this class -> fall back to the non-blocked rows
            sub = sub[~sub["wait_dominated"]]
        if len(sub):
            rows.append({
                "cls": cls,
                "total_dur": float(sub["duration"].sum()),
                "total_bytes": float(sub["comm_size"].sum()),
                "count": int(len(sub)),
            })

    # --- TP collective: one representative per logical all-reduce ---
    tp = comm[comm["cls"] == "TP"]
    if len(tp):
        keys = [c for c in ("pl", "ss", "L", "it", "op") if c in tp.columns]
        ded = tp.groupby(keys, dropna=False).agg(
            d=("duration", "max"), s=("comm_size", "first")).reset_index()
        rows.append({
            "cls": "TP",
            "total_dur": float(ded["d"].sum()),
            "total_bytes": float(ded["s"].sum()),
            "count": int(len(ded)),
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
    ax.set_title("Per-transfer achieved bandwidth (send side)  \u2014  congestion indicator")
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
    ax1.set_title("GPU occupancy per sys  \u2014  idle = stalled on peers / network")
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

    # shade the head-stage skew band
    head = [r for r in ready_sorted if str(r["stage"]) == "0"]
    if len(head) >= 2:
        hx = [(r["kv_ready_tick"] - t0) / 1e6 for r in head]
        ax.axvspan(min(hx), max(hx), color="#ff6b6b", alpha=0.10, zorder=0)
        ax.axvline(min(hx), color="#ff6b6b", lw=1, ls=":", alpha=0.7)
        ax.axvline(max(hx), color="#ff6b6b", lw=1.4, ls="--",
                   label=f"head-stage skew = {fmt_time(metrics.get('sync_skew_head_ns'))}")

    dstart = metrics.get("decode_start_ns")
    if dstart is not None:
        ax.axvline(dstart, color="#d97706", lw=2, ls="--",
                   label=f"decode start ({fmt_time(dstart)})")

    ax.set_yticks(y)
    ax.set_yticklabels([f"sys{r['sys_id']} (stage {r['stage']}, shard {r['shard']})"
                        for r in ready_sorted], fontsize=8)
    ax.set_xlabel("KV-ready instant since request start (ms)")
    skew = metrics.get("sync_skew_ns")
    ax.set_title("Per-rank KV-ready skew on the decode side\n"
                 f"global skew = {fmt_time(skew)}  "
                 f"(absorbed by the first decode all-reduce)", fontsize=11)

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

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Execution timeline &mdash; __TITLE__</title>
<style>
  :root{
    --bg:#0e1320; --panel:#161d2e; --panel2:#1d2740; --ink:#e7ecf3;
    --muted:#8a96ab; --line:#2a3550; --accent:#4f9bff;
    --ttft:#ff6b6b; --second:#22c08a; --dstart:#ff9f43;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);
       font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
       font-size:13px;overflow:hidden}
  header{padding:14px 18px;border-bottom:1px solid var(--line);
         display:flex;align-items:center;gap:26px;flex-wrap:wrap;background:var(--panel)}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
  header h1 .sub{color:var(--muted);font-weight:400;margin-left:8px}
  .metrics{display:flex;gap:22px;margin-left:auto;flex-wrap:wrap}
  .metric{display:flex;flex-direction:column;line-height:1.15}
  .metric .v{font-size:20px;font-weight:700}
  .metric .k{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
  .metric.ttft .v{color:var(--ttft)}
  .metric.kvexp .v{color:var(--dstart)}
  .metric.second .v{color:var(--second)}
  .metric.tpot .v{color:#7c5cff}
  #toolbar{display:flex;gap:10px;align-items:center;padding:8px 18px;
           border-bottom:1px solid var(--line);background:var(--panel2);flex-wrap:wrap}
  #toolbar button{background:#26314d;color:var(--ink);border:1px solid var(--line);
           border-radius:6px;padding:5px 11px;cursor:pointer;font:inherit;font-size:12px}
  #toolbar button:hover{background:#30406a}
  #toolbar .hint{color:var(--muted);font-size:11.5px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-left:auto}
  .legend span{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;color:var(--muted)}
  .legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
  #wrap{position:absolute;inset:0;top:var(--top);}
  #cv{display:block;width:100%;height:100%;cursor:grab}
  #cv.drag{cursor:grabbing}
  #tip{position:fixed;pointer-events:none;z-index:10;display:none;max-width:460px;
       background:#0a0f1ae6;border:1px solid var(--line);border-radius:8px;
       padding:9px 11px;font-size:12px;line-height:1.5;backdrop-filter:blur(3px);
       box-shadow:0 8px 30px #0009}
  #tip b{color:var(--accent)}
  #tip .name{color:#fff;word-break:break-all;font-size:12.5px;margin-bottom:4px}
  #tip .row{color:var(--muted)}
  #tip .row span{color:var(--ink)}
  #tip .warn{color:var(--ttft);margin-top:3px}
</style>
</head>
<body>
<header>
  <h1>Execution timeline<span class="sub">__TITLE__</span></h1>
  <div class="metrics">
    <div class="metric ttft"><span class="v">__TTFT__</span><span class="k">TTFT (first token, prefill)</span></div>
    <div class="metric kvexp"><span class="v">__KVEXP__</span><span class="k">KV transfer exposed</span></div>
    <div class="metric second"><span class="v">__SECOND__</span><span class="k">2nd token (prefill+KV+decode 1)</span></div>
    <div class="metric tpot"><span class="v">__TPOT__</span><span class="k">TPOT (per token)</span></div>
    <div class="metric"><span class="v">__MAKESPAN__</span><span class="k">total makespan</span></div>
  </div>
</header>
<div id="toolbar">
  <button id="reset">Reset view</button>
  <button id="zin">Zoom +</button>
  <button id="zout">Zoom &minus;</button>
  <button id="fit">Fit width</button>
  <span class="hint">drag = pan &nbsp;&middot;&nbsp; wheel = zoom (x) &nbsp;&middot;&nbsp; shift+wheel = scroll &nbsp;&middot;&nbsp; hover = node name</span>
  <div class="legend" id="legend"></div>
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const root = document.documentElement;
const headerH = document.querySelector('header').offsetHeight + document.getElementById('toolbar').offsetHeight;
root.style.setProperty('--top', headerH + 'px');

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');

const COLORS = DATA.colors;          // catName -> hex
const CATS   = DATA.cats;            // index -> catName
const LANES  = DATA.lanes;           // [{label, role}]
const BARS   = DATA.bars;            // {l,a,b,c,n,s,w,d}
const t0 = DATA.t0, tEnd = DATA.tEnd;
const ttftX = DATA.ttft, secondX = DATA.second; // absolute ticks or null
const decodeStartX = DATA.decodeStart;          // absolute tick or null

// layout constants
const LANE_H = 16, LANE_PAD = 2, LEFT = 168, TOP = 26, ROW = LANE_H + LANE_PAD;
const contentH = LANES.length * ROW + TOP + 20;

// view transform (x only zoom/pan; y = vertical scroll)
let scaleX, offX, offY = 0;
let dpr = Math.max(1, window.devicePixelRatio || 1);

function resize(){
  dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.floor(w*dpr); cv.height = Math.floor(h*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function fitWidth(){
  const w = cv.clientWidth;
  const span = (tEnd - t0) || 1;
  scaleX = (w - LEFT - 20) / span;
  offX = LEFT;
  offY = 0;
  draw();
}
function tickToX(t){ return offX + (t - t0)*scaleX; }
function xToTick(x){ return t0 + (x - offX)/scaleX; }

// build per-lane bar buckets for faster hit-testing
const laneBars = LANES.map(()=>[]);
for(const bar of BARS){ laneBars[bar.l].push(bar); }

function colorFor(b){ return COLORS[CATS[b.c]] || '#888'; }

function fmtTime(ns){
  if(ns==null||!isFinite(ns)) return 'n/a';
  const a=Math.abs(ns);
  if(a>=1e9) return (ns/1e9).toFixed(3)+' s';
  if(a>=1e6) return (ns/1e6).toFixed(3)+' ms';
  if(a>=1e3) return (ns/1e3).toFixed(3)+' \u00b5s';
  return ns.toFixed(0)+' ns';
}

function draw(){
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.clearRect(0,0,w,h);

  // --- time grid ---
  ctx.save();
  ctx.beginPath(); ctx.rect(LEFT,0,w-LEFT,h); ctx.clip();
  const leftTick = xToTick(LEFT), rightTick = xToTick(w);
  const span = rightTick-leftTick;
  let step = Math.pow(10, Math.floor(Math.log10(span/6)));
  [1,2,5,10].some(m=>{ if(span/(step*m) <= 8){ step*=m; return true;} return false;});
  const first = Math.ceil(leftTick/step)*step;
  ctx.font='10px ui-monospace,monospace';
  for(let t=first;t<rightTick;t+=step){
    const x=tickToX(t);
    ctx.strokeStyle='#1c2538'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x,TOP-6); ctx.lineTo(x,h); ctx.stroke();
    ctx.fillStyle='#6b7689';
    ctx.fillText(fmtTime(t-t0), x+3, TOP-8);
  }
  ctx.restore();

  // --- lane backgrounds + labels ---
  ctx.font='11px ui-monospace,monospace';
  for(let i=0;i<LANES.length;i++){
    const y = TOP + i*ROW - offY;
    if(y+ROW<0||y>h) continue;
    ctx.fillStyle = (i%2)? '#121829':'#0f1422';
    ctx.fillRect(0,y,w,ROW);
    const ln = LANES[i];
    ctx.fillStyle = ln.role==='decode' ? '#9fb4d8' : (ln.role==='prefill'?'#cdb88c':'#8a96ab');
    ctx.fillText(ln.label, 8, y+LANE_H-3);
  }
  // separator under labels
  ctx.strokeStyle='#2a3550'; ctx.beginPath(); ctx.moveTo(LEFT,0); ctx.lineTo(LEFT,h); ctx.stroke();

  // --- bars ---
  ctx.save();
  ctx.beginPath(); ctx.rect(LEFT,TOP-6,w-LEFT,h); ctx.clip();
  for(let i=0;i<LANES.length;i++){
    const y = TOP + i*ROW - offY;
    if(y+ROW<0||y>h) continue;
    for(const b of laneBars[i]){
      let x0 = tickToX(b.a), x1 = tickToX(b.b);
      if(x1<LEFT||x0>w) continue;
      let bw = Math.max(1, x1-x0);
      const col = colorFor(b);
      if(b.w){ // wait-dominated -> hollow/light to show it's mostly blocked
        ctx.fillStyle = col+'33';
        ctx.fillRect(x0, y+1, bw, LANE_H-2);
        ctx.strokeStyle = col; ctx.lineWidth=1;
        ctx.strokeRect(x0+0.5, y+1.5, Math.max(1,bw-1), LANE_H-3);
      }else{
        ctx.fillStyle = col;
        ctx.fillRect(x0, y+1, bw, LANE_H-2);
      }
    }
  }
  // --- markers ---
  function marker(tick,color,label,tier){
    if(tick==null) return;
    const x=tickToX(tick); if(x<LEFT||x>w) return;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(x,TOP-6); ctx.lineTo(x,h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=color; ctx.font='bold 11px ui-monospace,monospace';
    const ty = TOP+10 + (tier||0)*15;
    ctx.fillText(label, x+4, ty);
  }
  marker(ttftX,'#ff6b6b','TTFT '+fmtTime(ttftX!=null?ttftX-t0:null), 0);
  marker(decodeStartX,'#ff9f43','decode start '+fmtTime(decodeStartX!=null?decodeStartX-t0:null), 1);
  marker(secondX,'#22c08a','2nd '+fmtTime(secondX!=null?secondX-t0:null), 2);
  ctx.restore();
}

// --- hit testing for tooltip ---
function hit(px,py){
  if(px<LEFT) return null;
  const i = Math.floor((py + offY - TOP)/ROW);
  if(i<0||i>=LANES.length) return null;
  const y = TOP + i*ROW - offY;
  if(py<y+1||py>y+LANE_H-1) return null;
  const t = xToTick(px);
  // search bars in lane; prefer the narrowest containing bar (so thin bars win)
  let best=null, bestW=Infinity;
  for(const b of laneBars[i]){
    if(t>=b.a && t<=b.b){
      // also require pixel proximity for zero-width
      const w=b.b-b.a;
      if(w<bestW){best=b;bestW=w;}
    } else {
      // allow clicking a sub-pixel bar within 4px
      const x0=tickToX(b.a),x1=tickToX(b.b);
      if(px>=x0-3 && px<=x1+3){ if((b.b-b.a)<bestW){best=b;bestW=b.b-b.a;} }
    }
  }
  return best ? {bar:best, lane:LANES[i]} : null;
}

function showTip(h, px, py){
  const b=h.bar, ln=h.lane;
  let html = '<div class="name">'+b.n+'</div>';
  html += '<div class="row">lane: <span>'+ln.label+'</span> &middot; cat: <span style="color:'+colorFor(b)+'">'+CATS[b.c]+'</span></div>';
  html += '<div class="row">start: <span>'+fmtTime(b.a-t0)+'</span> &rarr; end: <span>'+fmtTime(b.b-t0)+'</span></div>';
  html += '<div class="row">duration: <span>'+fmtTime(b.b-b.a)+'</span></div>';
  if(b.s!=null && b.s>0){
    html += '<div class="row">payload: <span>'+(b.s/1e6).toFixed(2)+' MB</span> &middot; eff.bw: <span>'+(b.d!=null?b.d.toFixed(2)+' GB/s':'n/a')+'</span></div>';
  }
  if(b.w){ html += '<div class="warn">&#9888; wait-dominated &mdash; scheduled early, mostly blocked; reported bandwidth unreliable</div>'; }
  tip.innerHTML=html;
  tip.style.display='block';
  let tx=px+14, ty=py+14;
  const r=tip.getBoundingClientRect();
  if(tx+r.width>window.innerWidth) tx=px-r.width-14;
  if(ty+r.height>window.innerHeight) ty=py-r.height-14;
  tip.style.left=tx+'px'; tip.style.top=ty+'px';
}

// --- interaction ---
let dragging=false, lastX=0, lastY=0, moved=false;
cv.addEventListener('mousedown',e=>{dragging=true;moved=false;lastX=e.clientX;lastY=e.clientY;cv.classList.add('drag');});
window.addEventListener('mouseup',()=>{dragging=false;cv.classList.remove('drag');});
window.addEventListener('mousemove',e=>{
  const rect=cv.getBoundingClientRect();
  const px=e.clientX-rect.left, py=e.clientY-rect.top;
  if(dragging){
    moved=true;
    offX += e.clientX-lastX;
    offY -= e.clientY-lastY;
    offY = Math.max(0, Math.min(offY, Math.max(0, contentH - cv.clientHeight)));
    lastX=e.clientX; lastY=e.clientY;
    tip.style.display='none';
    draw();
    return;
  }
  const h=hit(px,py);
  if(h){ showTip(h,e.clientX,e.clientY); } else { tip.style.display='none'; }
});
cv.addEventListener('mouseleave',()=>tip.style.display='none');

cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const rect=cv.getBoundingClientRect();
  const px=e.clientX-rect.left, py=e.clientY-rect.top;
  if(e.shiftKey){
    offY += e.deltaY;
    offY = Math.max(0, Math.min(offY, Math.max(0, contentH - cv.clientHeight)));
  }else{
    const tAtCursor = xToTick(px);
    const factor = Math.exp(-e.deltaY*0.0015);
    scaleX *= factor;
    // keep cursor tick fixed
    offX = px - (tAtCursor - t0)*scaleX;
  }
  tip.style.display='none';
  draw();
},{passive:false});

document.getElementById('reset').onclick=fitWidth;
document.getElementById('fit').onclick=fitWidth;
document.getElementById('zin').onclick=()=>{ scaleX*=1.4; draw(); };
document.getElementById('zout').onclick=()=>{ scaleX/=1.4; draw(); };

// legend
const lg=document.getElementById('legend');
const order=['COMPUTE_attn','COMPUTE_ffw','TP','KV','PP','KVREQ','DECFB'];
const seen=new Set(CATS);
for(const c of order){ if(seen.has(c)){
  const s=document.createElement('span');
  s.innerHTML='<i style="background:'+(COLORS[c]||'#888')+'"></i>'+c.replace('COMPUTE_','compute:');
  lg.appendChild(s);
}}

window.addEventListener('resize',resize);
fitWidth();
resize();
</script>
</body>
</html>
"""


def build_timeline_html(df, metrics, roles, title):
    """Assemble lane layout + bar list and inject into the HTML template."""
    # Keep compute + non-control comm (drop tiny serialize/signal traffic).
    vis = df[df["is_compute"] | (df["is_comm"] & ~df["is_control"])].copy()

    # ---- lane layout: per sys -> [compute, TP, KV, PP, DECFB, OTHER] ----
    lane_order = ["compute", "TP", "KV", "PP", "KVREQ", "DECFB", "OTHER"]

    def lane_key_of(row):
        if row["is_compute"]:
            return "compute"
        return row["cls"] if row["cls"] in lane_order else "OTHER"

    vis["lane_key"] = vis.apply(lane_key_of, axis=1)

    lanes = []          # list of dicts {label, role}
    lane_index = {}     # (sys, lane_key) -> idx
    for sid in sorted(vis["sys_id"].unique()):
        sid = int(sid)
        role = roles.get(sid, {}).get("role", "?")
        present = vis[vis["sys_id"] == sid]["lane_key"].unique()
        for lk in lane_order:
            if lk in present:
                idx = len(lanes)
                lane_index[(sid, lk)] = idx
                pretty = "compute" if lk == "compute" else lk
                lanes.append({"label": f"sys{sid} \u00b7 {pretty}", "role": role})

    # ---- category index table ----
    cats = sorted(vis["category"].unique())
    cat_idx = {c: i for i, c in enumerate(cats)}

    # ---- bars ----
    bars = []
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
    }

    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__TITLE__", title)
            .replace("__TTFT__", fmt_time(metrics["ttft_ns"]))
            .replace("__KVEXP__", fmt_time(metrics.get("kv_exposure_ns")))
            .replace("__SECOND__", fmt_time(metrics["second_token_ns"]))
            .replace("__TPOT__", fmt_time(metrics["tpot_ns"]))
            .replace("__MAKESPAN__", fmt_time(metrics["makespan_ns"])))
    return html


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def _ask(prompt: str, required: bool = True, default: str | None = None) -> str:
    """Prompt the user on stdin. Loops until a non-empty answer when required.
    Aborts cleanly if stdin is closed (EOF) on a required prompt."""
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            ans = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            if default is not None:
                return default
            if not required:
                return ""
            raise SystemExit("\nNo input available (stdin closed). "
                             "Pass --model/--topology or --input on the command line.")
        if not ans and default is not None:
            return default
        if ans or not required:
            return ans
        print("  (a value is required)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input becomes a standard optional argument (to allow for the interactive prompt)
    ap.add_argument("input", nargs="?", default=None,
                    help="Full path of the input folder (e.g., model/topology) under DATA_DIR")

    ap.add_argument("--output", "-o", default=None,
                    help="Output folder under RESULT_DIR (default: same sub-path as input)")
    
    ap.add_argument("--title", default=None,
                    help="Label shown in the timeline header")
    
    args = ap.parse_args()

    # ---- Interactive fallback if input is not provided via CLI ------------- #
    if not args.input:
        # Prompt the user to manually enter the path.
        # Assuming your _ask function ensures the input cannot be empty.
        args.input = _ask("Enter the input folder path (e.g., model_name/topology_name)")

    # ---- Resolve input / output paths -------------------------------------- #
    # Input is now always args.input, so the path construction is straightforward
    in_dir = os.path.join(DATA_DIR, args.input)

    # If args.output is provided, use it. Otherwise, fallback to args.input
    out_subpath = args.output if args.output else args.input
    out_dir = os.path.join(RESULT_DIR, out_subpath)

    os.makedirs(out_dir, exist_ok=True)
    title = args.title or os.path.basename(os.path.normpath(in_dir)) or "run"

    print(f"Loading CSVs from {in_dir} ...")
    df = load_folder(in_dir)
    roles = sys_roles(df)
    df = tag_comm_role(df, roles)
    df = flag_wait_dominated(df)
    metrics = compute_metrics(df, roles)
    util = per_sys_utilisation(df, roles)

    print(f"  {len(df):,} timeline events across {df['sys_id'].nunique()} sys instances")
    print(f"  roles: " + ", ".join(
        f"sys{s}={r['role']}" for s, r in sorted(roles.items())))
    print(f"  TTFT={fmt_time(metrics['ttft_ns'])}  "
          f"decode-start={fmt_time(metrics['decode_start_ns'])}  "
          f"KV-exposed={fmt_time(metrics['kv_exposure_ns'])}")
    print(f"  2nd-token={fmt_time(metrics['second_token_ns'])}  "
          f"TPOT={fmt_time(metrics['tpot_ns'])}  "
          f"TPOT(steady)={fmt_time(metrics['tpot_steady_ns'])}  "
          f"sync-skew={fmt_time(metrics['sync_skew_ns'])}")

    # ---- charts ----
    _style()
    print("Rendering charts ...")
    chart_latency_breakdown(df, metrics, roles,
                            os.path.join(out_dir, "01_latency_breakdown.png"))
    chart_comm_breakdown(df, os.path.join(out_dir, "02_communication_mix.png"))
    chart_bandwidth(df, os.path.join(out_dir, "03_effective_bandwidth.png"))
    chart_utilisation(util, os.path.join(out_dir, "04_sys_utilisation.png"))
    chart_kv_timeline_impact(df, metrics,
                             os.path.join(out_dir, "05_kv_transfer_impact.png"))
    chart_sync_skew(df, metrics, roles,
                    os.path.join(out_dir, "06_sync_skew.png"))

    # ---- interactive timeline ----
    print("Building interactive timeline ...")
    html = build_timeline_html(df, metrics, roles, title)
    html_path = os.path.join(out_dir, "timeline.html")
    with open(html_path, "w") as f:
        f.write(html)

    # ---- summary ----
    summary = {
        "title": title,
        "n_events": int(len(df)),
        "n_sys": int(df["sys_id"].nunique()),
        "roles": {str(k): v for k, v in roles.items()},
        "ttft_ns": metrics["ttft_ns"],
        "ttft_human": fmt_time(metrics["ttft_ns"]),
        "decode_start_ns": metrics["decode_start_ns"],
        "decode_start_human": fmt_time(metrics["decode_start_ns"]),
        "kv_exposure_ns": metrics["kv_exposure_ns"],
        "kv_exposure_human": fmt_time(metrics["kv_exposure_ns"]),
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
        "makespan_ns": metrics["makespan_ns"],
        "makespan_human": fmt_time(metrics["makespan_ns"]),
        "n_decode_iters": metrics["n_decode_iters"],
        "kv_tail_vs_decode_start_ns": metrics["kv_tail_vs_decode_start_ns"],
        "comm_time_per_class": aggregate_comm_time(df),
        "per_sys_utilisation": util,
        "kv_ready_per_npu": metrics["kv_ready_per_npu"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Run: {title}\n")
        f.write(f"Events: {len(df):,}   sys instances: {df['sys_id'].nunique()}\n\n")
        f.write(f"TTFT (first token, prefill)   : {fmt_time(metrics['ttft_ns'])}\n")
        f.write(f"Decode start (KV-gated)       : {fmt_time(metrics['decode_start_ns'])}\n")
        f.write(f"KV transfer exposed           : {fmt_time(metrics['kv_exposure_ns'])}\n")
        f.write(f"2nd token (prefill+KV+decode1): {fmt_time(metrics['second_token_ns'])}\n")
        f.write(f"TPOT (per output token)       : {fmt_time(metrics['tpot_ns'])}\n")
        f.write(f"TPOT steady-state             : {fmt_time(metrics['tpot_steady_ns'])}\n")
        f.write(f"Sync skew (decode KV-ready)   : {fmt_time(metrics['sync_skew_ns'])}\n")
        f.write(f"Sync skew (head stage)        : {fmt_time(metrics['sync_skew_head_ns'])}\n")
        f.write(f"Total makespan                : {fmt_time(metrics['makespan_ns'])}\n")
        f.write(f"Decode iterations             : {metrics['n_decode_iters']}\n")
        if metrics["kv_tail_vs_decode_start_ns"] is not None:
            f.write(f"Last KV chunk vs decode start : {fmt_time(metrics['kv_tail_vs_decode_start_ns'])}\n")

    print(f"\nDone. Outputs in {out_dir}:")
    for p in sorted(os.listdir(out_dir)):
        print("  -", p)


if __name__ == "__main__":
    main()