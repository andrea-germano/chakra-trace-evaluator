#!/usr/bin/env python3
"""
utils.astra — readers for the ASTRA-sim per-operator CSVs (stats_sys*.csv).

Verified against a real stats_sys0.csv. The layout is::

    sys_id,node_id,name,type,comm_size,start_tick,end_tick,duration,
    bw_bytes_per_ns,operation_intensity,compute_utilization,memory_utilization,
    is_memory_bound

``type`` is **GPU or COMM** and carries no direction. The MLSynth naming scheme
(Utils/naming.py) is where the structure lives::

    COMP_pl=p_ss=0_sh=0_L=0_it=0_op=attn      GPU
    TP_pl=p_ss=0_sh=0_L=0_it=0_op=attn        COMM   collective
    KV_ss=0_ds=0_ssh=0_dsh=0_L=0_it=0         COMM   point-to-point, NO pl= field
    PP_pl=p_ss=0_ds=1_sh=0_it=0               COMM   point-to-point

    pl   pool: p = prefill, d = decode          L    layer
    ss   source stage      ds   dest stage      it   iteration
    sh   shard             ssh  source shard    op   attn | ffw
                           dsh  dest shard

Counting a transfer exactly once
--------------------------------------------------------------------------------
Two independent ways the naive count goes wrong, both confirmed on real data:

1. **Point-to-point transfers appear twice.** A SEND and its matching RECV carry
   the *same* node name -- that is how ASTRA-sim pairs them by tag -- so they show
   up as two rows in two different sys files. On the reference run, sys0 (a prefill
   rank) holds 20 KV rows of 80 MiB = 1.678 GB; across 4 prefill ranks that is the
   true 6.71 GB of KV, while the concatenated CSVs report 160 KV rows and 13.42 GB.
   Exactly 2x.

   The direction is NOT in the ``type`` column. It is recovered from the name plus
   the role of the sys that owns the row (``sys_roles`` -> ``tag_comm_role``):
   KV and FIRSTTOK flow prefill -> decode, KVREQ flows decode -> prefill, and for
   PP/DECFB the sender is the rank whose own stage equals the name's ``ss``.

2. **Collectives appear once per rank.** A TP all-reduce is one logical operation
   spread over the tp participating ranks: one row per rank, identical apart from
   ``sh``. Summing them multiplies the collective by tp. ``collapse_collectives``
   keeps one representative per (pl, ss, L, it, op), taking the slowest rank for
   the wall-clock duration.

Why the recv rows also poison *time*, not just bytes
--------------------------------------------------------------------------------
The receiving side pre-posts its recv at the very start of the run and then sits
blocked until the data is produced upstream, so a recv row's ``start_tick`` is the
simulation origin and its ``duration`` is (long wait + real transfer). Keeping
those rows therefore: collapses ``min(start_tick)`` to the origin, which stretches
any max(end) - min(start) window to the whole run; and drags any mean duration or
size/duration bandwidth toward a scheduling artefact rather than a link property.
``flag_wait_dominated`` marks them; being pre-posted at the origin is a clean
signal, independent of payload size and topology, unlike a bandwidth threshold
which would wrongly punish legitimately slow transfers.

The send side is the time on the wire. Use ``sends()`` / ``collapse_collectives()``
-- or ``unique_transfers()``, which picks the right one -- for anything that sums
bytes or durations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

NUMERIC_COLS = ["comm_size", "start_tick", "end_tick", "duration", "bw_bytes_per_ns",
                "operation_intensity", "compute_utilization", "memory_utilization",
                "is_memory_bound"]

# Fields of the MLSynth naming scheme, in canonical order.
FIELD_KEYS = ("pl", "ss", "ds", "sh", "ssh", "dsh", "L", "it", "op")

# Classes carried by one flow between two ranks, as opposed to collectives.
POINT_TO_POINT = ("KV", "KVREQ", "PP", "DECFB", "FIRSTTOK")
COLLECTIVE = ("TP",)

# Payloads at or below this are control / signalling traffic (the tiny FIRSTTOK
# markers), not bandwidth.
CONTROL_MAX_BYTES = 128


# --------------------------------------------------------------------------- #
# Name parsing
# --------------------------------------------------------------------------- #
def parse_name(name: str) -> dict:
    """'KV_ss=0_ds=0_ssh=0_dsh=0_L=0_it=0' -> {'cls': 'KV', 'ss': '0', ...}.
    Missing fields are simply absent; nothing is invented."""
    if not isinstance(name, str) or not name:
        return {"cls": "OTHER"}
    parts = name.split("_")
    out = {"cls": parts[0].upper()}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


def classify_op(name: str) -> tuple[str, str]:
    """(op_class, phase). KV carries no pl= field, so its phase is its own class
    rather than the pool of whichever rank owns the row."""
    f = parse_name(name)
    cls = f["cls"]
    op = cls if cls in {"COMP", "TP", "KV", "KVREQ", "PP", "FIRSTTOK", "DECFB"} else "OTHER"
    if op == "KV":
        return op, "kv_transfer"
    if op == "KVREQ":
        return op, "kv_request"
    if op == "FIRSTTOK":
        return op, "handoff"
    pl = f.get("pl")
    if pl == "p":
        return op, "prefill"
    if pl == "d" or op == "DECFB":
        return op, "decode"
    return op, "other"


# --------------------------------------------------------------------------- #
# Roles and direction
# --------------------------------------------------------------------------- #
def sys_roles(df: pd.DataFrame) -> dict[int, dict]:
    """sys_id -> {'role': 'prefill'|'decode'|None, 'ss': stage, 'sh': shard}.

    Derived from each rank's own COMP rows: they are the only ones carrying pl=
    for every rank, and under disaggregation a rank computes in exactly one pool.
    No external file needed."""
    roles: dict[int, dict] = {}
    comp = df[df["is_compute"]] if "is_compute" in df.columns else df[df["type"] == "GPU"]
    for sid, grp in comp.groupby("sys_id"):
        pls = set(grp["pl"].dropna())
        if len(pls) > 1:
            print(f"  ! sys {sid} computes in more than one pool ({sorted(pls)}): "
                  f"its send/recv tagging is unreliable", file=sys.stderr)
        pl = next(iter(pls)) if len(pls) == 1 else None
        ss, sh = grp["ss"].dropna(), grp["sh"].dropna()
        roles[int(sid)] = {"role": {"p": "prefill", "d": "decode"}.get(pl),
                           "ss": ss.iloc[0] if len(ss) else None,
                           "sh": sh.iloc[0] if len(sh) else None}
    return roles


def tag_comm_role(df: pd.DataFrame, roles: dict[int, dict]) -> pd.Series:
    """'send' / 'recv' / '' per row.

    A send and its recv share a name, so the owner decides which side a row is:
      KV, FIRSTTOK   source is the prefill pool            -> prefill side sends
      KVREQ          the pull request goes decode->prefill -> decode side sends
      PP, DECFB      source is the stage whose id is the name's ss
    Collectives have no direction and get ''.
    """
    def role_of(r) -> str:
        if not r.is_comm:
            return ""
        info = roles.get(int(r.sys_id), {})
        if r.cls in ("KV", "FIRSTTOK"):
            return "send" if info.get("role") == "prefill" else "recv"
        if r.cls == "KVREQ":
            return "send" if info.get("role") == "decode" else "recv"
        if r.cls in ("PP", "DECFB"):
            ss = info.get("ss")
            if r.ss is not None and ss is not None and str(ss) == str(r.ss):
                return "send"
            return "recv"
        return ""

    return pd.Series([role_of(r) for r in df.itertuples(index=False)], index=df.index)


def flag_wait_dominated(df: pd.DataFrame) -> pd.Series:
    """True where the reported duration is a scheduling artefact rather than time
    on the wire: the pre-posted recvs, identified by start_tick == the global
    simulation origin. The fallback when a class has no role information."""
    if df.empty:
        return pd.Series(dtype=bool)
    origin = int(df["start_tick"].min())
    return df["is_comm"] & (df["start_tick"] <= origin)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_run(run_dir: Path, pattern: str = "*.csv") -> pd.DataFrame | None:
    """Every stats_sys*.csv of one run, concatenated and annotated.

    Roles are resolved across the whole run rather than per file: a row's
    direction depends on the pool of the rank that owns it, which is only knowable
    once every rank's COMP rows are in hand."""
    frames = []
    for csv in sorted(run_dir.glob(pattern)):
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read {csv.name}: {exc}", file=sys.stderr)
            continue
        if df.empty or "name" not in df.columns:
            continue
        df["__file__"] = csv.name
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    fields = df["name"].map(parse_name)
    df["cls"] = fields.map(lambda d: d["cls"])
    for k in FIELD_KEYS:
        df[k] = fields.map(lambda d, k=k: d.get(k))
    cp = df["name"].map(classify_op)
    df["op_class"] = cp.map(lambda t: t[0])
    df["phase"] = cp.map(lambda t: t[1])

    df["is_compute"] = df["type"] == "GPU"
    df["is_comm"] = df["type"] == "COMM"
    df["comm_size"] = df["comm_size"].fillna(0)
    df["is_control"] = df["is_comm"] & (df["comm_size"] <= CONTROL_MAX_BYTES)

    df["comm_role"] = tag_comm_role(df, sys_roles(df))
    df["wait_dominated"] = flag_wait_dominated(df)
    return df


# --------------------------------------------------------------------------- #
# Counting each transfer once
# --------------------------------------------------------------------------- #
def sends(df: pd.DataFrame, mask=None) -> pd.DataFrame:
    """Point-to-point rows, one per transfer: the send side.

    Falls back to dropping the wait-dominated rows when a class has no role
    information, and says so, rather than silently returning double."""
    sub = df if mask is None else df[mask]
    if sub.empty:
        return sub
    if (sub["comm_role"] == "send").any():
        return sub[sub["comm_role"] == "send"]
    if sub["wait_dominated"].any():
        print(f"  ! no send/recv roles resolved for {sorted(set(sub['cls']))}: "
              f"falling back to dropping the pre-posted recvs", file=sys.stderr)
        return sub[~sub["wait_dominated"]]
    print(f"  ! cannot tell sends from recvs for {sorted(set(sub['cls']))}: "
          f"bytes and durations may be double-counted", file=sys.stderr)
    return sub


def collapse_collectives(df: pd.DataFrame, mask=None) -> pd.DataFrame:
    """One row per logical collective instead of one per participating rank.

    Keyed by (pl, ss, L, it, op); the slowest rank gives the wall-clock duration,
    and comm_size is the per-rank payload (identical across the group), so summing
    the result counts the collective once rather than tp times."""
    sub = df if mask is None else df[mask]
    if sub.empty:
        return sub
    keys = [c for c in ("pl", "ss", "L", "it", "op") if c in sub.columns]
    if not keys:
        return sub
    return (sub.groupby(keys, dropna=False)
            .agg(duration=("duration", "max"),
                 comm_size=("comm_size", "first"),
                 start_tick=("start_tick", "min"),
                 end_tick=("end_tick", "max"),
                 bw_bytes_per_ns=("bw_bytes_per_ns", "mean"),
                 n_ranks=("sys_id", "nunique"))
            .reset_index())


def unique_transfers(df: pd.DataFrame, op_class: str) -> pd.DataFrame:
    """One row per logical transfer of `op_class`, whichever kind it is."""
    mask = df["op_class"] == op_class
    if op_class in COLLECTIVE:
        return collapse_collectives(df, mask)
    return sends(df, mask)


def interval_union(starts, ends) -> float:
    """Total time covered by the union of intervals. Summing durations
    double-counts concurrent transfers; the union is the honest busy time."""
    iv = sorted(zip(starts, ends))
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s) if cur_s is not None else 0.0


def _merge(starts, ends) -> list[tuple[float, float]]:
    """Sorted intervals, overlapping ones fused -- the shared first step of
    union and overlap so both agree on what "an interval" means."""
    merged: list[list[float]] = []
    for s, e in sorted(zip(starts, ends)):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def interval_overlap(a_starts, a_ends, b_starts, b_ends) -> float:
    """Total time covered by BOTH interval sets -- how much of A is masked by B.
    Classic two-pointer sweep over two already-merged, sorted interval lists."""
    a, b = _merge(a_starts, a_ends), _merge(b_starts, b_ends)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total