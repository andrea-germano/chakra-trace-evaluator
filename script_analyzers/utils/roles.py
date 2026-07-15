#!/usr/bin/env python3
"""
utils.roles — which rank does what, declared rather than guessed.

The placement matrix is an exogenous input to a trace-driven simulator: the
generator already decided it, so the analyzer must be *told* it, not infer it.
Declaring it buys two things.

1. It replaces ``--bulk-mb``'s guesswork -- but NOT its job, and the difference
   matters. The role tells you what (src, dst) means; it does not tell you
   whether a flow is a bulk transfer or a control message. On the T1 reference
   run, prefill stage 1 sends its decode ranks two 8-BYTE messages alongside the
   80 MB per-layer KV, and folding them into 'kv' moves the KV slowdown p99 from
   36 to 4102 -- one flow dominating a percentile.

   So the split stays, but its line is READ rather than guessed: a flow that
   fits in one packet cannot be a bulk transfer, and the packet size is
   PACKET_PAYLOAD_SIZE in config.txt. Sub-MTU flows become 'ctrl'. That is a
   config value, not a tuned threshold: nothing to re-tune when the model
   changes, and the arbitrary 1 MiB is gone.

2. It makes the decode ranks -- the barrier population -- an assumption you can
   read at the top of a file, instead of a ``groupby(dst).size() > 1`` heuristic
   that silently picks a different set when the traffic changes.

Spec syntax, one token per stage:

    "p0=0,1 p1=2,3 d0=4,5 d1=6,7"     T1: prefill TP2/PP2, decode TP2/PP2

    p<i>   prefill pipeline stage i, ranks comma separated (the TP group)
    d<i>   decode  pipeline stage i

Flow classes, derived from the placement and the hop count alone:

    tp           1 hop: a host-to-host link. TP collectives; never crosses a
                 switch, so never congested.
    kv           prefill rank -> decode rank. The transfer the thesis is about.
    pp_prefill   prefill stage i -> prefill stage j. Activations.
    pp_decode    decode stage i -> decode stage j. Activations, once per step.
    other        anything else; if this is not ~0 the placement is wrong.
    ctrl         suffix on any fabric class: the flow fits in ONE packet, so it
                 carries a notification, not a payload. 'kv_ctrl' on T1 is
                 _emit_first_token: prefill tail -> decode head, 8 bytes, and on
                 the reference run it takes 99.9 ms because it queues behind the
                 bulk KV. That is head-of-line blocking of the control path, and
                 it is a result -- but only if it is not averaged into 'kv'.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

T1 = "p0=0,1 p1=2,3 d0=4,5 d1=6,7"

_TOK = re.compile(r"^([pd])(\d+)=([\d,]+)$")

FLOW_CLASSES = ("tp", "kv", "kv_ctrl", "pp_prefill", "pp_prefill_ctrl",
                "pp_decode", "pp_decode_ctrl", "other")


@dataclass(frozen=True)
class Placement:
    prefill: tuple[tuple[int, ...], ...]      # indexed by pipeline stage
    decode: tuple[tuple[int, ...], ...]

    @classmethod
    def parse(cls, spec: str) -> "Placement":
        pre: dict[int, tuple[int, ...]] = {}
        dec: dict[int, tuple[int, ...]] = {}
        for tok in spec.split():
            m = _TOK.match(tok)
            if not m:
                raise ValueError(
                    f"cannot parse placement token {tok!r}; expected 'p<stage>=<ranks>' "
                    f"or 'd<stage>=<ranks>', e.g. {T1!r}")
            kind, stage, ranks = m.group(1), int(m.group(2)), \
                tuple(int(x) for x in m.group(3).split(","))
            (pre if kind == "p" else dec)[stage] = ranks
        if not pre or not dec:
            raise ValueError(f"placement {spec!r} needs at least one p<i>= and one d<i>=")
        for name, d in (("prefill", pre), ("decode", dec)):
            if sorted(d) != list(range(len(d))):
                raise ValueError(f"{name} stages must be 0..{len(d)-1}, got {sorted(d)}")
        p = cls(tuple(pre[i] for i in range(len(pre))),
                tuple(dec[i] for i in range(len(dec))))
        seen: dict[int, str] = {}
        for role, stages in (("prefill", p.prefill), ("decode", p.decode)):
            for i, ranks in enumerate(stages):
                for r in ranks:
                    if r in seen:
                        raise ValueError(f"rank {r} is in both {seen[r]} and {role}{i}")
                    seen[r] = f"{role}{i}"
        return p

    @property
    def prefill_ranks(self) -> tuple[int, ...]:
        return tuple(r for s in self.prefill for r in s)

    @property
    def decode_ranks(self) -> tuple[int, ...]:
        return tuple(r for s in self.decode for r in s)

    def role(self, rank: int) -> tuple[str, int] | None:
        for name, stages in (("prefill", self.prefill), ("decode", self.decode)):
            for i, ranks in enumerate(stages):
                if rank in ranks:
                    return name, i
        return None

    def describe(self) -> str:
        return "  " + "  ".join(
            f"{k}{i}={','.join(map(str, r))}"
            for k, stages in (("p", self.prefill), ("d", self.decode))
            for i, r in enumerate(stages))


def classify(flows: pd.DataFrame, placement: Placement, mtu: int) -> pd.Series:
    """Flow class from (src, dst, hops, size). Requires `hops` -- i.e. the
    topology -- and `mtu` = PACKET_PAYLOAD_SIZE from config.txt. There is no
    class without both."""
    out = []
    for src, dst, hops, size in zip(flows["src"], flows["dst"], flows["hops"],
                                    flows["size"]):
        if hops == 1:
            out.append("tp")
            continue
        suffix = "_ctrl" if size <= mtu else ""
        a, b = placement.role(int(src)), placement.role(int(dst))
        if a is None or b is None:
            out.append("other")
        elif a[0] == "prefill" and b[0] == "decode":
            out.append("kv" + suffix)
        elif a[0] == b[0] == "prefill":
            out.append("pp_prefill" + suffix)
        elif a[0] == b[0] == "decode":
            out.append("pp_decode" + suffix)
        else:
            out.append("other")          # decode -> prefill: should not exist
    return pd.Series(out, index=flows.index, dtype="string")


def check(flows: pd.DataFrame, placement: Placement) -> list[str]:
    """Warnings that mean the declared placement does not match the traffic."""
    w = []
    counts = flows["flow_class"].value_counts()
    if counts.get("kv", 0) == 0:
        w.append("no flow is classified 'kv': --placement does not match the "
                 "traffic in fct.txt. Check the rank->node mapping.")
    if counts.get("other", 0):
        bad = flows.loc[flows["flow_class"] == "other", ["src", "dst"]]
        pairs = sorted({(int(s), int(d)) for s, d in zip(bad["src"], bad["dst"])})[:6]
        w.append(f"{counts['other']} flow(s) classified 'other' (src,dst e.g. "
                 f"{pairs}): ranks outside the declared placement, or a "
                 f"decode->prefill flow that should not exist.")
    seen = set(flows["src"]) | set(flows["dst"])
    declared = set(placement.prefill_ranks) | set(placement.decode_ranks)
    if extra := sorted(seen - declared):
        w.append(f"nodes {extra} appear in fct.txt but not in --placement.")
    if missing := sorted(declared - seen):
        w.append(f"ranks {missing} are declared but send/receive nothing.")
    return w


def from_astra(astra_run_dir: Path) -> Placement:
    """Recover the placement from the ASTRA-sim CSVs of one run.

    MLSynth writes pl=/ss=/sh= into every op name, so the generator's own
    decision is already in the trace and does not have to be remembered. This
    exists because the project has TWO sources of truth for the same fact --
    these tags, and the --placement string -- and nothing compared them. That is
    not a hypothetical: the whole reading of the T1 buffer sweep turned on which
    ranks were prefill, and getting it wrong changes which link is the bottleneck
    without changing a single number's plausibility.

    They cannot be merged: this needs the ASTRA CSVs, and the ns-3 analyzers see
    only IP addresses. So they get cross-checked instead. Print it, paste it,
    or let cross_check() compare the two."""
    from . import astra
    df = astra.read_run(astra_run_dir)
    if df is None or df.empty:
        raise ValueError(f"no readable stats_sys*.csv under {astra_run_dir}")
    pre: dict[int, list[int]] = {}
    dec: dict[int, list[int]] = {}
    for sid, info in astra.sys_roles(df).items():
        if info["role"] is None or info["ss"] is None:
            continue
        (pre if info["role"] == "prefill" else dec).setdefault(int(info["ss"]), []).append(sid)
    if not pre or not dec:
        raise ValueError(f"{astra_run_dir}: found prefill stages {sorted(pre)} and "
                         f"decode stages {sorted(dec)}; need at least one of each. "
                         f"The op names may not carry pl=/ss=.")
    return Placement.parse(" ".join(
        [f"p{i}={','.join(map(str, sorted(pre[i])))}" for i in sorted(pre)] +
        [f"d{i}={','.join(map(str, sorted(dec[i])))}" for i in sorted(dec)]))


def spec_of(p: Placement) -> str:
    return " ".join(
        [f"p{i}={','.join(map(str, r))}" for i, r in enumerate(p.prefill)] +
        [f"d{i}={','.join(map(str, r))}" for i, r in enumerate(p.decode)])


def cross_check(declared: Placement, astra_run_dir: Path) -> str | None:
    """None if the declared placement matches what MLSynth wrote into the trace,
    else the message saying how they differ. Cheap; run it whenever the ASTRA
    directory for the run is reachable."""
    try:
        found = from_astra(astra_run_dir)
    except Exception as e:                                  # noqa: BLE001
        return f"could not cross-check --placement against {astra_run_dir}: {e}"
    if found == declared:
        return None
    return (f"--placement disagrees with the ASTRA trace of this run.\n"
            f"      declared: {spec_of(declared)}\n"
            f"      in trace: {spec_of(found)}\n"
            f"    The trace is what was simulated. Every rank-dependent number "
            f"below is computed against the declared one.")


def add_argument(ap) -> None:
    ap.add_argument("--placement", default=T1,
                    help=f"rank->role map, 'p<stage>=<ranks> d<stage>=<ranks>'. "
                         f"Recover it from a run with "
                         f"`python3 -m utils.roles --from-astra <dir>` instead of "
                         f"remembering it (default, T1: {T1!r})")


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print the placement spec that the "
                                             "ASTRA-sim CSVs of a run imply.")
    ap.add_argument("--from-astra", required=True, type=Path, metavar="DIR",
                    help="an ASTRA run directory containing stats_sys*.csv")
    a = ap.parse_args(argv)
    try:
        print(spec_of(from_astra(a.from_astra)))
        return 0
    except Exception as e:                                  # noqa: BLE001
        print(f"ABORT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())