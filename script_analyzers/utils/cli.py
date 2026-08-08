#!/usr/bin/env python3
"""
utils.cli — the abort and warning conventions every analyzer shares.

`Abort` marks a condition under which no number the script could print would mean
anything (a missing run, an unresolved placement, a sweep that moves two knobs).
It is raised through `need`, caught once in each `main`, and turned into a
non-zero exit -- never downgraded to a default. Kept here so the four analyzers
and their two cross-model companions raise it the same way instead of each
re-declaring it.

`warn` is the other half: a condition the numbers survive but are conditional
on. Warnings accumulate in ONE stream (`WARNINGS`) across every module of a run
-- the sweeps and the measures they share all speak through it -- and `main`
drains them once at the end via `drain_warnings`, whose exit code says whether
there were any.
"""

from __future__ import annotations

import sys


class Abort(Exception):
    """A condition under which no number this script could print would mean
    anything. Never caught except at the top of main(), never given a default."""


def need(cond, msg: str) -> None:
    if not cond:
        raise Abort(msg)


# The single warning stream of a run. Module-level on purpose: buffer_sweep,
# incast_sweep, cc_sweep and utils.measures all warn into the same list, and
# whichever main() is running drains it exactly once.
WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  ! {msg}", file=sys.stderr)


def drain_warnings(note: str = "") -> int:
    """Print the accumulated warnings once, at the end of a main(), and return
    the exit code they imply (1 if any, else 0). `note` extends the header --
    e.g. ' — the numbers above are conditional on them'."""
    if not WARNINGS:
        return 0
    print(f"\n{len(WARNINGS)} WARNING(S){note}:")
    for w in WARNINGS:
        print(f"  ! {w}")
    return 1
