#!/usr/bin/env python3
"""
utils.cli — the abort convention every analyzer shares.

`Abort` marks a condition under which no number the script could print would mean
anything (a missing run, an unresolved placement, a sweep that moves two knobs).
It is raised through `need`, caught once in each `main`, and turned into a
non-zero exit -- never downgraded to a default. Kept here so the four analyzers
and their two cross-model companions raise it the same way instead of each
re-declaring it.
"""

from __future__ import annotations


class Abort(Exception):
    """A condition under which no number this script could print would mean
    anything. Never caught except at the top of main(), never given a default."""


def need(cond, msg: str) -> None:
    if not cond:
        raise Abort(msg)
