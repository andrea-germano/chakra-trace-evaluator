#!/usr/bin/env python3
"""
validate_topology.py - sanity-check ns-3 / ASTRA-sim physical_topology.txt files
and report the architecture and the collective groups the fabric supports.

Usage:
    python validate_topology.py <folder> [--config config.txt]

Scans <folder> recursively for physical_topology.txt and, for each file:
  1. Validates it structurally (header counts, ids, connectivity, no dangling nodes).
  2. If --config is given, checks the topology against THAT config
     (link speeds must appear in its KMAX/KMIN/PMAX maps; GLOBAL_T sanity).
  3. Prints the inferred architecture and the collective groups the fabric can host.

Everything config-dependent is read from the config file you pass in - the script
makes no assumptions about which link speeds or settings are "allowed".

Exit code: 0 if every file passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

TOPOLOGY_NAMES = ("physical_topology.txt", "physical_config.txt", "topology.txt")

RATE_UNITS = {"tbps": 10**12, "gbps": 10**9, "mbps": 10**6, "kbps": 10**3, "bps": 1}
TIME_UNITS = {"ms": 1e6, "us": 1e3, "ns": 1.0, "s": 1e9}  # -> nanoseconds


def parse_rate(tok: str) -> int:
    t = tok.strip().lower()
    for suf, mult in RATE_UNITS.items():
        if t.endswith(suf):
            return int(float(t[: -len(suf)]) * mult)
    raise ValueError(f"cannot parse rate {tok!r}")


def parse_delay_ns(tok: str) -> float:
    t = tok.strip().lower()
    for suf, mult in TIME_UNITS.items():
        if t.endswith(suf):
            return float(t[: -len(suf)]) * mult
    raise ValueError(f"cannot parse delay {tok!r}")


def fmt_rate(bps: int) -> str:
    for suf, mult in (("Tbps", 10**12), ("Gbps", 10**9), ("Mbps", 10**6)):
        if bps >= mult:
            return f"{bps/mult:g}{suf}"
    return f"{bps}bps"


def fmt_ranks(ranks) -> str:
    """[0,1,2,3,7] -> '0-3,7'"""
    ranks = sorted(ranks)
    if not ranks:
        return "-"
    out, start, prev = [], ranks[0], ranks[0]
    for r in ranks[1:]:
        if r == prev + 1:
            prev = r
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = r
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(out)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class Config:
    """Only the fields this validator cares about. Missing / templated values stay None."""

    def __init__(self, path: Path | None):
        self.path = path
        self.ecn_rates: set[int] | None = None  # rates present in KMAX_MAP
        self.global_t: int | None = None
        if path:
            self._parse(path)

    def _parse(self, path: Path) -> None:
        for line in path.read_text().splitlines():
            f = line.split()
            if not f:
                continue
            if f[0] == "KMAX_MAP" and len(f) >= 2:
                # KMAX_MAP <count> <rate> <k> <rate> <k> ...
                self.ecn_rates = set(int(f[i]) for i in range(2, len(f), 2))
            elif f[0] == "GLOBAL_T" and len(f) >= 2:
                try:
                    self.global_t = int(f[1])
                except ValueError:
                    pass


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
class Topology:
    def __init__(self, path: Path):
        self.path = path
        self.node_num = self.switch_num = self.link_num = 0
        self.switches: set[int] = set()
        self.hosts: set[int] = set()
        self.links: list = []                       # (src, dst, rate, delay, lineno)
        self.adj: dict = defaultdict(dict)          # a -> {b: (rate, delay)}


class Report:
    def __init__(self):
        self.errors, self.warnings, self.notes = [], [], []

    def error(self, m): self.errors.append(m)
    def warn(self, m): self.warnings.append(m)
    def note(self, m): self.notes.append(m)

    @property
    def ok(self): return not self.errors


def load_topology(path: Path, rep: Report) -> Topology | None:
    lines = [
        (i + 1, l.strip())
        for i, l in enumerate(path.read_text().splitlines())
        if l.strip() and not l.strip().startswith("#")
    ]
    if len(lines) < 2:
        rep.error("file needs at least a header line and a switch list")
        return None

    head = lines[0][1].split()
    if len(head) != 3 or not all(x.lstrip("-").isdigit() for x in head):
        rep.error(f"line 1: header must be 'node_num switch_num link_num', got {lines[0][1]!r}")
        return None

    topo = Topology(path)
    topo.node_num, topo.switch_num, topo.link_num = (int(x) for x in head)

    # switch list
    sw = lines[1][1].split()
    if not all(x.lstrip("-").isdigit() for x in sw):
        rep.error(f"line {lines[1][0]}: switch list must be integers")
        return None
    sw = [int(x) for x in sw]
    if len(sw) != topo.switch_num:
        rep.error(f"header says switch_num={topo.switch_num} but switch list has {len(sw)} entries")
    if len(set(sw)) != len(sw):
        rep.error(f"line {lines[1][0]}: duplicate switch ids")
    topo.switches = {s for s in sw if 0 <= s < topo.node_num}
    topo.hosts = set(range(topo.node_num)) - topo.switches

    # links
    seen: dict[tuple[int, int], int] = {}
    for lineno, line in lines[2:]:
        f = line.split()
        if len(f) != 5:
            rep.error(f"line {lineno}: link needs 5 fields 'src dst rate delay err', got {len(f)}")
            continue
        try:
            src, dst = int(f[0]), int(f[1])
            rate, delay = parse_rate(f[2]), parse_delay_ns(f[3])
        except ValueError as e:
            rep.error(f"line {lineno}: {e}")
            continue
        if not (0 <= src < topo.node_num and 0 <= dst < topo.node_num):
            rep.error(f"line {lineno}: node id out of range 0..{topo.node_num-1}")
            continue
        if src == dst:
            rep.error(f"line {lineno}: self-loop on node {src}")
            continue
        key = (min(src, dst), max(src, dst))
        if key in seen:
            rep.error(f"line {lineno}: duplicate link {src}<->{dst} (first on line {seen[key]})")
            continue
        seen[key] = lineno
        topo.links.append((src, dst, rate, delay, lineno))
        topo.adj[src][dst] = (rate, delay)
        topo.adj[dst][src] = (rate, delay)

    if len(topo.links) != topo.link_num:
        rep.error(f"header says link_num={topo.link_num} but {len(topo.links)} links parsed")
    return topo


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_structure(topo: Topology, rep: Report) -> None:
    H = len(topo.hosts)
    if topo.hosts != set(range(H)):
        rep.error(f"hosts must be the contiguous range 0..{H-1} (rank i maps to node i); "
                  f"got {fmt_ranks(topo.hosts)}")
    if H == 0:
        rep.error("no hosts: every node is a switch")

    for nid in range(topo.node_num):
        if len(topo.adj.get(nid, {})) == 0:
            kind = "switch" if nid in topo.switches else "host"
            rep.error(f"node {nid} ({kind}) has no links (dangling)")

    # connectivity
    if topo.adj:
        seen = {0}
        q = deque([0])
        while q:
            for nb in topo.adj[q.popleft()]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        missing = set(range(topo.node_num)) - seen
        if missing:
            rep.error(f"graph not connected; unreachable from node 0: {fmt_ranks(missing)}")


def switch_only_routes(topo: Topology):
    """BFS that only expands switches (mirrors common.h::CalculateRoute).
    Returns hops[(a,b)], bw[(a,b)] for host pairs, and ECMP next-hop lists."""
    hops, bw, nexthops = {}, {}, defaultdict(list)
    for host in sorted(topo.hosts):
        dis = {host: 0}
        bwm = {host: float("inf")}
        q, i = [host], 0
        while i < len(q):
            now = q[i]; i += 1
            for nb, (rate, _) in topo.adj[now].items():
                if nb not in dis:
                    dis[nb] = dis[now] + 1
                    bwm[nb] = min(bwm[now], rate)
                    if nb in topo.switches:          # never route through a host
                        q.append(nb)
                if dis[nb] == dis[now] + 1:
                    nexthops[(nb, host)].append(now)
        for other in topo.hosts:
            if other != host and other in dis:
                hops[(other, host)] = dis[other]
                bw[(other, host)] = int(bwm[other])
    return hops, bw, nexthops


def validate_routing(topo: Topology, hops, nexthops, rep: Report) -> None:
    hosts = sorted(topo.hosts)
    dead = [(a, b) for a in hosts for b in hosts if a != b and (a, b) not in hops]
    if dead:
        pairs = ", ".join(f"{a}->{b}" for a, b in dead[:6])
        rep.error(f"host pairs with no switch-only path (ns-3 won't route through a GPU): {pairs}"
                  + ("" if len(dead) <= 6 else f" (+{len(dead)-6} more)"))
    if any(len(v) > 1 for v in nexthops.values()):
        rep.note("ECMP present: some pairs have multiple equal-cost paths, so results are "
                 "hash/seed-dependent (expected in a Clos, not in a rail design).")


def validate_against_config(topo: Topology, cfg: Config, rep: Report) -> None:
    if cfg.path is None:
        rep.note("no --config given: skipping ECN-map / GLOBAL_T checks.")
        return

    # link speeds on switch ports must have an ECN entry in this config
    if cfg.ecn_rates is not None:
        switch_rates = {
            r for (s, d, r, _, _) in topo.links if s in topo.switches or d in topo.switches
        }
        missing = sorted(r for r in switch_rates if r not in cfg.ecn_rates)
        if missing:
            rep.error(
                "link speeds not present in the config's KMAX/KMIN/PMAX maps: "
                + ", ".join(fmt_rate(r) for r in missing)
                + ". ns-3 asserts on every switch-port rate, so the run aborts. Add "
                + ", ".join(str(r) for r in missing) + f" to the maps in {cfg.path.name}."
            )

    # GLOBAL_T only matters when link speeds are heterogeneous
    rates = {r for (_, _, r, _, _) in topo.links}
    if cfg.global_t == 1 and len(rates) > 1:
        rep.warn(
            f"config sets GLOBAL_T 1 but this topology mixes link speeds "
            f"({fmt_rate(min(rates))}..{fmt_rate(max(rates))}); one global window/RTT will be "
            "wrong for one class of link. Set GLOBAL_T 0."
        )


# --------------------------------------------------------------------------- #
# architecture + comm groups
# --------------------------------------------------------------------------- #
def domains_by_rate(topo: Topology):
    """rate -> {switch: [hosts attached at that rate]}, sorted fast-first."""
    tiers = {}
    for r in sorted({rt for (_, _, rt, _, _) in topo.links}, reverse=True):
        by_sw = defaultdict(list)
        for (s, d, rt, _, _) in topo.links:
            if rt != r:
                continue
            h, sw = (s, d) if s in topo.hosts else (d, s)
            if h in topo.hosts and sw in topo.switches:
                by_sw[sw].append(h)
        if by_sw:
            tiers[r] = {sw: sorted(hs) for sw, hs in sorted(by_sw.items())}
    return tiers


def full_bw_domains(topo: Topology, bw):
    """Groups of GPUs mutually reachable at the top bandwidth = valid TP groups."""
    if not bw:
        return None
    top = max(bw.values())
    parent = {h: h for h in topo.hosts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), v in bw.items():
        if v == top:
            parent[find(a)] = find(b)
    groups = defaultdict(list)
    for h in topo.hosts:
        groups[find(h)].append(h)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0]), top


def print_report(topo: Topology, hops, bw) -> None:
    tiers = domains_by_rate(topo)
    rates = list(tiers)
    host_facing = {sw for t in tiers.values() for sw in t}
    spine = sorted(topo.switches - host_facing)

    print("  ARCHITECTURE")
    print(f"    {len(topo.hosts)} GPUs ({fmt_ranks(topo.hosts)}), "
          f"{len(topo.switches)} switches, {len(topo.links)} links")
    for r in rates:
        tier = tiers[r]
        per_gpu = all(len(h) == 1 for h in tier.values())
        if per_gpu:
            label = "per-GPU leaf (1 GPU/switch)"
        elif r == rates[0] and len(rates) > 1:
            label = "scale-up domains"
        else:
            label = "scale-out"
        sizes = sorted({len(h) for h in tier.values()})
        print(f"    {fmt_rate(r)} - {label}: {len(tier)} switch(es), domain sizes {sizes}")
        if not per_gpu:
            for sw, hs in tier.items():
                print(f"        switch {sw:<3} <- GPUs {fmt_ranks(hs)}")
    if spine:
        print(f"    spine (no GPUs): {fmt_ranks(spine)}")

    result = full_bw_domains(topo, bw)
    if not result:
        return
    groups, top = result
    print("\n  COLLECTIVE GROUPS (from topology only; comm_groups.json not read)")
    print(f"    TP groups must fit in one full-bandwidth domain ({fmt_rate(top)}):")
    for i, g in enumerate(groups):
        print(f"        domain {i}: GPUs {fmt_ranks(g)}   ({len(g)} GPU{'s'*(len(g)!=1)})")
    sizes = sorted({len(g) for g in groups})
    legal = [n for n in range(1, max(sizes) + 1) if all(s % n == 0 for s in sizes)]
    print(f"    legal tp_size (fits every domain): {legal}")
    if len(sizes) > 1:
        print(f"    note: domains are non-uniform (sizes {sizes}) -> different TP per pool "
              f"(e.g. prefill vs decode); KV is resharded across the {fmt_rate(min(bw.values()))} plane.")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def process(path: Path, cfg: Config) -> bool:
    print(f"\n{'='*70}\n {path}\n{'='*70}")
    rep = Report()
    topo = load_topology(path, rep)
    if topo is None:
        for m in rep.errors:
            print(f"  [ERROR] {m}")
        print("\n  FAIL")
        return False

    validate_structure(topo, rep)
    hops = bw = None
    if topo.hosts and topo.links:
        hops, bw, nexthops = switch_only_routes(topo)
        validate_routing(topo, hops, nexthops, rep)
        validate_against_config(topo, cfg, rep)

    if hops and bw:
        print()
        print_report(topo, hops, bw)

    print()
    for m in rep.errors:
        print(f"  [ERROR] {m}")
    for m in rep.warnings:
        print(f"  [WARN]  {m}")
    for m in rep.notes:
        print(f"  [note]  {m}")
    print()
    if rep.ok and not rep.warnings:
        print("  PASS")
    elif rep.ok:
        print("  PASS (with warnings)")
    else:
        print(f"  FAIL - {len(rep.errors)} error(s)")
    return rep.ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ns-3 / ASTRA-sim topology files.")
    ap.add_argument("folder", help="folder scanned recursively for topology files")
    ap.add_argument("--config", help="network config to validate against (KMAX_MAP, GLOBAL_T, ...)")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    cfg_path = Path(args.config) if args.config else None
    if cfg_path and not cfg_path.is_file():
        print(f"error: config {cfg_path} not found", file=sys.stderr)
        return 2
    cfg = Config(cfg_path)

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name in TOPOLOGY_NAMES)
    if not files:
        print(f"error: no {' / '.join(TOPOLOGY_NAMES)} found under {root}", file=sys.stderr)
        return 2

    results = [process(p, cfg) for p in files]
    print(f"\n{'='*70}\n {sum(results)}/{len(results)} passed\n{'='*70}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())