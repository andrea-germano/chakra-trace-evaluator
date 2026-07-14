#!/usr/bin/env python3
"""
validate_topology.py — sanity-check ns-3 / ASTRA-sim `physical_topology.txt` files
and report the architecture and the collective groups the fabric can support.

Usage:
    python validate_topology.py <folder> [--buffer-size MB] [--headroom-factor N]
                                         [--payload BYTES] [--quiet]

Scans <folder> recursively for physical_topology.txt (also accepts
physical_config.txt / topology.txt), then for each file:

  1. Validates the file structurally (header counts, node ids, degrees,
     connectivity, no dangling nodes, no host-transit paths, ...).
  2. Replays the exact BFS that scratch/common.h::CalculateRoute performs,
     so the reported paths / ECMP / bottleneck bandwidths are the ones ns-3
     will actually use.
  3. Infers the architecture (scale-up domains, scale-out tiers, spine).
  4. Derives the collective groups the fabric *can* support, purely from the
     topology. It does NOT read or compare against comm_groups.json.

Exit code: 0 if every file passes with no ERRORs, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

TOPOLOGY_NAMES = ("physical_topology.txt", "physical_config.txt", "topology.txt")

# Link speeds that the shipped ns-3 configs provide KMIN/KMAX/PMAX entries for.
# scratch/common.h asserts on every *switch* port whose rate is missing here.
KNOWN_ECN_RATES_BPS = {
    25_000_000_000,
    40_000_000_000,
    100_000_000_000,
    200_000_000_000,
    400_000_000_000,
    2_400_000_000_000,
}

RATE_UNITS = {"bps": 1, "kbps": 10**3, "mbps": 10**6, "gbps": 10**9, "tbps": 10**12}
TIME_UNITS = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}  # -> nanoseconds

C_RESET, C_RED, C_YEL, C_GRN, C_DIM, C_BOLD = (
    "\033[0m",
    "\033[31m",
    "\033[33m",
    "\033[32m",
    "\033[2m",
    "\033[1m",
)


def _c(text: str, colour: str) -> str:
    return text if not sys.stdout.isatty() else f"{colour}{text}{C_RESET}"


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_rate(token: str) -> int:
    """'4800Gbps' -> 4_800_000_000_000 (bits per second)."""
    t = token.strip()
    for suffix, mult in sorted(RATE_UNITS.items(), key=lambda kv: -len(kv[0])):
        if t.lower().endswith(suffix):
            return int(float(t[: -len(suffix)]) * mult)
    raise ValueError(f"cannot parse data rate {token!r}")


def parse_delay_ns(token: str) -> float:
    """'0.0005ms' -> 500.0 (nanoseconds)."""
    t = token.strip()
    for suffix, mult in sorted(TIME_UNITS.items(), key=lambda kv: -len(kv[0])):
        if t.lower().endswith(suffix):
            return float(t[: -len(suffix)]) * mult
    raise ValueError(f"cannot parse delay {token!r}")


def fmt_rate(bps: int) -> str:
    for suffix, mult in (("Tbps", 10**12), ("Gbps", 10**9), ("Mbps", 10**6)):
        if bps >= mult:
            v = bps / mult
            return f"{v:g}{suffix}"
    return f"{bps}bps"


def fmt_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns/1e6:g}ms"
    if ns >= 1e3:
        return f"{ns/1e3:g}us"
    return f"{ns:g}ns"


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


@dataclass
class Link:
    src: int
    dst: int
    rate_bps: int
    delay_ns: float
    error_rate: float
    lineno: int


@dataclass
class Topology:
    path: Path
    node_num: int
    switch_num: int
    link_num: int
    switches: set = field(default_factory=set)
    links: list = field(default_factory=list)
    hosts: set = field(default_factory=set)
    adj: dict = field(default_factory=lambda: defaultdict(dict))  # a -> b -> Link


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, m):
        self.errors.append(m)

    def warn(self, m):
        self.warnings.append(m)

    def note(self, m):
        self.notes.append(m)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_topology(path: Path, rep: Report) -> Topology | None:
    raw = [l.strip() for l in path.read_text().splitlines()]
    lines = [(i + 1, l) for i, l in enumerate(raw) if l and not l.startswith("#")]
    if len(lines) < 2:
        rep.error("file has fewer than 2 non-empty lines; expected a header and a switch list")
        return None

    head = lines[0][1].split()
    if len(head) != 3:
        rep.error(
            f"line 1: header must be exactly 3 fields "
            f"'node_num switch_num link_num', got {len(head)}: {lines[0][1]!r}"
        )
        return None
    try:
        node_num, switch_num, link_num = (int(x) for x in head)
    except ValueError:
        rep.error(f"line 1: header fields must be integers: {lines[0][1]!r}")
        return None

    topo = Topology(path=path, node_num=node_num, switch_num=switch_num, link_num=link_num)

    # --- switch list -------------------------------------------------------
    try:
        sw = [int(x) for x in lines[1][1].split()]
    except ValueError:
        rep.error(f"line {lines[1][0]}: switch list must contain integers only")
        return None
    if len(sw) != switch_num:
        rep.error(
            f"line {lines[1][0]}: header declares switch_num={switch_num} "
            f"but the switch list has {len(sw)} entries"
        )
    if len(set(sw)) != len(sw):
        dup = sorted({s for s in sw if sw.count(s) > 1})
        rep.error(f"line {lines[1][0]}: duplicate switch ids {dup}")
    for s in sw:
        if not 0 <= s < node_num:
            rep.error(f"line {lines[1][0]}: switch id {s} outside 0..{node_num-1}")
    topo.switches = {s for s in sw if 0 <= s < node_num}
    topo.hosts = set(range(node_num)) - topo.switches

    # --- links -------------------------------------------------------------
    seen: dict[tuple[int, int], int] = {}
    for lineno, line in lines[2:]:
        f = line.split()
        if len(f) != 5:
            rep.error(
                f"line {lineno}: link needs 5 fields "
                f"'src dst rate delay error_rate', got {len(f)}: {line!r}"
            )
            continue
        try:
            src, dst = int(f[0]), int(f[1])
            rate = parse_rate(f[2])
            delay = parse_delay_ns(f[3])
            err = float(f[4])
        except ValueError as e:
            rep.error(f"line {lineno}: {e}")
            continue

        for nid in (src, dst):
            if not 0 <= nid < node_num:
                rep.error(f"line {lineno}: node id {nid} outside 0..{node_num-1}")
        if src == dst:
            rep.error(f"line {lineno}: self-loop on node {src}")
            continue
        key = (min(src, dst), max(src, dst))
        if key in seen:
            rep.error(
                f"line {lineno}: duplicate link {src}<->{dst} "
                f"(already declared on line {seen[key]}); ns-3 links are bidirectional, "
                f"declare each pair once"
            )
            continue
        seen[key] = lineno
        if rate <= 0:
            rep.error(f"line {lineno}: non-positive data rate {f[2]}")
        if err < 0 or err > 1:
            rep.error(f"line {lineno}: error_rate {err} outside [0,1]")

        link = Link(src, dst, rate, delay, err, lineno)
        topo.links.append(link)
        topo.adj[src][dst] = link
        topo.adj[dst][src] = link

    if len(topo.links) != link_num:
        rep.error(
            f"header declares link_num={link_num} but {len(topo.links)} valid links were parsed"
        )
    return topo


# --------------------------------------------------------------------------- #
# structural validation
# --------------------------------------------------------------------------- #
def validate_structure(topo: Topology, rep: Report) -> None:
    # Hosts must be 0..H-1. ASTRA-sim maps rank i -> ns-3 node i.
    H = len(topo.hosts)
    if topo.hosts != set(range(H)):
        rep.error(
            f"hosts must occupy the contiguous range 0..{H-1} (ASTRA-sim maps rank i to node i); "
            f"got hosts {fmt_ranks(topo.hosts)}"
        )
    if H == 0:
        rep.error("no hosts: every node is listed as a switch")

    # Dangling / isolated nodes.
    for nid in range(topo.node_num):
        deg = len(topo.adj.get(nid, {}))
        if deg == 0:
            kind = "switch" if nid in topo.switches else "host"
            rep.error(f"node {nid} ({kind}) is declared but has no links (dangling)")

    # A switch with a single link can never forward anything.
    for s in sorted(topo.switches):
        if len(topo.adj.get(s, {})) == 1:
            rep.warn(f"switch {s} has only one link; it is a dead end and can never forward")

    # Global connectivity (ignoring the host-transit rule).
    if topo.node_num and topo.adj:
        seen = {0}
        q = deque([0])
        while q:
            for nb in topo.adj[q.popleft()]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        missing = set(range(topo.node_num)) - seen
        if missing:
            rep.error(f"graph is not connected; nodes unreachable from node 0: {fmt_ranks(missing)}")


# --------------------------------------------------------------------------- #
# routing — mirrors scratch/common.h::CalculateRoute
# --------------------------------------------------------------------------- #
def calculate_routes(topo: Topology, payload: int = 1000):
    """
    BFS from every host. Crucially, only SWITCHES are expanded (common.h:240),
    so packets can never transit through another GPU. Returns:
        hops[(s,d)]      -- shortest-path hop count between hosts
        bw[(s,d)]        -- bottleneck bandwidth (bps) on that path
        delay[(s,d)]     -- one-way propagation delay (ns)
        rtt[(s,d)]       -- 2*delay + tx delay, as ns-3 computes it
        nexthops[(n,d)]  -- equal-cost next hops from node n toward host d (ECMP width)
    """
    hops, bw, delay, rtt, nexthops = {}, {}, {}, {}, defaultdict(list)

    for host in sorted(topo.hosts):
        dis = {host: 0}
        dly = {host: 0.0}
        tx = {host: 0.0}
        bwm = {host: float("inf")}
        q = [host]
        i = 0
        while i < len(q):
            now = q[i]
            i += 1
            d = dis[now]
            for nb, link in topo.adj[now].items():
                if nb not in dis:
                    dis[nb] = d + 1
                    dly[nb] = dly[now] + link.delay_ns
                    tx[nb] = tx[now] + payload * 8 * 1e9 / link.rate_bps
                    bwm[nb] = min(bwm[now], link.rate_bps)
                    if nb in topo.switches:  # hosts are never expanded
                        q.append(nb)
                if dis[nb] == d + 1:
                    nexthops[(nb, host)].append(now)

        for other in sorted(topo.hosts):
            if other == host or other not in dis:
                continue
            hops[(other, host)] = dis[other]
            bw[(other, host)] = int(bwm[other])
            delay[(other, host)] = dly[other]
            rtt[(other, host)] = 2 * dly[other] + tx[other]

    return hops, bw, delay, rtt, nexthops


def validate_routing(topo: Topology, hops, bw, nexthops, rep: Report) -> None:
    hosts = sorted(topo.hosts)
    unreachable = [
        (a, b) for a in hosts for b in hosts if a != b and (a, b) not in hops
    ]
    if unreachable:
        pairs = ", ".join(f"{a}->{b}" for a, b in unreachable[:6])
        more = "" if len(unreachable) <= 6 else f" (+{len(unreachable)-6} more)"
        rep.error(
            "some host pairs have NO switch-only path. ns-3 will not route through a GPU, "
            f"so these flows can never be delivered: {pairs}{more}"
        )

    ecmp = {k: v for k, v in nexthops.items() if len(v) > 1}
    if ecmp:
        widths = sorted({len(v) for v in ecmp.values()})
        rep.note(
            f"ECMP present: {len(ecmp)} (node,dst) entries have multiple equal-cost next hops "
            f"(widths {widths}). Flows are hashed across them (SwitchNode::EcmpHash), so results "
            f"become seed-dependent; this is expected in a Clos, unexpected in a rail design."
        )
    else:
        rep.note("no ECMP: every (src,dst) pair has a single shortest path; routing is deterministic")


# --------------------------------------------------------------------------- #
# ns-3 / ASTRA-sim specific hazards
# --------------------------------------------------------------------------- #
def validate_ns3_hazards(topo: Topology, bw, rtt, rep, buffer_mb, hdrm_factor, payload) -> None:
    # ---- 1. ECN maps must cover every SWITCH port rate (common.h asserts) ---
    switch_rates = {
        l.rate_bps for l in topo.links if l.src in topo.switches or l.dst in topo.switches
    }
    missing = sorted(r for r in switch_rates if r not in KNOWN_ECN_RATES_BPS)
    if missing:
        rep.error(
            "these link speeds terminate on a switch but are NOT in the KMIN/KMAX/PMAX maps of the "
            "shipped ns-3 configs: "
            + ", ".join(fmt_rate(r) for r in missing)
            + ". scratch/common.h asserts 'must set kmin for each link speed' -> the run will "
            "ABORT at startup. Add these rates (in bps: "
            + ", ".join(str(r) for r in missing)
            + ") to KMAX_MAP / KMIN_MAP / PMAX_MAP and bump their entry counts."
        )

    # ---- 2. get_nic_rate() is order-dependent with multi-homed hosts --------
    multihomed = {h for h in topo.hosts if len(topo.adj.get(h, {})) > 1}
    if multihomed:
        rep.note(
            f"{len(multihomed)} host(s) are multi-homed (degree > 1): {fmt_ranks(multihomed)}. "
            "Supported (RdmaDriver::Init builds one RdmaInterfaceMgr per NIC and RdmaHw picks the "
            "NIC per destination IP), but see the order-dependency below."
        )
        host0_links = [l for l in topo.links if 0 in (l.src, l.dst) and 0 in topo.hosts]
        if host0_links:
            first = min(host0_links, key=lambda l: l.lineno)
            rates = sorted({l.rate_bps for l in host0_links})
            nic_rate = first.rate_bps
            if len(rates) > 1:
                rep.warn(
                    f"get_nic_rate() (common.h:319) returns GetDevice(1) of host 0 = the FIRST link "
                    f"listed for node 0, i.e. line {first.lineno} -> nic_rate = {fmt_rate(nic_rate)}. "
                    f"Host 0 also has links at {', '.join(fmt_rate(r) for r in rates if r != nic_rate)}. "
                    "nic_rate drives pfc_a_shift on EVERY switch port (common.h:740), so simply "
                    "reordering the lines in this file silently changes the dynamic PFC threshold "
                    "across the whole fabric. Pin the intended NIC by listing it first."
                )
                if nic_rate < max(rates):
                    rep.warn(
                        f"  -> nic_rate is currently the SLOW link. Every port faster than "
                        f"{fmt_rate(nic_rate)} will have its alpha shift driven toward 0 "
                        f"(much larger PFC threshold). Probably not what you want."
                    )

    # ---- 3. per-switch headroom vs buffer ----------------------------------
    buffer_bytes = buffer_mb * 1024 * 1024
    for s in sorted(topo.switches):
        total = 0
        for nb, l in topo.adj[s].items():
            # common.h:735  headroom = rate * delay / 8 / 1e9 * headroom_factor
            total += int(l.rate_bps * l.delay_ns / 8 / 1e9 * hdrm_factor)
        frac = total / buffer_bytes if buffer_bytes else 99
        msg = (
            f"switch {s}: {len(topo.adj[s])} ports, PFC headroom {total/1e6:.2f} MB "
            f"= {frac*100:.0f}% of the {buffer_mb} MB buffer"
        )
        if frac >= 1.0:
            rep.error(
                msg
                + ". total_hdrm exceeds buffer_size -> SwitchMmu::GetPfcThreshold "
                "(switch-mmu.cc:94) underflows in uint32 -> PFC effectively never fires and you "
                f"get silent drops. Raise BUFFER_SIZE above {int(total/1024/1024)+1}."
            )
        elif frac >= 0.4:
            rep.warn(
                msg
                + ". Headroom is eating a large share of the buffer, leaving little for the shared "
                "pool. Raise BUFFER_SIZE, and note this scales linearly with the domain size — a "
                "wider NVSwitch domain at this rate/delay would overflow it."
            )

    # ---- 4. speed heterogeneity / GLOBAL_T ---------------------------------
    rates = sorted({l.rate_bps for l in topo.links})
    if len(rates) > 1:
        ratio = rates[-1] / rates[0]
        if ratio >= 4:
            max_bdp = max((rtt[k] * bw[k] / 8e9 for k in rtt), default=0)
            max_rtt = max(rtt.values(), default=0)
            rep.warn(
                f"link speeds span {fmt_rate(rates[0])}..{fmt_rate(rates[-1])} ({ratio:.0f}x). "
                f"Set GLOBAL_T 0 in the network config: with GLOBAL_T 1 every QP is given "
                f"win = maxBdp ({max_bdp/1e3:.0f} KB, set by the fastest path) and "
                f"baseRtt = maxRtt ({fmt_ns(max_rtt)}, set by the slowest path) — each wrong for "
                "the other class of link."
            )
            rep.warn(
                f"also check the CC constants: RATE_AI / MIN_RATE in the sample configs "
                f"(50 Mb/s, 100 Mb/s) are ~5 orders of magnitude below {fmt_rate(rates[-1])} and "
                "will never ramp a fast link to line rate."
            )

    # ---- 5. degenerate delays ----------------------------------------------
    tiny = [l for l in topo.links if l.delay_ns < 100]
    if tiny:
        rep.note(
            f"{len(tiny)} link(s) have a propagation delay below 100 ns "
            f"(min {fmt_ns(min(l.delay_ns for l in tiny))}); that is below any realistic cable, "
            "so those hops are effectively zero-latency."
        )


# --------------------------------------------------------------------------- #
# architecture inference
# --------------------------------------------------------------------------- #
def infer_architecture(topo: Topology, hops, bw):
    """Group hosts by the switch they attach to at each distinct link speed."""
    rates = sorted({l.rate_bps for l in topo.links}, reverse=True)
    # tier[rate] -> switch -> [hosts]
    tiers = {}
    for r in rates:
        by_sw = defaultdict(list)
        for l in topo.links:
            if l.rate_bps != r:
                continue
            h, s = (l.src, l.dst) if l.src in topo.hosts else (l.dst, l.src)
            if h in topo.hosts and s in topo.switches:
                by_sw[s].append(h)
        if by_sw:
            tiers[r] = {s: sorted(hs) for s, hs in sorted(by_sw.items())}

    # switches with no host attached = aggregation / spine
    host_facing = {s for t in tiers.values() for s in t}
    spine = sorted(topo.switches - host_facing)
    return rates, tiers, spine


def bandwidth_domains(topo: Topology, bw):
    """
    Partition hosts into cliques of mutually-highest-bandwidth reachability.
    These are the groups a collective can run in at full scale-up speed.
    """
    hosts = sorted(topo.hosts)
    if not bw:
        return []
    top = max(bw.values())
    parent = {h: h for h in hosts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), v in bw.items():
        if v == top:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups = defaultdict(list)
    for h in hosts:
        groups[find(h)].append(h)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def print_architecture(topo: Topology, hops, bw, rtt):
    rates, tiers, spine = infer_architecture(topo, hops, bw)
    hosts = sorted(topo.hosts)

    print(_c("  ARCHITECTURE", C_BOLD))
    print(f"    {len(hosts)} hosts (GPU ranks {fmt_ranks(hosts)}), "
          f"{len(topo.switches)} switches ({fmt_ranks(topo.switches)}), {len(topo.links)} links")

    for r in rates:
        if r not in tiers:
            continue
        tier = tiers[r]
        sizes = sorted({len(h) for h in tier.values()})
        label = "scale-up / NVSwitch-like" if r == rates[0] and len(rates) > 1 else "scale-out / NIC"
        if all(len(h) == 1 for h in tier.values()):
            label = "per-host leaf (1 GPU per switch — no aggregation)"
        print(f"\n    {_c(fmt_rate(r), C_BOLD)} tier — {label}")
        print(f"      {len(tier)} switch(es), {sum(len(h) for h in tier.values())} host ports, "
              f"domain sizes {sizes}")
        for s, hs in tier.items():
            if len(tier) <= 24:
                print(f"        switch {s:<3} <- hosts {fmt_ranks(hs)}   ({len(hs)} GPU"
                      f"{'s' if len(hs) != 1 else ''})")

    if spine:
        print(f"\n    Spine / aggregation (no hosts attached): {fmt_ranks(spine)}")
        for s in spine:
            print(f"        switch {s:<3} <- {len(topo.adj[s])} switch ports")

    # path classes
    print(f"\n    {_c('Path classes between GPU pairs', C_BOLD)}")
    classes = defaultdict(int)
    for (a, b), h in hops.items():
        if a < b:
            classes[(h, bw[(a, b)], round(rtt[(a, b)]))] += 1
    for (h, b, r), n in sorted(classes.items()):
        print(f"        {n:>4} pairs : {h} hops, bottleneck {fmt_rate(b):>8}, "
              f"base RTT {fmt_ns(r):>8}")


def print_comm_groups(topo: Topology, hops, bw):
    """Derive plausible collective groups from the fabric alone."""
    hosts = sorted(topo.hosts)
    domains = bandwidth_domains(topo, bw)

    print(f"\n  {_c('COLLECTIVE GROUPS SUPPORTED BY THIS FABRIC', C_BOLD)}")
    print(_c("    (inferred from the topology only; comm_groups.json is NOT read)", C_DIM))

    top = max(bw.values()) if bw else 0
    print(f"\n    {_c('Full-bandwidth domains', C_BOLD)} — all-to-all at {fmt_rate(top)}, "
          f"no oversubscribed hop:")
    for i, d in enumerate(domains):
        print(f"        domain {i}: ranks {fmt_ranks(d)}   ({len(d)} GPU"
              f"{'s' if len(d) != 1 else ''})")
    sizes = sorted({len(d) for d in domains})
    print(f"      -> a TENSOR-PARALLEL group must fit entirely inside one domain.")
    print(f"         Legal tp_size values here: {sorted({s for s in _divisors_of_all(sizes)})}")
    if len(sizes) > 1:
        print(_c(
            f"      !! domains are NOT uniform (sizes {sizes}). A single tp_size cannot tile them "
            f"evenly. This is fine ONLY if you intend different TP degrees per pool "
            f"(e.g. prefill vs decode).", C_YEL))

    # rank-aligned groups across domains (DP / PP style)
    if len(domains) > 1 and len({len(d) for d in domains}) == 1:
        k = len(domains[0])
        print(f"\n    {_c('Rank-aligned cross-domain groups', C_BOLD)} — one member per domain, "
              f"traverse the slow fabric:")
        for j in range(k):
            members = [d[j] for d in domains]
            b = min(bw[(members[0], m)] for m in members[1:])
            print(f"        group {j}: ranks {fmt_ranks(members)}   bottleneck {fmt_rate(b)}")
        print(f"      -> these are the natural DP all-reduce / PP send-recv groups: "
              f"{k} group(s) of {len(domains)}.")
    elif len(domains) > 1:
        print(f"\n    {_c('Cross-domain groups', C_BOLD)}: domains have unequal sizes, so no clean "
              f"rank-aligned grouping exists. Any DP/PP group spanning domains will cross the "
              f"{fmt_rate(min(bw.values()))} fabric.")

    # what a collective costs
    slow = min(bw.values()) if bw else 0
    if slow != top:
        print(f"\n    {_c('Cost asymmetry', C_BOLD)}: an intra-domain collective runs at "
              f"{fmt_rate(top)}; any collective that leaves a domain is capped at "
              f"{fmt_rate(slow)} — a {top/slow:.0f}x cliff. Placing a TP group across domains "
              f"would silently cost {top/slow:.0f}x more per all-reduce, with no error from ns-3.")


def _divisors_of_all(sizes):
    out = set()
    for n in range(1, max(sizes) + 1):
        if all(s % n == 0 for s in sizes):
            out.add(n)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def process(path: Path, args) -> bool:
    rel = path
    print()
    print(_c("=" * 78, C_DIM))
    print(_c(f" {rel}", C_BOLD))
    print(_c("=" * 78, C_DIM))

    rep = Report(path)
    topo = load_topology(path, rep)
    if topo is None:
        _print_findings(rep)
        return False

    validate_structure(topo, rep)

    hops = bw = rtt = None
    if topo.hosts and topo.links:
        hops, bw, delay, rtt, nexthops = calculate_routes(topo, args.payload)
        validate_routing(topo, hops, bw, nexthops, rep)
        validate_ns3_hazards(topo, bw, rtt, rep, args.buffer_size, args.headroom_factor, args.payload)

    if hops and bw and not args.quiet:
        print()
        print_architecture(topo, hops, bw, rtt)
        print_comm_groups(topo, hops, bw)

    print()
    _print_findings(rep)
    return rep.ok


def _print_findings(rep: Report) -> None:
    for m in rep.errors:
        print(f"  {_c('[ERROR]', C_RED)}   {m}")
    for m in rep.warnings:
        print(f"  {_c('[WARN]', C_YEL)}    {m}")
    for m in rep.notes:
        print(f"  {_c('[note]', C_DIM)}    {m}")
    print()
    if rep.ok and not rep.warnings:
        print(f"  {_c('PASS', C_GRN)} — topology is structurally valid and raises no ns-3 hazards.")
    elif rep.ok:
        print(f"  {_c('PASS with warnings', C_YEL)} — structurally valid; review the warnings above.")
    else:
        print(f"  {_c('FAIL', C_RED)} — {len(rep.errors)} error(s).")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate ns-3 / ASTRA-sim physical_topology.txt files in a folder."
    )
    ap.add_argument("folder", help="folder to scan (recursively) for topology files")
    ap.add_argument("--buffer-size", type=int, default=16,
                    help="switch BUFFER_SIZE in MB from your network config (default: 16)")
    ap.add_argument("--headroom-factor", type=int, default=3,
                    help="HEADROOM_FACTOR from common.h (default: 3)")
    ap.add_argument("--payload", type=int, default=1000,
                    help="PACKET_PAYLOAD_SIZE in bytes (default: 1000)")
    ap.add_argument("--quiet", action="store_true",
                    help="only print validation findings, skip the architecture report")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.name in TOPOLOGY_NAMES
    )
    if not files:
        print(f"error: no {' / '.join(TOPOLOGY_NAMES)} found under {root}", file=sys.stderr)
        return 2

    results = [process(p, args) for p in files]

    print(_c("=" * 78, C_DIM))
    passed = sum(results)
    print(f" {passed}/{len(results)} topolog{'y' if len(results)==1 else 'ies'} passed")
    print(_c("=" * 78, C_DIM))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())