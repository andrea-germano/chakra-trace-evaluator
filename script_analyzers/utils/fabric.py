#!/usr/bin/env python3
"""
utils.fabric — what the ns-3 switch actually does, reconstructed from the inputs.

This module is a deliberate mirror of the qos-impl sources. Every quantity here
has a named counterpart in C++, cited at the point of definition. Nothing in this
file reads simulation *output*: given only ``physical_topology.txt`` and
``config.txt`` it predicts how the fabric will behave. That is the point — it is
the pre-execution half of the analysis, and it is what makes the buffer sweep a
co-design tool rather than a measurement report.

Run it standalone to inspect the model before trusting any plot:

    python3 -m utils.fabric physical_topology.txt config.txt --buffers 2,4,8,16,32

The two quantities that are easy to confuse, and must never be:

    F_ports        number of INGRESS PORTS feeding the congested egress port.
                   PFC accounting is per (port, qIndex) -- see SwitchMmu::
                   UpdateIngressAdmission -- so this, and only this, is the
                   multiplier that converts the ingress threshold into an
                   egress-equivalent one. It lives in Bottleneck.ingress_ports
                   and is derived from the topology, never from flows.

    N_concurrent   number of FLOWS sharing the bottleneck at once. This sets the
                   fair-share slowdown (standalone_fct assumes a flow owns the
                   bottleneck, so N flows sharing it give slowdown ~ N). It is a
                   property of the workload, is measured from fct.txt, and lives
                   in the metrics layer -- not here.

Using N_concurrent where F_ports belongs produces an egress-equivalent threshold
several times larger than the physical buffer. ``egress_equivalent_threshold``
asserts against exactly that.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# --- ns-3 constants -------------------------------------------------------- #
RESERVE_BYTES = 4 * 1024        # SwitchMmu::reserve
DEFAULT_HEADROOM_FACTOR = 3     # common.h::headroom_factor
BASE_PFC_SHIFT = 3              # common.h: "uint32_t shift = 3; // by default 1/8"

_RATE_RE = re.compile(r"^\s*([\d.]+)\s*([kKmMgGtT]?)b(?:ps|/s)\s*$")
_TIME_RE = re.compile(r"^\s*([\d.]+)\s*(s|ms|us|ns)\s*$")
_RATE_MUL = {"": 1, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
_TIME_MUL = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}


def parse_rate(tok: str) -> int:
    """'1024Gbps' -> bits/s, as GetDataRate().GetBitRate() returns it."""
    m = _RATE_RE.match(tok)
    if not m:
        raise ValueError(f"cannot parse data rate {tok!r}")
    return int(float(m.group(1)) * _RATE_MUL[m.group(2).lower()])


def parse_delay_ns(tok: str) -> int:
    """'0.005ms' -> 5000. ns-3's default time resolution is ns."""
    m = _TIME_RE.match(tok)
    if not m:
        raise ValueError(f"cannot parse delay {tok!r}")
    return int(round(float(m.group(1)) * _TIME_MUL[m.group(2)]))


# --------------------------------------------------------------------------- #
# config.txt
# --------------------------------------------------------------------------- #
@dataclass
class Ns3Config:
    """The knobs that matter for the congestion regime. ConfigEcn multiplies the
    KMIN/KMAX map values by 1000 (decimal kB); ConfigBufferSize multiplies
    BUFFER_SIZE by 1024*1024 (MiB). Map keys are link rates in bit/s and must
    match GetBitRate() exactly or ns-3 NS_ASSERTs at startup."""
    path: Path | None = None
    buffer_mb: float | None = None
    cc_mode: int | None = None
    enable_qcn: int | None = None
    dynamic_pfc: int | None = None
    headroom_factor: int = DEFAULT_HEADROOM_FACTOR
    kmin: dict[int, int] = field(default_factory=dict)   # rate -> bytes
    kmax: dict[int, int] = field(default_factory=dict)
    pmax: dict[int, float] = field(default_factory=dict)

    @property
    def buffer_bytes(self) -> int | None:
        return int(self.buffer_mb * 1024 * 1024) if self.buffer_mb is not None else None

    def warnings(self) -> list[str]:
        out = []
        if self.cc_mode == 12:
            out.append("CC_MODE 12 has no handler in any RdmaHw/SwitchNode branch: "
                       "this is a PFC-only lossless fabric with NO rate-based CC. "
                       "It must be declared explicitly in the methodology.")
        if self.dynamic_pfc == 0:
            out.append("USE_DYNAMIC_PFC_THRESHOLD=0: the threshold model in this "
                       "module (buffer-proportional) does not apply.")
        if self.enable_qcn == 0:
            out.append("ENABLE_QCN=0: no ECN marking, so the ECN band is irrelevant "
                       "and every run is PFC-governed by construction.")
        return out


def parse_ns3_config(path: Path) -> Ns3Config:
    cfg = Ns3Config(path=path)
    for raw in path.read_text().splitlines():
        p = raw.split()
        if not p:
            continue
        key = p[0].upper()
        try:
            if key == "BUFFER_SIZE":
                cfg.buffer_mb = float(p[1])
            elif key == "CC_MODE":
                cfg.cc_mode = int(p[1])
            elif key == "ENABLE_QCN":
                cfg.enable_qcn = int(p[1])
            elif key == "USE_DYNAMIC_PFC_THRESHOLD":
                cfg.dynamic_pfc = int(p[1])
            elif key == "HEADROOM_FACTOR":
                cfg.headroom_factor = int(p[1])
            elif key in ("KMIN_MAP", "KMAX_MAP", "PMAX_MAP"):
                tgt = {"KMIN_MAP": cfg.kmin, "KMAX_MAP": cfg.kmax, "PMAX_MAP": cfg.pmax}[key]
                for i in range(int(p[1])):
                    rate, val = int(p[2 + 2 * i]), p[3 + 2 * i]
                    tgt[rate] = float(val) if key == "PMAX_MAP" else int(float(val) * 1000)
        except (IndexError, ValueError):
            # A template with __PLACEHOLDERS__ lands here. Tolerated, but the
            # caller is told, because a config with no BUFFER_SIZE is not a config.
            pass
    return cfg


# --------------------------------------------------------------------------- #
# physical_topology.txt
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Port:
    peer: int
    rate: int          # bits/s
    delay_ns: int


@dataclass
class Topology:
    """physical_topology.txt:
           <node_num> <switch_num> <link_num>
           <switch ids...>
           <src> <dst> <rate> <delay> <error_rate>   x link_num

    Device indices follow file order (device 0 is the loopback added by
    InternetStackHelper), exactly as the qbb.Install() loop in common.h assigns
    them. This matters: in a topology where a host's direct peer link is listed
    before its leaf link, the host's device facing the leaf is ifindex 2, not 1 --
    and that is the ifindex that appears in pfc.txt."""
    n_nodes: int
    switches: frozenset[int]
    ports: dict[int, dict[int, Port]]
    headroom_factor: int = DEFAULT_HEADROOM_FACTOR

    # derived
    nic_rate: int = 0
    hdrm: dict[int, dict[int, int]] = field(default_factory=dict)
    total_hdrm: dict[int, int] = field(default_factory=dict)
    total_rsrv: dict[int, int] = field(default_factory=dict)
    shift: dict[int, dict[int, int]] = field(default_factory=dict)
    pair_bw: dict[tuple[int, int], int] = field(default_factory=dict)
    dist: dict[int, dict[int, int]] = field(default_factory=dict)
    next_hop: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    ecmp_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def hosts(self) -> list[int]:
        return [i for i in range(self.n_nodes) if i not in self.switches]

    def is_switch(self, n: int) -> bool:
        return n in self.switches

    def port_facing(self, node: int, peer: int) -> int | None:
        for idx, p in self.ports.get(node, {}).items():
            if p.peer == peer:
                return idx
        return None

    def port_label(self, node: int, ifidx: int) -> str:
        """'p3->sw12' / 'p2->h0'. The ifindex alone is meaningless to a reader:
        it follows link-file order, so a host's port toward its leaf is not
        necessarily 1."""
        port = self.ports.get(node, {}).get(ifidx)
        if port is None:
            return f"p{ifidx}"
        return f"p{ifidx}->{'sw' if self.is_switch(port.peer) else 'h'}{port.peer}"

    def path(self, src: int, dst: int) -> list[tuple[int, int]] | None:
        """Directed links on the src->dst shortest path, or None if ambiguous
        (ECMP: SetRoutingEntries installs every equal-cost next hop and the
        runtime choice is a hash, so no path is reconstructible) or unreachable."""
        cur, out = src, []
        for _ in range(self.n_nodes + 1):
            if cur == dst:
                return out
            hops = self.next_hop.get(cur, {}).get(dst, [])
            if len(hops) != 1:
                return None
            out.append((cur, hops[0]))
            cur = hops[0]
        return None


def parse_topology(path: Path, headroom_factor: int = DEFAULT_HEADROOM_FACTOR) -> Topology:
    toks = iter(path.read_text().split())
    n_nodes, n_sw, n_link = int(next(toks)), int(next(toks)), int(next(toks))
    switches = frozenset(int(next(toks)) for _ in range(n_sw))

    ports: dict[int, dict[int, Port]] = defaultdict(dict)
    next_if: dict[int, int] = defaultdict(lambda: 1)     # 0 = loopback
    for _ in range(n_link):
        a, b = int(next(toks)), int(next(toks))
        rate, delay = parse_rate(next(toks)), parse_delay_ns(next(toks))
        next(toks)                                        # error_rate, unused
        ports[a][next_if[a]] = Port(b, rate, delay)
        ports[b][next_if[b]] = Port(a, rate, delay)
        next_if[a] += 1
        next_if[b] += 1

    topo = Topology(n_nodes=n_nodes, switches=switches, ports=dict(ports),
                    headroom_factor=headroom_factor)
    _derive_switch_state(topo)
    _derive_routes(topo)
    return topo


def _derive_switch_state(topo: Topology) -> None:
    # common.h::get_nic_rate walks the NodeContainer in id order and returns the
    # rate of device 1 of the FIRST host -- i.e. of the first link in the file
    # that mentions the lowest-id host. Fragile: it depends on line order.
    first_host = next((i for i in range(topo.n_nodes) if i not in topo.switches), None)
    topo.nic_rate = topo.ports[first_host][1].rate if first_host is not None else 0

    for sw in topo.switches:
        pmap = topo.ports.get(sw, {})
        topo.hdrm[sw], topo.shift[sw] = {}, {}
        for idx, port in pmap.items():
            # common.h: rate * delay / 8 / 1000000000 * headroom_factor, in uint64
            # integer arithmetic -- the truncation is replicated deliberately.
            topo.hdrm[sw][idx] = ((port.rate * port.delay_ns) // 8 // 1_000_000_000
                                  * topo.headroom_factor)
            # common.h: shift starts at 3 and is decremented while rate > nic_rate
            s, r = BASE_PFC_SHIFT, port.rate
            while r > topo.nic_rate and s > 0:
                s -= 1
                r //= 2
            topo.shift[sw][idx] = s
        # ConfigNPort(GetNDevices()-1)
        topo.total_hdrm[sw] = sum(topo.hdrm[sw].values())
        topo.total_rsrv[sw] = len(pmap) * RESERVE_BYTES


def _derive_routes(topo: Topology) -> None:
    """common.h::CalculateRoute: BFS from each host, expanding switches only, with
    bw[] carrying the min rate along the path. That bw is entry.h's pairBw, i.e.
    the denominator of standalone_fct."""
    INF = 1 << 62
    for host in topo.hosts:
        dis, bw = {host: 0}, {host: INF}
        q = deque([host])
        while q:
            now = q.popleft()
            for port in topo.ports.get(now, {}).values():
                nxt = port.peer
                if nxt not in dis:
                    dis[nxt] = dis[now] + 1
                    bw[nxt] = min(bw[now], port.rate)
                    if topo.is_switch(nxt):
                        q.append(nxt)
                if dis[nxt] == dis[now] + 1:
                    hops = topo.next_hop.setdefault(nxt, {}).setdefault(host, [])
                    if now not in hops:
                        hops.append(now)
        for node, d in dis.items():
            topo.dist.setdefault(node, {})[host] = d
            topo.pair_bw[(node, host)] = bw[node]

    for node, per_host in topo.next_hop.items():
        for host, hops in per_host.items():
            if len(hops) > 1 and topo.is_switch(node):
                topo.ecmp_pairs.append((node, host))


# --------------------------------------------------------------------------- #
# The bottleneck and its physics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bottleneck:
    """One congested directed link, plus the ingress ports that feed it.

    ``ingress_ports`` is the F_ports of the module docstring. It is a tuple of
    port indices on ``switch`` -- so len() of it is, by construction, a port
    count and cannot silently become a flow count."""
    switch: int
    egress_port: int
    peer: int
    rate: int
    ingress_ports: tuple[int, ...] = ()

    @property
    def f_ports(self) -> int:
        return len(self.ingress_ports)

    def __str__(self) -> str:
        return f"{self.switch}->{self.peer}"

    def pause_victims(self, topo: Topology) -> tuple[tuple[int, int], ...]:
        """(node, ifindex) of the devices that get PAUSEd when this bottleneck's
        ingress ports fill. QbbNetDevice::Receive fires m_tracePfc on the device
        that RECEIVES the pause frame -- the victim -- so these, not the switch's
        own ports, are the keys to look for in pfc.txt."""
        out = []
        for p in self.ingress_ports:
            up = topo.ports[self.switch][p].peer
            vp = topo.port_facing(up, self.switch)
            if vp is not None:
                out.append((up, vp))
        return tuple(out)


class FabricModel:
    """Predicts the congestion regime of one bottleneck at a given buffer size.

    Everything below is a function of topology + config only. No simulation
    output is involved, which is what lets the prediction be falsified by the
    simulation rather than fitted to it."""

    def __init__(self, topo: Topology, cfg: Ns3Config):
        self.topo, self.cfg = topo, cfg

    # -- PFC ---------------------------------------------------------------- #
    def pfc_threshold(self, bn: Bottleneck, buffer_bytes: int,
                      shared_used: int = 0) -> int:
        """SwitchMmu::GetPfcThreshold
               (buffer_size - total_hdrm - total_rsrv - shared_used_bytes) >> pfc_a_shift

        Called with shared_used=0 this is the MAXIMUM: the real threshold shrinks
        as the shared pool fills, so PFC is entered earlier than predicted here.
        Note that ``buffer/8`` is not an acceptable approximation: on a leaf whose
        host ports are fast (large rate*delay headroom) the subtracted terms can
        be ~half the buffer at small sizes, exactly where the regime flips."""
        sw = bn.switch
        avail = buffer_bytes - self.topo.total_hdrm[sw] - self.topo.total_rsrv[sw] - shared_used
        return max(0, int(avail) >> self.topo.shift[sw][bn.egress_port])

    def egress_equivalent_threshold(self, bn: Bottleneck, buffer_bytes: int) -> int:
        """PFC watches per-ingress-port occupancy; ECN watches per-egress-port
        occupancy. The same bytes sit on F_ports ingress counters but on one
        egress counter, so comparing the threshold against KMIN/KMAX requires
        multiplying by the number of ingress ports."""
        eq = self.pfc_threshold(bn, buffer_bytes) * bn.f_ports
        if eq > buffer_bytes:
            raise ValueError(
                f"egress-equivalent threshold ({eq/1e6:.1f} MB) exceeds the physical "
                f"buffer ({buffer_bytes/1e6:.1f} MB). The threshold is carved out of "
                f"the buffer, so this is impossible: f_ports={bn.f_ports} is not a "
                f"port count.")
        return eq

    def pfc_egress_ceiling(self, bn: Bottleneck, buffer_bytes: int) -> int:
        """Where the egress queue settles once PFC has paused every ingress port:
        each one still holds reserve + threshold + its own headroom (packets
        already in flight when the pause was sent land in hdrm_bytes).

        This is the decisive regime test. Measured peak egress at the ceiling
        means the queue is held by backpressure; below it, the rate control is
        what limits. Note the ceiling can exceed KMAX on its own when the ingress
        headroom is large -- in which case a PFC-only regime is unreachable and
        ECN marks at every buffer size."""
        thr = self.pfc_threshold(bn, buffer_bytes)
        return sum(RESERVE_BYTES + thr + self.topo.hdrm[bn.switch][p]
                   for p in bn.ingress_ports)

    # -- ECN ---------------------------------------------------------------- #
    def ecn_band(self, bn: Bottleneck) -> tuple[int | None, int | None]:
        """SwitchMmu::ShouldSendCN tests egress_bytes[ifindex][qIndex] against
        kmin/kmax[ifindex], keyed on that port's own link rate."""
        return self.cfg.kmin.get(bn.rate), self.cfg.kmax.get(bn.rate)

    # -- verdicts ----------------------------------------------------------- #
    def regime(self, bn: Bottleneck, buffer_bytes: int) -> str:
        kmin, kmax = self.ecn_band(bn)
        if kmin is None or kmax is None or not bn.f_ports:
            return "?"
        eq = self.egress_equivalent_threshold(bn, buffer_bytes)
        if eq < kmin:
            return "PFC"        # PFC fires before ECN even starts marking
        if eq > kmax:
            return "DCQCN"      # ECN saturates long before PFC can trigger
        return "MIXED"

    def flip_band(self, bn: Bottleneck) -> tuple[float, float] | None:
        """Buffers (MiB) where F_ports*threshold crosses KMIN and KMAX:
               F * ((B - hdrm - rsrv) >> shift) == K
           ->  B = K * 2^shift / F + hdrm + rsrv
        Below the KMIN crossing the fabric is PFC-dominated, above the KMAX one
        it is ECN-dominated, and in between mixed. A band, not a line."""
        kmin, kmax = self.ecn_band(bn)
        if kmin is None or kmax is None or not bn.f_ports:
            return None
        base = self.topo.total_hdrm[bn.switch] + self.topo.total_rsrv[bn.switch]
        mul = 1 << self.topo.shift[bn.switch][bn.egress_port]
        return ((kmin * mul / bn.f_ports + base) / 2**20,
                (kmax * mul / bn.f_ports + base) / 2**20)


# --------------------------------------------------------------------------- #
# Standalone inspection
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topology")
    ap.add_argument("config", nargs="?")
    ap.add_argument("--buffers", default="2,4,8,16,32", help="MiB, comma separated")
    ap.add_argument("--bottleneck", help="'sw->peer', e.g. '8->12'. Default: the "
                                        "most oversubscribed switch egress link.")
    ap.add_argument("--headroom-factor", type=int, default=DEFAULT_HEADROOM_FACTOR)
    a = ap.parse_args(argv)

    topo = parse_topology(Path(a.topology), a.headroom_factor)
    cfg = parse_ns3_config(Path(a.config)) if a.config else Ns3Config()
    for w in cfg.warnings():
        print(f"WARNING: {w}\n", file=sys.stderr)

    print(f"{len(topo.hosts)} hosts, {len(topo.switches)} switches, "
          f"nic_rate = {topo.nic_rate/1e9:g} Gbps "
          f"(device 1 of host {topo.hosts[0]} -> first link in the file naming it)")
    if topo.ecmp_pairs:
        print(f"ECMP ties on {len(topo.ecmp_pairs)} (node, host) pairs: runtime paths "
              f"are hash-chosen and not reconstructible.")
    print("\nHost ports (the ifindex in pfc.txt follows file order, not intuition):")
    for h in topo.hosts:
        print(f"  host {h}: " + "  ".join(
            f"if{i}->n{p.peer} @{p.rate/1e9:g}G" for i, p in sorted(topo.ports[h].items())))

    # pick the bottleneck: switch egress link with the worst in:out rate ratio
    if a.bottleneck:
        sw, peer = (int(x) for x in a.bottleneck.split("->"))
    else:
        cand = []
        for sw in topo.switches:
            for idx, port in topo.ports[sw].items():
                other = sum(p.rate for i, p in topo.ports[sw].items() if i != idx)
                cand.append((other / port.rate, sw, port.peer))
        _, sw, peer = max(cand)
    egress = topo.port_facing(sw, peer)
    ingress = tuple(i for i in topo.ports[sw] if i != egress)
    bn = Bottleneck(sw, egress, peer, topo.ports[sw][egress].rate, ingress)
    model = FabricModel(topo, cfg)

    print(f"\nBottleneck {bn}: egress if{bn.egress_port} @{bn.rate/1e9:g} Gbps, "
          f"F_ports = {bn.f_ports} (ingress if{list(bn.ingress_ports)})")
    print(f"  PAUSE victims in pfc.txt: {list(bn.pause_victims(topo))}")
    print(f"  total_hdrm = {topo.total_hdrm[sw]:,} B, total_rsrv = {topo.total_rsrv[sw]:,} B, "
          f"pfc_a_shift = {topo.shift[sw][egress]} (divide by {1<<topo.shift[sw][egress]})")
    kmin, kmax = model.ecn_band(bn)
    print(f"  ECN band at {bn.rate/1e9:g} Gbps: KMIN = {kmin}, KMAX = {kmax}")
    band = model.flip_band(bn)
    if band:
        print(f"  predicted flip band: PFC below {band[0]:.2f} MiB, "
              f"DCQCN above {band[1]:.2f} MiB")

    print(f"\n{'buf MiB':>8} {'threshold':>11} {'x F_ports':>11} {'buffer/8':>11} "
          f"{'PFC ceiling':>12} {'hdrm+rsrv':>10} {'regime':>7}")
    for b in (float(x) for x in a.buffers.split(",")):
        B = int(b * 1024 * 1024)
        thr = model.pfc_threshold(bn, B)
        eq = model.egress_equivalent_threshold(bn, B)
        ceil = model.pfc_egress_ceiling(bn, B)
        pct = 100 * (topo.total_hdrm[sw] + topo.total_rsrv[sw]) / B
        print(f"{b:>8g} {thr:>11,} {eq:>11,} {B//8:>11,} {ceil:>12,} "
              f"{pct:>9.1f}% {model.regime(bn, B):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())