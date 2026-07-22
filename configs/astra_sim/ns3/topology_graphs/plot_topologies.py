#!/usr/bin/env python3
"""
plot_topologies.py
==================

Render a graphical representation (PNG) of each network topology described in
the ns-3 / ASTRA-sim "physical topology" file format.

Input file format
------------------
    line 1 : <total_nodes> <num_switches> <num_links>
    line 2 : <switch_id_1> <switch_id_2> ... (num_switches values)
    line N : <src> <dst> <bandwidth> <latency> <error_rate>

The node ids listed on line 2 are *switches*; every other id in
[0, total_nodes) is a *host* (compute node / GPU).  The bandwidth may be a
numeric value (e.g. "4800Gbps") or a symbol (e.g. "Bx"); in the latter case it
is reported verbatim.

Layout
------
The graph is drawn with a hierarchical, spine-leaf style layout (Graphviz
"dot"), the way real datacenter/network diagrams are usually presented:

    * the spine / core switches sit at the top,
    * per-host (scale-out) switches sit in the middle,
    * hosts sit below them,
    * intra-node "scale-up" switches (those wired only to hosts through
      high-bandwidth, e.g. 4800Gbps, links) are pushed to the very bottom so
      they do not clutter the middle of the picture.

Usage
-----
    # process every .txt in a directory, writing PNGs into ./<out>/
    python3 plot_topologies.py [input_dir] [output_dir]

    # or a single file
    python3 plot_topologies.py physical_topology_T1.txt out/

With no arguments it reads the files from "../topology_templates" and writes
the PNGs into the current directory.

The script is intentionally generic: it works with any topology that follows
the format above, regardless of the number of nodes/switches.
"""

from __future__ import annotations

import os
import sys
import glob

import networkx as nx
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------- #
# Bandwidth helpers
# --------------------------------------------------------------------------- #
def bandwidth_to_gbps(bw):
    """Convert a bandwidth string to a value in Gbps, or None if symbolic.

    Recognizes the suffixes T/G/M (Tbps/Gbps/Mbps). A symbolic bandwidth such
    as "Bx" returns None.
    """
    s = bw.lower().replace("bps", "")
    mult = 1.0
    if s.endswith("t"):
        mult, s = 1000.0, s[:-1]
    elif s.endswith("g"):
        mult, s = 1.0, s[:-1]
    elif s.endswith("m"):
        mult, s = 0.001, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None  # symbolic (e.g. "Bx")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
class Topology:
    """In-memory representation of a topology."""

    def __init__(self, name):
        self.name = name
        self.num_nodes = 0
        self.num_switches = 0
        self.num_links = 0
        self.switches = set()      # switch ids
        self.hosts = set()         # host ids
        self.links = []            # list of (src, dst, bandwidth, latency)

    @property
    def num_hosts(self):
        return len(self.hosts)


def parse_topology(path):
    """Read a topology file and return a Topology object."""
    name = os.path.splitext(os.path.basename(path))[0]
    topo = Topology(name)

    with open(path) as fh:
        # keep only non-empty, non-comment lines
        lines = [ln.strip() for ln in fh
                 if ln.strip() and not ln.lstrip().startswith("#")]

    if len(lines) < 2:
        raise ValueError(f"File '{path}' is too short to be a valid topology")

    header = lines[0].split()
    topo.num_nodes = int(header[0])
    topo.num_switches = int(header[1])
    topo.num_links = int(header[2]) if len(header) > 2 else 0

    topo.switches = set(int(x) for x in lines[1].split())
    topo.hosts = set(range(topo.num_nodes)) - topo.switches

    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        src, dst = int(parts[0]), int(parts[1])
        bw = parts[2]
        lat = parts[3] if len(parts) > 3 else ""
        topo.links.append((src, dst, bw, lat))

    return topo


def classify_bottom_switches(topo):
    """Return the set of "intra-node" (scale-up) switches to place at the bottom.

    These are switches whose neighbors are *only* hosts (no switch-to-switch
    link) and that carry at least one numeric high-bandwidth link (e.g.
    4800Gbps).  They represent the intra-node scale-up domain and are pushed
    below the hosts to keep the middle of the diagram uncluttered.

    A switch wired to hosts only through symbolic ("Bx") links is *not*
    considered a bottom switch (e.g. the single core switch in a star
    topology), so this rule never sinks the actual core of a topology.
    """
    neighbors = {s: set() for s in topo.switches}
    has_numeric = {s: False for s in topo.switches}
    for src, dst, bw, lat in topo.links:
        for a, b in ((src, dst), (dst, src)):
            if a in topo.switches:
                neighbors[a].add(b)
                if bandwidth_to_gbps(bw) is not None:
                    has_numeric[a] = True

    bottom = set()
    for s in topo.switches:
        only_hosts = neighbors[s] and neighbors[s].issubset(topo.hosts)
        if only_hosts and has_numeric[s]:
            bottom.add(s)
    return bottom


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
HOST_COLOR = "#f4a259"        # orange -> hosts / compute nodes
HOST_EDGE = "#b5651d"
SWITCH_COLOR = "#4a80bd"      # blue -> switches
SWITCH_EDGE = "#22456e"

# palette used to distinguish bandwidth classes (edge color + width)
_BW_PALETTE = [
    "#2a2a2a", "#c0392b", "#27ae60", "#8e44ad",
    "#d35400", "#16a085", "#2980b9", "#7f8c8d",
]


def bandwidth_style(topo):
    """Assign a color and a width to every bandwidth value.

    Higher numeric bandwidths -> thicker edges; symbolic values (e.g. Bx) get
    an intermediate width.  Returns dict bandwidth -> (color, width).
    """
    uniq = list(dict.fromkeys(bw for *_, bw, _ in topo.links))
    # numeric first (ascending), then symbolic
    numeric = sorted((b for b in uniq if bandwidth_to_gbps(b) is not None),
                     key=bandwidth_to_gbps)
    symbolic = [b for b in uniq if bandwidth_to_gbps(b) is None]
    ordered = numeric + symbolic

    style = {}
    n = max(len(ordered), 1)
    for i, bw in enumerate(ordered):
        color = _BW_PALETTE[i % len(_BW_PALETTE)]
        width = 1.4 + 2.2 * (i / max(n - 1, 1))  # width between 1.4 and 3.6
        style[bw] = (color, width)
    return style


# --------------------------------------------------------------------------- #
# Hierarchical layout via Graphviz (dot)
# --------------------------------------------------------------------------- #
def compute_layout(topo, bottom_switches):
    """Compute node positions with a hierarchical (spine-leaf) layout.

    Uses Graphviz "dot" when available (through pydot); otherwise it falls back
    to a networkx force-directed layout.

    Edge directions are chosen so that, with rankdir=BT:
        * hosts rank above their intra-node scale-up switches (bottom),
        * per-host/leaf switches rank above the hosts,
        * spine/core switches end up at the top.
    """
    G = nx.DiGraph()
    # graph-level attributes for Graphviz "dot":
    # rankdir=BT -> sources at the bottom, core (spine) at the top
    G.graph["graph"] = {"rankdir": "BT", "nodesep": "0.6", "ranksep": "1.4"}
    G.add_nodes_from(topo.hosts)
    G.add_nodes_from(topo.switches)

    for src, dst, bw, lat in topo.links:
        both_hosts = (src in topo.hosts) and (dst in topo.hosts)
        if both_hosts:
            # host<->host "scale-up" links must not drive the rank hierarchy
            G.add_edge(src, dst, constraint="false")
            continue

        a, b = src, dst
        # a bottom (intra-node) switch must be the tail of the edge so it sinks
        # *below* the host it connects to
        if b in bottom_switches and a in topo.hosts:
            a, b = b, a
        G.add_edge(a, b)

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog="dot", root=None)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[warn] Graphviz unavailable ({exc}); "
                         f"falling back to a force-directed layout.\n")
        UG = nx.Graph()
        UG.add_nodes_from(G.nodes())
        UG.add_edges_from((u, v) for u, v in G.edges())
        return nx.spring_layout(UG, seed=42, k=1.5, iterations=200)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_topology(topo, out_path):
    bottom_switches = classify_bottom_switches(topo)
    pos = compute_layout(topo, bottom_switches)
    bw_style = bandwidth_style(topo)

    # figure size scales with the number of nodes
    n = topo.num_nodes
    fig_w = max(9.0, min(28.0, 1.05 * n ** 0.5 * 2.2))
    fig_h = max(6.5, min(20.0, 0.85 * n ** 0.5 * 2.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # undirected graph used only for drawing
    G = nx.Graph()
    G.add_nodes_from(topo.hosts)
    G.add_nodes_from(topo.switches)

    # marker sizes and font size adapt to the number of nodes
    host_size = max(300, 1500 - 12 * n)
    switch_size = max(500, 2200 - 15 * n)
    font_size = max(6, min(11, int(150 / max(n ** 0.5, 1))))

    # --- links ------------------------------------------------------------ #
    for src, dst, bw, lat in topo.links:
        color, width = bw_style.get(bw, ("#555555", 1.6))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(src, dst)], ax=ax,
            edge_color=color, width=width, alpha=0.75,
        )

    # Bandwidth labels on the links. To avoid an unreadable tangle, labels with
    # the *same* text that fall too close together are collapsed into a single
    # one (the bandwidth is anyway encoded by the link color and width, shown in
    # the legend). Each "bundle" of same-bandwidth links thus shows one clean
    # label.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    min_dist = 0.06 * diag if diag > 0 else 1.0

    # each bandwidth class gets a different position along the link (between
    # 0.30 and 0.70) so labels of different bandwidths do not overlap
    bw_index = {bw: i for i, bw in enumerate(bw_style)}
    n_bw = max(len(bw_style), 1)

    def _frac(bw):
        if n_bw == 1:
            return 0.5
        return 0.30 + 0.40 * (bw_index.get(bw, 0) / (n_bw - 1))

    placed = []  # (x, y, text) already drawn
    for src, dst, bw, lat in topo.links:
        # host<->host links (usually horizontal): center the label; the
        # per-class offset is only useful for vertical links
        host_to_host = (src in topo.hosts) and (dst in topo.hosts)
        f = 0.5 if host_to_host else _frac(bw)
        mx = pos[src][0] + f * (pos[dst][0] - pos[src][0])
        my = pos[src][1] + f * (pos[dst][1] - pos[src][1])
        # skip if an identical label already exists nearby
        if any(t == bw and (px - mx) ** 2 + (py - my) ** 2 < min_dist ** 2
               for px, py, t in placed):
            continue
        placed.append((mx, my, bw))
        ax.text(mx, my, bw, fontsize=max(5, font_size - 2), color="#333333",
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.8))

    # --- nodes ------------------------------------------------------------ #
    nx.draw_networkx_nodes(
        G, pos, nodelist=sorted(topo.hosts), ax=ax,
        node_color=HOST_COLOR, edgecolors=HOST_EDGE, linewidths=1.5,
        node_size=host_size, node_shape="o",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=sorted(topo.switches), ax=ax,
        node_color=SWITCH_COLOR, edgecolors=SWITCH_EDGE, linewidths=1.8,
        node_size=switch_size, node_shape="s",
    )

    # node labels
    host_labels = {h: str(h) for h in topo.hosts}
    switch_labels = {s: str(s) for s in topo.switches}
    nx.draw_networkx_labels(G, pos, labels=host_labels, ax=ax,
                            font_size=font_size, font_color="black")
    nx.draw_networkx_labels(G, pos, labels=switch_labels, ax=ax,
                            font_size=font_size, font_color="white",
                            font_weight="bold")

    # --- legend / title --------------------------------------------------- #
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Compute node (host)",
               markerfacecolor=HOST_COLOR, markeredgecolor=HOST_EDGE,
               markersize=13, markeredgewidth=1.5),
        Line2D([0], [0], marker="s", color="w", label="Switch",
               markerfacecolor=SWITCH_COLOR, markeredgecolor=SWITCH_EDGE,
               markersize=13, markeredgewidth=1.8),
    ]
    # one legend entry per bandwidth class
    for bw, (color, width) in bw_style.items():
        legend_handles.append(
            Line2D([0], [0], color=color, lw=max(width, 2.0),
                   label=f"Bandwidth: {bw}")
        )

    leg = ax.legend(handles=legend_handles, loc="upper left",
                    fontsize=9, frameon=True, framealpha=0.95,
                    borderpad=0.8, labelspacing=0.6)
    leg.get_frame().set_edgecolor("#cccccc")

    title = (f"{topo.name}\n"
             f"{topo.num_hosts} nodes  •  {topo.num_switches} switches  "
             f"•  {len(topo.links)} links")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    ax.axis("off")
    ax.margins(0.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.normpath(os.path.join(here, "..", "topology_templates"))

    in_arg = argv[1] if len(argv) > 1 else default_in
    out_dir = argv[2] if len(argv) > 2 else here

    os.makedirs(out_dir, exist_ok=True)

    # collect the files to process
    if os.path.isdir(in_arg):
        files = sorted(glob.glob(os.path.join(in_arg, "*.txt")))
    else:
        files = [in_arg]

    if not files:
        sys.stderr.write(f"No topology file found in '{in_arg}'\n")
        return 1

    print(f"Found {len(files)} topology file(s). Output in: {out_dir}\n")
    for path in files:
        try:
            topo = parse_topology(path)
        except Exception as exc:
            sys.stderr.write(f"[error] {path}: {exc}\n")
            continue
        out_png = os.path.join(out_dir, topo.name + ".png")
        draw_topology(topo, out_png)
        print(f"  ok {topo.name}: "
              f"{topo.num_hosts} hosts, {topo.num_switches} switches, "
              f"{len(topo.links)} links  ->  {os.path.basename(out_png)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
