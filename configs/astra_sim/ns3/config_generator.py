from pathlib import Path
import json
from datetime import datetime

# Determina la directory dove si trova questo script
SCRIPT_DIR = Path(__file__).parent

# edit once (or just press enter at the prompts to keep these)
TEMPLATE  = SCRIPT_DIR / "config.template.txt"
CONF_DIR  = SCRIPT_DIR
OUT_DIR   = "/home/andre/tesi/trace_evaluator/output/ns3"

# cc name -> CC_MODE  (ideal = window-only, NO congestion; use a real CC otherwise)
CC_MODE = {"dcqcn": 1, "hpcc": 3, "timely": 7, "dctcp": 8, "hpcc-pint": 10, "none": 12}

# Bandwidths covered by the ECN maps in the template (Gbps). The script only
# *warns* if you pick something outside this set, since you write the physical
# topology by hand; it does not block you.
KNOWN_BX = {"400", "200", "100", "50", "40", "25"}

# Defaults that, when unchanged, are omitted from the tag to keep names short.
BUFFER_DEFAULT = "16"


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def ask_topo() -> str:
    """Free-text topology label. Whatever you type becomes the first token of
    the tag. Variant suffixes (e.g. oversubscription, rail alignment) are just
    part of the string you write, so encode them as you like, e.g. 2c-os2."""
    return ask("topology label (free text, e.g. 2a / 2c-os2 / single-switch)", "2a")


def build_fabric_tag(topo_token: str, bx: str, cc: str, buf: str) -> str:
    parts = [topo_token, f"bx{bx}", cc]
    parts.append(f"buf{buf}")
    return "_".join(parts)


def main() -> int:
    # --- experiment parameters (these decide the tag) ---
    subdir = ask("subdirectory under ns3 default directory (leave empty for none)", "")
    topo_token = ask_topo()

    num_nodes = ask("number of NPUs (compute nodes, = number of .et files)", "8")
    while not (num_nodes.isdigit() and int(num_nodes) > 0):
        num_nodes = ask("  must be a positive integer", "8")

    bx = ask(f"fabric bandwidth Bx in Gbps {sorted(KNOWN_BX, key=int, reverse=True)}", "50")
    if bx not in KNOWN_BX:
        print(f"  ! warning: Bx={bx}Gbps is not in the template ECN maps {sorted(KNOWN_BX, key=int)}.")
        print(f"    ns-3 will crash if a link with this speed appears in physical_topology.txt")
        print(f"    unless you add it to KMAX/KMIN/PMAX in the template first.")

    cc = ask(f"cc {list(CC_MODE)}", "dcqcn")
    while cc not in CC_MODE:
        cc = ask(f"  not valid, choose one of {list(CC_MODE)}", "dcqcn")

    buffer_size = ask("buffer size", BUFFER_DEFAULT)

    # --- instrumentation (NOT part of the tag; recorded in the manifest) ---
    enable_trace = ask("enable packet trace (0/1)", "0")
    qlen_start   = ask("qlen monitor start", "0")
    qlen_end     = ask("qlen monitor end", "0")

    # tag is derived, not typed
    tag = build_fabric_tag(topo_token, bx, cc, buffer_size)
    print(f"\n==> derived tag: {tag}")

    # path segment used by __CONF_DIR__/__TAG__ (TOPOLOGY_FILE, pointing at this
    # config's own folder); must include subdir when set so it matches the
    # folder actually created below.
    tag_path = f"{subdir}/{tag}" if subdir else tag

    # NOTE: __RUN_DIR__ (in the *_OUTPUT_FILE/QLEN_MON_FILE lines) is
    # deliberately NOT resolved here. A fabric config is model-agnostic and
    # reused across models/experiments; which run is using it is only known at
    # launch time, so generate_log_ns3.sh substitutes __RUN_DIR__ with the
    # exact output-dir name it was given, right before the ns-3 run. This makes
    # ns-3 outputs at output/ns3/<output_dir_name> mirror
    # output/astra_logs/<output_dir_name> exactly, whatever <output_dir_name>
    # is chosen at launch time and regardless of this fabric config's own
    # folder location. Do not add __RUN_DIR__ to `repl`.
    repl = {
        "__CONF_DIR__":     str(CONF_DIR),
        "__OUT_DIR__":      str(OUT_DIR),
        "__TAG__":          tag_path,
        "__CC_MODE__":      str(CC_MODE[cc]),
        "__BUFFER_SIZE__":  buffer_size,
        "__ENABLE_TRACE__": enable_trace,
        "__QLEN_START__":   qlen_start,
        "__QLEN_END__":     qlen_end,
    }

    txt = Path(TEMPLATE).read_text()
    for k, v in repl.items():
        txt = txt.replace(k, v)

    folder = Path(CONF_DIR) / subdir / tag if subdir else Path(CONF_DIR) / tag
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "config.txt"
    out.write_text(txt)

    # logical topology: always a flat 1-D dimension holding all NPUs. The real
    # collective structure comes from comm_groups.json (ASTRA-sim rebuilds a 1-D
    # ring over the involved ranks), so logical-dims only has to satisfy the two
    # formal constraints from the README: product of dims == N, and number of
    # dims == length of the *-implementation lists in the 1-D system config.
    logical = {"logical-dims": [num_nodes]}
    (folder / "logical_topology.json").write_text(json.dumps(logical, indent=2))

    # manifest: the authoritative, complete record of this run's fabric config.
    # The folder name is the human handle; this file is the source of truth.
    manifest = {
        "tag": tag,
        "topo": topo_token,
        "num_nodes": int(num_nodes),
        "bx_gbps": bx,
        "cc": cc,
        "cc_mode": CC_MODE[cc],
        "buffer_size": buffer_size,
        "enable_trace": enable_trace,
        "qlen_start": qlen_start,
        "qlen_end": qlen_end,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"written {out}")
    print(f"written {folder / 'logical_topology.json'}  -> logical-dims [{num_nodes}]")
    print(f"written {folder / 'manifest.json'}")
    print(f"now drop your physical_topology.txt into {folder}")
    print(f"  reminder: physical_topology.txt must have exactly {num_nodes} compute nodes,")
    print(f"            matching the number of .et files MLSynth produces")
    print(f"  reminder: fabric links must read {bx}Gbps; NVLink (intra-island) stays 4800Gbps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())