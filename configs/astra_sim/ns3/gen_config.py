from pathlib import Path

# Determina la directory dove si trova questo script
SCRIPT_DIR = Path(__file__).parent

# edit once (or just press enter at the prompts to keep these)
TEMPLATE  = SCRIPT_DIR / "ns3_config.template.txt"
CONF_DIR  = SCRIPT_DIR
OUT_DIR   = "/home/andre/tesi/trace_evaluator/output/ns3"

# cc name -> CC_MODE  (ideal = window-only, NO congestion; use a real CC otherwise)
CC_MODE = {"dcqcn": 1, "hpcc": 3, "timely": 7, "dctcp": 8, "hpcc-pint": 10, "none": 12}


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def main() -> int:
    tag = ask("tag (folder name)", "test")

    cc = ask(f"cc {list(CC_MODE)}", "dcqcn")
    while cc not in CC_MODE:
        cc = ask(f"  not valid, choose one of {list(CC_MODE)}", "dcqcn")

    buffer_size  = ask("buffer size", "32")
    enable_trace = ask("enable packet trace (0/1)", "0")
    qlen_start   = ask("qlen monitor start", "0")
    qlen_end     = ask("qlen monitor end", "2000000000")

    repl = {
        "__CONF_DIR__":     str(CONF_DIR),
        "__OUT_DIR__":      str(OUT_DIR),
        "__TAG__":          tag,
        "__CC_MODE__":      str(CC_MODE[cc]),
        "__BUFFER_SIZE__":  buffer_size,
        "__ENABLE_TRACE__": enable_trace,
        "__QLEN_START__":   qlen_start,
        "__QLEN_END__":     qlen_end,
    }

    txt = Path(TEMPLATE).read_text()
    for k, v in repl.items():
        txt = txt.replace(k, v)

    folder = Path(CONF_DIR) / tag
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "config.txt"
    out.write_text(txt)

    print(f"\nwritten {out}")
    print(f"now drop your topology.txt into {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())