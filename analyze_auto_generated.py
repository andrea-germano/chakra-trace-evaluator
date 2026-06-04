from __future__ import annotations
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import pandas as pd
# --------------------------------------------------------------------------- #
# Import the single-run analysis engine (analyze_logs.py)                     #
# --------------------------------------------------------------------------- #
import analyze_log as analyzer

# --------------------------------------------------------------------------- #
# Configuration — edit here, no CLI                                            #
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

# where generate_all_logs.sh dropped the per-scenario log folders
LOGS_ROOT = BASE_DIR / "astra_logs" / "auto_generated"
# the auto-generated config trees
MLSYNTH_CFG_ROOT = BASE_DIR / "configs" / "mlsynth" / "auto_generated"
NETWORK_CFG_ROOT = BASE_DIR / "configs" / "astra_sim" / "auto_generated"
# output (global constant, as in SUMMARY.md)
OUTPUT_DIR = BASE_DIR / "compare_results"

# name suffixes used by generate_configs.py
MLSYNTH_SUFFIX = "_mlsynth.yaml"
NETWORK_SUFFIX = "_network.yml"

# --------------------------------------------------------------------------- #
# Discovery + per-run metric extraction                                        #
# --------------------------------------------------------------------------- #
def find_log_file(scenario_dir: Path) -> Path | None:
    """The batch script moves ASTRA-sim's `log/` folder to the scenario dir; the
    main log inside is `log.log`. Prefer that, else the largest *.log file."""
    explicit = scenario_dir / "log.log"
    if explicit.is_file():
        return explicit
    candidates = sorted(scenario_dir.rglob("*.log"), key=lambda p: p.stat().st_size,
                        reverse=True)
    return candidates[0] if candidates else None


def parse_scenario_id(scenario_id: str) -> dict:
    """Loose fallback parse of `bw{bw}_prompt{p}_{mode}` for any field a config
    fails to provide. Missing pieces stay None."""
    out = {"bandwidth_gbps": None, "prompt_len": None, "kv_mode": None}
    m = re.search(r"bw(\d+(?:\.\d+)?)", scenario_id)
    if m:
        out["bandwidth_gbps"] = float(m.group(1))
    m = re.search(r"prompt(\d+)", scenario_id)
    if m:
        out["prompt_len"] = int(m.group(1))
    m = re.search(r"(bulk|streaming)", scenario_id)
    if m:
        out["kv_mode"] = m.group(1)
    return out


def read_bandwidth(scenario_id: str) -> float | None:
    """Bandwidth is in the network YAML, not the MLSynth one (per SUMMARY.md)."""
    path = NETWORK_CFG_ROOT / f"{scenario_id}{NETWORK_SUFFIX}"
    if not path.is_file():
        return None
    try:
        net = yaml.safe_load(path.read_text())
        bw = net.get("bandwidth")
        if isinstance(bw, (list, tuple)):
            bw = bw[0] if bw else None
        return float(bw) if bw is not None else None
    except Exception:
        return None


def analyse_log(log_path: Path, cfg_path: Path | None):
    """Run prova.py's engine on one log; return its analysis dict (the same
    object prova.py serialises to summary.json minus the per-NPU detail)."""
    ops, finished, stats = analyzer.parse_log(str(log_path))
    if not stats and not finished:
        return None
    warns: list[str] = []
    cfg = None
    if cfg_path and cfg_path.is_file():
        cfg = analyzer.load_config(str(cfg_path))
        part, npf, ndc = analyzer.build_partition(cfg)
    else:
        part, npf, ndc = analyzer.heuristic_partition(stats, warns)
    ms = analyzer.derive_ms_per_cycle(ops, stats)
    return analyzer.analyse(ops, finished, stats, ms, part, cfg, warns)


def collect_runs() -> pd.DataFrame:
    """Build one row per completed scenario. Discovery is filesystem-driven:
    every sub-directory of LOGS_ROOT that contains a log is a candidate."""
    if not LOGS_ROOT.is_dir():
        sys.exit(f"Logs directory not found: {LOGS_ROOT}\n"
                 "Run generate_all_logs.sh first.")

    rows = []
    for scenario_dir in sorted(p for p in LOGS_ROOT.iterdir() if p.is_dir()):
        scenario_id = scenario_dir.name
        log_path = find_log_file(scenario_dir)
        if log_path is None:
            warnings.warn(f"{scenario_id}: no log file found, skipping.")
            continue

        cfg_path = MLSYNTH_CFG_ROOT / f"{scenario_id}{MLSYNTH_SUFFIX}"
        if not cfg_path.is_file():
            warnings.warn(f"{scenario_id}: MLSynth config not found "
                          f"({cfg_path.name}), partition will use a heuristic.")
            cfg_path = None

        a = analyse_log(log_path, cfg_path)
        if a is None:
            warnings.warn(f"{scenario_id}: empty/unparseable log, skipping.")
            continue

        # sweep axes: prefer config values, fall back to the scenario_id
        fb = parse_scenario_id(scenario_id)
        cfg = a.get("config") or {}
        prompt_len = (cfg.get("requests") or [{}])[0].get("prompt_len") if cfg else None
        kv_mode = (cfg.get("kv_transfer") or {}).get("mode") if cfg else None
        bandwidth = read_bandwidth(scenario_id)

        pm = a.get("pool_means", {})
        rows.append({
            "scenario_id": scenario_id,
            # --- swept parameters ---
            "bandwidth_gbps": bandwidth if bandwidth is not None else fb["bandwidth_gbps"],
            "prompt_len": prompt_len if prompt_len is not None else fb["prompt_len"],
            "kv_mode": kv_mode or fb["kv_mode"] or "unknown",
            "num_layers": (cfg.get("model") or {}).get("num_layers"),
            "hidden_size": (cfg.get("model") or {}).get("hidden_size"),
            "bytes_per_val": (cfg.get("model") or {}).get("bytes_per_val"),
            # --- headline metrics ---
            "ttft_ms": a["ttft_ms"],
            "prefill_compute_ms": a["prefill_compute_ms"],
            "kv_transfer_ms": a["kv_transfer_ms"],
            "ttft_first_step_ms": a["ttft_first_step_ms"],
            "kv_share_pct": a["kv_share_of_ttft_pct"],
            "kv_in_tokens": a["kv_in_tokens"],
            "avg_itl_ms": a["avg_itl_ms"],
            "decode_total_ms": a["decode_total_ms"],
            "decode_wait_ms": a["decode_wait_ms"],
            "makespan_ms": a["makespan_ms"],
            "prefill_compute_util": (pm.get("prefill") or {}).get("compute_util"),
            "decode_mem_util": (pm.get("decode") or {}).get("mem_util"),
        })

    if not rows:
        sys.exit("No analysable runs found under " + str(LOGS_ROOT))
    df = pd.DataFrame(rows).sort_values(
        ["kv_mode", "prompt_len", "bandwidth_gbps"], na_position="last"
    ).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Plotting helpers                                                             #
# --------------------------------------------------------------------------- #
C_PREFILL = "#3B7DD8"
C_TRANSFER = "#E8743B"
C_DECODE = "#2E8B57"
C_INK = "#1F2937"
# distinct hues per kv_mode; falls back to a cycle for unexpected modes
MODE_COLORS = {"bulk": "#E8743B", "streaming": "#3B7DD8"}
_FALLBACK = ["#8E44AD", "#16A085", "#C0392B", "#2C3E50"]
# distinct markers/linestyles per prompt_len bucket
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", ":", "-."]


def _mode_color(mode, idx=0):
    return MODE_COLORS.get(mode, _FALLBACK[idx % len(_FALLBACK)])


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _line_vs_x(df, xcol, ycol, title, ylabel, fname, ylog=False):
    """Generic: y vs xcol, one line per (kv_mode, prompt_len). Skips if xcol has
    a single value (a line plot would be a dot)."""
    sub = df.dropna(subset=[xcol, ycol])
    if sub.empty or sub[xcol].nunique() < 2:
        return None
    modes = sorted(sub["kv_mode"].unique())
    plens = sorted(p for p in sub["prompt_len"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mi, mode in enumerate(modes):
        for pi, plen in enumerate(plens):
            g = sub[(sub["kv_mode"] == mode) & (sub["prompt_len"] == plen)]
            if g.empty:
                continue
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g[ycol], marker=MARKERS[pi % len(MARKERS)],
                    linestyle=LINESTYLES[mi % len(LINESTYLES)],
                    color=_mode_color(mode, mi), lw=2, ms=7,
                    label=f"{mode}, prompt={int(plen)}")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xcol.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=max(1, len(modes)), framealpha=0.9)
    return _save(fig, fname)


def plot_ttft_vs_bandwidth(df):
    return _line_vs_x(df, "bandwidth_gbps", "ttft_ms",
                      "TTFT vs network bandwidth\n(link saturation: where TTFT stops improving)",
                      "TTFT (ms)", "ttft_vs_bandwidth.png")


def plot_kv_share_vs_bandwidth(df):
    p = _line_vs_x(df, "bandwidth_gbps", "kv_share_pct",
                   "KV-transfer share of TTFT vs bandwidth\n(when does the transfer stop dominating?)",
                   "KV transfer as % of TTFT", "kv_share_vs_bandwidth.png")
    return p


def plot_itl_vs_bandwidth(df):
    return _line_vs_x(df, "bandwidth_gbps", "avg_itl_ms",
                      "Mean ITL vs bandwidth (control: decode is memory-bound,\n"
                      "so it should be ~flat — bandwidth shouldn't move it)",
                      "mean inter-token latency (ms)", "itl_vs_bandwidth.png")


def plot_kv_transfer_vs_prompt(df):
    """KV transfer time vs prompt length, a line per (bandwidth, kv_mode).
    Shows the (roughly linear) growth of the KV volume to move."""
    sub = df.dropna(subset=["prompt_len", "kv_transfer_ms"])
    if sub.empty or sub["prompt_len"].nunique() < 2:
        return None
    bws = sorted(b for b in sub["bandwidth_gbps"].dropna().unique())
    modes = sorted(sub["kv_mode"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mi, mode in enumerate(modes):
        for bi, bw in enumerate(bws):
            g = sub[(sub["kv_mode"] == mode) & (sub["bandwidth_gbps"] == bw)]
            if g.empty:
                continue
            g = g.sort_values("prompt_len")
            ax.plot(g["prompt_len"], g["kv_transfer_ms"],
                    marker=MARKERS[bi % len(MARKERS)],
                    linestyle=LINESTYLES[mi % len(LINESTYLES)],
                    color=_mode_color(mode, mi), lw=2, ms=7,
                    label=f"{mode}, bw={int(bw)} GB/s")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("KV transfer time (ms)")
    ax.set_title("KV-transfer cost vs prompt length\n(volume to move grows with the prompt)",
                 fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, framealpha=0.9)
    return _save(fig, "kv_transfer_vs_prompt_len.png")


def plot_ttft_breakdown(df):
    """Stacked TTFT decomposition for every run: prefill compute / KV transfer /
    1st decode step. The single most communicative 'where is the bottleneck'
    view across the whole sweep."""
    d = df.copy()
    if d.empty:
        return None
    labels = d["scenario_id"].tolist()
    pf = d["prefill_compute_ms"].to_numpy()
    kv = d["kv_transfer_ms"].to_numpy()
    fs = d["ttft_first_step_ms"].to_numpy()
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(d) + 4), 6))
    ax.bar(x, pf, 0.7, color=C_PREFILL, label="Prefill compute")
    ax.bar(x, kv, 0.7, bottom=pf, color=C_TRANSFER, label="KV transfer")
    ax.bar(x, fs, 0.7, bottom=pf + kv, color=C_DECODE, label="1st decode step")
    # annotate KV share on top of each bar
    for xi, p_, k_, f_ in zip(x, pf, kv, fs):
        tot = p_ + k_ + f_
        if tot > 0:
            ax.text(xi, tot, f"{k_ / tot * 100:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C_TRANSFER)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT breakdown across all runs (label on top = KV share of TTFT)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "ttft_breakdown_all_runs.png")


def plot_streaming_vs_bulk(df):
    """Direct streaming-vs-bulk contrast on the KV transfer and TTFT, paired by
    (bandwidth, prompt_len). Central to the thesis: streaming overlaps the KV
    transfer with prefill compute, so its exposed transfer should be smaller."""
    if set(df["kv_mode"].unique()) < {"bulk", "streaming"}:
        return None
    piv = df.pivot_table(index=["bandwidth_gbps", "prompt_len"], columns="kv_mode",
                         values=["kv_transfer_ms", "ttft_ms"], aggfunc="mean")
    if piv.empty:
        return None
    pairs = [idx for idx in piv.index]
    labels = [f"bw{int(b)}\np{int(p)}" for (b, p) in pairs]
    x = np.arange(len(pairs)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(pairs) + 5), 5))
    for ax, metric, ttl in [(ax1, "kv_transfer_ms", "KV transfer (ms)"),
                            (ax2, "ttft_ms", "TTFT (ms)")]:
        bulk = [piv.loc[idx, (metric, "bulk")] if (metric, "bulk") in piv.columns else np.nan
                for idx in pairs]
        strm = [piv.loc[idx, (metric, "streaming")] if (metric, "streaming") in piv.columns else np.nan
                for idx in pairs]
        ax.bar(x - w / 2, bulk, w, color=MODE_COLORS["bulk"], label="bulk")
        ax.bar(x + w / 2, strm, w, color=MODE_COLORS["streaming"], label="streaming")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ttl); ax.set_title(ttl, fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Streaming vs bulk KV transfer (paired by bandwidth × prompt length)",
                 fontsize=12.5, fontweight="bold")
    return _save(fig, "streaming_vs_bulk.png")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    df = collect_runs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "all_runs.csv"
    df.to_csv(csv_path, index=False)

    figs = [
        plot_ttft_breakdown(df),
        plot_ttft_vs_bandwidth(df),
        plot_kv_share_vs_bandwidth(df),
        plot_kv_transfer_vs_prompt(df),
        plot_itl_vs_bandwidth(df),
        plot_streaming_vs_bulk(df),
    ]
    figs = [f for f in figs if f]

    print("=" * 70)
    print(f"  CROSS-RUN COMPARISON — {len(df)} scenarios")
    print("=" * 70)
    cols = ["scenario_id", "bandwidth_gbps", "prompt_len", "kv_mode",
            "ttft_ms", "kv_transfer_ms", "kv_share_pct", "avg_itl_ms"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df[cols].to_string(index=False))
    print("=" * 70)
    print("\nGenerated:")
    for p in [csv_path] + figs:
        print("  ", p)
    skipped = 6 - len(figs)
    if skipped:
        print(f"  ({skipped} plot(s) skipped: the relevant sweep axis had a single value "
              "or data was missing.)")


if __name__ == "__main__":
    main()