"""The six figures of the benchmark README (results/plots/*.png).

    python -m analysis.plots

1 throughput vs concurrency (E2)          4 FP16 vs AWQ: memory, throughput, quality (E3)
2 p95 TTFT vs concurrency, log y (E2)     5 TTFT with / without prefix caching (E4)
3 throughput vs p95 latency trade-off (E2) 6 TTFT vs input tokens: prefill cost (E1, c=1)

Colours follow the configuration (never its rank), marker shapes double-encode it, and
lines are labelled directly; results/summary.csv is the table behind every chart.
A plot whose data is missing is skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from bench.config import BENCH_ROOT  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Fixed identity -> slot, so a configuration keeps its colour in every chart.
FIXED = {"vllm_fp16": 0, "vllm_awq": 1, "ollama_q4_k_m": 2, "hf_fp16": 3, "tgi_fp16": 4}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LABELS = {"vllm_fp16": "vLLM FP16", "vllm_awq": "vLLM AWQ (4-bit)", "ollama_q4_k_m": "Ollama GGUF Q4_K_M",
          "hf_fp16": "HF Transformers FP16", "tgi_fp16": "TGI FP16",
          "vllm_fp16_prefix_off": "prefix caching off", "vllm_fp16_prefix_on": "prefix caching on"}


def style(name: str) -> tuple[str, str]:
    slot = FIXED.get(name)
    if slot is None:  # unknown configs take the remaining slots in a stable order
        free = [i for i in range(len(SLOTS)) if i not in FIXED.values()]
        slot = free[sum(map(ord, name)) % len(free)]
    return SLOTS[slot], MARKERS[slot]


def label(name: str) -> str:
    return LABELS.get(name, name)


def setup(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)


def plain_numbers(axis) -> None:
    """Log axes default to 6x10^0 style ticks; show plain 1-2-5 numbers instead."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def end_labels(ax, items: list[tuple[str, float, float]], min_gap_px: float = 11) -> None:
    """Label line ends, nudging labels apart vertically so converging lines stay readable."""
    ax.figure.canvas.draw()  # final limits and scales
    pts = sorted(((ax.transData.transform((x, y))[1], text, x, y) for text, x, y in items))
    placed: list[float] = []
    for disp_y, text, x, y in pts:
        target = max(disp_y, placed[-1] + min_gap_px) if placed else disp_y
        placed.append(target)
        dpi_scale = ax.figure.dpi / 72  # offset points -> pixels
        ax.annotate(text, (x, y), xytext=(6, (target - disp_y) / dpi_scale),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)


def new_fig(w: float = 7.5, h: float = 4.5, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    return fig, axes


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / name}")


def line_by_config(df: pd.DataFrame, x: str, y: str, out: Path, fname: str, title: str,
                   xlabel: str, ylabel: str, logy: bool = False) -> None:
    fig, ax = new_fig()
    setup(ax, title, xlabel, ylabel)
    ends = []
    for name, g in df.groupby("config_name"):
        g = g.sort_values(x)
        color, marker = style(name)
        ax.plot(g[x], g[y], color=color, marker=marker, linewidth=2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label(name))
        ends.append((label(name), g[x].iloc[-1], g[y].iloc[-1]))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df[x].unique()), [str(int(v)) for v in sorted(df[x].unique())])
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
        plain_numbers(ax.yaxis)
    ax.margins(x=0.25)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    end_labels(ax, ends)
    save(fig, out, fname)


def tradeoff(e2: pd.DataFrame, out: Path) -> None:
    fig, ax = new_fig()
    setup(ax, "Throughput vs p95 latency (each point = one concurrency level)",
          "p95 end-to-end latency (s)", "Throughput (output tokens/s)")
    for i, (name, g) in enumerate(e2.groupby("config_name")):
        g = g.sort_values("concurrency")
        color, marker = style(name)
        ax.plot(g["e2e_p95_ms"] / 1000, g["throughput_tok_s"], color=color, marker=marker,
                linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label(name))
        for _, row in g.iloc[[0, -1]].iterrows():  # label only the ends: c=min and c=max
            ax.annotate(f"c={int(row['concurrency'])}", (row["e2e_p95_ms"] / 1000, row["throughput_tok_s"]),
                        xytext=(5, 5 + 10 * i), textcoords="offset points", fontsize=8,
                        color=INK_2)
    ax.set_xscale("log")
    plain_numbers(ax.xaxis)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out, "3_throughput_latency_tradeoff.png")


def bars(ax, names: list[str], values: list[float], fmt: str) -> None:
    colors = [style(n)[0] for n in names]
    pos = np.arange(len(names))
    ax.bar(pos, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(pos, [label(n) for n in names], fontsize=8)
    for p, v in zip(pos, values, strict=True):
        if not np.isnan(v):
            ax.annotate(fmt.format(v), (p, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)


def fp16_vs_awq(summary: pd.DataFrame, results: Path, out: Path) -> None:
    names = ["vllm_fp16", "vllm_awq"]
    e = summary[(summary["config_name"].isin(names)) & (summary["concurrency"] == 16)
                & (summary["experiment"].isin(["E2", "E3"]))]
    if e.empty:
        print("skip 4_fp16_vs_awq.png (no E2/E3 c=16 runs)")
        return
    e = e.sort_values("experiment").groupby("config_name").first()  # prefer E2 (3 repeats)
    quality = {}
    for qf in results.glob("*/quality_summary.json"):
        q = json.loads(qf.read_text(encoding="utf-8"))
        quality[q["config_name"]] = q.get("correctness_mean")
    fig, axes = new_fig(10, 3.8, ncols=3)
    panels = [("Peak GPU memory (GiB)", lambda n: e.loc[n, "gpu_mem_peak_mib"] / 1024, "{:.1f}"),
              ("Throughput at 16 users (tok/s)", lambda n: e.loc[n, "throughput_tok_s"], "{:.0f}"),
              ("Answer correctness (1-5, judge)", lambda n: quality.get(n, np.nan), "{:.2f}")]
    for ax, (title, get, fmt) in zip(axes, panels, strict=True):
        setup(ax, title, "", "")
        vals = [float(get(n)) if n in e.index or "correctness" in title else np.nan for n in names]
        bars(ax, names, vals, fmt)
    save(fig, out, "4_fp16_vs_awq.png")


def prefix_caching(summary: pd.DataFrame, out: Path) -> None:
    e4 = summary[summary["experiment"] == "E4"]
    if e4.empty:
        print("skip 5_prefix_caching_ttft.png (no E4 runs)")
        return
    fig, ax = new_fig(6.5, 4)
    setup(ax, "TTFT with and without prefix caching (16 users)", "", "TTFT (ms)")
    order = [n for n in ("vllm_fp16_prefix_off", "vllm_fp16_prefix_on") if n in set(e4["config_name"])]
    colors = {"vllm_fp16_prefix_off": MUTED, "vllm_fp16_prefix_on": SLOTS[0]}
    width = 0.35
    for i, name in enumerate(order):
        row = e4[e4["config_name"] == name].iloc[0]
        vals = [row["ttft_p50_ms"], row["ttft_p95_ms"]]
        pos = np.arange(2) + (i - (len(order) - 1) / 2) * width
        ax.bar(pos, vals, width=width, color=colors[name], edgecolor=SURFACE, linewidth=2, label=label(name))
        for p, v in zip(pos, vals, strict=True):
            ax.annotate(f"{v:.0f}", (p, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["p50", "p95"])
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out, "5_prefix_caching_ttft.png")


def ttft_vs_input(results: Path, out: Path) -> None:
    frames = []
    for f in results.glob("*/requests.csv"):
        df = pd.read_csv(f, keep_default_na=False)
        df = df[(df["experiment"] == "E1") & (df["concurrency"] == 1) & (df["error"].astype(str) == "")]
        frames.append(df)
    data = pd.concat(frames) if frames else pd.DataFrame()
    if data.empty:
        print("skip 6_ttft_vs_input_tokens.png (no E1 runs)")
        return
    names = sorted(data["config_name"].unique(), key=lambda n: FIXED.get(n, 99))
    fig, axes = new_fig(3.4 * len(names), 3.8, ncols=len(names))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        g = data[data["config_name"] == name]
        x = pd.to_numeric(g["prompt_tokens"], errors="coerce")
        y = pd.to_numeric(g["ttft_ms"], errors="coerce")
        ok = x.notna() & y.notna()
        color, marker = style(name)
        setup(ax, label(name), "input tokens", "TTFT (ms)")
        ax.scatter(x[ok], y[ok], s=18, color=color, marker=marker, alpha=0.7, edgecolors=SURFACE, linewidths=0.8)
        if ok.sum() > 2:
            slope, intercept = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=INK_2, linewidth=1.2)
            ax.annotate(f"≈ {slope * 1000:.0f} ms per 1k tokens", (0.03, 0.95), xycoords="axes fraction",
                        va="top", fontsize=8, color=INK_2)
    save(fig, out, "6_ttft_vs_input_tokens.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=BENCH_ROOT / "results")
    args = ap.parse_args()
    out = args.results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.results / "summary.csv")
    e2 = summary[summary["experiment"] == "E2"]
    if e2.empty:
        print("skip plots 1-3 (no E2 runs)")
    else:
        line_by_config(e2, "concurrency", "throughput_tok_s", out, "1_throughput_vs_concurrency.png",
                       "Throughput vs concurrent users", "concurrent users", "Throughput (output tokens/s)")
        line_by_config(e2, "concurrency", "ttft_p95_ms", out, "2_ttft_p95_vs_concurrency.png",
                       "p95 time to first token vs concurrent users", "concurrent users",
                       "p95 TTFT (ms, log scale)", logy=True)
        tradeoff(e2, out)
    fp16_vs_awq(summary, args.results, out)
    prefix_caching(summary, out)
    ttft_vs_input(args.results, out)


if __name__ == "__main__":
    main()
