"""Combine every results/<run_id>/ into results/summary.csv.

    python -m analysis.aggregate

One row per (experiment, backend, config_name, concurrency), each metric being the median
over repeats. Per-repeat values are kept in results/summary_repeats.csv.

Throughput = completion tokens of successful requests / wall time of the measured phase
(warm-up excluded). GPU numbers come from the nvidia-smi samples inside that phase's window;
vLLM numbers (running/waiting requests, KV-cache usage) from /metrics samples likewise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bench.config import BENCH_ROOT

KEYS = ["experiment", "backend", "config_name", "concurrency"]


def _pct(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")


def _window(df: pd.DataFrame | None, t0: float, t1: float) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    w = df[(df["unix_time"] >= t0) & (df["unix_time"] <= t1)]
    return w if not w.empty else None


def _vllm_col(df: pd.DataFrame, *needles: str) -> str | None:
    for c in df.columns:
        if all(n in c for n in needles):
            return c
    return None


def summarise_run(run_dir: Path) -> list[dict]:
    req = pd.read_csv(run_dir / "requests.csv", keep_default_na=False)
    phases = pd.read_csv(run_dir / "phases.csv")
    gpu = pd.read_csv(run_dir / "gpu.csv") if (run_dir / "gpu.csv").exists() else None
    if gpu is not None:
        gpu = gpu.apply(pd.to_numeric, errors="coerce")
    vm = (pd.read_csv(run_dir / "vllm_metrics.csv")
          if (run_dir / "vllm_metrics.csv").exists() else None)
    rows = []
    for ph in phases.to_dict("records"):
        r = req[(req["concurrency"] == ph["concurrency"]) & (req["repeat"] == ph["repeat"])]
        ok = r[r["error"].astype(str) == ""]
        wall = ph["wall_s"]
        out_tokens = pd.to_numeric(ok["completion_tokens"], errors="coerce").sum()
        row = {k: ph[k] for k in ("run_id", *KEYS, "repeat")}
        row.update({
            "requests": len(r), "error_rate": 1 - len(ok) / len(r) if len(r) else float("nan"),
            "throughput_tok_s": out_tokens / wall if wall else float("nan"),
            "req_s": len(ok) / wall if wall else float("nan"),
            "ttft_p50_ms": _pct(ok["ttft_ms"], 50), "ttft_p95_ms": _pct(ok["ttft_ms"], 95),
            "tpot_p50_ms": _pct(ok["tpot_ms"], 50), "tpot_p95_ms": _pct(ok["tpot_ms"], 95),
            "e2e_p50_ms": _pct(ok["e2e_ms"], 50), "e2e_p95_ms": _pct(ok["e2e_ms"], 95),
            "prompt_tokens_mean": pd.to_numeric(ok["prompt_tokens"], errors="coerce").mean(),
        })
        g = _window(gpu, ph["t_start"], ph["t_end"])
        if g is not None:
            row.update({"gpu_mem_peak_mib": g["memory_used"].max(),
                        "gpu_util_mean": g["utilization_gpu"].mean(),
                        "gpu_power_mean_w": g["power_draw"].mean()})
        v = _window(vm, ph["t_start"], ph["t_end"])
        if v is not None:
            for name, needles in {"vllm_running_max": ("num_requests_running",),
                                  "vllm_waiting_max": ("num_requests_waiting",),
                                  "vllm_kv_usage_max": ("cache_usage",)}.items():
                col = _vllm_col(v, *needles)
                if col:
                    row[name] = v[col].max()
        rows.append(row)
    return rows


def aggregate(results_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in results_dir.iterdir() if (p / "phases.csv").exists()):
        rows += summarise_run(run_dir)
    if not rows:
        raise SystemExit(f"no completed runs under {results_dir}")
    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(results_dir / "summary_repeats.csv", index=False)
    metrics = [c for c in per_repeat.columns if c not in (*KEYS, "run_id", "repeat")]
    summary = (per_repeat.groupby(KEYS, as_index=False)
               .agg(**{m: (m, "median") for m in metrics}, repeats=("repeat", "nunique"))
               .sort_values(KEYS))
    summary.to_csv(results_dir / "summary.csv", index=False, float_format="%.2f")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=BENCH_ROOT / "results")
    args = ap.parse_args()
    summary = aggregate(args.results)
    cols = ["experiment", "config_name", "concurrency", "throughput_tok_s", "ttft_p95_ms",
            "tpot_p50_ms", "e2e_p95_ms", "error_rate", "repeats"]
    print(summary[[c for c in cols if c in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
