"""Aggregation and plotting on a synthetic results/ tree with the real file layout."""

import csv
import json

import numpy as np
import pandas as pd

from analysis.aggregate import aggregate
from analysis.plots import main as plots_main

# (experiment, config, concurrency levels, tokens/s per user at c=1, saturation)
FAKE = [
    ("E1", "vllm_fp16", [1], 40, 64), ("E1", "hf_fp16", [1], 25, 1),
    ("E2", "vllm_fp16", [1, 4, 16], 40, 64), ("E2", "vllm_awq", [1, 4, 16], 55, 96),
    ("E2", "hf_fp16", [1, 4, 16], 25, 1), ("E4", "vllm_fp16_prefix_off", [16], 40, 64),
    ("E4", "vllm_fp16_prefix_on", [16], 40, 64),
]


def write_run(root, experiment, config, levels, speed, sat, rng):
    run_id = f"x_{experiment}_{config}"
    d = root / run_id
    d.mkdir(parents=True)
    req_rows, phase_rows, gpu_rows = [], [], []
    t = 1_800_000_000.0
    for c in levels:
        for rep in (1, 2, 3):
            wall = 30.0
            n = int(min(c, sat) * wall / 7)
            for i in range(max(n, 2)):
                pt = int(rng.integers(800, 3000))
                ttft = 0.1 * pt * (0.5 if config.endswith("prefix_on") else 1) * (1 + c / sat)
                req_rows.append({"run_id": run_id, "experiment": experiment, "backend": "x",
                                 "config_name": config, "concurrency": c, "repeat": rep,
                                 "prompt_id": f"q_{i}", "lang": "vi", "prompt_tokens": pt,
                                 "completion_tokens": 200, "ttft_ms": ttft, "e2e_ms": ttft + 5000,
                                 "itl_mean_ms": 25, "itl_p95_ms": 30, "tpot_ms": 25, "n_chunks": 200,
                                 "t_send": t, "t_end": t + 5, "usage_source": "server",
                                 "error": "" if i else ("timeout" if c > 8 else "")})
            phase_rows.append({"run_id": run_id, "experiment": experiment, "backend": "x",
                               "config_name": config, "concurrency": c, "repeat": rep,
                               "t_start": t, "t_end": t + wall, "wall_s": wall})
            for s in range(int(wall)):
                gpu_rows.append({"unix_time": t + s, "timestamp": "", "memory_used": 15000 + 100 * c,
                                 "memory_total": 23034, "utilization_gpu": 90, "power_draw": 70})
            t += wall + 5
    for name, rows in (("requests.csv", req_rows), ("phases.csv", phase_rows), ("gpu.csv", gpu_rows)):
        with open(d / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (d / "DONE").write_text("ok")
    if experiment == "E2" and config in ("vllm_fp16", "vllm_awq"):
        (d / "quality_summary.json").write_text(json.dumps(
            {"config_name": config, "correctness_mean": 4.1 if config == "vllm_fp16" else 3.9}))


def test_aggregate_and_plots(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for exp, cfg, levels, speed, sat in FAKE:
        write_run(tmp_path, exp, cfg, levels, speed, sat, rng)
    # an interrupted attempt of the same configuration must not be counted
    write_run(tmp_path / "interrupted", "E2", "vllm_fp16", [16], 40, 64, rng)
    (tmp_path / "interrupted" / "x_E2_vllm_fp16").rename(tmp_path / "y_E2_vllm_fp16")
    (tmp_path / "y_E2_vllm_fp16" / "DONE").unlink()
    (tmp_path / "interrupted").rmdir()
    summary = aggregate(tmp_path)
    row = summary[(summary["config_name"] == "vllm_fp16") & (summary["experiment"] == "E2")
                  & (summary["concurrency"] == 16)].iloc[0]
    assert row["repeats"] == 3
    assert row["gpu_mem_peak_mib"] == 15000 + 1600
    assert 0 < row["error_rate"] < 1  # one timeout per repeat at c=16
    repeats = pd.read_csv(tmp_path / "summary_repeats.csv")
    c16 = repeats[(repeats["experiment"] == "E2") & (repeats["config_name"] == "vllm_fp16")
                  & (repeats["concurrency"] == 16)]
    assert len(c16) == 3  # 3 repeats of the finished run; the unfinished one is skipped
    hf = summary[(summary["config_name"] == "hf_fp16") & (summary["experiment"] == "E2")
                 & (summary["concurrency"] <= 4)]  # c=16 has an injected timeout
    assert hf["throughput_tok_s"].nunique() == 1  # sequential server: flat throughput

    monkeypatch.setattr("sys.argv", ["plots", "--results", str(tmp_path)])
    plots_main()
    pngs = sorted(p.name for p in (tmp_path / "plots").glob("*.png"))
    assert pngs == ["1_throughput_vs_concurrency.png", "2_ttft_p95_vs_concurrency.png",
                    "3_throughput_latency_tradeoff.png", "4_fp16_vs_awq.png",
                    "5_prefix_caching_ttft.png", "6_ttft_vs_input_tokens.png"]
    assert pd.read_csv(tmp_path / "summary.csv").shape[0] == len(summary)
