"""Run one experiment configuration end to end.

    python -m bench.run configs/e2_vllm_fp16.yaml

1. start the server (if `server_command` is set) and wait for its health endpoint
2. record env.json and command.txt
3. for each concurrency level and repeat: warm up, then run the closed-loop load while
   sampling nvidia-smi (and vLLM /metrics); append to requests.csv and phases.csv
4. stop the server

Fairness details:
- Warm-up prompts are the real prompts with a marker prepended to the user message, so they
  share only the system prompt with the measured requests (as real traffic would).
- With prefix caching on, a second repeat would find whole prompts cached from the first.
  Set `reset_prefix_cache_url` (vLLM: POST /reset_prefix_cache, needs VLLM_SERVER_DEV_MODE=1)
  or run E1-E3 with prefix caching off.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx

from bench import env_info
from bench.config import BENCH_ROOT, BenchConfig, load_config
from bench.gpu_monitor import GpuMonitor, MetricsPoller
from bench.load_gen import TokenCounter, load_prompts, run_load, summary, write_records
from bench.server import ManagedServer


def warmup_variant(prompt: dict) -> dict:
    msgs = [dict(m) for m in prompt["messages"]]
    for m in msgs:
        if m["role"] == "user":
            m["content"] = "(warm-up) " + m["content"]
    return {**prompt, "id": "warmup-" + prompt["id"], "messages": msgs}


def reset_prefix_cache(url: str | None) -> None:
    if url:
        try:
            httpx.post(url, timeout=30).raise_for_status()
        except httpx.HTTPError as e:
            print(f"warning: prefix cache reset failed: {e}", file=sys.stderr)


def run(cfg: BenchConfig, config_path: Path, run_id: str | None = None) -> Path:
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.experiment}_{cfg.config_name}"
    out = BENCH_ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out / "config.yaml")
    (out / "command.txt").write_text(
        (cfg.server_command or "(server started outside bench.run)") + "\n", encoding="utf-8")

    all_prompts = load_prompts(cfg.prompts_path, seed=cfg.seed)
    measured = all_prompts[: cfg.num_prompts] if cfg.num_prompts else all_prompts
    warmup = [warmup_variant(p) for p in all_prompts[: cfg.warmup_requests]]
    counter = TokenCounter(cfg.tokenizer)
    common = {"url": cfg.chat_url, "model": cfg.model, "max_tokens": cfg.max_tokens,
              "temperature": cfg.temperature, "timeout_s": cfg.timeout_s,
              "include_usage": cfg.include_usage, "counter": counter}
    ident = {"run_id": run_id, "experiment": cfg.experiment, "backend": cfg.backend,
             "config_name": cfg.config_name}

    with ManagedServer(cfg.server_command, cfg.resolved_health_url, out / "server.log",
                       cfg.server_startup_timeout_s):
        (out / "env.json").write_text(json.dumps(env_info.collect(cfg), indent=2), encoding="utf-8")
        with GpuMonitor(out / "gpu.csv"), MetricsPoller(cfg.metrics_url, out / "vllm_metrics.csv"):
            phases = []
            for concurrency in cfg.concurrency:
                for repeat in range(1, cfg.repeats + 1):
                    reset_prefix_cache(cfg.reset_prefix_cache_url)
                    asyncio.run(run_load(warmup, concurrency=min(concurrency, len(warmup) or 1),
                                         **common))
                    order = measured[:]
                    random.Random(cfg.seed * 1000 + repeat).shuffle(order)
                    records, start, wall = asyncio.run(run_load(
                        order, concurrency=concurrency, max_duration_s=cfg.max_duration_s, **common))
                    rows = [{**ident, "concurrency": concurrency, "repeat": repeat,
                             **{k: v for k, v in asdict(r).items() if k != "text"}} for r in records]
                    write_records(out / "requests.csv", rows)
                    with open(out / "outputs.jsonl", "a", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps({"prompt_id": r.prompt_id, "concurrency": concurrency,
                                                "repeat": repeat, "text": r.text, "error": r.error},
                                               ensure_ascii=False) + "\n")
                    phases.append({**ident, "concurrency": concurrency, "repeat": repeat,
                                   "t_start": round(start, 3), "t_end": round(start + wall, 3),
                                   "wall_s": round(wall, 3)})
                    print(f"[{cfg.config_name}] c={concurrency} rep={repeat}: "
                          f"{json.dumps(summary(records, wall))}", flush=True)
            with open(out / "phases.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(phases[0]))
                w.writeheader()
                w.writerows(phases)
    print(f"results in {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.config, args.run_id)


if __name__ == "__main__":
    main()
