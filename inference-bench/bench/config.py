"""Experiment configuration (one YAML file per experiment x backend configuration)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]  # inference-bench/
REPO_ROOT = BENCH_ROOT.parent


@dataclass
class BenchConfig:
    experiment: str  # E1..E5
    backend: str  # hf | vllm | ollama | tgi
    config_name: str  # e.g. vllm_fp16; one line per config in the plots
    base_url: str  # OpenAI-compatible root, e.g. http://localhost:8000/v1
    model: str  # name sent in the request ("served model name")
    server_command: str | None = None  # started and stopped by bench.run; None = already running
    health_url: str | None = None  # default: {base_url}/models
    concurrency: list[int] = field(default_factory=lambda: [1])
    repeats: int = 3
    warmup_requests: int = 10
    max_tokens: int = 256
    temperature: float = 0.0
    prompts: str = "../prompts/medical_rag_prompts.jsonl"
    num_prompts: int | None = None  # per (concurrency, repeat); None = all
    max_duration_s: float = 600.0  # per (concurrency, repeat)
    timeout_s: float = 120.0
    seed: int = 0
    tokenizer: str | None = None  # HF id for counting tokens when the server reports no usage
    include_usage: bool = True  # stream_options.include_usage
    server_startup_timeout_s: float = 900.0
    metrics_url: str | None = None  # vLLM Prometheus endpoint, e.g. http://localhost:8000/metrics
    reset_prefix_cache_url: str | None = None  # called before every (concurrency, repeat)
    model_revision: str | None = None  # recorded in env.json
    quantization: str | None = None  # recorded: fp16 | awq-int4 | gguf-q4_k_m ...
    notes: str = ""

    @property
    def prompts_path(self) -> Path:
        p = Path(self.prompts)
        return p if p.is_absolute() else (BENCH_ROOT / p).resolve()

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def resolved_health_url(self) -> str:
        return self.health_url or self.base_url.rstrip("/") + "/models"


def load_config(path: str | Path) -> BenchConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data.get("concurrency"), int):
        data["concurrency"] = [data["concurrency"]]
    return BenchConfig(**data)
