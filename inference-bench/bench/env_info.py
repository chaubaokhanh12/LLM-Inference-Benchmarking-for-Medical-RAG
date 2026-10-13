"""Everything needed to reproduce a run: hardware, drivers, library and server versions."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone

import httpx

from bench.config import BenchConfig


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _server_version(cfg: BenchConfig) -> str | None:
    root = cfg.base_url.rstrip("/").removesuffix("/v1")
    for path in ("/version", "/api/version", "/info"):  # vLLM, Ollama, TGI
        try:
            r = httpx.get(root + path, timeout=5)
            if r.status_code == 200:
                return r.text.strip()[:500]
        except httpx.HTTPError:
            continue
    return None


def collect(cfg: BenchConfig) -> dict:
    gpu = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
        "cuda_driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "nvcc": _cmd(["nvcc", "--version"]),
        "libraries": {p: _version(p) for p in
                      ("vllm", "transformers", "torch", "text-generation", "httpx", "openai")},
        "ollama": _cmd(["ollama", "--version"]),
        "server_version_endpoint": _server_version(cfg),
        "backend": cfg.backend,
        "config_name": cfg.config_name,
        "model": cfg.model,
        "model_revision": cfg.model_revision,
        "quantization": cfg.quantization,
        "server_command": cfg.server_command,
    }
