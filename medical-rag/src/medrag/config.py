"""Runtime settings, read from environment variables and `medical-rag/.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# medical-rag/ (the directory holding pyproject.toml). Relative paths are resolved against it,
# so commands behave the same regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

Backend = Literal["ollama", "vllm", "openai", "groq", "gemini"]

DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM backend
    llm_backend: Backend = "ollama"
    llm_model: str = "medrag-qwen2.5-3b"
    llm_base_url: str | None = None
    llm_api_key: str = "EMPTY"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.0
    llm_timeout_s: float = 120.0
    llm_context_window: int = 4096

    # Retrieval
    embed_model: str = "intfloat/multilingual-e5-base"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_device: str = "auto"
    rerank_device: str = "auto"
    retrieve_k_dense: int = 50
    retrieve_k_bm25: int = 50
    retrieve_k_out: int = 30
    rrf_k: int = 60
    rerank_top_n: int = 5
    rerank_min_score: float | None = None
    index_dir: Path = Path("data/index")

    # Logging / API
    log_requests: bool = True
    log_dir: Path = Path("logs")
    api_port: int = 8080

    # Evaluation (QA generation and LLM judge)
    openai_api_key: str | None = None
    eval_llm_model: str = "gpt-4.1-mini-2025-04-14"

    @field_validator("llm_base_url", "rerank_min_score", "openai_api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        return None if isinstance(v, str) and v.strip() == "" else v

    @field_validator("index_dir", "log_dir", mode="after")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v

    @property
    def resolved_base_url(self) -> str | None:
        return self.llm_base_url or DEFAULT_BASE_URLS.get(self.llm_backend)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
