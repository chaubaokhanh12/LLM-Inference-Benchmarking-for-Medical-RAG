"""LLM backends. `create_backend(settings)` picks one from `LLM_BACKEND`."""

from __future__ import annotations

from medrag.config import Settings
from medrag.llm.base import DoneEvent, LLMBackend, LLMError, LLMResult, TokenEvent

__all__ = ["DoneEvent", "LLMBackend", "LLMError", "LLMResult", "TokenEvent", "create_backend"]


def create_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "gemini":
        from medrag.llm.gemini import GeminiBackend

        return GeminiBackend(settings.llm_model, settings.llm_api_key, settings.llm_timeout_s)

    from medrag.llm.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(
        name=settings.llm_backend, model=settings.llm_model,
        base_url=settings.resolved_base_url, api_key=settings.llm_api_key,
        timeout_s=settings.llm_timeout_s,
    )
