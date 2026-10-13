"""Gemini through the `google-genai` SDK, behind the same interface (pip install -e .[gemini])."""

from __future__ import annotations

import time
from collections.abc import Iterator

from medrag.llm.base import (
    DoneEvent,
    LLMError,
    LLMResult,
    StreamEvent,
    StreamingBackend,
    TokenEvent,
)


class GeminiBackend(StreamingBackend):
    def __init__(self, model: str, api_key: str, timeout_s: float = 120.0):
        from google import genai

        self.name = "gemini"
        self.model = model
        self.client = genai.Client(api_key=api_key,
                                   http_options={"timeout": int(timeout_s * 1000)})

    def stream(self, messages: list[dict], max_tokens: int,
               temperature: float) -> Iterator[StreamEvent]:
        from google.genai import types

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None, max_output_tokens=max_tokens,
            temperature=temperature,
        )
        t0 = time.perf_counter()
        ttft_ms, parts, usage, finish = None, [], None, None
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                if chunk.usage_metadata:
                    usage = chunk.usage_metadata
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish = str(chunk.candidates[0].finish_reason)
                text = chunk.text or ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    parts.append(text)
                    yield TokenEvent(text)
        except Exception as e:  # noqa: BLE001 - SDK raises several error types
            raise LLMError(f"gemini request failed: {e}") from e
        yield DoneEvent(LLMResult(
            text="".join(parts),
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            ttft_ms=ttft_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            finish_reason=finish,
        ))
