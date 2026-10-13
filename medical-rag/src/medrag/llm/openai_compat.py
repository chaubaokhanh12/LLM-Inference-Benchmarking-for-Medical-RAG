"""One client for every OpenAI-compatible server: OpenAI, vLLM, Ollama (/v1), Groq.

Only `base_url`, `api_key` and `model` differ between them, so switching backend is a
`.env` change. Usage numbers are requested in the stream (`stream_options.include_usage`);
servers that reject the option are retried without it and report `None` token counts.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import openai

from medrag.llm.base import (
    DoneEvent,
    LLMError,
    LLMResult,
    StreamEvent,
    StreamingBackend,
    TokenEvent,
)


class OpenAICompatBackend(StreamingBackend):
    def __init__(self, name: str, model: str, base_url: str, api_key: str = "EMPTY",
                 timeout_s: float = 120.0, max_retries: int = 2, http_client=None):
        self.name = name
        self.model = model
        self.base_url = base_url
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                                    timeout=timeout_s, max_retries=max_retries,
                                    http_client=http_client)
        self._include_usage = True
        self._send_temperature = True
        # OpenAI's newer models only accept max_completion_tokens; self-hosted servers use max_tokens.
        self._max_tokens_field = "max_completion_tokens" if name == "openai" else "max_tokens"

    def _open_stream(self, messages: list[dict], max_tokens: int, temperature: float):
        kwargs = {"model": self.model, "messages": messages, "stream": True,
                  self._max_tokens_field: max_tokens}
        if self._send_temperature:
            kwargs["temperature"] = temperature
        if self._include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            msg = str(e).lower()
            if self._include_usage and "stream_options" in msg:
                self._include_usage = False
            elif self._send_temperature and "temperature" in msg:
                self._send_temperature = False
            else:
                raise LLMError(f"{self.name} rejected the request: {e}") from e
            return self._open_stream(messages, max_tokens, temperature)
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise LLMError(f"cannot reach {self.name} at {self.base_url}: {e}") from e
        except openai.APIStatusError as e:
            raise LLMError(f"{self.name} returned HTTP {e.status_code}: {e.message}") from e

    def stream(self, messages: list[dict], max_tokens: int,
               temperature: float) -> Iterator[StreamEvent]:
        t0 = time.perf_counter()
        stream = self._open_stream(messages, max_tokens, temperature)
        ttft_ms = None
        parts: list[str] = []
        usage = None
        finish = None
        try:
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                for choice in chunk.choices:
                    # Only answer text counts for TTFT; reasoning deltas of "thinking" models
                    # (delta.reasoning_content) are not part of the answer.
                    text = choice.delta.content if choice.delta else None
                    if text:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        parts.append(text)
                        yield TokenEvent(text)
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise LLMError(f"{self.name} stream interrupted: {e}") from e
        yield DoneEvent(LLMResult(
            text="".join(parts),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            ttft_ms=ttft_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            finish_reason=finish,
        ))
