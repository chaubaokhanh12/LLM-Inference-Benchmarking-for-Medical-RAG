"""Backend-independent LLM interface.

Every backend streams internally, even for non-streaming calls: it is the only way to measure
time-to-first-token (TTFT), which is what a RAG user feels and what the benchmark compares.
`stream()` yields `TokenEvent`s and ends with one `DoneEvent` carrying timings and usage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Protocol


class LLMError(RuntimeError):
    """The backend could not be reached or rejected the request."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_ms: float | None  # until the first non-empty content delta
    total_ms: float
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenEvent:
    text: str


@dataclass
class DoneEvent:
    result: LLMResult


StreamEvent = TokenEvent | DoneEvent


class LLMBackend(Protocol):
    name: str
    model: str

    def stream(self, messages: list[dict], max_tokens: int,
               temperature: float) -> Iterator[StreamEvent]: ...

    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult: ...


class StreamingBackend:
    """Implements `generate()` on top of `stream()`."""

    name: str
    model: str

    def stream(self, messages: list[dict], max_tokens: int,
               temperature: float) -> Iterator[StreamEvent]:
        raise NotImplementedError

    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult:
        for event in self.stream(messages, max_tokens, temperature):
            if isinstance(event, DoneEvent):
                return event.result
        raise LLMError(f"{self.name} stream ended without a result")
