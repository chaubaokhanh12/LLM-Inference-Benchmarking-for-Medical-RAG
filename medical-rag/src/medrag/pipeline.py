"""retrieve -> rerank -> prompt -> LLM, with a timing for every stage.

    timings = {retrieve_ms, rerank_ms, llm_ttft_ms, llm_total_ms, total_ms}

`answer_stream()` is the single code path; `answer()` just drains it, so the streaming API
and the blocking API cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from medrag.config import Settings, get_settings
from medrag.generate.prompt import NOT_FOUND, build_citations, fit_messages, is_not_found
from medrag.ingest.chunker import get_token_counter
from medrag.lang import detect_lang
from medrag.llm import DoneEvent, LLMBackend, LLMResult, TokenEvent, create_backend
from medrag.retrieve.hybrid import HybridRetriever, Mode
from medrag.retrieve.rerank import Reranker
from medrag.types import ScoredChunk

log = logging.getLogger(__name__)

PROMPT_SAFETY_TOKENS = 64  # tokenizer mismatch between our estimate and the served model


@dataclass
class Retrieval:
    lang: str
    contexts: list[ScoredChunk]
    n_candidates: int
    timings: dict[str, float]


@dataclass
class AnswerResult:
    request_id: str
    question: str
    lang: str
    answer: str
    citations: list[dict]
    invalid_citations: list[int]
    timings: dict[str, float | None]
    backend: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    not_found: bool
    llm_called: bool
    contexts: list[dict] = field(default_factory=list)

    def to_dict(self, with_contexts: bool = True) -> dict:
        d = asdict(self)
        if not with_contexts:
            d.pop("contexts")
        return d


class RequestLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: HybridRetriever, reranker: Reranker | None,
                 backend: LLMBackend, logger: RequestLogger | None = None):
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.backend = backend
        self.logger = logger
        counter = get_token_counter(settings.embed_model)
        self.count_tokens = lambda msgs: sum(counter.count(m["content"]) + 8 for m in msgs)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RAGPipeline:
        from medrag.index.dense import Embedder
        from medrag.index.store import load_index

        s = settings or get_settings()
        bundle = load_index(s.index_dir, embed_model=s.embed_model)
        retriever = HybridRetriever(bundle, Embedder(s.embed_model, device=s.embed_device))
        reranker = Reranker(s.rerank_model, device=s.rerank_device)
        logger = RequestLogger(s.log_dir / "requests.jsonl") if s.log_requests else None
        return cls(s, retriever, reranker, create_backend(s), logger)

    # ---------------------------------------------------------------- retrieval

    def retrieve(self, question: str, *, top_n: int | None = None, lang: str | None = None,
                 mode: Mode = "hybrid", rerank: bool = True,
                 min_score: float | None = None, k_out: int | None = None) -> Retrieval:
        s = self.settings
        lang = lang or detect_lang(question)
        t0 = time.perf_counter()
        candidates = self.retriever.search(
            question, mode=mode, k_dense=s.retrieve_k_dense, k_bm25=s.retrieve_k_bm25,
            k_out=k_out or s.retrieve_k_out, rrf_k=s.rrf_k, lang=lang,
        )
        t1 = time.perf_counter()
        timings = {"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": 0.0}
        top_n = top_n or s.rerank_top_n
        if rerank and self.reranker is not None:
            contexts = self.reranker.rerank(question, candidates, top_n=top_n, min_score=min_score)
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
        else:
            contexts = candidates[:top_n]
        return Retrieval(lang, contexts, len(candidates), timings)

    def prepare_messages(self, question: str, lang: str, contexts: list[ScoredChunk]
                         ) -> tuple[list[dict], list[ScoredChunk]]:
        s = self.settings
        budget = s.llm_context_window - s.llm_max_tokens - PROMPT_SAFETY_TOKENS
        return fit_messages(question, lang, contexts, budget, self.count_tokens)

    # ---------------------------------------------------------------- generation

    def answer(self, question: str, top_n: int | None = None) -> AnswerResult:
        result = None
        for event, payload in self.answer_stream(question, top_n=top_n):
            if event == "done":
                result = payload
        assert result is not None
        return result

    def answer_stream(self, question: str, top_n: int | None = None
                      ) -> Iterator[tuple[str, object]]:
        """Yields ("meta", dict), then ("token", str) events, then ("done", AnswerResult)."""
        s = self.settings
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        r = self.retrieve(question, top_n=top_n, min_score=s.rerank_min_score)

        llm: LLMResult | None = None
        if r.contexts:
            messages, contexts = self.prepare_messages(question, r.lang, r.contexts)
            citations, _ = build_citations(contexts, "")
            yield "meta", {"request_id": request_id, "lang": r.lang, "citations": citations}
            for ev in self.backend.stream(messages, s.llm_max_tokens, s.llm_temperature):
                if isinstance(ev, TokenEvent):
                    yield "token", ev.text
                elif isinstance(ev, DoneEvent):
                    llm = ev.result
            self._check_truncation(messages, llm)
        else:  # nothing relevant enough: answer without calling the LLM
            contexts = []
            yield "meta", {"request_id": request_id, "lang": r.lang, "citations": []}
            yield "token", NOT_FOUND[r.lang]

        total_ms = (time.perf_counter() - t0) * 1000
        result = self.build_result(request_id, question, r, contexts, llm, total_ms)
        if self.logger:
            self.logger.write({"ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                               **result.to_dict(with_contexts=False),
                               "context_ids": [c.chunk.chunk_id for c in contexts]})
        yield "done", result

    def build_result(self, request_id: str, question: str, r: Retrieval,
                     contexts: list[ScoredChunk], llm: LLMResult | None,
                     total_ms: float) -> AnswerResult:
        """Assemble the response; also used by evaluation runs that call the LLM separately."""
        answer = llm.text.strip() if llm else NOT_FOUND[r.lang]
        citations, invalid = build_citations(contexts, answer)
        timings = {
            **{k: round(v, 2) for k, v in r.timings.items()},
            "llm_ttft_ms": round(llm.ttft_ms, 2) if llm and llm.ttft_ms is not None else None,
            "llm_total_ms": round(llm.total_ms, 2) if llm else None,
            "total_ms": round(total_ms, 2),
        }
        return AnswerResult(
            request_id=request_id, question=question, lang=r.lang, answer=answer,
            citations=citations, invalid_citations=invalid, timings=timings,
            backend=self.backend.name, model=self.backend.model,
            prompt_tokens=llm.prompt_tokens if llm else None,
            completion_tokens=llm.completion_tokens if llm else None,
            not_found=is_not_found(answer), llm_called=llm is not None,
            contexts=[c.to_dict() for c in contexts],
        )

    def _check_truncation(self, messages: list[dict], llm: LLMResult) -> None:
        """Ollama silently cuts prompts longer than its num_ctx; catch it from the usage numbers."""
        if llm.prompt_tokens is None:
            return
        expected = self.count_tokens(messages)
        if llm.prompt_tokens < 0.6 * expected:
            log.warning(
                "server counted %d prompt tokens but ~%d were sent: the prompt was probably "
                "truncated (for Ollama set OLLAMA_CONTEXT_LENGTH >= %d)",
                llm.prompt_tokens, expected, self.settings.llm_context_window,
            )

    def release_retrieval_models(self) -> None:
        """Free GPU memory held by the encoder and reranker (two-phase runs on small GPUs)."""
        import gc

        import torch

        self.retriever.embedder = None
        self.reranker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        """Load kernels and caches so the first real request is not measured cold."""
        self.retrieve("đau đầu và sốt kéo dài nên làm gì?")
        self.retrieve("What are the side effects of metformin?")
        try:
            self.backend.generate([{"role": "user", "content": "Hi"}], max_tokens=1, temperature=0)
        except Exception as e:  # noqa: BLE001 - the API can still serve /retrieve
            log.warning("LLM warm-up failed (%s); /ask will fail until the backend is up", e)
