"""Light stand-ins for models and servers, so pipeline/API tests run without GPUs or downloads."""

from __future__ import annotations

from medrag.config import Settings
from medrag.llm.base import DoneEvent, LLMError, LLMResult, StreamingBackend, TokenEvent
from medrag.pipeline import RAGPipeline
from medrag.types import Chunk, ScoredChunk


def make_chunk(i: int, text: str, lang: str = "vi") -> Chunk:
    return Chunk(chunk_id=f"doc{i}:0", doc_id=f"doc{i}", title=f"Tài liệu {i}",
                 section_path=["Điều trị"], text=text, lang=lang, n_tokens=50)


class FakeBundle:
    meta = {"embed_model": "e5", "n_chunks": 3, "built_at": "2026-01-01", "chunker_hash": "abc"}


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bundle = FakeBundle()

    def search(self, question, **kw) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=1.0 / (i + 1), rrf_score=1.0 / (i + 1),
                            dense_rank=i + 1) for i, c in enumerate(self.chunks)]


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores

    def rerank(self, question, candidates, top_n=None, min_score=None):
        for c, s in zip(candidates, self.scores, strict=False):
            c.rerank_score = c.score = s
        ranked = sorted(candidates, key=lambda c: -c.score)
        if min_score is not None:
            ranked = [c for c in ranked if c.score >= min_score]
        return ranked[:top_n] if top_n else ranked


class FakeBackend(StreamingBackend):
    def __init__(self, tokens: list[str], fail: bool = False):
        self.name = "fake"
        self.model = "fake-1"
        self.tokens = tokens
        self.fail = fail
        self.calls: list[list[dict]] = []

    def stream(self, messages, max_tokens, temperature):
        if self.fail:
            raise LLMError("backend down")
        self.calls.append(messages)
        for t in self.tokens:
            yield TokenEvent(t)
        yield DoneEvent(LLMResult(text="".join(self.tokens), prompt_tokens=900,
                                  completion_tokens=len(self.tokens), ttft_ms=12.0, total_ms=40.0))


def make_pipeline(tokens=("Uống paracetamol", " [1]."), scores=(0.9, 0.2, 0.7),
                  min_score=None, fail=False, log_dir=None) -> RAGPipeline:
    chunks = [
        make_chunk(0, "Paracetamol 500 mg mỗi 6 giờ khi sốt."),
        make_chunk(1, "Bệnh sởi lây qua đường hô hấp."),
        make_chunk(2, "Không dùng aspirin cho trẻ em dưới 16 tuổi."),
    ]
    settings = Settings(_env_file=None, rerank_min_score=min_score, rerank_top_n=2,
                        log_requests=log_dir is not None, log_dir=log_dir or "logs")
    return RAGPipeline(settings, FakeRetriever(chunks), FakeReranker(list(scores)),
                       FakeBackend(list(tokens), fail=fail),
                       logger=None if log_dir is None else
                       __import__("medrag.pipeline", fromlist=["RequestLogger"]).RequestLogger(
                           settings.log_dir / "requests.jsonl"))
