"""Core data types passed between ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Document:
    doc_id: str
    title: str
    lang: str  # "vi" | "en"
    text: str
    source_path: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}:{index}"
    doc_id: str
    title: str
    section_path: list[str]
    text: str
    lang: str
    n_tokens: int  # tokens of the full embedding input ("passage: " + header + text, with specials)
    oversized: bool = False  # an atomic block (table/list) longer than the token limit

    @property
    def header(self) -> str:
        return " > ".join([self.title, *self.section_path])

    @property
    def contextual_text(self) -> str:
        """Text with its title/section header; used for embedding, BM25 and reranking."""
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # score used for the current ordering (RRF, or rerank once reranked)
    rrf_score: float | None = None
    dense_rank: int | None = None  # 1-based; None = not in that branch's top-k
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None  # sigmoid probability in [0, 1]

    def to_dict(self, with_text: bool = True) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "chunk"}
        c = self.chunk
        d.update(chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title,
                 section_path=c.section_path, lang=c.lang)
        if with_text:
            d["text"] = c.text
        return d
