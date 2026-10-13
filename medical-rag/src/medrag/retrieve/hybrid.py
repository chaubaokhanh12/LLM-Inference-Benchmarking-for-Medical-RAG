"""Hybrid retrieval: dense (e5) + BM25, fused with Reciprocal Rank Fusion.

RRF only looks at ranks, never at raw scores, so it needs no calibration between a cosine
similarity (~0.8) and a BM25 score (~15):

    score(d) = sum over branches of 1 / (rrf_k + rank_branch(d)),  rank starting at 1

A document missing from a branch's top-k simply gets nothing from that branch.
"""

from __future__ import annotations

import threading
from typing import Literal

from medrag.index.dense import Embedder
from medrag.index.store import IndexBundle
from medrag.lang import detect_lang
from medrag.types import ScoredChunk

Mode = Literal["hybrid", "dense", "bm25"]


def rrf_fuse(rankings: list[list[int]], rrf_k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists of row ids; returns (row, score) sorted by score, ties by first seen."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    def __init__(self, bundle: IndexBundle, embedder: Embedder):
        self.bundle = bundle
        self.embedder = embedder
        self._lock = threading.Lock()  # one GPU, one encoder call at a time

    def dense_search(self, question: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            qvec = self.embedder.embed_queries([question])
        scores, ids = self.bundle.dense.search(qvec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def bm25_search(self, question: str, lang: str, k: int) -> list[tuple[int, float]]:
        return self.bundle.bm25.search(question, lang, k)

    def search(self, question: str, *, mode: Mode = "hybrid", k_dense: int = 50,
               k_bm25: int = 50, k_out: int = 30, rrf_k: int = 60,
               lang: str | None = None) -> list[ScoredChunk]:
        lang = lang or detect_lang(question)
        dense = self.dense_search(question, k_dense) if mode != "bm25" else []
        sparse = self.bm25_search(question, lang, k_bm25) if mode != "dense" else []

        dense_info = {row: (rank, s) for rank, (row, s) in enumerate(dense, start=1)}
        bm25_info = {row: (rank, s) for rank, (row, s) in enumerate(sparse, start=1)}

        if mode == "dense":
            ordered = dense
        elif mode == "bm25":
            ordered = sparse
        else:
            ordered = rrf_fuse([[r for r, _ in dense], [r for r, _ in sparse]], rrf_k)

        results = []
        for row, score in ordered[:k_out]:
            d_rank, d_score = dense_info.get(row, (None, None))
            b_rank, b_score = bm25_info.get(row, (None, None))
            results.append(ScoredChunk(
                chunk=self.bundle.chunks[row], score=score,
                rrf_score=score if mode == "hybrid" else None,
                dense_rank=d_rank, dense_score=d_score, bm25_rank=b_rank, bm25_score=b_score,
            ))
        return results
