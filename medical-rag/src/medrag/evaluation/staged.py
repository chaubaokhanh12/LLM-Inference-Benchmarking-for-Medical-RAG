"""Batch retrieval with one model in memory at a time (for machines with little RAM/VRAM).

The online pipeline keeps the encoder and the reranker loaded together. Evaluation runs over
hundreds of questions do not need that: embed + search everything first, free the encoder,
then load the reranker and rerank everything. Results are identical to the online pipeline.
"""

from __future__ import annotations

import dataclasses
import gc
import time
from dataclasses import dataclass, field

from tqdm import tqdm

from medrag.config import Settings
from medrag.index.dense import Embedder
from medrag.index.store import IndexBundle, load_index
from medrag.retrieve.hybrid import HybridRetriever, Mode
from medrag.retrieve.rerank import Reranker
from medrag.types import ScoredChunk


@dataclass
class StagedResult:
    question: str
    lang: str
    by_mode: dict[str, list[ScoredChunk]] = field(default_factory=dict)  # bm25 / dense / hybrid
    reranked: list[ScoredChunk] = field(default_factory=list)  # all hybrid candidates, reordered
    latency_ms: dict[str, float] = field(default_factory=dict)


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def staged_retrieval(questions: list[tuple[str, str]], s: Settings,
                     modes: tuple[Mode, ...] = ("hybrid",), rerank: bool = True,
                     k_out: int | None = None, bundle: IndexBundle | None = None
                     ) -> list[StagedResult]:
    """questions = [(text, lang)]; `hybrid` must be in modes when rerank=True."""
    bundle = bundle or load_index(s.index_dir, embed_model=s.embed_model)
    kw = {"k_dense": s.retrieve_k_dense, "k_bm25": s.retrieve_k_bm25,
          "k_out": k_out or s.retrieve_k_out, "rrf_k": s.rrf_k}
    results = [StagedResult(q, lang) for q, lang in questions]

    retriever = HybridRetriever(bundle, Embedder(s.embed_model, device=s.embed_device))
    retriever.search("warm-up", mode="dense", lang="en", **kw)
    for res in tqdm(results, desc="stage 1: dense + BM25"):
        for mode in modes:
            t0 = time.perf_counter()
            res.by_mode[mode] = retriever.search(res.question, mode=mode, lang=res.lang, **kw)
            res.latency_ms[mode] = (time.perf_counter() - t0) * 1000
    retriever.embedder = None
    free_gpu()

    if rerank:
        reranker = Reranker(s.rerank_model, device=s.rerank_device)
        reranker.score("warm-up", ["warm-up"])
        for res in tqdm(results, desc="stage 2: rerank"):
            candidates = [dataclasses.replace(c) for c in res.by_mode["hybrid"]]
            t0 = time.perf_counter()
            res.reranked = reranker.rerank(res.question, candidates)
            res.latency_ms["rerank"] = (time.perf_counter() - t0) * 1000
        del reranker
        free_gpu()
    return results
