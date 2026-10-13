"""Cross-encoder reranking.

A bi-encoder embeds question and passage separately; a cross-encoder reads them together,
which is slower but much more precise. It therefore only sees the ~30 fused candidates.
Raw logits are turned into probabilities with a sigmoid so that a threshold is meaningful.
"""

from __future__ import annotations

import threading

import numpy as np

from medrag.config import resolve_device
from medrag.types import ScoredChunk


class Reranker:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16,
                 max_length: int = 512):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        model_kwargs = {"dtype": torch.float16} if self.device == "cuda" else {}
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length,
                                  model_kwargs=model_kwargs)
        self._identity = torch.nn.Identity()
        self._lock = threading.Lock()

    def score(self, question: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            logits = self.model.predict(
                [(question, p) for p in passages], batch_size=self.batch_size,
                activation_fn=self._identity, convert_to_numpy=True, show_progress_bar=False,
            )
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))

    def rerank(self, question: str, candidates: list[ScoredChunk], top_n: int | None = None,
               min_score: float | None = None) -> list[ScoredChunk]:
        """Reorder by cross-encoder probability; optionally drop weak ones and keep top_n."""
        probs = self.score(question, [c.chunk.contextual_text for c in candidates])
        for c, p in zip(candidates, probs, strict=True):
            c.rerank_score = float(p)
            c.score = float(p)
        ranked = sorted(candidates, key=lambda c: -c.score)
        if min_score is not None:
            ranked = [c for c in ranked if c.score >= min_score]
        return ranked[:top_n] if top_n else ranked
