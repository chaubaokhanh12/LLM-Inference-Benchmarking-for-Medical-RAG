"""Dense retrieval: multilingual-e5 embeddings in a FAISS inner-product index.

e5 models are trained with prefixes; leaving them out costs several points of recall.
The prefixes are added here, inside the embedder, so callers cannot forget them.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from medrag.config import resolve_device

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        model_kwargs = {"dtype": torch.float16} if self.device == "cuda" else {}
        self.model = SentenceTransformer(model_name, device=self.device, model_kwargs=model_kwargs)
        self.model.max_seq_length = 512

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def _encode(self, texts: list[str], progress: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=progress,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self._encode([PASSAGE_PREFIX + t for t in texts], progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([QUERY_PREFIX + t for t in texts], progress=False)


class DenseIndex:
    """Exact cosine search (vectors are L2-normalised, so inner product == cosine)."""

    def __init__(self, index: faiss.Index):
        self.index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> DenseIndex:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    def search(self, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.index.ntotal)
        return self.index.search(query_vecs, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        return cls(faiss.read_index(str(path)))

    def __len__(self) -> int:
        return self.index.ntotal
