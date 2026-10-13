"""On-disk layout of an index directory and loading it back.

    data/index/
    ├── chunks.jsonl   row i of the FAISS index and of BM25 == line i of this file
    ├── dense.index    FAISS IndexFlatIP
    ├── bm25.pkl       BM25Okapi + tokenizer options
    └── meta.json      models, chunker config hash, build date, corpus fingerprint
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medrag.index.dense import DenseIndex
from medrag.index.sparse import BM25Index
from medrag.types import Chunk

CHUNKS_FILE = "chunks.jsonl"
DENSE_FILE = "dense.index"
BM25_FILE = "bm25.pkl"
META_FILE = "meta.json"


class IndexMismatchError(RuntimeError):
    pass


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    dense: DenseIndex
    bm25: BM25Index
    meta: dict


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def load_index(index_dir: Path, embed_model: str | None = None) -> IndexBundle:
    index_dir = Path(index_dir)
    meta_path = index_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no index at {index_dir}; build it with: python -m medrag.index.build"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if embed_model and meta["embed_model"] != embed_model:
        raise IndexMismatchError(
            f"index was built with {meta['embed_model']!r} but EMBED_MODEL={embed_model!r}; "
            "rebuild the index or fix the setting"
        )
    chunks = read_chunks(index_dir / CHUNKS_FILE)
    dense = DenseIndex.load(index_dir / DENSE_FILE)
    bm25 = BM25Index.load(index_dir / BM25_FILE)
    if not (len(chunks) == len(dense) == meta["n_chunks"]):
        raise IndexMismatchError(
            f"index files disagree: {len(chunks)} chunks, {len(dense)} vectors, "
            f"meta says {meta['n_chunks']}"
        )
    return IndexBundle(chunks=chunks, dense=dense, bm25=bm25, meta=meta)
