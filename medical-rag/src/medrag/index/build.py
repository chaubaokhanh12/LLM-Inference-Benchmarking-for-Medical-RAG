"""Build the retrieval index from raw documents.

    python -m medrag.index.build --input data/raw --out data/index

The index is written to a temporary directory and swapped in at the end, so a failed build
never leaves a half-written index behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medrag import __version__
from medrag.config import PROJECT_ROOT, get_settings
from medrag.index.dense import PASSAGE_PREFIX, QUERY_PREFIX, DenseIndex, Embedder
from medrag.index.sparse import TOKENIZER_VERSION, BM25Index
from medrag.index.store import BM25_FILE, CHUNKS_FILE, DENSE_FILE, META_FILE, write_chunks
from medrag.ingest.chunker import ChunkerConfig, chunk_documents
from medrag.ingest.loader import SUPPORTED, load_documents

log = logging.getLogger("medrag.build")


def corpus_fingerprint(input_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            h.update(p.relative_to(input_dir).as_posix().encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def build(input_dir: Path, out_dir: Path, *, embed_model: str, device: str, batch_size: int,
          chunker_cfg: ChunkerConfig, bm25_syllables: bool, processed_path: Path) -> dict:
    t0 = time.perf_counter()
    docs = load_documents(input_dir)
    if not docs:
        raise SystemExit(f"no documents found under {input_dir}")
    log.info("loaded %d documents (%s)", len(docs), dict(Counter(d.lang for d in docs)))

    chunks = chunk_documents(docs, chunker_cfg)
    write_chunks(processed_path, chunks)
    toks = np.array([c.n_tokens for c in chunks])
    log.info("%d chunks; tokens p50=%d p95=%d max=%d; oversized=%d", len(chunks),
             np.percentile(toks, 50), np.percentile(toks, 95), toks.max(),
             sum(c.oversized for c in chunks))

    texts = [c.contextual_text for c in chunks]
    embedder = Embedder(embed_model, device=device, batch_size=batch_size)
    log.info("embedding on %s ...", embedder.device)
    vectors = embedder.embed_passages(texts, progress=True)

    log.info("building BM25 (Vietnamese word segmentation may take a minute) ...")
    bm25 = BM25Index.build(texts, [c.lang for c in chunks], add_syllables=bm25_syllables)

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    write_chunks(tmp / CHUNKS_FILE, chunks)
    DenseIndex.build(vectors).save(tmp / DENSE_FILE)
    bm25.save(tmp / BM25_FILE)
    meta = {
        "medrag_version": __version__,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embed_model": embed_model,
        "embed_dim": int(vectors.shape[1]),
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "chunks_by_lang": dict(Counter(c.lang for c in chunks)),
        "n_oversized": sum(c.oversized for c in chunks),
        "chunker_hash": chunker_cfg.hash(),
        "chunker_config": asdict(chunker_cfg),
        "bm25": {"tokenizer": TOKENIZER_VERSION, "add_syllables": bm25_syllables},
        "corpus_fingerprint": corpus_fingerprint(input_dir),
        "input_dir": input_dir.relative_to(PROJECT_ROOT).as_posix()
        if input_dir.is_relative_to(PROJECT_ROOT) else str(input_dir),
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    (tmp / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    old = out_dir.with_name(out_dir.name + ".old")
    if out_dir.exists():
        if old.exists():
            shutil.rmtree(old)
        out_dir.rename(old)
    tmp.rename(out_dir)
    if old.exists():
        shutil.rmtree(old)
    log.info("index written to %s in %.1fs", out_dir, meta["build_seconds"])
    return meta


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Build the dense + BM25 index.")
    ap.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=s.index_dir)
    ap.add_argument("--processed", type=Path, default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
    ap.add_argument("--device", default=s.embed_device)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split-oversized", action="store_true",
                    help="split tables/lists longer than the limit at row/item boundaries")
    ap.add_argument("--bm25-syllables", action="store_true",
                    help="also index the syllables of Vietnamese compound words")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    resolve = lambda p: p if p.is_absolute() else (Path.cwd() / p).resolve()  # noqa: E731
    build(resolve(args.input), resolve(args.out), embed_model=s.embed_model, device=args.device,
          batch_size=args.batch_size,
          chunker_cfg=ChunkerConfig(tokenizer_name=s.embed_model,
                                    split_oversized=args.split_oversized),
          bm25_syllables=args.bm25_syllables, processed_path=resolve(args.processed))


if __name__ == "__main__":
    main()
