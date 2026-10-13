"""Synthetic QA items and how their gold chunks survive re-chunking.

Each item stores the `evidence` sentence copied from its source chunk. After the chunker
changes, the gold chunks are found again as the chunks of the same document whose text
contains that evidence, so the QA set does not have to be regenerated.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from medrag.config import PROJECT_ROOT
from medrag.types import Chunk

QA_FILE = PROJECT_ROOT / "eval" / "data" / "qa.jsonl"


@dataclass
class QAItem:
    id: str
    question: str
    lang: str
    type: str  # answerable | crosslingual | unanswerable
    reference_answer: str = ""
    evidence: str = ""
    gold_doc_id: str | None = None
    gold_chunk_ids: list[str] = field(default_factory=list)
    source_chunk_id: str | None = None  # chunk the question was written from
    generator_model: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def read_qa(path: Path) -> list[QAItem]:
    with open(path, encoding="utf-8") as f:
        return [QAItem(**json.loads(line)) for line in f if line.strip()]


def write_qa(path: Path, items: list[QAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def find_gold_chunks(evidence: str, doc_id: str, chunks_by_doc: dict[str, list[Chunk]]) -> list[str]:
    """Chunks of `doc_id` containing the evidence (whitespace/case-insensitive)."""
    ev = normalize_ws(evidence)
    if not ev:
        return []
    return [c.chunk_id for c in chunks_by_doc.get(doc_id, []) if ev in normalize_ws(c.text)]


def remap_gold(items: list[QAItem], chunks: list[Chunk]) -> tuple[int, int]:
    """Recompute gold_chunk_ids against the current index; returns (remapped, lost)."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    remapped = lost = 0
    for it in items:
        if it.type == "unanswerable" or not it.gold_doc_id:
            continue
        ids = find_gold_chunks(it.evidence, it.gold_doc_id, by_doc)
        if ids:
            remapped += ids != it.gold_chunk_ids
            it.gold_chunk_ids = ids
        else:
            lost += 1
    return remapped, lost
