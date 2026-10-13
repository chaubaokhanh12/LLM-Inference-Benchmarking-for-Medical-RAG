"""Prompt construction and citation parsing.

The system prompt is a constant: byte-identical for every request and placed first, so an
inference server with prefix caching can reuse its KV cache. Everything request-specific
(contexts, question, answer language) goes into the user message.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from medrag.text import strip_accents
from medrag.types import Chunk, ScoredChunk

NOT_FOUND = {
    "vi": "Không tìm thấy thông tin này trong tài liệu.",
    "en": "I could not find this information in the documents.",
}
# Closing instruction of the user message. Small models follow what is nearest the end of the
# prompt, so the answer language and the citation format are repeated here (constant per
# language, so the system prompt stays identical across requests).
ANSWER_LANGUAGE = {
    "vi": "Hãy trả lời bằng tiếng Việt. Cuối mỗi câu ghi số đoạn nguồn trong ngoặc vuông, như [1] hoặc [2][3].",
    "en": "Answer in English. End each sentence with its source passage number in square brackets, like [1] or [2][3].",
}

SYSTEM_PROMPT = f"""You are a medical information assistant. Answer the question using ONLY the numbered context passages in the user message.

Rules:
1. Use only facts stated in the context passages. Do not add outside knowledge.
2. Answer in the language requested at the end of the user message, even if the passages are in another language.
3. Cite the source of every factual sentence with its passage number in square brackets, e.g. [1] or [2][3]. Only cite passage numbers that exist.
4. You may report doses and treatments exactly as the passages state them, with citations. Do not diagnose or choose a dose for a specific person; when it is relevant, advise seeing a doctor or pharmacist.
5. If the passages do not contain the answer, reply with exactly one sentence: "{NOT_FOUND['vi']}" for a Vietnamese answer, or "{NOT_FOUND['en']}" for an English answer.
6. Be concise: at most five sentences or a short list."""


def format_context(index: int, chunk: Chunk) -> str:
    return f"[{index}] {chunk.header}\n{chunk.text}"


def build_messages(question: str, lang: str, contexts: list[Chunk]) -> list[dict]:
    passages = "\n\n".join(format_context(i, c) for i, c in enumerate(contexts, start=1))
    user = f"Context:\n{passages}\n\nQuestion: {question.strip()}\n{ANSWER_LANGUAGE[lang]}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def fit_messages(question: str, lang: str, contexts: list[ScoredChunk], max_input_tokens: int,
                 count_tokens: Callable[[list[dict]], int]
                 ) -> tuple[list[dict], list[ScoredChunk]]:
    """Drop the lowest-ranked contexts until the prompt fits the model's context window."""
    kept = list(contexts)
    while True:
        messages = build_messages(question, lang, [c.chunk for c in kept])
        if len(kept) <= 1 or count_tokens(messages) <= max_input_tokens:
            return messages, kept
        kept.pop()


_CITATION = re.compile(r"\[(\s*\d+\s*(?:[,;–-]\s*\d+\s*)*)\]")


def parse_citations(answer: str) -> list[int]:
    """Passage numbers cited in the answer, in order of first appearance: "[1][2, 4-5]"."""
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        for part in re.split(r"[,;]", m.group(1)):
            bounds = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
            nums = range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 else bounds
            for n in nums:
                if n not in seen:
                    seen.append(n)
    return seen


def build_citations(contexts: list[ScoredChunk], answer: str) -> tuple[list[dict], list[int]]:
    """Citation records for every context passed to the LLM, plus cited numbers that don't exist."""
    cited = parse_citations(answer)
    records = []
    for i, sc in enumerate(contexts, start=1):
        c = sc.chunk
        records.append({
            "index": i, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title,
            "section_path": c.section_path, "score": sc.score, "cited": i in cited,
        })
    invalid = [n for n in cited if not 1 <= n <= len(contexts)]
    return records, invalid


_NOT_FOUND_PATTERNS = [
    strip_accents(NOT_FOUND["vi"]).lower().rstrip("."),
    NOT_FOUND["en"].lower().rstrip("."),
    "khong tim thay thong tin", "khong co thong tin", "could not find", "not found in the",
    "does not contain", "do not contain", "no information",
]


def is_not_found(answer: str) -> bool:
    norm = strip_accents(answer).lower()
    return any(p in norm for p in _NOT_FOUND_PATTERNS) and len(answer) < 300
