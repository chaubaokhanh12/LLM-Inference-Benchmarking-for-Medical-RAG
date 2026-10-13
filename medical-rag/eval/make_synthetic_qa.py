"""Build the synthetic QA set (OpenAI model as question writer and checker).

    python eval/make_synthetic_qa.py                      # full run
    python eval/make_synthetic_qa.py --apply-review       # apply eval/data/qa_review.csv

Three kinds of questions:
  answerable    written from one corpus chunk; gold = chunks containing the copied evidence
  crosslingual  an answerable question translated to the other language (same gold)
  unanswerable  written from held-out articles that were never indexed, and checked
                against what the retriever actually returns from the corpus

Quality controls: the evidence must be found verbatim in the chunk, a second LLM pass checks
that each question is answerable, self-contained and specific, and a ~10% sample is written
to qa_review.csv for manual review.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from medrag.config import PROJECT_ROOT, get_settings
from medrag.evaluation.llm_client import EvalLLM
from medrag.evaluation.qa import QA_FILE, QAItem, find_gold_chunks, normalize_ws, read_qa, write_qa
from medrag.index.sparse import tokenize
from medrag.index.store import CHUNKS_FILE, read_chunks
from medrag.ingest.chunker import Chunker, split_sentences
from medrag.ingest.loader import load_documents
from medrag.types import Chunk

REVIEW_FILE = PROJECT_ROOT / "eval" / "data" / "qa_review.csv"
LANG_NAME = {"vi": "Vietnamese", "en": "English"}
SKIP_SECTIONS = re.compile(r"history|etymology|lịch sử|từ nguyên|tên gọi|research|nghiên cứu",
                           re.IGNORECASE)
PASSAGE_REFERENCE = re.compile(
    r"\b(passage|the text|this article|the article|according to the)\b|đoạn văn|bài viết|văn bản trên|theo tài liệu",
    re.IGNORECASE,
)

GEN_SYSTEM = """You create test questions for a medical document search system.
Given one passage, write {n} question(s) that the passage answers.

Requirements:
- Write in {language}.
- Each question must be self-contained, phrased the way a patient, student or clinician would ask it: name the disease, drug or topic explicitly. Never refer to "the passage", "the text", "this article", "đoạn văn" or "bài viết".
- Paraphrase: do not copy more than three consecutive words from the passage, except names of diseases, drugs and other technical terms.
- The answer must be stated in the passage, not inferred and not merely common knowledge.
- Prefer specific medical facts: symptoms, causes, transmission, diagnosis, doses, side effects, contraindications, prevention, prognosis, epidemiology.
- For each question also give "answer" (one or two sentences in {language}) and "evidence": ONE sentence or table row copied EXACTLY, character for character, from the passage, that supports the answer.

Return JSON: {{"items": [{{"question": "...", "answer": "...", "evidence": "..."}}]}}
If the passage has no answerable medical fact (for example only names or anecdotes), return {{"items": []}}."""

VERIFY_SYSTEM = """You review test questions for a medical document search system.
Given a passage and a question, decide:
- "answerable": the passage alone fully answers the question.
- "self_contained": the question makes sense without seeing the passage (explicit topic, no reference to "the passage/text").
- "specific": the question asks for a specific fact with one clear answer, not a broad essay.
Return JSON: {"answerable": true|false, "self_contained": true|false, "specific": true|false, "reason": "<short>"}"""

TRANSLATE_SYSTEM = """Translate the medical question into {language}. Use the usual {language} medical terms and keep drug and disease names correct. Do not answer it.
Return JSON: {{"question": "..."}}"""

ANSWERED_SYSTEM = """You check whether a search result answers a medical question.
Given a question and several numbered passages, decide whether ANY passage contains the answer.
Return JSON: {"answered": true|false, "passage": <number or null>}"""


def eligible(c: Chunk) -> bool:
    return (c.n_tokens >= 100 and not c.oversized
            and not any(SKIP_SECTIONS.search(p) for p in c.section_path)
            and (len(split_sentences(c.text)) >= 2 or "|" in c.text))


def sample_chunks(chunks: list[Chunk], lang: str, n: int, per_doc: int, rng: random.Random) -> list[Chunk]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.lang == lang and eligible(c):
            by_doc[c.doc_id].append(c)
    picked: list[Chunk] = []
    for cs in by_doc.values():
        picked += rng.sample(cs, min(per_doc, len(cs)))
    rng.shuffle(picked)
    return picked[:n]


def snap_evidence(evidence: str, chunk: Chunk) -> str | None:
    """Return the evidence if it is verbatim in the chunk, else the chunk sentence/row that
    shares at least 70% of its words, else None."""
    if normalize_ws(evidence) and normalize_ws(evidence) in normalize_ws(chunk.text):
        return evidence.strip()
    words = set(normalize_ws(evidence).split())
    if not words:
        return None
    candidates = [ln for ln in chunk.text.split("\n") if ln.strip().startswith("|")]
    candidates += [s for para in chunk.text.split("\n") for s in split_sentences(para)]
    best, best_score = None, 0.0
    for cand in candidates:
        cw = set(normalize_ws(cand).split())
        score = len(words & cw) / len(words)
        if score > best_score:
            best, best_score = cand.strip(), score
    return best if best_score >= 0.7 else None


def lexical_overlap(question: str, chunk: Chunk) -> float:
    q = set(tokenize(question, chunk.lang))
    return len(q & set(tokenize(chunk.text, chunk.lang))) / len(q) if q else 0.0


def passage_block(c: Chunk) -> str:
    return f"Document: {c.title}\nSection: {' > '.join(c.section_path) or '-'}\n\nPassage:\n{c.text}"


def write_questions(llm: EvalLLM, chunks: list[Chunk], n_per_chunk: int, workers: int,
                    desc: str) -> list[tuple[Chunk, dict]]:
    def one(c: Chunk) -> list[tuple[Chunk, dict]]:
        sys_prompt = GEN_SYSTEM.format(n=n_per_chunk, language=LANG_NAME[c.lang])
        out = llm.json_call(sys_prompt, passage_block(c), temperature=0.7, tag="gen-v1")
        return [(c, it) for it in out.get("items", []) if isinstance(it, dict)]

    with ThreadPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(one, chunks), total=len(chunks), desc=desc))
    return [pair for r in results for pair in r]


def verify(llm: EvalLLM, pairs: list[tuple[Chunk, dict]], workers: int) -> list[dict]:
    def one(pair: tuple[Chunk, dict]) -> dict:
        c, it = pair
        return llm.json_call(VERIFY_SYSTEM, f"{passage_block(c)}\n\nQuestion: {it['question']}",
                             tag="verify-v1")

    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(one, pairs), total=len(pairs), desc="verifying"))


def build_answerable(llm: EvalLLM, chunks: list[Chunk], args, rng: random.Random) -> list[QAItem]:
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    items: list[QAItem] = []
    for lang in ("vi", "en"):
        picked = sample_chunks(chunks, lang, int(args.per_lang * 1.4), args.per_doc, rng)
        pairs = write_questions(llm, picked, args.questions_per_chunk, args.workers, f"writing {lang}")
        pairs = [(c, it) for c, it in pairs
                 if isinstance(it.get("question"), str) and 10 <= len(it["question"]) <= 300
                 and not PASSAGE_REFERENCE.search(it["question"])]
        checks = verify(llm, pairs, args.workers)
        seen: set[str] = set()
        kept: list[QAItem] = []
        stats = defaultdict(int)
        for (c, it), chk in zip(pairs, checks, strict=True):
            evidence = snap_evidence(str(it.get("evidence", "")), c)
            key = normalize_ws(it["question"])
            if not evidence:
                stats["evidence_not_in_chunk"] += 1
            elif not (chk.get("answerable") and chk.get("self_contained") and chk.get("specific")):
                stats["failed_check"] += 1
            elif key in seen:
                stats["duplicate"] += 1
            else:
                seen.add(key)
                gold = find_gold_chunks(evidence, c.doc_id, by_doc) or [c.chunk_id]
                kept.append(QAItem(
                    id="", question=it["question"].strip(), lang=lang, type="answerable",
                    reference_answer=str(it.get("answer", "")).strip(), evidence=evidence,
                    gold_doc_id=c.doc_id, gold_chunk_ids=gold, source_chunk_id=c.chunk_id,
                    generator_model=llm.model,
                    meta={"lexical_overlap": round(lexical_overlap(it["question"], c), 3)},
                ))
        print(f"[{lang}] {len(pairs)} written -> {len(kept)} kept; rejected: {dict(stats)}")
        items += kept[: args.per_lang]
    return items


def build_crosslingual(llm: EvalLLM, answerable: list[QAItem], n_per_lang: int,
                       rng: random.Random) -> list[QAItem]:
    out: list[QAItem] = []
    for src in ("vi", "en"):
        tgt = "en" if src == "vi" else "vi"
        pool = [it for it in answerable if it.lang == src]
        for it in rng.sample(pool, min(n_per_lang, len(pool))):
            r = llm.json_call(TRANSLATE_SYSTEM.format(language=LANG_NAME[tgt]), it.question,
                              tag="translate-v1")
            if isinstance(r.get("question"), str) and r["question"].strip():
                out.append(QAItem(
                    id="", question=r["question"].strip(), lang=tgt, type="crosslingual",
                    reference_answer=it.reference_answer, evidence=it.evidence,
                    gold_doc_id=it.gold_doc_id, gold_chunk_ids=list(it.gold_chunk_ids),
                    source_chunk_id=it.source_chunk_id, generator_model=llm.model,
                    meta={"translated_from": src},
                ))
    return out


def build_unanswerable(llm: EvalLLM, args, rng: random.Random) -> list[QAItem]:
    """Questions from held-out articles, kept only if the corpus does not answer them.

    The check reads the top-10 hybrid (dense + BM25) results: wider than the 5 reranked
    passages the pipeline would show the LLM, so it is the stricter test, and it does not need
    the reranker in memory.
    """
    from medrag.index.dense import Embedder
    from medrag.index.store import load_index
    from medrag.retrieve.hybrid import HybridRetriever

    holdout = load_documents(PROJECT_ROOT / "data" / "holdout")
    if not holdout:
        print("no held-out documents (data/holdout); skipping unanswerable questions")
        return []
    chunker = Chunker()
    hchunks = [c for d in holdout for c in chunker.chunk(d)]
    s = get_settings()
    retriever = HybridRetriever(load_index(s.index_dir, s.embed_model),
                                Embedder(s.embed_model, device=s.embed_device))
    out: list[QAItem] = []
    for lang in ("vi", "en"):
        picked = sample_chunks(hchunks, lang, int(args.unanswerable_per_lang * 1.5), 2, rng)
        pairs = write_questions(llm, picked, 1, args.workers, f"held-out {lang}")
        kept = 0
        for c, it in pairs:
            q = it.get("question")
            if not isinstance(q, str) or PASSAGE_REFERENCE.search(q) or kept >= args.unanswerable_per_lang:
                continue
            ctx = retriever.search(q, k_out=10, lang=lang)
            passages = "\n\n".join(f"[{i}] {sc.chunk.contextual_text}" for i, sc in enumerate(ctx, 1))
            r = llm.json_call(ANSWERED_SYSTEM, f"Question: {q}\n\n{passages}", tag="answered-v1")
            if r.get("answered") is False:
                kept += 1
                out.append(QAItem(
                    id="", question=q.strip(), lang=lang, type="unanswerable",
                    reference_answer="", evidence="", generator_model=llm.model,
                    meta={"holdout_doc": c.doc_id},
                ))
        print(f"[{lang}] unanswerable kept: {kept}")
    return out


def assign_ids(items: list[QAItem]) -> None:
    counters: dict[tuple[str, str], int] = defaultdict(int)
    prefix = {"answerable": "a", "crosslingual": "x", "unanswerable": "u"}
    for it in items:
        counters[(it.type, it.lang)] += 1
        it.id = f"{prefix[it.type]}-{it.lang}-{counters[(it.type, it.lang)]:04d}"


def write_review_sample(items: list[QAItem], share: float, rng: random.Random) -> None:
    sample = rng.sample(items, max(1, int(len(items) * share)))
    with open(REVIEW_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "lang", "question", "reference_answer", "evidence", "keep", "note"])
        for it in sorted(sample, key=lambda i: i.id):
            w.writerow([it.id, it.type, it.lang, it.question, it.reference_answer, it.evidence, "", ""])
    print(f"review sample ({len(sample)} items) -> {REVIEW_FILE.relative_to(PROJECT_ROOT)}; "
          "mark keep=n to drop, then run with --apply-review")


def apply_review(path: Path) -> None:
    items = read_qa(path)
    with open(REVIEW_FILE, encoding="utf-8-sig") as f:
        drop = {row["id"] for row in csv.DictReader(f)
                if row.get("keep", "").strip().lower() in {"n", "no", "0", "x"}}
    kept = [it for it in items if it.id not in drop]
    write_qa(path, kept)
    print(f"dropped {len(items) - len(kept)} reviewed items; {len(kept)} remain")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=QA_FILE)
    ap.add_argument("--per-lang", type=int, default=150)
    ap.add_argument("--per-doc", type=int, default=3, help="max source chunks per document")
    ap.add_argument("--questions-per-chunk", type=int, default=1)
    ap.add_argument("--crosslingual-per-lang", type=int, default=20)
    ap.add_argument("--unanswerable-per-lang", type=int, default=25)
    ap.add_argument("--review-share", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--apply-review", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.apply_review:
        apply_review(args.out)
        return
    rng = random.Random(args.seed)
    llm = EvalLLM()
    chunks = read_chunks(get_settings().index_dir / CHUNKS_FILE)
    answerable = build_answerable(llm, chunks, args, rng)
    crosslingual = build_crosslingual(llm, answerable, args.crosslingual_per_lang, rng)
    unanswerable = build_unanswerable(llm, args, rng)
    items = answerable + crosslingual + unanswerable
    assign_ids(items)
    write_qa(args.out, items)
    counts = defaultdict(int)
    for it in items:
        counts[f"{it.type}/{it.lang}"] += 1
    print(f"wrote {len(items)} items to {args.out}: {dict(counts)}")
    print(f"OpenAI usage: {llm.usage}")
    write_review_sample(items, args.review_share, rng)


if __name__ == "__main__":
    main()
