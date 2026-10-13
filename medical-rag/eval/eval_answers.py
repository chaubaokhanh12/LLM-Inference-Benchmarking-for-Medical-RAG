"""Answer evaluation of the full pipeline with the LLM backend configured in .env.

    python eval/eval_answers.py [--n-answerable 100] [--tag ollama-qwen2.5-3b]

Metrics
  faithfulness      share of the answer's claims supported by the given contexts (LLM judge)
  relevance         1-5, does the answer address the question (LLM judge)
  correctness       1-5, agreement with the reference answer (LLM judge)
  language match    answer language == question language
  citations         answers with >=1 valid [i]; answers citing a non-existent passage
  abstention        unanswerable -> "not found" (good); answerable -> "not found" (bad)
  latency           per-stage timings and the share of total time spent in the LLM
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from medrag.config import PROJECT_ROOT, get_settings
from medrag.evaluation.llm_client import EvalLLM
from medrag.evaluation.metrics import bootstrap_ci
from medrag.evaluation.qa import QA_FILE, read_qa
from medrag.evaluation.staged import staged_retrieval
from medrag.lang import detect_lang
from medrag.llm import create_backend
from medrag.pipeline import RAGPipeline, Retrieval

RESULTS = PROJECT_ROOT / "eval" / "results"

JUDGE_SYSTEM = """You grade answers of a medical question-answering system that must answer only from retrieved passages.
You get the question, the numbered passages the system saw, its answer, and possibly a reference answer.

1. List the factual claims made in the answer (ignore citations, greetings and advice to see a doctor). For each claim decide if the passages support it.
2. "relevance" 1-5: how directly and completely the answer addresses the question (5 = fully; 1 = off-topic). A "not found" reply scores 1 unless the passages truly lack the answer, then 5.
3. "correctness" 1-5 against the reference answer (5 = same facts, 1 = contradicts or misses them). Use null when no reference is given.
Return JSON: {"claims": [{"claim": "...", "supported": true|false}], "relevance": <1-5>, "correctness": <1-5 or null>}"""


def judge(llm: EvalLLM, item, result) -> dict:
    passages = "\n\n".join(
        f"[{i}] {c['title']} > {' > '.join(c['section_path'])}\n{c['text']}"
        for i, c in enumerate(result.contexts, 1)
    ) or "(no passages)"
    ref = item.reference_answer or "(none)"
    user = (f"Question: {item.question}\n\nPassages:\n{passages}\n\nAnswer:\n{result.answer}\n\n"
            f"Reference answer: {ref}")
    return llm.json_call(JUDGE_SYSTEM, user, max_tokens=1200, tag="judge-v1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", type=Path, default=QA_FILE)
    ap.add_argument("--n-answerable", type=int, default=100)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--two-phase", action="store_true",
                    help="retrieve everything first, free the GPU, then generate (small GPUs)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    s = get_settings()
    tag = args.tag or f"{s.llm_backend}-{s.llm_model}".replace(":", "-").replace("/", "-")
    rng = random.Random(args.seed)
    items = read_qa(args.qa)
    answerable = [it for it in items if it.type == "answerable"]
    sample = []
    for lang in ("vi", "en"):
        pool = [it for it in answerable if it.lang == lang]
        sample += rng.sample(pool, min(args.n_answerable // 2, len(pool)))
    sample += [it for it in items if it.type in ("crosslingual", "unanswerable")]

    if args.two_phase:
        # no models loaded here: staged_retrieval loads encoder, then reranker, one at a time
        pipeline = RAGPipeline(s, retriever=None, reranker=None, backend=create_backend(s))
        results = answer_two_phase(pipeline, sample, s)
    else:
        pipeline = RAGPipeline.from_settings(s)
        pipeline.logger = None
        pipeline.warmup()
        results = [pipeline.answer(it.question) for it in tqdm(sample, desc=f"answering ({tag})")]

    llm = EvalLLM()
    with ThreadPoolExecutor(8) as ex:
        verdicts = list(tqdm(ex.map(lambda p: judge(llm, *p), zip(sample, results, strict=True)),
                             total=len(sample), desc="judging"))

    detail = []
    for it, r, v in zip(sample, results, verdicts, strict=True):
        claims = v.get("claims") or []
        faith = (sum(bool(c.get("supported")) for c in claims) / len(claims)
                 if claims and not r.not_found else None)
        valid_cites = [c for c in r.citations if c["cited"]]
        detail.append({
            "id": it.id, "type": it.type, "lang": it.lang, "question": it.question,
            "answer": r.answer, "not_found": r.not_found,
            "answer_lang": detect_lang(r.answer), "lang_match": detect_lang(r.answer) == it.lang,
            "has_citation": bool(valid_cites), "invalid_citations": r.invalid_citations,
            "faithfulness": faith, "relevance": v.get("relevance"),
            "correctness": v.get("correctness") if it.type != "unanswerable" else None,
            "context_ids": [c["chunk_id"] for c in r.contexts],
            "gold_in_context": bool(set(it.gold_chunk_ids) & {c["chunk_id"] for c in r.contexts}),
            **{f"t_{k}": v_ for k, v_ in r.timings.items()},
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"answers_{tag}_detail.jsonl", "w", encoding="utf-8") as f:
        for d in detail:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    summary = summarise(detail)
    with open(RESULTS / f"answers_{tag}_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "group", "value", "ci_low", "ci_high", "n"])
        w.writerows(summary)
    md = render(summary, detail, tag, s)
    (RESULTS / f"answers_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"OpenAI judge usage: {llm.usage}")


def answer_two_phase(pipeline: RAGPipeline, sample, s) -> list:
    """Same steps as RAGPipeline.answer, split so the encoder, the reranker and the LLM never
    share memory. total_ms is the sum of the stage timings."""
    staged = staged_retrieval([(it.question, it.lang) for it in sample], s)
    prepared = []
    for it, res in zip(sample, staged, strict=True):
        contexts = res.reranked
        if s.rerank_min_score is not None:
            contexts = [c for c in contexts if c.score >= s.rerank_min_score]
        contexts = contexts[: s.rerank_top_n]
        r = Retrieval(it.lang, contexts, len(res.by_mode["hybrid"]),
                      {"retrieve_ms": res.latency_ms["hybrid"], "rerank_ms": res.latency_ms["rerank"]})
        messages, contexts = (pipeline.prepare_messages(it.question, r.lang, r.contexts)
                              if r.contexts else (None, []))
        prepared.append((it, r, messages, contexts, sum(r.timings.values())))
    pipeline.backend.generate([{"role": "user", "content": "Hi"}], max_tokens=1, temperature=0)
    results = []
    for it, r, messages, contexts, retrieval_ms in tqdm(prepared, desc="phase 2: generate"):
        llm = (pipeline.backend.generate(messages, s.llm_max_tokens, s.llm_temperature)
               if messages else None)
        total = retrieval_ms + (llm.total_ms if llm else 0.0)
        results.append(pipeline.build_result(it.id, it.question, r, contexts, llm, total))
    return results


def summarise(detail: list[dict]) -> list[list]:
    rows = []

    def add(metric: str, group: str, values: list[float]) -> None:
        values = [float(v) for v in values if v is not None]
        if values:
            mean, lo, hi = bootstrap_ci(values)
            rows.append([metric, group, round(mean, 4), round(lo, 4), round(hi, 4), len(values)])

    groups = defaultdict(list)
    for d in detail:
        groups[d["type"] if d["type"] != "answerable" else f"answerable/{d['lang']}"].append(d)
        if d["type"] == "answerable":
            groups["answerable/all"].append(d)
    for g, ds in sorted(groups.items()):
        if g == "unanswerable":
            add("abstained (good)", g, [d["not_found"] for d in ds])
            add("language match", g, [d["lang_match"] for d in ds])
            continue
        add("faithfulness", g, [d["faithfulness"] for d in ds])
        add("relevance (1-5)", g, [d["relevance"] for d in ds])
        add("correctness (1-5)", g, [d["correctness"] for d in ds])
        add("language match", g, [d["lang_match"] for d in ds])
        add("has valid citation", g, [d["has_citation"] for d in ds if not d["not_found"]])
        add("cites missing passage", g, [bool(d["invalid_citations"]) for d in ds])
        add("abstained (bad)", g, [d["not_found"] for d in ds])
        add("gold chunk in context", g, [d["gold_in_context"] for d in ds])
    return rows


def render(summary: list[list], detail: list[dict], tag: str, s) -> str:
    lines = [f"Backend `{s.llm_backend}` / model `{s.llm_model}` (tag `{tag}`), judge `{s.eval_llm_model}`\n",
             "| Metric | Group | Value | 95% CI | n |", "| --- | --- | --- | --- | --- |"]
    for m, g, v, lo, hi, n in summary:
        lines.append(f"| {m} | {g} | {v:.3f} | [{lo:.3f}, {hi:.3f}] | {n} |")
    timed = [d for d in detail if d["t_llm_total_ms"] is not None]
    if timed:
        def med(key):
            return statistics.median(d[key] for d in timed if d[key] is not None)
        share = statistics.median(d["t_llm_total_ms"] / d["t_total_ms"] for d in timed)
        lines += ["", "Latency (median over requests that called the LLM)\n",
                  "| retrieve | rerank | LLM TTFT | LLM total | end-to-end | LLM share |",
                  "| --- | --- | --- | --- | --- | --- |",
                  f"| {med('t_retrieve_ms'):.0f} ms | {med('t_rerank_ms'):.0f} ms | "
                  f"{med('t_llm_ttft_ms'):.0f} ms | {med('t_llm_total_ms'):.0f} ms | "
                  f"{med('t_total_ms'):.0f} ms | {share:.0%} |"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
