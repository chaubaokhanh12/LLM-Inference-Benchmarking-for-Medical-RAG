"""Export the exact LLM messages for every QA question, for the inference benchmark.

    python -m medrag.tools.export_prompts --tokenizer Qwen/Qwen2.5-7B-Instruct

Retrieval and reranking run once here, so every benchmarked backend receives identical
inputs and the benchmark measures only the LLM. Messages are built by the same function the
pipeline uses. With --tokenizer, prompts are trimmed to fit `--max-model-len` minus the output
budget, measured with that model's chat template (the benchmark serves max_model_len=4096).

Outputs (repo-level prompts/ folder):
    medical_rag_prompts.jsonl   {"id", "lang", "messages", "approx_input_tokens", ...}
    stats.json                  input-length distribution per language
    manifest.json               what produced the file (index, models, system prompt hash)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medrag.config import REPO_ROOT, get_settings
from medrag.evaluation.qa import QA_FILE, read_qa, remap_gold
from medrag.evaluation.staged import staged_retrieval
from medrag.generate.prompt import SYSTEM_PROMPT, fit_messages
from medrag.index.store import load_index
from medrag.pipeline import PROMPT_SAFETY_TOKENS, RAGPipeline


def chat_template_counter(tokenizer_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(messages: list[dict]) -> int:
        # Render the template to text, then tokenize: the return type of
        # apply_chat_template(tokenize=True) differs between transformers versions.
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    probe = count([{"role": "user", "content": "Bệnh sốt xuất huyết Dengue lây qua muỗi vằn."}])
    if probe < 10:
        raise RuntimeError(f"token counting looks broken ({probe} tokens for a sentence)")

    return count


def stats(values: list[int]) -> dict:
    a = np.asarray(values)
    return {"n": int(a.size), "min": int(a.min()), "p50": int(np.percentile(a, 50)),
            "p95": int(np.percentile(a, 95)), "max": int(a.max()), "mean": round(float(a.mean()), 1)}


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", type=Path, default=QA_FILE)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "prompts")
    ap.add_argument("--name", default="medical_rag_prompts")
    ap.add_argument("--types", default="answerable", help="comma list: answerable,crosslingual,unanswerable")
    ap.add_argument("--top-n", type=int, default=s.rerank_top_n)
    ap.add_argument("--tokenizer", default=None, help="HF id of the benchmarked model's tokenizer")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    types = set(args.types.split(","))
    items = [it for it in read_qa(args.qa) if it.type in types]
    bundle = load_index(s.index_dir, embed_model=s.embed_model)
    remap_gold(items, bundle.chunks)
    # Same retrieval as RAGPipeline.retrieve (no threshold), one model in memory at a time.
    staged = staged_retrieval([(it.question, it.lang) for it in items], s, bundle=bundle)
    approx = RAGPipeline(s, retriever=None, reranker=None, backend=None).count_tokens
    count = chat_template_counter(args.tokenizer) if args.tokenizer else approx
    budget = args.max_model_len - args.max_tokens - (0 if args.tokenizer else PROMPT_SAFETY_TOKENS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.name}.jsonl"
    lengths: dict[str, list[int]] = defaultdict(list)
    n_trimmed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (it, res) in enumerate(zip(items, staged, strict=True), start=1):
            contexts = res.reranked[: args.top_n]
            messages, kept = fit_messages(it.question, it.lang, contexts, budget, count)
            n_trimmed += len(kept) < len(contexts)
            n_tok = count(messages)
            lengths[it.lang].append(n_tok)
            lengths["all"].append(n_tok)
            f.write(json.dumps({
                "id": f"q_{i:04d}", "qa_id": it.id, "lang": it.lang, "type": it.type,
                "messages": messages, "approx_input_tokens": n_tok,
                "max_tokens": args.max_tokens, "temperature": 0,
                "context_chunk_ids": [c.chunk.chunk_id for c in kept],
                "gold_chunk_ids": it.gold_chunk_ids, "reference_answer": it.reference_answer,
            }, ensure_ascii=False) + "\n")

    meta = bundle.meta
    stats_doc = {
        "tokenizer": args.tokenizer or f"{s.embed_model} (approximation)",
        "max_model_len": args.max_model_len, "max_tokens": args.max_tokens, "top_n": args.top_n,
        "prompts_with_trimmed_contexts": n_trimmed,
        "input_tokens": {k: stats(v) for k, v in lengths.items()},
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "file": out_path.name, "n_prompts": len(items), "types": sorted(types),
        "qa_sha256": hashlib.sha256(args.qa.read_bytes()).hexdigest()[:16],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "embed_model": meta["embed_model"], "rerank_model": s.rerank_model,
        "chunker_hash": meta["chunker_hash"], "corpus_fingerprint": meta["corpus_fingerprint"],
        "index_built_at": meta["built_at"], **{k: stats_doc[k] for k in ("tokenizer", "top_n")},
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats_doc, indent=2), encoding="utf-8")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats_doc, indent=2))
    print(f"wrote {len(items)} prompts to {out_path}")


if __name__ == "__main__":
    main()
