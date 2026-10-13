"""Smoke-test the LLM backend configured in .env with real RAG prompts.

    python scripts/check_backend.py [--n 4]

Sends the first N prompts of prompts/medical_rag_prompts.jsonl (exact pipeline inputs, so no
retrieval models are loaded) to LLM_BACKEND / LLM_MODEL / LLM_BASE_URL and prints, per prompt:
answer language, citation validity, TTFT, total time and token usage as reported by the server.
Use it after switching backends (e.g. ollama -> vllm) to confirm that only .env changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medrag.config import REPO_ROOT, get_settings
from medrag.generate.prompt import parse_citations
from medrag.lang import detect_lang
from medrag.llm import create_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts" / "medical_rag_prompts.jsonl")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    s = get_settings()
    backend = create_backend(s)
    print(f"backend={backend.name} model={backend.model} base_url={s.resolved_base_url}\n")
    with open(args.prompts, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    # half Vietnamese, half English, so both are exercised
    rows = [p for p in prompts if p["lang"] == "vi"][: (args.n + 1) // 2]
    rows += [p for p in prompts if p["lang"] == "en"][: args.n // 2]
    ok = 0
    for r in rows:
        res = backend.generate(r["messages"], s.llm_max_tokens, s.llm_temperature)
        n_ctx = len(r["context_chunk_ids"])
        cites = parse_citations(res.text)
        valid = bool(cites) and all(1 <= c <= n_ctx for c in cites)
        lang_ok = detect_lang(res.text) == r["lang"]
        ok += valid and lang_ok
        print(f"[{r['id']} {r['lang']}] ttft={res.ttft_ms:.0f} ms total={res.total_ms:.0f} ms "
              f"prompt_tokens={res.prompt_tokens} completion_tokens={res.completion_tokens} "
              f"lang_ok={lang_ok} citations={cites} valid={valid}")
        print("   ", res.text.replace("\n", " ")[:220], "\n")
    print(f"{ok}/{len(rows)} answers in the right language with valid citations")


if __name__ == "__main__":
    main()
