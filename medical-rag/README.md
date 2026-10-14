# Medical RAG (Vietnamese / English)

Question answering over medical documents in Vietnamese and English, with retrieval written
from scratch (no LangChain / LlamaIndex), a FastAPI service, a swappable LLM backend, an
evaluation suite, and an export of the exact LLM inputs for the
[inference benchmark](../inference-bench/).

## Architecture

```
             ┌──────────────────────── ingestion (offline) ─────────────────────────┐
 Wikipedia ─►│ loader → cleaner → section-aware chunker ──► chunks.jsonl            │
 VI + EN     │                    └─► e5 "passage:" embeddings ──► FAISS (dense)   │
             │                    └─► pyvi / regex tokens ──────► BM25 (sparse)    │
             └───────────────────────────────────────────────────────────────────────┘
             ┌──────────────────────── query (online) ──────────────────────────────┐
 question ──►│ detect language → dense top-50 ─┐                                     │
             │                → BM25 top-50  ──┴► RRF fusion → top-30              │
             │                → cross-encoder rerank → top-5                       │
             │                → prompt (fixed system prompt + numbered contexts)  │
             │                → LLM backend (Ollama | vLLM | OpenAI | Groq | Gemini)│
             │                → answer with [i] citations + per-stage timings      │
             └───────────────────────────────────────────────────────────────────────┘
```

| Stage | Choice | Why |
| --- | --- | --- |
| Chunking | Section-aware, 350–450 tokens, tables and lists never split | A dose table cut in half answers nothing |
| Dense | `intfloat/multilingual-e5-base`, `query:` / `passage:` prefixes, FAISS `IndexFlatIP` | Multilingual, small enough for a 4 GB GPU; exact search is fine at this scale |
| Sparse | BM25 with `pyvi` word segmentation for Vietnamese | "đau đầu" must be one term (`đau_đầu`), not two |
| Fusion | Reciprocal Rank Fusion (k = 60) | Combines ranks, so cosine and BM25 scores need no calibration |
| Rerank | `BAAI/bge-reranker-v2-m3` cross-encoder | Reads question and passage together; much more precise on 30 candidates |
| LLM | One OpenAI-compatible client + Gemini | Switching backend is a `.env` change |

### Design notes

- **Chunker** ([chunker.py](src/medrag/ingest/chunker.py)). Headings of mixed styles (Markdown,
  `Chương/Mục/Điều`, `I.`, `1.`, `1.1.`) are nested with a stack ordered by first appearance.
  A numbered line is a heading only if it does not continue or start a numbered run (otherwise
  "1. Sốt / 2. Đau đầu" would become two sections). Roman numerals require a capital letter
  after them, so "V. cholerae" stays text. The 512-token limit is checked on the exact string
  that is embedded (`passage: ` + title/section header + text). An oversized table or list
  stays whole and is flagged `oversized`; `--split-oversized` splits it at row/item
  boundaries, repeating the table header.
- **Prompt** ([prompt.py](src/medrag/generate/prompt.py)). The system prompt is a constant
  placed first, so a server with prefix caching can reuse it. The answer language is
  requested at the end of the user message. If no passage clears the rerank threshold the
  pipeline answers "not found" without calling the LLM.
- **Timing** ([pipeline.py](src/medrag/pipeline.py)). Every backend streams internally so
  time-to-first-token is measured even for blocking calls. Each request records
  `retrieve_ms`, `rerank_ms`, `llm_ttft_ms`, `llm_total_ms`, `total_ms` (and is logged to
  `logs/requests.jsonl`).

## Setup

```bash
conda create -p D:/envs/medrag python=3.12 -y
D:/envs/medrag/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
D:/envs/medrag/python.exe -m pip install -e ".[data,eval,dev]"
cp .env.example .env        # then set OPENAI_API_KEY for evaluation
```

LLM for development (Ollama). Ollama silently truncates prompts longer than its context
length, so the project uses model variants with a larger `num_ctx`
([docker/ollama/](docker/ollama/)):

```bash
ollama pull qwen2.5:3b
ollama create medrag-qwen2.5-3b -f docker/ollama/Modelfile              # shares a 4 GB GPU with retrieval
ollama create medrag-qwen2.5-3b-gpu -f docker/ollama/Modelfile.full-gpu # LLM alone on the GPU
```

## Run

```bash
python scripts/fetch_wikipedia.py                 # ~250 VI/EN articles -> data/raw, data/holdout
python -m medrag.index.build                      # chunk, embed, BM25 -> data/index
uvicorn medrag.api.main:app --port 8080
```

```bash
curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
     -d '{"question": "Sốt xuất huyết có nên dùng ibuprofen không?"}'
```

| Method | Path | |
| --- | --- | --- |
| GET | `/health` | backend, model, index size and build info |
| POST | `/ask` | `{question, top_n?, return_contexts?}` → `{answer, citations[], timings, backend, model, ...}` |
| POST | `/ask/stream` | same, as Server-Sent Events: `meta` (citations), `token`…, `done` (timings) |
| POST | `/retrieve` | retrieval + rerank only; `mode` = `hybrid` / `dense` / `bm25` |

Switching to vLLM is a `.env` change ([docker/vllm-compose.yml](docker/vllm-compose.yml)):

```
LLM_BACKEND=vllm
LLM_MODEL=qwen2.5-1.5b-awq
LLM_BASE_URL=http://localhost:8000/v1
```

Verified on the laptop with vLLM 0.31.0 in Docker (Qwen2.5-1.5B-Instruct-AWQ): with only those
three lines changed, `python scripts/check_backend.py` sends real exported prompts through the
same backend code and gets answers in the right language with `[i]` citations; the server's
`prompt_tokens` match the token counts of the export.

**Which backend to serve with.** The [inference benchmark](../inference-bench/README.md) measured
this system's own prompts on an NVIDIA L4: **vLLM with Qwen2.5-7B-Instruct-AWQ** is the recommended
serving configuration (2.4× FP16 for one user, ~1.3 requests/s at 8 users with p95 TTFT < 2 s,
no measurable quality loss). Ollama remains the development default because it runs on a laptop
alongside the retrieval models; switching is the `.env` change above.

## Evaluation

```bash
python eval/make_synthetic_qa.py     # synthetic QA (OpenAI), review sample in eval/data/qa_review.csv
python eval/eval_retrieval.py        # BM25 / dense / hybrid / hybrid+rerank
python eval/eval_answers.py --two-phase   # faithfulness, relevance, language, citations, abstention, latency
python -m medrag.tools.export_prompts --tokenizer Qwen/Qwen2.5-7B-Instruct   # -> ../prompts/
```

Evaluation scripts load one model at a time (encoder, then reranker, then the LLM), so they run
on a 4 GB GPU; `--two-phase` does the same for answer generation. OpenAI responses are cached in
`eval/data/_llm_cache.jsonl`, so re-running an evaluation is free and deterministic.

Results: see [Results](#results) below (generated files in [eval/results/](eval/results/)).

## Results

### Data

| | |
| --- | --- |
| Corpus | 289 Wikipedia articles (150 EN, 139 VI) → 6,780 chunks (median 242 tokens, 73 oversized tables/lists) |
| QA set | 390 questions written by `gpt-4.1-mini-2025-04-14`: 300 answerable (150 VI / 150 EN), 40 cross-lingual, 50 unanswerable |
| Quality control | evidence must be verbatim in its chunk (rejects 16%), second LLM check (rejects 7%), manual review of a 10% sample (39/39 kept) |

The synthetic questions reuse many words of their source chunk (lexical overlap: VI mean
0.78, EN 0.67), which favours BM25 and makes recall easy to saturate. The cross-lingual set
(question in one language, gold passage in the other) has no shared words and is the harder test.

### Retrieval (`eval/eval_retrieval.py`)

Answerable questions (n = 300), 95% bootstrap interval for Recall@10:

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |
| --- | --- | --- | --- | --- | --- |
| BM25 only | 0.937 | 0.960 [0.937, 0.980] | 0.823 | 0.855 | 33 ms |
| Dense only (e5) | 0.973 | 0.993 [0.983, 1.000] | 0.870 | 0.901 | 40 ms |
| Hybrid (RRF) | 0.967 | 0.987 [0.973, 0.997] | 0.867 | 0.895 | 72 ms |
| **Hybrid + rerank** | **1.000** | **1.000** [1.000, 1.000] | **0.959** | **0.970** | 920 ms |

Cross-lingual questions (n = 40): Recall@10 is 0.025 (BM25), 0.300 (dense), 0.225 (hybrid),
0.300 (hybrid + rerank).

What this shows:

- **Hybrid + rerank has the highest Recall@10**, as required, but the margin is significant
  only against BM25 (+0.040, paired bootstrap p < 0.001) and hybrid (+0.013, p = 0.017), not
  against dense alone (+0.007, p = 0.14): recall is saturated on this QA set. The reranker's
  real gain is ordering — MRR@10 rises from 0.87 to 0.96, i.e. the right passage is usually
  first, which matters because only the top 5 reach the LLM.
- **RRF does not beat dense alone here** (0.987 vs 0.993 Recall@10; English 0.980 vs 0.993),
  even though the questions favour lexical matching. BM25's mistakes enter the fused list and
  push down good dense hits. Hybrid is still the safer default for exact terms that an encoder
  may blur (drug names, doses, unaccented Vietnamese) and the reranker repairs the order.
- **Cross-lingual retrieval is the weak point.** BM25 cannot match across languages and
  multilingual-e5-base only reaches 0.30. These numbers are a lower bound: the article on the
  same topic in the question's own language may also answer it but is not counted as gold.
- **Rerank threshold.** With the top reranker score as an "is there an answer?" signal:

  | Threshold | Unanswerable caught | Answerable lost |
  | --- | --- | --- |
  | 0.1 | 50% | 0.3% |
  | **0.3 (default)** | **70%** | **1.2%** |
  | 0.5 | 82% | 1.8% |

  Below the threshold the pipeline answers "not found" without calling the LLM.

### Answers (`eval/eval_answers.py`)

Generator: `qwen2.5:3b` (Q4_K_M) in Ollama on a laptop RTX 2050 4 GB; judge:
`gpt-4.1-mini-2025-04-14`. 100 answerable questions (50 VI + 50 EN), 40 cross-lingual,
50 unanswerable; rerank threshold 0.3.

| Metric | Answerable VI | Answerable EN | Cross-lingual | Unanswerable |
| --- | --- | --- | --- | --- |
| Faithfulness (claims supported by context) | 0.970 | 0.987 | 0.924 | – |
| Correctness vs reference (1–5) | 4.54 | 4.84 | 3.38 | – |
| Relevance (1–5) | 4.90 | 4.98 | 3.93 | – |
| Answer in the question's language | 100% | 100% | 97.5% | 96% |
| Has a valid `[i]` citation | 100% | 100% | 97% | – |
| Cites a passage that does not exist | 2% | 0% | 2.5% | – |
| Said "not found" | 0% (good) | 0% (good) | 17.5% | **90%** (good) |

- **Citations needed a prompt fix.** With the citation rule only in the system prompt, the 3B
  model cited a passage in 33% of answers. Repeating it in one line at the end of the user
  message ("End each sentence with its source passage number…") raised that to 100%, with
  correctness unchanged within noise (4.78 → 4.69, overlapping intervals). The system prompt
  stays constant, so prefix caching is unaffected. Previous run:
  [answers_ollama-qwen2.5-3b_prompt-v1.md](eval/results/answers_ollama-qwen2.5-3b_prompt-v1.md).
- **Unanswerable questions:** 90% end in "not found" — most stopped by the rerank threshold
  without an LLM call, the rest refused by the model itself.
- **Cross-lingual answers** inherit the retrieval gap: the gold passage reaches the context in
  only 30% of cases, so correctness drops to 3.4/5 and 17.5% are (correctly cautious) refusals.

### Where the time goes

Median per request that reached the LLM (laptop RTX 2050; retrieval measured in a separate
phase because 4 GB cannot hold all three models):

| Retrieve (dense + BM25 + RRF) | Rerank (30 candidates) | LLM time to first token | LLM total | End to end | LLM share |
| --- | --- | --- | --- | --- | --- |
| 79 ms | 835 ms | 2,934 ms | 3,959 ms | 4,920 ms | **81%** |

The LLM is four fifths of the latency, and its time to first token (prefill of ~2–3k prompt
tokens) is most of that. This is what the [inference benchmark](../inference-bench/) measures
on a datacenter GPU.

### Live API check

The running service (`uvicorn`, embedder + reranker + Ollama `qwen2.5:3b` on the 4 GB laptop GPU)
was called over HTTP with hand-picked questions, including deliberately awkward ones:

| Question | Result |
| --- | --- |
| Sốt xuất huyết Dengue có nên dùng ibuprofen để hạ sốt không? | ✓ "Không nên dùng ibuprofen … [1]", Vietnamese, citation matches |
| What are the common side effects of metformin? | ✓ English, cites the *Adverse effects* section |
| lieu vitamin A dieu tri soi cho tre em la bao nhieu (no diacritics) | ✓ finds *Sởi > Điều trị > Vitamin A*, answers in Vietnamese with the doses |
| Bệnh sởi lây truyền qua đường nào? (`/ask/stream`) | ✓ SSE: `meta`, tokens arriving every ~60 ms, `done` with timings |
| What is the recommended treatment for Kawasaki disease? | ✗ retrieval correct (aspirin), but the 3B model appended an unrelated sentence from the same passage and cited a passage `[3]` that does not exist — reported in `invalid_citations` |
| Trẻ em có được dùng aspirin để hạ sốt không? | ✗ the answer (Reye syndrome) is in passages 1, 2 and 5, yet the model replied "not found" |
| How is tuberculosis transmitted? | ✗ only the Vietnamese *Lao* article is in the corpus; cross-lingual retrieval misses it and the pipeline safely answers "not found" |

Every answer was in the question's language and every `[i]` either matched `citations` or was flagged.
The failures match the measured weak spots — a small 3B generator and cross-lingual retrieval. Serving
with the larger 7B model from the [inference benchmark](../inference-bench/README.md) is recommended;
whether it fixes these particular cases was not tested.

On Windows, connect to `127.0.0.1` rather than `localhost`: a new connection to `localhost` first
tries IPv6 and costs ~2 s against a server listening on IPv4 only (the default backend URLs use
`127.0.0.1` for this reason).

### Benchmark prompts (`medrag.tools.export_prompts`)

[`../prompts/medical_rag_prompts.jsonl`](../prompts/): the 300 answerable questions as the exact
messages sent to the LLM (one shared system prompt, top-5 reranked passages, question), with
`reference_answer`, gold and context chunk ids. Lengths counted with the Qwen2.5-7B-Instruct
chat template ([stats.json](../prompts/stats.json)):

| | min | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| Vietnamese | 973 | 1,859 | 2,384 | 2,686 |
| English | 1,017 | 1,666 | 2,024 | 2,208 |

All fit `max_model_len = 4096` with 256 output tokens, so no context had to be dropped.
Vietnamese prompts are ~12% longer in Qwen tokens for similar content.

### Known limitations

- The QA set is synthetic and lexically close to its sources; retrieval recall is near its
  ceiling, so MRR/nDCG and the cross-lingual subset are the more informative numbers.
- One judge model (`gpt-4.1-mini`) scores everything; its scores are comparable across runs,
  not absolute truth.
- Oversized tables/lists (73 chunks) are kept whole; the encoder only reads their first 512
  tokens (`--split-oversized` splits them at row/item boundaries).
- Latency numbers come from a 4 GB laptop GPU and a 3B model — they show proportions, not
  production speed.

## Tests

```bash
pytest -q          # unit tests; models are loaded only from the local Hugging Face cache
```
