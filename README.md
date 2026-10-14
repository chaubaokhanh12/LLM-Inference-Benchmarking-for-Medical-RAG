# LLM Inference Benchmarking for Medical RAG

Two projects in one repository, connected by a folder of prompts:

```
 medical-rag/                      prompts/                          inference-bench/
 ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
 │ bilingual VI/EN medical  │      │ medical_rag_prompts.jsonl│      │ HF Transformers vs vLLM  │
 │ RAG: hybrid retrieval +  │ ───► │ stats.json               │ ───► │ vs Ollama: TTFT, latency,│
 │ rerank + LLM, FastAPI,   │      │ manifest.json            │      │ throughput, GPU memory   │
 │ evaluation               │      │ (exact LLM inputs)       │      │                          │
 └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
```

1. **[medical-rag/](medical-rag/)** answers medical questions in Vietnamese and English from
   Wikipedia articles. Retrieval is built from scratch (no LangChain/LlamaIndex): a
   section-aware chunker, dense e5 embeddings + BM25 fused with Reciprocal Rank Fusion, and a
   cross-encoder reranker. The LLM backend (Ollama, vLLM, OpenAI, Groq, Gemini) is chosen in
   `.env`. Every request records how long each stage takes.
2. **[prompts/](prompts/)** holds the final chat messages the RAG system sends to its LLM, one per
   evaluation question. Retrieval is done once, so every benchmarked server receives identical
   inputs and the benchmark measures only inference.
3. **[inference-bench/](inference-bench/)** serves one open model with several inference
   engines and compares them on that real workload.

## Results at a glance

**Medical RAG** ([details](medical-rag/README.md#results)) — 289 Wikipedia articles, 390 synthetic questions:
hybrid retrieval + reranking finds the source passage in the top 10 for 100% of answerable questions
(MRR@10 0.96 vs 0.87 for dense retrieval alone); answers are 98% faithful to their sources, always in the
question's language and cited; 90% of unanswerable questions end in "not found". Cross-lingual
questions remain the weak spot (Recall@10 0.30).

**Inference benchmark** ([details](inference-bench/README.md#summary)) — Qwen2.5-7B-Instruct on one L4,
300 RAG prompts (~1,700 tokens in, ~78 out):
at one user, 4-bit weights matter more than the engine (2.4×); at 32 users vLLM delivers 7.6× the
throughput of HF Transformers; vLLM AWQ gives up to 2.2× Ollama's throughput at equal latency with no
measurable quality loss; prefix caching cuts TTFT by 11–14%. The workload is prefill-bound: shorter
retrieved context is the biggest remaining lever.

Specifications and plan: [docs/medical_rag_spec.md](docs/medical_rag_spec.md),
[docs/llm_benchmark_spec.md](docs/llm_benchmark_spec.md), [docs/PLAN.md](docs/PLAN.md);
interview questions answered with the measured numbers: [docs/interview_answers.md](docs/interview_answers.md).
