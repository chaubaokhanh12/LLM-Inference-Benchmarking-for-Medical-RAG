# Bilingual Medical RAG + LLM Inference Benchmark

A Vietnamese–English medical question-answering system built from scratch (hybrid retrieval,
reranking, cited answers), and a benchmark of how to serve its LLM: **vLLM vs Ollama vs Hugging Face
Transformers** on an NVIDIA L4, replaying the system's own prompts.

- **vLLM: 7.6× the throughput of Hugging Face Transformers** at 32 concurrent users (Qwen2.5-7B-Instruct, L4).
- **AWQ 4-bit: 2.4× the single-user speed of FP16** and 2.75× the KV-cache capacity, with no measurable quality loss.
- **Retrieval: reranking lifts MRR@10 from 0.87 to 0.96**; answers are 98% faithful to their cited sources and
  90% of questions the documents cannot answer are refused.
- **Why:** the RAG workload is prefill-bound — 22 prompt tokens per generated token.

![Throughput vs concurrent users: vLLM AWQ and FP16 scale to ~115 tok/s, Ollama to 99, HF Transformers stays at 15](inference-bench/results/plots/1_throughput_vs_concurrency.png)

| Qwen2.5-7B on 1× L4 | 1 user | 8 users | 32 users | p95 time to first token at 8 users |
| --- | --- | --- | --- | --- |
| HF Transformers FP16 | 14.9 tok/s | 15.0 | 15.0 | 46 s |
| vLLM FP16 | 15.9 | 73.3 | 114.1 | 1.2 s |
| **vLLM AWQ 4-bit** | **38.1** | **102.0** | **118.0** | 1.7 s |
| Ollama GGUF Q4_K_M | 38.7 | 47.1 | 98.6 | 1.7 s |

Details: [inference benchmark](inference-bench/README.md) (Q1–Q5, profiling, all six figures) ·
[RAG evaluation](medical-rag/README.md#results) · [interview notes](docs/interview_answers.md)

## How the two parts fit together

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
   engines, replays that workload at 1–32 concurrent users, and explains the differences with GPU,
   KV-cache and scheduler data. A Colab notebook reproduces it on an L4.

Stack: Python, PyTorch, Hugging Face Transformers, sentence-transformers, FAISS, rank-bm25, pyvi,
FastAPI, vLLM, Ollama, Docker, OpenAI API (QA generation and LLM judge).

Specifications and plan: [docs/medical_rag_spec.md](docs/medical_rag_spec.md),
[docs/llm_benchmark_spec.md](docs/llm_benchmark_spec.md), [docs/PLAN.md](docs/PLAN.md).
