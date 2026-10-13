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

Specifications and plan: [docs/medical_rag_spec.md](docs/medical_rag_spec.md),
[docs/llm_benchmark_spec.md](docs/llm_benchmark_spec.md), [docs/PLAN.md](docs/PLAN.md).
