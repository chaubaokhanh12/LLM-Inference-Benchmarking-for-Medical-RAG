# LLM Inference Benchmark on a Medical RAG Workload

How much faster is vLLM than plain Hugging Face Transformers or Ollama for the prompts a
medical RAG system actually sends — long inputs (1–3k tokens of retrieved passages), short
answers (≤256 tokens)? This project serves one open model with each engine, replays the same
prompts at 1–32 concurrent users, and explains the differences with GPU and KV-cache data.

> **Status:** harness implemented and smoke-tested locally against Ollama. The GPU runs (L4)
> are pending; results and figures will be filled in below.

## Questions

| # | Question | Experiment |
| --- | --- | --- |
| Q1 | At one user, how much faster is vLLM than HF Transformers and Ollama? | E1 |
| Q2 | From 1 to 32 concurrent users, how do latency and throughput change? | E2 |
| Q3 | What does AWQ 4-bit save in memory, and what does it cost in speed and quality? | E3 |
| Q4 | How much TTFT does prefix caching save with the RAG system prompt? | E4 |
| Q5 | Why? (batching, KV cache, GPU utilisation) | profiling |

## Workload

`../prompts/medical_rag_prompts.jsonl`, exported by [medical-rag](../medical-rag/): one line
per evaluation question with the exact chat messages (fixed system prompt + top-5 reranked
passages + question), Vietnamese and English mixed. Retrieval is not part of the benchmark;
every engine gets byte-identical inputs. `prompts/stats.json` has the input-length
distribution.

## Setup (fixed)

| | |
| --- | --- |
| GPU | 1× NVIDIA L4 24 GB |
| Model | Qwen2.5-7B-Instruct — FP16, official AWQ, GGUF Q4_K_M (Ollama) |
| Context | `max_model_len = 4096`; `max_tokens = 256`, `temperature = 0` |
| Repeats | 3 per configuration (median reported), 10 warm-up requests first |

## How it works

```
configs/e2_vllm_fp16.yaml ─► bench.run ─► start server ─► wait for /health
                                       ├─► env.json, command.txt
                                       ├─► for each concurrency × repeat:
                                       │     warm-up → closed-loop load (bench.load_gen)
                                       │     + nvidia-smi sampler + vLLM /metrics poller
                                       └─► results/<run_id>/{requests,phases,gpu,vllm_metrics}.csv
analysis.aggregate ─► results/summary.csv   (median over repeats)
analysis.plots     ─► results/plots/*.png
bench.quality_eval ─► quality_summary.json (LLM judge: faithfulness, correctness)
```

- **Closed loop:** N workers each send the next request as soon as the previous one returns;
  N is the number of concurrent users.
- **Timing:** every request streams. TTFT = first content chunk − send; TPOT =
  (e2e − TTFT) / (output tokens − 1) is the main decode metric (a chunk can hold several
  tokens, so inter-chunk gaps are only indicative). Token counts come from the server's
  `usage`, or the model tokenizer when a server does not report it.
- **HF baseline** ([servers/hf_server.py](servers/hf_server.py)): a minimal OpenAI-compatible
  FastAPI server that generates one request at a time — the naive way to serve a model.

### Fair comparison

- Ollama runs a 4-bit GGUF, so it is compared with **vLLM AWQ** (also 4-bit), not FP16.
- E1–E3 run vLLM with prefix caching **off** (`--no-enable-prefix-caching`), so a repeated
  prompt cannot hit a cache filled by the previous repeat; only E4 turns it on, and resets the
  cache (`/reset_prefix_cache`, `VLLM_SERVER_DEV_MODE=1`) before each repeat. Warm-up prompts
  share only the system prompt with the measured ones. llama.cpp (Ollama) reuses a slot's
  prompt prefix and cannot be switched off — noted in the results.
- Ollama is started with `OLLAMA_NUM_PARALLEL=32` and `OLLAMA_CONTEXT_LENGTH=4096`; its
  defaults (few parallel slots, short context) would distort the comparison.
- Server flags change between versions: each run stores the exact command and library
  versions (`command.txt`, `env.json`).

## Run

```bash
pip install -e ".[hf,dev]"     # on the GPU VM also: pip install vllm
python -m bench.run configs/e1_vllm_fp16.yaml      # one experiment configuration
python -m bench.quality_eval results/<fp16_run> results/<awq_run>   # E3 quality (OPENAI_API_KEY)
python -m analysis.aggregate && python -m analysis.plots
```

Local smoke test (any OpenAI-compatible server already running, e.g. Ollama):

```bash
python -m bench.load_gen --base-url http://localhost:11434/v1 --model qwen2.5:3b \
    --prompts ../prompts/medical_rag_prompts.jsonl --concurrency 2 --num-prompts 10
```

### Google Colab (L4)

[colab/run_benchmark.ipynb](colab/run_benchmark.ipynb) runs E1–E4 on a Colab Pro L4 runtime: it checks
the GPU and driver, clones this repository, installs vLLM / Ollama / the HF baseline, pre-downloads the
weights, runs each configuration and copies it to Google Drive as soon as it finishes. Re-running a cell
skips configurations already finished on Drive, so the ~9–10 GPU hours can be spread over several sessions;
an interrupted configuration is re-run from the start and its partial copy is ignored by `analysis.aggregate`
(only runs with a `DONE` file count). E5 (TGI) needs Docker and is not run on Colab.

Each measured phase is capped by `max_duration_s` (E1 600 s, E2 300 s, HF in E2 240 s) so the whole plan fits
the budget; the closed loop stops sending new requests at the cap and lets in-flight ones finish.

### Google Cloud checklist

1. Request L4 GPU quota for the region on day one (can take 1–2 days); add a budget alert.
2. VM with a Deep Learning image (driver + CUDA included), disk ≥ 100 GB.
3. Pre-download weights: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct` (and `-AWQ`);
   `ollama pull qwen2.5:7b-instruct-q4_K_M`.
4. Run inside `tmux`; copy `results/` off the VM and **stop the VM** after each session.
   Budget ≈ 10–15 GPU hours.

## Results

_Pending GPU runs (L4)._

### Local smoke test (not a benchmark result)

The harness was exercised end to end on a laptop RTX 2050 (4 GB) with vLLM 0.31.0 in Docker
serving Qwen2.5-1.5B-Instruct-AWQ, 24 real RAG prompts per level, 128 output tokens:

| Concurrent users | Throughput | TPOT p50 | End-to-end p50 | Max KV-cache use | Prefix-cache hit rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 24 tok/s | 35 ms | 4.2 s | 4% | 28% |
| 4 | 96 tok/s | 37 ms | 3.7 s | 13% | 99% |
| 8 | 188 tok/s | 35 ms | 3.7 s | 23% | 99% |

- **Continuous batching:** 8 users get 7.8× the throughput of one user with the same
  per-token speed; no request waited (`vllm:num_requests_waiting` stayed 0).
- **KV-cache arithmetic checks out:** 2 × 28 layers × 2 KV heads × 128 dims × 2 bytes =
  28 KB per token; vLLM reported 1.54 GiB of KV cache = 57,504 tokens, ≈ 14 concurrent
  4,096-token requests.
- **Why E1–E3 turn prefix caching off.** With it on (the v0.31 default), the 4- and 8-user
  phases re-sent prompts already seen at 1 user, hit the cache for 99% of prompt tokens, and
  showed a misleadingly low TTFT (~110 ms vs ~440 ms). On fresh prompts the hit rate was 28%:
  roughly the share of the shared system prompt and chat template — a first estimate for E4.
