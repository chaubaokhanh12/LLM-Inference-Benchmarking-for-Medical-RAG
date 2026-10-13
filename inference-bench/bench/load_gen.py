"""Closed-loop load generator for OpenAI-compatible chat servers.

N workers each send their next request as soon as the previous one finishes, so N is the
number of concurrent users. Every request streams; we timestamp the send, every content
chunk and the end, and take token counts from the final `usage` chunk (or count them with
the model's tokenizer when the server does not report usage).

    ttft_ms   = t_first - t_send          (prefill + queueing)
    e2e_ms    = t_end - t_send
    tpot_ms   = (e2e - ttft) / (completion_tokens - 1)   (decode speed; the main metric)
    itl_*_ms  = gaps between consecutive chunks (a chunk may hold several tokens)

Standalone use:
    python -m bench.load_gen --base-url http://localhost:11434/v1 --model qwen2.5:3b \
        --prompts ../prompts/medical_rag_prompts.jsonl --concurrency 4 --num-prompts 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

FIELDS = ["prompt_id", "lang", "prompt_tokens", "completion_tokens", "ttft_ms", "e2e_ms",
          "itl_mean_ms", "itl_p95_ms", "tpot_ms", "n_chunks", "t_send", "t_end",
          "usage_source", "error"]


@dataclass
class RequestRecord:
    prompt_id: str
    lang: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    itl_mean_ms: float | None = None
    itl_p95_ms: float | None = None
    tpot_ms: float | None = None
    n_chunks: int = 0
    t_send: float = 0.0  # unix time, to align with GPU samples
    t_end: float = 0.0
    usage_source: str = "server"  # server | tokenizer
    error: str = ""
    text: str = ""  # kept in memory for quality evaluation; not written to requests.csv


def load_prompts(path: Path, n: int | None = None, seed: int = 0) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    random.Random(seed).shuffle(prompts)
    return prompts[:n] if n else prompts


class TokenCounter:
    """Lazy tokenizer, only loaded if a server does not report usage."""

    def __init__(self, name: str | None):
        self.name = name
        self._tok = None

    def count(self, text: str) -> int | None:
        if not self.name:
            return None
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.name)
        return len(self._tok.encode(text, add_special_tokens=False))


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: dict,
                      max_tokens: int, temperature: float, timeout_s: float,
                      include_usage: bool, counter: TokenCounter) -> RequestRecord:
    rec = RequestRecord(prompt_id=prompt["id"], lang=prompt.get("lang", ""))
    body = {"model": model, "messages": prompt["messages"], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    chunk_times: list[float] = []
    parts: list[str] = []
    usage = None
    rec.t_send = time.time()
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url, json=body, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                rec.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        chunk_times.append(time.perf_counter())
                        parts.append(content)
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as e:
        rec.error = f"{type(e).__name__}: {e}"[:300]
    t_end = time.perf_counter()
    rec.t_end = rec.t_send + (t_end - t0)
    rec.e2e_ms = (t_end - t0) * 1000
    rec.n_chunks = len(chunk_times)
    rec.text = "".join(parts)
    if not chunk_times:
        rec.error = rec.error or "no content received"
        return rec
    rec.ttft_ms = (chunk_times[0] - t0) * 1000
    if len(chunk_times) > 1:
        gaps = np.diff(chunk_times) * 1000
        rec.itl_mean_ms = float(gaps.mean())
        rec.itl_p95_ms = float(np.percentile(gaps, 95))
    if usage and usage.get("completion_tokens"):
        rec.prompt_tokens = usage.get("prompt_tokens")
        rec.completion_tokens = usage["completion_tokens"]
    else:
        rec.usage_source = "tokenizer"
        rec.completion_tokens = counter.count(rec.text)
        rec.prompt_tokens = prompt.get("approx_input_tokens")
    if rec.completion_tokens and rec.completion_tokens > 1:
        rec.tpot_ms = (rec.e2e_ms - rec.ttft_ms) / (rec.completion_tokens - 1)
    return rec


async def run_load(prompts: list[dict], *, url: str, model: str, concurrency: int,
                   max_tokens: int, temperature: float, timeout_s: float,
                   max_duration_s: float | None = None, include_usage: bool = True,
                   counter: TokenCounter | None = None,
                   transport: httpx.AsyncBaseTransport | None = None
                   ) -> tuple[list[RequestRecord], float, float]:
    """Closed loop over `prompts`; returns (records, wall_start_unix, wall_seconds)."""
    counter = counter or TokenCounter(None)
    queue = deque(prompts)
    records: list[RequestRecord] = []
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    deadline = time.perf_counter() + max_duration_s if max_duration_s else None

    async with httpx.AsyncClient(limits=limits, timeout=timeout_s, transport=transport) as client:
        async def worker() -> None:
            while queue and (deadline is None or time.perf_counter() < deadline):
                prompt = queue.popleft()
                records.append(await one_request(client, url, model, prompt, max_tokens,
                                                 temperature, timeout_s, include_usage, counter))

        start_unix, t0 = time.time(), time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    return records, start_unix, wall


def write_records(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def summary(records: list[RequestRecord], wall_s: float) -> dict:
    ok = [r for r in records if not r.error]
    def pct(key: str, q: float) -> float | None:
        vals = [getattr(r, key) for r in ok if getattr(r, key) is not None]
        return round(float(np.percentile(vals, q)), 1) if vals else None
    out_tokens = sum(r.completion_tokens or 0 for r in ok)
    return {
        "requests": len(records), "errors": len(records) - len(ok),
        "ttft_p50_ms": pct("ttft_ms", 50), "ttft_p95_ms": pct("ttft_ms", 95),
        "tpot_p50_ms": pct("tpot_ms", 50), "e2e_p50_ms": pct("e2e_ms", 50),
        "throughput_tok_s": round(out_tokens / wall_s, 1) if wall_s else None,
        "req_s": round(len(ok) / wall_s, 3) if wall_s else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", type=Path, default=None, help="append rows to this CSV")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts, args.num_prompts)
    records, _, wall = asyncio.run(run_load(
        prompts, url=args.base_url.rstrip("/") + "/chat/completions", model=args.model,
        concurrency=args.concurrency, max_tokens=args.max_tokens, temperature=0.0,
        timeout_s=args.timeout, counter=TokenCounter(args.tokenizer)))
    if args.out:
        write_records(args.out, [{k: v for k, v in asdict(r).items() if k != "text"} for r in records])
    print(json.dumps(summary(records, wall), indent=2))


if __name__ == "__main__":
    main()
