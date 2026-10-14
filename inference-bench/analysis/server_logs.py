"""Facts that only the server logs know, distilled into results/<run>/server_facts.json.

server.log files are large (llama.cpp logs every prompt-cache decision) and are not committed;
the few numbers the analysis needs are kept in server_facts.json instead.

vLLM     model weight memory, KV-cache memory and capacity in tokens (from start-up lines)
Ollama   how much of each prompt llama.cpp reused from its slot cache instead of recomputing:
         the median share, and the share of requests whose prompt was reused almost entirely
         (it cannot be turned off, so it is reported next to Ollama's TTFT)
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

FACTS_FILE = "server_facts.json"

_VLLM = {
    "model_weights_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
}
_TASK_PROMPT = re.compile(r"task (\d+) \| new prompt, .*task\.n_tokens = (\d+)")
_TASK_EVAL = re.compile(r"task (\d+) \| prompt eval time = +[\d.]+ ms / +(\d+) tokens")


def parse_server_log(path: Path) -> dict:
    facts: dict = {}
    total: dict[str, int] = {}
    evaluated: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for key, rx in _VLLM.items():
                if key not in facts and (m := rx.search(line)):
                    facts[key] = float(m.group(1).replace(",", ""))
            if m := _TASK_PROMPT.search(line):
                total[m.group(1)] = int(m.group(2))
            elif m := _TASK_EVAL.search(line):
                evaluated[m.group(1)] = int(m.group(2))
    reused = [1 - evaluated[t] / total[t] for t in total if t in evaluated and total[t]]
    if reused:
        facts["prompt_reuse_median"] = round(statistics.median(reused), 3)
        facts["prompt_reuse_full_share"] = round(sum(r >= 0.9 for r in reused) / len(reused), 3)
        facts["prompt_reuse_requests"] = len(reused)
    if "kv_cache_tokens" in facts:
        facts["kv_cache_tokens"] = int(facts["kv_cache_tokens"])
    return facts


def server_facts(run_dir: Path) -> dict:
    """Facts for a run: parsed from server.log when present (and cached), else from the cache."""
    cached = run_dir / FACTS_FILE
    log = run_dir / "server.log"
    if log.exists():
        facts = parse_server_log(log)
        cached.write_text(json.dumps(facts, indent=2), encoding="utf-8")
        return facts
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    return {}
