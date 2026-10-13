"""OpenAI client for evaluation work (writing synthetic questions, judging answers).

Responses are cached on disk by request hash: re-running an evaluation costs nothing and
gives identical results, which matters when the same judge must score several configurations.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import openai

from medrag.config import PROJECT_ROOT, get_settings

CACHE_FILE = PROJECT_ROOT / "eval" / "data" / "_llm_cache.jsonl"


class EvalLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 cache_file: Path | None = CACHE_FILE):
        s = get_settings()
        key = api_key or s.openai_api_key
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set (medical-rag/.env)")
        self.model = model or s.eval_llm_model
        self.client = openai.OpenAI(api_key=key, timeout=120, max_retries=4)
        self.cache_file = cache_file
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._send_temperature = True
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "cached": 0}
        if cache_file and cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]

    def json_call(self, system: str, user: str, *, temperature: float = 0.0,
                  max_tokens: int = 800, tag: str = "") -> dict:
        """Chat completion that must return a JSON object."""
        key = hashlib.sha256(json.dumps([self.model, system, user, temperature, tag],
                                        ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self.usage["cached"] += 1
                return self._cache[key]
        response = self._call(system, user, temperature, max_tokens)
        with self._lock:
            self._cache[key] = response
            if self.cache_file:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        return response

    def _call(self, system: str, user: str, temperature: float, max_tokens: int) -> dict:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
        }
        for attempt in range(3):
            if self._send_temperature:
                kwargs["temperature"] = temperature
            else:
                kwargs.pop("temperature", None)
            try:
                r = self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                if self._send_temperature and "temperature" in str(e).lower():
                    self._send_temperature = False  # reasoning models accept only the default
                    continue
                raise
            with self._lock:
                self.usage["calls"] += 1
                if r.usage:
                    self.usage["prompt_tokens"] += r.usage.prompt_tokens
                    self.usage["completion_tokens"] += r.usage.completion_tokens
            try:
                return json.loads(r.choices[0].message.content or "{}")
            except json.JSONDecodeError:
                time.sleep(1 + attempt)
        return {}
