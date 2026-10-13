"""Baseline server: plain Hugging Face Transformers behind a minimal OpenAI-compatible API.

    python -m servers.hf_server --model Qwen/Qwen2.5-7B-Instruct --port 8001

This is deliberately the naive way to serve a model: one request at a time (a global lock),
no batching, no paged KV cache. Under concurrent load, requests simply queue, which is the
behaviour the benchmark contrasts with vLLM's continuous batching.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Iterator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = 256
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    stream: bool = False
    stream_options: dict | None = None


def create_app(model_id: str, dtype: str = "float16") -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    lock = threading.Lock()  # one generation at a time: the point of this baseline
    app = FastAPI(title="HF Transformers baseline")

    def chunk(rid: str, created: int, delta: dict, finish: str | None = None, usage=None) -> str:
        body = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            body["usage"] = usage
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def generate(req: ChatRequest) -> Iterator[tuple[str, dict | None]]:
        """Yields text pieces, then ("", usage) at the end. Holds the lock throughout."""
        max_new = req.max_completion_tokens or req.max_tokens or 256
        with lock:
            inputs = tokenizer.apply_chat_template(req.messages, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True).to(model.device)
            n_in = inputs["input_ids"].shape[1]
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            result: dict = {}

            def run() -> None:
                with torch.inference_mode():
                    result["out"] = model.generate(
                        **inputs, max_new_tokens=max_new, do_sample=req.temperature > 0,
                        temperature=req.temperature if req.temperature > 0 else None,
                        top_p=None, top_k=None, streamer=streamer,
                        pad_token_id=tokenizer.eos_token_id)

            worker = threading.Thread(target=run)
            worker.start()
            for text in streamer:
                if text:
                    yield text, None
            worker.join()
            n_out = int(result["out"].shape[1] - n_in)
            yield "", {"prompt_tokens": n_in, "completion_tokens": n_out,
                       "total_tokens": n_in + n_out, "finish_reason": "length" if n_out >= max_new else "stop"}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        if not req.stream:
            parts, usage = [], None
            for text, u in generate(req):
                parts.append(text)
                usage = u or usage
            finish = usage.pop("finish_reason")
            return JSONResponse({
                "id": rid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                             "finish_reason": finish}],
                "usage": usage})

        include_usage = bool((req.stream_options or {}).get("include_usage"))

        def events() -> Iterator[str]:
            yield chunk(rid, created, {"role": "assistant", "content": ""})
            for text, usage in generate(req):
                if usage is None:
                    yield chunk(rid, created, {"content": text})
                else:
                    yield chunk(rid, created, {}, finish=usage.pop("finish_reason"))
                    if include_usage:
                        yield chunk(rid, created, {}, usage=usage)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()
    uvicorn.run(create_app(args.model, args.dtype), host="0.0.0.0", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
