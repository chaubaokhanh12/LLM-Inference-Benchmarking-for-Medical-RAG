"""FastAPI app.

    uvicorn medrag.api.main:app --port 8080

Endpoints are plain `def` functions, so FastAPI runs them in its thread pool; the encoder and
the reranker each hold a lock, while LLM calls run concurrently (a vLLM server batches them).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from medrag.config import get_settings
from medrag.llm import LLMError
from medrag.pipeline import AnswerResult, RAGPipeline

log = logging.getLogger("medrag.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_n: int | None = Field(default=None, ge=1, le=10)
    return_contexts: bool = False


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_n: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool = True
    return_text: bool = True


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            p = await run_in_threadpool(RAGPipeline.from_settings, get_settings())
            await run_in_threadpool(p.warmup)
            app.state.pipeline = p
        yield

    app = FastAPI(title="Medical RAG", version="0.1.0", lifespan=lifespan)

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/health")
    def health(request: Request) -> dict:
        p = get_pipeline(request)
        meta = p.retriever.bundle.meta
        return {
            "status": "ok",
            "backend": p.backend.name,
            "model": p.backend.model,
            "base_url": p.settings.resolved_base_url,
            "embed_model": meta["embed_model"],
            "rerank_model": p.settings.rerank_model,
            "n_chunks": meta["n_chunks"],
            "index_built_at": meta["built_at"],
            "chunker_hash": meta["chunker_hash"],
        }

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest, request: Request) -> dict:
        p = get_pipeline(request)
        r = p.retrieve(req.question, top_n=req.top_n, mode=req.mode, rerank=req.rerank)
        return {
            "lang": r.lang,
            "n_candidates": r.n_candidates,
            "timings": {k: round(v, 2) for k, v in r.timings.items()},
            "results": [c.to_dict(with_text=req.return_text) for c in r.contexts],
        }

    @app.post("/ask")
    def ask(req: AskRequest, request: Request) -> dict:
        try:
            result = get_pipeline(request).answer(req.question, top_n=req.top_n)
        except LLMError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return result.to_dict(with_contexts=req.return_contexts)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        p = get_pipeline(request)

        def sse(event: str, data: object) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def events() -> Iterator[str]:
            try:
                for event, payload in p.answer_stream(req.question, top_n=req.top_n):
                    if isinstance(payload, AnswerResult):
                        payload = payload.to_dict(with_contexts=req.return_contexts)
                    elif event == "token":
                        payload = {"text": payload}
                    yield sse(event, payload)
            except LLMError as e:
                yield sse("error", {"detail": str(e)})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
