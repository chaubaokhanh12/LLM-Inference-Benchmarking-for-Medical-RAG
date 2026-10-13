import json

import httpx
import pytest
import respx

from medrag.llm import LLMError
from medrag.llm.base import DoneEvent, TokenEvent
from medrag.llm.openai_compat import OpenAICompatBackend
from tests.fakes import make_pipeline

BASE = "http://vllm.test/v1"


def backend(name: str, **kw) -> OpenAICompatBackend:
    # A plain httpx client, so respx can intercept the SDK's requests.
    return OpenAICompatBackend(name, "m", BASE, http_client=httpx.Client(), **kw)


def sse_body(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def delta(content=None, role=None, finish=None) -> dict:
    d = {}
    if role:
        d["role"] = role
    if content is not None:
        d["content"] = content
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


USAGE = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": [],
         "usage": {"prompt_tokens": 1200, "completion_tokens": 3, "total_tokens": 1203}}


@respx.mock
def test_openai_compat_stream_measures_ttft_and_usage():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content=sse_body([delta(role="assistant", content=""), delta("Xin"), delta(" chào"),
                          delta(finish="stop"), USAGE]),
    ))
    events = list(backend("vllm").stream([{"role": "user", "content": "hi"}], 8, 0))
    assert [e.text for e in events if isinstance(e, TokenEvent)] == ["Xin", " chào"]
    result = events[-1].result
    assert isinstance(events[-1], DoneEvent)
    assert result.text == "Xin chào"
    assert (result.prompt_tokens, result.completion_tokens) == (1200, 3)
    assert result.ttft_ms is not None and result.ttft_ms <= result.total_ms
    assert result.finish_reason == "stop"
    sent = json.loads(route.calls[0].request.content)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["max_tokens"] == 8


@respx.mock
def test_openai_compat_retries_without_stream_options():
    route = respx.post(f"{BASE}/chat/completions").mock(side_effect=[
        httpx.Response(400, json={"error": {"message": "unknown field stream_options"}}),
        httpx.Response(200, headers={"content-type": "text/event-stream"},
                       content=sse_body([delta("ok", finish="stop")])),
    ])
    result = backend("ollama").generate(
        [{"role": "user", "content": "hi"}], 8, 0)
    assert result.text == "ok" and result.prompt_tokens is None
    assert "stream_options" not in json.loads(route.calls[1].request.content)


@respx.mock
def test_openai_compat_unreachable_raises_llm_error():
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(LLMError, match="cannot reach vllm"):
        backend("vllm", max_retries=0).generate([{"role": "user", "content": "hi"}], 8, 0)


def test_pipeline_answer_has_timings_and_citations():
    p = make_pipeline()
    r = p.answer("Sốt nên uống thuốc gì?")
    assert r.lang == "vi"
    assert r.answer == "Uống paracetamol [1]."
    assert set(r.timings) == {"retrieve_ms", "rerank_ms", "llm_ttft_ms", "llm_total_ms", "total_ms"}
    assert r.timings["llm_ttft_ms"] == 12.0
    # reranker scores 0.9, 0.2, 0.7 -> top 2 are doc0, doc2
    assert [c["chunk_id"] for c in r.citations] == ["doc0:0", "doc2:0"]
    assert [c["cited"] for c in r.citations] == [True, False]
    assert r.invalid_citations == [] and r.llm_called and not r.not_found
    user_msg = p.backend.calls[0][1]["content"]
    assert "[1] Tài liệu 0 > Điều trị" in user_msg and "[2] Tài liệu 2" in user_msg


def test_pipeline_short_circuits_when_nothing_passes_threshold():
    p = make_pipeline(min_score=0.95)
    r = p.answer("What is the capital of France?")
    assert r.answer == "I could not find this information in the documents."
    assert not r.llm_called and r.not_found and p.backend.calls == []
    assert r.timings["llm_total_ms"] is None


def test_pipeline_stream_events_and_request_log(tmp_path):
    p = make_pipeline(log_dir=tmp_path)
    events = list(p.answer_stream("Sốt nên uống thuốc gì?"))
    assert [e for e, _ in events] == ["meta", "token", "token", "done"]
    assert events[0][1]["citations"][0]["chunk_id"] == "doc0:0"
    logged = json.loads((tmp_path / "requests.jsonl").read_text(encoding="utf-8"))
    assert logged["answer"] == "Uống paracetamol [1]." and logged["context_ids"] == ["doc0:0", "doc2:0"]
