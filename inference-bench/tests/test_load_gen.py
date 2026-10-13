import asyncio
import json

import httpx
import pytest

from bench.gpu_monitor import parse_prometheus
from bench.load_gen import TokenCounter, run_load
from bench.run import warmup_variant


def sse(chunks: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def delta(text=None, finish=None):
    d = {} if text is None else {"content": text}
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def prompts(n: int) -> list[dict]:
    return [{"id": f"q_{i:04d}", "lang": "vi", "approx_input_tokens": 1000,
             "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": f"Q{i}"}]}
            for i in range(n)]


def make_transport(with_usage: bool = True, fail_ids: tuple[str, ...] = ()):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["messages"][-1]["content"] in fail_ids:
            return httpx.Response(500, text="boom")
        chunks = [delta(""), delta("Xin"), delta(" chào"), delta(" bạn"), delta(finish="stop")]
        if with_usage:
            chunks.append({"choices": [], "usage": {"prompt_tokens": 1200, "completion_tokens": 3}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(chunks))

    return httpx.MockTransport(handler), seen


def run(transport, items, concurrency=2, **kw):
    return asyncio.run(run_load(items, url="http://x/v1/chat/completions", model="m",
                                concurrency=concurrency, max_tokens=8, temperature=0,
                                timeout_s=10, transport=transport, **kw))


def test_closed_loop_records_every_request():
    transport, seen = make_transport()
    records, start, wall = run(transport, prompts(7), concurrency=3)
    assert len(records) == 7 and len(seen) == 7
    assert {r.prompt_id for r in records} == {f"q_{i:04d}" for i in range(7)}
    r = records[0]
    assert r.error == "" and r.prompt_tokens == 1200 and r.completion_tokens == 3
    assert r.n_chunks == 3 and r.ttft_ms <= r.e2e_ms
    assert r.tpot_ms == pytest.approx((r.e2e_ms - r.ttft_ms) / 2)
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}
    assert wall > 0 and start > 1e9


def test_errors_are_recorded_not_raised():
    transport, _ = make_transport(fail_ids=("Q1",))
    records, _, _ = run(transport, prompts(3))
    failed = [r for r in records if r.error]
    assert len(failed) == 1 and failed[0].prompt_id == "q_0001" and "HTTP 500" in failed[0].error


def test_missing_usage_falls_back_to_tokenizer_and_prompt_estimate():
    class FakeCounter(TokenCounter):
        def count(self, text):
            return len(text.split())

    transport, _ = make_transport(with_usage=False)
    records, _, _ = run(transport, prompts(1), counter=FakeCounter(None))
    r = records[0]
    assert r.usage_source == "tokenizer" and r.completion_tokens == 3 and r.prompt_tokens == 1000


def test_warmup_variant_shares_only_the_system_prompt():
    p = prompts(1)[0]
    w = warmup_variant(p)
    assert w["messages"][0] == p["messages"][0]
    assert w["messages"][1]["content"] != p["messages"][1]["content"]
    assert p["messages"][1]["content"] == "Q0"  # original untouched


def test_parse_prometheus_keeps_vllm_gauges_only():
    text = """# HELP x
vllm:num_requests_running{model_name="m"} 3.0
vllm:num_requests_waiting{model_name="m"} 1
vllm:gpu_cache_usage_perc{model_name="m"} 0.42
vllm:e2e_request_latency_seconds_bucket{le="1"} 5
python_gc_objects_collected_total 10
"""
    assert parse_prometheus(text) == {"vllm:num_requests_running": 3.0,
                                      "vllm:num_requests_waiting": 1.0,
                                      "vllm:gpu_cache_usage_perc": 0.42}
