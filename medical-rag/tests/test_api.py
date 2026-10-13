import json

from fastapi.testclient import TestClient

from medrag.api.main import create_app
from tests.fakes import make_pipeline


def client(**kw) -> TestClient:
    return TestClient(create_app(make_pipeline(**kw)))


def test_health():
    with client() as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["n_chunks"] == 3 and body["backend"] == "fake"


def test_ask():
    with client() as c:
        body = c.post("/ask", json={"question": "Sốt nên uống thuốc gì?"}).json()
    assert body["answer"] == "Uống paracetamol [1]."
    assert body["citations"][0]["cited"] is True
    assert "contexts" not in body
    assert set(body["timings"]) >= {"retrieve_ms", "rerank_ms", "llm_ttft_ms", "total_ms"}


def test_ask_validation():
    with client() as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "x", "top_n": 99}).status_code == 422


def test_ask_backend_down_returns_503():
    with client(fail=True) as c:
        r = c.post("/ask", json={"question": "Sốt nên uống thuốc gì?"})
    assert r.status_code == 503 and "backend down" in r.json()["detail"]


def test_ask_stream_sse():
    with client() as c:
        r = c.post("/ask/stream", json={"question": "Sốt nên uống thuốc gì?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in r.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    assert [e for e, _ in events] == ["meta", "token", "token", "done"]
    assert "".join(d["text"] for e, d in events if e == "token") == "Uống paracetamol [1]."
    assert events[-1][1]["timings"]["llm_ttft_ms"] == 12.0


def test_retrieve_without_llm():
    with client() as c:
        body = c.post("/retrieve", json={"question": "sốt", "top_n": 3}).json()
    assert [r["chunk_id"] for r in body["results"]] == ["doc0:0", "doc2:0", "doc1:0"]
    assert body["results"][0]["rerank_score"] == 0.9
