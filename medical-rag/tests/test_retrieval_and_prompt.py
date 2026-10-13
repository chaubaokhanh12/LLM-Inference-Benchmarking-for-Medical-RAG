from medrag.generate.prompt import (
    ANSWER_LANGUAGE,
    NOT_FOUND,
    SYSTEM_PROMPT,
    build_citations,
    build_messages,
    fit_messages,
    is_not_found,
    parse_citations,
)
from medrag.index.sparse import tokenize
from medrag.retrieve.hybrid import rrf_fuse
from medrag.types import ScoredChunk
from tests.fakes import make_chunk


def test_rrf_by_hand():
    dense = [10, 20, 30]
    bm25 = [30, 40]
    fused = dict(rrf_fuse([dense, bm25], rrf_k=60))
    assert fused[10] == 1 / 61
    assert fused[30] == 1 / 63 + 1 / 61  # rank 3 in dense, rank 1 in BM25
    assert fused[40] == 1 / 62
    order = [row for row, _ in rrf_fuse([dense, bm25], rrf_k=60)]
    assert order[0] == 30  # present in both lists beats rank 1 in a single list


def test_rrf_empty_branch_equals_single_branch_order():
    assert [r for r, _ in rrf_fuse([[3, 1, 2], []])] == [3, 1, 2]


def test_tokenize_vietnamese_compounds_and_stopwords():
    toks = tokenize("Bệnh nhân bị đau đầu và sốt xuất huyết.", "vi")
    assert "đau_đầu" in toks
    assert "và" not in toks and "bị" not in toks
    assert all(t == t.lower() for t in toks)
    with_syll = tokenize("Bệnh nhân bị đau đầu.", "vi", add_syllables=True)
    assert {"đau_đầu", "đau", "đầu"} <= set(with_syll)


def test_tokenize_english():
    assert tokenize("What is the dose of COVID-19 vaccine (0.5 mL)?", "en") == [
        "dose", "covid-19", "vaccine", "0.5", "ml"]


def test_system_prompt_is_identical_across_requests():
    a = build_messages("Sốt nên làm gì?", "vi", [make_chunk(0, "A")])
    b = build_messages("What is gout?", "en", [make_chunk(1, "B"), make_chunk(2, "C")])
    assert a[0] == b[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert a[1]["content"].startswith("Context:\n[1] Tài liệu 0 > Điều trị\nA")
    assert a[1]["content"].endswith(ANSWER_LANGUAGE["vi"])
    assert b[1]["content"].endswith(ANSWER_LANGUAGE["en"])


def test_fit_messages_drops_lowest_ranked_contexts():
    contexts = [ScoredChunk(make_chunk(i, "x " * 100), score=1.0) for i in range(5)]
    count = lambda msgs: sum(len(m["content"].split()) for m in msgs)  # noqa: E731
    budget = count(build_messages("q", "vi", [c.chunk for c in contexts[:3]]))
    _, kept = fit_messages("q", "vi", contexts, budget, count)
    assert [c.chunk.chunk_id for c in kept] == ["doc0:0", "doc1:0", "doc2:0"]


def test_parse_citations():
    assert parse_citations("A [1]. B [2][3]. C [1, 4-5]. D [2]") == [1, 2, 3, 4, 5]
    assert parse_citations("no citations") == []


def test_build_citations_flags_invalid_numbers():
    ctx = [ScoredChunk(make_chunk(i, "t"), score=0.5) for i in range(2)]
    records, invalid = build_citations(ctx, "Theo [2] và [7].")
    assert [r["cited"] for r in records] == [False, True]
    assert invalid == [7]


def test_is_not_found():
    assert is_not_found(NOT_FOUND["vi"])
    assert is_not_found("Khong tim thay thong tin nay trong tai lieu.")
    assert is_not_found(NOT_FOUND["en"])
    assert not is_not_found("Paracetamol 500 mg mỗi 6 giờ [1].")
