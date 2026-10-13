import math

from medrag.evaluation.metrics import (
    bootstrap_ci,
    first_hit_rank,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)
from medrag.evaluation.qa import QAItem, find_gold_chunks, remap_gold
from medrag.types import Chunk


def test_ranking_metrics_single_gold():
    ranked = ["a", "b", "c", "d"]
    gold = {"c"}
    assert first_hit_rank(ranked, gold) == 3
    assert recall_at_k(ranked, gold, 2) == 0.0 and recall_at_k(ranked, gold, 3) == 1.0
    assert mrr_at_k(ranked, gold, 10) == 1 / 3
    assert math.isclose(ndcg_at_k(ranked, gold, 10), 1 / math.log2(4))


def test_ndcg_with_two_relevant_chunks():
    assert ndcg_at_k(["x", "y"], {"x", "y"}, 10) == 1.0
    assert ndcg_at_k(["y", "z"], {"x", "y"}, 10) < 1.0


def test_bootstrap_and_paired():
    mean, lo, hi = bootstrap_ci([1, 1, 0, 1, 0, 1, 1, 1])
    assert lo <= mean <= hi and mean == 0.75
    diff, p = paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0])
    assert diff == 1.0 and p == 0.0


def chunk(i: int, text: str) -> Chunk:
    return Chunk(f"d:{i}", "d", "T", [], text, "vi", 10)


def test_gold_remap_follows_evidence_after_rechunking():
    old = [chunk(0, "A. B."), chunk(1, "Liều 500 mg mỗi 6 giờ. C.")]
    item = QAItem(id="a-vi-0001", question="q", lang="vi", type="answerable",
                  evidence="Liều 500 mg  mỗi 6 giờ.", gold_doc_id="d", gold_chunk_ids=["d:1"])
    assert find_gold_chunks(item.evidence, "d", {"d": old}) == ["d:1"]
    # after re-chunking the sentence sits in chunk 0, and also in chunk 1 through overlap
    new = [chunk(0, "A. B. Liều 500 mg mỗi 6 giờ."), chunk(1, "Liều 500 mg mỗi 6 giờ. C.")]
    remapped, lost = remap_gold([item], new)
    assert (remapped, lost) == (1, 0)
    assert item.gold_chunk_ids == ["d:0", "d:1"]
