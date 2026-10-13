"""Ranking metrics with a set of relevant chunks per question, plus bootstrap intervals.

A question usually has one gold chunk, but the evidence sentence can also sit in the
neighbouring chunk (overlap), so relevance is a set.
"""

from __future__ import annotations

import math

import numpy as np


def first_hit_rank(ranked_ids: list[str], relevant: set[str]) -> int | None:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return i
    return None


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Hit rate: 1 if any relevant chunk is in the top k."""
    return float(any(cid in relevant for cid in ranked_ids[:k]))


def mrr_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    rank = first_hit_rank(ranked_ids[:k], relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked_ids[:k], start=1)
              if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values: list[float] | np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """Mean difference a-b over the same questions, and the share of resamples where a <= b
    (a one-sided p-value for "a is better than b")."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float((means <= 0).mean())
