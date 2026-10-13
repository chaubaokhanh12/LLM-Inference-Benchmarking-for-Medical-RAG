"""Retrieval evaluation: BM25 vs dense vs hybrid (RRF) vs hybrid + rerank.

    python eval/eval_retrieval.py [--index-dir data/index] [--tag baseline]

Reports Recall@5/10, MRR@10, nDCG@10 (chunk level) and Recall@10 at document level, per
language and for the cross-lingual subset, with 95% bootstrap intervals; a paired test of
hybrid+rerank against each other configuration; and a rerank-threshold table built from the
unanswerable questions (how often a threshold abstains correctly vs. wrongly).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from medrag.config import PROJECT_ROOT, get_settings
from medrag.evaluation.metrics import (
    bootstrap_ci,
    first_hit_rank,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)
from medrag.evaluation.qa import QA_FILE, read_qa, remap_gold
from medrag.evaluation.staged import staged_retrieval
from medrag.index.store import load_index

RESULTS = PROJECT_ROOT / "eval" / "results"
CONFIGS = ["bm25", "dense", "hybrid", "hybrid+rerank"]
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", type=Path, default=QA_FILE)
    ap.add_argument("--index-dir", type=Path, default=s.index_dir)
    ap.add_argument("--k-out", type=int, default=s.retrieve_k_out)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    bundle = load_index(args.index_dir, embed_model=s.embed_model)
    items = read_qa(args.qa)
    remapped, lost = remap_gold(items, bundle.chunks)
    print(f"{len(items)} questions; gold remapped for {remapped}, lost for {lost} "
          f"(index chunker {bundle.meta['chunker_hash']})")
    chunk_doc = {c.chunk_id: c.doc_id for c in bundle.chunks}

    staged = staged_retrieval([(it.question, it.lang) for it in items], s,
                              modes=("bm25", "dense", "hybrid"), k_out=args.k_out, bundle=bundle)
    rows = []  # one row per (question, config)
    unanswerable_top = []
    answerable_top: dict[str, float] = {}
    for it, res in zip(items, staged, strict=True):
        ranked = {m: [r.chunk.chunk_id for r in res.by_mode[m]] for m in ("bm25", "dense", "hybrid")}
        ranked["hybrid+rerank"] = [r.chunk.chunk_id for r in res.reranked]
        latency = {**res.latency_ms,
                   "hybrid+rerank": res.latency_ms["hybrid"] + res.latency_ms["rerank"]}
        top = res.reranked[0].score if res.reranked else 0.0

        if it.type == "unanswerable":
            unanswerable_top.append(top)
            continue
        answerable_top[it.id] = top
        gold = set(it.gold_chunk_ids)
        for cfg in CONFIGS:
            ids = ranked[cfg]
            docs = [chunk_doc[c] for c in ids]
            rows.append({
                "id": it.id, "lang": it.lang, "type": it.type, "config": cfg,
                "rank": first_hit_rank(ids, gold) or "",
                "R@5": recall_at_k(ids, gold, 5), "R@10": recall_at_k(ids, gold, 10),
                "MRR@10": mrr_at_k(ids, gold, 10), "nDCG@10": ndcg_at_k(ids, gold, 10),
                "docR@10": float(it.gold_doc_id in docs[:10]),
                "latency_ms": round(latency[cfg], 1),
            })

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / f"retrieval_{args.tag}_detail.csv", rows)
    summary = summarise(rows)
    write_csv(RESULTS / f"retrieval_{args.tag}_summary.csv", summary)
    paired = paired_tests(rows)
    thresholds = threshold_table(unanswerable_top, list(answerable_top.values()))
    write_csv(RESULTS / f"retrieval_{args.tag}_threshold.csv", thresholds)
    md = render_markdown(summary, paired, thresholds, bundle.meta, len(unanswerable_top))
    (RESULTS / f"retrieval_{args.tag}.md").write_text(md, encoding="utf-8")
    print(md)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["type"] == "answerable":
            groups[(r["config"], "all")].append(r)
            groups[(r["config"], r["lang"])].append(r)
        else:
            groups[(r["config"], "crosslingual")].append(r)
    out = []
    for (cfg, group), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], CONFIGS.index(kv[0][0]))):
        row = {"config": cfg, "group": group, "n": len(rs)}
        for m in ("R@5", "R@10", "MRR@10", "nDCG@10", "docR@10"):
            mean, lo, hi = bootstrap_ci([r[m] for r in rs])
            row[m] = round(mean, 4)
            row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        lat = [r["latency_ms"] for r in rs]
        row["latency_p50_ms"] = round(statistics.median(lat), 1)
        out.append(row)
    return out


def paired_tests(rows: list[dict]) -> list[tuple[str, float, float]]:
    by_cfg: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["type"] == "answerable":
            by_cfg[r["config"]][r["id"]] = r["R@10"]
    ids = sorted(by_cfg["hybrid+rerank"])
    out = []
    for other in ("bm25", "dense", "hybrid"):
        diff, p = paired_bootstrap([by_cfg["hybrid+rerank"][i] for i in ids],
                                   [by_cfg[other][i] for i in ids])
        out.append((other, diff, p))
    return out


def threshold_table(unanswerable: list[float], answerable: list[float]) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        rows.append({
            "threshold": t,
            "unanswerable_abstain": round(sum(x < t for x in unanswerable) / max(1, len(unanswerable)), 3),
            "answerable_false_abstain": round(sum(x < t for x in answerable) / max(1, len(answerable)), 3),
        })
    return rows


def render_markdown(summary, paired, thresholds, meta, n_unans) -> str:
    lines = [f"Index: {meta['n_chunks']} chunks, embed `{meta['embed_model']}`, "
             f"chunker `{meta['chunker_hash']}`\n"]
    for group in ("all", "vi", "en", "crosslingual"):
        rs = [r for r in summary if r["group"] == group]
        if not rs:
            continue
        lines += [f"**{group}** (n={rs[0]['n']})\n",
                  "| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Doc Recall@10 | p50 ms |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rs:
            lines.append(f"| {r['config']} | {r['R@5']:.3f} | {r['R@10']:.3f} {r['R@10_ci']} | "
                         f"{r['MRR@10']:.3f} | {r['nDCG@10']:.3f} | {r['docR@10']:.3f} | "
                         f"{r['latency_p50_ms']} |")
        lines.append("")
    lines.append("Paired bootstrap on Recall@10 (answerable): hybrid+rerank minus ...\n")
    lines += ["| vs | Δ Recall@10 | p (Δ ≤ 0) |", "| --- | --- | --- |"]
    lines += [f"| {o} | {d:+.3f} | {p:.3f} |" for o, d, p in paired]
    lines += ["", f"Rerank threshold (top score; {n_unans} unanswerable questions)\n",
              "| Threshold | Unanswerable correctly abstained | Answerable wrongly abstained |",
              "| --- | --- | --- |"]
    lines += [f"| {r['threshold']} | {r['unanswerable_abstain']:.1%} | {r['answerable_false_abstain']:.1%} |"
              for r in thresholds]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
