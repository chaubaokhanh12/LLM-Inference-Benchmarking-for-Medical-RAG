"""BM25 over a bilingual corpus.

Vietnamese is written in syllables separated by spaces, so "đau đầu" (headache) would be two
unrelated BM25 terms. pyvi joins compound words ("đau_đầu"); it runs on the original casing
(its CRF uses capitalisation features) and the result is lower-cased afterwards.
Optionally the syllables of each compound are indexed too, which helps when the query is
segmented differently from the document.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from medrag.text import nfc

TOKENIZER_VERSION = "pyvi+regex-1"

VI_STOPWORDS = {
    "và", "của", "là", "có", "được", "các", "những", "cho", "với", "trong", "một", "này", "đó",
    "khi", "thì", "mà", "để", "từ", "theo", "về", "như", "nhưng", "hoặc", "hay", "cũng", "đã",
    "đang", "sẽ", "bị", "do", "vì", "nên", "rất", "lại", "ra", "vào", "trên", "dưới", "tại",
    "nếu", "thế", "nào", "gì", "sao", "ai", "đâu", "bao_nhiêu", "là_gì", "như_thế_nào",
    "thế_nào", "tại_sao", "vì_sao", "hãy", "cần", "phải", "còn", "đến", "ở", "những_gì",
    "người_ta", "ta", "chúng", "nó", "họ", "tôi", "bạn", "đây", "kia", "ấy", "vậy", "thường",
    "có_thể", "không", "các_loại", "nhiều", "mỗi", "cả", "đều", "chỉ", "rằng",
}
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "which", "who", "whom", "what", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "may", "might", "will", "shall",
    "has", "have", "had", "i", "you", "he", "she", "we", "they", "them", "their", "there",
    "than", "then", "so", "such", "also", "into", "about", "over", "after", "before", "if",
    "not", "no", "other", "some", "any", "most", "more", "very", "s", "my", "your", "our",
}

_EN_TOKEN = re.compile(r"\w+(?:[-.']\w+)*")
_HAS_WORD_CHAR = re.compile(r"\w")


def tokenize(text: str, lang: str, add_syllables: bool = False) -> list[str]:
    text = nfc(text)
    if lang == "vi":
        from pyvi import ViTokenizer

        raw = ViTokenizer.tokenize(text).split()
        stop = VI_STOPWORDS
    else:
        raw = _EN_TOKEN.findall(text)
        stop = EN_STOPWORDS
    out: list[str] = []
    for tok in raw:
        tok = tok.lower().strip(".,;:!?()[]{}\"'“”‘’/\\|-_")
        if not tok or not _HAS_WORD_CHAR.search(tok) or tok in stop:
            continue
        out.append(tok)
        if add_syllables and "_" in tok:
            out.extend(s for s in tok.split("_") if s and s not in stop)
    return out


class BM25Index:
    def __init__(self, bm25: BM25Okapi, add_syllables: bool):
        self.bm25 = bm25
        self.add_syllables = add_syllables

    @classmethod
    def build(cls, texts: list[str], langs: list[str], add_syllables: bool = False) -> BM25Index:
        corpus = [tokenize(t, lang, add_syllables) for t, lang in zip(texts, langs, strict=True)]
        return cls(BM25Okapi(corpus), add_syllables)

    def search(self, query: str, lang: str, k: int) -> list[tuple[int, float]]:
        """Top-k (row, score), dropping documents that share no term with the query."""
        tokens = tokenize(query, lang, self.add_syllables)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "add_syllables": self.add_syllables,
                         "tokenizer": TOKENIZER_VERSION}, f)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        # The pickle is produced by our own build step; never load one from an untrusted source.
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["add_syllables"])
