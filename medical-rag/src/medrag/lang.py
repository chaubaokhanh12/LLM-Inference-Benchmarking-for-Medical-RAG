"""Vietnamese / English language detection tuned for short queries.

Accented Vietnamese is easy: its diacritic letters never appear in English. Unaccented
Vietnamese ("dau dau uong thuoc gi") is caught by a list of common Vietnamese words written
without accents; words that are also English ("the", "can", "an", "do", "to", "in") are left out.
"""

from __future__ import annotations

import re

VI_CHARS = set(
    "ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
)
# Characters that only Vietnamese uses (French/Spanish share à, é, ...; these they do not).
VI_ONLY = set("ăđơưảãạằẳẵắặầẩẫấậẻẽẹềểễếệỉĩịỏọồổỗốộờởỡớợủũụừửữứựỳỷỹỵ")

VI_UNACCENTED = {
    "khong", "cua", "cac", "nhung", "duoc", "trong", "nguoi", "benh", "thuoc", "nhieu", "nao",
    "gi", "bao", "sao", "lam", "cho", "voi", "nhu", "dieu", "tri", "chung", "uong", "trieu",
    "nen", "phai", "gay", "mot", "va", "la", "khi", "hay", "bi", "co", "tre", "em",
    "dau", "sot", "ho", "viem", "nhiem", "lieu", "dung", "tac", "phong", "ngua", "vi", "rut",
}
# English words whose spelling is not also unaccented Vietnamese ("the"=thể, "can"=cần, "it"=ít).
EN_WORDS = {
    "of", "and", "is", "are", "what", "how", "which", "for", "with", "does", "why", "when",
    "should", "this", "that", "symptoms", "treatment", "dose", "disease", "cause", "causes",
    "from", "or", "by", "used", "patients", "risk", "if", "after", "during", "your",
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_lang(text: str) -> str:
    """Return "vi" or "en"."""
    lower = text.lower()
    letters = [ch for ch in lower if ch.isalpha()]
    if not letters:
        return "en"
    vi_chars = sum(ch in VI_CHARS for ch in letters)
    if any(ch in VI_ONLY for ch in letters) or vi_chars / len(letters) > 0.05:
        return "vi"
    words = _WORD.findall(lower)
    vi = sum(w in VI_UNACCENTED for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "vi" if vi > en else "en"
