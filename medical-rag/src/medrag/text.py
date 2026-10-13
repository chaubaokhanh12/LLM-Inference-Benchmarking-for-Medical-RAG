"""Small text helpers shared by ingestion, indexing and language detection."""

from __future__ import annotations

import re
import unicodedata


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_accents(text: str) -> str:
    """Remove Vietnamese/Latin diacritics: 'Đau đầu' -> 'Dau dau'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def slugify(text: str, max_len: int = 80) -> str:
    """ASCII, filesystem-safe identifier: 'Sốt xuất huyết Dengue' -> 'sot_xuat_huyet_dengue'."""
    s = strip_accents(text).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len] or "untitled"
