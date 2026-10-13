"""Text normalisation that is safe for the chunker.

Deliberately does NOT collapse spaces inside a line: runs of spaces/tabs are how the chunker
recognises column-aligned tables in text extracted from PDFs.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from medrag.text import nfc

log = logging.getLogger(__name__)

# zero-width space/joiners, word joiner, BOM, soft hyphen
_INVISIBLE = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD], None)
# no-break / figure / narrow no-break spaces -> one plain space (keeps column alignment)
_SPACES = {0x00A0: " ", 0x2007: " ", 0x202F: " "}
# Characters typical of Vietnamese text in legacy TCVN3/VNI fonts decoded as Latin-1.
_LEGACY_VN = set("¸µ¶·¹¨©ª«¬®¯°±²³´½¾ÆÇÐÑØ")
_PAGE_NUMBER = re.compile(r"^\s*(trang|page)?\s*\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = nfc(text).translate(_INVISIBLE).translate(_SPACES)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if looks_like_legacy_encoding(text):
        log.warning("text looks like legacy TCVN3/VNI encoding; convert it to Unicode first")
    return text


def looks_like_legacy_encoding(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    legacy = sum(ch in _LEGACY_VN for ch in text)
    return letters > 0 and legacy / letters > 0.01


def remove_page_furniture(pages: list[str], min_share: float = 0.5) -> list[str]:
    """Drop running headers/footers and page numbers from PDF pages.

    A short line counts as furniture when it appears on at least `min_share` of the pages.
    """
    if len(pages) < 3:
        return [_drop_page_numbers(p) for p in pages]
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({ln.strip() for ln in page.split("\n") if 0 < len(ln.strip()) < 100})
    repeated = {ln for ln, c in counts.items() if c >= min_share * len(pages)}
    cleaned = []
    for page in pages:
        kept = [ln for ln in page.split("\n") if ln.strip() not in repeated]
        cleaned.append(_drop_page_numbers("\n".join(kept)))
    return cleaned


def _drop_page_numbers(page: str) -> str:
    return "\n".join(ln for ln in page.split("\n") if not _PAGE_NUMBER.match(ln))
