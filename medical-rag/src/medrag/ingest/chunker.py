"""Section-aware chunker.

Pipeline for one document:

1. classify lines   heading / table row / list item / text / blank
2. build sections   a heading stack gives every section its `section_path`
3. build blocks     paragraphs, and atomic tables and lists (never split)
4. pack             fill chunks up to a token target inside each section, with a
                    one-sentence overlap between neighbouring chunks of the same section

Token counts use the embedding model's tokenizer and are measured on the exact string that
gets embedded ("passage: " + header + text), so no chunk is silently truncated by the encoder.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from medrag.types import Chunk, Document

# --------------------------------------------------------------------------------------------
# Configuration and token counting
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkerConfig:
    tokenizer_name: str = "intfloat/multilingual-e5-base"
    max_tokens: int = 512  # hard limit of the encoder, special tokens included
    target_tokens: int = 450  # stop filling a chunk around here
    min_tokens: int = 350  # below this, keep filling past paragraph boundaries
    overlap_tokens: int = 50  # ~one sentence repeated from the previous chunk
    passage_prefix: str = "passage: "
    safety_margin: int = 8
    min_section_tokens: int = 40  # tinier sections are merged into their first subsection
    max_heading_chars: int = 80
    split_oversized: bool = False  # split oversized tables/lists at row/item boundaries
    version: str = "1"

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


class TokenCounter:
    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = 10**9  # we only count; silence the >512 warning

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_with_specials(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))


@lru_cache(maxsize=4)
def get_token_counter(tokenizer_name: str) -> TokenCounter:
    return TokenCounter(tokenizer_name)


# --------------------------------------------------------------------------------------------
# 1. Line classification
# --------------------------------------------------------------------------------------------

MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
VN_HEADING = re.compile(
    r"^(chương|chuong|phần|phan|mục|muc|điều|dieu)\s+(\d{1,3}|[IVXLC]{1,6})\b\s*[.:\-–]?\s*(.*)$",
    re.IGNORECASE,
)
VN_RANK = {"phan": 1, "phần": 1, "chuong": 2, "chương": 2, "muc": 3, "mục": 3, "dieu": 4, "điều": 4}
ROMAN_HEADING = re.compile(
    r"^(?=[IVXLC]+\.)(C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\.\s+(\S.*)$"
)
NUM_HEADING = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)(\.?)\s+(\S.*)$")
LIST_ITEM = re.compile(r"^\s*(?:[-*•+▪◦●]|\d{1,3}[.)]|[a-zđ][.)])\s+\S")
NUM_ITEM = re.compile(r"^\s*(\d{1,3})[.)]\s+\S")
TABLE_MD = re.compile(r"^\s*\|")
COLUMN_GAP = re.compile(r"\S(?:\t+| {3,})(?=\S)")
CAPTION = re.compile(r"^(bảng|bang|table)\s*\d+", re.IGNORECASE)
MD_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


@dataclass
class Line:
    text: str
    kind: str  # blank | heading | table | list | text
    heading_key: tuple[str, int] | None = None  # (family, rank)
    heading_title: str | None = None


def _heading_text_ok(title: str, line: str) -> bool:
    """Guards against list items or sentences that merely start with a number."""
    title = title.strip()
    return (
        bool(title)
        and title[0].isupper()
        and len(line) <= 120
        and len(title.split()) <= 15
        and not title.endswith((".", ";", ","))
    )


def classify_lines(text: str, cfg: ChunkerConfig) -> list[Line]:
    raw = text.split("\n")
    lines: list[Line] = []
    num1_candidates: list[int] = []
    for s in raw:
        st = s.strip()
        if not st:
            lines.append(Line(s, "blank"))
            continue
        if m := MD_HEADING.match(st):
            title = m.group(2).strip().strip("*").strip()
            lines.append(Line(s, "heading", ("md", len(m.group(1))), title))
            continue
        if (m := VN_HEADING.match(st)) and len(st) <= 150 and (
            not m.group(3) or m.group(3)[0].isupper()
        ):
            rank = VN_RANK[m.group(1).lower()]
            lines.append(Line(s, "heading", ("vn", rank), st))
            continue
        if (m := ROMAN_HEADING.match(st)) and _heading_text_ok(m.group(2), st):
            lines.append(Line(s, "heading", ("roman", 1), st))
            continue
        if TABLE_MD.match(s):
            lines.append(Line(s, "table"))
            continue
        if m := NUM_HEADING.match(st):
            depth = m.group(1).count(".") + 1
            if depth >= 2 and _heading_text_ok(m.group(3), st):
                lines.append(Line(s, "heading", ("num", depth), st))
                continue
            if depth == 1 and m.group(2) == "." and _heading_text_ok(m.group(3), st):
                num1_candidates.append(len(lines))
                lines.append(Line(s, "heading", ("num", 1), st))  # may be demoted below
                continue
        if LIST_ITEM.match(s):
            lines.append(Line(s, "list"))
            continue
        lines.append(Line(s, "columnar" if COLUMN_GAP.search(s.strip()) else "text"))

    for i in num1_candidates:
        if _is_numbered_list_item(lines, i):
            lines[i] = Line(lines[i].text, "list")
    _resolve_columnar(lines)
    return lines


def _neighbour(lines: list[Line], i: int, step: int) -> Line | None:
    j = i + step
    while 0 <= j < len(lines):
        if lines[j].kind != "blank":
            return lines[j]
        j += step
    return None


def _item_number(line: Line | None) -> int | None:
    if line is None:
        return None
    m = NUM_ITEM.match(line.text)
    return int(m.group(1)) if m else None


def _is_numbered_list_item(lines: list[Line], i: int) -> bool:
    """`1. Foo` is a list item, not a heading, when it continues or starts a numbered run.

    "1. Sốt / 2. Đau đầu / 3. Buồn nôn" -> list. "2. Điều trị" followed by a paragraph -> heading.
    """
    n = _item_number(lines[i])
    prev_n = _item_number(_neighbour(lines, i, -1))
    next_n = _item_number(_neighbour(lines, i, +1))
    return (prev_n is not None and prev_n == n - 1) or (next_n is not None and next_n == n + 1)


def _resolve_columnar(lines: list[Line]) -> None:
    """Runs of column-aligned lines become table rows (>=2 lines with tabs, >=3 with spaces)."""
    i = 0
    while i < len(lines):
        if lines[i].kind != "columnar":
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].kind == "columnar":
            j += 1
        run = lines[i:j]
        has_tab = any("\t" in ln.text for ln in run)
        kind = "table" if len(run) >= (2 if has_tab else 3) else "text"
        for ln in run:
            ln.kind = kind
        i = j


# --------------------------------------------------------------------------------------------
# 2 + 3. Sections and blocks
# --------------------------------------------------------------------------------------------


@dataclass
class Block:
    kind: str  # text | table | list
    text: str


@dataclass
class Section:
    path: list[str]
    blocks: list[Block] = field(default_factory=list)


class HeadingStack:
    """Nesting of mixed heading styles (Markdown, Chương/Mục/Điều, I., 1., 1.1.).

    Heading families are ordered by first appearance in the document; a new heading pops
    every entry of a later family, and every entry of its own family with rank >= its own.
    """

    def __init__(self) -> None:
        self.stack: list[tuple[tuple[str, int], str]] = []
        self.family_order: list[str] = []

    def push(self, key: tuple[str, int], title: str) -> list[str]:
        family, rank = key
        if family not in self.family_order:
            self.family_order.append(family)
        pos = self.family_order.index(family)
        keep = []
        for k, t in self.stack:
            kpos = self.family_order.index(k[0])
            if kpos < pos or (kpos == pos and k[1] < rank):
                keep.append((k, t))
            else:
                break
        self.stack = keep + [(key, title)]
        return [t for _, t in self.stack]


def build_sections(lines: list[Line], cfg: ChunkerConfig) -> list[Section]:
    stack = HeadingStack()
    sections = [Section(path=[])]
    para: list[str] = []
    atomic: Block | None = None

    def close_para() -> None:
        if para:
            sections[-1].blocks.append(Block("text", " ".join(p.strip() for p in para)))
            para.clear()

    def close_atomic() -> None:
        nonlocal atomic
        if atomic is not None:
            sections[-1].blocks.append(atomic)
            atomic = None

    def open_atomic(kind: str, first_line: str) -> None:
        """Start a table/list block, pulling in a caption or a short intro ending with ':'."""
        nonlocal atomic
        intro = None
        if len(para) == 1 and len(para[0]) <= 300 and (
            (kind == "table" and CAPTION.match(para[0].strip()))
            or para[0].rstrip().endswith(":")
        ):
            intro = para.pop().strip()
        close_para()
        atomic = Block(kind, (intro + "\n" if intro else "") + first_line)

    for idx, ln in enumerate(lines):
        if ln.kind == "heading":
            close_para()
            close_atomic()
            title = ln.heading_title[: cfg.max_heading_chars].strip()
            sections.append(Section(path=stack.push(ln.heading_key, title)))
        elif ln.kind == "blank":
            if atomic is not None and atomic.kind == "list":
                nxt = _neighbour(lines, idx, +1)
                if nxt is not None and nxt.kind == "list":
                    continue  # a blank line inside a list does not end it
            close_para()
            close_atomic()
        elif ln.kind in ("table", "list"):
            if atomic is not None and atomic.kind == ln.kind:
                atomic.text += "\n" + ln.text.rstrip()
            else:
                close_atomic()
                open_atomic(ln.kind, ln.text.rstrip())
        else:  # text
            if atomic is not None and atomic.kind == "list":
                atomic.text += "\n" + ln.text.rstrip()  # wrapped continuation of a list item
            else:
                close_atomic()
                para.append(ln.text)
    close_para()
    close_atomic()
    return [s for s in sections if s.blocks]


# --------------------------------------------------------------------------------------------
# 4. Packing
# --------------------------------------------------------------------------------------------

_ABBREV = {
    "e.g", "i.e", "vs", "dr", "mr", "mrs", "ms", "prof", "st", "no", "fig", "approx",
    "al", "ts", "bs", "ths", "pgs", "gs", "tp", "ca", "inc", "jr", "sr", "u.s",
}
_SENT_END = re.compile(r"([.!?…]+)([\"'”’)\]]*)\s+")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        nxt = text[m.end() : m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘(["):
            continue
        before = text[start : m.start()].split()
        word = before[-1] if before else ""
        if m.group(1) == "." and (
            word.lower().rstrip(".") in _ABBREV or (len(word) == 1 and word.isalpha())
        ):
            continue  # "Dr. Smith", "E. coli" ("v.v."/"etc." before a capital end a sentence)
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass
class Unit:
    text: str
    n_tokens: int
    atomic: bool
    para_start: bool  # first unit of a paragraph or block (joined with a blank line)
    para_tokens: int = 0  # tokens of the whole paragraph this unit starts (para_start only)


class Chunker:
    def __init__(self, cfg: ChunkerConfig | None = None, counter: TokenCounter | None = None):
        self.cfg = cfg or ChunkerConfig()
        self.counter = counter or get_token_counter(self.cfg.tokenizer_name)

    # -- public ------------------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        sections = build_sections(classify_lines(doc.text, self.cfg), self.cfg)
        sections = self._merge_tiny_sections(sections)
        chunks: list[Chunk] = []
        for sec in sections:
            for text, oversized in self._pack_section(doc.title, sec):
                chunks.append(self._make_chunk(doc, sec.path, text, oversized, len(chunks)))
        return chunks

    # -- helpers -----------------------------------------------------------------------------

    def _header(self, title: str, path: list[str]) -> str:
        return " > ".join([title, *path])

    def _embed_string(self, title: str, path: list[str], text: str) -> str:
        return f"{self.cfg.passage_prefix}{self._header(title, path)}\n{text}"

    def _make_chunk(self, doc: Document, path: list[str], text: str, oversized: bool,
                    index: int) -> Chunk:
        n = self.counter.count_with_specials(self._embed_string(doc.title, path, text))
        return Chunk(
            chunk_id=f"{doc.doc_id}:{index}", doc_id=doc.doc_id, title=doc.title,
            section_path=list(path), text=text, lang=doc.lang, n_tokens=n,
            oversized=oversized and n > self.cfg.max_tokens,
        )

    def _section_tokens(self, sec: Section) -> int:
        return sum(self.counter.count(b.text) for b in sec.blocks)

    def _merge_tiny_sections(self, sections: list[Section]) -> list[Section]:
        """A near-empty parent section ("Điều trị gồm:") is folded into its first subsection."""
        out: list[Section] = []
        carry: list[Block] = []
        for i, sec in enumerate(sections):
            sec = Section(sec.path, carry + sec.blocks)
            carry = []
            nxt = sections[i + 1] if i + 1 < len(sections) else None
            is_parent = (
                nxt is not None
                and len(nxt.path) > len(sec.path)
                and nxt.path[: len(sec.path)] == sec.path
            )
            if is_parent and self._section_tokens(sec) < self.cfg.min_section_tokens:
                carry = sec.blocks
                continue
            out.append(sec)
        return out

    def _units(self, sec: Section, budget: int) -> list[Unit]:
        units: list[Unit] = []
        for block in sec.blocks:
            if block.kind != "text":
                n = self.counter.count(block.text)
                units.append(Unit(block.text, n, atomic=True, para_start=True, para_tokens=n))
                continue
            first = len(units)
            for sent in split_sentences(block.text):
                n = self.counter.count(sent)
                pieces = [(sent, n)] if n <= budget else self._split_words(sent, budget)
                for text, k in pieces:
                    units.append(Unit(text, k, atomic=False, para_start=len(units) == first))
            if len(units) > first:
                units[first].para_tokens = sum(u.n_tokens for u in units[first:])
        return units

    def _split_words(self, text: str, budget: int) -> list[tuple[str, int]]:
        """Last resort for a single sentence longer than the budget."""
        out, cur = [], []
        for word in text.split():
            if cur and self.counter.count(" ".join(cur + [word])) > budget:
                out.append(" ".join(cur))
                cur = []
            cur.append(word)
        if cur:
            out.append(" ".join(cur))
        return [(t, self.counter.count(t)) for t in out]

    def _overlap_unit(self, chunk_units: list[Unit]) -> Unit | None:
        last = chunk_units[-1]
        if last.atomic:
            return None  # never repeat part of a table or list
        if last.n_tokens <= int(self.cfg.overlap_tokens * 1.6):
            return Unit(last.text, last.n_tokens, atomic=False, para_start=False)
        words = last.text.split()
        tail: list[str] = []
        while words and self.counter.count(" ".join([words[-1], *tail])) <= self.cfg.overlap_tokens:
            tail.insert(0, words.pop())
        if not tail:
            return None
        text = " ".join(tail)
        return Unit(text, self.counter.count(text), atomic=False, para_start=False)

    def _pack_section(self, title: str, sec: Section) -> list[tuple[str, bool]]:
        cfg = self.cfg
        header_cost = self.counter.count_with_specials(self._embed_string(title, sec.path, ""))
        budget = cfg.max_tokens - header_cost - cfg.safety_margin
        target = min(cfg.target_tokens, budget)
        min_fill = min(cfg.min_tokens, target)

        out: list[tuple[list[Unit], bool]] = []
        cur: list[Unit] = []  # may start with an overlap unit copied from the previous chunk
        cur_tok = 0
        fresh = False  # cur holds at least one unit that is not overlap

        def emit(next_unit: Unit) -> None:
            nonlocal cur, cur_tok, fresh
            out.append((cur, False))
            ov = self._overlap_unit(cur)
            if ov is not None and ov.n_tokens + next_unit.n_tokens > budget:
                ov = None
            cur, cur_tok, fresh = ([ov], ov.n_tokens, False) if ov else ([], 0, False)

        for u in self._units(sec, budget):
            if u.atomic and u.n_tokens > budget:
                if fresh:
                    out.append((cur, False))
                cur, cur_tok, fresh = [], 0, False
                pieces = self._split_atomic(u, budget) if cfg.split_oversized else [u]
                out.extend(([p], p.n_tokens > budget) for p in pieces)
                continue
            # Prefer cutting at a paragraph boundary once the chunk is reasonably full.
            if fresh and u.para_start and cur_tok >= min_fill and cur_tok + u.para_tokens > target:
                emit(u)
            if fresh and cur_tok + u.n_tokens > target:
                emit(u)
            if cur_tok + u.n_tokens > budget:  # only an overlap unit can be in the way here
                cur, cur_tok = [], 0
            cur.append(u)
            cur_tok += u.n_tokens
            fresh = True
        if fresh:
            out.append((cur, False))

        results: list[tuple[str, bool]] = []
        for units, oversized in out:
            results.extend(self._finalize(title, sec.path, units, oversized))
        return results

    def _join(self, units: list[Unit]) -> str:
        parts: list[str] = []
        for i, u in enumerate(units):
            if i and (u.para_start or u.atomic or units[i - 1].atomic):
                parts.append("\n\n")
            elif i:
                parts.append(" ")
            parts.append(u.text)
        return "".join(parts).strip()

    def _finalize(self, title: str, path: list[str], units: list[Unit],
                  oversized: bool) -> list[tuple[str, bool]]:
        """Guarantee the token limit on the real string; split in half if estimates were off."""
        text = self._join(units)
        if oversized or len(units) == 1:
            return [(text, oversized)]
        n = self.counter.count_with_specials(self._embed_string(title, path, text))
        if n <= self.cfg.max_tokens:
            return [(text, False)]
        mid = len(units) // 2
        return (self._finalize(title, path, units[:mid], False)
                + self._finalize(title, path, units[mid:], False))

    def _split_atomic(self, unit: Unit, budget: int) -> list[Unit]:
        """Split an oversized table at row boundaries (repeating its header) or a list at items."""
        lines = unit.text.split("\n")
        header: list[str] = []
        if TABLE_MD.match(lines[0]) or CAPTION.match(lines[0]):
            # caption/intro lines, header row, separator row are repeated in every piece
            k = 0
            while k < len(lines) and not TABLE_MD.match(lines[k]):
                k += 1
            if k + 1 < len(lines) and MD_TABLE_SEPARATOR.match(lines[k + 1]):
                header, lines = lines[: k + 2], lines[k + 2 :]
            items = [[ln] for ln in lines]
        else:
            items = []
            for ln in lines:
                if not items or (LIST_ITEM.match(ln) and not ln[:1].isspace()):
                    items.append([ln])
                else:
                    items[-1].append(ln)
        pieces: list[Unit] = []
        cur: list[str] = []
        for item in items:
            candidate = header + cur + item
            if cur and self.counter.count("\n".join(candidate)) > budget:
                text = "\n".join(header + cur)
                pieces.append(Unit(text, self.counter.count(text), True, True))
                cur = []
            cur.extend(item)
        if cur:
            text = "\n".join(header + cur)
            pieces.append(Unit(text, self.counter.count(text), True, True))
        return pieces


def chunk_documents(docs: list[Document], cfg: ChunkerConfig | None = None) -> list[Chunk]:
    chunker = Chunker(cfg)
    return [c for doc in docs for c in chunker.chunk(doc)]
