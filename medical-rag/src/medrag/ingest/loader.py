"""Read Markdown / TXT / PDF files into `Document`s.

Markdown and TXT files may start with a simple front-matter block:

    ---
    title: Sốt xuất huyết Dengue
    lang: vi
    ---
"""

from __future__ import annotations

import logging
from pathlib import Path

from medrag.ingest.cleaner import clean_text, remove_page_furniture
from medrag.lang import detect_lang
from medrag.text import slugify
from medrag.types import Document

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


def load_documents(input_dir: str | Path) -> list[Document]:
    root = Path(input_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())
    docs = []
    for path in files:
        try:
            docs.append(load_document(path, root))
        except Exception as e:  # noqa: BLE001 - one bad file should not stop the build
            log.error("skipping %s: %s", path, e)
    return docs


def load_document(path: Path, root: Path | None = None) -> Document:
    if path.suffix.lower() == ".pdf":
        meta, body = {}, _read_pdf(path)
    else:
        meta, body = split_front_matter(path.read_text(encoding="utf-8-sig"))
    text = clean_text(body)
    rel = path.relative_to(root) if root else Path(path.name)
    doc_id = "/".join(slugify(part) for part in rel.with_suffix("").parts)
    title = meta.pop("title", None) or _first_heading(text) or path.stem.replace("_", " ")
    lang = meta.pop("lang", None) or detect_lang(text[:5000])
    return Document(doc_id=doc_id, title=title, lang=lang, text=text,
                    source_path=rel.as_posix(), meta=meta)


def split_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw
    meta = {}
    for line in raw[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, raw[end + 4 :].lstrip("\n")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    return "\n\n".join(remove_page_furniture(pages))


def _first_heading(text: str) -> str | None:
    for line in text.split("\n", 20)[:20]:
        if line.startswith("# "):
            return line[2:].strip()
    return None
