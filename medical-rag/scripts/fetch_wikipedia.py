"""Fetch medical Wikipedia articles (English + Vietnamese) as Markdown files with front-matter.

HTML (not the plain-text extract) is fetched so that tables and lists survive the conversion;
infoboxes, navboxes, references and "See also"-type sections are removed.

Usage (from medical-rag/):
    python scripts/fetch_wikipedia.py                  # resolve topics, fetch latest revisions
    python scripts/fetch_wikipedia.py --from-manifest  # re-fetch the exact revisions in the manifest

Outputs:
    data/raw/wikipedia/{en,vi}/*.md       corpus (indexed)
    data/holdout/wikipedia/{en,vi}/*.md   held-out topics (never indexed; for unanswerable questions)
    data/sources/wikipedia_manifest.json  titles + revision ids, for reproducibility
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from medrag.config import PROJECT_ROOT
from medrag.text import nfc, slugify

USER_AGENT = (
    "medrag-dataset/0.1 "
    "(https://github.com/chaubaokhanh12/LLM-Inference-Benchmarking-for-Medical-RAG)"
)
SOURCES = PROJECT_ROOT / "data" / "sources"
MANIFEST = SOURCES / "wikipedia_manifest.json"
HTML_CACHE = PROJECT_ROOT / "data" / "cache" / "wikipedia_html"
OUT_DIRS = {
    "corpus": PROJECT_ROOT / "data" / "raw" / "wikipedia",
    "holdout": PROJECT_ROOT / "data" / "holdout" / "wikipedia",
}

REMOVE_SELECTORS = [
    "style", "script", "link", "meta", "figure", ".thumb", ".gallery", ".infobox",
    ".navbox", ".vertical-navbox", ".sidebar", ".metadata", ".ambox", ".mbox-small",
    ".hatnote", ".shortdescription", ".toc", ".noprint", ".mw-editsection", ".reflist",
    ".references", ".mw-references-wrap", "sup.reference", ".mw-cite-backlink", ".portalbox",
    ".catlinks", ".mw-empty-elt", ".navbox-styles",
]

# Sections dropped together with their subsections: a heading matching any of these words
# ("Notes and references", "Works cited", "Tham khảo và chú thích", ...).
STOP_SECTIONS = {
    "en": re.compile(
        r"\b(references?|see also|external links?|further reading|notes|bibliography|sources|"
        r"citations|footnotes|works cited|cited works|society and culture|popular culture|"
        r"other animals|gallery)\b", re.IGNORECASE),
    "vi": re.compile(
        # "nguồn" only as the whole title: "Nguồn lây" (source of infection) is clinical
        r"(tham khảo|chú thích|xem thêm|liên kết ngoài|đọc thêm|ghi chú|thư mục|^nguồn$|"
        r"văn hóa đại chúng)", re.IGNORECASE),
}


class WikiClient:
    def __init__(self) -> None:
        self.http = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60)

    def api(self, lang: str, params: dict) -> dict:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        for attempt in range(5):
            r = self.http.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504) or "maxlag" in r.text[:200]:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Wikipedia API kept failing for {params}")

    def resolve(self, titles: list[str]) -> dict[str, dict]:
        """EN title -> {"en": canonical EN title, "vi": VI title or None}."""
        out: dict[str, dict] = {}
        for i in range(0, len(titles), 50):
            batch = titles[i : i + 50]
            data = self.api("en", {
                "action": "query", "titles": "|".join(batch), "prop": "langlinks",
                "lllang": "vi", "lllimit": "max", "redirects": 1,
            })
            q = data["query"]
            renames = {r["from"]: r["to"] for r in q.get("normalized", []) + q.get("redirects", [])}
            pages = {p["title"]: p for p in q["pages"]}
            for t in batch:
                canon = t
                while canon in renames:
                    canon = renames[canon]
                page = pages.get(canon)
                if page is None or page.get("missing"):
                    print(f"  [missing] {t}", file=sys.stderr)
                    continue
                vi = next((ll["title"] for ll in page.get("langlinks", []) if ll["lang"] == "vi"), None)
                out[t] = {"en": canon, "vi": vi}
        return out

    def parse(self, lang: str, title: str | None = None, revid: int | None = None) -> dict:
        params = {
            "action": "parse", "prop": "text|revid|displaytitle", "redirects": 1,
            "disableeditsection": 1, "disabletoc": 1, "disablelimitreport": 1,
        }
        if revid:
            params["oldid"] = revid
        else:
            params["page"] = title
        return self.api(lang, params)["parse"]


def html_to_markdown(html: str, lang: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in REMOVE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    # Keep only content tables; layout tables disappear (with any tables nested inside them).
    for table in soup.find_all("table"):
        if table.decomposed:
            continue
        if "wikitable" not in (table.get("class") or []):
            table.decompose()
    # Formulas: keep the TeX source instead of garbled MathML.
    for el in soup.select(".mwe-math-element"):
        math = el.find("math")
        tex = (math.get("alttext") if math else "") or ""
        tex = re.sub(r"^\{\\displaystyle\s*(.*)\}$", r"\1", tex.strip())
        el.replace_with(f" {tex} ")

    md = markdownify(
        # links, images and bold/italic markers would only add noise for embeddings and BM25
        str(soup), heading_style="ATX", bullets="-",
        strip=["a", "img", "b", "strong", "i", "em", "sup", "sub"],
        escape_underscores=False, escape_asterisks=False, escape_misc=False,
    )
    md = nfc(md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return drop_sections(md, STOP_SECTIONS[lang])


def drop_sections(md: str, stop: re.Pattern) -> str:
    out: list[str] = []
    skip_level: int | None = None
    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip().rstrip(":").lower()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and stop.search(title):
                skip_level = level
        if skip_level is None:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def write_doc(split: str, lang: str, title: str, revid: int, topic_en: str, md: str) -> Path:
    path = OUT_DIRS[split] / lang / f"{slugify(title)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    front = "\n".join([
        "---", f"title: {title}", f"lang: {lang}", "source: wikipedia", f"url: {url}",
        f"revid: {revid}", "license: CC BY-SA 4.0", f"topic_en: {topic_en}", "---", "",
    ])
    path.write_text(front + md + "\n", encoding="utf-8")
    return path


def read_topics(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def fetch_one(client: WikiClient, lang: str, title: str, revid: int | None = None) -> dict | None:
    """Fetch (or reuse the cached HTML of) one article and convert it to Markdown."""
    cache = HTML_CACHE / lang / f"{slugify(title)}.json"
    try:
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
        if cached and (revid is None or cached["revid"] == revid):
            p = cached
        else:
            p = client.parse(lang, title=title, revid=revid)
            p = {"title": p["title"], "revid": p["revid"], "text": p["text"]}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")
        return {"title": p["title"], "revid": p["revid"], "md": html_to_markdown(p["text"], lang)}
    except Exception as e:  # noqa: BLE001 - report and continue with the other articles
        print(f"  [error] {lang}:{title}: {e!r}", file=sys.stderr)
        return None


def fresh(args: argparse.Namespace) -> list[dict]:
    client = WikiClient()
    rng = random.Random(args.seed)
    records: list[dict] = []
    for split, topics_file in (("corpus", "topics_en.txt"), ("holdout", "topics_holdout_en.txt")):
        topics = read_topics(SOURCES / topics_file)
        print(f"[{split}] resolving {len(topics)} topics")
        resolved = client.resolve(topics)
        docs = {"en": [], "vi": []}
        for i, names in enumerate(resolved.values(), 1):
            for lang in ("en", "vi"):
                if names[lang] is None:
                    continue
                doc = fetch_one(client, lang, names[lang])
                if doc and len(doc["md"]) >= args.min_chars:
                    docs[lang].append({**doc, "topic_en": names["en"]})
            if i % 25 == 0:
                print(f"  {i}/{len(resolved)} topics fetched")
        for lang in ("en", "vi"):
            chosen = docs[lang]
            if split == "corpus" and len(chosen) > args.max_per_lang:
                chosen = rng.sample(chosen, args.max_per_lang)
            for d in sorted(chosen, key=lambda d: d["title"]):
                path = write_doc(split, lang, d["title"], d["revid"], d["topic_en"], d["md"])
                records.append({
                    "split": split, "lang": lang, "title": d["title"], "revid": d["revid"],
                    "topic_en": d["topic_en"], "n_chars": len(d["md"]),
                    "file": path.relative_to(PROJECT_ROOT).as_posix(),
                })
            print(f"[{split}] {lang}: kept {len(chosen)} (of {len(docs[lang])} long enough)")
    return records


def from_manifest() -> list[dict]:
    client = WikiClient()
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for i, r in enumerate(records, 1):
        doc = fetch_one(client, r["lang"], r["title"], revid=r["revid"])
        if doc:
            write_doc(r["split"], r["lang"], r["title"], r["revid"], r["topic_en"], doc["md"])
        if i % 50 == 0:
            print(f"  {i}/{len(records)}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-manifest", action="store_true", help="re-fetch exact revisions")
    ap.add_argument("--min-chars", type=int, default=3000, help="drop shorter (stub) articles")
    ap.add_argument("--max-per-lang", type=int, default=150, help="corpus cap per language")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.from_manifest:
        from_manifest()
        return
    records = fresh(args)
    MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    n = {(r["split"], r["lang"]): 0 for r in records}
    for r in records:
        n[(r["split"], r["lang"])] += 1
    print("done:", {f"{s}/{lang}": c for (s, lang), c in sorted(n.items())})


if __name__ == "__main__":
    main()
