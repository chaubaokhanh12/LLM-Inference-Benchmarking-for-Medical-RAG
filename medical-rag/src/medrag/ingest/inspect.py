"""Print how a document is chunked, for eyeballing the chunker.

    python -m medrag.ingest.inspect data/raw/wikipedia/vi/sot_xuat_huyet_dengue.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medrag.ingest.chunker import Chunker, ChunkerConfig
from medrag.ingest.loader import load_document


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--width", type=int, default=160, help="characters of text shown per chunk")
    ap.add_argument("--split-oversized", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    doc = load_document(args.path)
    chunks = Chunker(ChunkerConfig(split_oversized=args.split_oversized)).chunk(doc)
    print(f"{doc.title} [{doc.lang}] -> {len(chunks)} chunks\n")
    for c in chunks:
        flag = " OVERSIZED" if c.oversized else ""
        print(f"#{c.chunk_id.rsplit(':', 1)[1]:>3} {c.n_tokens:>4} tok{flag} | {' > '.join(c.section_path)}")
        body = c.text.replace("\n", " ⏎ ")
        print(f"      {body[: args.width]}{'…' if len(body) > args.width else ''}\n")


if __name__ == "__main__":
    main()
