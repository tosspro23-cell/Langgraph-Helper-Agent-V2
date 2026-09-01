#!/usr/bin/env python3
"""Data preparation, step 2: turn the raw docs dump into retrieval chunks.

Fully offline -- only reads data/raw/*.txt and writes data/index/chunks.jsonl.
No network access, no embedding model download.

Parsing: the llms-full.txt corpus concatenates one markdown page per doc,
each starting with a "# Title" line immediately followed by a
"Source: <url>" line. We split on that pattern, then split each page's body
into overlapping character-based chunks (paragraph-aware) so the BM25
retriever in langgraph_helper/retrievers/offline.py has reasonably sized,
self-contained passages to score and return.

Why BM25 instead of embeddings: it needs no ML model download/inference
(fully offline, installs in seconds via `rank-bm25`, a pure-Python package),
and for API-reference-style docs with lots of exact identifiers
(StateGraph, add_conditional_edges, ...) lexical matching is a strong,
predictable baseline. See README.md for the tradeoff discussion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
OUT_PATH = INDEX_DIR / "chunks.jsonl"

# Primary corpus: contains both LangChain and LangGraph Python docs.
# See download_docs.py for why this is the file we index.
SOURCE_FILES = ["langchain_oss_python_llms_full.txt"]

PAGE_RE = re.compile(r"(?m)^# (.+?)\nSource: (\S+)\n")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def split_pages(text: str) -> list[dict]:
    matches = list(PAGE_RE.finditer(text))
    pages = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        url = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            pages.append({"title": title, "url": url, "body": body})
    return pages


def chunk_text(body: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current.strip())
            # start next chunk with the tail of the previous one for continuity
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for filename in SOURCE_FILES:
            path = RAW_DIR / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run: python data_prep/download_docs.py"
                )
            text = path.read_text(encoding="utf-8")
            pages = split_pages(text)
            total_pages += len(pages)
            print(f"{filename}: {len(pages)} pages")

            chunk_id = 0
            for page in pages:
                for piece in chunk_text(page["body"], CHUNK_SIZE, CHUNK_OVERLAP):
                    record = {
                        "id": f"{filename}:{page['url']}#{chunk_id}",
                        "title": page["title"],
                        "url": page["url"],
                        "text": f"# {page['title']}\n\n{piece}",
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_id += 1
                    total_chunks += 1

    print(f"\nWrote {total_chunks} chunks from {total_pages} pages -> {OUT_PATH}")


if __name__ == "__main__":
    main()
