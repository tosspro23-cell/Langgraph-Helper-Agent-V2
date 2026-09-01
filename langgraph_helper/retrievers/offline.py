"""Offline retrieval over a locally pre-built BM25 index.

The index is built once (offline, at prep time) by data_prep/build_index.py
from the raw llms.txt / llms-full.txt dumps in data/raw/. At query time this
module only reads the small chunks.jsonl file that ships in the repo and
scores it with BM25 -- no network access, no vector DB, no ML model
download required. See README.md "Data Freshness Strategy" for the
rationale.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from ..config import CHUNKS_PATH
from ..state import SourceChunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _load_index():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"No offline index found at {CHUNKS_PATH}.\n"
            "Build it first with:\n"
            "  python data_prep/download_docs.py\n"
            "  python data_prep/build_index.py"
        )

    from rank_bm25 import BM25Okapi

    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, chunks


def search(query: str, k: int = 5) -> list[SourceChunk]:
    bm25, chunks = _load_index()
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    results: list[SourceChunk] = []
    for i in ranked[:k]:
        if scores[i] <= 0:
            continue
        c = chunks[i]
        results.append(
            SourceChunk(
                title=c["title"],
                url=c["url"],
                text=c["text"],
                score=float(scores[i]),
            )
        )
    return results
