#!/usr/bin/env python3
"""Data preparation, step 1: download the llms.txt documentation sources.

This is the ONLY step in the whole pipeline that needs internet access, and
it only needs to run once (re-run it later to refresh the docs -- see
"Data Freshness Strategy" in README.md).

Source note: the assignment lists
    https://langchain-ai.github.io/langgraph/llms.txt
    https://langchain-ai.github.io/langgraph/llms-full.txt
as the LangGraph sources. As of this writing those URLs redirect to a stub
page: LangGraph's docs were merged into docs.langchain.com alongside
LangChain's (both now live under /oss/python/). We fetch the legacy URLs
anyway (for the record / transparency) and additionally fetch
https://docs.langchain.com/oss/python/llms-full.txt, which is the actual
current full-text corpus and already contains both the LangChain and the
LangGraph Python documentation (confirmed by grepping for
"Source: .../oss/python/langgraph/..." entries within it). That combined
file is what data_prep/build_index.py indexes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    # small sitemap/index files, kept for reference
    "langchain_llms.txt": "https://docs.langchain.com/llms.txt",
    "langgraph_llms.txt": "https://langchain-ai.github.io/langgraph/llms.txt",
    # primary corpus actually indexed by build_index.py
    "langchain_oss_python_llms_full.txt": "https://docs.langchain.com/oss/python/llms-full.txt",
}

TIMEOUT = 60


def download(name: str, url: str, force: bool) -> None:
    dest = RAW_DIR / name
    if dest.exists() and not force:
        print(f"skip  {name} (already exists, use --force to re-download)")
        return
    print(f"fetch {url}")
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")
    print(f"saved {dest} ({len(resp.text):,} bytes)")


def main() -> None:
    force = "--force" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        download(name, url, force)
    print("\nDone. Next step: python data_prep/build_index.py")


if __name__ == "__main__":
    main()
