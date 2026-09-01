"""Online retrieval via live web search.

Backend selection:
  - Tavily is used automatically when TAVILY_API_KEY is set (higher quality,
    LLM-oriented search, free tier: 1,000 searches/month).
  - Otherwise falls back to DuckDuckGo via the `ddgs` package, which needs
    no API key and no signup at all.

Queries are biased towards the official LangChain/LangGraph docs and GitHub
so results stay relevant to the assignment's domain, with a second, broader
pass if that turns up too little.
"""

from __future__ import annotations

from ..config import settings
from ..state import SourceChunk

_SITE_BIAS = (
    "(site:docs.langchain.com OR site:langchain-ai.github.io OR "
    "site:github.com/langchain-ai)"
)


def _search_tavily(query: str, k: int) -> list[SourceChunk]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    resp = client.search(query=f"{query} LangGraph LangChain", max_results=k, search_depth="basic")
    return [
        SourceChunk(
            title=r.get("title", ""),
            url=r.get("url", ""),
            text=r.get("content", ""),
            score=float(r.get("score", 0.0)),
        )
        for r in resp.get("results", [])
    ]


def _search_ddgs(query: str, k: int) -> list[SourceChunk]:
    from ddgs import DDGS

    def _run(q: str) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(q, max_results=k))

    results = _run(f"{query} {_SITE_BIAS}")
    if len(results) < max(2, k // 2):
        results = _run(f"{query} LangGraph LangChain")

    return [
        SourceChunk(
            title=r.get("title", ""),
            url=r.get("href", r.get("url", "")),
            text=r.get("body", ""),
            score=0.0,
        )
        for r in results[:k]
    ]


def search(query: str, k: int = 5) -> tuple[list[SourceChunk], str]:
    """Returns (results, backend_name_used)."""
    if settings.tavily_api_key:
        try:
            return _search_tavily(query, k), "tavily"
        except Exception:
            pass  # fall through to ddgs
    return _search_ddgs(query, k), "ddgs"
