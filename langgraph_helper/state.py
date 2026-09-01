"""Shared graph state."""

from __future__ import annotations

from typing import Literal, TypedDict


class SourceChunk(TypedDict):
    title: str
    url: str
    text: str
    score: float


class AgentState(TypedDict, total=False):
    question: str
    mode: Literal["offline", "online"]

    # populated by the retrieval node
    context_chunks: list[SourceChunk]
    retrieval_backend: str  # "bm25" | "tavily" | "ddgs" | "none"
    retrieval_warning: str | None

    # populated by the generation node
    answer: str
    sources: list[str]
