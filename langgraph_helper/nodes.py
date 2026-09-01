"""Graph node functions.

Each node takes the current AgentState and returns a partial state update
(LangGraph merges dict updates into state), keeping nodes small and testable
in isolation.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .llm import get_llm
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from .retrievers import offline as offline_retriever
from .retrievers import online as online_retriever
from .state import AgentState


def route_by_mode(state: AgentState) -> str:
    """Conditional-edge function: send the question down the offline or
    online retrieval branch based on state["mode"]."""
    return "online" if state.get("mode") == "online" else "offline"


def retrieve_offline(state: AgentState) -> dict:
    try:
        chunks = offline_retriever.search(state["question"], k=settings.retrieval_k)
        return {
            "context_chunks": chunks,
            "retrieval_backend": "bm25",
            "retrieval_warning": None if chunks else "No matching passages found in the local index.",
        }
    except FileNotFoundError as e:
        return {"context_chunks": [], "retrieval_backend": "none", "retrieval_warning": str(e)}


def retrieve_online(state: AgentState) -> dict:
    try:
        chunks, backend = online_retriever.search(state["question"], k=settings.retrieval_k)
        warning = None if chunks else "Web search returned no results."
        return {"context_chunks": chunks, "retrieval_backend": backend, "retrieval_warning": warning}
    except Exception as e:
        # Fall back to the offline index rather than failing outright.
        chunks = []
        try:
            chunks = offline_retriever.search(state["question"], k=settings.retrieval_k)
        except FileNotFoundError:
            pass
        return {
            "context_chunks": chunks,
            "retrieval_backend": "bm25" if chunks else "none",
            "retrieval_warning": f"Online search failed ({e}); fell back to the offline index.",
        }


def _extract_text(content) -> str:
    """Normalize a LangChain message's .content into plain text.

    Most chat models return a plain string. Some (e.g. newer Gemini models)
    return a list of content blocks (text, thought signatures, etc.) --
    concatenate just the text blocks in that case.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def generate_answer(state: AgentState) -> dict:
    llm = get_llm()
    context = format_context(state.get("context_chunks", []))
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_TEMPLATE.format(question=state["question"], context=context)),
    ]
    response = llm.invoke(messages)
    sources = [c["url"] for c in state.get("context_chunks", []) if c.get("url")]
    return {"answer": _extract_text(response.content), "sources": sources}
