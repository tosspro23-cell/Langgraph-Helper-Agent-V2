"""Offline retriever tests. No network, no LLM/API key required.

Run: pytest tests/ -v
"""

from langgraph_helper.retrievers import offline


def test_returns_ranked_results():
    results = offline.search("How do I add persistence to a LangGraph agent?", k=5)
    assert results, "expected at least one retrieved chunk"
    assert results == sorted(results, key=lambda c: c["score"], reverse=True)


def test_results_have_required_fields():
    results = offline.search("StateGraph checkpointer", k=3)
    for c in results:
        assert c["title"]
        assert c["url"].startswith("https://")
        assert c["text"]


def test_relevant_chunk_is_retrieved_for_known_topic():
    results = offline.search("add memory persistence checkpointer LangGraph", k=5)
    urls = [c["url"] for c in results]
    assert any("add-memory" in u or "persistence" in u for u in urls)
