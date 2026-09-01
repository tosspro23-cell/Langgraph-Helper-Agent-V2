"""Graph structure smoke tests -- verify the graph compiles and routes
correctly without ever calling an LLM (no API key required)."""

from langgraph_helper.graph import build_graph
from langgraph_helper.nodes import route_by_mode


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_route_by_mode():
    assert route_by_mode({"mode": "offline"}) == "offline"
    assert route_by_mode({"mode": "online"}) == "online"
    assert route_by_mode({}) == "offline"  # default


def test_offline_retrieval_node_runs_without_llm():
    from langgraph_helper.nodes import retrieve_offline

    result = retrieve_offline({"question": "How do I use checkpointers?", "mode": "offline"})
    assert result["retrieval_backend"] == "bm25"
    assert result["context_chunks"]
