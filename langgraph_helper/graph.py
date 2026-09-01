"""Graph assembly.

    START
      |
      v
  route_by_mode  --offline--> retrieve_offline --\
      \--online---> retrieve_online -------------+--> generate_answer --> END

`mode` lives in the graph state (not baked into the graph structure), so a
single compiled graph handles both offline and online questions -- the mode
is just another field on the input state. This mirrors how you'd extend the
agent with more branches later (e.g. a "hybrid" mode) without rebuilding
the graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_offline", nodes.retrieve_offline)
    graph.add_node("retrieve_online", nodes.retrieve_online)
    graph.add_node("generate_answer", nodes.generate_answer)

    graph.add_conditional_edges(
        START,
        nodes.route_by_mode,
        {"offline": "retrieve_offline", "online": "retrieve_online"},
    )
    graph.add_edge("retrieve_offline", "generate_answer")
    graph.add_edge("retrieve_online", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
