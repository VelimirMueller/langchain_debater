"""Graph topology for the debate agent.

The moderator function is registered at two distinct graph positions
(moderator_open, moderator_distill) — this is the idiomatic LangGraph way
to reuse a Python callable at multiple flow positions.
"""

from langgraph.graph import END, START, StateGraph

from debate.nodes import (
    critic_node,
    judge_node,
    judge_router,
    moderator_node,
    proposer_node,
    setup_node,
)
from debate.state import DebateState


def build_graph():
    """Construct and compile the debate StateGraph."""
    graph = StateGraph(DebateState)

    graph.add_node("setup", setup_node)
    graph.add_node("moderator_open", moderator_node)
    graph.add_node("proposer", proposer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("moderator_distill", moderator_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "setup")
    graph.add_edge("setup", "moderator_open")
    graph.add_edge("moderator_open", "proposer")
    graph.add_edge("proposer", "critic")
    graph.add_edge("critic", "moderator_distill")
    graph.add_edge("moderator_distill", "judge")
    graph.add_conditional_edges(
        "judge",
        judge_router,
        {"continue": "proposer", "end": END},
    )

    return graph.compile()
