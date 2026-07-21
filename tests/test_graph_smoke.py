"""Smoke tests — graph builds and runs offline.

Run fully offline (stub retriever/searcher + mock router/answer LLMs): no
database, no network, no API keys.
"""

from __future__ import annotations

from langgraph.graph import END, START

from langconnect_agent.graph import build_graph


def test_graph_topology_route_retrieve_grade_answer():
    """Graph compiles with the Phase 3 routing + fallback topology."""
    compiled = build_graph()

    drawable = compiled.get_graph()

    node_ids = set(drawable.nodes)
    for expected in (
        "route", "retrieve", "web_search", "grade", "answer", "verify"
    ):
        assert expected in node_ids, f"missing node {expected!r}"

    edges = {(e.source, e.target) for e in drawable.edges}
    # START enters the router; both retrieval paths converge on grade; grade
    # feeds answer (and can loop back to web_search for the 1-hop fallback);
    # answer feeds the verify gate, which ends (or loops back to answer once).
    assert (START, "route") in edges
    assert ("retrieve", "grade") in edges
    assert ("web_search", "grade") in edges
    assert ("answer", "verify") in edges
    assert ("verify", END) in edges
    assert ("verify", "answer") in edges  # 1-hop regeneration loop


def test_graph_invoke_returns_non_empty_answer():
    """Invoking the compiled graph on a sample query returns a non-empty answer."""
    compiled = build_graph()

    result = compiled.invoke({"query": "What is vector similarity search?"})

    # Retrieval populated documents.
    assert result.get("documents"), "expected retrieved documents in state"

    # The answer node wrote a non-empty answer string (mock LLM, offline).
    answer_text = result.get("answer")
    assert isinstance(answer_text, str)
    assert answer_text.strip(), "expected a non-empty answer string"
