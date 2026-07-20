"""Phase 2 routing gate (BUILD_SPEC §5.1, §5.4).

Offline: mock router + stub retrievers/searcher, no keys.

Gate assertions:
  1. Each of the 3 query types routes to its intended path (A/B/C).
  2. A router that emits off-schema output falls back to semantic (A).
  3. Route C (web) excerpts are isolated as untrusted data and cite URLs.
  4. Every run emits a one-line decision trace.
"""

from __future__ import annotations

import pytest

from langconnect_agent.graph import build_graph
from langconnect_agent.router import classify_query, parse_route
from langconnect_agent.state import AgentState


class _StubRouterLLM:
    """Router stub that always returns a fixed (possibly off-schema) string."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def invoke(self, prompt):  # noqa: ANN001 - matches the LLM .invoke contract
        return self._reply


@pytest.mark.parametrize(
    "query, expected_route",
    [
        ("How does vector similarity search work conceptually?", "semantic"),
        ('What does the "str_replace_based_edit_tool" error mean?', "keyword"),
        ("What are the latest LangGraph features released in 2026?", "web"),
    ],
)
def test_three_query_types_route_as_intended(query, expected_route):
    """The 3 canonical query shapes each reach their intended route."""
    result = build_graph().invoke({"query": query})
    assert result["route"] == expected_route


def test_offschema_router_output_falls_back_to_semantic():
    """A malformed router response defaults to semantic (BUILD_SPEC §5.1)."""
    graph = build_graph(router_llm=_StubRouterLLM("banana pancakes"))
    result = graph.invoke({"query": "anything at all"})

    assert result["route"] == "semantic"
    assert "off-schema" in result["router_rationale"]
    # The graph still completes and answers despite the bad router output.
    assert result["answer"].strip()


def test_web_route_isolates_excerpts_and_cites_sources():
    """Route C fences web excerpts as untrusted data and includes source URLs."""
    graph = build_graph(router_llm=_StubRouterLLM("web"))
    result = graph.invoke({"query": "current status of something"})

    assert result["route"] == "web"
    # Web documents carry source + url metadata for citation.
    docs = result["documents"]
    assert docs and all(d.metadata.get("source") == "web" for d in docs)
    assert all(d.metadata.get("url") for d in docs)
    # The mock answer echoes the isolation-prompt framing (untrusted DATA).
    assert "untrusted" in result["answer"].lower()
    assert "<web_excerpts>" in result["answer"]


def test_decision_trace_is_emitted():
    """Every run produces the one-line decision trace (BUILD_SPEC §5.4)."""
    result = build_graph().invoke({"query": "How does reranking improve recall?"})
    trace = result["trace"]

    assert "route=A(semantic)" in trace
    assert "grade=0.85" in trace  # in-corpus query graded sufficient
    assert "fallback=no" in trace


def test_parse_route_accepts_labels_letters_and_rejects_noise():
    """Unit-level guardrails for the schema-safe route parser."""
    assert parse_route("web") == "web"
    assert parse_route("Route: keyword") == "keyword"
    assert parse_route("A") == "semantic"
    assert parse_route("banana") is None
    assert parse_route("") is None
    assert parse_route("semantic or web") is None  # ambiguous -> caller defaults


def test_classify_query_heuristics():
    """The offline mock's classification matches the intended cues."""
    assert classify_query("why does embedding dimensionality matter") == "semantic"
    assert classify_query("the latest news on pgvector") == "web"
    assert classify_query('exact match for "RFC 822"') == "keyword"
