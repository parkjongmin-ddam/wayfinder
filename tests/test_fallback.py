"""Phase 3 fallback-loop gate (BUILD_SPEC §5.1 / Phase 3).

Offline: mock router + stub sources + mock grader, no keys.

Gate: an internal query with weak grounding falls back to web (1-hop) and
recovers with a non-empty, web-grounded answer.
"""

from __future__ import annotations

from langconnect_agent.grading import GradeResult, MockGrader
from langconnect_agent.graph import build_graph


class _AlwaysInsufficientGrader:
    """Grader that never judges grounding sufficient (forces fallback)."""

    def grade(self, query, documents):  # noqa: ANN001 - matches Grader protocol
        return GradeResult(0.0, False, "forced insufficient")


class _StubRouterLLM:
    """Router stub that always returns a fixed route label."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def invoke(self, prompt):  # noqa: ANN001 - matches the LLM .invoke contract
        return self._reply


def test_in_corpus_query_does_not_fall_back():
    """A query within corpus scope is graded sufficient — no fallback."""
    result = build_graph().invoke(
        {"query": "How does semantic vector retrieval rank documents?"}
    )
    assert result["route"] == "semantic"
    assert not result.get("needs_fallback")
    assert not result.get("fallbacks_used")
    assert result["grade"] >= 0.5


def test_weak_internal_grounding_falls_back_to_web_and_recovers():
    """Out-of-corpus internal query → web fallback → recovered answer."""
    result = build_graph().invoke({"query": "What is the capital of France?"})

    # Router kept its decision (internal); the fallback is recorded separately.
    assert result["route"] == "semantic"
    assert result["fallbacks_used"] == ["web"]
    assert result["fallback_count"] == 1

    # Recovered: the final documents are web-sourced and the answer is grounded
    # in them via the injection-isolation prompt.
    docs = result["documents"]
    assert docs and all(d.metadata.get("source") == "web" for d in docs)
    assert result["answer"].strip()
    assert "<web_excerpts>" in result["answer"]


def test_web_route_never_falls_back():
    """A query the router sends to web has no further path to fall back to."""
    result = build_graph(router_llm=_StubRouterLLM("web")).invoke(
        {"query": "latest happenings"}
    )
    assert result["route"] == "web"
    assert not result.get("fallbacks_used")


def test_fallback_is_capped_at_one_hop():
    """Even with a grader that always fails, fallback fires at most once."""
    result = build_graph(grader=_AlwaysInsufficientGrader()).invoke(
        {"query": "How does dense retrieval work?"}
    )
    # One hop taken, then the graph proceeds to answer (no infinite loop).
    assert result["fallbacks_used"] == ["web"]
    assert result["fallback_count"] == 1
    assert result["answer"].strip()


def test_mock_grader_scope_judgment():
    """MockGrader: in-scope corpus query sufficient, out-of-scope not."""
    from langconnect_agent.retrievers import StubRetriever

    grader = MockGrader(threshold=0.5)
    in_scope = StubRetriever().search("vector similarity reranking", k=3)
    out_scope = StubRetriever().search("weather in Paris today", k=3)

    assert grader.grade("vector similarity reranking", in_scope).sufficient
    assert not grader.grade("weather in Paris tomorrow", out_scope).sufficient
