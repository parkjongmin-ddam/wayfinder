"""Agent state schema.

The full field set (routing + fallback) is declared now for schema stability
across phases, even though Phase 1 only reads/writes ``query``, ``documents``,
and ``answer``.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between graph nodes.

    Phase 1 uses: query, documents, answer.
    Phase 2 adds: route, router_rationale, trace (routing + decision trace).
    Phase 3 adds: grade, grade_reason, and the fallback_* / needs_fallback loop
    bookkeeping.

    ``route`` records the router's decision and is never mutated by fallback —
    a web fallback is recorded in ``fallbacks_used`` instead, so routing
    accuracy stays measurable independent of fallback (Phase 4).
    """

    query: str
    route: str
    router_rationale: str
    documents: list[Any]
    grade: float
    grade_reason: str
    answer: str
    trace: str
    fallback_count: int
    fallbacks_used: list[str]
    needs_fallback: bool
    # Phase 2 verification harness: post-answer faithfulness gate + regen loop.
    faithfulness: float
    faithfulness_reason: str
    needs_regen: bool
    regen_count: int
    # Step 4 multi-agent orchestration (separate orchestrator graph).
    next_agent: str          # supervisor's chosen next sub-agent
    agent_steps: int         # supervisor step counter (termination cap)
    agent_log: list[str]     # ordered record of sub-agents invoked
    rewrites: list[str]      # query reformulations from the retrieval agent
    retrieval_sufficient: bool  # retrieval agent's self-check verdict
