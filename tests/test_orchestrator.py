"""Multi-agent orchestrator (Step 4) — offline, mock providers only."""

from __future__ import annotations

from langconnect_agent.agents import (
    MockRewriteLLM,
    _merge_docs,
    retrieval_agent,
)
from langconnect_agent.orchestrator import build_orchestrator
from langconnect_agent.retrievers import Document
from langconnect_agent.supervisor import (
    RETRIEVAL,
    SYNTHESIS,
    WEB,
    _decide,
    supervisor,
)


# --- helpers --------------------------------------------------------------
def _doc(i, score, source="langconnect"):
    return Document(page_content=f"doc {i}", metadata={"id": i, "score": score,
                                                       "source": source})


def test_merge_docs_dedups_and_keeps_top_k_by_score():
    a = [_doc(1, 0.4), _doc(2, 0.9)]
    b = [_doc(2, 0.5), _doc(3, 0.7)]  # id 2 duplicated (keep higher 0.9)
    merged = _merge_docs(a, b, top_k=2)
    ids = [d.metadata["id"] for d in merged]
    assert ids == [2, 3]  # sorted by score desc, deduped, top-2


def test_mock_rewrite_llm_is_deterministic():
    llm = MockRewriteLLM(n=2)
    out1 = llm.invoke("Original query: what is pgvector\nRewrites:")
    out2 = llm.invoke("Original query: what is pgvector\nRewrites:")
    assert out1 == out2
    assert len(out1.splitlines()) == 2


# --- supervisor policy ----------------------------------------------------
def test_supervisor_policy_paths():
    # nothing gathered: semantic -> retrieval, web -> web
    assert _decide({"route": "semantic", "agent_log": []}) == RETRIEVAL
    assert _decide({"route": "web", "agent_log": []}) == WEB
    # after retrieval: sufficient -> synthesis, weak -> web
    assert _decide(
        {"agent_log": [RETRIEVAL], "retrieval_sufficient": True}
    ) == SYNTHESIS
    assert _decide(
        {"agent_log": [RETRIEVAL], "retrieval_sufficient": False}
    ) == WEB
    # after web -> synthesis
    assert _decide({"agent_log": [RETRIEVAL, WEB]}) == SYNTHESIS


def test_supervisor_step_cap_forces_synthesis():
    from langconnect_agent.config import get_config

    cfg = get_config()
    cfg.max_agent_steps = 1
    out = supervisor({"agent_log": [], "route": "semantic", "agent_steps": 1}, cfg=cfg)
    assert out["next_agent"] == SYNTHESIS  # budget spent -> forced synthesis


# --- retrieval agent (multi-query + self-check) ---------------------------
def test_retrieval_agent_multi_query_and_self_check():
    out = retrieval_agent({"query": "How are documents ranked by relevance?"})
    # rewrites were added (mock rewriter), docs merged/deduped, self-check set.
    assert len(out["rewrites"]) == 2
    assert out["documents"]
    assert isinstance(out["retrieval_sufficient"], bool)
    assert out["agent_log"] == ["retrieval_agent"]


# --- end-to-end orchestration (offline) -----------------------------------
def test_orchestrator_semantic_sufficient_path():
    app = build_orchestrator()
    # "documents" is in the MockGrader CORPUS_VOCAB -> retrieval judged sufficient
    state = app.invoke({"query": "How does ranking of documents work?"})
    assert state["agent_log"] == ["retrieval_agent", "synthesis_agent"]
    assert state["route"] == "semantic"
    assert state["answer"].strip()
    assert "agents=retrieval_agent->synthesis_agent" in state["trace"]


def test_orchestrator_augments_with_web_when_retrieval_weak():
    app = build_orchestrator()
    # out-of-corpus semantic query -> retrieval self-check weak -> web augment
    state = app.invoke({"query": "How do I bake sourdough bread at home?"})
    assert state["agent_log"] == [
        "retrieval_agent", "web_agent", "synthesis_agent"
    ]
    assert state["answer"].strip()


def test_orchestrator_web_route_skips_retrieval():
    app = build_orchestrator()
    state = app.invoke({"query": "What are the latest releases in 2026?"})
    assert state["route"] == "web"
    assert state["agent_log"] == ["web_agent", "synthesis_agent"]
    assert state["answer"].strip()
