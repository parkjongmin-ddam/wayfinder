"""Phase 4 evaluation gate (BUILD_SPEC §4 / Phase 4) — offline, mock data.

Verifies the differentiator metrics compute correctly over the fixed eval set:
routing accuracy, fallback firing/recovery, per-route faithfulness. No keys.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from langconnect_agent.evaluation import (
    EVAL_DATASET,
    MockFaithfulness,
    RagasFaithfulness,
    eval_faithfulness,
    eval_fallback_match,
    eval_routing_accuracy,
    evaluate,
    get_faithfulness,
    make_target,
    register_langsmith_dataset,
    run_langsmith_experiment,
)
from langconnect_agent.retrievers import Document


def test_routing_accuracy_is_perfect_on_labeled_set():
    report = evaluate()
    assert report.routing_accuracy == 1.0
    assert all(r.routed_correctly for r in report.results)


def test_fallback_fires_only_on_weak_grounding_cases():
    report = evaluate()

    fell_back = {r.case.query for r in report.results if r.fell_back}
    expected = {c.query for c in EVAL_DATASET if c.expect_fallback}
    assert fell_back == expected  # fires exactly on the expected cases

    # 2 of 7 cases expect fallback (rate is rounded to 4 dp in the report).
    assert sum(r.fell_back for r in report.results) == 2
    assert report.fallback_rate == pytest.approx(2 / 7, abs=1e-3)


def test_all_expected_fallbacks_recover_via_web():
    report = evaluate()
    assert report.recovery_rate == 1.0
    for r in report.results:
        if r.case.expect_fallback:
            assert r.recovered


def test_per_route_faithfulness_covers_all_routes():
    report = evaluate()
    routes = set(report.per_route_faithfulness)
    assert {"semantic", "keyword", "web"} <= routes
    # All grounded answers score strictly positive and within [0, 1].
    for route, score in report.per_route_faithfulness.items():
        assert 0.0 < score <= 1.0, route


def test_mock_faithfulness_is_deterministic_and_bounded():
    metric = MockFaithfulness()
    docs = [Document(page_content="vector similarity search over embeddings")]

    grounded = metric.score("vector similarity search", docs)
    ungrounded = metric.score("completely unrelated banana pancakes", docs)

    assert grounded == 1.0        # every answer token appears in context
    assert ungrounded == 0.0
    assert metric.score("", docs) == 0.0
    assert metric.score("anything", []) == 0.0


def test_report_summary_has_headline_metrics():
    summary = evaluate().summary()
    assert "routing accuracy" in summary
    assert "fallback firing rate" in summary
    assert "per-route faithfulness" in summary


def test_langsmith_dataset_registration_is_noop_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    msg = register_langsmith_dataset()
    assert "skip" in msg
    assert str(len(EVAL_DATASET)) in msg


# --- LangSmith experiment pieces (offline) --------------------------------
def test_make_target_returns_scoreable_outputs():
    target = make_target()
    out = target({"query": "How does semantic vector search rank documents?"})
    assert out["route"] == "semantic"
    assert out["fell_back"] is False
    assert 0.0 <= out["faithfulness"] <= 1.0
    assert isinstance(out["answer"], str)


def test_experiment_evaluators_score_correctly():
    run = SimpleNamespace(
        outputs={"route": "semantic", "fell_back": True, "faithfulness": 0.7}
    )
    ex_ok = SimpleNamespace(
        outputs={"expected_route": "semantic", "expect_fallback": True}
    )
    ex_bad = SimpleNamespace(
        outputs={"expected_route": "web", "expect_fallback": False}
    )

    assert eval_routing_accuracy(run, ex_ok)["score"] == 1.0
    assert eval_routing_accuracy(run, ex_bad)["score"] == 0.0
    assert eval_fallback_match(run, ex_ok)["score"] == 1.0
    assert eval_fallback_match(run, ex_bad)["score"] == 0.0
    assert eval_faithfulness(run, ex_ok)["score"] == 0.7


def test_run_experiment_is_noop_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    msg = run_langsmith_experiment()
    assert "skip" in msg


def test_get_faithfulness_selects_ragas_only_when_opted_in(monkeypatch):
    cfg = SimpleNamespace(llm_provider="mock", router_model="mock-router")

    monkeypatch.setenv("FAITHFULNESS", "ragas")
    assert isinstance(get_faithfulness(cfg), RagasFaithfulness)

    monkeypatch.setenv("FAITHFULNESS", "")
    assert isinstance(get_faithfulness(cfg), MockFaithfulness)


def test_ragas_faithfulness_requires_documents_and_answer():
    # Short-circuits (returns 0.0) before importing ragas when there's nothing
    # to score, so this stays offline.
    metric = RagasFaithfulness(llm=object())
    assert metric.score("", [], "q") == 0.0
    assert metric.score("some answer", [], "q") == 0.0


def test_ragas_judge_wires_to_local_ollama(monkeypatch):
    # RAGAS works with a local Ollama judge, not just hosted providers. Stub
    # langchain_ollama so the wiring is verified with no server (mirrors the
    # test_llm_ollama pattern).
    import sys
    import types

    class _FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.ModuleType("langchain_ollama")
    module.ChatOllama = _FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)

    cfg = SimpleNamespace(
        llm_provider="ollama",
        router_model="qwen2.5:3b",
        ollama_base_url="http://localhost:11434",
        router_temperature=0.0,
    )
    monkeypatch.setenv("FAITHFULNESS", "ragas")

    metric = get_faithfulness(cfg)

    assert isinstance(metric, RagasFaithfulness)
    # The judge is a real langchain ChatOllama (local), wired from config.
    assert type(metric.llm).__name__ == "_FakeChatOllama"
    assert metric.llm.kwargs["model"] == "qwen2.5:3b"
    assert metric.llm.kwargs["base_url"] == "http://localhost:11434"
