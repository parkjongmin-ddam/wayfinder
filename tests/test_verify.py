"""Post-answer faithfulness verification harness (offline).

Exercises the ``verify`` node and the 1-hop answer-regeneration loop with an
injected fake metric, so no LLM or keys are needed.
"""

from __future__ import annotations

from types import SimpleNamespace

from langconnect_agent.config import get_config
from langconnect_agent.graph import build_graph
from langconnect_agent.nodes import verify
from langconnect_agent.retrievers import Document


class _FakeMetric:
    """Returns preset faithfulness scores in sequence; records call count."""

    def __init__(self, *scores):
        self.scores = list(scores)
        self.calls = 0

    def score(self, answer, documents, query=""):
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return s


class _RecordingLLM:
    """Mock LLM that records every prompt it is asked to answer."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return f"answer #{len(self.prompts)}"


def _cfg(threshold=0.35, retries=1):
    c = get_config()
    c.faithfulness_threshold = threshold
    c.max_verify_retries = retries
    return c


def _docs():
    return [Document(page_content="grounding context about vectors", metadata={})]


def test_verify_accepts_grounded_answer():
    out = verify(
        {"query": "q", "answer": "a", "documents": _docs()},
        metric=_FakeMetric(0.9),
        cfg=_cfg(),
    )
    assert out["needs_regen"] is False
    assert out["faithfulness"] == 0.9
    assert "grounded" in out["faithfulness_reason"]
    assert "regen_count" not in out  # no regeneration scheduled


def test_verify_flags_ungrounded_answer_for_regen():
    out = verify(
        {"query": "q", "answer": "a", "documents": _docs()},
        metric=_FakeMetric(0.1),
        cfg=_cfg(),
    )
    assert out["needs_regen"] is True
    assert out["regen_count"] == 1
    assert "ungrounded" in out["faithfulness_reason"]


def test_verify_respects_retry_budget():
    # Already regenerated once; even a low score must not loop again.
    out = verify(
        {"query": "q", "answer": "a", "documents": _docs(), "regen_count": 1},
        metric=_FakeMetric(0.1),
        cfg=_cfg(retries=1),
    )
    assert out["needs_regen"] is False
    assert "retry budget spent" in out["faithfulness_reason"]


def test_graph_regenerates_once_then_terminates():
    """Ungrounded-then-grounded metric drives exactly one regeneration."""
    llm = _RecordingLLM()
    metric = _FakeMetric(0.1, 0.9)  # first fail, second pass
    compiled = build_graph(llm=llm, faithfulness=metric)

    result = compiled.invoke({"query": "How does vector search rank documents?"})

    assert len(llm.prompts) == 2, "answer should be generated twice (1 regen)"
    # The regeneration prompt tightens grounding.
    assert "Use ONLY facts stated in the context" in llm.prompts[1]
    assert result["regen_count"] == 1
    assert result["faithfulness"] == 0.9
    assert result["answer"].strip()
    assert "regen x1" in result["trace"]
