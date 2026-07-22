"""LLMGrader: graded 1..5 grounding judgment + robust parsing.

Offline: a fake LLM returns canned judge replies, so no provider or network is
needed. Exercises the local-grader path (LLM_PROVIDER=ollama wires this same
LLMGrader to a local model via get_grader).
"""

from __future__ import annotations

from langconnect_agent.config import Config
from langconnect_agent.grading import LLMGrader, get_grader
from langconnect_agent.retrievers import Document


class _FakeLLM:
    """Return a fixed reply; record the prompt it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    def invoke(self, prompt):  # noqa: ANN001 - matches the .invoke contract
        self.last_prompt = prompt
        return self.reply


DOCS = [Document(page_content="Chunking splits documents into passages.")]


def test_high_rating_is_sufficient():
    grader = LLMGrader(_FakeLLM("5"), threshold=0.5)
    r = grader.grade("What is chunking?", DOCS)
    assert r.score == 1.0
    assert r.sufficient
    assert "5/5" in r.reason


def test_low_rating_is_insufficient():
    grader = LLMGrader(_FakeLLM("2"), threshold=0.5)
    r = grader.grade("Unrelated question?", DOCS)
    assert r.score == 0.25
    assert not r.sufficient


def test_midpoint_rating_meets_default_threshold():
    grader = LLMGrader(_FakeLLM("The rating is 3 out of 5."), threshold=0.5)
    r = grader.grade("Partially covered?", DOCS)
    assert r.score == 0.5
    assert r.sufficient  # 0.5 >= 0.5


def test_unparseable_reply_defaults_neutral_sufficient():
    """Runtime defense: no rating -> neutral, sufficient (verify backstops)."""
    grader = LLMGrader(_FakeLLM("I cannot decide."), threshold=0.5)
    r = grader.grade("Ambiguous?", DOCS)
    assert r.score == 0.5
    assert r.sufficient
    assert "unparseable" in r.reason


def test_no_documents_is_insufficient():
    grader = LLMGrader(_FakeLLM("5"), threshold=0.5)
    r = grader.grade("Anything?", [])
    assert r.score == 0.0
    assert not r.sufficient


def test_content_object_reply_is_coerced():
    """A langchain-style message (has .content) is handled, not just str."""

    class _Msg:
        content = "4"

    class _MsgLLM:
        def invoke(self, prompt):  # noqa: ANN001
            return _Msg()

    r = LLMGrader(_MsgLLM(), threshold=0.5).grade("q", DOCS)
    assert r.score == 0.75
    assert r.sufficient


def test_get_grader_ollama_returns_llm_grader(monkeypatch):
    """get_grader wires LLMGrader for a non-mock (local) provider."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    # Stub the router LLM factory so no ollama package/server is needed.
    monkeypatch.setattr(
        "langconnect_agent.llm.get_router_llm",
        lambda config: _FakeLLM("4"),
    )
    grader = get_grader(Config.from_env())
    assert isinstance(grader, LLMGrader)
    r = grader.grade("q", DOCS)
    assert r.score == 0.75
