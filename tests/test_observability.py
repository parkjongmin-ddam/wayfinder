"""LangSmith observability helper — offline-safe behavior.

These tests do not touch the LangSmith network; they verify the helper attaches
the right RunnableConfig and stays a transparent pass-through when tracing is
off (no API key needed to run the graph).
"""

from __future__ import annotations

from langconnect_agent.graph import build_graph
from langconnect_agent.observability import run_with_trace, tracing_enabled


class _SpyGraph:
    """Captures the config passed to invoke and echoes a minimal result."""

    def __init__(self):
        self.config = None

    def invoke(self, inputs, config=None):
        self.config = config
        return {"query": inputs["query"], "trace": "spy-trace"}


def test_tracing_disabled_without_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert tracing_enabled() is False


def test_tracing_requires_both_flag_and_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert tracing_enabled() is False

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    assert tracing_enabled() is True


def test_run_with_trace_attaches_tags_and_metadata():
    spy = _SpyGraph()
    run_with_trace(
        spy,
        {"query": "hi"},
        tags=["route-demo"],
        metadata={"experiment": "phase2"},
    )
    cfg = spy.config
    assert "langconnect-agent" in cfg["tags"]
    assert "route-demo" in cfg["tags"]
    assert cfg["metadata"]["app"] == "langconnect-agent"
    assert cfg["metadata"]["experiment"] == "phase2"


def test_run_with_trace_is_transparent_offline():
    """Runs a real compiled graph with tracing off and returns the result."""
    result = run_with_trace(
        build_graph(), {"query": "How does dense vector retrieval work?"}
    )
    assert result["route"] == "semantic"
    assert result["answer"].strip()
    assert result["trace"]
