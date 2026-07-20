"""Route C web-searcher selection (offline — no network calls)."""

from __future__ import annotations

from types import SimpleNamespace

from langconnect_agent.web import (
    StubWebSearcher,
    TavilyWebSearcher,
    get_web_searcher,
)


def _cfg(web_provider="auto"):
    return SimpleNamespace(web_provider=web_provider)


def test_auto_uses_stub_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert isinstance(get_web_searcher(_cfg("auto")), StubWebSearcher)


def test_auto_uses_tavily_when_key_present(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
    # Constructor reads the key lazily at search time, so this makes no call.
    assert isinstance(get_web_searcher(_cfg("auto")), TavilyWebSearcher)


def test_explicit_providers_override_auto(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
    assert isinstance(get_web_searcher(_cfg("stub")), StubWebSearcher)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert isinstance(get_web_searcher(_cfg("tavily")), TavilyWebSearcher)


def test_stub_web_searcher_returns_web_sourced_docs():
    docs = StubWebSearcher().search("anything", k=3)
    assert len(docs) == 3
    assert all(d.metadata["source"] == "web" for d in docs)
    assert all(d.metadata["url"] for d in docs)
