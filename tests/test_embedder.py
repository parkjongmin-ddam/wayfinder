"""Embedding-provider seam: config resolution + get_embedder dispatch.

Offline: the ollama/openai embedders are stubbed in ``sys.modules`` so these
tests need no packages and no network.
"""

from __future__ import annotations

import sys
import types

import pytest

from langconnect_agent.config import Config
from langconnect_agent.retrievers import get_embedder


@pytest.fixture
def fake_embedders(monkeypatch):
    """Stub ``langchain_ollama.OllamaEmbeddings`` and ``langchain_openai.OpenAIEmbeddings``."""

    class _FakeOllamaEmbeddings:
        provider = "ollama"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeOpenAIEmbeddings:
        provider = "openai"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    om = types.ModuleType("langchain_ollama")
    om.OllamaEmbeddings = _FakeOllamaEmbeddings
    oa = types.ModuleType("langchain_openai")
    oa.OpenAIEmbeddings = _FakeOpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_ollama", om)
    monkeypatch.setitem(sys.modules, "langchain_openai", oa)


# ---- config: embedding provider resolution --------------------------------


def test_embedding_auto_follows_ollama_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    cfg = Config.from_env()

    assert cfg.embedding_provider == "ollama"
    assert cfg.embedding_model == "nomic-embed-text"


def test_embedding_auto_defaults_to_openai_for_hosted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    cfg = Config.from_env()

    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "text-embedding-3-small"


def test_embedding_provider_explicit_overrides_auto(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    cfg = Config.from_env()

    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "text-embedding-3-small"


# ---- get_embedder dispatch ------------------------------------------------


def test_get_embedder_ollama(fake_embedders):
    cfg = Config(
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://localhost:11434",
    )

    emb = get_embedder(cfg)

    assert emb.provider == "ollama"
    assert emb.kwargs["model"] == "nomic-embed-text"
    assert emb.kwargs["base_url"] == "http://localhost:11434"


def test_get_embedder_openai(fake_embedders):
    cfg = Config(embedding_provider="openai", embedding_model="text-embedding-3-small")

    emb = get_embedder(cfg)

    assert emb.provider == "openai"
    assert emb.kwargs["model"] == "text-embedding-3-small"
