"""Local Ollama provider seam (LLM_PROVIDER=ollama).

Fully offline: the ollama wiring is exercised against a stub ``langchain_ollama``
module injected into ``sys.modules``, so these tests need neither the
``langchain-ollama`` package nor a running Ollama server.
"""

from __future__ import annotations

import sys
import types

import pytest

from langconnect_agent.config import Config
from langconnect_agent.llm import get_llm, get_router_llm


@pytest.fixture
def fake_ollama(monkeypatch):
    """Install a stub ``langchain_ollama.ChatOllama`` capturing its kwargs."""
    calls: list[dict] = []

    class _FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

        def invoke(self, prompt):  # pragma: no cover - not exercised here
            return f"[fake-ollama] {prompt}"

    module = types.ModuleType("langchain_ollama")
    module.ChatOllama = _FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)
    return calls


# ---- Config: provider-aware model defaults --------------------------------


def test_config_ollama_defaults_to_local_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("ANSWER_MODEL", raising=False)

    cfg = Config.from_env()

    assert cfg.llm_provider == "ollama"
    assert cfg.router_model == "qwen2.5:3b"
    assert cfg.answer_model == "llama3.1:8b"
    assert cfg.ollama_base_url == "http://localhost:11434"


def test_config_hosted_provider_keeps_claude_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("ANSWER_MODEL", raising=False)

    cfg = Config.from_env()

    assert cfg.router_model == "claude-haiku-4-5"
    assert cfg.answer_model == "claude-opus-4-8"


def test_config_explicit_model_overrides_win(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ANSWER_MODEL", "mistral:7b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")

    cfg = Config.from_env()

    assert cfg.answer_model == "mistral:7b"
    assert cfg.ollama_base_url == "http://gpu-box:11434"


# ---- llm factory: ollama wiring -------------------------------------------


def test_get_llm_ollama_uses_answer_model_and_base_url(fake_ollama):
    cfg = Config(
        llm_provider="ollama",
        answer_model="llama3.1:8b",
        ollama_base_url="http://localhost:11434",
    )

    client = get_llm(cfg)

    assert client.kwargs["model"] == "llama3.1:8b"
    assert client.kwargs["base_url"] == "http://localhost:11434"


def test_get_router_llm_ollama_uses_router_model(fake_ollama):
    cfg = Config(
        llm_provider="ollama",
        router_model="qwen2.5:3b",
        ollama_base_url="http://gpu-box:11434",
    )

    client = get_router_llm(cfg)

    assert client.kwargs["model"] == "qwen2.5:3b"
    assert client.kwargs["base_url"] == "http://gpu-box:11434"


def test_get_llm_ollama_explicit_overrides(fake_ollama):
    cfg = Config(llm_provider="ollama")

    client = get_llm(cfg, model="gemma2:9b")

    assert client.kwargs["model"] == "gemma2:9b"


def test_unknown_provider_message_lists_ollama():
    with pytest.raises(ValueError, match="ollama"):
        get_llm(provider="does-not-exist")
