"""Config env resolution (Config.from_env / get_config).

Complements test_llm_ollama (which covers the ollama/anthropic model + temperature
defaults). Here: the untested surface — plain defaults, "auto" embedding-provider
resolution, numeric/string env overrides, and get_config() delegation.
"""

from __future__ import annotations

import pytest

from langconnect_agent.config import Config, get_config

# Every env var Config.from_env reads, so a "defaults" test can start clean.
_CONFIG_ENV = (
    "LLM_PROVIDER", "ROUTER_MODEL", "ANSWER_MODEL", "OLLAMA_BASE_URL",
    "ROUTER_TEMPERATURE", "ANSWER_TEMPERATURE", "TOP_K", "ROUTE_DEFAULT",
    "GRADE_THRESHOLD", "MAX_FALLBACKS", "WEB_PROVIDER", "FAITHFULNESS_THRESHOLD",
    "MAX_VERIFY_RETRIES", "RETRIEVER_PROVIDER", "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL", "AGENT_MODE", "MAX_AGENT_STEPS", "REWRITE_COUNT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Drop every Config env var so from_env() sees pristine defaults."""
    for key in _CONFIG_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_defaults_when_env_absent(clean_env):
    cfg = Config.from_env()

    assert cfg.llm_provider == "mock"
    assert cfg.router_model == "claude-haiku-4-5"   # hosted fallback
    assert cfg.answer_model == "claude-opus-4-8"
    assert cfg.top_k == 5
    assert cfg.route_default == "semantic"
    assert cfg.grade_threshold == 0.5
    assert cfg.max_fallbacks == 1
    assert cfg.web_provider == "auto"
    assert cfg.retriever_provider == "auto"
    assert cfg.faithfulness_threshold == 0.35
    assert cfg.max_verify_retries == 1
    assert cfg.router_temperature == 0.0
    assert cfg.answer_temperature == 0.0
    assert cfg.agent_mode == "single"
    assert cfg.max_agent_steps == 4
    assert cfg.rewrite_count == 2
    # Non-ollama provider → hosted embeddings.
    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "text-embedding-3-small"


def test_embedding_auto_follows_ollama_provider(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")  # EMBEDDING_PROVIDER stays "auto"

    cfg = Config.from_env()

    assert cfg.embedding_provider == "ollama"
    assert cfg.embedding_model == "nomic-embed-text"


def test_embedding_auto_stays_hosted_for_non_ollama(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")

    cfg = Config.from_env()

    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "text-embedding-3-small"


def test_explicit_embedding_provider_overrides_auto(clean_env):
    # Explicit EMBEDDING_PROVIDER wins even when the LLM provider is local.
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("EMBEDDING_PROVIDER", "openai")

    cfg = Config.from_env()

    assert cfg.embedding_provider == "openai"
    assert cfg.embedding_model == "text-embedding-3-small"


def test_numeric_and_string_overrides_are_parsed(clean_env):
    clean_env.setenv("TOP_K", "8")
    clean_env.setenv("ROUTE_DEFAULT", "web")
    clean_env.setenv("GRADE_THRESHOLD", "0.7")
    clean_env.setenv("MAX_FALLBACKS", "2")
    clean_env.setenv("WEB_PROVIDER", "tavily")
    clean_env.setenv("RETRIEVER_PROVIDER", "langconnect")
    clean_env.setenv("FAITHFULNESS_THRESHOLD", "0.6")
    clean_env.setenv("MAX_VERIFY_RETRIES", "2")
    clean_env.setenv("AGENT_MODE", "multi")
    clean_env.setenv("MAX_AGENT_STEPS", "6")
    clean_env.setenv("REWRITE_COUNT", "3")

    cfg = Config.from_env()

    assert cfg.top_k == 8 and isinstance(cfg.top_k, int)
    assert cfg.route_default == "web"
    assert cfg.grade_threshold == 0.7 and isinstance(cfg.grade_threshold, float)
    assert cfg.max_fallbacks == 2
    assert cfg.web_provider == "tavily"
    assert cfg.retriever_provider == "langconnect"
    assert cfg.faithfulness_threshold == 0.6
    assert cfg.max_verify_retries == 2
    assert cfg.agent_mode == "multi"
    assert cfg.max_agent_steps == 6
    assert cfg.rewrite_count == 3


def test_get_config_delegates_to_from_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("TOP_K", "9")

    cfg = get_config()

    assert isinstance(cfg, Config)
    assert cfg.llm_provider == "ollama"
    assert cfg.top_k == 9
