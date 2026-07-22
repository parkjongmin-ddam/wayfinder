"""Provider-agnostic LLM factory.

``get_llm`` returns an object exposing ``.invoke(prompt) -> str``-like output.
The "mock" provider is fully deterministic and requires no network and no API
keys. The "anthropic", "openai", and "ollama" providers lazily import their
langchain packages only when selected, so those deps stay optional. "ollama"
runs a model locally against an Ollama server (no API key, offline-capable).
"""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_base_url(config: Any) -> str:
    """Resolve the Ollama server URL from config, then env, then default."""
    return (
        getattr(config, "ollama_base_url", None)
        or os.getenv("OLLAMA_BASE_URL")
        or _DEFAULT_OLLAMA_BASE_URL
    )


class MockLLM:
    """Deterministic, offline LLM. No network, no API key required."""

    def __init__(self, model: str = "mock", **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    def invoke(self, prompt: Any) -> str:
        """Return a deterministic canned answer derived from the prompt."""
        text = prompt if isinstance(prompt, str) else str(prompt)
        return f"[mock:{self.model}] Based on the provided context: {text}"


def get_llm(
    config: Any = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Return an LLM client with an ``.invoke(prompt)`` interface.

    Args:
        config: optional Config-like object. When provided, its ``llm_provider``
            and ``answer_model`` supply defaults for ``provider`` / ``model``.
        provider: override provider ("mock", "anthropic", "openai").
        model: override model name.
        **kwargs: forwarded to the underlying client constructor.

    The "mock" provider works with zero configuration. "anthropic" and "openai"
    import their langchain packages lazily, only when selected.
    """
    if provider is None:
        provider = getattr(config, "llm_provider", None) or "mock"
    provider = provider.lower()

    if model is None and config is not None:
        model = getattr(config, "answer_model", None)

    temperature = kwargs.pop("temperature", None)

    if provider == "mock":
        return MockLLM(model=model or "mock", **kwargs)

    if temperature is None:
        temperature = getattr(config, "answer_temperature", 0.0)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # lazy import

        return ChatAnthropic(
            model=model or "claude-opus-4-8", temperature=temperature, **kwargs
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # lazy import

        return ChatOpenAI(
            model=model or "gpt-4o-mini", temperature=temperature, **kwargs
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # lazy import

        return ChatOllama(
            model=model or "llama3.1:8b",
            base_url=_ollama_base_url(config),
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(
        f"Unknown llm provider {provider!r}. "
        "Expected one of: 'mock', 'anthropic', 'openai', 'ollama'."
    )


def get_router_llm(
    config: Any = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Return the fast routing LLM with an ``.invoke(prompt)`` interface.

    Parallels ``get_llm`` but defaults the model to ``config.router_model`` (a
    fast model, e.g. Claude Haiku). The "mock" provider returns a deterministic
    offline ``MockRouterLLM`` so Phase 2 routing runs with zero configuration.
    """
    from langconnect_agent.router import MockRouterLLM  # local import avoids cycle

    if provider is None:
        provider = getattr(config, "llm_provider", None) or "mock"
    provider = provider.lower()

    if model is None and config is not None:
        model = getattr(config, "router_model", None)

    temperature = kwargs.pop("temperature", None)

    if provider == "mock":
        return MockRouterLLM(model=model or "mock-router", **kwargs)

    if temperature is None:
        temperature = getattr(config, "router_temperature", 0.0)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # lazy import

        return ChatAnthropic(
            model=model or "claude-haiku-4-5", temperature=temperature, **kwargs
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # lazy import

        return ChatOpenAI(
            model=model or "gpt-4o-mini", temperature=temperature, **kwargs
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # lazy import

        return ChatOllama(
            model=model or "qwen2.5:3b",
            base_url=_ollama_base_url(config),
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(
        f"Unknown llm provider {provider!r}. "
        "Expected one of: 'mock', 'anthropic', 'openai', 'ollama'."
    )
