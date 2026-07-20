"""Configuration for the langconnect_agent skeleton.

Dependency-light: a plain dataclass populated from environment variables via
``os.getenv``. No pydantic / pydantic-settings required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock")


@dataclass
class Config:
    """Runtime settings for the agent.

    Attributes:
        llm_provider: which LLM backend to use ("mock", "anthropic", "openai").
            Defaults to the ``LLM_PROVIDER`` env var, or "mock".
        router_model: fast model name used for routing decisions (Phase 2+).
        answer_model: strong model name used for answer synthesis.
        top_k: number of documents to retrieve.
        route_default: safe fallback route when the router output is malformed
            or off-schema (Phase 2 runtime defense, BUILD_SPEC §5.1).
        grade_threshold: minimum grounding-sufficiency score to skip fallback
            (Phase 3).
        max_fallbacks: maximum number of fallback hops (Phase 3, 1-hop cap).
        web_provider: route C searcher — "auto" (Tavily if TAVILY_API_KEY set,
            else stub), "stub", or "tavily".
    """

    llm_provider: str = field(default_factory=_default_provider)
    router_model: str = "claude-haiku-4-5"
    answer_model: str = "claude-opus-4-8"
    top_k: int = 5
    route_default: str = "semantic"
    grade_threshold: float = 0.5
    max_fallbacks: int = 1
    web_provider: str = "auto"  # auto | stub | tavily (route C searcher)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config, reading overridable values from the environment."""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock"),
            router_model=os.getenv("ROUTER_MODEL", "claude-haiku-4-5"),
            answer_model=os.getenv("ANSWER_MODEL", "claude-opus-4-8"),
            top_k=int(os.getenv("TOP_K", "5")),
            route_default=os.getenv("ROUTE_DEFAULT", "semantic"),
            grade_threshold=float(os.getenv("GRADE_THRESHOLD", "0.5")),
            max_fallbacks=int(os.getenv("MAX_FALLBACKS", "1")),
            web_provider=os.getenv("WEB_PROVIDER", "auto"),
        )


def get_config() -> Config:
    """Convenience factory returning a Config populated from the environment."""
    return Config.from_env()
