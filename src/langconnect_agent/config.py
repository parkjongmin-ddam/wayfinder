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
        faithfulness_threshold: minimum post-answer faithfulness to accept an
            answer; below it the verify node regenerates once (verification
            harness). Calibrated LOW for the default lexical proxy
            (``MockFaithfulness``), which underestimates grounding of
            paraphrased answers — it should catch gross hallucination, not
            penalize paraphrase. Raise it when using RAGAS (``FAITHFULNESS=ragas``).
        max_verify_retries: max answer regenerations from the faithfulness gate
            (1-hop cap, mirrors ``max_fallbacks``).
        retriever_provider: routes A/B retriever — "auto" (real pgvector
            ``LangConnectRetriever`` if ``PGVECTOR_CONNINFO`` is set, else the
            offline ``StubRetriever``), "stub", or "langconnect".
        embedding_model: embedding model used for both ingestion and query
            (must match the model the corpus was ingested with).
    """

    llm_provider: str = field(default_factory=_default_provider)
    router_model: str = "claude-haiku-4-5"
    answer_model: str = "claude-opus-4-8"
    top_k: int = 5
    route_default: str = "semantic"
    grade_threshold: float = 0.5
    max_fallbacks: int = 1
    web_provider: str = "auto"  # auto | stub | tavily (route C searcher)
    faithfulness_threshold: float = 0.35  # low: lexical proxy underestimates
    max_verify_retries: int = 1  # 1-hop answer-regeneration cap
    retriever_provider: str = "auto"  # auto | stub | langconnect (routes A/B)
    embedding_model: str = "text-embedding-3-small"
    # Step 4 multi-agent orchestration.
    agent_mode: str = "single"  # single (graph.py) | multi (orchestrator.py)
    max_agent_steps: int = 4    # supervisor step cap (guarantees termination)
    rewrite_count: int = 2      # query reformulations the retrieval agent adds

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
            faithfulness_threshold=float(
                os.getenv("FAITHFULNESS_THRESHOLD", "0.35")
            ),
            max_verify_retries=int(os.getenv("MAX_VERIFY_RETRIES", "1")),
            retriever_provider=os.getenv("RETRIEVER_PROVIDER", "auto"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            agent_mode=os.getenv("AGENT_MODE", "single"),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "4")),
            rewrite_count=int(os.getenv("REWRITE_COUNT", "2")),
        )


def get_config() -> Config:
    """Convenience factory returning a Config populated from the environment."""
    return Config.from_env()
