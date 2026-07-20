"""Phase 3 grading: RAGAS-style grounding-sufficiency judgment.

The grade node asks: do the retrieved documents sufficiently ground an answer?
If not — and a fallback hop is still available — the graph falls back to the
web path (1-hop cap, BUILD_SPEC §5.1 / Phase 3).

Offline default is ``MockGrader``, a deterministic simulation of context
relevance: web excerpts count as fresh external grounding (sufficient), while
corpus (internal) grounding is judged sufficient only when the query falls
within the stub corpus's known scope. This keeps the fallback loop reproducible
with no network and no API key. ``LLMGrader`` is the seam for a real fast-model
judge (superseded by the RAGAS evaluation layer in Phase 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from langconnect_agent.retrievers import Document

# Topics the stub corpus is treated as "covering". A query with no overlap is
# judged out-of-scope, so internal grounding is graded insufficient and the
# graph falls back to the web path.
CORPUS_VOCAB: frozenset[str] = frozenset(
    {
        "embedding", "embeddings", "vector", "vectors", "similarity", "search",
        "retrieval", "retriever", "retrieve", "rag", "pgvector", "rerank",
        "reranking", "recall", "precision", "chunk", "chunking", "index",
        "indexing", "semantic", "keyword", "hybrid", "langgraph", "langchain",
        "agent", "agents", "router", "routing", "faithfulness", "ragas",
        "corpus", "document", "documents", "query", "relevance", "dense",
        "sparse", "bm25", "cosine", "knn",
    }
)

_SUFFICIENT_SCORE = 0.85
_INSUFFICIENT_SCORE = 0.25


@dataclass
class GradeResult:
    """Outcome of a grounding-sufficiency judgment."""

    score: float
    sufficient: bool
    reason: str


def _source(doc: Document) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("source", "")


@runtime_checkable
class Grader(Protocol):
    """Protocol every grader must satisfy."""

    def grade(self, query: str, documents: list[Document]) -> GradeResult:
        ...


class MockGrader:
    """Deterministic, offline grounding grader (RAGAS-style simulation)."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def grade(self, query: str, documents: list[Document]) -> GradeResult:
        if not documents:
            return GradeResult(0.0, False, "no documents retrieved")

        if any(_source(d) == "web" for d in documents):
            return GradeResult(
                _SUFFICIENT_SCORE, True, "web excerpts provide fresh grounding"
            )

        tokens = set(re.findall(r"[a-z]+", (query or "").lower()))
        covered = bool(tokens & CORPUS_VOCAB)
        score = _SUFFICIENT_SCORE if covered else _INSUFFICIENT_SCORE
        sufficient = score >= self.threshold
        reason = (
            "query within corpus scope"
            if covered
            else "query outside corpus scope"
        )
        return GradeResult(score, sufficient, reason)


class LLMGrader:
    """Seam for a real fast-model grounding judge (Phase 4 → RAGAS)."""

    _PROMPT = (
        "You judge whether the CONTEXT sufficiently grounds an answer to the "
        "QUESTION. Reply with ONLY one word: SUFFICIENT or INSUFFICIENT.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nVerdict:"
    )

    def __init__(self, llm: Any, threshold: float = 0.5) -> None:
        self.llm = llm
        self.threshold = threshold

    def grade(self, query: str, documents: list[Document]) -> GradeResult:
        if not documents:
            return GradeResult(0.0, False, "no documents retrieved")
        context = "\n".join(
            getattr(d, "page_content", str(d)) for d in documents
        )
        raw = self.llm.invoke(self._PROMPT.format(context=context, query=query))
        text = str(getattr(raw, "content", raw)).strip().lower()
        sufficient = "insufficient" not in text and "sufficient" in text
        score = _SUFFICIENT_SCORE if sufficient else _INSUFFICIENT_SCORE
        return GradeResult(score, sufficient, f"llm judge: {text[:40]}")


def get_grader(
    config: Any = None,
    *,
    provider: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Grader:
    """Return a grader. Offline "mock" → MockGrader; real providers → LLMGrader."""
    if provider is None:
        provider = getattr(config, "llm_provider", None) or "mock"
    provider = provider.lower()
    thr = (
        threshold
        if threshold is not None
        else getattr(config, "grade_threshold", 0.5)
    )

    if provider == "mock":
        return MockGrader(threshold=thr)

    from langconnect_agent.llm import get_router_llm  # local import avoids cycle

    return LLMGrader(get_router_llm(config), threshold=thr)
