"""Specialized sub-agents for the multi-agent orchestrator (Step 4, B-lite).

Three sub-agents, each a single node with its own logic, coordinated by the
supervisor (see ``supervisor.py``) and assembled in ``orchestrator.py``:

- ``retrieval_agent`` — rewrites the query into a few reformulations, runs a
  **multi-query** vector search, merges + dedups the hits, and self-checks
  grounding sufficiency (so the supervisor can decide whether to augment with
  the web). This is genuinely more than the single-graph ``retrieve`` node.
- ``web_agent`` — route-C web search with prompt-injection isolation.
- ``synthesis_agent`` — the only strong-model (Opus) step: synthesizes a final
  answer over the **accumulated** internal + web context and verifies its
  faithfulness, regenerating once (stricter prompt) if it is ungrounded.

Everything runs offline with mocks (``MockRewriteLLM`` + the stub retriever /
searcher / grader), so the orchestrator is testable with no keys.
"""

from __future__ import annotations

from typing import Any, Optional

from langconnect_agent.config import Config, get_config
from langconnect_agent.grading import Grader, get_grader
from langconnect_agent.llm import get_llm, get_router_llm
from langconnect_agent.nodes import (
    _STRICT_ADDENDUM,
    _build_answer_prompt,
    _build_web_answer_prompt,
    _format_context,
    _format_web_context,
    _is_web,
)
from langconnect_agent.retrievers import Document, Retriever, get_retriever
from langconnect_agent.state import AgentState
from langconnect_agent.web import WebSearcher, get_web_searcher

# ---------------------------------------------------------------------------
# Query rewriting (retrieval agent) — LLM seam + deterministic offline mock.
# ---------------------------------------------------------------------------
_REWRITE_MARKER = "Original query:"


def build_rewrite_prompt(query: str, n: int) -> str:
    """Prompt asking a fast model for ``n`` retrieval reformulations."""
    return (
        f"Rewrite the query below into {n} alternative search queries that "
        "capture the same information need with different wording, to improve "
        "recall in vector search. Respond with one rewrite per line, no "
        "numbering.\n\n"
        f"{_REWRITE_MARKER} {query}\n"
        "Rewrites:"
    )


def parse_rewrites(text: Any, n: int) -> list[str]:
    """Parse an LLM rewrite response into up to ``n`` clean query strings."""
    raw = getattr(text, "content", text)
    lines = [ln.strip(" -\t") for ln in str(raw or "").splitlines()]
    return [ln for ln in lines if ln][:n]


def mock_rewrites(query: str, n: int) -> list[str]:
    """Deterministic offline reformulations (no LLM)."""
    base = (query or "").strip().rstrip("?.!").strip()
    templates = [
        f"What is meant by {base}",
        f"Explain {base} in detail",
        f"Definition and mechanism of {base}",
    ]
    return templates[:n]


class MockRewriteLLM:
    """Deterministic, offline query-rewrite LLM (no network, no key)."""

    def __init__(self, model: str = "mock-rewrite", n: int = 2, **kwargs: Any):
        self.model = model
        self.n = n
        self.options = kwargs

    def invoke(self, prompt: Any) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        query = text
        if _REWRITE_MARKER in text:
            query = text.split(_REWRITE_MARKER, 1)[1].split("\n", 1)[0].strip()
        return "\n".join(mock_rewrites(query, self.n))


def get_rewriter(config: Any = None) -> Any:
    """Return the query-rewrite LLM (fast model / mock), mirroring get_llm."""
    provider = (getattr(config, "llm_provider", None) or "mock").lower()
    if provider == "mock":
        n = int(getattr(config, "rewrite_count", 2) or 2)
        return MockRewriteLLM(n=n)
    # Real providers reuse the fast router model (Haiku) for cheap rewriting.
    return get_router_llm(config)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _doc_id(doc: Document) -> Any:
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("id", getattr(doc, "page_content", id(doc)))


def _merge_docs(*groups: list[Document], top_k: int) -> list[Document]:
    """Union documents across groups, dedup by id, keep the best ``top_k`` by score."""
    seen: dict[Any, Document] = {}
    for group in groups:
        for doc in group or []:
            key = _doc_id(doc)
            prev = seen.get(key)
            if prev is None:
                seen[key] = doc
                continue
            # keep the higher-scored duplicate
            ps = (getattr(prev, "metadata", {}) or {}).get("score") or 0
            cs = (getattr(doc, "metadata", {}) or {}).get("score") or 0
            if cs > ps:
                seen[key] = doc
    merged = list(seen.values())
    merged.sort(
        key=lambda d: (getattr(d, "metadata", {}) or {}).get("score") or 0,
        reverse=True,
    )
    return merged[:top_k]


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------
def retrieval_agent(
    state: AgentState,
    *,
    retriever: Optional[Retriever] = None,
    rewriter: Any = None,
    grader: Optional[Grader] = None,
    cfg: Optional[Config] = None,
) -> dict[str, Any]:
    """Multi-query retrieval sub-agent with a grounding self-check.

    Rewrites the query, retrieves for the original + each rewrite, merges and
    dedups, then grades sufficiency so the supervisor can decide whether to
    augment with a web search. Accumulates into ``documents``.
    """
    cfg = cfg or get_config()
    retriever = retriever or get_retriever(cfg)
    rewriter = rewriter if rewriter is not None else get_rewriter(cfg)
    grader = grader or get_grader(cfg)

    query = state.get("query", "") or ""
    variants = parse_rewrites(
        rewriter.invoke(build_rewrite_prompt(query, cfg.rewrite_count)),
        cfg.rewrite_count,
    )

    groups = [retriever.search(query, k=cfg.top_k, route="semantic")]
    for v in variants:
        groups.append(retriever.search(v, k=cfg.top_k, route="semantic"))
    prior = state.get("documents", []) or []
    docs = _merge_docs(prior, *groups, top_k=cfg.top_k)

    verdict = grader.grade(query, docs)
    log = list(state.get("agent_log", []) or []) + ["retrieval_agent"]
    return {
        "documents": docs,
        "rewrites": variants,
        "retrieval_sufficient": bool(verdict.sufficient),
        "grade": verdict.score,
        "grade_reason": verdict.reason,
        "agent_log": log,
    }


def web_agent(
    state: AgentState,
    *,
    searcher: Optional[WebSearcher] = None,
    cfg: Optional[Config] = None,
) -> dict[str, Any]:
    """Web-search sub-agent (route C). Accumulates web hits into ``documents``."""
    cfg = cfg or get_config()
    searcher = searcher or get_web_searcher(cfg)

    query = state.get("query", "") or ""
    hits = searcher.search(query, k=cfg.top_k)
    prior = state.get("documents", []) or []
    docs = _merge_docs(prior, hits, top_k=cfg.top_k + len(prior))
    log = list(state.get("agent_log", []) or []) + ["web_agent"]
    return {"documents": docs, "agent_log": log}


def synthesis_agent(
    state: AgentState,
    *,
    llm: Any = None,
    faithfulness: Any = None,
    cfg: Optional[Config] = None,
) -> dict[str, Any]:
    """Final synthesis over accumulated context + faithfulness verification.

    The only strong-model (Opus) step. Builds a grounded answer from ALL
    gathered documents (web excerpts kept under injection isolation), scores
    faithfulness, and regenerates once with a stricter prompt if the answer is
    ungrounded (1-hop cap). Writes the trace.
    """
    from langconnect_agent.trace import build_trace

    cfg = cfg or get_config()
    llm = llm if llm is not None else get_llm(cfg)
    if faithfulness is None:
        from langconnect_agent.evaluation import get_faithfulness

        faithfulness = get_faithfulness(cfg)

    query = state.get("query", "") or ""
    documents = state.get("documents", []) or []
    has_web = _is_web(documents)

    def _synthesize(strict: bool) -> str:
        if has_web:
            prompt = _build_web_answer_prompt(
                query, _format_web_context(documents), strict=strict
            )
        else:
            prompt = _build_answer_prompt(
                query, _format_context(documents), strict=strict
            )
        result = llm.invoke(prompt)
        return str(getattr(result, "content", result))

    answer = _synthesize(strict=False)
    score = float(faithfulness.score(answer, documents, query))
    regens = 0
    if score < cfg.faithfulness_threshold and cfg.max_verify_retries > 0:
        answer = _synthesize(strict=True)
        score = float(faithfulness.score(answer, documents, query))
        regens = 1

    reason = (
        f"grounded ({score:.2f} >= {cfg.faithfulness_threshold:.2f})"
        if score >= cfg.faithfulness_threshold
        else f"ungrounded ({score:.2f} < {cfg.faithfulness_threshold:.2f})"
    )
    log = list(state.get("agent_log", []) or []) + ["synthesis_agent"]
    partial = {
        "answer": answer,
        "faithfulness": round(score, 4),
        "faithfulness_reason": reason,
        "regen_count": regens,
        "agent_log": log,
    }
    partial["trace"] = build_trace({**state, **partial})
    return partial
