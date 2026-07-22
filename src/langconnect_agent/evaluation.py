"""Phase 4 evaluation layer (BUILD_SPEC §4 differentiator, Phase 4 gate).

The point of the project is not "we built routing" but "we measured it". This
module runs the graph over a fixed, route-typed eval set and computes the three
differentiator metrics offline (no keys):

* **routing accuracy** — did the router pick the intended route?
* **fallback firing rate** — how often did grounding fall back to web? (and,
  for cases where the corpus is expected to be weak, did it **recover**?)
* **per-route faithfulness** — is the answer grounded in the retrieved context?

Faithfulness here is a deterministic lexical proxy (``MockFaithfulness``) so the
harness is provable with mock data. ``RagasFaithfulness`` is the real RAGAS
metric (needs only an LLM judge — runs locally on Ollama or on a hosted
provider); ``get_faithfulness`` picks the mock offline and the RAGAS metric when
``FAITHFULNESS=ragas`` with a real provider. The same eval set is meant
to be registered as a LangSmith dataset once a key is available
(``register_langsmith_dataset``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from langconnect_agent.retrievers import Document


# --- fixed eval set -------------------------------------------------------
@dataclass(frozen=True)
class EvalCase:
    """A labeled evaluation query."""

    query: str
    expected_route: str          # semantic | keyword | web
    expect_fallback: bool        # should internal grounding be weak -> web?
    note: str = ""


# Covers all 3 route types + weak-grounding fallback/recovery cases.
EVAL_DATASET: list[EvalCase] = [
    EvalCase(
        "How does semantic vector search rank documents?",
        "semantic", False, "A: conceptual, in-corpus",
    ),
    EvalCase(
        "Why does reranking improve recall in retrieval?",
        "semantic", False, "A: conceptual, in-corpus",
    ),
    EvalCase(
        'What is the exact meaning of the "pgvector" operator?',
        "keyword", False, "B: exact term, in-corpus",
    ),
    EvalCase(
        'What does the "str_replace_based_edit_tool" error mean?',
        "keyword", True, "B: exact term, out-of-corpus -> web fallback",
    ),
    EvalCase(
        "What are the latest LangGraph releases in 2026?",
        "web", False, "C: time-sensitive",
    ),
    EvalCase(
        "current news about vector databases",
        "web", False, "C: time-sensitive",
    ),
    EvalCase(
        "What is the capital of France?",
        "semantic", True, "A: out-of-corpus -> web fallback",
    ),
]


# --- faithfulness metric --------------------------------------------------
def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


@runtime_checkable
class FaithfulnessMetric(Protocol):
    """Score how well an answer is grounded in its retrieved context (0..1)."""

    def score(
        self, answer: str, documents: list[Document], query: str = ""
    ) -> float:
        ...


class MockFaithfulness:
    """Deterministic offline faithfulness proxy.

    Fraction of the answer's content tokens that also appear in the retrieved
    context — a lexical stand-in for RAGAS faithfulness (which checks whether
    the answer's claims are supported by the context). ``query`` is accepted for
    interface parity but unused.
    """

    def score(
        self, answer: str, documents: list[Document], query: str = ""
    ) -> float:
        ans = _tokenize(answer)
        if not ans:
            return 0.0
        ctx: set[str] = set()
        for d in documents:
            ctx |= _tokenize(getattr(d, "page_content", ""))
        if not ctx:
            return 0.0
        return round(len(ans & ctx) / len(ans), 4)


class RagasFaithfulness:
    """Real RAGAS faithfulness metric.

    Uses ragas to extract the claims in ``answer`` and verify each against the
    retrieved contexts via ``llm`` (a langchain chat model — pass a real
    provider, e.g. ChatOllama for a local/offline judge or ChatAnthropic; the
    offline MockLLM won't work). Faithfulness needs only an LLM, not embeddings.
    """

    def __init__(self, llm: Any = None, embeddings: Any = None) -> None:
        self.llm = llm
        self.embeddings = embeddings

    def score(
        self, answer: str, documents: list[Document], query: str = ""
    ) -> float:
        contexts = [getattr(d, "page_content", str(d)) for d in documents]
        if not contexts or not (answer or "").strip():
            return 0.0  # nothing to score — short-circuit before importing ragas
        if self.llm is None:
            raise ValueError(
                "RagasFaithfulness needs a langchain chat model (real provider)."
            )

        _ensure_ragas_importable()
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness

        metric = Faithfulness(llm=LangchainLLMWrapper(self.llm))
        sample = SingleTurnSample(
            user_input=query or "",
            response=answer,
            retrieved_contexts=contexts,
        )

        async def _run() -> float:
            return float(await metric.single_turn_ascore(sample))

        try:
            return _run_sync(_run())
        except Exception as e:
            import os

            if os.getenv("RAGAS_DEBUG"):
                raise
            # RAGAS can raise on parse/timeouts; treat as unscoreable (0.0)
            # rather than failing the whole evaluation.
            import logging

            logging.getLogger(__name__).warning("RAGAS score failed: %r", e)
            return 0.0


def _ensure_ragas_importable() -> None:
    """Work around ragas 0.4.x hard-importing modules removed from langchain.

    ragas imports ``langchain_community.chat_models.vertexai`` (and a few other
    optional integrations) at package load; newer langchain-community dropped
    them. We don't use Vertex, so stub any missing submodule with a dummy class
    that nothing is an instance of, letting ragas import cleanly.
    """
    import importlib
    import sys
    import types

    optional = [
        ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    ]
    for mod_name, attr in optional:
        try:
            importlib.import_module(mod_name)
        except Exception:
            stub = types.ModuleType(mod_name)
            setattr(stub, attr, type(attr, (), {}))
            sys.modules[mod_name] = stub


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine to completion from sync code."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop: run in a fresh loop on a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def get_faithfulness(config: Any = None) -> FaithfulnessMetric:
    """Return the faithfulness metric.

    Defaults to the provider-independent lexical proxy (``MockFaithfulness``),
    which works on real answers too and needs no keys. Real RAGAS
    (``RagasFaithfulness``) is opt-in via ``FAITHFULNESS=ragas`` and uses the
    fast router model as the judge to keep cost down. It works with any real
    provider: ``LLM_PROVIDER=ollama`` runs the judge locally (e.g. qwen2.5:3b,
    free/offline), ``LLM_PROVIDER=anthropic`` uses Haiku. Only the mock provider
    won't work — its judge LLM isn't a langchain chat model.
    """
    import os

    if os.getenv("FAITHFULNESS", "").lower() == "ragas":
        from langconnect_agent.llm import get_router_llm

        return RagasFaithfulness(llm=get_router_llm(config))
    return MockFaithfulness()


# --- report ---------------------------------------------------------------
@dataclass
class CaseResult:
    case: EvalCase
    actual_route: str
    routed_correctly: bool
    fell_back: bool
    recovered: bool
    faithfulness: float


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    routing_accuracy: float = 0.0
    fallback_rate: float = 0.0
    recovery_rate: float = 0.0        # over cases where fallback was expected
    per_route_faithfulness: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== langconnect-agent evaluation (offline, mock data) ===",
            f"cases                : {len(self.results)}",
            f"routing accuracy     : {self.routing_accuracy:.2%}",
            f"fallback firing rate : {self.fallback_rate:.2%}",
            f"fallback recovery    : {self.recovery_rate:.2%}",
            "per-route faithfulness:",
        ]
        for route, score in sorted(self.per_route_faithfulness.items()):
            lines.append(f"  {route:9s}: {score:.3f}")
        return "\n".join(lines)


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def evaluate(
    *,
    graph: Any = None,
    dataset: Optional[list[EvalCase]] = None,
    faithfulness: Optional[FaithfulnessMetric] = None,
    config: Any = None,
) -> EvalReport:
    """Run the graph over the eval set and compute the differentiator metrics."""
    from langconnect_agent.config import get_config
    from langconnect_agent.graph import build_graph  # local import avoids cycle

    config = config or get_config()  # resolve so faithfulness gets the real provider
    graph = graph or build_graph(config=config)
    dataset = dataset or EVAL_DATASET
    faithfulness = faithfulness or get_faithfulness(config)

    results: list[CaseResult] = []
    for case in dataset:
        state = graph.invoke({"query": case.query})
        actual_route = state.get("route", "")
        documents = state.get("documents", []) or []
        fell_back = bool(state.get("fallbacks_used"))
        is_web = any(
            (getattr(d, "metadata", {}) or {}).get("source") == "web"
            for d in documents
        )
        results.append(
            CaseResult(
                case=case,
                actual_route=actual_route,
                routed_correctly=actual_route == case.expected_route,
                fell_back=fell_back,
                recovered=fell_back and is_web and bool(state.get("answer")),
                faithfulness=faithfulness.score(
                    state.get("answer", ""), documents, case.query
                ),
            )
        )

    n = len(results)
    routing_accuracy = _mean([1.0 if r.routed_correctly else 0.0 for r in results])
    fallback_rate = _mean([1.0 if r.fell_back else 0.0 for r in results])

    expected_fb = [r for r in results if r.case.expect_fallback]
    recovery_rate = _mean([1.0 if r.recovered else 0.0 for r in expected_fb])

    per_route: dict[str, list[float]] = {}
    for r in results:
        per_route.setdefault(r.actual_route, []).append(r.faithfulness)
    per_route_faithfulness = {k: _mean(v) for k, v in per_route.items()}

    return EvalReport(
        results=results,
        routing_accuracy=routing_accuracy,
        fallback_rate=fallback_rate,
        recovery_rate=recovery_rate,
        per_route_faithfulness=per_route_faithfulness,
    )


def register_langsmith_dataset(
    name: str = "wayfinder-eval",
    dataset: Optional[list[EvalCase]] = None,
) -> str:
    """Seam: register the eval set as a LangSmith dataset (needs a key).

    Offline (no LANGSMITH_API_KEY) this is a no-op that reports what it would do,
    so the harness stays runnable without a key.
    """
    import os

    dataset = dataset or EVAL_DATASET
    if not os.getenv("LANGSMITH_API_KEY"):
        return (
            f"[skip] would register {len(dataset)} cases as LangSmith dataset "
            f"{name!r} (set LANGSMITH_API_KEY to enable)"
        )

    from langsmith import Client  # lazy import

    client = Client()
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(dataset_name=name)
        for case in dataset:
            client.create_example(
                inputs={"query": case.query},
                outputs={
                    "expected_route": case.expected_route,
                    "expect_fallback": case.expect_fallback,
                },
                dataset_id=ds.id,
            )
    return f"[ok] registered {len(dataset)} cases as LangSmith dataset {name!r}"


# --- LangSmith experiment -------------------------------------------------
def make_target(
    graph: Any = None,
    faithfulness: Optional[FaithfulnessMetric] = None,
    config: Any = None,
):
    """Build the ``evaluate`` target: run the graph, return scoreable outputs."""
    from langconnect_agent.config import get_config
    from langconnect_agent.graph import build_graph  # local import avoids cycle

    config = config or get_config()  # resolve so faithfulness gets the real provider
    graph = graph or build_graph(config=config)
    faithfulness = faithfulness or get_faithfulness(config)

    def target(inputs: dict) -> dict:
        state = graph.invoke({"query": inputs["query"]})
        documents = state.get("documents", []) or []
        return {
            "route": state.get("route"),
            "fell_back": bool(state.get("fallbacks_used")),
            "grade": state.get("grade"),
            "faithfulness": faithfulness.score(
                state.get("answer", ""), documents, inputs["query"]
            ),
            "answer": (state.get("answer", "") or "")[:500],
        }

    return target


def eval_routing_accuracy(run: Any, example: Any) -> dict:
    """Did the router pick the expected route?"""
    correct = run.outputs.get("route") == example.outputs.get("expected_route")
    return {"key": "routing_correct", "score": float(correct)}


def eval_fallback_match(run: Any, example: Any) -> dict:
    """Did fallback fire iff the case expected weak grounding?"""
    match = bool(run.outputs.get("fell_back")) == bool(
        example.outputs.get("expect_fallback")
    )
    return {"key": "fallback_match", "score": float(match)}


def eval_faithfulness(run: Any, example: Any) -> dict:
    """Surface the target's faithfulness score as experiment feedback."""
    return {"key": "faithfulness", "score": float(run.outputs.get("faithfulness", 0.0))}


def run_langsmith_experiment(
    *,
    dataset_name: str = "wayfinder-eval",
    experiment_prefix: str = "wayfinder-routing",
    graph: Any = None,
    config: Any = None,
) -> Any:
    """Run a LangSmith experiment over the eval dataset (needs a key).

    Offline (no LANGSMITH_API_KEY) returns a skip message so the harness stays
    runnable without a key. When enabled, creates one experiment (visible in the
    dataset's Experiments tab) scored by routing accuracy, fallback match, and
    faithfulness.
    """
    import os

    if not os.getenv("LANGSMITH_API_KEY"):
        return (
            "[skip] set LANGSMITH_API_KEY to run a LangSmith experiment "
            f"over dataset {dataset_name!r}"
        )

    experiment_prefix = os.getenv("EXPERIMENT_PREFIX", experiment_prefix)

    from langsmith import evaluate  # lazy import

    return evaluate(
        make_target(graph=graph, config=config),
        data=dataset_name,
        evaluators=[eval_routing_accuracy, eval_fallback_match, eval_faithfulness],
        experiment_prefix=experiment_prefix,
    )
