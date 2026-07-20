# Wayfinder — langconnect_agent

A **multi-source routing + self-correcting fallback** [LangGraph](https://github.com/langchain-ai/langgraph)
RAG agent. It runs **fully offline by default** — no database, no web API, no keys: stub
retriever/web-searcher, a mock router, a mock grader, and a mock LLM. Real providers, a real
pgvector store, and Tavily web search swap in behind clean seams with **no node changes**.

Built in staged phases, each with a passing gate (see `BUILD_SPEC.md`).

## Topology (Phase 3)

```
START -> route -> retrieve | web_search -> grade -> answer -> END
                                            grade -> web_search   (1-hop fallback)
```

- **route** — an LLM classifies the query into one of three routes; off-schema/unparseable
  output safely defaults to **semantic** (runtime defense, BUILD_SPEC §5.1):
  - **A · semantic** — conceptual, meaning-based vector search
  - **B · keyword** — exact terms, proper nouns, identifiers, versions, error codes
  - **C · web** — latest/external info not in the corpus (Tavily)
- **retrieve** — routes A/B via the `Retriever` protocol.
- **web_search** — route C via the `WebSearcher` protocol.
- **grade** — RAGAS-style grounding-sufficiency judgment. If the grounding is weak it falls
  back **once** to the web path (1-hop cap), then answers.
- **answer** — synthesizes a grounded answer. Web excerpts are isolated as **untrusted data**
  (prompt-injection isolation) and cited by URL (BUILD_SPEC §5.1).

Every run emits a one-line **decision trace** (BUILD_SPEC §5.4), e.g.:

```
query='What is the capital of France?' | route=A(semantic) | rationale='classified as semantic' | grade=0.85 | fallback=web
```

## Install & run

```sh
pip install -e ".[test]"
cp .env.example .env
pytest -q                     # 26 tests: routing + fallback + isolation + seam mapping
python scripts/demo_trace.py  # print the decision trace for one query per route
```

Run the graph directly:

```python
from langconnect_agent.graph import graph
result = graph.invoke({"query": "What is Wayfinder?"})
print(result["answer"], result["trace"])
```

Serve it for the LangGraph dev UI (`langgraph.json`): `langgraph dev`.

## Provider swap

Set `LLM_PROVIDER` in `.env`: `mock` (default, offline), `anthropic`
(`pip install -e ".[anthropic]"`, `ANTHROPIC_API_KEY`), or `openai`
(`pip install -e ".[openai]"`, `OPENAI_API_KEY`). Router = fast model
(`claude-haiku-4-5`), answer = strong model (`claude-opus-4-8`); both overridable via env.

## Seams (swap the stubs for the real thing)

| Seam | Protocol | Stub (offline) | Real | Wire in |
|---|---|---|---|---|
| Retrieval A/B | `Retriever` | `StubRetriever` | `LangConnectRetriever` (pgvector) | `build_graph(retriever=...)` |
| Web C | `WebSearcher` | `StubWebSearcher` | `TavilyWebSearcher` | `build_graph(web_searcher=...)` |
| Grading | `Grader` | `MockGrader` | `LLMGrader` → RAGAS (Phase 4) | `build_graph(grader=...)` |

`LangConnectRetriever` (pgvector) is fully implemented — inject an embedder + connection
(or set `PGVECTOR_*` env vars) and it drops into the graph unchanged. `pip install -e ".[pgvector]"`.
The live Phase 1 parity gate additionally needs the real DB, the ingestion-time embedding
model, and the Phase 0 baseline.

## Evaluation (Phase 4 — the differentiator)

The point isn't "we built routing", it's "we measured it". `scripts/run_eval.py` runs the graph
over a fixed, route-typed eval set and prints the differentiator metrics — **offline, on mock
data, no keys**:

```sh
python scripts/run_eval.py
```

```
routing accuracy     : 100.00%
fallback firing rate : 28.57%
fallback recovery    : 100.00%
per-route faithfulness:
  keyword  : 0.462
  semantic : 0.469
  web      : 0.276
```

Faithfulness is a deterministic lexical proxy (`MockFaithfulness`); `RagasFaithfulness` is the
seam for the real RAGAS metric (needs an LLM + embeddings). The same eval set registers as a
LangSmith dataset via `register_langsmith_dataset()` once `LANGSMITH_API_KEY` is set.

## Observability (LangSmith)

LangGraph auto-instruments every node as a LangSmith span. Enable with
`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` (`pip install -e ".[langsmith]"`).
`observability.run_with_trace(graph, inputs, ...)` tags runs with queryable metadata and is a
transparent pass-through when tracing is off.

## Layout

```
src/langconnect_agent/
  state.py          AgentState (TypedDict)
  config.py         Config + get_config() (env-aware, no pydantic)
  retrievers.py     Document, Retriever, StubRetriever, LangConnectRetriever (pgvector)
  web.py            WebSearcher, StubWebSearcher, TavilyWebSearcher (route C seam)
  router.py         route classification, schema-safe parse, MockRouterLLM
  grading.py        Grader, MockGrader, LLMGrader (RAGAS-style sufficiency)
  llm.py            get_llm() / get_router_llm() factories, MockLLM
  nodes.py          route, retrieve, web_search, grade, answer
  graph.py          build_graph(), module-level graph
  trace.py          one-line decision trace (§5.4)
  observability.py  LangSmith helpers
  evaluation.py     fixed eval set, faithfulness metric, metrics report (Phase 4)
scripts/
  demo_trace.py     print the decision trace per route
  run_eval.py       offline metrics dashboard (routing acc / fallback / faithfulness)
tests/              routing, fallback, isolation, pgvector-mapping, observability, evaluation
```
