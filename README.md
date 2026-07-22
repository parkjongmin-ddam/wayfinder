# Wayfinder — langconnect_agent

A **multi-source routing + self-correcting fallback** [LangGraph](https://github.com/langchain-ai/langgraph)
RAG agent. It runs **fully offline by default** — no database, no web API, no keys: stub
retriever/web-searcher, a mock router, a mock grader, and a mock LLM. Real providers, a real
pgvector store, and Tavily web search swap in behind clean seams with **no node changes**.

Built in staged phases, each with a passing gate (see `BUILD_SPEC.md`).

## Topology

```
START -> route -> retrieve | web_search -> grade -> answer -> verify -> END
                                            grade  -> web_search   (1-hop fallback)
                                            verify -> answer       (1-hop regen)
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
- **verify** — post-answer **faithfulness gate** (verification harness). Scores how well the
  answer is grounded in the retrieved context; if it is below threshold, regenerates **once**
  with a stricter grounding prompt (1-hop cap), else finishes. Metric is pluggable: the
  offline lexical proxy by default, real RAGAS via `FAITHFULNESS=ragas`. The threshold is
  calibrated low (`0.35`) for the lexical proxy — which underestimates paraphrased answers —
  so it catches gross hallucination without penalizing paraphrase.

Every run emits a one-line **decision trace** (BUILD_SPEC §5.4), e.g.:

```
query='What is the capital of France?' | route=A(semantic) | rationale='classified as semantic' | grade=0.85 | fallback=web | faith=0.62
```

## Install & run

```sh
pip install -e ".[test]"
cp .env.example .env
pytest -q                     # 68 tests: routing + fallback + verify + mcp + orchestrator + chat + seams
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
(`pip install -e ".[anthropic]"`, `ANTHROPIC_API_KEY`), `openai`
(`pip install -e ".[openai]"`, `OPENAI_API_KEY`), or `ollama`
(`pip install -e ".[ollama]"`, **no API key** — models run locally). Router =
fast model, answer = strong model, both overridable via `ROUTER_MODEL` /
`ANSWER_MODEL`. Defaults are provider-aware: hosted → `claude-haiku-4-5` /
`claude-opus-4-8`; ollama → `qwen2.5:3b` / `llama3.1:8b`. `OLLAMA_BASE_URL`
defaults to `http://localhost:11434`.

**Deterministic by default.** Sampling temperature defaults to `0.0`
(`ROUTER_TEMPERATURE` / `ANSWER_TEMPERATURE`): the router and grader are
classifiers and the answer is grounded, so `0.0` is the right default and makes
the whole decision trace **reproducible** — `python scripts/demo_trace.py` is
byte-identical across runs, even on a small local model. Raise it per role for
more varied answers.

## Seams (swap the stubs for the real thing)

| Seam | Protocol | Stub (offline) | Real | Wire in |
|---|---|---|---|---|
| Retrieval A/B | `Retriever` | `StubRetriever` | `LangConnectRetriever` (pgvector) | `build_graph(retriever=...)` |
| Web C | `WebSearcher` | `StubWebSearcher` | `TavilyWebSearcher` | `build_graph(web_searcher=...)` |
| Grading | `Grader` | `MockGrader` | `LLMGrader` → RAGAS (Phase 4) | `build_graph(grader=...)` |
| Embeddings | `get_embedder` | — | `OpenAIEmbeddings` \| `OllamaEmbeddings` | `EMBEDDING_PROVIDER` |

`LangConnectRetriever` (pgvector) is fully implemented — inject an embedder + connection
(or set `PGVECTOR_*` env vars) and it drops into the graph unchanged. `pip install -e ".[pgvector]"`.

## Phase 1 parity (pgvector — live)

Routes A/B run on a **real pgvector store** by default. `get_retriever(config)`
selects `LangConnectRetriever` (cosine `<=>`) when `PGVECTOR_CONNINFO` is set,
else the offline `StubRetriever` — `auto` | `stub` | `langconnect` via
`RETRIEVER_PROVIDER`. The embedder is pluggable via `EMBEDDING_PROVIDER`
(`auto` follows `LLM_PROVIDER`): OpenAI `text-embedding-3-small` or local
Ollama `nomic-embed-text`. The **same** model must embed both ingestion and query.

```sh
pip install -e ".[openai,pgvector]"
# Postgres 16 + pgvector; set PGVECTOR_CONNINFO + OPENAI_API_KEY in .env
python scripts/ingest.py            # chunk + embed ./corpus -> langconnect_embeddings
```

### Fully-local stack (Ollama + local embeddings + pgvector — no API keys)

The whole pipeline can run offline on local hardware — local router, local
embeddings, local answerer, all against a local pgvector. Measured on a 6 GB
GTX 1660 SUPER: `qwen2.5:3b` routes at ~87 tok/s (100% GPU), `llama3.1:8b`
answers at ~24 tok/s.

```sh
# 1. Ollama (https://ollama.com) — pull router, answerer, and embedder
ollama pull qwen2.5:3b llama3.1:8b nomic-embed-text

# 2. Local pgvector
docker run -d --name wayfinder-pgvector -e POSTGRES_PASSWORD=wayfinder \
  -e POSTGRES_DB=wayfinder -p 5433:5432 pgvector/pgvector:pg17

# 3. Install + configure .env
pip install -e ".[ollama,pgvector]"
#   LLM_PROVIDER=ollama                 # EMBEDDING_PROVIDER=auto -> local nomic-embed-text
#   PGVECTOR_CONNINFO=postgresql://postgres:wayfinder@localhost:5433/wayfinder

# 4. Ingest + run
python scripts/ingest.py            # embeds ./corpus with nomic-embed-text (768-dim)
python scripts/demo_trace.py
```

End-to-end verified: `"What is retrieval-augmented generation?"` routes semantic,
retrieves the in-corpus RAG doc from pgvector (top cosine ~0.82), and
`llama3.1:8b` answers grounded with **faithfulness 1.0, no web fallback** — the
full agent, running with zero API keys.

**Parity result (measured).** The `corpus/` RAG-concept docs answer the eval's
in-corpus route A/B queries. Same real `LLMGrader`, stub → real docs: the two
semantic in-corpus cases flip from *insufficient* (spurious web fallback) to
*sufficient* (grounded answer, no fallback) — real retrieval removes the stub
artifact that inflated the fallback rate (§4.1). The remaining fallback is the
grader's genuine judgment, not a stub effect.

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

Faithfulness defaults to a deterministic lexical proxy (`MockFaithfulness`). The real RAGAS
metric (`RagasFaithfulness`, an LLM judge — no embeddings needed) is opt-in via
`FAITHFULNESS=ragas` and runs **fully locally on Ollama** (`LLM_PROVIDER=ollama`, judge
`qwen2.5:3b`) or on a hosted provider. Verified live: a grounded answer scores **1.0**, a
fabricated one **0.0**. The same eval set registers as a LangSmith dataset via
`register_langsmith_dataset()` once `LANGSMITH_API_KEY` is set.

```bash
# real RAGAS faithfulness, offline on local Ollama
FAITHFULNESS=ragas LLM_PROVIDER=ollama python scripts/run_eval.py
```

## Observability (LangSmith)

LangGraph auto-instruments every node as a LangSmith span. Enable with
`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` (`pip install -e ".[langsmith]"`).
`observability.run_with_trace(graph, inputs, ...)` tags runs with queryable metadata and is a
transparent pass-through when tracing is off.

## MCP server (expose the agent)

The whole agent is exposed as a single MCP tool, `ask_wayfinder(query)`, so any MCP client
(Claude Desktop, an agent-builder platform) can call it. `run_agent(graph, query)` shapes the
result (answer + route + faithfulness + fallbacks + citations + trace); `create_server()` wraps
it in a FastMCP server. `pip install -e ".[mcp]"`.

```sh
python -m langconnect_agent.mcp_server     # serve over stdio
# or the console script:
wayfinder-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{"mcpServers": {"wayfinder": {"command": "wayfinder-mcp"}}}
```

Verified end-to-end over stdio: an MCP client `initialize`s, lists `ask_wayfinder`, and calls it
to get the routed, verified answer with citations — the same graph, now callable as a tool.

## Multi-agent orchestration (optional)

An **additive** orchestrator graph (`build_orchestrator()`) runs a **supervisor** over three
specialized sub-agents — the single-agent `graph.py` above is unchanged and still backs MCP,
eval, and most tests. Topology:

```
START -> route -> supervisor --(select)--> retrieval_agent -> supervisor
                                         -> web_agent        -> supervisor
                                         -> synthesis_agent  -> END
```

- **retrieval_agent** — rewrites the query into a few reformulations, runs a **multi-query**
  vector search, merges + dedups, and **self-checks** grounding sufficiency.
- **web_agent** — route-C web search with prompt-injection isolation.
- **synthesis_agent** — the only strong-model (Opus) step: synthesizes over the **accumulated**
  internal + web context and verifies faithfulness (1-hop regen).
- **supervisor** — an explainable **policy** controller (with an LLM-planner seam) that picks
  the next sub-agent from the route + the retrieval self-check; a `max_agent_steps` cap plus the
  policy guarantee termination. Sub-agents use the fast model; only synthesis uses the strong one.

Enable it as the served agent with `AGENT_MODE=multi` (the MCP server then exposes the
orchestrator behind the same `ask_wayfinder` tool), or run `python scripts/demo_orchestrator.py`.
The decision trace gains an `agents=retrieval_agent->web_agent->synthesis_agent` segment.

## Deploy (Phase 6)

`langgraph.json` serves two graphs — `agent` and `orchestrator` — over the LangGraph Server API.
Run locally with `langgraph dev` (`pip install "langgraph-cli[inmem]"`). Both graphs accept a chat
`messages` input and append the answer as an AIMessage with a `🧭 route · fallback · faith` badge,
so **agent-chat-ui** renders the path with no custom UI. Managed deploy (LangGraph Platform) +
Supabase pgvector + Vercel frontend: see [`DEPLOY.md`](./DEPLOY.md).

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
  nodes.py          route, retrieve, web_search, grade, answer, verify
  graph.py          build_graph(), module-level graph
  trace.py          one-line decision trace (§5.4)
  observability.py  LangSmith helpers
  evaluation.py     fixed eval set, faithfulness metric, metrics report (Phase 4)
  mcp_server.py     FastMCP server exposing ask_wayfinder (run_agent / create_server)
  agents.py         retrieval / web / synthesis sub-agents + query rewriter (Step 4)
  supervisor.py     supervisor policy controller (Step 4)
  orchestrator.py   build_orchestrator() multi-agent graph (Step 4)
scripts/
  demo_trace.py     print the decision trace per route
  demo_orchestrator.py  run the multi-agent orchestrator per path
  run_eval.py       offline metrics dashboard (routing acc / fallback / faithfulness)
  ingest.py         chunk + embed ./corpus into pgvector (Phase 1 parity)
tests/              routing, fallback, verify, mcp, orchestrator, isolation, pgvector, eval
```
