# Deploy (Phase 6) — LangGraph Platform + Supabase + Vercel

Goal (gate): an outside person reproduces the three-route demo from a link alone.

Stack:
- **Graph** → LangGraph Platform (managed; checkpointing/persistence provided by the platform).
- **Retriever corpus** → Supabase Postgres + pgvector (cloud-reachable, unlike local WSL).
- **Frontend** → agent-chat-ui on Vercel, pointed at the deployment. The route/fallback/faith
  badge is prepended to each answer (see `trace.chat_message_content`), so the UI shows the path
  with no custom components.

Two graphs are served (`langgraph.json`): `agent` (single-agent) and `orchestrator` (multi-agent).

---

## 1. Supabase — cloud pgvector

1. Create a project at supabase.com (free tier).
2. SQL editor: `create extension if not exists vector;`
3. Copy the **connection string** (Project settings → Database → Connection string → URI; use the
   pooled/`6543` connection for serverless). It looks like
   `postgresql://postgres.<ref>:<pwd>@aws-0-<region>.pooler.supabase.com:6543/postgres`.
4. Ingest the corpus into Supabase (locally, one time):
   ```sh
   PGVECTOR_CONNINFO="<supabase-uri>" OPENAI_API_KEY="sk-..." python scripts/ingest.py
   ```
   This creates `langconnect_embeddings` and loads the 18 corpus chunks.

## 2. LangGraph Platform — deploy the graph

1. Push the repo to GitHub (done: `parkjongmin-ddam/wayfinder`).
2. LangSmith (APAC: smith.langchain.com, APAC region) → **Deployments** → **+ New Deployment**.
3. Connect the GitHub repo; LangGraph config path = `langgraph.json`.
4. Set environment variables (secrets):
   | Var | Value |
   |---|---|
   | `ANTHROPIC_API_KEY` | your key (router=Haiku, synthesis=Opus) |
   | `OPENAI_API_KEY` | your key (query embeddings) |
   | `TAVILY_API_KEY` | your key (route C) |
   | `PGVECTOR_CONNINFO` | the Supabase URI from step 1 |
   | `LLM_PROVIDER` | `anthropic` |
   | `WEB_PROVIDER` | `auto` |
   | `RETRIEVER_PROVIDER` | `auto` |
   | `LANGSMITH_API_KEY` | your APAC key (tracing) |

   > Secrets live **in the deployment settings above**, not in the repo. The
   > repo's `.env` is gitignored (never pushed); `langgraph.json`'s `"env":
   > "./.env"` is only a convenience for local `langgraph dev` and is harmless
   > when absent in the cloud build. `langgraph.json` also lists the hosted
   > runtime deps (`langchain-anthropic`, `langchain-openai`, `psycopg`,
   > `tavily-python`, `langsmith`) so the platform installs them — the base
   > package alone would `ImportError` at runtime under `LLM_PROVIDER=anthropic`.
5. Deploy → copy the **deployment URL** and create an API key for it.

## 3. agent-chat-ui — frontend on Vercel

1. Deploy `github.com/langchain-ai/agent-chat-ui` to Vercel (its README has a one-click deploy).
2. Set the app config / env:
   | Var | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | the LangGraph deployment URL |
   | `NEXT_PUBLIC_ASSISTANT_ID` | `orchestrator` (or `agent`) |
   | `LANGSMITH_API_KEY` | deployment API key (proxied server-side) |
3. Open the Vercel URL → ask the three demo queries:
   - "How does semantic vector search rank documents?" → **A/semantic**, no fallback
   - "What are the latest LangGraph releases in 2026?" → **C/web**
   - a question weakly grounded internally → self-corrects to **web fallback**

   Each answer shows the `🧭 route … · fallback … · faith …` badge.

## Gate

Share the Vercel link. An outsider reproduces all three routes with no setup. Done.
