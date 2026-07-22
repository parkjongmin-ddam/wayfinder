# Demo — the decision trace, fully local

`scripts/demo_trace.py` runs one query per route (A/B/C) plus a fallback case and
prints the one-line decision trace for each. With `temperature=0` (the default)
the output is **deterministic**, so a recorded GIF matches every rerun.

## Run it (fully local — no API keys)

```sh
# prerequisites (once)
ollama pull qwen2.5:3b llama3.1:8b nomic-embed-text
docker run -d --name wayfinder-pgvector -e POSTGRES_PASSWORD=wayfinder \
  -e POSTGRES_DB=wayfinder -p 5433:5432 pgvector/pgvector:pg17
pip install -e ".[ollama,pgvector]"
python scripts/ingest.py     # embed ./corpus with nomic-embed-text

# the demo
LLM_PROVIDER=ollama \
PGVECTOR_CONNINFO=postgresql://postgres:wayfinder@localhost:5433/wayfinder \
python scripts/demo_trace.py
```

## Expected output (deterministic, byte-identical across runs)

```
[LangSmith tracing: off (set LANGSMITH_TRACING + key)]
query='How does vector similarity search work conceptually?' | route=A(semantic) | rationale='classified as semantic' | grade=0.50 | fallback=no | faith=0.76
query='pgvector <=> operator exact syntax' | route=B(keyword) | rationale='classified as keyword' | grade=0.50 | fallback=no | faith=0.70
query='What are the latest LangGraph features released in 2026?' | route=C(web) | rationale='classified as web' | grade=0.50 | fallback=no | faith=0.17 (regen x1)
query='What is the capital of France?' | route=A(semantic) | rationale='classified as semantic' | grade=0.50 | fallback=web | faith=0.18 (regen x1)
```

Reading it: **A** stays on the corpus, **B** (keyword) stays on the corpus, **C**
goes to web, and the out-of-corpus "capital of France?" is routed A but falls back
to web when the local grader rates corpus grounding weak — the self-correcting loop.

> Route C uses the offline `StubWebSearcher` unless `TAVILY_API_KEY` is set; with a
> key, `get_web_searcher` auto-selects `TavilyWebSearcher` for real web answers.

## Recording a GIF

The terminal output is short and deterministic, which makes it ideal for a GIF.
Any of these work:

- **Windows** — [ScreenToGif](https://www.screentogif.com/): start recording, run
  the demo command, stop, trim, export GIF.
- **asciinema + agg** (cross-platform, crisp text):
  ```sh
  asciinema rec demo.cast -c "LLM_PROVIDER=ollama PGVECTOR_CONNINFO=... python scripts/demo_trace.py"
  agg demo.cast demo.gif
  ```
- **terminalizer**:
  ```sh
  terminalizer record demo && terminalizer render demo -o demo.gif
  ```

Tips: use a dark theme, ~16px font, and pause ~1s at the end so the final trace
lines are readable. Drop the GIF at `docs/demo.gif` and embed it at the top of the
README (`![demo](docs/demo.gif)`).
