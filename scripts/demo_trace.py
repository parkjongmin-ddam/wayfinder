"""Phase 2 demo: run one query per route and print the decision trace.

The minimal "vision" surface from BUILD_SPEC §5.4 — eyeball the routing
decision in the dev loop. Run offline (mock router + stub sources):

    python scripts/demo_trace.py
"""

from __future__ import annotations

from langconnect_agent.env import load_env

load_env()  # pull .env (keys, LangSmith config) into the environment

from langconnect_agent.graph import build_graph  # noqa: E402
from langconnect_agent.observability import (  # noqa: E402
    run_with_trace,
    tracing_enabled,
)

QUERIES = [
    "How does vector similarity search work conceptually?",   # -> A semantic
    'What does the "str_replace_based_edit_tool" error mean?',  # -> B keyword
    "What are the latest LangGraph features released in 2026?",  # -> C web
    "What is the capital of France?",  # -> A semantic, weak grounding -> web fallback
]


def main() -> None:
    status = "ON" if tracing_enabled() else "off (set LANGSMITH_TRACING + key)"
    print(f"[LangSmith tracing: {status}]")
    graph = build_graph()
    for q in QUERIES:
        # run_with_trace tags the LangSmith run and echoes the local trace line.
        run_with_trace(graph, {"query": q}, print_local_trace=True)


if __name__ == "__main__":
    main()
