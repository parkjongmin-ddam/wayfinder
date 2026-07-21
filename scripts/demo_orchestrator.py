"""Run the multi-agent orchestrator on one query per path and print the trace.

Offline by default (mock rewriter/router, stub retriever/searcher, MockGrader,
MockFaithfulness) unless ``.env`` selects real providers. Shows the supervisor's
sub-agent path via the ``agents=...`` segment of the decision trace.

    python scripts/demo_orchestrator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langconnect_agent.env import load_env  # noqa: E402
from langconnect_agent.orchestrator import build_orchestrator  # noqa: E402

QUERIES = [
    "How does semantic vector search rank documents?",   # in-corpus -> synth
    "How do I bake sourdough bread at home?",             # out-of-corpus -> web
    "What are the latest LangGraph releases in 2026?",    # web route
]


def main() -> int:
    load_env()
    app = build_orchestrator()
    for q in QUERIES:
        state = app.invoke({"query": q})
        print(state.get("trace", ""))
        print(f"  answer: {(state.get('answer', '') or '')[:90]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
