"""LangSmith observability (BUILD_SPEC Phase 2 gate: routing decision trace).

LangGraph auto-instruments every node (route, grade, ...) as a LangSmith span
when ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` are set — so the
routing decision and grade land in the trace with zero extra code. This module
adds the small ergonomics on top:

* ``tracing_enabled()`` — is LangSmith configured?
* ``run_with_trace(graph, inputs, ...)`` — invoke the graph while tagging the
  run with queryable tags/metadata (filter runs by app/route in the UI), and
  echo the local decision trace. Works identically offline (tracing off): it is
  a plain ``graph.invoke`` with a config attached, so it never requires a key.

This complements the local one-line decision trace (``trace.py``): the local
trace is for the dev loop; LangSmith is the post-hoc analysis surface.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langconnect_agent.trace import build_trace

_TRUTHY = {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    """True when LangSmith tracing is configured via environment."""
    flag = os.getenv("LANGSMITH_TRACING", "").strip().lower() in _TRUTHY
    return flag and bool(os.getenv("LANGSMITH_API_KEY"))


def run_with_trace(
    graph: Any,
    inputs: dict[str, Any],
    *,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    print_local_trace: bool = False,
) -> dict[str, Any]:
    """Invoke ``graph`` with LangSmith-friendly tags/metadata on the run.

    ``tags`` and ``metadata`` are passed through the RunnableConfig, which
    LangSmith records on the top-level run (queryable/filterable in the UI).
    Safe with tracing off — it is then just ``graph.invoke`` with an extra,
    ignored config. Returns the graph result.
    """
    md = {"app": "langconnect-agent", **(metadata or {})}
    config = {
        "tags": ["langconnect-agent", *(tags or [])],
        "metadata": md,
    }
    result = graph.invoke(inputs, config=config)

    if print_local_trace:
        line = result.get("trace") or build_trace(result)
        print(line)

    return result
