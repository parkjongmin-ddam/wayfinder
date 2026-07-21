"""Ultra-light decision trace (BUILD_SPEC §5.4).

A single line per query making the routing decision auditable *during
development* — the minimal version of the Phase 6 UI's path/fallback display,
pulled forward to Phase 2. Complements LangSmith traces (post-hoc) with an
immediate, eyeball-able signal in the dev loop.

Format:
  query='...' | route=A(semantic) | rationale='...' | grade=n/a | fallback=no

``grade`` is populated by the Phase 3 grade node (``n/a`` before grading);
``fallback`` shows the fallback hop(s) taken, or ``no``.
"""

from __future__ import annotations

from typing import Any

from langconnect_agent.router import ROUTE_LETTERS
from langconnect_agent.state import AgentState


def _clip(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_trace(state: AgentState) -> str:
    """Render the one-line decision trace from the current graph state."""
    query = state.get("query", "") or ""
    route = state.get("route", "") or "?"
    letter = ROUTE_LETTERS.get(route, "?")
    rationale = state.get("router_rationale", "") or ""

    grade = state.get("grade", None)  # Phase 3
    grade_str = "n/a" if grade is None else f"{float(grade):.2f}"

    fallbacks = state.get("fallbacks_used", []) or []
    fallback_str = "->".join(fallbacks) if fallbacks else "no"

    faith = state.get("faithfulness", None)  # Phase 2 verification harness
    faith_str = "n/a" if faith is None else f"{float(faith):.2f}"
    regens = int(state.get("regen_count", 0) or 0)
    faith_suffix = f" (regen x{regens})" if regens else ""

    line = (
        f"query='{_clip(query)}' | route={letter}({route}) | "
        f"rationale='{_clip(rationale)}' | grade={grade_str} | "
        f"fallback={fallback_str} | faith={faith_str}{faith_suffix}"
    )

    agents = state.get("agent_log", []) or []  # Step 4 orchestrator only
    if agents:
        line += f" | agents={'->'.join(agents)}"
    return line


def print_trace(state: AgentState, **kwargs: Any) -> str:
    """Build the trace, print it, and return it (dev convenience)."""
    line = build_trace(state)
    print(line, **kwargs)
    return line
