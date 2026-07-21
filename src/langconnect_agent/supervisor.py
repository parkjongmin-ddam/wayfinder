"""Supervisor controller for the multi-agent orchestrator (Step 4, B-lite).

A bounded policy controller: given the routing decision and the sub-agents'
signals, it picks the next sub-agent (retrieval / web / synthesis) and finishes
by handing off to synthesis. Termination is guaranteed two ways — a hard
``max_agent_steps`` cap and a policy that reaches synthesis within at most two
gather steps.

Kept as an explainable policy (not an opaque LLM loop) for determinism, offline
testability, and cost — with a ``planner`` seam for a future LLM planner. This
mirrors production "supervisor" agents, which are usually policies over agent
signals rather than free-running loops.
"""

from __future__ import annotations

from typing import Any, Optional

from langconnect_agent.config import Config, get_config
from langconnect_agent.state import AgentState

RETRIEVAL = "retrieval_agent"
WEB = "web_agent"
SYNTHESIS = "synthesis_agent"


def _decide(state: AgentState) -> str:
    """Pure policy: choose the next sub-agent from route + accumulated signals."""
    log = state.get("agent_log", []) or []
    last = log[-1] if log else None
    route = state.get("route", "semantic") or "semantic"

    if not log:  # nothing gathered yet
        return WEB if route == "web" else RETRIEVAL
    if last == RETRIEVAL:
        # augment with the web only when the retrieval self-check was weak
        return SYNTHESIS if state.get("retrieval_sufficient") else WEB
    if last == WEB:
        return SYNTHESIS
    return SYNTHESIS


def supervisor(
    state: AgentState,
    *,
    planner: Any = None,
    cfg: Optional[Config] = None,
) -> dict[str, Any]:
    """Decide the next sub-agent and advance the step counter.

    ``planner`` (optional) is a seam for an LLM planner; when absent the
    deterministic ``_decide`` policy is used. Once the step budget is spent, the
    supervisor forces synthesis so the graph always terminates.
    """
    cfg = cfg or get_config()
    steps = int(state.get("agent_steps", 0) or 0)

    if steps >= cfg.max_agent_steps:
        nxt = SYNTHESIS
    elif planner is not None:
        nxt = planner(state)  # LLM-planner seam (unused by default)
    else:
        nxt = _decide(state)

    return {"next_agent": nxt, "agent_steps": steps + 1}


def select_agent(state: AgentState) -> str:
    """Conditional-edge selector: route to the supervisor's chosen sub-agent."""
    return state.get("next_agent", SYNTHESIS) or SYNTHESIS
