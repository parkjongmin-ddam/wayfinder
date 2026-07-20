"""Phase 3 LangGraph assembly.

Topology (route branches to retrieve OR web_search; grade may fall back once to
web_search — 1-hop cap — before answering)::

    START -> route -> retrieve|web_search -> grade -> answer -> END
                                              grade -> web_search (fallback, 1-hop)

``build_graph()`` wires the nodes with offline defaults (StubRetriever,
StubWebSearcher, MockRouterLLM, MockGrader, mock answer LLM), so the compiled
graph runs with no database, no web API, and no keys. A module-level
``graph = build_graph()`` is exposed for ``langgraph dev`` and
``from langconnect_agent.graph import graph``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from langconnect_agent.config import Config, get_config
from langconnect_agent.grading import Grader, get_grader
from langconnect_agent.llm import get_llm, get_router_llm
from langconnect_agent.nodes import answer, grade, retrieve, route, web_search
from langconnect_agent.retrievers import Retriever, StubRetriever
from langconnect_agent.state import AgentState
from langconnect_agent.web import WebSearcher, get_web_searcher


def _select_path(state: AgentState) -> str:
    """Conditional-edge selector: route C goes to web_search, else retrieve."""
    return "web_search" if state.get("route") == "web" else "retrieve"


def _after_grade(state: AgentState) -> str:
    """Conditional-edge selector: fall back to web_search, else answer."""
    return "web_search" if state.get("needs_fallback") else "answer"


def build_graph(
    *,
    retriever: Optional[Retriever] = None,
    web_searcher: Optional[WebSearcher] = None,
    router_llm: Any = None,
    grader: Optional[Grader] = None,
    llm: Any = None,
    config: Optional[Config] = None,
) -> Any:
    """Build and compile the Phase 3 agent graph.

    Args:
        retriever: retriever for routes A/B. Defaults to offline StubRetriever.
        web_searcher: searcher for route C. Defaults to offline StubWebSearcher.
        router_llm: fast routing LLM. Defaults to the provider's router LLM
            (offline MockRouterLLM unless ``LLM_PROVIDER`` is overridden).
        grader: grounding-sufficiency grader. Defaults to the provider's grader
            (offline MockGrader).
        llm: answer-synthesis LLM. Defaults to the provider's answer LLM.
        config: runtime Config. Defaults to ``get_config()``.

    Returns:
        A compiled LangGraph graph with the Phase 3 routing + fallback topology.
    """
    config = config or get_config()
    retriever = retriever or StubRetriever()
    web_searcher = web_searcher or get_web_searcher(config)
    router_llm = router_llm if router_llm is not None else get_router_llm(config)
    grader = grader or get_grader(config)
    llm = llm if llm is not None else get_llm(config)

    builder = StateGraph(AgentState)
    builder.add_node("route", partial(route, router_llm=router_llm, cfg=config))
    builder.add_node(
        "retrieve", partial(retrieve, retriever=retriever, cfg=config)
    )
    builder.add_node(
        "web_search", partial(web_search, searcher=web_searcher, cfg=config)
    )
    builder.add_node("grade", partial(grade, grader=grader, cfg=config))
    builder.add_node("answer", partial(answer, llm=llm, cfg=config))

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _select_path,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )
    builder.add_edge("retrieve", "grade")
    builder.add_edge("web_search", "grade")
    builder.add_conditional_edges(
        "grade",
        _after_grade,
        {"web_search": "web_search", "answer": "answer"},
    )
    builder.add_edge("answer", END)

    return builder.compile()


# Module-level compiled graph for `langgraph dev` and direct imports.
# Offline by default: stub retriever/searcher + mock router/answer LLMs.
graph = build_graph()
