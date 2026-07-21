"""Multi-agent orchestrator graph (Step 4, B-lite) — ADDITIVE to graph.py.

A supervisor coordinates specialized sub-agents; the single-agent ``graph.py``
is unchanged and still backs MCP / eval / tests. Topology::

    START -> route -> supervisor --(select)--> retrieval_agent -> supervisor
                                              -> web_agent        -> supervisor
                                              -> synthesis_agent  -> END

``route`` sets the routing decision (reused, so routing stays measurable); the
supervisor picks sub-agents from that decision plus the retrieval agent's
self-check; synthesis is the single strong-model step and ends the run. Offline
defaults (mock rewriter / router, stub retriever+searcher, MockGrader,
MockFaithfulness) make the whole orchestrator runnable with no keys.

``build_orchestrator()`` mirrors ``build_graph()``'s injection surface.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from langconnect_agent.agents import (
    get_rewriter,
    retrieval_agent,
    synthesis_agent,
    web_agent,
)
from langconnect_agent.config import Config, get_config
from langconnect_agent.grading import Grader, get_grader
from langconnect_agent.llm import get_llm, get_router_llm
from langconnect_agent.nodes import route
from langconnect_agent.retrievers import Retriever, get_retriever
from langconnect_agent.state import AgentState
from langconnect_agent.supervisor import (
    RETRIEVAL,
    SYNTHESIS,
    WEB,
    select_agent,
    supervisor,
)
from langconnect_agent.web import WebSearcher, get_web_searcher


def build_orchestrator(
    *,
    retriever: Optional[Retriever] = None,
    web_searcher: Optional[WebSearcher] = None,
    router_llm: Any = None,
    rewriter: Any = None,
    grader: Optional[Grader] = None,
    llm: Any = None,
    faithfulness: Any = None,
    planner: Any = None,
    config: Optional[Config] = None,
) -> Any:
    """Build and compile the multi-agent orchestrator graph.

    All dependencies are injectable (defaults resolve to offline mocks/stubs),
    mirroring ``build_graph``. ``planner`` is an optional LLM-planner seam for
    the supervisor; by default the supervisor uses its deterministic policy.
    """
    config = config or get_config()
    retriever = retriever or get_retriever(config)
    web_searcher = web_searcher or get_web_searcher(config)
    router_llm = router_llm if router_llm is not None else get_router_llm(config)
    rewriter = rewriter if rewriter is not None else get_rewriter(config)
    grader = grader or get_grader(config)
    llm = llm if llm is not None else get_llm(config)
    if faithfulness is None:
        from langconnect_agent.evaluation import get_faithfulness

        faithfulness = get_faithfulness(config)

    builder = StateGraph(AgentState)
    builder.add_node("route", partial(route, router_llm=router_llm, cfg=config))
    builder.add_node(
        "supervisor", partial(supervisor, planner=planner, cfg=config)
    )
    builder.add_node(
        RETRIEVAL,
        partial(
            retrieval_agent,
            retriever=retriever,
            rewriter=rewriter,
            grader=grader,
            cfg=config,
        ),
    )
    builder.add_node(
        WEB, partial(web_agent, searcher=web_searcher, cfg=config)
    )
    builder.add_node(
        SYNTHESIS,
        partial(synthesis_agent, llm=llm, faithfulness=faithfulness, cfg=config),
    )

    builder.add_edge(START, "route")
    builder.add_edge("route", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        select_agent,
        {RETRIEVAL: RETRIEVAL, WEB: WEB, SYNTHESIS: SYNTHESIS},
    )
    builder.add_edge(RETRIEVAL, "supervisor")
    builder.add_edge(WEB, "supervisor")
    builder.add_edge(SYNTHESIS, END)

    return builder.compile()


# Module-level compiled orchestrator for `langgraph dev` / LangGraph Platform
# and `from langconnect_agent.orchestrator import orchestrator`.
orchestrator = build_orchestrator()
