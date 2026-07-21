"""Expose the Wayfinder agent as an MCP tool (Step 3 / BUILD_SPEC §Phase 5).

Wraps the compiled LangGraph agent in a FastMCP server so any MCP client
(Claude Desktop, an agent-builder platform, ...) can call it as a single tool.
The heavy logic lives in ``run_agent`` (pure, offline-testable); ``create_server``
registers the tool, and ``main`` serves it over stdio.

Run it::

    python -m langconnect_agent.mcp_server      # stdio server
    # or, after install, the console script:
    wayfinder-mcp

Claude Desktop config (``claude_desktop_config.json``)::

    {"mcpServers": {"wayfinder": {"command": "wayfinder-mcp"}}}
"""

from __future__ import annotations

from typing import Any

from langconnect_agent.env import load_env
from langconnect_agent.graph import build_graph


def _citations(documents: list) -> list[dict]:
    """Extract compact, client-friendly citations from retrieved documents."""
    cites: list[dict] = []
    for d in documents or []:
        meta = getattr(d, "metadata", {}) or {}
        cites.append(
            {
                "id": meta.get("id"),
                "source": meta.get("source"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "score": meta.get("score"),
            }
        )
    return cites


def run_agent(graph: Any, query: str) -> dict:
    """Invoke the agent graph and shape its state for MCP consumers.

    Returns the answer plus the routing decision, faithfulness score, any
    fallbacks taken, source citations, and the one-line decision trace — the
    same signals the agent uses internally, surfaced for the caller.
    """
    state = graph.invoke({"query": query})
    return {
        "answer": state.get("answer", ""),
        "route": state.get("route"),
        "faithfulness": state.get("faithfulness"),
        "fallbacks_used": state.get("fallbacks_used", []) or [],
        "citations": _citations(state.get("documents", [])),
        "trace": state.get("trace", ""),
    }


def create_server(graph: Any = None, *, name: str = "wayfinder") -> Any:
    """Build the FastMCP server exposing ``ask_wayfinder``.

    With no ``graph``, loads ``.env`` and builds the real graph (the server
    serves the live agent). Tests inject an offline graph to stay hermetic.
    """
    from mcp.server.fastmcp import FastMCP

    if graph is None:
        load_env()
        graph = build_graph()

    mcp = FastMCP(name)

    @mcp.tool()
    def ask_wayfinder(query: str) -> dict:
        """Answer a question with the Wayfinder RAG agent.

        Routes the query (semantic / keyword / web), retrieves grounding
        context (pgvector or web search), self-corrects with a 1-hop web
        fallback when internal grounding is weak, and verifies the answer's
        faithfulness (regenerating once if it is ungrounded). Returns the
        answer plus route, faithfulness score, fallbacks taken, source
        citations, and a one-line decision trace.
        """
        return run_agent(graph, query)

    return mcp


def main() -> None:
    """Console entry point: serve the agent over stdio for MCP clients."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
