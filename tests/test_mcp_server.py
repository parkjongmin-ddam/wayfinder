"""MCP server exposure (offline — mock graph, no keys, no transport)."""

from __future__ import annotations

import asyncio
import json

from langconnect_agent.graph import build_graph
from langconnect_agent.mcp_server import create_server, run_agent


def test_run_agent_shapes_state_for_mcp():
    graph = build_graph()  # offline (conftest forces mock providers)
    out = run_agent(graph, "How does semantic vector search rank documents?")

    assert set(out) >= {
        "answer", "route", "faithfulness", "fallbacks_used", "citations", "trace"
    }
    assert isinstance(out["answer"], str) and out["answer"].strip()
    assert out["route"] == "semantic"
    assert isinstance(out["citations"], list) and out["citations"]
    assert isinstance(out["trace"], str) and out["trace"]


def test_create_server_registers_ask_wayfinder_tool():
    server = create_server(graph=build_graph(), name="wayfinder-test")
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert "ask_wayfinder" in names
    tool = next(t for t in tools if t.name == "ask_wayfinder")
    assert "query" in tool.inputSchema.get("properties", {})


def test_ask_wayfinder_tool_runs_end_to_end_offline():
    server = create_server(graph=build_graph(), name="wayfinder-test")
    result = asyncio.run(
        server.call_tool("ask_wayfinder", {"query": "What is vector search?"})
    )
    # FastMCP returns the tool's dict serialized as JSON inside TextContent
    # blocks (possibly wrapped in a (content, structured) tuple). Recover the
    # answer payload wherever it lands.
    def _find_payload(obj):
        if isinstance(obj, dict) and "answer" in obj:
            return obj
        text = getattr(obj, "text", None)
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except ValueError:
                return None
            if isinstance(data, dict) and "answer" in data:
                return data
        if isinstance(obj, (list, tuple)):
            for part in obj:
                found = _find_payload(part)
                if found is not None:
                    return found
        return None

    payload = _find_payload(result)
    assert payload is not None, f"no structured answer in {result!r}"
    assert payload["answer"].strip()
    assert payload["route"] == "semantic"
