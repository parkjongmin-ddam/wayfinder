"""agent-chat-ui / LangGraph Server chat adapter (offline).

The graphs accept a chat ``messages`` input (last human message seeds the query)
and append the final answer as an AIMessage carrying a route/fallback badge, so a
generic chat UI renders it.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from langconnect_agent.graph import build_graph
from langconnect_agent.orchestrator import build_orchestrator
from langconnect_agent.trace import chat_badge, query_from_messages


def test_query_from_messages_reads_last_human():
    state = {
        "messages": [
            AIMessage(content="hi"),
            HumanMessage(content="How does ranking work?"),
        ]
    }
    assert query_from_messages(state) == "How does ranking work?"
    assert query_from_messages({"messages": []}) == ""


def test_chat_badge_includes_route_and_fallback():
    badge = chat_badge({"route": "semantic", "fallbacks_used": [], "faithfulness": 0.6})
    assert "route A(semantic)" in badge
    assert "fallback no" in badge
    assert "faith 0.60" in badge


def test_single_graph_accepts_messages_and_emits_ai_answer():
    result = build_graph().invoke(
        {"messages": [HumanMessage(content="How are documents ranked?")]}
    )
    assert result["query"] == "How are documents ranked?"      # resolved from chat
    msgs = result["messages"]
    assert isinstance(msgs[-1], AIMessage)
    assert "🧭" in msgs[-1].content and "route" in msgs[-1].content
    assert result["answer"] in msgs[-1].content


def test_orchestrator_accepts_messages_and_emits_ai_answer():
    result = build_orchestrator().invoke(
        {"messages": [HumanMessage(content="How are documents ranked?")]}
    )
    msgs = result["messages"]
    assert isinstance(msgs[-1], AIMessage)
    assert "agents" in msgs[-1].content  # orchestrator path shown in the badge
