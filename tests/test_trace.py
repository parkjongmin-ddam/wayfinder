"""Decision-trace formatter (trace.build_trace / print_trace, BUILD_SPEC §5.4).

The one-line routing trace is a Phase 2 deliverable but had no dedicated test
(chat_badge / query_from_messages are covered in test_chat_adapter). Pin the
format: route letter, grade, fallback chain, faithfulness, regen + agents.
"""

from __future__ import annotations

from langconnect_agent.trace import build_trace, print_trace


def test_basic_routing_trace_shows_route_letter_and_placeholders():
    line = build_trace(
        {"query": "How does ranking work?", "route": "semantic",
         "router_rationale": "classified as semantic"}
    )

    assert "query='How does ranking work?'" in line
    assert "route=A(semantic)" in line
    assert "rationale='classified as semantic'" in line
    assert "grade=n/a" in line       # no grade yet
    assert "fallback=no" in line
    assert "faith=n/a" in line


def test_trace_renders_grade_and_fallback_chain():
    line = build_trace(
        {"query": "q", "route": "keyword", "grade": 0.42,
         "fallbacks_used": ["web"]}
    )

    assert "route=B(keyword)" in line
    assert "grade=0.42" in line
    assert "fallback=web" in line


def test_trace_shows_faithfulness_and_regen_suffix():
    line = build_trace(
        {"query": "q", "route": "web", "faithfulness": 0.8, "regen_count": 2}
    )

    assert "route=C(web)" in line
    assert "faith=0.80 (regen x2)" in line


def test_trace_marks_unknown_route_with_question_mark():
    line = build_trace({"query": "q", "route": "nonsense"})

    assert "route=?(nonsense)" in line
    # An empty/missing route falls back to the "?" placeholder route too.
    assert "route=?(?)" in build_trace({"query": "q"})


def test_trace_clips_a_long_query():
    long_q = "word " * 40  # ~200 chars
    line = build_trace({"query": long_q, "route": "semantic"})

    assert "…" in line                       # clipped
    assert long_q.strip() not in line         # full text not present


def test_trace_appends_agent_log_for_orchestrator():
    line = build_trace(
        {"query": "q", "route": "semantic",
         "agent_log": ["router", "retrieval", "answer"]}
    )

    assert "agents=router->retrieval->answer" in line


def test_print_trace_returns_and_prints_same_line(capsys):
    state = {"query": "q", "route": "semantic"}

    returned = print_trace(state)
    printed = capsys.readouterr().out.strip()

    assert returned == build_trace(state)
    assert printed == returned
