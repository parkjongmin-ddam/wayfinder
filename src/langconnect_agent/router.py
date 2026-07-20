"""Phase 2 routing: LLM intent classification into routes A/B/C.

BUILD_SPEC §5.1: the router is an LLM classifier (not a rule-based if/else),
and because an LLM is non-deterministic, any output that falls outside the
A/B/C schema must **default to semantic (A)** — a runtime safety net that is
separate from routing-accuracy measurement (Phase 4).

Routes:
  - semantic (A): conceptual / meaning-based questions
  - keyword  (B): exact terms, proper nouns, identifiers, versions, error codes
  - web      (C): needs the latest / external information not in the corpus

This module provides the routing prompt, a schema-safe ``parse_route``, and an
offline ``MockRouterLLM`` whose classification is deterministic so the routing
gate runs with no network and no API key. The mock stands in for a real fast
LLM (Haiku); it simulates the LLM's classification, it is not the production
router.
"""

from __future__ import annotations

import re
from typing import Any, Optional

ROUTES: tuple[str, ...] = ("semantic", "keyword", "web")

# route -> demo letter used in the decision trace (A/B/C, BUILD_SPEC §5.1)
ROUTE_LETTERS: dict[str, str] = {"semantic": "A", "keyword": "B", "web": "C"}

# Query is embedded after this marker so MockRouterLLM can recover it from the
# rendered prompt (a real LLM reads the whole prompt and returns a label).
_QUERY_MARKER = "User query:"

# Time-sensitive / external-info cues → route C (web). English + Korean.
_WEB_HINTS: tuple[str, ...] = (
    "latest",
    "today",
    "recent",
    "recently",
    "current",
    "currently",
    "news",
    "this year",
    "right now",
    "up to date",
    "2024",
    "2025",
    "2026",
    "최신",
    "오늘",
    "최근",
    "현재",
    "뉴스",
    "지금",
)

# Exact-match cues → route B (keyword).
_KEYWORD_HINTS: tuple[str, ...] = (
    "exact",
    "error code",
    "stack trace",
    "cve-",
    "rfc ",
)

_VERSION_RE = re.compile(r"\bv?\d+\.\d+")
_ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,}\b")


def build_router_prompt(query: str) -> str:
    """Render the routing classification prompt for a query."""
    return (
        "You are a routing classifier for a retrieval agent. Classify the "
        "user query into exactly one retrieval route:\n"
        "- semantic: conceptual / how / why questions best answered by "
        "meaning-based vector search\n"
        "- keyword: exact terms, proper nouns, identifiers, versions, or error "
        "codes that need precise matching\n"
        "- web: needs the latest or external information not in the corpus\n\n"
        "Respond with ONLY one word: semantic, keyword, or web.\n\n"
        f"{_QUERY_MARKER} {query}\n"
        "Route:"
    )


def parse_route(text: Any) -> Optional[str]:
    """Parse an LLM routing response into a canonical route, or None.

    Accepts a bare label ("semantic"/"keyword"/"web") or a letter (A/B/C).
    Returns None when the response is empty, unrecognized, or ambiguous
    (names more than one distinct route) — the caller then applies the safe
    semantic default (BUILD_SPEC §5.1).
    """
    t = str(text or "").strip().lower()
    if not t:
        return None

    named = {r for r in ROUTES if re.search(rf"\b{r}\b", t)}
    if len(named) == 1:
        return next(iter(named))
    if len(named) > 1:
        return None  # ambiguous — let the caller default

    letter_to_route = {v.lower(): k for k, v in ROUTE_LETTERS.items()}
    lettered = {
        letter_to_route[m] for m in re.findall(r"\b([abc])\b", t)
        if m in letter_to_route
    }
    if len(lettered) == 1:
        return next(iter(lettered))
    return None


def classify_query(query: str) -> str:
    """Deterministic heuristic classification used by the offline mock router.

    Order: web cues first (most specific), then keyword cues, else semantic.
    """
    raw = query or ""
    low = raw.lower()

    if any(hint in low for hint in _WEB_HINTS):
        return "web"

    if any(hint in low for hint in _KEYWORD_HINTS):
        return "keyword"
    if '"' in raw or "'" in raw:
        return "keyword"
    if _VERSION_RE.search(raw):
        return "keyword"
    if _ACRONYM_RE.search(raw):
        return "keyword"

    return "semantic"


def _extract_query(prompt: Any) -> str:
    """Recover the embedded query from a rendered routing prompt."""
    text = prompt if isinstance(prompt, str) else str(prompt)
    if _QUERY_MARKER in text:
        after = text.split(_QUERY_MARKER, 1)[1]
        # Query runs to end-of-line (the prompt ends with "\nRoute:").
        return after.split("\n", 1)[0].strip()
    return text.strip()


class MockRouterLLM:
    """Deterministic, offline router LLM. No network, no API key.

    Recovers the query from the rendered routing prompt and returns a route
    label via ``classify_query``. Stands in for a real fast model so the
    routing gate is reproducible offline.
    """

    def __init__(self, model: str = "mock-router", **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    def invoke(self, prompt: Any) -> str:
        return classify_query(_extract_query(prompt))
