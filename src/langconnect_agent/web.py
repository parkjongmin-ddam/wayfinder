"""Web-search seam for route C (external, latest information).

Mirrors ``retrievers.py``: a ``WebSearcher`` protocol, an offline
``StubWebSearcher`` returning deterministic canned results (so the graph runs
route C with no network and no API key), and ``TavilyWebSearcher`` as the real
seam to the Tavily API.

Web results carry a ``source="web"`` and a ``url`` in metadata. Per BUILD_SPEC
§5.1, downstream the answer node treats these excerpts as **data, not
instructions** (prompt-injection isolation) and cites the source URLs.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Protocol, runtime_checkable

from langconnect_agent.retrievers import Document


@runtime_checkable
class WebSearcher(Protocol):
    """Protocol every web searcher must satisfy (route C)."""

    def search(self, query: str, k: int = 5) -> list[Document]:
        ...


class StubWebSearcher:
    """Deterministic, offline web searcher returning canned results.

    Default for route C so the graph runs with no network and no API key.
    """

    def search(self, query: str, k: int = 5) -> list[Document]:
        docs = [
            Document(
                page_content=(
                    f"Web result {i} discussing recent information about "
                    f"{query!r}."
                ),
                metadata={
                    "id": f"web-{i}",
                    "source": "web",
                    "url": f"https://example.com/result/{i}",
                    "title": f"Result {i} for {query!r}",
                    "score": round(1.0 - i * 0.1, 4),
                },
            )
            for i in range(1, max(k, 0) + 1)
        ]
        return docs


class TavilyWebSearcher:
    """Real web searcher backed by the Tavily API (route C seam).

    Lazily imports ``tavily`` only when used, so the dependency stays optional.
    Requires ``TAVILY_API_KEY`` (or an explicit ``api_key``).
    """

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.options = kwargs

    def search(self, query: str, k: int = 5) -> list[Document]:
        import os

        from tavily import TavilyClient  # lazy import

        client = TavilyClient(api_key=self.api_key or os.getenv("TAVILY_API_KEY"))
        response = client.search(query, max_results=k)
        docs: list[Document] = []
        for i, hit in enumerate(response.get("results", []), start=1):
            docs.append(
                Document(
                    page_content=hit.get("content", ""),
                    metadata={
                        "id": f"web-{i}",
                        "source": "web",
                        "url": hit.get("url", ""),
                        "title": hit.get("title", ""),
                        "score": hit.get("score"),
                    },
                )
            )
        return docs


def get_web_searcher(config: Any = None) -> WebSearcher:
    """Select the route C searcher.

    "auto" (default) → ``TavilyWebSearcher`` when ``TAVILY_API_KEY`` is set,
    else the offline ``StubWebSearcher``. "stub"/"tavily" force one explicitly.
    """
    provider = (getattr(config, "web_provider", None) or "auto").lower()
    if provider == "stub":
        return StubWebSearcher()
    if provider == "tavily":
        return TavilyWebSearcher()
    # auto
    if os.getenv("TAVILY_API_KEY"):
        return TavilyWebSearcher()
    return StubWebSearcher()
