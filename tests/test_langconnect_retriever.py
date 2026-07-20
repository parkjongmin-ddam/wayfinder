"""Phase 1 parity seam: LangConnectRetriever row->Document mapping.

Exercised with a fake connection + fake embedder so the pgvector query/mapping
logic is verified without a live database. The live parity gate (single-path
output matching the Phase 0 baseline) still needs the real DB, ingestion-time
embedding model, and Phase 0 baseline — none of which are in this workspace.
"""

from __future__ import annotations

import pytest

from langconnect_agent.graph import build_graph
from langconnect_agent.retrievers import (
    Document,
    LangConnectRetriever,
    Retriever,
)


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.executed = (sql, params)

    def fetchall(self):
        return self.rows


class _FakeConn:
    def __init__(self, rows):
        self.cur = _FakeCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cur


def test_maps_rows_to_documents_and_parses_json_metadata():
    rows = [
        (1, "Vector search explained", {"title": "A"}, 0.91),
        (2, "Reranking overview", '{"title": "B"}', 0.80),  # JSON-string meta
    ]
    conn = _FakeConn(rows)
    retr = LangConnectRetriever(
        embedder=lambda q: [0.1, 0.2, 0.3], connect=lambda: conn
    )

    docs = retr.search("what is vector search", k=2, route="semantic")

    assert [d.page_content for d in docs] == [
        "Vector search explained",
        "Reranking overview",
    ]
    assert docs[0].metadata["title"] == "A"
    assert docs[1].metadata["title"] == "B"  # parsed from JSON string
    assert docs[0].metadata["score"] == 0.91
    assert docs[0].metadata["source"] == "langconnect"
    assert docs[0].metadata["id"] == 1


def test_query_embedding_is_bound_as_pgvector_literal():
    conn = _FakeConn([(1, "x", {}, 0.5)])
    retr = LangConnectRetriever(
        embedder=lambda q: [0.1, 0.2, 0.3], connect=lambda: conn
    )

    retr.search("q", k=2)

    sql, params = conn.cur.executed
    assert params["vec"] == "[0.1,0.2,0.3]"
    assert params["k"] == 2
    assert "::vector" in sql


def test_accepts_langchain_style_embeddings_object():
    class _Embeddings:
        def embed_query(self, q):
            return [1.0, 2.0]

    conn = _FakeConn([(7, "hit", {}, 0.42)])
    retr = LangConnectRetriever(embedder=_Embeddings(), connect=lambda: conn)

    docs = retr.search("q", k=1)

    assert docs[0].metadata["id"] == 7
    _, params = conn.cur.executed
    assert params["vec"] == "[1.0,2.0]"


def test_missing_embedder_raises_clear_error():
    retr = LangConnectRetriever(connect=lambda: _FakeConn([]))
    with pytest.raises(ValueError, match="embedder"):
        retr.search("q")


def test_missing_connection_raises_clear_error():
    retr = LangConnectRetriever(embedder=lambda q: [0.0])
    with pytest.raises(ValueError, match="conninfo|connect"):
        retr.search("q")


def test_satisfies_retriever_protocol():
    retr = LangConnectRetriever(
        embedder=lambda q: [0.0], connect=lambda: _FakeConn([])
    )
    assert isinstance(retr, Retriever)


def test_drops_into_graph_with_no_node_changes():
    """Injecting LangConnectRetriever reaches parity without touching nodes."""
    rows = [
        (1, "Dense vector retrieval ranks documents by cosine similarity.",
         {"title": "Retrieval"}, 0.93),
    ]
    conn = _FakeConn(rows)
    retr = LangConnectRetriever(
        embedder=lambda q: [0.1, 0.2], connect=lambda: conn
    )

    result = build_graph(retriever=retr).invoke(
        {"query": "How does semantic vector retrieval rank documents?"}
    )

    assert result["route"] == "semantic"
    assert result["documents"][0].metadata["source"] == "langconnect"
    assert not result.get("fallbacks_used")  # in-corpus query, graded sufficient
    assert result["answer"].strip()
    assert isinstance(result["documents"][0], Document)
