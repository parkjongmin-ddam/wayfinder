"""Retriever protocol, Document type, and concrete retriever implementations.

``Document`` is defined here and re-used across the codebase. ``StubRetriever``
returns deterministic canned docs so tests run fully offline. ``LangConnectRetriever``
is a connection SEAM: its ``search`` intentionally raises ``NotImplementedError``
until real pgvector wiring is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Document:
    """A retrieved chunk of content plus arbitrary metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Protocol every retriever must satisfy.

    ``route`` selects a retrieval strategy (e.g. "semantic", "keyword") in later
    phases; Phase 1 always passes "semantic".
    """

    def search(
        self, query: str, k: int = 5, route: str = "semantic"
    ) -> list[Document]:
        ...


class StubRetriever:
    """Deterministic, offline retriever returning canned documents.

    Used as the default retriever so the graph runs with no DB and no keys.
    """

    def search(
        self, query: str, k: int = 5, route: str = "semantic"
    ) -> list[Document]:
        docs = [
            Document(
                page_content=(
                    f"Stub document {i} relevant to query: {query!r}."
                ),
                metadata={
                    "id": f"stub-{i}",
                    "source": "stub",
                    "route": route,
                    "score": round(1.0 - i * 0.1, 4),
                },
            )
            for i in range(1, max(k, 0) + 1)
        ]
        return docs


def _as_vector_literal(vector: "list[float]") -> str:
    """Render an embedding as a pgvector text literal, e.g. ``[0.1,0.2,0.3]``."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class LangConnectRetriever:
    """Real pgvector retriever — the langconnect-v2 backend behind the SEAM.

    The graph depends only on the ``Retriever`` protocol, so wiring this in
    reaches Phase 1 parity with **no node changes**. Everything is injectable
    (embedder + connection factory) so the row→Document mapping is unit-testable
    without a live database.

    Connection and schema are configured via constructor args or environment
    variables (``PGVECTOR_CONNINFO``, ``PGVECTOR_TABLE``, ``PGVECTOR_*_COLUMN``).

    Note: Phase 1 parity only needs the single **semantic** (vector-similarity)
    path. Keyword/hybrid (route B) depends on what langconnect-v2 supports and
    is deferred (BUILD_SPEC §12); until then, all routes use vector similarity.
    """

    def __init__(
        self,
        *,
        embedder: Any = None,
        connect: Any = None,
        conninfo: Optional[str] = None,
        table: Optional[str] = None,
        id_column: Optional[str] = None,
        content_column: Optional[str] = None,
        embedding_column: Optional[str] = None,
        metadata_column: Optional[str] = None,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        import os

        self.embedder = embedder
        self._connect = connect
        self.conninfo = conninfo or os.getenv("PGVECTOR_CONNINFO")
        self.table = table or os.getenv("PGVECTOR_TABLE", "langconnect_embeddings")
        self.id_column = id_column or os.getenv("PGVECTOR_ID_COLUMN", "id")
        self.content_column = content_column or os.getenv(
            "PGVECTOR_CONTENT_COLUMN", "content"
        )
        self.embedding_column = embedding_column or os.getenv(
            "PGVECTOR_EMBEDDING_COLUMN", "embedding"
        )
        self.metadata_column = metadata_column or os.getenv(
            "PGVECTOR_METADATA_COLUMN", "metadata"
        )
        self.config = config
        self.options = kwargs

    # -- embedding -----------------------------------------------------------
    def _embed(self, query: str) -> list[float]:
        embedder = self.embedder
        if embedder is None:
            raise ValueError(
                "LangConnectRetriever needs an `embedder`: a langchain "
                "Embeddings object (with .embed_query) or a callable "
                "str -> list[float] using the corpus's ingestion-time model."
            )
        if hasattr(embedder, "embed_query"):
            return list(embedder.embed_query(query))
        if callable(embedder):
            return list(embedder(query))
        raise TypeError(f"Unsupported embedder type: {type(embedder)!r}")

    # -- connection ----------------------------------------------------------
    def _open(self) -> Any:
        if self._connect is not None:
            return self._connect()
        if not self.conninfo:
            raise ValueError(
                "LangConnectRetriever needs a `connect` factory or a "
                "`conninfo` / PGVECTOR_CONNINFO connection string."
            )
        import psycopg  # lazy import; optional dependency

        return psycopg.connect(self.conninfo)

    def _build_sql(self) -> str:
        # Cosine distance (<=>) ordering; score = 1 - distance (higher = closer).
        return (
            f"SELECT {self.id_column}, {self.content_column}, "
            f"{self.metadata_column}, "
            f"1 - ({self.embedding_column} <=> %(vec)s::vector) AS score "
            f"FROM {self.table} "
            f"ORDER BY {self.embedding_column} <=> %(vec)s::vector "
            f"LIMIT %(k)s"
        )

    @staticmethod
    def _row_to_document(row: Any) -> Document:
        row_id, content, metadata, score = row
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except (ValueError, TypeError):
                metadata = {"raw": metadata}
        meta = dict(metadata or {})
        meta.setdefault("id", row_id)
        meta.setdefault("source", "langconnect")
        meta["score"] = score
        return Document(page_content=content or "", metadata=meta)

    def search(
        self, query: str, k: int = 5, route: str = "semantic"
    ) -> list[Document]:
        """Vector-similarity search over the pgvector store (semantic path)."""
        vector = self._embed(query)
        params = {"vec": _as_vector_literal(vector), "k": k}
        sql = self._build_sql()

        with self._open() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_document(r) for r in rows]


def get_openai_embedder(config: Any = None) -> Any:
    """Return an OpenAI embeddings client (``.embed_query`` / ``.embed_documents``).

    Lazily imports ``langchain_openai`` so the dependency stays optional. The
    model defaults to ``config.embedding_model`` (``text-embedding-3-small``);
    the SAME model must be used for ingestion and query.
    """
    model = getattr(config, "embedding_model", None) or "text-embedding-3-small"
    from langchain_openai import OpenAIEmbeddings  # lazy import

    return OpenAIEmbeddings(model=model)


def get_ollama_embedder(config: Any = None) -> Any:
    """Return a local Ollama embeddings client (``.embed_query`` / ``.embed_documents``).

    Lazily imports ``langchain_ollama`` so the dependency stays optional. The
    model defaults to ``config.embedding_model`` (``nomic-embed-text``); the SAME
    model must be used for ingestion and query. Runs against the local Ollama
    server (``config.ollama_base_url`` / ``OLLAMA_BASE_URL``, default :11434).
    """
    import os

    model = getattr(config, "embedding_model", None) or "nomic-embed-text"
    base_url = (
        getattr(config, "ollama_base_url", None)
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    )
    from langchain_ollama import OllamaEmbeddings  # lazy import

    return OllamaEmbeddings(model=model, base_url=base_url)


def get_embedder(config: Any = None) -> Any:
    """Return the embeddings client for the configured provider.

    ``config.embedding_provider`` selects "ollama" (local) or "openai" (hosted);
    both expose ``.embed_query`` / ``.embed_documents`` so the retriever and the
    ingest script are provider-agnostic.
    """
    provider = (getattr(config, "embedding_provider", None) or "openai").lower()
    if provider == "ollama":
        return get_ollama_embedder(config)
    return get_openai_embedder(config)


def get_retriever(config: Any = None) -> Retriever:
    """Select the routes A/B retriever.

    "auto" (default) → real ``LangConnectRetriever`` (pgvector) when
    ``PGVECTOR_CONNINFO`` is set, else the offline ``StubRetriever``.
    "stub"/"langconnect" force one explicitly. The real retriever is wired with
    the configured embedder (``get_embedder``: local ollama or hosted openai),
    which MUST match the model the corpus was ingested with.
    """
    import os

    provider = (getattr(config, "retriever_provider", None) or "auto").lower()
    if provider == "stub":
        return StubRetriever()
    if provider == "langconnect":
        return LangConnectRetriever(embedder=get_embedder(config))
    # auto
    if os.getenv("PGVECTOR_CONNINFO"):
        return LangConnectRetriever(embedder=get_embedder(config))
    return StubRetriever()
