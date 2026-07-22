"""Ingest the local corpus into pgvector (Phase 1 parity).

Reads every ``*.md`` under ``corpus/``, splits each file into
heading/paragraph-aware chunks, embeds them with the configured OpenAI model
(``text-embedding-3-small`` by default), and (re)loads them into the
``langconnect_embeddings`` table that ``LangConnectRetriever`` queries.

The table is created if missing and TRUNCATEd before each load, so re-running is
idempotent. Connection comes from ``PGVECTOR_CONNINFO`` (via ``.env``).

Usage::

    python scripts/ingest.py            # ingest ./corpus
    python scripts/ingest.py path/to/dir
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make ``src`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langconnect_agent.config import get_config  # noqa: E402
from langconnect_agent.env import load_env  # noqa: E402
from langconnect_agent.retrievers import (  # noqa: E402
    _as_vector_literal,
    get_embedder,
)

TABLE = os.getenv("PGVECTOR_TABLE", "langconnect_embeddings")
MAX_CHARS = 1200  # soft cap per chunk; paragraphs are packed up to this size


def _chunk_markdown(text: str) -> list[str]:
    """Split markdown into coherent chunks: group paragraphs under each heading,
    packing them up to ``MAX_CHARS`` so no chunk blurs too many topics."""
    # Split into blocks on blank lines; keep headings attached to their section.
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    heading = ""
    for block in blocks:
        if block.startswith("#"):
            # flush the previous section, start a new one led by this heading
            if current:
                chunks.append("\n\n".join(current))
                current = []
                size = 0
            heading = block
            current.append(block)
            size += len(block)
            continue
        if size + len(block) > MAX_CHARS and current:
            chunks.append("\n\n".join(current))
            # carry the heading into the continuation for context
            current = [heading] if heading else []
            size = len(heading)
        current.append(block)
        size += len(block)
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _connect(conninfo: str):
    import psycopg  # lazy import

    return psycopg.connect(conninfo)


def main() -> int:
    load_env()
    config = get_config()
    conninfo = os.getenv("PGVECTOR_CONNINFO")
    if not conninfo:
        print("ERROR: PGVECTOR_CONNINFO is not set (check .env).", file=sys.stderr)
        return 2

    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "corpus"
    )
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        print(f"ERROR: no *.md files under {corpus_dir}", file=sys.stderr)
        return 2

    # Build (chunk text, metadata, id) records.
    records: list[tuple[str, str, dict]] = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        title = _title_of(raw, path.stem)
        for i, chunk in enumerate(_chunk_markdown(raw)):
            rid = f"{path.stem}-{i}"
            meta = {
                "source": path.name,
                "title": title,
                "chunk_index": i,
            }
            records.append((rid, chunk, meta))

    print(
        f"Read {len(files)} files -> {len(records)} chunks. "
        f"Embedding with {config.embedding_model}..."
    )
    embedder = get_embedder(config)
    vectors = embedder.embed_documents([c for _, c, _ in records])
    dim = len(vectors[0])
    print(f"Embedded {len(vectors)} chunks, dim={dim}. Loading into {TABLE}...")

    with _connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {TABLE} ("
                "  id text PRIMARY KEY,"
                "  content text NOT NULL,"
                "  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,"
                f"  embedding vector({dim}) NOT NULL"
                ");"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_cos_idx "
                f"ON {TABLE} USING hnsw (embedding vector_cosine_ops);"
            )
            cur.execute(f"TRUNCATE {TABLE};")
            for (rid, content, meta), vec in zip(records, vectors):
                cur.execute(
                    f"INSERT INTO {TABLE} (id, content, metadata, embedding) "
                    "VALUES (%(id)s, %(content)s, %(meta)s::jsonb, "
                    "%(vec)s::vector)",
                    {
                        "id": rid,
                        "content": content,
                        "meta": json.dumps(meta),
                        "vec": _as_vector_literal(vec),
                    },
                )
        conn.commit()

    print(f"Done. {len(records)} chunks loaded into {TABLE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
