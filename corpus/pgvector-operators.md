# pgvector Distance Operators

`pgvector` is a PostgreSQL extension that adds a `vector` column type and a set
of **distance operators** used to order rows by similarity to a query vector.
Each operator is a two-character token of the form `<?>`.

## The operators and their exact meaning

- **`<=>` — cosine distance.** Returns `1 - cosine_similarity` between the two
  vectors. This is the operator used for semantic search over
  embedding-normalized text. `ORDER BY embedding <=> '[...]'::vector` sorts the
  nearest (smallest cosine distance = most similar) rows first. A score of
  `1 - (embedding <=> query)` recovers the cosine similarity so that higher is
  better.
- **`<->` — L2 (Euclidean) distance.** The straight-line distance between the
  two points in vector space.
- **`<#>` — negative inner product.** Returns the negative dot product; pgvector
  negates it so that, like the other operators, smaller values mean closer.
- **`<+>` — L1 (taxicab / Manhattan) distance.** Sum of absolute per-dimension
  differences (available in newer pgvector versions).

## Why the `<=>` cosine operator is the default for RAG

Text embeddings encode meaning primarily in their *direction*, not their
magnitude, so the angle between vectors is the meaningful signal. The `<=>`
cosine-distance operator compares exactly that angle, which is why it is the
standard choice for semantic document retrieval. A typical query is:

```sql
SELECT id, content, metadata,
       1 - (embedding <=> %(vec)s::vector) AS score
FROM   langconnect_embeddings
ORDER  BY embedding <=> %(vec)s::vector
LIMIT  %(k)s;
```

To make these operators fast at scale, create an ANN index whose `vector_ops`
class matches the operator — e.g. `vector_cosine_ops` for `<=>`,
`vector_l2_ops` for `<->`, `vector_ip_ops` for `<#>` — using either an HNSW or
IVFFlat index.
