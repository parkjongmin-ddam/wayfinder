# Hybrid Search: Combining Dense and Keyword Retrieval

**Hybrid search** runs two retrievers over the same corpus and merges their
results: a **dense** semantic retriever (embedding similarity) and a **sparse**
keyword retriever (BM25 / full-text). Each covers the other's blind spot.

- **Dense** search captures meaning and paraphrase — it finds "how documents are
  ordered by relevance" for a query about "ranking search results" — but it can
  miss rare exact tokens (a product code, an error string, an API name) that
  never appeared in its training data.
- **Keyword** search nails exact-term matches and out-of-vocabulary strings but
  is blind to synonyms and paraphrase.

## Merging the two rankings

The two result lists are fused into one ranking. A popular, model-free method is
**Reciprocal Rank Fusion (RRF)**, which scores each document by the sum of
`1 / (k + rank)` across the lists it appears in, rewarding documents ranked
highly by *either* retriever. Weighted score combination is an alternative when
the two scores are calibrated.

## When routing chooses one over the other

A routing agent may send a **conceptual / paraphrased** question down the dense
(semantic) path, and an **exact-term** question — one that hinges on a specific
literal token such as an operator name or an error code — down the keyword path.
Time-sensitive questions that the internal corpus cannot answer are instead sent
to a web search. This is the multi-source routing pattern: pick the retrieval
strategy that fits the query, and fall back to the web when internal grounding is
insufficient.
