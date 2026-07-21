# Semantic Vector Search: How Documents Are Ranked

Semantic (dense) vector search ranks documents by *meaning* rather than by
literal keyword overlap. At ingestion time every document chunk is passed
through an embedding model that maps its text to a fixed-length vector (an
embedding) in a high-dimensional space. Chunks with similar meaning land close
together in that space, even when they share no words.

## The ranking procedure

1. **Embed the query.** The user's query is embedded with the *same* model used
   for the corpus, producing a query vector `q`.
2. **Measure similarity.** For every candidate document vector `d`, the engine
   computes a similarity score against `q`. The most common measure is **cosine
   similarity**, which compares the *angle* between the two vectors and ignores
   their magnitude: `cos(q, d) = (q · d) / (‖q‖ · ‖d‖)`. Scores range from -1
   (opposite) to 1 (identical direction); higher means more semantically
   similar.
3. **Sort and cut.** Documents are ranked in descending order of similarity and
   the top-k are returned. Approximate-nearest-neighbor (ANN) indexes such as
   HNSW or IVFFlat make this fast over millions of vectors by avoiding a full
   brute-force scan.

## Why ranking works on meaning

Because the embedding model was trained so that semantically related text has a
small angular distance, ranking by cosine similarity surfaces documents that
*answer* the query rather than merely *mention* its words. A query like "how are
documents ordered by relevance?" can retrieve a passage about "ranking search
results by similarity score" even with zero shared keywords.

Distance and similarity are two sides of the same coin: cosine **distance** is
`1 - cosine similarity`, so sorting ascending by distance is identical to
sorting descending by similarity. Vector databases expose this as a distance
operator (see the pgvector operators note) and typically report a `score` of
`1 - distance` so that higher is better.
