# Reranking and Why It Improves Recall

Retrieval is usually a two-stage pipeline. A fast **first-stage retriever**
(dense vector search, BM25 keyword search, or a hybrid of both) pulls a broad
candidate set — say the top 50–100 chunks — optimizing for **recall**: making
sure the relevant documents are somewhere in that set. A slower, more accurate
**reranker** then reorders those candidates and keeps only the top few that are
actually fed to the LLM.

## Why add a reranker

The first stage is deliberately cheap so it can scan the whole corpus. That
speed comes at a cost: a single query embedding cannot capture every nuance, so
the *ordering* of the candidate list is noisy even when the relevant document is
present. A **cross-encoder reranker** scores each (query, document) pair
*jointly* — the query and the candidate are concatenated and passed through the
model together, letting every query token attend to every document token. This
joint attention is far more precise than comparing two independently-computed
vectors, but it is too expensive to run over the entire corpus.

## How reranking improves recall in practice

Strictly speaking, reranking improves **precision@k** — the fraction of the
final top-k that are relevant. It improves *effective* recall in the way that
matters for RAG: because the reranker is more accurate, you can set the
first-stage retriever to return a **larger** candidate pool (raising true recall)
without flooding the LLM with noise, since the reranker will filter that pool
down to the genuinely relevant few. In other words:

- Retrieve wide (high recall) with the cheap first stage.
- Rerank narrow (high precision) with the expensive cross-encoder.

The net effect is that relevant documents that the first stage ranked at, say,
position 30 get promoted into the top 5 the model actually sees — so answers are
grounded in context that a one-stage system would have discarded. That is why
adding a reranker raises end-to-end answer quality even when the first-stage
retriever is unchanged.
