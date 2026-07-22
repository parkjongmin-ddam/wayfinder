# Retrieval-Augmented Generation (RAG)

**Retrieval-augmented generation (RAG)** is a technique that grounds a language
model's answer in documents fetched from an external knowledge source at query
time, instead of relying only on the parameters the model learned during
training. A RAG system first **retrieves** the passages most relevant to the
user's question, then hands them to the model as context so it can **generate**
an answer supported by that evidence. The point is to make answers current,
verifiable, and specific to a private or domain corpus the base model never saw.

## Why it is used

- **Reduces hallucination.** By conditioning the answer on retrieved passages,
  the model has real evidence to quote rather than guessing from memory.
- **Keeps knowledge fresh.** Updating the answer set is as simple as re-indexing
  documents — no expensive model retraining or fine-tuning.
- **Enables citation.** Because each answer traces back to specific source
  chunks, the system can show *where* a claim came from.
- **Works over private data.** A general model can answer questions about
  internal documents it was never trained on.

## How the pipeline works

1. **Ingest.** Source documents are split into [chunks](chunking.md), each
   embedded into a vector and stored in a vector database.
2. **Retrieve.** The user query is embedded with the same model and matched
   against the store by vector similarity (often with keyword or hybrid search
   and a reranking stage to improve recall).
3. **Augment.** The top retrieved chunks are inserted into the prompt as
   grounding context, usually with their source metadata for citation.
4. **Generate.** The language model reads the question plus the retrieved
   context and produces a grounded, cited answer.

## Grounding and faithfulness

A RAG answer is only as trustworthy as its grounding. A **faithful** answer
states only what the retrieved context supports; an unfaithful one adds claims
the context never made. Good retrieval (relevant chunks), good chunking (coherent
units with clean metadata), and a verification step that checks the answer
against its context are what separate a reliable RAG system from a plausible-
sounding but unreliable one.
