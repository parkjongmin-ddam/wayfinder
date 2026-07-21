# Text Embeddings

An **embedding** is a fixed-length vector of floating-point numbers that
represents the meaning of a piece of text. An embedding model is trained so that
texts with similar meaning are mapped to vectors that are close together (small
angular distance), and unrelated texts to vectors that are far apart.

## Properties that matter for retrieval

- **Dimensionality.** A model emits vectors of a fixed size. For example,
  OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors. Every
  vector stored in a given index must share the same dimensionality.
- **Same model for ingestion and query.** The corpus and the incoming query must
  be embedded with the *same* model. Mixing models places their vectors in
  incompatible spaces and makes similarity scores meaningless.
- **Normalization.** Many embedding models output (near) unit-length vectors,
  which is why cosine similarity — an angle-only measure — is the natural way to
  compare them.

## Where embeddings sit in a RAG pipeline

At ingestion, documents are split into chunks and each chunk is embedded and
stored alongside its text and metadata in a vector store such as pgvector. At
query time the question is embedded once and compared against the stored vectors
to retrieve the most semantically similar chunks, which then become the grounding
context for the language model's answer.
