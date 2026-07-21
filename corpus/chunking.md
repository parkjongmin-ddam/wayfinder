# Chunking Documents for Retrieval

Before documents can be embedded and retrieved they are split into **chunks** —
smaller passages that each become one retrievable, independently-embedded unit.
Chunking matters because embeddings summarize a whole passage into a single
vector: too large a chunk blurs several topics into one averaged vector and
hurts ranking precision, while too small a chunk loses the surrounding context
needed to answer a question.

## Common strategies

- **Fixed-size chunks** of N tokens or characters, often with an **overlap** of
  10–20% so that a sentence split across a boundary still appears whole in one of
  the chunks.
- **Structure-aware chunking** that splits on natural boundaries — Markdown
  headings, paragraphs, or code blocks — so each chunk is topically coherent.
- **Sentence-window / parent-document** approaches that embed small units for
  precise matching but return a larger surrounding window as context.

## Metadata

Each chunk is stored with metadata (source document id, title, section, position)
so that retrieved context can be cited back to its origin and, if needed,
filtered before the similarity search. Good chunking plus clean metadata is often
a larger lever on RAG answer quality than swapping the embedding model.
