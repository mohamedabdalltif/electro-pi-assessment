# Section 2 Notes

## Chunking

I used `RecursiveCharacterTextSplitter` with `chunk_size=400` and `chunk_overlap=50`. The overlap is important — without it, a sentence that spans a chunk boundary gets cut and neither chunk has the full context. 400 chars is small enough that each chunk stays focused on one idea.

## What I'd improve for longer documents

**Semantic chunking** — instead of splitting by character count, split by paragraph or section boundary so each chunk is a complete thought. Much better recall on long policy docs.

**Hybrid search** — combine FAISS (dense vector) with BM25 (keyword). Vector search is great for semantic similarity but misses exact keyword matches like order IDs or product names. Merging both with Reciprocal Rank Fusion covers both cases well.

**Cross-encoder re-ranking** — after retrieving the top-k chunks, run them through a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score them against the query. The bi-encoder used for retrieval is fast but approximate; the cross-encoder is slower but much more precise. Worth it when answer quality matters.

## Hallucination guardrail

The prompt explicitly instructs the LLM to respond with a fixed string if the context doesn't contain the answer. This isn't foolproof (LLMs can still ignore instructions) but it works well enough for a bounded domain like this. A stricter version would check retrieved chunk scores and refuse to answer if the max similarity is below a threshold.
