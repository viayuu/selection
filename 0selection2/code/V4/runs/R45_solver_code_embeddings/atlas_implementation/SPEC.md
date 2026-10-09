# Atlas Semantic Views

Target: use MongoDB Atlas `https://ai.mongodb.com/v1/embeddings`,
`voyage-code-4`, 1024-dimensional float embeddings, no truncation.
Input limit: 32,000 official model tokens per text; at most 320,000 per request.
Account pacing: 2,000 RPM / 8,000,000 TPM.

Inputs: the existing reviewed solver manifest and explicitly accepted defaults.
Outputs: `corpus_atlas/`, `embedding_cache_atlas/`, and a locked
`solver_code_vectors_atlas.pt`, including all component vectors and provenance.
Each complete role is one input unless it exceeds the context window.
Oversized roles retain all semantic components for trainable attention pooling.

Base commit: `307a9b6dfda6797dc782550bca1017d3e8b01421`.
The pre-existing R45 files are untracked user work and must be preserved.
No paid API calls or selector training are part of this implementation check.

Success: offline Atlas/payload, semantic splitting, aggregation, cache isolation,
checkpoint replay tests; prepare the real-source corpus and print its upload plan.
