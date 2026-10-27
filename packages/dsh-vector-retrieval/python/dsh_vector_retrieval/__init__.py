"""Domain-independent scoped vector indexing and retrieval."""

from .embeddings import EmbeddingProvider, build_embedding_provider, cosine_similarity
from .index import DocumentChunk, Scope, VectorIndex
from .qdrant import PayloadFields, Qdrant, VectorServiceError

__all__ = [
    "DocumentChunk", "EmbeddingProvider", "PayloadFields", "Qdrant", "Scope",
    "VectorIndex", "VectorServiceError", "build_embedding_provider", "cosine_similarity",
]
