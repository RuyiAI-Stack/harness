"""Compatibility imports; embedding implementations live in dsh-vector-retrieval."""
from dsh_vector_retrieval.embeddings import (
    EmbeddingProvider, OllamaEmbeddingProvider, OpenAICompatibleEmbeddingProvider,
    SentenceTransformerEmbeddingProvider, build_embedding_provider, cosine_similarity,
    _sentence_encoder, urllib,
)

__all__ = [
    "EmbeddingProvider", "OllamaEmbeddingProvider", "OpenAICompatibleEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider", "build_embedding_provider", "cosine_similarity",
]
