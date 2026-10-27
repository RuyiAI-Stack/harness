"""Scoped document/chunk/version API. Callers choose chunking and source validation."""
import hashlib
import json
import math
from dataclasses import dataclass, field
from uuid import UUID

from .validation import MAX_QUERY_TOKENS, page_budget, required_text, validate_vector


@dataclass(frozen=True)
class Scope:
    workspace_id: str
    generation: str
    index_fingerprint: str

    def __post_init__(self):
        for key, value in vars(self).items():
            required_text(value, key)

    def matches(self, payload):
        return isinstance(payload, dict) and all(payload.get(k) == v for k, v in vars(self).items())


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    version: int
    chunk_id: str
    source_version: str
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("document_id", "chunk_id", "source_version", "text"):
            required_text(getattr(self, key), key)
        if type(self.version) is not int or self.version < 1 or not isinstance(self.metadata, dict):
            raise ValueError("invalid document chunk")


class VectorIndex:
    def __init__(self, transport, scope: Scope):
        if not isinstance(scope, Scope):
            raise ValueError("workspace scope is required")
        self.transport, self.scope = transport, scope

    def embed(self, texts, provider):
        if provider is None:
            raise ValueError("embedding provider is not configured")
        if not isinstance(texts, list) or not 1 <= len(texts) <= 100:
            raise ValueError("embedding batch must contain 1 to 100 texts")
        for text in texts:
            required_text(text, "text")
            if provider.count_tokens(text) > provider.max_input_tokens:
                raise ValueError("embedding input exceeds token budget")
        vectors = provider.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedding count mismatch")
        for vector in vectors:
            validate_vector(vector, self.transport.dimensions)
        return vectors

    def index_chunks(self, chunks, *, vectors=None, provider=None):
        if not isinstance(chunks, list) or not 1 <= len(chunks) <= 100:
            raise ValueError("index batch must contain 1 to 100 chunks")
        if any(not isinstance(chunk, DocumentChunk) for chunk in chunks):
            raise ValueError("invalid document chunks")
        if vectors is None:
            vectors = self.embed([chunk.text for chunk in chunks], provider)
        if len(vectors) != len(chunks):
            raise ValueError("embedding count mismatch")
        points = []
        for chunk, vector in zip(chunks, vectors):
            validate_vector(vector, self.transport.dimensions)
            # Point IDs are scope-bound and cannot be supplied by an untrusted request.
            identity = [*vars(self.scope).values(), chunk.document_id, chunk.version,
                        chunk.source_version, chunk.chunk_id]
            identifier = str(UUID(hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]))
            points.append({"id": identifier, "vector": vector, "payload": {
                **vars(self.scope), self.transport.fields.document: chunk.document_id,
                self.transport.fields.version: chunk.version, "chunk_id": chunk.chunk_id,
                "source_version": chunk.source_version, "text": chunk.text, "metadata": chunk.metadata,
            }})
        self.transport.ensure()
        self.transport.upsert(points)
        return [point["id"] for point in points]

    def search(self, vector, *, limit=20, offset=0):
        page_budget(limit, offset)
        validate_vector(vector, self.transport.dimensions)
        hits = self.transport.query(vector, self.scope.workspace_id, self.scope.generation,
                                    self.scope.index_fingerprint, limit=limit, offset=offset)
        # Never trust even a successful service response to respect its filter.
        return [hit for hit in hits if isinstance(hit, dict) and self.scope.matches(hit.get("payload"))
                and type(hit.get("score")) in {int, float} and math.isfinite(hit["score"])]

    def search_text(self, text, provider, *, limit=20, offset=0):
        page_budget(limit, offset)
        required_text(text, "text")
        if provider is None or provider.count_tokens(text) > min(MAX_QUERY_TOKENS, provider.max_input_tokens):
            raise ValueError("query embedding provider or token budget is invalid")
        return self.search(self.embed([text], provider)[0], limit=limit, offset=offset)

    def delete_version(self, document_id, version, *, older=False, include=False):
        return self.transport.delete_version(self.scope.workspace_id, document_id, version,
            older=older, include=include, generation=self.scope.generation,
            fingerprint=self.scope.index_fingerprint)
