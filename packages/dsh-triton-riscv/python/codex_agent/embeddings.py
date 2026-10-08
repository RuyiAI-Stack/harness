#!/usr/bin/env python3
"""Pluggable embedding providers for hybrid agent-memory retrieval."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Minimal provider contract used by the memory store."""

    name: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""

    def count_tokens(self, text: str) -> int:
        """Count tokens with the tokenizer used by this embedding model."""

    @property
    def max_input_tokens(self) -> int:
        """Maximum safe input length including special tokens."""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


@dataclass
class SentenceTransformerEmbeddingProvider:
    """Optional local provider; sentence-transformers is imported lazily."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "sentence-transformers"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install it or disable embeddings"
            ) from exc
        self._encoder = SentenceTransformer(self.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._encoder.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.tokenizer.encode(text, add_special_tokens=True))

    @property
    def max_input_tokens(self) -> int:
        return int(self._encoder.max_seq_length)


@dataclass
class OpenAICompatibleEmbeddingProvider:
    """Provider for company or hosted APIs exposing an OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60
    name: str = "openai-compatible"
    tokenizer_json: str | None = None
    token_budget: int | None = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        if self.tokenizer_json:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError("tokenizers is required for tokenizer_json") from exc
            self._tokenizer = Tokenizer.from_file(self.tokenizer_json)
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token_budget must be positive")

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None or self.token_budget is None:
            raise RuntimeError(
                "OpenAI-compatible embedding needs the model's tokenizer_json and "
                "token_budget; vector indexing is disabled without exact counting"
            )
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)

    @property
    def max_input_tokens(self) -> int:
        if self.token_budget is None or self._tokenizer is None:
            raise RuntimeError("embedding tokenizer and token budget are not configured")
        return self.token_budget

    def embed(self, texts: list[str]) -> list[list[float]]:
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/embeddings"):
            endpoint += "/embeddings"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [row.get("embedding") for row in rows]
        if len(vectors) != len(texts) or any(not isinstance(item, list) for item in vectors):
            raise RuntimeError("embedding endpoint returned an invalid response")
        return [[float(value) for value in vector] for vector in vectors]


@dataclass
class OllamaEmbeddingProvider:
    """Local Ollama embeddings with model-native token counting."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    token_budget: int = 256
    timeout_seconds: int = 60
    name: str = "ollama"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("Ollama embedding model is required")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        self._token_counts: dict[str, int] = {}

    @property
    def max_input_tokens(self) -> int:
        return self.token_budget

    def _request(self, inputs: list[str], *, truncate: bool) -> dict:
        endpoint = self.base_url.rstrip("/") + "/api/embed"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "model": self.model,
                "input": inputs,
                "truncate": truncate,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama embedding endpoint returned an invalid response")
        return payload

    def count_tokens(self, text: str) -> int:
        cached = self._token_counts.get(text)
        if cached is not None:
            return cached
        # Counting only needs to distinguish in-budget inputs from oversized
        # ones. Ollama reports the model limit for truncated oversized input,
        # which remains above this provider's smaller chunk budget.
        payload = self._request([text], truncate=True)
        count = payload.get("prompt_eval_count")
        if not isinstance(count, int) or count < 0:
            raise RuntimeError("Ollama embedding endpoint did not return prompt_eval_count")
        self._token_counts[text] = count
        return count

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self._request(texts, truncate=False)
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama embedding endpoint returned invalid vectors")
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise RuntimeError("Ollama embedding endpoint returned empty vectors")
        return [[float(value) for value in vector] for vector in vectors]


def build_embedding_provider(
    provider: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "AGENT_EMBEDDING_API_KEY",
    tokenizer_json: str | None = None,
    token_budget: int | None = None,
    use_environment: bool = True,
) -> EmbeddingProvider | None:
    """Build an optional provider without making an embedding request."""
    env = os.environ if use_environment else {}
    if provider == "none":
        return None
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model=model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    if provider == "ollama":
        resolved_model = model or env.get("AGENT_EMBEDDING_MODEL")
        if not resolved_model:
            raise ValueError("Ollama embeddings require a model")
        return OllamaEmbeddingProvider(
            model=resolved_model,
            base_url=base_url or env.get(
                "AGENT_EMBEDDING_BASE_URL", "http://127.0.0.1:11434"
            ),
            token_budget=token_budget or (
                int(env["AGENT_EMBEDDING_TOKEN_BUDGET"])
                if env.get("AGENT_EMBEDDING_TOKEN_BUDGET") else 256
            ),
        )
    if provider == "openai-compatible":
        resolved_url = base_url or env.get("AGENT_EMBEDDING_BASE_URL")
        api_key = os.getenv(api_key_env)
        resolved_model = model or env.get("AGENT_EMBEDDING_MODEL")
        missing = [
            name
            for name, value in (
                ("base URL", resolved_url),
                (f"environment variable {api_key_env}", api_key),
                ("model", resolved_model),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "openai-compatible embeddings require " + ", ".join(missing)
            )
        return OpenAICompatibleEmbeddingProvider(
            base_url=str(resolved_url),
            api_key=str(api_key),
            model=str(resolved_model),
            tokenizer_json=tokenizer_json or env.get("AGENT_EMBEDDING_TOKENIZER_JSON"),
            token_budget=token_budget or (
                int(env["AGENT_EMBEDDING_TOKEN_BUDGET"])
                if env.get("AGENT_EMBEDDING_TOKEN_BUDGET") else None
            ),
        )
    raise ValueError(f"unknown embedding provider: {provider}")
