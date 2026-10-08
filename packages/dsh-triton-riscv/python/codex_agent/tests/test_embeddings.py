from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from codex_agent.embeddings import (
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
)


class EmbeddingTests(unittest.TestCase):
    def test_cosine_similarity_handles_matches_and_invalid_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_builds_optional_openai_compatible_provider_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TEST_EMBEDDING_KEY": "secret",
                "AGENT_EMBEDDING_MODEL": "company-embedding-v1",
            },
            clear=False,
        ):
            provider = build_embedding_provider(
                "openai-compatible",
                base_url="https://models.example/v1",
                api_key_env="TEST_EMBEDDING_KEY",
            )

        self.assertIsInstance(provider, OpenAICompatibleEmbeddingProvider)
        self.assertEqual(provider.model, "company-embedding-v1")
        self.assertIsNone(build_embedding_provider("none"))

    def test_openai_compatible_provider_validates_and_orders_response(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        ).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        context.__exit__.return_value = False
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="https://models.example/v1",
            api_key="secret",
            model="embedding-v1",
        )

        with patch("codex_agent.embeddings.urllib.request.urlopen", return_value=context):
            vectors = provider.embed(["first", "second"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])

    def test_ollama_provider_uses_native_token_count_and_vectors(self) -> None:
        token_response = MagicMock()
        token_response.read.return_value = json.dumps({
            "embeddings": [[1.0, 0.0]],
            "prompt_eval_count": 7,
        }).encode()
        vector_response = MagicMock()
        vector_response.read.return_value = json.dumps({
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "prompt_eval_count": 12,
        }).encode()
        for response in (token_response, vector_response):
            response.__enter__.return_value = response
            response.__exit__.return_value = False
        provider = OllamaEmbeddingProvider(model="all-minilm:v2")

        with patch(
            "codex_agent.embeddings.urllib.request.urlopen",
            side_effect=[token_response, vector_response],
        ):
            self.assertEqual(provider.count_tokens("hello"), 7)
            self.assertEqual(provider.count_tokens("hello"), 7)
            self.assertEqual(
                provider.embed(["first", "second"]),
                [[1.0, 0.0], [0.0, 1.0]],
            )

    def test_builds_ollama_provider_without_api_key(self) -> None:
        provider = build_embedding_provider(
            "ollama",
            model="all-minilm:v2",
            base_url="http://127.0.0.1:11434",
            token_budget=256,
        )
        self.assertIsInstance(provider, OllamaEmbeddingProvider)
        self.assertEqual(provider.max_input_tokens, 256)

    def test_rejects_incomplete_or_unknown_provider_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                build_embedding_provider("openai-compatible")
        with self.assertRaises(ValueError):
            build_embedding_provider("not-a-provider")


if __name__ == "__main__":
    unittest.main()
