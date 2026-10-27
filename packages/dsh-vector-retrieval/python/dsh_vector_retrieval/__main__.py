"""Single-request JSON bridge. Configuration comes only from the host environment."""
import contextlib
import json
import os
import re
import sys
from urllib.parse import urlsplit

from . import DocumentChunk, Qdrant, Scope, VectorIndex, build_embedding_provider

CONFIG_ENV = "DSH_VECTOR_RETRIEVAL_CONFIG"
MAX_BYTES = 2_000_000


def keys(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("invalid fields")
    return value


def configuration(value):
    keys(value, {"workspaceId", "generation", "fingerprint", "collection", "qdrantUrl",
                 "dimensions", "timeoutSeconds", "apiKeyEnv", "embedding"})
    Scope(value["workspaceId"], value["generation"], value["fingerprint"])
    if type(value["dimensions"]) is not int or not 1 <= value["dimensions"] <= 65536:
        raise ValueError("invalid dimensions")
    timeout = value.get("timeoutSeconds", 30)
    if type(timeout) not in {int, float} or not 0 < timeout <= 300:
        raise ValueError("invalid timeout")
    embedding = keys(value.get("embedding", {}), {"provider", "model", "baseUrl", "apiKeyEnv",
        "tokenizerJson", "tokenBudget", "revision"})
    for name in (value.get("apiKeyEnv", "DSH_VECTOR_API_KEY"),
                 embedding.get("apiKeyEnv", "DSH_EMBEDDING_API_KEY")):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("invalid credential environment name")
    if embedding.get("baseUrl"):
        url = urlsplit(embedding["baseUrl"])
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("invalid embedding URL")
    return {**value, "timeoutSeconds": timeout, "apiKeyEnv": value.get("apiKeyEnv", "DSH_VECTOR_API_KEY")}


def provider_for(config):
    e = config.get("embedding", {})
    return build_embedding_provider(e.get("provider", "none"), model=e.get("model"),
        base_url=e.get("baseUrl"), api_key_env=e.get("apiKeyEnv", "DSH_EMBEDDING_API_KEY"),
        tokenizer_json=e.get("tokenizerJson"), token_budget=e.get("tokenBudget"),
        revision=e.get("revision"), use_environment=False)


def dispatch(config, request):
    config = configuration(config)
    transport = Qdrant(config, config["collection"])
    index = VectorIndex(transport, Scope(config["workspaceId"], config["generation"], config["fingerprint"]))
    if not isinstance(request, dict):
        raise ValueError("invalid request")
    action = request.get("action")
    if action == "search":
        keys(request, {"action", "vector", "text", "limit", "offset"})
        if ("text" in request) == ("vector" in request):
            raise ValueError("provide text or vector")
        options = dict(limit=request.get("limit", 20), offset=request.get("offset", 0))
        if "vector" in request:
            return index.search(request["vector"], **options)
        return index.search_text(request["text"], provider_for(config), **options)
    if action == "index":
        keys(request, {"action", "chunks", "vectors"})
        chunks = [DocumentChunk(**chunk) for chunk in request["chunks"]]
        return index.index_chunks(chunks, vectors=request.get("vectors"),
            provider=provider_for(config) if request.get("vectors") is None else None)
    if action == "deleteVersion":
        keys(request, {"action", "documentId", "version", "older", "include"})
        if any(type(request.get(k, False)) is not bool for k in ("older", "include")):
            raise ValueError("invalid deletion mode")
        index.delete_version(request["documentId"], request["version"],
            older=request.get("older", False), include=request.get("include", False))
        return {"completed": True}
    if action == "embed":
        keys(request, {"action", "texts"})
        return index.embed(request["texts"], provider_for(config))
    raise ValueError("unsupported action")


def main():
    try:
        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("request too large")
        # Model libraries may write diagnostics to stdout; the protocol is JSON only.
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(json.loads(os.environ[CONFIG_ENV]), json.loads(data))
        output = json.dumps({"result": result}, allow_nan=False)
        if len(output.encode()) > MAX_BYTES:
            raise ValueError("response too large")
        print(output)
        return 0
    except Exception:
        # Never relay provider bodies, document text, config, paths or credentials.
        print(json.dumps({"error": "Vector retrieval operation failed"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
