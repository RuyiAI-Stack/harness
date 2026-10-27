"""Bounded Qdrant REST adapter. No credentials or document bodies in errors."""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

from .validation import page_budget, required_text


@dataclass(frozen=True)
class PayloadFields:
    document: str = "document_id"
    version: str = "version"


class VectorServiceError(RuntimeError):
    pass


class Qdrant:
    def __init__(self, definition, collection, *, fields=PayloadFields()):
        import re
        if not isinstance(collection, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", collection):
            raise ValueError("invalid vector collection")
        url = urlsplit(definition["qdrantUrl"])
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("invalid vector service URL")
        self.url = definition["qdrantUrl"].rstrip("/") + "/collections/" + collection
        self.timeout = definition["timeoutSeconds"]
        self.dimensions = definition["dimensions"]
        self.key = os.environ.get(definition["apiKeyEnv"], "")
        self.fields = fields

    def request(self, method, path="", value=None, *, missing_ok=False):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["api-key"] = self.key
        try:
            request = urllib.request.Request(self.url + path, method=method, headers=headers,
                data=None if value is None else json.dumps(value, allow_nan=False).encode())
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read(8 * 1024 * 1024))
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                raise VectorServiceError("Qdrant did not confirm the operation")
            return payload.get("result")
        except urllib.error.HTTPError as error:
            if error.code == 404 and missing_ok:
                return None
            raise VectorServiceError(f"Qdrant HTTP {error.code}") from None
        except (OSError, ValueError):
            raise VectorServiceError("Qdrant connection or response failed") from None

    def ensure(self):
        current = self.request("GET", missing_ok=True)
        if current is None:
            try:
                self.request("PUT", value={"vectors": {"size": self.dimensions, "distance": "Cosine"}})
            except VectorServiceError:
                # Another worker can win collection creation; verify its schema, never overwrite it.
                if self.request("GET", missing_ok=True) is None:
                    raise
            current = self.request("GET")
        try:
            vectors = current["config"]["params"]["vectors"]
        except (KeyError, TypeError):
            raise VectorServiceError("Qdrant returned invalid collection metadata") from None
        if not isinstance(vectors, dict):
            raise VectorServiceError("Qdrant returned invalid collection metadata")
        if vectors.get("size") != self.dimensions or vectors.get("distance") != "Cosine":
            raise VectorServiceError("Qdrant collection dimensions or distance mismatch")

    def upsert(self, points):
        if points:
            result = self.request("PUT", "/points?wait=true", {"points": points})
            if not isinstance(result, dict) or result.get("status") != "completed":
                raise VectorServiceError("Qdrant upsert is not completed")

    def query(self, vector, workspace, generation, fingerprint, *, limit, offset=0):
        page_budget(limit, offset)
        required_text(workspace, "workspace")
        required_text(generation, "generation")
        required_text(fingerprint, "fingerprint")
        result = self.request("POST", "/points/query", {
            "query": vector, "limit": limit, "offset": offset,
            "with_vector": False, "with_payload": True,
            "filter": {"must": [
                {"key": key, "match": {"value": value}}
                for key, value in (("workspace_id", workspace), ("generation", generation),
                                   ("index_fingerprint", fingerprint))]},
        })
        if not isinstance(result, dict) or not isinstance(result.get("points"), list):
            raise VectorServiceError("Qdrant returned invalid candidates")
        if len(result["points"]) > limit:
            raise VectorServiceError("Qdrant exceeded requested candidate budget")
        return result["points"]

    def delete_version(self, workspace, document_id, version, *, older=False, include=False,
                       generation=None, fingerprint=None):
        required_text(workspace, "workspace")
        if (type(document_id) not in {str, int} or not str(document_id).strip()
                or type(version) is not int or version < 1):
            raise ValueError("document and positive version are required")
        condition = {"key": self.fields.version, "range": {"lte" if include else "lt": version}} if older else {
            "key": self.fields.version, "match": {"value": version}}
        scope = []
        if generation is not None or fingerprint is not None:
            required_text(generation, "generation")
            required_text(fingerprint, "fingerprint")
            scope = [{"key": key, "match": {"value": value}} for key, value in
                     (("generation", generation), ("index_fingerprint", fingerprint))]
        result = self.request("POST", "/points/delete?wait=true", {"filter": {"must": [
            {"key": "workspace_id", "match": {"value": workspace}},
            {"key": self.fields.document, "match": {"value": document_id}}, condition, *scope]}}, missing_ok=True)
        if result is not None and (not isinstance(result, dict) or result.get("status") != "completed"):
            raise VectorServiceError("Qdrant deletion is not completed")
