"""Bounded Qdrant REST adapter. No credentials or document bodies in errors."""
import json
import os
import urllib.error
import urllib.request


class VectorServiceError(RuntimeError):
    pass


class Qdrant:
    def __init__(self, definition, collection):
        self.url = definition["qdrantUrl"].rstrip("/") + "/collections/" + collection
        self.timeout = definition["timeoutSeconds"]
        self.dimensions = definition["dimensions"]
        self.key = os.environ.get(definition["apiKeyEnv"], "")

    def request(self, method, path="", value=None, *, missing_ok=False):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["api-key"] = self.key
        request = urllib.request.Request(self.url + path, method=method, headers=headers,
            data=None if value is None else json.dumps(value, allow_nan=False).encode())
        try:
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
        vectors = current["config"]["params"]["vectors"]
        if vectors.get("size") != self.dimensions or vectors.get("distance") != "Cosine":
            raise VectorServiceError("Qdrant collection dimensions or distance mismatch")

    def upsert(self, points):
        if points:
            result = self.request("PUT", "/points?wait=true", {"points": points})
            if result.get("status") != "completed":
                raise VectorServiceError("Qdrant upsert is not completed")

    def query(self, vector, workspace, generation, fingerprint, *, limit, offset=0):
        if not 1 <= limit <= 100 or not 0 <= offset < 200:
            raise ValueError("vector query exceeds candidate budget")
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

    def delete_revision(self, workspace, memory_id, revision, *, older=False, include=False):
        condition = {"key": "revision", "range": {"lte" if include else "lt": revision}} if older else {
            "key": "revision", "match": {"value": revision}}
        result = self.request("POST", "/points/delete?wait=true", {"filter": {"must": [
            {"key": "workspace_id", "match": {"value": workspace}},
            {"key": "memory_id", "match": {"value": memory_id}}, condition]}}, missing_ok=True)
        if result is not None and result.get("status") != "completed":
            raise VectorServiceError("Qdrant deletion is not completed")
