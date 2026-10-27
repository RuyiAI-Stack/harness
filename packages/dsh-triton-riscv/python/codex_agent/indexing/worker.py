"""Replay-safe vector writes followed by a lease- and version-fenced SQL checkpoint."""
import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

from dsh_vector_retrieval.embeddings import build_embedding_provider
from codex_agent.memory_chunking import ChunkPolicy, case_chunk_parts, embedding_input
from codex_agent.memory_lineage import chunk_lineage, policy_metadata, digest, SPLITTER_VERSION
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.schema import (memories, memory_versions, memory_indexes as indexes,
    memory_index_entries as entries, memory_index_chunks as chunks)
from codex_agent.tasks.store import digest as task_digest
from .qdrant import Qdrant


def provider_for(definition):
    e = definition["embedding"]
    path = e["tokenizerJson"]
    if path and hashlib.sha256(Path(path).read_bytes()).hexdigest() != definition["tokenizer_sha256"]:
        raise ValueError("tokenizer changed; create a new index generation")
    return build_embedding_provider(e["provider"], model=e["model"], base_url=e["baseUrl"],
        api_key_env=e["apiKeyEnv"], tokenizer_json=path, token_budget=e["tokenBudget"], use_environment=False,
        revision=definition["modelRevision"])


def current(db, conn, payload, identifier):
    rows = db.rows(entries, entries.c.generation == payload["generation"],
                   entries.c.memory_id == payload["memory_id"], connection=conn)
    if (not rows or rows[0]["job_id"] != identifier
            or any(rows[0][key] != payload[key] for key in ("revision", "case_version", "desired"))):
        return False
    parents = db.rows(memories, memories.c.id == payload["memory_id"], connection=conn)
    if payload["desired"] == "deleted":
        return not parents or not parents[0]["active"]
    return bool(parents and parents[0]["active"] and parents[0]["case_version"] == payload["case_version"])


def execute_index(jobs, job, owner, *, provider_factory=provider_for, vector_factory=Qdrant):
    if not runtime_config().indexing.enabled:
        raise PermissionError("indexing is disabled")
    db, identifier = jobs.db, job["id"]
    with db.transaction() as conn:
        job = jobs.require_owner(conn, identifier, owner)
        if job["kind"] != "index":
            raise ValueError("only an index job can write vectors")
    payload = json.loads(job["payload_json"])
    if job["idempotency_key"] != task_digest("index:" + task_digest(payload)):
        raise ValueError("job is not bound to an indexing request")
    generation, memory_id, revision = (payload[k] for k in ("generation", "memory_id", "revision"))
    with db.transaction() as conn:
        jobs.require_owner(conn, identifier, owner)
        index = db.rows(indexes, indexes.c.generation == generation, connection=conn)[0]
        if index["fingerprint"] != payload["index_fingerprint"]:
            raise ValueError("index definition fingerprint mismatch")
        live = current(db, conn, payload, identifier) and (
            index["status"] != "retired" or payload["desired"] == "deleted")
        versions = db.rows(memory_versions, memory_versions.c.memory_id == memory_id,
            memory_versions.c.case_version == payload["case_version"], connection=conn) if live and payload["desired"] == "present" else []
    definition = json.loads(index["config_json"])
    if definition["splitter"] != SPLITTER_VERSION:
        raise ValueError("splitter implementation changed; build a new index generation")
    vectors = vector_factory(definition, index["collection"])
    jobs.begin_effects(identifier, owner)
    records = []
    if live and payload["desired"] == "present":
        if not versions:
            raise ValueError("immutable source snapshot is missing")
        source = json.loads(versions[0]["payload_json"])
        if digest(source) != payload["case_version"]:
            raise ValueError("immutable source snapshot checksum mismatch")
        provider = provider_factory(definition)
        if provider is None or (provider.name, provider.model) != (
                definition["embedding"]["provider"], definition["embedding"]["model"]):
            raise ValueError("embedding provider identity mismatch")
        limit = min(definition["maxTokens"], provider.max_input_tokens)
        policy = ChunkPolicy(max_chars=definition["maxChars"], max_tokens=limit,
                             overlap_tokens=min(definition["overlapTokens"], limit - 1))
        parts = case_chunk_parts(source, token_count=provider.count_tokens, policy=policy)
        policy_data = {**policy_metadata(policy, provider), "model_revision": definition["modelRevision"],
                       "tokenizer_sha256": definition["tokenizer_sha256"]}
        vectors.ensure()
        for offset in range(0, len(parts), definition["batchSize"]):
            jobs.begin_effects(identifier, owner)
            batch = parts[offset:offset + definition["batchSize"]]
            texts = [embedding_input(source.get("operator") or "", part) for part in batch]
            if any(provider.count_tokens(text) > limit for text in texts):
                raise ValueError("indexed input exceeds token budget including title")
            embedded = provider.embed(texts)
            if len(embedded) != len(texts):
                raise ValueError("embedding count mismatch")
            points = []
            for part, vector in zip(batch, embedded):
                if (len(vector) != definition["dimensions"]
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)
                        or not any(vector)):
                    raise ValueError("invalid embedding dimension, value or zero norm")
                lineage = chunk_lineage(db.workspace_id, memory_id, payload["case_version"], source, part, policy_data)
                provenance = json.loads(lineage["provenance_json"])
                provenance.update(kind=part.kind, position=part.position,
                                  part_index=part.part_index, part_count=part.part_count)
                lineage["provenance_json"] = json.dumps(provenance, sort_keys=True)
                point_id = str(UUID(digest([db.workspace_id, generation, memory_id, revision, lineage["chunk_id"]])[:32]))
                records.append(dict(generation=generation, memory_id=memory_id, revision=revision,
                    chunk_id=lineage["chunk_id"], point_id=point_id, case_version=payload["case_version"],
                    text=part.text, provenance_json=lineage["provenance_json"]))
                points.append({"id": point_id, "vector": vector, "payload": {
                    "workspace_id": db.workspace_id, "generation": generation, "memory_id": memory_id,
                    "revision": revision, "case_version": payload["case_version"],
                    "chunk_id": lineage["chunk_id"], "unit_id": lineage["unit_id"],
                    "kind": part.kind, "position": part.position, "source_field": part.source_field,
                    "model_revision": definition["modelRevision"], "index_fingerprint": index["fingerprint"]}})
            vectors.upsert(points)
        vectors.delete_revision(db.workspace_id, memory_id, revision, older=True)
    elif live:
        vectors.delete_revision(db.workspace_id, memory_id, revision, older=True, include=True)
    else:
        vectors.delete_revision(db.workspace_id, memory_id, revision)

    with db.transaction() as conn:
        db.lock(conn, "memory-write")
        jobs.require_owner(conn, identifier, owner)
        valid = live and current(db, conn, payload, identifier)
        if valid:
            # Deterministic IDs make redelivery an overwrite, never a second set of points.
            db.delete(conn, chunks, chunks.c.generation == generation, chunks.c.memory_id == memory_id,
                      chunks.c.revision == revision)
            for record in records:
                db.insert(conn, chunks, record)
            db.update(conn, entries, {"status": "ready" if payload["desired"] == "present" else "deleted",
                                     "point_count": len(records)},
                      entries.c.generation == generation, entries.c.memory_id == memory_id)
            jobs.finish(identifier, owner, "succeeded", {"index_status": "ready" if records else payload["desired"],
                        "points": len(records), "generation": generation, "revision": revision}, connection=conn)
            return
    # Newer content won while embedding was running. Delete only this obsolete revision.
    vectors.delete_revision(db.workspace_id, memory_id, revision)
    jobs.finish(identifier, owner, "succeeded", {"index_status": "superseded", "generation": generation,
                                               "revision": revision, "points": 0})
