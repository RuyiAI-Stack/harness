"""Bounded vector recall; MySQL, not vector payloads, authorizes source hydration."""
import hashlib
import json
import math

from sqlalchemy import and_, select, tuple_
from dsh_vector_retrieval.validation import MAX_CANDIDATES, MAX_QUERY_TOKENS

from codex_agent.memory_lineage import case_payload, digest, SPLITTER_VERSION
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.schema import (
    memories, memory_indexes as indexes, memory_index_entries as entries,
    memory_index_chunks as chunks,
)
from .qdrant import Qdrant
from .worker import provider_for

class RetrievalUnavailable(RuntimeError):
    pass


def _index(db, generation, allow_staging):
    conditions = [indexes.c.generation == generation] if generation else [indexes.c.status == "active"]
    rows = db.rows(indexes, *conditions, limit=2)
    if len(rows) != 1 or rows[0]["status"] not in ({"active", "staging"} if allow_staging else {"active"}):
        raise RetrievalUnavailable("No eligible index generation; prepare, backfill and activate first")
    row = rows[0]
    definition = json.loads(row["config_json"])
    if digest(definition) != row["fingerprint"] or definition["splitter"] != SPLITTER_VERSION:
        raise RetrievalUnavailable("Index definition mismatch; rebuild the generation")
    return row, definition


def _hydrate(store, index, hits, excluded, trace):
    db = store.db
    valid = []
    for hit in hits:
        p = hit.get("payload", {}) if isinstance(hit, dict) else {}
        score = hit.get("score") if isinstance(hit, dict) else None
        if (not isinstance(p, dict) or type(p.get("memory_id")) is not int
                or not 0 < p["memory_id"] < 2**63
                or type(p.get("revision")) is not int or not 0 < p["revision"] < 2**63
                or not isinstance(p.get("chunk_id"), str)
                or not isinstance(p.get("case_version"), str)
                or not isinstance(hit.get("id"), str)
                or type(score) not in {int, float} or not math.isfinite(score)
                or p.get("workspace_id") != db.workspace_id
                or p.get("generation") != index["generation"]
                or p.get("index_fingerprint") != index["fingerprint"]):
            trace["rejected"] += 1
            continue
        valid.append(hit)
    if not valid:
        return []
    # Point candidates supply only lookup keys. Every join is scoped to this workspace.
    join = chunks.join(entries, and_(
        entries.c.workspace_id == chunks.c.workspace_id, entries.c.generation == chunks.c.generation,
        entries.c.memory_id == chunks.c.memory_id, entries.c.revision == chunks.c.revision,
        entries.c.case_version == chunks.c.case_version,
    )).join(memories, and_(memories.c.workspace_id == chunks.c.workspace_id,
        memories.c.id == chunks.c.memory_id, memories.c.case_version == chunks.c.case_version))
    stmt = select(memories, chunks.c.chunk_id, chunks.c.point_id, chunks.c.revision,
                  chunks.c.text.label("chunk_text"), chunks.c.provenance_json).select_from(join).where(
        *db.predicate(chunks, chunks.c.generation == index["generation"]),
        entries.c.desired == "present", entries.c.status == "ready", memories.c.active == 1,
        tuple_(chunks.c.memory_id, chunks.c.revision, chunks.c.chunk_id).in_([
            (h["payload"]["memory_id"], h["payload"]["revision"], h["payload"]["chunk_id"]) for h in valid]))
    with db.transaction() as conn:
        rows = {r["point_id"]: dict(r) for r in conn.execute(stmt).mappings()}
    items = []
    for hit in valid:
        row = rows.get(hit["id"])
        p = hit["payload"]
        if row is None or any(row[k] != p[k] for k in ("revision", "chunk_id", "case_version")) or row["id"] != p["memory_id"]:
            trace["rejected"] += 1
            continue
        item = store._row_dict({c.name: row[c.name] for c in memories.columns})
        provenance = json.loads(row["provenance_json"])
        if (digest(case_payload(item)) != row["case_version"]
                or provenance.get("case_id") != item["case_id"]
                or provenance.get("text_sha256") != hashlib.sha256(row["chunk_text"].encode()).hexdigest()):
            trace["rejected"] += 1
            continue
        if item["source_run"] in excluded:
            trace["excluded"] += 1
            continue
        item["matched_evidence"] = {
            "text": row["chunk_text"], "kind": provenance.get("kind", "indexed-evidence"),
            "source_field": provenance["source_field"], "source_run": item["source_run"],
            "case_id": item["case_id"], "case_version": row["case_version"],
            "chunk_id": row["chunk_id"], "provenance": provenance,
        }
        item["retrieval"] = {"mode": "qdrant", "score": hit["score"], "semantic_score": hit["score"],
            "generation": index["generation"], "revision": row["revision"],
            "index_fingerprint": index["fingerprint"], "best_chunk": row["chunk_id"]}
        items.append(item)
    return items


def retrieve(store, query, limit=5, *, exclude_source_runs=(), trace=None, generation=None,
             allow_staging=False, provider_factory=None, vector_factory=None):
    """Return distinct cases from at most 200 vector hits; never scan all source rows."""
    from codex_agent.memory import normalize_text
    if not runtime_config().indexing.enabled:
        raise RetrievalUnavailable("Enable indexing before selecting qdrant retrieval")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("qdrant limit must be between 1 and 50")
    if allow_staging and not generation:
        raise ValueError("preview requires an explicit generation")
    trace = trace if trace is not None else {}
    trace.update(mode="qdrant", scanned=0, rejected=0, excluded=0, deduplicated=0,
                 candidate_budget=MAX_CANDIDATES, budget_exhausted=False, cache="bypassed")
    index, definition = _index(store.db, generation, allow_staging)
    trace["generation"] = index["generation"]
    provider = (provider_factory or provider_for)(definition)
    if provider is None or (provider.name, provider.model) != (
            definition["embedding"]["provider"], definition["embedding"]["model"]):
        raise RetrievalUnavailable("Query embedding provider does not match indexed documents")
    text = query.searchable_text().strip()
    if not text:
        return []
    count = provider.count_tokens(text)
    if count > min(MAX_QUERY_TOKENS, provider.max_input_tokens):
        raise ValueError("Query exceeds embedding token budget; provide focused task/error evidence")
    trace["query_tokens"] = count
    vectors = provider.embed([text])
    if (len(vectors) != 1 or len(vectors[0]) != definition["dimensions"]
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in vectors[0]) or not any(vectors[0])):
        raise RetrievalUnavailable("Invalid query embedding; no vector search performed")
    service = (vector_factory or Qdrant)(definition, index["collection"])
    excluded = {normalize_text(s) for s in exclude_source_runs if s}
    found, point_hits = {}, {}
    while trace["scanned"] < MAX_CANDIDATES:
        size = min(max(20, limit * 4), 100, MAX_CANDIDATES - trace["scanned"])
        hits = service.query(vectors[0], store.db.workspace_id, index["generation"], index["fingerprint"],
                             limit=size, offset=trace["scanned"])
        trace["scanned"] += len(hits)
        point_hits.update({h["id"]: h for h in hits if isinstance(h, dict) and isinstance(h.get("id"), str)})
        for item in _hydrate(store, index, hits, excluded, trace):
            old = found.get(item["id"])
            if old:
                trace["deduplicated"] += 1
            if not old or item["retrieval"]["score"] > old["retrieval"]["score"]:
                found[item["id"]] = item
        if len(found) >= limit or len(hits) < size:
            break
    # A generation switch during network/model work must not silently mix indexes.
    latest, _ = _index(store.db, generation, allow_staging)
    if (latest["generation"], latest["fingerprint"]) != (index["generation"], index["fingerprint"]):
        raise RetrievalUnavailable("Index changed during retrieval; retry the query")
    # Earlier pages may have been superseded while fetching later pages.
    winning_chunks = {i["matched_evidence"]["chunk_id"] for i in found.values()}
    final_hits = [h for h in point_hits.values() if isinstance(h.get("payload"), dict)
                  and h["payload"].get("chunk_id") in winning_chunks]
    found = {i["id"]: i for i in _hydrate(store, index, final_hits, excluded, trace)}
    trace["budget_exhausted"] = trace["scanned"] == MAX_CANDIDATES and len(found) < limit
    return sorted(found.values(), key=lambda i: (-i["retrieval"]["score"], i["id"]))[:limit]
