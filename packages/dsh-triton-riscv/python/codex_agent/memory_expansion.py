"""Deterministic parent/stage/neighbor expansion after vector ranking, never before."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from sqlalchemy import func

from codex_agent.memory_lineage import case_payload, digest, source_binding
from codex_agent.storage.schema import memories, memory_indexes, memory_index_entries, memory_index_chunks

TaskStage = Literal["lookup", "generation", "diagnosis", "repair", "validation"]
STAGES = {
    "lookup": ("validation", "test_summary"),
    "generation": ("contract", "validation", "test_summary", "recommendation"),
    "diagnosis": ("error", "diagnosis", "environment", "validation", "test_summary"),
    "repair": ("error", "action", "patch", "validation", "test_summary", "recommendation", "diagnosis"),
    "validation": ("command", "validation", "test_summary", "error"),
}
SCHEMA = "three-layer-v1"


def resolve_stage(stage: TaskStage | None, *, diagnostic=False) -> TaskStage:
    value = stage or ("diagnosis" if diagnostic else "lookup")
    if value not in STAGES:
        raise ValueError("task_stage must be lookup, generation, diagnosis, repair or validation")
    return value


def _chunk_entry(item, row, layer):
    p = json.loads(row["provenance_json"])
    source = p.get("source") or {}
    if (p.get("case_id") != item["case_id"]
        or p.get("text_sha256") != hashlib.sha256(row["text"].encode()).hexdigest()
        or source != source_binding(item, p.get("source_field", ""))):
        return None
    return {**source, "evidence_id": "indexed:" + row["chunk_id"],
            "kind": "retrieved_chunk" if layer == 0 else "neighbor",
            "source": source.get("source") or item["source_run"],
            "pointer": source.get("pointer") or p["source_field"],
            "text": row["text"], "layer": layer, "chunk_id": row["chunk_id"],
            "case_version": row["case_version"], "part_index": p.get("part_index"),
            "part_count": p.get("part_count"), "overlap_may_repeat": layer == 3}


def _background(item):
    from codex_agent.memory_context import serialize
    keys = ("architecture", "execution_mode", "triton", "triton_version", "torch", "python", "llvm", "buddy")
    env = item.get("environment") or {}
    facts = {"semantics": item.get("semantics") or None, "pytorch_reference": item.get("pytorch_reference") or None,
             "environment": {k: env[k] for k in keys if env.get(k)},
             "missing_fields": [k for k in ("semantics", "pytorch_reference", "environment") if not item.get(k)]}
    omitted = []
    # Oversized fields remain source-addressable, never shown as complete definitions.
    for container, key in [(facts, "semantics"), (facts, "pytorch_reference"), *[(facts["environment"], k) for k in facts["environment"]]]:
        value = container[key]
        if len(serialize(value)) > 300:
            container[key] = {"omitted": True, "sha256": digest(value)}
            omitted.append(key)
    while len(serialize(facts)) > 1200 and facts["environment"]:
        key, _ = facts["environment"].popitem()
        omitted.append("environment." + key)
    if omitted:
        facts["omitted_fields"] = omitted
    return {"evidence_id": "background:" + str(item["id"]), "layer": 1, "kind": "background",
            "state": "recorded-fields-not-current-environment", "source": item["source_run"],
            "pointer": "/semantics,/pytorch_reference,/environment", "text": serialize(facts),
            "case_version": item["case_version"], "truncated": bool(omitted)}


def _task_entries(item, anchor, stage):
    from codex_agent.memory_view import entries
    from codex_agent.memory_context import _variants
    allowed = ("conflict", "gap", *STAGES[stage])
    candidates = [e for e in entries(item) if e.get("kind") in allowed]
    counts = {}
    ranked = []
    for entry in candidates:
        kind = entry["kind"]
        ordinal = counts.get(kind, 0)
        counts[kind] = ordinal + 1
        ranked.append((ordinal, allowed.index(kind), entry))
    candidates = [entry for _, _, entry in sorted(ranked, key=lambda pair: pair[:2])]
    chosen, seen, omitted = [], set(), 0
    source_id = (item["matched_evidence"].get("provenance") or {}).get("source", {}).get("evidence_id")
    for e in candidates:
        # The hit/its neighbors already represent this source field. Do not also paste the full parent log.
        if (source_id and e.get("evidence_id") == source_id) or (
                e.get("source") == anchor.get("source") and e.get("pointer") and (
                    e["pointer"] == anchor.get("pointer") or str(anchor.get("pointer", "")).startswith(e["pointer"] + "/"))):
            continue
        identity = digest({k: e.get(k) for k in ("kind", "state", "source", "pointer", "run_id", "proposal_id", "attempt", "text")})
        if identity in seen:
            continue
        seen.add(identity)
        if len(chosen) >= 8:
            omitted += 1
            continue
        shown = dict(e, layer=2, case_version=item["case_version"])
        if len(str(shown.get("text", ""))) > 1200:
            variants = [v for v in _variants(str(shown["text"]), shown["kind"]) if len(v["text"]) <= 1200]
            if not variants:
                omitted += 1
                continue
            shown.update(variants[-1], truncated=True)
        chosen.append(shown)
    return chosen, omitted


def expand_cases(store, items, task_stage: TaskStage, *, trace=None, allow_staging=False):
    """Only selected cases are read, under a short source-write fence; no network/model calls."""
    stage = resolve_stage(task_stage)
    trace = trace if trace is not None else {}
    stats = dict(schema=SCHEMA, stage=stage, rejected_cases=0, neighbors=0,
                 neighbor_unavailable=0, rejected_neighbors=0, max_neighbors_per_case=2)
    result, db = [], store.db
    with db.transaction() as conn:
        db.lock(conn, "memory-write")
        for item in items:
            r = item.get("retrieval") or {}
            if r.get("mode") != "qdrant":
                raise ValueError("three-layer expansion requires version-bound qdrant hits")
            parents = db.rows(memories, memories.c.id == item["id"], memories.c.active == 1,
                              memories.c.case_version == item["case_version"], connection=conn)
            indexes = db.rows(memory_indexes, memory_indexes.c.generation == r["generation"],
                              memory_indexes.c.fingerprint == r["index_fingerprint"],
                              memory_indexes.c.status.in_(["active", "staging"] if allow_staging else ["active"]), connection=conn)
            ready = db.rows(memory_index_entries, memory_index_entries.c.generation == r["generation"],
                            memory_index_entries.c.memory_id == item["id"],
                            memory_index_entries.c.revision == r["revision"],
                            memory_index_entries.c.case_version == item["case_version"],
                            memory_index_entries.c.status == "ready", memory_index_entries.c.desired == "present", connection=conn)
            if not parents or not indexes or not ready or digest(case_payload(store._row_dict(parents[0]))) != item["case_version"]:
                stats["rejected_cases"] += 1
                continue
            c = memory_index_chunks.c
            scope = (c.generation == r["generation"], c.memory_id == item["id"],
                     c.revision == r["revision"], c.case_version == item["case_version"])
            rows = db.rows(memory_index_chunks, *scope, c.chunk_id == r["best_chunk"], connection=conn)
            anchor = _chunk_entry(item, rows[0], 0) if rows else None
            if not anchor:
                stats["rejected_cases"] += 1
                continue
            p = json.loads(rows[0]["provenance_json"])
            candidates, omitted = _task_entries(item, anchor, stage)
            evidence = [anchor, _background(item), *candidates]
            i, count = p.get("part_index"), p.get("part_count")
            unavailable_before = stats["neighbor_unavailable"]
            if (type(i) is int and type(count) is int and 0 <= i < count
                    and all(p.get(k) is not None for k in ("source_field", "unit_sha256", "unit_position"))):
                if count > 1:
                    neighbor_indices = [n for n in (i - 1, i + 1) if 0 <= n < count]
                    json_value = lambda key: func.json_unquote(func.json_extract(c.provenance_json, "$." + key))
                    neighbors = db.rows(memory_index_chunks, *scope,
                        json_value("source_field") == p["source_field"],
                        json_value("unit_sha256") == p["unit_sha256"],
                        json_value("unit_position") == str(p["unit_position"]),
                        json_value("part_index").in_([str(n) for n in neighbor_indices]), limit=3, connection=conn)
                    by_part = {}
                    for row in neighbors:
                        np = json.loads(row["provenance_json"])
                        by_part.setdefault(np.get("part_index"), []).append(row)
                    for n in neighbor_indices:
                        matches = by_part.get(n, [])
                        row = matches[0] if len(matches) == 1 else None
                        np = json.loads(row["provenance_json"]) if row else {}
                        entry = _chunk_entry(item, row, 3) if row else None
                        if not entry or any(np.get(k) != p.get(k) for k in ("source", "policy", "kind", "part_count")):
                            stats["rejected_neighbors"] += len(matches)
                            stats["neighbor_unavailable"] += 1
                            continue
                        evidence.append(entry)
                        stats["neighbors"] += 1
            else:
                stats["neighbor_unavailable"] += 1
            chain = (item.get("evidence") or {}).get("chain") or {}
            result.append({**item, "expanded_evidence": {"items": evidence, "omitted_count": omitted,
                "remote_version_binding": chain.get("remote_version_binding", "unknown"),
                "gaps": chain.get("gaps", []), "conflicts": chain.get("conflicts", [])},
                "expansion": dict(schema=SCHEMA, stage=stage, case_version=item["case_version"],
                                  generation=r["generation"],
                                  neighbor_parts_included=sum(e["layer"] == 3 for e in evidence),
                                  neighbor_unavailable=stats["neighbor_unavailable"] - unavailable_before,
                                  layers=["background", "task-evidence", "same-unit-neighbors"])})
    trace["expansion"] = stats
    return result
