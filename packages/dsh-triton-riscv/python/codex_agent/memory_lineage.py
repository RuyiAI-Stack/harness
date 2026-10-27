"""Versioned case/chunk identities; evidence labels are copied, never promoted."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re

LINEAGE_SCHEMA = "case-chunk-v1"
SPLITTER_VERSION = "structural-v1"
CASE_FIELDS = (
    "memory_type", "operator", "semantics", "pytorch_reference", "summary", "outcome",
    "confidence_grade", "source_run", "tl_ops", "failure_stage", "error_signature",
    "environment", "evidence", "test_sha256",
)


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def case_payload(item):
    # Counters, timestamps, archiving and derived vectors do not change source content.
    return {key: item.get(key) for key in CASE_FIELDS}


def case_identity(workspace_id, memory_id):
    return digest(["case", workspace_id, memory_id])


def policy_metadata(policy, provider=None):
    return {"splitter": SPLITTER_VERSION, **asdict(policy),
            "counting": "provider-tokenizer" if provider else "characters-only",
            "provider": provider.name if provider else None,
            "model": provider.model if provider else None}


def source_binding(item, field):
    evidence = item.get("evidence") or {}
    chain = evidence.get("chain") or {}
    matches = [entry for entry in chain.get("items", [])
               if field == "evidence.chain." + str(entry.get("evidence_id", "unknown"))]
    if len(matches) == 1:
        entry = matches[0]
        keys = ("evidence_id", "kind", "state", "source", "pointer", "lines", "run_id",
                "proposal_id", "attempt", "relation", "source_sha256", "content_sha256",
                "value_sha256", "available_at", "time_basis", "truncated")
        return {"binding": "explicit-evidence-entry", **{key: entry.get(key) for key in keys}}
    # A source_run can be a directory, not a file; do not invent an artifact path/run ID.
    pointer = "/" + re.sub(r"\[(\d+)\]", r"/\1", field.replace(".", "/"))
    state = "recorded-field"
    if field.startswith("evidence.recommended_actions["):
        state = "not-executed"
    elif field in {"evidence.patch_excerpt", "evidence.attempted_action"}:
        state = "application-unknown"
    elif field in {"outcome", "evidence.test_summary", "evidence.correctness"}:
        state = "reported-result"
    elif field == "summary" and item.get("outcome") != "passed":
        state = "reported-not-proven"
    return {"binding": "ambiguous-evidence-entry" if matches else "record-field",
            "source": None, "pointer": None if matches else pointer,
            "source_run": item.get("source_run") or None, "state": state,
            "run_id": None, "proposal_id": None, "attempt": None,
            "source_sha256": None, "available_at": None, "time_basis": "unknown"}


def chunk_lineage(workspace_id, memory_id, version, item, chunk, policy):
    case_id = case_identity(workspace_id, memory_id)
    source = source_binding(item, chunk.source_field)
    unit_id = digest([case_id, version, chunk.source_field, chunk.unit_position,
                      chunk.unit_sha256, source])
    chunk_id = digest([unit_id, policy, chunk.part_index, chunk.text])
    provenance = {"schema": LINEAGE_SCHEMA, "case_id": case_id, "source_field": chunk.source_field,
                  "unit_position": chunk.unit_position, "unit_sha256": chunk.unit_sha256,
                  "unit_scope": "normalized-field-segment-not-original-file-offset",
                  "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                  "policy": policy, "source": source}
    return dict(case_version=version, chunk_id=chunk_id, unit_id=unit_id,
                part_index=chunk.part_index, part_count=chunk.part_count,
                provenance_json=encode(provenance))
