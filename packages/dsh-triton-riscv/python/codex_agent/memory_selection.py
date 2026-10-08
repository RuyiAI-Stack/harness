"""Optional record selection after scoring; buckets never imply a repair chain."""
from __future__ import annotations

import json
from typing import Literal

CandidateStrategy = Literal["record-top5", "evidence-diverse-v1"]
CANDIDATE_STRATEGIES = {"record-top5", "evidence-diverse-v1"}
_ROLES = {"error", "diagnosis", "recommendation", "action", "patch",
          "validation", "test_summary", "rationale", "conflict", "gap"}


def diversity_bucket(record: dict) -> tuple | None:
    """Use exact public metadata, not evaluator families or inferred causality."""
    operator, kind, outcome = (record.get(k) for k in ("operator", "memory_type", "outcome"))
    environment = record.get("environment")
    if not all(isinstance(v, str) and v.strip() for v in (operator, kind, outcome)):
        return None
    if not isinstance(environment, dict) or not any(v not in (None, "") for v in environment.values()):
        return None
    stage, signature = record.get("failure_stage"), record.get("error_signature")
    if outcome != "passed" and (not stage or not signature):
        return None
    return (operator, kind, outcome, stage, signature,
            json.dumps(environment, sort_keys=True, ensure_ascii=False))


def evidence_roles(record: dict) -> set[tuple[str, str]]:
    evidence = record.get("evidence") or {}
    chain = evidence.get("chain") or {}
    return {(e["kind"], str(e.get("state") or "unknown"))
            for e in chain.get("items", []) if isinstance(e, dict) and e.get("kind") in _ROLES}


def select_candidates(
    candidates: list[dict], limit: int = 5, *,
    strategy: CandidateStrategy = "record-top5", trace: dict | None = None,
) -> list[dict]:
    """Select at most limit unchanged records from an already ranked candidate pool."""
    if strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(f"unsupported candidate strategy: {strategy}")
    limit = max(0, limit)
    if strategy == "record-top5" and trace is None:
        return candidates[:limit]
    decisions = []
    selected_indices = []
    deferred = []
    seen: dict[tuple, set[tuple[str, str]]] = {}
    for index, record in enumerate(candidates):
        bucket = diversity_bucket(record) if strategy != "record-top5" else None
        roles = evidence_roles(record) if strategy != "record-top5" else set()
        new_roles = roles - seen.get(bucket, set()) if bucket is not None else roles
        if len(selected_indices) >= limit:
            reason = "budget-full"
        elif strategy == "record-top5" or len(candidates) <= limit:
            selected_indices.append(index)
            reason = "legacy-order"
        elif bucket is None:
            selected_indices.append(index)
            reason = "missing-group-fields-keep-order"
        elif bucket not in seen or new_roles:
            selected_indices.append(index)
            reason = "first-in-heuristic-bucket" if bucket not in seen else "additional-evidence-role"
            seen.setdefault(bucket, set()).update(roles)
        else:
            deferred.append(index)
            reason = "deferred-repeated-roles"
        decisions.append({"rank": index + 1, "record_id": record["id"],
                          "bucket": bucket, "roles": sorted(roles),
                          "additional_roles": sorted(new_roles), "reason": reason})
    for index in deferred[:max(0, limit - len(selected_indices))]:
        selected_indices.append(index)
        decisions[index]["reason"] = "backfill-original-order"
    selected_indices.sort()
    if trace is not None:
        trace.update(strategy=strategy, candidate_count=len(candidates), limit=limit,
                     decisions=decisions, selected_ids=[candidates[i]["id"] for i in selected_indices],
                     grouping="metadata-heuristic-not-causal", source_records_merged=False)
    return [candidates[index] for index in selected_indices]
