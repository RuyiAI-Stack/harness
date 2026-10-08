"""Bridge structured validation receipts into evidence-gated memory."""

from __future__ import annotations

from .memory_selection import CandidateStrategy
from codex_agent.runtime_config import CONFIG_ENV, runtime_config

import os
from pathlib import Path
from typing import Any

from codex_agent.embeddings import build_embedding_provider
from codex_agent.memory_evidence import (
    EvidenceSource, assemble_receipt_evidence, collect_lifecycle_sources,
)
from codex_agent.memory_view import bounded_chain, clip, encode, PUBLIC_ITEM_CHARS
from codex_agent.memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryStore,
    normalized_error_signature,
)


MEMORY_DB = Path("agent-results/memory.sqlite3")
MAX_MEMORY_RESULTS = 10


def memory_database_path(repo_root: Path) -> Path:
    from codex_agent.paths import state_root
    value = runtime_config().memory.database
    if value is None:
        return state_root(repo_root) / "memory.sqlite3"
    configured = Path(value)
    if configured.is_absolute():
        return configured
    return repo_root.resolve() / configured


def build_configured_embedding_provider():
    """Build an optional provider; lexical retrieval remains the default."""

    settings = runtime_config().memory.embedding
    return build_embedding_provider(
        settings.provider,
        model=settings.model or None,
        base_url=settings.baseUrl or None,
        api_key_env=settings.apiKeyEnv,
        tokenizer_json=settings.tokenizerJson or None,
        token_budget=settings.tokenBudget,
        use_environment=CONFIG_ENV not in os.environ,
    )


def _diagnosis_evidence(diagnosis: dict[str, Any]) -> list[str]:
    return [
        str(item.get("text", ""))
        for item in (diagnosis.get("evidence") or [])
        if isinstance(item, dict) and item.get("text")
    ]


def _operator_context(operator: dict[str, Any] | None, name: str) -> dict[str, Any]:
    operator = operator or {}
    references = [str(item) for item in operator.get("torch_references", [])]
    functions = [str(item) for item in operator.get("public_functions", [])]
    return {
        "operator": name,
        "semantics": (
            f"Repository operator {name}; public functions: {', '.join(functions)}"
            if functions
            else ""
        ),
        "pytorch_reference": ", ".join(references),
        "tl_ops": [str(item) for item in operator.get("tl_ops", [])],
    }


def record_from_validation_receipt(
    receipt: dict[str, Any],
    operator: dict[str, Any] | None = None,
    *, sources: list[EvidenceSource] | None = None, as_of: str | None = None,
) -> MemoryRecord | None:
    """Compress an executed validation receipt into one auditable memory."""

    status = str(receipt.get("status", "unknown"))
    if status not in {"passed", "failed"} or receipt.get("dry_run", False):
        return None

    name = str(receipt.get("operator", "unknown"))
    context = _operator_context(operator, name)
    diagnosis = receipt.get("diagnosis")
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    excerpts = _diagnosis_evidence(diagnosis) or [
        str(item) for item in (receipt.get("error_excerpt") or []) if item
    ]
    stage = diagnosis.get("failure_stage") or receipt.get("failure_stage")
    summary = diagnosis.get("summary") or receipt.get("likely_reason")
    source_run = f"operator-lifecycle:{receipt.get('run_id', receipt.get('operator', 'unknown'))}"
    environment = {
        "architecture": receipt.get("architecture"),
        "execution_mode": receipt.get("execution_mode"),
        "triton": receipt.get("triton_version"),
    }
    chain = assemble_receipt_evidence(receipt, sources, as_of=as_of) if sources is not None else None
    if chain and not any(e["kind"] == "validation" for e in chain["items"]):
        return None
    if chain:
        remote = receipt.get("remote_preflight") if isinstance(receipt.get("remote_preflight"), dict) else {}
        for target, field in (("architecture", "architecture"), ("triton", "triton_version")):
            local, nested = receipt.get(field), remote.get(field)
            environment[target] = None if local and nested and local != nested else local or nested
    extra_evidence = {"chain":chain} if chain else {}

    if status == "passed":
        grade = "A" if receipt.get("execution_mode") == "native-riscv" else "B"
        return MemoryRecord(
            memory_type="successful-run",
            summary=(
                "Validation completed successfully with the recorded acceptance "
                f"test; correctness={receipt.get('correctness', 'unknown')}."
            ),
            outcome="passed",
            confidence_grade=grade,
            source_run=source_run,
            environment=environment,
            evidence={
                **extra_evidence,
                "command": receipt.get("command"),
                "exit_code": receipt.get("exit_code"),
                "correctness": receipt.get("correctness"),
                "test_summary": receipt.get("test_summary"),
                "test_files": receipt.get("test_files", []),
                "log_path": receipt.get("log_path"),
                "receipt_path": receipt.get("receipt_path"),
            },
            **context,
        )

    try:
        confidence = float(diagnosis.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0
    grade = "C" if confidence >= 0.6 else "D"
    return MemoryRecord(
        memory_type="failure-diagnosis",
        summary=str(summary or "Validation failed without a classified reason."),
        outcome="failed",
        confidence_grade=grade,
        source_run=source_run,
        failure_stage=str(stage) if stage else None,
        error_signature=normalized_error_signature(
            str(stage) if stage else None,
            str(diagnosis.get("rule_id") or summary or ""),
            excerpts,
        ),
        environment=environment,
        evidence={
            **extra_evidence,
            "rule_id": diagnosis.get("rule_id"),
            "category": diagnosis.get("category"),
            "confidence": confidence,
            "failed_command": diagnosis.get("failed_command") or receipt.get("command"),
            "error_excerpt": excerpts[:4],
            "recommended_actions": (diagnosis.get("recommended_actions") or [])[:4],
            "log_path": receipt.get("log_path"),
            "receipt_path": receipt.get("receipt_path"),
        },
        **context,
    )


def remember_validation(
    repo_root: Path,
    receipt: dict[str, Any],
    operator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort write which never turns a validation result into a failure."""

    try:
        record = record_from_validation_receipt(
            receipt, operator, sources=collect_lifecycle_sources(repo_root, receipt),
        )
        if record is None:
            return {"status": "not-recorded", "reason": "validation was not executed"}
        provider = build_configured_embedding_provider()
        with MemoryStore(memory_database_path(repo_root), provider) as store:
            memory_id, created = store.add(record)
        return {
            "status": "recorded" if created else "deduplicated",
            "memory_id": memory_id,
            "memory_type": record.memory_type,
        }
    except Exception as error:
        return {"status": "unavailable", "reason": str(error)[:500]}


def _public_item(item: dict[str, Any], query_text: str = "") -> dict[str, Any]:
    evidence = item.get("evidence") or {}
    result = {
        "memory_id": item["id"],
        "memory_type": item["memory_type"],
        "operator": item["operator"],
        "failure_stage": item.get("failure_stage"),
        "summary": item["summary"],
        "outcome": item["outcome"],
        "confidence_grade": item["confidence_grade"],
        "score": item.get("retrieval", {}).get("score", 0.0),
        "score_breakdown": item.get("retrieval", {}),
        "rule_id": evidence.get("rule_id"),
        "evidence": evidence.get("error_excerpt", [])[:4],
        "recommended_actions": evidence.get("recommended_actions", [])[:4],
        "reported_cause": item["summary"] if item["outcome"] != "passed" else None,
        "attempted_action": evidence.get("attempted_action"),
        "applied_action": evidence.get("applied_action"),
        "patch_excerpt": (evidence.get("patch_excerpt") or "")[:1200] or None,
        "patch_path": evidence.get("patch_path"),
        "patch_truncated": bool(evidence.get("patch_truncated") or len(evidence.get("patch_excerpt") or "") > 1200),
        "validation_result": encode(evidence.get("test_summary") or evidence.get("correctness")) if evidence.get("test_summary") or evidence.get("correctness") else None,
        "matched_evidence": item.get("matched_evidence"),
        "source_run": item["source_run"],
    }
    # Retain the existing flat fields for clients while bounding untrusted payloads.
    trimmed = False
    for key, value in list(result.items()):
        if isinstance(value, str):
            result[key], cut = clip(value, 600)
            trimmed |= cut
            if key == "patch_excerpt" and cut:
                result["patch_truncated"] = True
        elif isinstance(value, list):
            result[key] = [clip(str(v), 400)[0] for v in value[:4]]
            trimmed |= len(value) > 4 or any(len(str(v)) > 400 for v in value)
    result["evidence_chain"] = bounded_chain(item, query_text=query_text)
    if len(encode(result)) > PUBLIC_ITEM_CHARS:
        result["matched_evidence"] = None
        result["score_breakdown"] = {}
        result["evidence_chain"] = bounded_chain(item, max_chars=4000, query_text=query_text)
        trimmed = True
    result["output_truncated"] = trimmed or result["evidence_chain"]["truncated"]
    result["output_budget_chars"] = PUBLIC_ITEM_CHARS
    if len(encode(result)) > PUBLIC_ITEM_CHARS:
        result["evidence_chain"] = bounded_chain(item, max_chars=256, query_text=query_text)
        result["output_truncated"] = True
    return result


def retrieve_memories(
    repo_root: Path,
    *,
    operator_name: str,
    operator: dict[str, Any] | None = None,
    semantics: str = "",
    pytorch_reference: str = "",
    failure_stage: str | None = None,
    rule_id: str | None = None,
    evidence: list[str] | None = None,
    environment: dict[str, Any] | None = None,
    exclude_source_runs: tuple[str, ...] = (),
    limit: int = 5,
    candidate_strategy: CandidateStrategy = "record-top5",
) -> dict[str, Any]:
    """Return bounded, provenance-preserving historical evidence."""

    if limit < 1 or limit > MAX_MEMORY_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_MEMORY_RESULTS}")
    context = _operator_context(operator, operator_name)
    if semantics:
        context["semantics"] = semantics
    if pytorch_reference:
        context["pytorch_reference"] = pytorch_reference
    evidence = evidence or []
    signature = None
    if failure_stage or rule_id or evidence:
        signature = normalized_error_signature(failure_stage, rule_id, evidence)
    diagnostic_text = " ".join([rule_id or "", *evidence])
    try:
        provider = build_configured_embedding_provider()
        database = memory_database_path(repo_root)
        with MemoryStore(database, provider) as store:
            items = store.retrieve(
                MemoryQuery(
                    operator=context["operator"],
                    semantics=context["semantics"],
                    pytorch_reference=context["pytorch_reference"],
                    tl_ops=context["tl_ops"],
                    failure_stage=failure_stage,
                    error_signature=signature,
                    environment=environment or {},
                    diagnostic_text=diagnostic_text,
                ),
                limit,
                exclude_source_runs=exclude_source_runs,
                score_mode=runtime_config().memory.retrievalMode,
                candidate_strategy=candidate_strategy,
            )
        from codex_agent.memory import render_memory_context
        public = [_public_item(item, semantics + " " + diagnostic_text) for item in items]
        return {
            "status": "found" if items else "empty",
            "database": database.as_posix(),
            "query": {
                "operator": operator_name,
                "failure_stage": failure_stage,
                "rule_id": rule_id,
            },
            "items": public,
            "context_excerpt": render_memory_context(public, query_text=semantics + " " + diagnostic_text),
            "context_unit": "unicode-characters-not-tokens",
        }
    except Exception as error:
        return {
            "status": "unavailable",
            "database": memory_database_path(repo_root).as_posix(),
            "query": {
                "operator": operator_name,
                "failure_stage": failure_stage,
                "rule_id": rule_id,
            },
            "items": [],
            "warning": str(error)[:500],
        }
