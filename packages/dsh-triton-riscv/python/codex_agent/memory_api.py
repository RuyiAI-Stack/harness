"""Typed historical evidence interface shared by the lifecycle and MCP host."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from codex_agent.diagnostic_memory import retrieve_memories
from codex_agent.failure_diagnosis import diagnose_log
from codex_agent.memory_selection import CandidateStrategy
from codex_agent.operator_tools import discover_operator_evidence


class RetrievedMemoryCase(BaseModel):
    memory_id: int
    memory_type: str
    operator: str
    failure_stage: str | None = None
    summary: str
    outcome: str
    confidence_grade: str
    score: float
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    rule_id: str | None = None
    evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    reported_cause: str | None = None
    attempted_action: str | None = None
    applied_action: str | None = None
    patch_excerpt: str | None = None
    patch_path: str | None = None
    patch_truncated: bool = False
    validation_result: str | None = None
    matched_evidence: dict[str, Any] | None = None
    source_run: str
    evidence_chain: dict[str, Any] = Field(default_factory=dict)
    output_truncated: bool = False
    output_budget_chars: int | None = None


class MemoryRetrievalToolResult(BaseModel):
    status: Literal["found", "empty", "unavailable"]
    database: str
    query: dict[str, Any]
    items: list[RetrievedMemoryCase] = Field(default_factory=list)
    warning: str | None = None
    context_excerpt: str | None = None
    context_unit: str | None = None


def retrieve_operator_memory(
    repo_root: Path, operator_name: str = "", *, run_id: str | None = None,
    semantics: str = "", pytorch_reference: str = "",
    failure_stage: str | None = None, error_text: str = "", limit: int = 5,
    candidate_strategy: CandidateStrategy = "record-top5",
) -> MemoryRetrievalToolResult:
    root = repo_root.resolve()
    rule_id = None
    evidence = [error_text] if error_text.strip() else []
    environment: dict[str, Any] = {}
    excluded: tuple[str, ...] = ()
    if run_id:
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            raise ValueError("run_id contains unsupported characters")
        receipt = json.loads((root / "agent-results/operator-lifecycle/receipts" / f"{run_id}.json").read_text())
        if operator_name and operator_name != receipt.get("operator"):
            raise ValueError("operator_name does not match the validation receipt")
        operator_name = receipt["operator"]
        structured = receipt.get("diagnosis") or {}
        if not structured:
            path = Path(receipt.get("log_path") or "")
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            text = path.read_text(errors="replace") if path.is_relative_to(root / "agent-results") and path.is_file() else ""
            structured = diagnose_log(text, receipt.get("exit_code"), validation_command=receipt.get("command"))
        failure_stage = structured.get("failure_stage") or receipt.get("failure_stage")
        rule_id = structured.get("rule_id")
        evidence = [str(e.get("text", "")) for e in structured.get("evidence", []) if isinstance(e, dict)]
        remote = receipt.get("remote_preflight") or {}
        environment = {
            "architecture": receipt.get("architecture") or remote.get("architecture"),
            "execution_mode": receipt.get("execution_mode"),
            "triton": receipt.get("triton_version") or remote.get("triton_version"),
        }
        excluded = (f"operator-lifecycle:{run_id}",)
    if not operator_name:
        raise ValueError("operator_name or run_id is required")
    found = discover_operator_evidence(root, operator_name)
    result = retrieve_memories(
        root, operator_name=operator_name,
        operator=found.operator.model_dump() if found.operator else None,
        semantics=semantics, pytorch_reference=pytorch_reference,
        failure_stage=failure_stage, rule_id=rule_id, evidence=evidence,
        environment=environment, exclude_source_runs=excluded, limit=limit,
        candidate_strategy=candidate_strategy,
    )
    return MemoryRetrievalToolResult.model_validate(result)
