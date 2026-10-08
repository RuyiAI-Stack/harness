"""Guarded operator validation, diagnosis, and repair lifecycle tools."""

from __future__ import annotations
from codex_agent.runtime_config import approval_required, permission_enabled, runtime_config

from codex_agent.execution_guard import atomic_json, approval_digest, begin_effects, guarded_execution, locked_decision

import ast
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from codex_agent.operator_tools import discover_operator_evidence
from codex_agent.diagnostic_memory import remember_validation
from codex_agent.memory_api import MemoryRetrievalToolResult, RetrievedMemoryCase, retrieve_operator_memory
from codex_agent.remote_executor import (
    RemotePreflightResult,
    RemoteValidationConfig,
    remote_validation_command,
    run_remote_operator,
)
from codex_agent.validation_evidence import (
    ValidationEvidence,
    audit_validation_receipt,
    capture_source_snapshot,
    workspace_artifact_sha256,
)
from codex_agent.validate_operator import run_operator


ARTIFACT_ROOT = Path("agent-results/operator-lifecycle")
REPAIRABLE_STAGES = {"correctness", "pytest", "triton-frontend"}
COMPILER_WORKAROUND_STAGES = {
    "triton-shared-opt",
    "buddy-opt",
    "mlir-translate",
    "llc",
    "compilation",
}
MAX_REPAIR_ATTEMPTS = 3
MAX_REPLACEMENT_BYTES = 1_000_000


class ValidationToolResult(BaseModel):
    """Structured validation plan or execution receipt."""

    run_id: str
    operator: str
    status: str
    command: str | None = None
    exit_code: int | None = None
    failure_stage: str | None = None
    likely_reason: str | None = None
    error_excerpt: list[str] = Field(default_factory=list)
    implementation_file: str | None = None
    test_files: list[str] = Field(default_factory=list)
    log_path: str | None = None
    execution_target: Literal["local", "remote"]
    remote_preflight: RemotePreflightResult | None = None
    receipt_path: str
    evidence: ValidationEvidence
    memory_write: dict[str, Any] = Field(default_factory=dict)
    isolation: dict[str, Any] = Field(default_factory=dict)


class DiagnosisToolResult(BaseModel):
    """Actionable interpretation of one validation receipt."""

    run_id: str
    operator: str
    status: Literal["passed", "failed", "planned", "blocked", "unknown"]
    failure_stage: str | None = None
    likely_reason: str | None = None
    recommended_action: str
    source_repair_allowed: bool
    repair_strategy: Literal["source-fix", "compiler-workaround", "none"] = "none"
    repair_attempts: int = 0
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS
    attempts_remaining: int = MAX_REPAIR_ATTEMPTS
    stop_reason: str | None = None
    user_action: str | None = None
    evidence: list[str] = Field(default_factory=list)

    rule_id: str | None = None
    confidence: float = 0.0
    failed_command: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)
    memory_write_status: str = "not-recorded"
    memory_status: str = "empty"
    similar_cases: list[RetrievedMemoryCase] = Field(default_factory=list)


class RepairProposalResult(BaseModel):
    """A source replacement waiting for human approval."""

    proposal_id: str
    run_id: str
    operator: str
    status: str
    implementation_file: str
    rationale: str
    diff_preview: str
    proposal_path: str


class ApplyRepairResult(BaseModel):
    """Result of attempting to apply an approved proposal."""

    proposal_id: str
    operator: str
    status: str
    implementation_file: str
    message: str
    patch_path: str | None = None
    followup_plan: dict[str, Any] | None = None


def _safe_id(value: str, label: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
        raise ValueError(f"{label} contains unsupported characters")
    return value


def _artifact_dir(repo_root: Path, name: str) -> Path:
    path = repo_root.resolve() / ARTIFACT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _relative_file(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("repository file escaped the workspace") from error
    return candidate


def _validate_replacement_source(source: str) -> None:
    if len(source.encode("utf-8")) > MAX_REPLACEMENT_BYTES:
        raise ValueError("replacement source exceeds the 1 MB limit")
    ast.parse(source)
    if "@triton.jit" not in source:
        raise ValueError("replacement must retain at least one @triton.jit kernel")
    from codex_agent.source_risk import assert_source_screened
    assert_source_screened(source, "replacement_source")


def _load_receipt(repo_root: Path, run_id: str) -> dict[str, Any]:
    safe_run_id = _safe_id(run_id, "run_id")
    return _read_json(_artifact_dir(repo_root, "receipts") / f"{safe_run_id}.json")


def _load_proposal(repo_root: Path, proposal_id: str) -> tuple[Path, dict[str, Any]]:
    safe_proposal_id = _safe_id(proposal_id, "proposal_id")
    path = _artifact_dir(repo_root, "proposals") / f"{safe_proposal_id}.json"
    return path, _read_json(path)


def _repair_attempts(repo_root: Path, operator: str) -> int:
    """Count current repair proposals, resetting after a passing receipt."""

    latest_pass = ""
    for path in _artifact_dir(repo_root, "receipts").glob("run-*.json"):
        try:
            receipt = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if receipt.get("operator") == operator and receipt.get("status") == "passed":
            latest_pass = max(latest_pass, str(receipt.get("created_at", "")))

    attempts = 0
    for path in _artifact_dir(repo_root, "proposals").glob("repair-*.json"):
        try:
            proposal = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if proposal.get("operator") != operator:
            continue
        if str(proposal.get("created_at", "")) <= latest_pass:
            continue
        if proposal.get("status") in {
            "pending_approval",
            "approved",
            "applied",
            "rejected",
        }:
            attempts += 1
    return attempts


@guarded_execution("validation", "approved_run_id")
def validate_operator_target(
    repo_root: Path,
    operator_name: str,
    *,
    execute: bool = False,
    approved_run_id: str | None = None,
    source_env: bool = True,
    timeout_seconds: int = 900,
) -> ValidationToolResult:
    """Plan validation by default; execute only when the host explicitly enables it."""

    root = repo_root.resolve()
    evidence = discover_operator_evidence(root, operator_name)
    if evidence.status != "found" or evidence.operator is None:
        raise ValueError(evidence.message)
    if timeout_seconds < 1 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be between 1 and 3600")
    if approval_required() and timeout_seconds > 900:
        raise ValueError("Harness validation timeout must not exceed the 900 second total budget")
    if execute and not permission_enabled("validation"):
        raise PermissionError(
            "live validation is disabled; the host must set "
            "permissions.validation=true (legacy: TRITON_RISCV_ALLOW_VALIDATION=1)"
        )

    operator = evidence.operator.model_dump()
    source_snapshot = capture_source_snapshot(
        root,
        operator["implementation_file"],
        operator["test_files"],
    )
    approved_plan: dict[str, Any] | None = None
    require_approval = approval_required()
    if execute and (require_approval or approved_run_id is not None):
        if not approved_run_id:
            raise PermissionError(
                "live validation requires an approved_run_id from the host"
            )
        approved_plan = _load_receipt(root, approved_run_id)
        approval = approved_plan.get("approval", {})
        if approved_plan.get("status") != "planned":
            raise PermissionError("approved_run_id is not a validation plan")
        if approved_plan.get("operator") != evidence.operator.name:
            raise PermissionError("approved validation plan targets another operator")
        if approval.get("status") != "approved":
            raise PermissionError("validation plan has not been approved by the host")
        if approval.get("execution_run_id"):
            raise PermissionError("validation plan has already been executed")

    remote = RemoteValidationConfig.from_env()
    if (
        execute
        and runtime_config().remote.required
        and remote is None
    ):
        raise PermissionError(
            "remote validation is required; configure remote.host and remote.repository"
        )

    from codex_agent.execution_guard import execution_details, execution_kind, record_execution_detail
    recoverable_operation = execute and execution_kind() == "validation"
    run_id = execution_details().get("validation_result_run_id") if recoverable_operation else None
    run_id = run_id or _new_id("run")
    if recoverable_operation:
        record_execution_detail(validation_result_run_id=run_id)
    validation_run_id = (approved_plan or {}).get("validation_run_id") or run_id
    if execute and approved_plan is not None:
        if (
            approved_plan.get("source_env", source_env) != source_env
            or approved_plan.get("timeout_seconds", timeout_seconds) != timeout_seconds
        ):
            raise PermissionError("validation settings changed after approval")
        current_plan = run_operator(
            operator,
            repo_root=root,
            results_dir=_artifact_dir(root, "validation"),
            source_env=source_env,
            dry_run=True,
            timeout_seconds=timeout_seconds,
            run_id=validation_run_id,
        )
        if remote is not None:
            _, current_plan.command = remote_validation_command(operator, remote)
        if current_plan.command != approved_plan.get("command"):
            raise PermissionError(
                "validation command changed after approval; create and approve a new plan"
            )
        approved_snapshot = approved_plan.get("source_snapshot")
        if not isinstance(approved_snapshot, dict):
            raise PermissionError(
                "approved plan predates source locking; create and approve a new plan"
            )
        if approved_snapshot != source_snapshot:
            raise PermissionError(
                "validation source or tests changed after approval; "
                "create and approve a new plan"
            )

    preflight: RemotePreflightResult | None = None
    if execute:
        from codex_agent.source_risk import screen_operator_files
        screen_operator_files(root, [operator["implementation_file"], *operator["test_files"]])
        begin_effects()
    if execute and remote is not None:
        result, preflight = run_remote_operator(
            operator,
            repo_root=root,
            results_dir=_artifact_dir(root, "validation"),
            timeout_seconds=timeout_seconds,
            config=remote,
        )
        execution_target: Literal["local", "remote"] = "remote"
    else:
        result = run_operator(
            operator,
            repo_root=root,
            results_dir=_artifact_dir(root, "validation"),
            source_env=source_env,
            dry_run=not execute,
            timeout_seconds=timeout_seconds,
            run_id=validation_run_id,
        )
        execution_target = "remote" if remote is not None else "local"
        if not execute and remote is not None:
            _, result.command = remote_validation_command(operator, remote)
    final_source_snapshot = capture_source_snapshot(
        root,
        operator["implementation_file"],
        operator["test_files"],
    )
    source_snapshot_stable = source_snapshot == final_source_snapshot
    if execute and not source_snapshot_stable:
        result.status = "failed"
        result.failure_stage = "evidence"
        result.likely_reason = "implementation or acceptance tests changed during validation"
        result.error_excerpt = [
            "validation evidence rejected because source files changed during execution"
        ]
    receipt_path = _artifact_dir(root, "receipts") / f"{run_id}.json"
    payload = {
        **asdict(result),
        "run_id": run_id,
        "validation_run_id": validation_run_id,
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "source_env": source_env,
        "timeout_seconds": timeout_seconds,
        "execution_target": execution_target,
        "remote_preflight": preflight.model_dump() if preflight else None,
        "source_snapshot": final_source_snapshot,
        "source_snapshot_stable": source_snapshot_stable,
        "log_sha256": workspace_artifact_sha256(root, result.log_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if approved_run_id:
        payload["approved_run_id"] = approved_run_id
    _write_json(receipt_path, payload)
    if approved_plan is not None:
        approval = dict(approved_plan.get("approval", {}))
        approval.update(
            {
                "execution_run_id": run_id,
                "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        approved_plan["approval"] = approval
        approved_path = _artifact_dir(root, "receipts") / f"{approved_run_id}.json"
        _write_json(approved_path, approved_plan)
    validation_evidence = audit_validation_receipt(
        root,
        {
            "run_id": run_id,
            "operator": result.operator,
            "status": result.status,
            "receipt_path": receipt_path.relative_to(root).as_posix(),
        },
    )
    # Only executed, audited outcomes become historical evidence.
    payload["memory_write"] = (
        remember_validation(root, payload, operator)
        if execute and result.failure_stage not in {"capacity", "cancellation"} and validation_evidence.verdict in {"verified-passed", "verified-failed"}
        else {"status": "not-recorded", "reason": "no verified execution"}
    )
    _write_json(receipt_path, payload)
    return ValidationToolResult(
        run_id=run_id,
        operator=result.operator,
        status=result.status,
        command=result.command,
        exit_code=result.exit_code,
        failure_stage=result.failure_stage,
        likely_reason=result.likely_reason,
        error_excerpt=result.error_excerpt,
        implementation_file=result.implementation_file,
        test_files=result.test_files,
        log_path=result.log_path,
        execution_target=execution_target,
        remote_preflight=preflight,
        receipt_path=receipt_path.relative_to(root).as_posix(),
        evidence=validation_evidence,
        memory_write=payload["memory_write"],
        isolation=result.isolation,
    )


def get_validation_plan(repo_root: Path, run_id: str) -> dict[str, Any]:
    """Return a reviewable validation command without executing it."""

    receipt = _load_receipt(repo_root.resolve(), run_id)
    if receipt.get("status") != "planned":
        raise ValueError("run_id is not a pending validation plan")
    return {
        "run_id": receipt["run_id"],
        "operator": receipt["operator"],
        "status": receipt["status"],
        "command": receipt.get("command"),
        "execution_target": receipt.get("execution_target"),
        "implementation_file": receipt.get("implementation_file"),
        "test_files": receipt.get("test_files", []),
        "receipt_path": (
            ARTIFACT_ROOT / "receipts" / f"{receipt['run_id']}.json"
        ).as_posix(),
        "approval": receipt.get("approval", {"status": "pending_approval"}),
    }


@locked_decision("validation")
def decide_validation_plan(
    repo_root: Path,
    run_id: str,
    *,
    approve: bool,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    """Record the human decision for one exact validation command."""

    if not reviewer.strip():
        raise ValueError("reviewer cannot be empty")
    root = repo_root.resolve()
    receipt = _load_receipt(root, run_id)
    if receipt.get("status") != "planned":
        raise ValueError("run_id is not a pending validation plan")
    current = receipt.get("approval", {})
    if current.get("status") in {"approved", "rejected"}:
        raise ValueError(f"validation plan is already {current['status']}")
    receipt["approval"] = {
        "status": "approved" if approve else "rejected",
        "reviewer": reviewer.strip(),
        "note": note.strip(),
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = _artifact_dir(root, "receipts") / f"{run_id}.json"
    receipt["approval_digest"] = approval_digest(root, "validation", receipt)
    _write_json(path, receipt)
    return get_validation_plan(root, run_id)


def diagnose_failure_run(repo_root: Path, run_id: str) -> DiagnosisToolResult:
    """Diagnose a stored validation result without rerunning the operator."""

    receipt = _load_receipt(repo_root, run_id)
    status = receipt.get("status", "unknown")
    stage = receipt.get("failure_stage")
    reason = receipt.get("likely_reason")
    has_tests = bool(receipt.get("test_files"))
    attempts = _repair_attempts(repo_root.resolve(), str(receipt.get("operator", "")))
    attempts_remaining = max(0, MAX_REPAIR_ATTEMPTS - attempts)
    source_failure = stage in REPAIRABLE_STAGES
    compiler_failure = stage in COMPILER_WORKAROUND_STAGES
    eligible = source_failure or compiler_failure
    repairable = (
        status == "failed"
        and eligible
        and has_tests
        and attempts_remaining > 0
    )
    strategy: Literal["source-fix", "compiler-workaround", "none"] = "none"
    stop_reason: str | None = None
    user_action: str | None = None
    if repairable:
        strategy = "source-fix" if source_failure else "compiler-workaround"

    if status == "passed":
        action = "No repair is needed; preserve the passing receipt as evidence."
    elif status == "planned":
        action = "Request approval and run live validation before diagnosing a failure."
    elif status == "failed" and stage in REPAIRABLE_STAGES and not has_tests:
        action = "Define and review an acceptance test before allowing source repair."
    elif repairable:
        if strategy == "compiler-workaround":
            action = (
                "Inspect passing repository references and prepare a semantics-preserving "
                "implementation workaround that avoids the unsupported lowering path."
            )
        else:
            action = "Inspect the implementation and prepare a bounded source repair proposal."
    elif status == "failed" and eligible and attempts_remaining == 0:
        stop_reason = f"bounded repair limit reached ({MAX_REPAIR_ATTEMPTS} attempts)"
        action = "Stop automatic source changes and escalate with the collected artifacts."
        user_action = (
            "Provide the latest receipt, full log, generated patch history, and validation "
            "command to the Triton-RISCV compiler owner."
        )
    elif stage == "isolation":
        action = "Repair the sandbox setup; do not edit operator source or disable isolation."
        stop_reason = "required execution isolation could not be established"
        user_action = "Inspect the sandbox error and host runtime mounts/capabilities; no unrestricted retry is allowed."
    elif stage == "cancellation":
        action = "Preserve the host cancellation evidence; do not repair operator source."
        stop_reason = "validation cancelled, not an operator correctness failure"
    elif stage == "capacity":
        action = "Check busy or fenced server slots; do not edit operator source or bypass admission."
        stop_reason = "server capacity unavailable; operator test was not started"
        user_action = "Wait for running validations, or ask the host operator to inspect an unknown task before releasing its slot."
    elif stage in {"environment", "import", "build", "timeout"}:
        action = "Repair the environment or retry policy; do not change operator source."
        stop_reason = f"{stage or 'environment'} failure is outside operator source"
        user_action = (
            "Check the remote environment, dependency versions, available disk space, and "
            "the failing command before rerunning validation."
        )
    else:
        action = "Collect a fuller log and request human triage before editing source."
        stop_reason = "failure stage is not classified for safe automatic repair"
        user_action = (
            "Review the full validation log and add a failure classifier or an explicitly "
            "approved repair policy before changing code."
        )

    normalized_status = status if status in {"passed", "failed", "planned", "blocked"} else "unknown"
    memory = retrieve_operator_memory(repo_root, run_id=run_id)
    structured = receipt.get("diagnosis") or {}
    return DiagnosisToolResult(
        run_id=receipt["run_id"],
        operator=receipt["operator"],
        status=normalized_status,
        failure_stage=stage,
        likely_reason=reason,
        recommended_action=action,
        source_repair_allowed=repairable,
        repair_strategy=strategy,
        repair_attempts=attempts,
        attempts_remaining=attempts_remaining,
        stop_reason=stop_reason,
        user_action=user_action,
        evidence=list(receipt.get("error_excerpt", [])),
        rule_id=structured.get("rule_id"),
        confidence=float(structured.get("confidence") or 0),
        failed_command=structured.get("failed_command") or receipt.get("command"),
        recommended_actions=list(structured.get("recommended_actions") or []),
        memory_write_status=(receipt.get("memory_write") or {}).get("status", "not-recorded"),
        memory_status=memory.status,
        similar_cases=memory.items,
    )


def propose_operator_repair(
    repo_root: Path,
    run_id: str,
    replacement_source: str,
    rationale: str,
) -> RepairProposalResult:
    """Store a source-only repair proposal without changing repository files."""

    root = repo_root.resolve()
    receipt = _load_receipt(root, run_id)
    diagnosis = diagnose_failure_run(root, run_id)
    if not diagnosis.source_repair_allowed:
        raise ValueError(
            f"source repair is not allowed for stage {diagnosis.failure_stage!r}"
        )
    if not rationale.strip():
        raise ValueError("rationale cannot be empty")
    _validate_replacement_source(replacement_source)

    relative_implementation = receipt["implementation_file"]
    implementation = _relative_file(root, relative_implementation)
    allowed_root = (root / "python/examples/flaggems").resolve()
    if implementation.parent != allowed_root or implementation.name.startswith("test_"):
        raise ValueError("repair target must be a FlagGems implementation file")
    current_source = implementation.read_text(encoding="utf-8")
    if current_source == replacement_source:
        raise ValueError("replacement source does not change the implementation")

    test_hashes = {
        relative: _sha256(_relative_file(root, relative))
        for relative in receipt.get("test_files", [])
    }
    diff = "".join(
        unified_diff(
            current_source.splitlines(keepends=True),
            replacement_source.splitlines(keepends=True),
            fromfile=f"a/{relative_implementation}",
            tofile=f"b/{relative_implementation}",
        )
    )
    proposal_id = _new_id("repair")
    proposal_path = _artifact_dir(root, "proposals") / f"{proposal_id}.json"
    payload = {
        "proposal_id": proposal_id,
        "run_id": run_id,
        "operator": receipt["operator"],
        "status": "pending_approval",
        "implementation_file": relative_implementation,
        "source_sha256": _sha256(implementation),
        "test_sha256": test_hashes,
        "replacement_source": replacement_source,
        "rationale": rationale.strip(),
        "attempt_number": diagnosis.repair_attempts + 1,
        "repair_strategy": diagnosis.repair_strategy,
        "diff": diff,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(proposal_path, payload)
    return RepairProposalResult(
        proposal_id=proposal_id,
        run_id=run_id,
        operator=receipt["operator"],
        status="pending_approval",
        implementation_file=relative_implementation,
        rationale=rationale.strip(),
        diff_preview=diff[:20_000],
        proposal_path=proposal_path.relative_to(root).as_posix(),
    )


def get_repair_proposal(repo_root: Path, proposal_id: str) -> dict[str, Any]:
    """Return a proposal without exposing its full replacement source."""

    _, proposal = _load_proposal(repo_root, proposal_id)
    return {
        key: value
        for key, value in proposal.items()
        if key not in {"replacement_source", "source_sha256", "test_sha256"}
    }


@locked_decision("repair")
def decide_repair_proposal(
    repo_root: Path,
    proposal_id: str,
    *,
    approve: bool,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    """Record a host-side human decision; this function is not exposed as an MCP tool."""

    if not reviewer.strip():
        raise ValueError("reviewer cannot be empty")
    path, proposal = _load_proposal(repo_root, proposal_id)
    if proposal["status"] != "pending_approval":
        raise ValueError(f"proposal is already {proposal['status']}")
    proposal["status"] = "approved" if approve else "rejected"
    proposal["review"] = {
        "reviewer": reviewer.strip(),
        "note": note.strip(),
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    proposal["approval_digest"] = approval_digest(repo_root, "repair", proposal)
    _write_json(path, proposal)
    return get_repair_proposal(repo_root, proposal_id)


@guarded_execution("repair", "proposal_id")
def apply_operator_repair(repo_root: Path, proposal_id: str) -> ApplyRepairResult:
    """Apply an approved proposal after source and acceptance-test integrity checks."""

    root = repo_root.resolve()
    _, proposal = _load_proposal(root, proposal_id)
    relative_implementation = proposal["implementation_file"]
    implementation = _relative_file(root, relative_implementation)

    if proposal["status"] != "approved":
        return ApplyRepairResult(
            proposal_id=proposal_id,
            operator=proposal["operator"],
            status="not_approved",
            implementation_file=relative_implementation,
            message="The proposal must be approved by the host before it can be applied.",
        )
    if not permission_enabled("repair"):
        return ApplyRepairResult(
            proposal_id=proposal_id,
            operator=proposal["operator"],
            status="blocked",
            implementation_file=relative_implementation,
            message="The host has not enabled permissions.repair.",
        )
    if _sha256(implementation) != proposal["source_sha256"]:
        raise RuntimeError("implementation changed after the proposal was created")
    for relative, expected_hash in proposal["test_sha256"].items():
        if _sha256(_relative_file(root, relative)) != expected_hash:
            raise RuntimeError("an acceptance test changed after the proposal was created")

    replacement = proposal["replacement_source"]
    _validate_replacement_source(replacement)
    original = implementation.read_text(encoding="utf-8")
    patch_path = _artifact_dir(root, "patches") / f"{proposal_id}.diff"
    patch_path.write_text(proposal["diff"], encoding="utf-8")

    temporary_name: str | None = None
    begin_effects()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=implementation.parent,
            prefix=f".{implementation.name}.",
            delete=False,
        ) as temporary:
            temporary.write(replacement)
            temporary_name = temporary.name
        Path(temporary_name).replace(implementation)
    except Exception:
        implementation.write_text(original, encoding="utf-8")
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise

    proposal_path, updated = _load_proposal(root, proposal_id)
    updated["status"] = "applied"
    updated["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    updated["applied_source_sha256"] = _sha256(implementation)
    _write_json(proposal_path, updated)
    return ApplyRepairResult(
        proposal_id=proposal_id,
        operator=proposal["operator"],
        status="applied",
        implementation_file=relative_implementation,
        message="Approved source replacement was applied; validation must run next.",
        patch_path=patch_path.relative_to(root).as_posix(),
    )
