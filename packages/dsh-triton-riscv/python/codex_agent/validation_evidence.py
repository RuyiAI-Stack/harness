"""Independent evidence checks for operator validation receipts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


RECEIPT_ROOT = Path("agent-results/operator-lifecycle/receipts")
RUN_ID_RE = re.compile(r"^run-\d{8}-\d{6}-[0-9a-f]{8}$")


class ValidationEvidence(BaseModel):
    """Machine-verifiable validation outcome, separate from model prose."""

    verdict: Literal[
        "not-observed",
        "planned",
        "verified-passed",
        "verified-failed",
        "invalid",
    ]
    trusted: bool
    success: bool
    run_id: str | None = None
    operator: str | None = None
    receipt_path: str | None = None
    log_path: str | None = None
    reasons: list[str] = Field(default_factory=list)


def no_validation_evidence() -> ValidationEvidence:
    return ValidationEvidence(
        verdict="not-observed",
        trusted=False,
        success=False,
        reasons=["no structured validation receipt was observed"],
    )


def _workspace_file(repo_root: Path, value: str) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"artifact escaped repository: {value}") from error
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_source_snapshot(
    repo_root: Path,
    implementation_file: str,
    test_files: list[str],
) -> dict[str, str]:
    """Hash the exact implementation and acceptance tests being validated."""

    root = repo_root.resolve()
    snapshot: dict[str, str] = {}
    for relative in dict.fromkeys([implementation_file, *test_files]):
        path = _workspace_file(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot[path.relative_to(root).as_posix()] = file_sha256(path)
    return snapshot


def workspace_artifact_sha256(repo_root: Path, value: str | None) -> str | None:
    if not value:
        return None
    try:
        path = _workspace_file(repo_root.resolve(), value)
    except ValueError:
        return None
    return file_sha256(path) if path.is_file() else None


def _invalid(
    run_id: str | None,
    operator: str | None,
    receipt_path: str | None,
    reasons: list[str],
) -> ValidationEvidence:
    return ValidationEvidence(
        verdict="invalid",
        trusted=False,
        success=False,
        run_id=run_id,
        operator=operator,
        receipt_path=receipt_path,
        reasons=reasons,
    )


def audit_validation_receipt(
    repo_root: Path,
    reference: dict[str, Any],
) -> ValidationEvidence:
    """Verify a tool result against durable artifacts and the current source tree."""

    root = repo_root.resolve()
    run_id = reference.get("run_id")
    operator = reference.get("operator")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        return _invalid(
            None,
            operator if isinstance(operator, str) else None,
            None,
            ["invalid run_id"],
        )

    relative_receipt = (RECEIPT_ROOT / f"{run_id}.json").as_posix()
    receipt_path = root / relative_receipt
    if not receipt_path.is_file():
        return _invalid(run_id, operator, relative_receipt, ["receipt file is missing"])

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _invalid(
            run_id,
            operator,
            relative_receipt,
            ["receipt is not valid JSON"],
        )
    if not isinstance(receipt, dict):
        return _invalid(
            run_id,
            operator,
            relative_receipt,
            ["receipt root is not an object"],
        )

    reasons: list[str] = []
    status = receipt.get("status")
    receipt_operator = receipt.get("operator")
    if receipt.get("run_id") != run_id:
        reasons.append("receipt run_id does not match the tool result")
    if isinstance(operator, str) and receipt_operator != operator:
        reasons.append("receipt operator does not match the tool result")
    if reference.get("status") not in {None, status}:
        reasons.append("receipt status does not match the tool result")
    if reference.get("receipt_path") not in {None, relative_receipt}:
        reasons.append("tool result referenced a noncanonical receipt path")
    if not isinstance(receipt_operator, str) or not receipt_operator:
        reasons.append("receipt operator is missing")
    if not isinstance(receipt.get("command"), str) or not receipt["command"].strip():
        reasons.append("validation command is missing")

    implementation = receipt.get("implementation_file")
    tests = receipt.get("test_files")
    expected_snapshot = receipt.get("source_snapshot")
    if not isinstance(implementation, str) or not isinstance(tests, list):
        reasons.append("implementation or test file list is missing")
    elif not isinstance(expected_snapshot, dict) or not expected_snapshot:
        reasons.append("receipt does not contain a source snapshot")
    else:
        try:
            current_snapshot = capture_source_snapshot(root, implementation, tests)
        except (OSError, ValueError) as error:
            reasons.append(f"cannot read current source snapshot: {error}")
        else:
            if current_snapshot != expected_snapshot:
                reasons.append("implementation or acceptance tests changed after validation")
    if receipt.get("source_snapshot_stable") is not True:
        reasons.append("source changed while validation was running")

    if status == "planned":
        if receipt.get("dry_run") is not True or receipt.get("exit_code") is not None:
            reasons.append("planned receipt contains execution results")
        if reasons:
            return _invalid(run_id, receipt_operator, relative_receipt, reasons)
        return ValidationEvidence(
            verdict="planned",
            trusted=True,
            success=False,
            run_id=run_id,
            operator=receipt_operator,
            receipt_path=relative_receipt,
            reasons=["validation is planned but has not executed"],
        )

    if receipt.get("dry_run") is not False:
        reasons.append("receipt is not from a live validation")
    exit_code = receipt.get("exit_code")
    if status == "passed" and exit_code != 0:
        reasons.append("passed receipt does not have exit code 0")
    elif status == "failed" and (
        not isinstance(exit_code, int) or exit_code == 0
    ):
        reasons.append("failed receipt does not have a nonzero exit code")
    elif status not in {"passed", "failed"}:
        reasons.append(f"unsupported executed status: {status}")

    log_path = receipt.get("log_path")
    log_sha256 = receipt.get("log_sha256")
    actual_log_sha256 = workspace_artifact_sha256(root, log_path)
    if not isinstance(log_path, str) or actual_log_sha256 is None:
        reasons.append("validation log is missing or outside the repository")
    elif not isinstance(log_sha256, str) or log_sha256 != actual_log_sha256:
        reasons.append("validation log hash is missing or does not match")

    if receipt.get("execution_target") == "remote":
        preflight = receipt.get("remote_preflight")
        if not isinstance(preflight, dict) or preflight.get("status") != "passed":
            reasons.append("remote RISC-V preflight was not successful")
        elif preflight.get("architecture") != "riscv64":
            reasons.append("remote validation did not run on riscv64")
        approved_run_id = receipt.get("approved_run_id")
        if (
            not isinstance(approved_run_id, str)
            or not RUN_ID_RE.fullmatch(approved_run_id)
        ):
            reasons.append("remote validation has no approved plan")
        else:
            plan_path = root / RECEIPT_ROOT / f"{approved_run_id}.json"
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                reasons.append("approved validation plan is missing")
            else:
                approval = plan.get("approval", {}) if isinstance(plan, dict) else {}
                if (
                    plan.get("status") != "planned"
                    or approval.get("status") != "approved"
                    or approval.get("execution_run_id") != run_id
                ):
                    reasons.append("approved plan is not linked to this execution")

    if reasons:
        return _invalid(run_id, receipt_operator, relative_receipt, reasons)
    verdict = "verified-passed" if status == "passed" else "verified-failed"
    return ValidationEvidence(
        verdict=verdict,
        trusted=True,
        success=status == "passed",
        run_id=run_id,
        operator=receipt_operator,
        receipt_path=relative_receipt,
        log_path=log_path,
    )
