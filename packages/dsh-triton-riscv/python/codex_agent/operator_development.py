"""Guarded new-operator development lifecycle for model-facing tools."""

from __future__ import annotations
from codex_agent.runtime_config import permission_enabled

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
from typing import Any

from pydantic import BaseModel, Field

from codex_agent.develop_operator import (
    OperatorSpec,
    audit_test_contract,
    choose_references,
    parse_operator_spec,
    render_task,
)
from codex_agent.discover_operators import discover_operators


ARTIFACT_ROOT = Path("agent-results/operator-development")
MAX_SOURCE_BYTES = 1_000_000


class OperatorInputContract(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)


class OperatorTolerances(BaseModel):
    rtol: float = Field(ge=0)
    atol: float = Field(ge=0)


class NewOperatorSpec(BaseModel):
    """Model-facing semantic contract for a new operator."""

    schema_version: int = 1
    name: str = Field(min_length=1, max_length=120)
    semantics: str = Field(min_length=10, max_length=8000)
    pytorch_reference: str = Field(min_length=6, max_length=4000)
    inputs: list[OperatorInputContract] = Field(min_length=1, max_length=16)
    output: str = Field(min_length=1, max_length=4000)
    shape_cases: list[list[int]] = Field(min_length=1, max_length=64)
    input_shape_cases: list[dict[str, list[int]]] = Field(
        default_factory=list,
        max_length=64,
    )
    dtypes: list[str] = Field(min_length=1, max_length=16)
    tolerances: OperatorTolerances
    backward: bool = False
    reference_operators: list[str] = Field(default_factory=list, max_length=16)
    notes: str = Field(default="", max_length=4000)


class DevelopmentPlanResult(BaseModel):
    development_id: str
    operator: str
    status: str
    implementation_file: str
    test_file: str
    task_file: str
    references: list[dict[str, Any]] = Field(default_factory=list)
    task_markdown: str
    request_path: str
    next_action: str


class DevelopmentPreparationToolResult(BaseModel):
    """Model-facing result that preserves lifecycle rejections as structured data."""

    development_id: str | None = None
    operator: str
    status: str
    implementation_file: str | None = None
    test_file: str | None = None
    task_file: str | None = None
    references: list[dict[str, Any]] = Field(default_factory=list)
    task_markdown: str = ""
    request_path: str | None = None
    existing_files: list[str] = Field(default_factory=list)
    existing_operator: dict[str, Any] | None = None
    blocked_reason: str | None = None
    next_action: str


class DevelopmentProposalResult(BaseModel):
    proposal_id: str
    development_id: str
    operator: str
    status: str
    implementation_file: str
    test_file: str
    rationale: str
    audit_warnings: list[str] = Field(default_factory=list)
    diff_preview: str
    proposal_path: str
    next_action: str


class ApplyDevelopmentResult(BaseModel):
    proposal_id: str
    development_id: str
    operator: str
    status: str
    created_files: list[str] = Field(default_factory=list)
    message: str
    patch_path: str | None = None
    followup_plan: dict[str, Any] | None = None


def _safe_id(value: str, label: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"{label} contains unsupported characters")
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


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


def _relative_file(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("repository file escaped the workspace") from error
    return candidate


def _fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_source(source: str, *, implementation: bool) -> None:
    label = "implementation" if implementation else "test"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(f"{label} source exceeds the 1 MB limit")
    try:
        ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"{label} source is not valid Python: {error}") from error
    if implementation and "@triton.jit" not in source:
        raise ValueError("implementation must contain at least one @triton.jit kernel")
    from codex_agent.source_risk import assert_source_screened
    assert_source_screened(source, label)


def _request_path(repo_root: Path, development_id: str) -> Path:
    safe_id = _safe_id(development_id, "development_id")
    return _artifact_dir(repo_root, "requests") / f"{safe_id}.json"


def _proposal_path(repo_root: Path, proposal_id: str) -> Path:
    safe_id = _safe_id(proposal_id, "proposal_id")
    return _artifact_dir(repo_root, "proposals") / f"{safe_id}.json"


def _load_request(repo_root: Path, development_id: str) -> tuple[Path, dict[str, Any]]:
    path = _request_path(repo_root, development_id)
    return path, _read_json(path)


def _load_proposal(repo_root: Path, proposal_id: str) -> tuple[Path, dict[str, Any]]:
    path = _proposal_path(repo_root, proposal_id)
    return path, _read_json(path)


def _reference_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "implementation_file": item["implementation_file"],
        "test_files": item.get("test_files", []),
        "public_functions": item.get("public_functions", []),
        "triton_kernels": item.get("triton_kernels", []),
        "tl_ops": item.get("tl_ops", []),
        "torch_references": item.get("torch_references", []),
    }


def prepare_operator_development(
    repo_root: Path,
    specification: NewOperatorSpec | dict[str, Any],
) -> DevelopmentPlanResult:
    """Validate a semantic contract and prepare context without tracked writes."""

    root = repo_root.resolve()
    data = (
        specification.model_dump()
        if isinstance(specification, NewOperatorSpec)
        else dict(specification)
    )
    spec = parse_operator_spec(data, root)
    task_file = f"tasks/operators/{spec.name}.md"
    guarded_files = [spec.implementation_file, spec.test_file, task_file]
    existing = [relative for relative in guarded_files if _relative_file(root, relative).exists()]
    if existing:
        raise ValueError(
            "new operator development refuses existing files: " + ", ".join(existing)
        )

    references = choose_references(spec, discover_operators(root))
    summarized_references = [_reference_summary(item) for item in references]
    task_markdown = render_task(spec, references)
    development_id = _new_id("development")
    request_path = _request_path(root, development_id)
    payload = {
        "development_id": development_id,
        "operator": spec.name,
        "status": "planned",
        "specification": asdict(spec),
        "implementation_file": spec.implementation_file,
        "test_file": spec.test_file,
        "task_file": task_file,
        "references": summarized_references,
        "task_markdown": task_markdown,
        "initial_fingerprints": {
            relative: _fingerprint(_relative_file(root, relative))
            for relative in guarded_files
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(request_path, payload)
    return DevelopmentPlanResult(
        development_id=development_id,
        operator=spec.name,
        status="planned",
        implementation_file=spec.implementation_file,
        test_file=spec.test_file,
        task_file=task_file,
        references=summarized_references,
        task_markdown=task_markdown,
        request_path=request_path.relative_to(root).as_posix(),
        next_action=(
            "Inspect the selected references, generate implementation and test source, "
            "then call propose_operator_implementation."
        ),
    )


def _audit_candidate(
    spec: OperatorSpec,
    implementation_source: str,
    test_source: str,
) -> tuple[list[str], str]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        implementation = _relative_file(root, spec.implementation_file)
        test = _relative_file(root, spec.test_file)
        implementation.parent.mkdir(parents=True, exist_ok=True)
        test.parent.mkdir(parents=True, exist_ok=True)
        implementation.write_text(implementation_source, encoding="utf-8")
        test.write_text(test_source, encoding="utf-8")
        audit = audit_test_contract(spec, root)
    if audit.status != "passed":
        raise ValueError("test contract audit failed: " + "; ".join(audit.errors))
    return audit.warnings, audit.test_sha256 or ""


def _source_diff(relative: str, current: str, replacement: str) -> str:
    return "".join(
        unified_diff(
            current.splitlines(keepends=True),
            replacement.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


def propose_operator_implementation(
    repo_root: Path,
    development_id: str,
    implementation_source: str,
    test_source: str,
    rationale: str,
) -> DevelopmentProposalResult:
    """Audit a model-generated implementation and store it for human approval."""

    root = repo_root.resolve()
    _, request = _load_request(root, development_id)
    if request["status"] != "planned":
        raise ValueError(f"development request is already {request['status']}")
    if not rationale.strip():
        raise ValueError("rationale cannot be empty")
    _validate_source(implementation_source, implementation=True)
    _validate_source(test_source, implementation=False)
    spec = parse_operator_spec(request["specification"], root)
    warnings, test_sha256 = _audit_candidate(spec, implementation_source, test_source)

    guarded = [request["implementation_file"], request["test_file"], request["task_file"]]
    for relative in guarded:
        current = _fingerprint(_relative_file(root, relative))
        if current != request["initial_fingerprints"][relative]:
            raise RuntimeError(f"{relative} changed after development was prepared")

    implementation_diff = _source_diff(
        request["implementation_file"],
        "",
        implementation_source,
    )
    test_diff = _source_diff(request["test_file"], "", test_source)
    task_diff = _source_diff(request["task_file"], "", request["task_markdown"])
    combined_diff = implementation_diff + test_diff + task_diff
    proposal_id = _new_id("development-proposal")
    proposal_path = _proposal_path(root, proposal_id)
    proposal = {
        "proposal_id": proposal_id,
        "development_id": development_id,
        "operator": request["operator"],
        "status": "pending_approval",
        "implementation_file": request["implementation_file"],
        "test_file": request["test_file"],
        "task_file": request["task_file"],
        "implementation_source": implementation_source,
        "test_source": test_source,
        "task_markdown": request["task_markdown"],
        "expected_fingerprints": request["initial_fingerprints"],
        "test_sha256": test_sha256,
        "audit_warnings": warnings,
        "rationale": rationale.strip(),
        "diff": combined_diff,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(proposal_path, proposal)
    request_path, updated_request = _load_request(root, development_id)
    updated_request["status"] = "pending_approval"
    updated_request["proposal_id"] = proposal_id
    _write_json(request_path, updated_request)
    return DevelopmentProposalResult(
        proposal_id=proposal_id,
        development_id=development_id,
        operator=request["operator"],
        status="pending_approval",
        implementation_file=request["implementation_file"],
        test_file=request["test_file"],
        rationale=rationale.strip(),
        audit_warnings=warnings,
        diff_preview=combined_diff[:30_000],
        proposal_path=proposal_path.relative_to(root).as_posix(),
        next_action="Ask a human to inspect and approve this proposal through FastAPI.",
    )


def get_operator_development_proposal(repo_root: Path, proposal_id: str) -> dict[str, Any]:
    """Return review metadata without exposing full generated source."""

    _, proposal = _load_proposal(repo_root.resolve(), proposal_id)
    hidden = {
        "implementation_source",
        "test_source",
        "task_markdown",
        "expected_fingerprints",
        "test_sha256",
    }
    return {key: value for key, value in proposal.items() if key not in hidden}


@locked_decision("development")
def decide_operator_development_proposal(
    repo_root: Path,
    proposal_id: str,
    *,
    approve: bool,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    """Record a host-side human decision unavailable to the model-facing MCP server."""

    if not reviewer.strip():
        raise ValueError("reviewer cannot be empty")
    path, proposal = _load_proposal(repo_root.resolve(), proposal_id)
    if proposal["status"] != "pending_approval":
        raise ValueError(f"proposal is already {proposal['status']}")
    proposal["status"] = "approved" if approve else "rejected"
    proposal["review"] = {
        "reviewer": reviewer.strip(),
        "note": note.strip(),
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    proposal["approval_digest"] = approval_digest(repo_root, "development", proposal)
    _write_json(path, proposal)
    request_path, request = _load_request(repo_root.resolve(), proposal["development_id"])
    request["status"] = proposal["status"]
    _write_json(request_path, request)
    return get_operator_development_proposal(repo_root, proposal_id)


def _restore_files(originals: dict[Path, str | None]) -> None:
    for path, content in originals.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def _write_files_atomically(files: dict[Path, str]) -> None:
    originals = {
        path: path.read_text(encoding="utf-8") if path.is_file() else None
        for path in files
    }
    temporary_paths: list[Path] = []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            temporary_paths.append(temporary_path)
            temporary_path.replace(path)
    except Exception:
        _restore_files(originals)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


@guarded_execution("development", "proposal_id")
def apply_operator_implementation(
    repo_root: Path,
    proposal_id: str,
) -> ApplyDevelopmentResult:
    """Apply an approved implementation, test, and task after integrity checks."""

    root = repo_root.resolve()
    proposal_path, proposal = _load_proposal(root, proposal_id)
    relative_files = [
        proposal["implementation_file"],
        proposal["test_file"],
        proposal["task_file"],
    ]
    if proposal["status"] == "applied":
        patch_path = _artifact_dir(root, "patches") / f"{proposal_id}.diff"
        return ApplyDevelopmentResult(
            proposal_id=proposal_id,
            development_id=proposal["development_id"],
            operator=proposal["operator"],
            status="already_applied",
            created_files=relative_files,
            message=(
                "This proposal was already applied. Do not apply it again; "
                "call validate_operator for the operator next."
            ),
            patch_path=(
                patch_path.relative_to(root).as_posix() if patch_path.is_file() else None
            ),
        )
    if proposal["status"] != "approved":
        return ApplyDevelopmentResult(
            proposal_id=proposal_id,
            development_id=proposal["development_id"],
            operator=proposal["operator"],
            status="not_approved",
            message="The development proposal must be approved by the host first.",
        )
    if not permission_enabled("development"):
        return ApplyDevelopmentResult(
            proposal_id=proposal_id,
            development_id=proposal["development_id"],
            operator=proposal["operator"],
            status="blocked",
            message=(
                "The host has not enabled permissions.development."
            ),
        )

    for relative in relative_files:
        current = _fingerprint(_relative_file(root, relative))
        if current != proposal["expected_fingerprints"][relative]:
            raise RuntimeError(f"{relative} changed after the proposal was created")
    _validate_source(proposal["implementation_source"], implementation=True)
    _validate_source(proposal["test_source"], implementation=False)
    _, request = _load_request(root, proposal["development_id"])
    spec = parse_operator_spec(request["specification"], root)
    _audit_candidate(spec, proposal["implementation_source"], proposal["test_source"])

    files = {
        _relative_file(root, proposal["implementation_file"]): proposal["implementation_source"],
        _relative_file(root, proposal["test_file"]): proposal["test_source"],
        _relative_file(root, proposal["task_file"]): proposal["task_markdown"],
    }
    patch_path = _artifact_dir(root, "patches") / f"{proposal_id}.diff"
    patch_path.write_text(proposal["diff"], encoding="utf-8")
    begin_effects()
    _write_files_atomically(files)

    proposal["status"] = "applied"
    proposal["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    proposal["applied_fingerprints"] = {
        relative: _fingerprint(_relative_file(root, relative))
        for relative in relative_files
    }
    _write_json(proposal_path, proposal)
    request_path, updated_request = _load_request(root, proposal["development_id"])
    updated_request["status"] = "applied"
    _write_json(request_path, updated_request)
    return ApplyDevelopmentResult(
        proposal_id=proposal_id,
        development_id=proposal["development_id"],
        operator=proposal["operator"],
        status="applied",
        created_files=relative_files,
        message=(
            "Approved operator files were created. Call validate_operator next; "
            "live execution still requires host approval."
        ),
        patch_path=patch_path.relative_to(root).as_posix(),
    )
