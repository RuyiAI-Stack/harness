"""MCP server exposing guarded Triton-RISCV repository tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from codex_agent.memory_api import MemoryRetrievalToolResult, retrieve_operator_memory
from codex_agent.memory_selection import CandidateStrategy

from codex_agent.operator_development import (
    ApplyDevelopmentResult,
    DevelopmentPreparationToolResult,
    DevelopmentProposalResult,
    NewOperatorSpec,
    get_operator_development_proposal,
    apply_operator_implementation,
    prepare_operator_development,
    propose_operator_implementation,
)
from codex_agent.operator_lifecycle import (
    ApplyRepairResult,
    DiagnosisToolResult,
    RepairProposalResult,
    ValidationToolResult,
    apply_operator_repair,
    diagnose_failure_run,
    propose_operator_repair,
    validate_operator_target,
)
from codex_agent.operator_tools import (
    DiscoverOperatorResult,
    OperatorFileResult,
    discover_operator_evidence,
    read_operator_file,
)
from codex_agent.remote_executor import (
    RemotePreflightResult,
    check_remote_environment,
)


server = MCPServer(
    name="triton-riscv-tools",
    title="Triton-RISCV Operator Tools",
    description="Guarded operator discovery, validation, diagnosis, and repair tools.",
)


def repository_root() -> Path:
    """Resolve the repository selected by the Harness host."""

    from codex_agent.paths import repository_root as resolve_root
    return resolve_root()


@server.tool(name="inspect_project", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def inspect_project_tool(kind: Literal["operator", "project", "pytest", "lit", "build"] = "operator",
                         offset: int = 0, limit: int = 25, contains: str = "") -> dict[str, Any]:
    """Discover paginated operator/project test targets. Inventory is not executed coverage."""
    from codex_agent.project_tools import inspect_project
    return inspect_project(repository_root(), kind=kind, offset=offset, limit=limit, contains=contains)


@server.tool(name="prepare_validation_job", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
def prepare_validation_job_tool(targets: list[str],
    kind: Literal["operator-batch", "project"] = "operator-batch",
    source_env: bool = True, timeout_seconds: int = 300) -> dict[str, Any]:
    """Plan 1..20 discovered operator names or project IDs; total timeout <=900s. No tests run. Project checks run on the host, operator batches can use configured SSH."""
    from codex_agent.project_tools import prepare_validation_job
    return prepare_validation_job(repository_root(), targets, kind=kind,
                                  source_env=source_env, timeout_seconds=timeout_seconds)


@server.tool(name="execute_validation_job", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True))
def execute_validation_job_tool(job_id: str) -> dict[str, Any]:
    """Request native approval then execute a source-locked job. New remote operator batches resume the SAME job_id after disconnect: replay committed children, collect active children, start only pending children. Completed jobs replay results; never replace unknown jobs with new IDs."""
    from codex_agent.project_tools import execute_validation_job
    return execute_validation_job(repository_root(), job_id)


@server.tool(name="get_validation_job", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def get_validation_job_tool(job_id: str) -> dict[str, Any]:
    """Read persisted results and local batch progress. No SSH or execution; a progress cursor is not evidence of test success."""
    from codex_agent.project_tools import get_validation_job
    return get_validation_job(repository_root(), job_id)


@server.tool(name="evaluate_plugin_fixtures", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def evaluate_plugin_fixtures_tool() -> dict[str, Any]:
    """Evaluate bundled diagnosis/retrieval fixtures offline. This is not real-business effectiveness evidence."""
    from codex_agent.project_tools import evaluate_plugin
    return evaluate_plugin()


@server.tool(name="retrieve_operator_memory", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def retrieve_operator_memory_tool(
    operator_name: str = "", run_id: str | None = None, semantics: str = "",
    pytorch_reference: str = "", failure_stage: str | None = None,
    error_text: str = "", limit: int = 5,
    candidate_strategy: CandidateStrategy = "record-top5",
) -> MemoryRetrievalToolResult:
    """Retrieve source-bound historical evidence; never treat it as current execution proof."""
    return retrieve_operator_memory(repository_root(), operator_name, run_id=run_id,
        semantics=semantics, pytorch_reference=pytorch_reference, failure_stage=failure_stage,
        error_text=error_text, limit=limit, candidate_strategy=candidate_strategy)


def _submit_validation_task(root, operator, run_id):
    from codex_agent.tasks.service import TaskService
    job = TaskService(root).submit_validation(operator, run_id, "validation:" + run_id)
    if job["status"] in {"succeeded", "failed"} and job["result"].get("receipt_path"):
        return job["result"]
    return {"status": "queued" if job["status"] in {"queued", "running"} else "blocked",
            "async_job": True, "queue_status": job["status"], "job_id": job["id"], "run_id": run_id,
            "message": "Not a test result. End this turn; inspect_queued_task reads persisted progress later."}


@server.tool(name="execute_approved_validation", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True))
def execute_approved_validation_tool(run_id: str) -> dict[str, Any]:
    """Execute the exact host-approved plan. After SSH loss, use the SAME run_id to collect its original remote job, not restart. Unknown evidence needs host inspection. The model cannot grant approval."""
    from codex_agent.operator_lifecycle import _load_receipt
    root = repository_root()
    receipt = _load_receipt(root, run_id)
    from codex_agent.runtime_config import runtime_config
    if runtime_config().queue.enabled:
        return _submit_validation_task(root, receipt["operator"], run_id)
    return validate_operator_target(root, receipt["operator"], execute=True, approved_run_id=run_id,
        source_env=receipt.get("source_env", True), timeout_seconds=receipt.get("timeout_seconds", 900)).model_dump()


@server.tool(name="inspect_queued_task", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def inspect_queued_task_tool(job_id: str) -> dict[str, Any]:
    """Read persisted queue/worker results; queued or published never means tests passed. No wait loop, replay or execution."""
    from codex_agent.tasks.store import JobStore
    return JobStore(repository_root()).detail(job_id)


@server.tool(name="get_validation_status", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True))
def get_validation_status_tool(run_id: str, remote: bool = False) -> dict[str, Any]:
    """Inspect the original approved plan ID. Default reads local evidence; remote=true only queries its existing remote job. Never starts, repairs, cleans or reruns anything. A remote completed state is not a verified pass until the local receipt is committed. Report unknown honestly."""
    from codex_agent.remote_maintenance import validation_status
    return validation_status(repository_root(), run_id, remote=remote)


@server.tool(name="apply_development_proposal", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False))
def apply_development_proposal_tool(proposal_id: str) -> ApplyDevelopmentResult:
    """Apply only the exact host-approved proposal; alias retained for Harness plugin policy."""
    return apply_operator_implementation(repository_root(), proposal_id)


@server.tool(name="get_operator_implementation_proposal", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def get_operator_implementation_proposal_tool(proposal_id: str) -> dict[str, Any]:
    """Read proposal review metadata without changing its approval status."""
    return get_operator_development_proposal(repository_root(), proposal_id)


@server.tool(
    name="check_validation_environment",
    title="Check RISC-V validation environment",
    description=(
        "Check the host-configured RISC-V machine, repository, Python, Triton, "
        "Triton-Shared, and Buddy tools. The model cannot choose the host."
    ),
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
    structured_output=True,
)
def check_validation_environment_tool() -> RemotePreflightResult:
    return check_remote_environment()


@server.tool(
    name="discover_operator",
    title="Discover Triton-RISCV operator",
    description=(
        "Inspect one existing FlagGems operator and return its implementation file, "
        "mapped tests, validation command, Torch references, Triton operations, "
        "and static risk hints. This tool never executes tests or modifies files."
    ),
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
    structured_output=True,
)
def discover_operator_tool(operator_name: str) -> DiscoverOperatorResult:
    """Discover repository evidence for an exact operator name."""

    return discover_operator_evidence(repository_root(), operator_name)


@server.tool(name="read_operator_file", structured_output=True,
             annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False,
                                         idempotent_hint=True, open_world_hint=False))
def read_operator_file_tool(path: str, start_line: int = 1, max_lines: int = 200) -> OperatorFileResult:
    """Read repository-relative FlagGems .py source/tests or tasks/operators .md. No execution. At most 400 lines/32 KiB per page; use next_line for more. Returns a source hash, not validation proof."""
    return read_operator_file(repository_root(), path, start_line, max_lines)


@server.tool(
    name="prepare_operator_development",
    title="Prepare a new Triton-RISCV operator",
    description=(
        "Validate a structured semantic contract, select nearby repository "
        "references, and prepare a development task. This does not create tracked files."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def prepare_operator_development_tool(
    specification: NewOperatorSpec,
) -> DevelopmentPreparationToolResult:
    root = repository_root()
    try:
        plan = prepare_operator_development(root, specification)
        return DevelopmentPreparationToolResult(**plan.model_dump())
    except ValueError as error:
        reason = str(error)
        if "new operator development refuses existing files:" in reason:
            existing_files = [
                item.strip()
                for item in reason.split(":", 1)[1].split(",")
                if item.strip()
            ]
            evidence = discover_operator_evidence(root, specification.name)
            return DevelopmentPreparationToolResult(
                operator=specification.name,
                status="existing_target",
                existing_files=existing_files,
                existing_operator=evidence.model_dump(),
                blocked_reason=reason,
                next_action=(
                    "Do not retry new-operator preparation or overwrite existing files. "
                    "Use discover_operator, then call validate_operator with execute=false."
                ),
            )
        return DevelopmentPreparationToolResult(
            operator=specification.name,
            status="blocked",
            blocked_reason=reason,
            next_action=(
                "Correct the semantic contract using the reported reason, then call "
                "prepare_operator_development once more."
            ),
        )


@server.tool(
    name="propose_operator_implementation",
    title="Propose a new operator implementation",
    description=(
        "Audit generated implementation and pytest source against a prepared "
        "contract, then store a reviewable proposal without changing tracked files."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def propose_operator_implementation_tool(
    development_id: str,
    implementation_source: str,
    test_source: str,
    rationale: str,
) -> DevelopmentProposalResult:
    return propose_operator_implementation(
        repository_root(),
        development_id,
        implementation_source,
        test_source,
        rationale,
    )


@server.tool(
    name="apply_operator_implementation",
    title="Apply an approved new operator",
    description=(
        "Create implementation, test, and task files from a separately approved "
        "proposal. The host must enable permissions.development in plugin configuration."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def apply_operator_implementation_tool(
    proposal_id: str,
) -> ApplyDevelopmentResult:
    return apply_operator_implementation(repository_root(), proposal_id)


@server.tool(
    name="validate_operator",
    title="Validate Triton-RISCV operator",
    description=(
        "Create a validation plan by default. Set execute=true only after user "
        "approval; the host must also enable permissions.validation in plugin configuration."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def validate_operator_tool(
    operator_name: str,
    execute: bool = False,
    approved_run_id: str | None = None,
    source_env: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    from codex_agent.runtime_config import runtime_config
    if execute and runtime_config().queue.enabled:
        if not approved_run_id:
            raise PermissionError("an approved validation plan is required")
        return _submit_validation_task(repository_root(), operator_name, approved_run_id)
    return validate_operator_target(
        repository_root(),
        operator_name,
        execute=execute,
        approved_run_id=approved_run_id,
        source_env=source_env,
        timeout_seconds=timeout_seconds,
    ).model_dump()


@server.tool(
    name="diagnose_failure",
    title="Diagnose validation failure",
    description="Classify one stored validation receipt and recommend the next action.",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
    structured_output=True,
)
def diagnose_failure_tool(run_id: str) -> DiagnosisToolResult:
    return diagnose_failure_run(repository_root(), run_id)


@server.tool(
    name="propose_repair",
    title="Propose operator source repair",
    description=(
        "Store a replacement for one failed operator implementation as a pending "
        "proposal. This tool does not modify source or tests."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def propose_repair_tool(
    run_id: str,
    replacement_source: str,
    rationale: str,
) -> RepairProposalResult:
    return propose_operator_repair(
        repository_root(),
        run_id,
        replacement_source,
        rationale,
    )


@server.tool(
    name="apply_repair",
    title="Apply approved operator repair",
    description=(
        "Apply a separately approved repair proposal. The host must enable "
        "permissions.repair in plugin configuration; tests are integrity locked."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=False,
        open_world_hint=False,
    ),
    structured_output=True,
)
def apply_repair_tool(proposal_id: str) -> ApplyRepairResult:
    return apply_operator_repair(repository_root(), proposal_id)


def main() -> None:
    """Run the local MCP server over stdio for DeepSeek Harness."""

    from codex_agent.runtime_config import runtime_config
    runtime_config()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
