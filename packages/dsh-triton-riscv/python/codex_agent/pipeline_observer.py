#!/usr/bin/env python3
"""Collect compiler-pipeline, hardware, and correctness evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import time
from pathlib import Path


PIPELINE_STAGES = (
    {
        "id": "ttir",
        "name": "Python/Triton frontend to TTIR",
        "artifacts": ("tt.mlir",),
    },
    {
        "id": "linalg-mlir",
        "name": "Triton-Shared to Linalg MLIR",
        "artifacts": ("ttshared.mlir",),
    },
    {
        "id": "llvm-mlir",
        "name": "Buddy lowering to LLVM MLIR",
        "artifacts": ("ll.mlir",),
    },
    {
        "id": "llvm-ir",
        "name": "MLIR translation to LLVM IR",
        "artifacts": ("ll.ir", "kernel.ll"),
    },
    {
        "id": "riscv-object",
        "name": "LLVM code generation to RISC-V object",
        "artifacts": ("kernel.o",),
    },
    {
        "id": "link-load",
        "name": "Object linking and loading",
        "artifacts": (),
    },
    {
        "id": "hardware-capability",
        "name": "RISC-V hardware capability",
        "artifacts": (),
    },
    {
        "id": "runtime",
        "name": "RISC-V runtime execution",
        "artifacts": (),
    },
    {
        "id": "correctness",
        "name": "PyTorch numerical comparison",
        "artifacts": (),
    },
)

FAILURE_STAGE_MAP = {
    "triton-frontend": "ttir",
    "triton-shared-opt": "linalg-mlir",
    "buddy-opt": "llvm-mlir",
    "mlir-translate": "llvm-ir",
    "llc": "riscv-object",
    "link": "link-load",
    "target-capability": "hardware-capability",
    "runtime": "runtime",
    "correctness": "correctness",
}

EVENT_STAGE_PREFIXES = (
    ("triton-shared-opt", "linalg-mlir"),
    ("buddy-opt", "llvm-mlir"),
    ("buddy-translate", "llvm-ir"),
    ("mlir-translate", "llvm-ir"),
    ("llvm-opt", "riscv-object"),
    ("clang-codegen", "riscv-object"),
    ("buddy-llc", "riscv-object"),
    ("llc", "riscv-object"),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _shell_command(command: str, source_env: bool) -> str:
    if not source_env:
        return command
    helper = shlex.quote("scripts/triton-riscv-env.sh")
    return f"source {helper} >/dev/null 2>&1 && {command}"


def _probe(
    repo_root: Path,
    command: str,
    *,
    source_env: bool,
    timeout_seconds: int = 20,
) -> dict:
    shell_command = _shell_command(command, source_env)
    try:
        completed = subprocess.run(
            ["bash", "-lc", shell_command],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "command": shell_command,
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return {
            "command": shell_command,
            "exit_code": 124,
            "output": output.strip(),
        }


def collect_environment(
    repo_root: Path,
    *,
    source_env: bool,
    dry_run: bool = False,
) -> dict:
    """Collect the software toolchain and execution-target snapshot."""
    if dry_run:
        return {
            "schema_version": 1,
            "status": "not-checked",
            "reason": "dry-run",
            "captured_at": None,
            "architecture": None,
            "execution": {"mode": "not-checked", "native_riscv": None},
            "tools": {},
        }

    architecture_probe = _probe(repo_root, "uname -m", source_env=source_env)
    system_probe = _probe(repo_root, "uname -srm", source_env=source_env)
    python_probe = _probe(
        repo_root,
        "python -c 'import platform, sys; "
        "print(platform.python_version()); print(sys.executable)'",
        source_env=source_env,
    )
    triton_probe = _probe(
        repo_root,
        "python -c 'import triton; print(triton.__version__)'",
        source_env=source_env,
    )
    cpu_probe = _probe(
        repo_root,
        "(command -v lscpu >/dev/null && lscpu) || sed -n '1,80p' /proc/cpuinfo",
        source_env=source_env,
    )
    target_probe = _probe(
        repo_root,
        "printf '%s\\n' \"${TRITON_RISCV_LOWERING_MODE:-}\" "
        "\"${TRITON_RISCV_CC:-}\" \"${TRITON_RISCV_QEMU:-}\"",
        source_env=source_env,
    )

    tools: dict[str, dict] = {}
    tool_commands = {
        "triton-shared-opt": "triton-shared-opt --version",
        "buddy-opt": "buddy-opt --version",
        "mlir-translate": "mlir-translate --version",
        "llc": "llc --version",
    }
    optional_commands = {
        "riscv-cc": "command -v \"${TRITON_RISCV_CC:-riscv64-unknown-linux-gnu-gcc}\"",
        "riscv-objdump": "command -v \"${TRITON_RISCV_OBJDUMP:-riscv64-unknown-linux-gnu-objdump}\"",
        "qemu-riscv64": "command -v \"${TRITON_RISCV_QEMU:-qemu-riscv64}\"",
    }
    for name, version_command in tool_commands.items():
        path_probe = _probe(
            repo_root,
            f"command -v {shlex.quote(name)}",
            source_env=source_env,
        )
        version_probe = _probe(
            repo_root,
            version_command,
            source_env=source_env,
        )
        tools[name] = {
            "available": path_probe["exit_code"] == 0,
            "path": path_probe["output"].splitlines()[0] if path_probe["output"] else None,
            "version": version_probe["output"].splitlines()[0]
            if version_probe["output"]
            else None,
        }
    for name, command in optional_commands.items():
        probe = _probe(repo_root, command, source_env=source_env)
        tools[name] = {
            "available": probe["exit_code"] == 0,
            "path": probe["output"].splitlines()[0] if probe["output"] else None,
            "version": None,
        }

    architecture = architecture_probe["output"].splitlines()[0]
    architecture = architecture if architecture_probe["exit_code"] == 0 else platform.machine()
    native_riscv = architecture.lower() in {"riscv64", "riscv"}
    qemu_available = tools["qemu-riscv64"]["available"]
    if native_riscv:
        execution_mode = "native-riscv"
    else:
        execution_mode = "non-riscv-host"

    required_tools = tuple(tool_commands)
    software_ready = (
        python_probe["exit_code"] == 0
        and triton_probe["exit_code"] == 0
        and all(tools[name]["available"] for name in required_tools)
    )
    return {
        "schema_version": 1,
        "status": "ready" if software_ready else "incomplete",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "architecture": architecture,
        "system": system_probe["output"] or None,
        "cpu_details": cpu_probe["output"] or None,
        "python": {
            "available": python_probe["exit_code"] == 0,
            "version": python_probe["output"].splitlines()[0]
            if python_probe["output"]
            else None,
            "executable": python_probe["output"].splitlines()[1]
            if len(python_probe["output"].splitlines()) > 1
            else None,
        },
        "triton": {
            "available": triton_probe["exit_code"] == 0,
            "version": triton_probe["output"].splitlines()[0]
            if triton_probe["output"]
            else None,
        },
        "execution": {
            "mode": execution_mode,
            "native_riscv": native_riscv,
            "qemu_available": qemu_available,
            "hardware_execution_established": native_riscv,
            "configured_target": target_probe["output"].splitlines(),
        },
        "tools": tools,
    }


def _artifact_stage(path: Path) -> str | None:
    name = path.name
    for stage in PIPELINE_STAGES:
        if name in stage["artifacts"]:
            return stage["id"]
    return None


def _elf_metadata(data: bytes) -> dict:
    if len(data) < 20 or data[:4] != b"\x7fELF":
        return {}
    byte_order = "little" if data[5] == 1 else "big"
    machine_id = int.from_bytes(data[18:20], byte_order)
    machine_names = {62: "x86-64", 183: "AArch64", 243: "RISC-V"}
    return {
        "format": "ELF",
        "elf_class": 64 if data[4] == 2 else 32,
        "machine_id": machine_id,
        "machine": machine_names.get(machine_id, f"unknown-{machine_id}"),
        "is_riscv": machine_id == 243,
    }


def collect_artifacts(dump_dir: Path) -> list[dict]:
    if not dump_dir.exists():
        return []
    artifacts: list[dict] = []
    for path in sorted(item for item in dump_dir.rglob("*") if item.is_file()):
        data = path.read_bytes()
        artifact = {
            "name": path.name,
            "path": path.as_posix(),
            "stage": _artifact_stage(path),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if path.suffix == ".o":
            artifact.update(_elf_metadata(data))
        artifacts.append(artifact)
    return artifacts


def load_stage_events(dump_dir: Path) -> list[dict]:
    path = dump_dir / "stage-events.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        tool_stage = str(event.get("stage", ""))
        event["pipeline_stage"] = next(
            (
                pipeline_stage
                for prefix, pipeline_stage in EVENT_STAGE_PREFIXES
                if tool_stage.startswith(prefix)
            ),
            None,
        )
        events.append(event)
    return events


def build_pipeline_report(
    *,
    operator: dict,
    validation_status: str,
    failure_stage: str | None,
    likely_reason: str | None,
    exit_code: int | None,
    artifacts: list[dict],
    stage_events: list[dict] | None = None,
    environment: dict | None = None,
    diagnosis: dict | None = None,
) -> dict:
    """Build an evidence-based stage report for one validation attempt."""
    artifact_stages = {
        item["stage"] for item in artifacts if item.get("stage") is not None
    }
    stage_events = stage_events or []
    failed_pipeline_stage = FAILURE_STAGE_MAP.get(failure_stage or "")
    if failed_pipeline_stage is None:
        failed_pipeline_stage = next(
            (
                item.get("pipeline_stage")
                for item in stage_events
                if item.get("status") == "failed" and item.get("pipeline_stage")
            ),
            None,
        )
    stage_ids = [item["id"] for item in PIPELINE_STAGES]
    failed_index = (
        stage_ids.index(failed_pipeline_stage)
        if failed_pipeline_stage in stage_ids
        else None
    )
    has_numerical_assertion = bool(
        operator.get("test_contract", {}).get("numerical_assertion")
    )
    hardware_execution_established = bool(
        (environment or {})
        .get("execution", {})
        .get("hardware_execution_established")
    )

    stages: list[dict] = []
    for index, definition in enumerate(PIPELINE_STAGES):
        stage_id = definition["id"]
        evidence = [
            item["path"] for item in artifacts if item.get("stage") == stage_id
        ]
        if stage_id == "hardware-capability":
            architecture = (environment or {}).get("architecture")
            if architecture:
                evidence.append(f"architecture={architecture}")
            evidence.extend(
                f"{item['path']}: ELF {item.get('machine')}"
                for item in artifacts
                if item.get("is_riscv")
            )
        events = [
            item for item in stage_events if item.get("pipeline_stage") == stage_id
        ]
        event_failed = any(item.get("status") == "failed" for item in events)
        event_passed = bool(events) and not event_failed
        if failed_index is not None:
            if index < failed_index:
                status = "passed" if evidence or event_passed else "passed-inferred"
            elif index == failed_index:
                status = "failed"
            else:
                status = "not-run"
        elif validation_status == "passed":
            if stage_id == "hardware-capability":
                status = "passed" if hardware_execution_established else "not-established"
            elif stage_id in {"runtime", "correctness"}:
                if stage_id == "correctness" and not has_numerical_assertion:
                    status = "not-established"
                else:
                    status = "passed"
            else:
                status = (
                    "passed"
                    if evidence or event_passed
                    else "passed-cached-or-inferred"
                )
        elif validation_status == "planned":
            status = "planned"
        elif stage_id == "runtime" and failure_stage in {"pytest", "unknown"}:
            status = "failed-or-not-established"
        else:
            status = "not-established"
        stages.append(
            {
                "id": stage_id,
                "name": definition["name"],
                "status": status,
                "evidence": evidence,
                "commands": [item.get("command", []) for item in events],
                "duration_seconds": round(
                    sum(float(item.get("duration_seconds", 0)) for item in events),
                    6,
                ),
            }
        )

    execution_mode = (environment or {}).get("execution", {}).get("mode", "unknown")
    correctness = "not-established"
    if validation_status == "passed" and has_numerical_assertion:
        correctness = "basic-tests-passed"
    elif failure_stage == "correctness":
        correctness = "basic-tests-failed"

    return {
        "schema_version": 1,
        "operator": operator["name"],
        "validation_status": validation_status,
        "exit_code": exit_code,
        "first_failure_stage": failed_pipeline_stage or failure_stage,
        "likely_reason": likely_reason,
        "diagnosis": diagnosis or {},
        "execution_mode": execution_mode,
        "correctness": correctness,
        "test_contract": operator.get("test_contract", {}),
        "stages": stages,
        "stage_events": stage_events,
        "artifacts": artifacts,
        "environment": {
            "status": (environment or {}).get("status"),
            "architecture": (environment or {}).get("architecture"),
            "execution": (environment or {}).get("execution", {}),
            "python": (environment or {}).get("python", {}),
            "triton": (environment or {}).get("triton", {}),
        },
    }


def render_pipeline_markdown(report: dict) -> str:
    lines = [
        f"# Operator Validation: {report['operator']}",
        "",
        f"- Result: {report['validation_status']}",
        f"- First failure stage: {report.get('first_failure_stage') or 'none'}",
        f"- Likely reason: {report.get('likely_reason') or 'none'}",
        f"- Execution mode: {report.get('execution_mode') or 'unknown'}",
        f"- Correctness evidence: {report.get('correctness') or 'not-established'}",
        f"- Architecture: {report.get('environment', {}).get('architecture') or 'unknown'}",
        "",
        "## Pipeline Stages",
        "",
        "| Stage | Status | Tool time (s) | Evidence |",
        "| --- | --- | ---: | --- |",
    ]
    for stage in report["stages"]:
        evidence = ", ".join(stage["evidence"]) or ""
        lines.append(
            f"| {stage['name']} | {stage['status']} | "
            f"{stage.get('duration_seconds', 0)} | {evidence} |"
        )

    diagnosis = report.get("diagnosis") or {}
    lines.extend(["", "## Failure Diagnosis", ""])
    if diagnosis.get("status") == "failed":
        lines.extend(
            [
                f"- Rule: {diagnosis.get('rule_id') or 'unknown'}",
                f"- Category: {diagnosis.get('category') or 'unknown'}",
                f"- Confidence: {diagnosis.get('confidence', 0):.2f}",
                f"- Repair scope: {diagnosis.get('repair_scope') or 'human-triage'}",
            ]
        )
        if diagnosis.get("failed_command"):
            lines.extend(
                [
                    "",
                    "### Failed Command",
                    "",
                    "```sh",
                    str(diagnosis["failed_command"]),
                    "```",
                ]
            )
        lines.extend(["", "### Log Evidence", ""])
        evidence = diagnosis.get("evidence") or []
        if evidence:
            for item in evidence:
                lines.append(
                    f"- Line {item.get('line_number', '?')}: `{item.get('text', '')}`"
                )
        else:
            lines.append("No diagnostic evidence line was captured.")
        lines.extend(["", "### Recommended Actions", ""])
        for action in diagnosis.get("recommended_actions") or []:
            lines.append(f"- {action}")
    else:
        lines.append("No failure diagnosis is required for this validation result.")

    commands = [event for event in report.get("stage_events", []) if event.get("command")]
    lines.extend(["", "## Compiler Commands", ""])
    if commands:
        for event in commands:
            command = " ".join(shlex.quote(str(item)) for item in event["command"])
            lines.extend(
                [
                    f"### {event.get('stage', 'compiler stage')}",
                    "",
                    f"- Status: {event.get('status', 'unknown')}",
                    f"- Duration: {event.get('duration_seconds', 0)} seconds",
                    "",
                    "```sh",
                    command,
                    "```",
                    "",
                ]
            )
    else:
        lines.append("No fresh compiler command was observed in this run.")

    contract = report.get("test_contract") or {}
    lines.extend(
        [
            "",
            "## Test Evidence",
            "",
            f"- PyTorch reference found: {bool(contract.get('pytorch_reference'))}",
            f"- Numerical assertion found: {bool(contract.get('numerical_assertion'))}",
            f"- Backward coverage signal: {bool(contract.get('backward'))}",
            f"- Broadcast coverage signal: {bool(contract.get('broadcast'))}",
            f"- Dtypes mentioned: {', '.join(contract.get('dtypes', [])) or 'none detected'}",
            "",
            "## Interpretation",
            "",
            "`passed-cached-or-inferred` means the final test passed but this run did not",
            "produce a fresh artifact for that compiler stage. Use `--fresh-compile` when",
            "a complete per-stage evidence trail is required.",
        ]
    )
    return "\n".join(lines) + "\n"
