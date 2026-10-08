#!/usr/bin/env python3
"""Validate discovered operators and classify the results."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from codex_agent.process_control import run_bounded, validation_environment
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from .failure_diagnosis import diagnose_log
    from .pipeline_observer import (
        build_pipeline_report,
        collect_artifacts,
        collect_environment,
        load_stage_events,
        render_pipeline_markdown,
        write_json,
    )
except ImportError:  # Allow direct script execution from the repository root.
    from failure_diagnosis import diagnose_log  # type: ignore
    from pipeline_observer import (  # type: ignore
        build_pipeline_report,
        collect_artifacts,
        collect_environment,
        load_stage_events,
        render_pipeline_markdown,
        write_json,
    )


DEFAULT_OPERATORS = Path("agent-results/operators.json")
DEFAULT_RESULTS_DIR = Path("agent-results")


@dataclass
class OperatorValidationResult:
    operator: str
    implementation_file: str
    test_files: list[str]
    command: str
    dry_run: bool
    exit_code: int | None
    status: str
    failure_stage: str | None
    likely_reason: str | None
    error_excerpt: list[str]
    duration_seconds: float
    log_path: str | None
    run_id: str = ""
    fresh_compile: bool = False
    artifact_dir: str | None = None
    pipeline_report_path: str | None = None
    first_failure_stage: str | None = None
    execution_mode: str = "unknown"
    correctness: str = "unknown"
    stages: list[dict] = field(default_factory=list)
    diagnosis: dict = field(default_factory=dict)
    isolation: dict = field(default_factory=dict)


def load_operators(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("operators", [])


def load_completed_operators(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") in {"passed", "failed", "skipped"}:
            completed.add(item["operator"])
    return completed


def find_operator(operators: list[dict], name: str) -> dict:
    for operator in operators:
        if operator["name"] == name:
            return operator
    available = ", ".join(item["name"] for item in operators)
    raise SystemExit(f"operator {name!r} not found; available: {available}")


def slugify(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_").replace(":", "_")


def build_shell_command(command: str, source_env: bool) -> str:
    if not source_env:
        return command
    return f"source scripts/triton-riscv-env.sh && {command}"


def operator_visibility(operator: dict) -> str:
    return operator.get(
        "visibility",
        "internal" if operator["name"].startswith("_") else "public",
    )


def operator_matches(
    operator: dict,
    contains: str | None,
    visibility: str | None,
) -> bool:
    if contains and contains not in operator["name"]:
        return False
    if visibility and operator_visibility(operator) != visibility:
        return False
    return True


def extract_error_excerpt(text: str, max_lines: int = 8) -> list[str]:
    diagnosis = diagnose_log(text, 1, max_evidence=max_lines)
    return [item["text"] for item in diagnosis["evidence"]]


def classify_log(text: str, exit_code: int | None) -> tuple[str, str | None, str | None]:
    diagnosis = diagnose_log(text, exit_code)
    return (
        diagnosis["status"],
        diagnosis["failure_stage"],
        diagnosis["summary"],
    )


def run_operator(
    operator: dict,
    *,
    repo_root: Path,
    results_dir: Path,
    source_env: bool,
    dry_run: bool,
    timeout_seconds: int,
    capture_pipeline: bool = True,
    fresh_compile: bool = False,
    environment: dict | None = None,
    run_id: str | None = None,
) -> OperatorValidationResult:
    run_id = run_id or (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{time.time_ns() % 1_000_000_000:09d}-{slugify(operator['name'])}"
    )
    run_dir = results_dir / "runs" / run_id
    dump_dir = run_dir / "artifacts"
    raw_command = operator["validation_command"]
    exports: list[str] = []
    if capture_pipeline:
        exports.append(
            f"export TRITON_SHARED_DUMP_PATH={shlex.quote(dump_dir.as_posix())}"
        )
    if fresh_compile:
        exports.append(
            f"export TRITON_CACHE_DIR={shlex.quote((run_dir / 'cache').as_posix())}"
        )
    if exports and raw_command:
        raw_command = " && ".join([*exports, raw_command])
    command = build_shell_command(raw_command, source_env)
    start = time.monotonic()
    log_path: Path | None = None
    exit_code: int | None = None
    output = ""

    if not operator["validation_command"]:
        status = "skipped"
        failure_stage = "selection"
        likely_reason = "operator has no validation command"
        diagnosis = {
            "schema_version": 1,
            "status": status,
            "rule_id": "missing-validation-command",
            "category": "selection",
            "failure_stage": failure_stage,
            "pipeline_stage": "selection",
            "summary": likely_reason,
            "confidence": 1.0,
            "repair_scope": "test-definition",
            "failed_command": None,
            "evidence": [],
            "recommended_actions": [
                "Add or map an acceptance test before attempting validation."
            ],
        }
        duration = time.monotonic() - start
        return OperatorValidationResult(
            operator=operator["name"],
            implementation_file=operator["implementation_file"],
            test_files=operator["test_files"],
            command=command,
            dry_run=dry_run,
            exit_code=None,
            status=status,
            failure_stage=failure_stage,
            likely_reason=likely_reason,
            error_excerpt=[],
            duration_seconds=round(duration, 3),
            log_path=None,
            run_id=run_id,
            fresh_compile=fresh_compile,
            artifact_dir=None,
            pipeline_report_path=None,
            first_failure_stage="selection",
            execution_mode=(environment or {}).get("execution", {}).get("mode", "unknown"),
            correctness="not-established",
            stages=[],
            diagnosis=diagnosis,
        )

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"
        try:
            completed = run_bounded(
                ["bash", "-lc", command],
                cwd=repo_root,
                env=validation_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            output = stdout + stderr + f"\nTIMEOUT after {timeout_seconds} seconds\n"
            exit_code = 124
        log_path.write_text(output, encoding="utf-8")

    artifacts = collect_artifacts(dump_dir) if not dry_run else []
    stage_events = load_stage_events(dump_dir) if not dry_run else []
    diagnosis = diagnose_log(
        output,
        exit_code,
        stage_events=stage_events,
        validation_command=command,
    )
    status = diagnosis["status"]
    failure_stage = diagnosis["failure_stage"]
    likely_reason = diagnosis["summary"]
    error_excerpt = [item["text"] for item in diagnosis["evidence"]]
    pipeline_report = build_pipeline_report(
        operator=operator,
        validation_status=status,
        failure_stage=failure_stage,
        likely_reason=likely_reason,
        exit_code=exit_code,
        artifacts=artifacts,
        stage_events=stage_events,
        environment=environment,
        diagnosis=diagnosis,
    )
    pipeline_report_path: Path | None = None
    if not dry_run:
        pipeline_report_path = run_dir / "pipeline.json"
        write_json(pipeline_report_path, pipeline_report)
        (run_dir / "report.md").write_text(
            render_pipeline_markdown(pipeline_report),
            encoding="utf-8",
        )
    duration = time.monotonic() - start
    result = OperatorValidationResult(
        operator=operator["name"],
        implementation_file=operator["implementation_file"],
        test_files=operator["test_files"],
        command=command,
        dry_run=dry_run,
        exit_code=exit_code,
        status=status,
        failure_stage=failure_stage,
        likely_reason=likely_reason,
        error_excerpt=error_excerpt,
        duration_seconds=round(duration, 3),
        log_path=log_path.as_posix() if log_path else None,
        run_id=run_id,
        fresh_compile=fresh_compile,
        artifact_dir=dump_dir.as_posix() if not dry_run else None,
        pipeline_report_path=(
            pipeline_report_path.as_posix() if pipeline_report_path else None
        ),
        first_failure_stage=pipeline_report["first_failure_stage"],
        execution_mode=pipeline_report["execution_mode"],
        correctness=pipeline_report["correctness"],
        stages=pipeline_report["stages"],
        diagnosis=diagnosis,
    )
    if not dry_run:
        write_json(run_dir / "result.json", asdict(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate operator targets discovered by discover_operators.py."
    )
    parser.add_argument(
        "operator",
        nargs="?",
        help="Operator name, for example silu_and_mul.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--operators",
        default=DEFAULT_OPERATORS.as_posix(),
        help="Operator discovery JSON.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR.as_posix(),
        help="Directory for logs and result JSONL files.",
    )
    parser.add_argument(
        "--source-env",
        action="store_true",
        help="Source scripts/triton-riscv-env.sh before validation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered operators and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all discovered operators, optionally filtered by --contains.",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Filter operator names when using --list or --all.",
    )
    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--public-only",
        action="store_true",
        help="Select public operators whose names do not start with an underscore.",
    )
    visibility_group.add_argument(
        "--internal-only",
        action="store_true",
        help="Select internal operators whose names start with an underscore.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of selected operators.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--capture-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture compiler IR artifacts and write a per-stage report.",
    )
    parser.add_argument(
        "--fresh-compile",
        action="store_true",
        help="Use an isolated Triton cache so compiler stages run again.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Append to an existing JSONL batch and skip operators that already "
            "have passed, failed, or skipped results."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    operators_path = Path(args.operators)
    if not operators_path.is_absolute():
        operators_path = repo_root / operators_path
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(
        repo_root,
        source_env=args.source_env,
        dry_run=args.dry_run,
    )
    environment_path = results_dir / "environment.json"
    write_json(environment_path, environment)

    visibility = None
    if args.public_only:
        visibility = "public"
    elif args.internal_only:
        visibility = "internal"

    operators = [
        operator
        for operator in load_operators(operators_path)
        if operator_matches(operator, args.contains, visibility)
    ]
    if args.limit is not None:
        operators = operators[: args.limit]

    if args.list:
        for operator in operators:
            print(
                f"{operator['name']}\t"
                f"visibility={operator_visibility(operator)}\t"
                f"tests={len(operator['test_nodes'])}\t"
                f"tl_ops={','.join(operator['tl_ops'])}"
            )
        return 0

    if args.all:
        selected = operators
    elif args.operator:
        selected = [find_operator(load_operators(operators_path), args.operator)]
    else:
        raise SystemExit("pass an operator name, or use --list, or use --all")

    if args.resume_from:
        result_path = Path(args.resume_from)
        if not result_path.is_absolute():
            result_path = repo_root / result_path
        completed_operators = load_completed_operators(result_path)
        original_count = len(selected)
        selected = [
            operator
            for operator in selected
            if operator["name"] not in completed_operators
        ]
        print(
            f"resume: skipped {original_count - len(selected)} completed operators"
        )
        file_mode = "a"
    else:
        if not selected:
            raise SystemExit("no operators matched the selection")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = slugify(selected[0]["name"]) if len(selected) == 1 else "batch"
        result_path = results_dir / f"operator-validation-{timestamp}-{suffix}.jsonl"
        file_mode = "w"

    if not selected:
        print(f"nothing to run; all selected operators are recorded in {result_path}")
        return 0

    result_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with result_path.open(file_mode, encoding="utf-8") as handle:
        for operator in selected:
            result = run_operator(
                operator,
                repo_root=repo_root,
                results_dir=results_dir,
                source_env=args.source_env,
                dry_run=args.dry_run,
                timeout_seconds=args.timeout_seconds,
                capture_pipeline=args.capture_pipeline,
                fresh_compile=args.fresh_compile,
                environment=environment,
            )
            result_json = json.dumps(asdict(result), sort_keys=True)
            handle.write(result_json + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            if result.exit_code not in (0, None):
                failures += 1

    try:
        display_path = result_path.relative_to(repo_root)
    except ValueError:
        display_path = result_path
    print(f"wrote {display_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
