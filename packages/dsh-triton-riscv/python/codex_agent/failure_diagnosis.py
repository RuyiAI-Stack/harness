"""Evidence-based diagnosis for Triton-RISCV validation logs."""

from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass
from typing import Any, Iterable


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
TEMP_PATH_RE = re.compile(r"/tmp/tmp[^/\s'\"]+")
EVENT_STAGE_MAP = (
    ("triton-shared-opt", "triton-shared-opt", "linalg-mlir"),
    ("buddy-opt", "buddy-opt", "llvm-mlir"),
    ("buddy-translate", "mlir-translate", "llvm-ir"),
    ("mlir-translate", "mlir-translate", "llvm-ir"),
    ("llvm-opt", "llc", "riscv-object"),
    ("clang-codegen", "llc", "riscv-object"),
    ("buddy-llc", "llc", "riscv-object"),
    ("llc", "llc", "riscv-object"),
)


@dataclass(frozen=True)
class DiagnosisRule:
    rule_id: str
    failure_stage: str
    pipeline_stage: str
    category: str
    summary: str
    confidence: float
    repair_scope: str
    match_patterns: tuple[str, ...]
    evidence_patterns: tuple[str, ...]
    recommended_actions: tuple[str, ...]

    def matches(self, text: str) -> bool:
        return all(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in self.match_patterns)


RULES = (
    DiagnosisRule(
        "environment-helper-missing",
        "environment",
        "environment",
        "environment",
        "The Triton-RISCV environment helper could not be found.",
        0.99,
        "environment",
        (r"triton-riscv-env\.sh", r"no such file or directory"),
        (r"triton-riscv-env\.sh.*no such file or directory",),
        (
            "Verify the repository path and scripts/triton-riscv-env.sh.",
            "Source the environment helper before running validation again.",
        ),
    ),
    DiagnosisRule(
        "python-import-failed",
        "import",
        "environment",
        "dependency",
        "Python could not import a required module.",
        0.98,
        "environment",
        (r"(?:ModuleNotFoundError|ImportError)",),
        (r"(?:ModuleNotFoundError|ImportError).*",),
        (
            "Install the missing package in the Python environment used by validation.",
            "Confirm that the selected interpreter matches the Triton build environment.",
        ),
    ),
    DiagnosisRule(
        "triton-wheel-build-failed",
        "build",
        "environment",
        "dependency",
        "The Triton wheel or build dependency could not be built.",
        0.97,
        "environment",
        (r"(?:failed building wheel|building wheel for triton)",),
        (r".*(?:failed building wheel|building wheel for triton).*", r"error:.*"),
        (
            "Inspect the first dependency download or compiler error above the wheel failure.",
            "Repair the build environment before changing operator source.",
        ),
    ),
    DiagnosisRule(
        "fp8-target-unsupported",
        "target-capability",
        "hardware-capability",
        "target-capability",
        "The RISC-V target does not support the requested FP8 dtype.",
        0.99,
        "hardware",
        (r"fp8", r"not supported in this architecture"),
        (r".*fp8.*not supported in this architecture.*",),
        (
            "Confirm the operator contract requires FP8 on this target.",
            "Record the target limitation instead of weakening the acceptance test.",
        ),
    ),
    DiagnosisRule(
        "triton-math-api-missing",
        "triton-frontend",
        "ttir",
        "compiler-api",
        "The operator uses a Triton math API unavailable in this Triton version.",
        0.99,
        "operator-source",
        (r"triton\.language\.math", r"has no attribute"),
        (r"AttributeError:.*triton\.language\.math.*has no attribute.*",),
        (
            "Find a supported Triton primitive or an equivalent implementation in a nearby operator.",
            "Change only the implementation and preserve the acceptance test.",
        ),
    ),
    DiagnosisRule(
        "triton-frontend-compilation",
        "triton-frontend",
        "ttir",
        "compiler-frontend",
        "Triton frontend compilation failed.",
        0.82,
        "operator-source",
        (r"triton\.compiler\.errors\.CompilationError",),
        (r".*CompilationError.*", r".*error:.*"),
        (
            "Inspect the reported source location and unsupported Triton expression.",
            "Prepare a source-only repair and rerun the same acceptance test.",
        ),
    ),
    DiagnosisRule(
        "buddy-tptr-dialect-missing",
        "buddy-opt",
        "llvm-mlir",
        "compiler-capability",
        "Buddy could not load the tptr dialect produced by pointer lowering.",
        0.99,
        "compiler",
        (r"Dialect [`']tptr['`] not found",),
        (r".*Dialect [`']tptr['`] not found.*",),
        (
            "Preserve the failing IR and report the missing Buddy dialect registration.",
            "Do not rewrite operator source unless a reviewed compiler workaround is requested.",
        ),
    ),
    DiagnosisRule(
        "buddy-ttx-dialect-missing",
        "buddy-opt",
        "llvm-mlir",
        "compiler-capability",
        "Buddy could not load the ttx dialect produced by Triton-Shared.",
        0.99,
        "compiler",
        (r"Dialect [`']ttx['`] not found",),
        (r".*Dialect [`']ttx['`] not found.*",),
        (
            "Preserve the failing IR and report the missing Buddy dialect registration.",
            "Check whether the producer and consumer builds use compatible dialect versions.",
        ),
    ),
    DiagnosisRule(
        "buddy-linalg-reduce-unsupported",
        "buddy-opt",
        "llvm-mlir",
        "compiler-capability",
        "Buddy VIR lowering does not support the generated linalg.reduce form.",
        0.99,
        "compiler",
        (r"unsupported linalg\.reduce for -lower-linalg-to-vir",),
        (r".*unsupported linalg\.reduce for -lower-linalg-to-vir.*",),
        (
            "Save the linalg.reduce operation and its iterator/reduction shape as compiler evidence.",
            "Open compiler triage before considering an operator-level workaround.",
        ),
    ),
    DiagnosisRule(
        "pointer-sequence-unsupported",
        "triton-shared-opt",
        "linalg-mlir",
        "compiler-capability",
        "Pointer lowering could not represent an operation in the pointer sequence.",
        0.98,
        "compiler",
        (r"unexpected op in ptr sequence",),
        (r".*unexpected op in ptr sequence.*",),
        (
            "Capture TTIR and the failing pointer operation for Triton-Shared triage.",
            "Avoid changing the acceptance test or hiding the unsupported pointer pattern.",
        ),
    ),
    DiagnosisRule(
        "unlowered-linalg-translation",
        "mlir-translate",
        "llvm-ir",
        "compiler-capability",
        "An unlowered Linalg operation reached LLVM IR translation.",
        0.99,
        "compiler",
        (r"(?:Dialect [`']linalg['`] not found|linalg\.generic)",),
        (r".*(?:Dialect [`']linalg['`] not found|linalg\.generic).*",),
        (
            "Inspect ll.mlir and identify the pass that should have lowered the remaining Linalg operation.",
            "Treat this as a compiler-pipeline gap before proposing an operator workaround.",
        ),
    ),
    DiagnosisRule(
        "triton-shared-generic",
        "triton-shared-opt",
        "linalg-mlir",
        "compiler-lowering",
        "Triton-to-Linalg conversion failed in triton-shared-opt.",
        0.78,
        "compiler",
        (r"triton-shared-opt", r"(?:error:|failed)"),
        (r".*triton-shared-opt.*", r".*error:.*"),
        (
            "Capture TTIR and the exact triton-shared-opt command.",
            "Classify the unsupported operation before considering source changes.",
        ),
    ),
    DiagnosisRule(
        "buddy-lowering-generic",
        "buddy-opt",
        "llvm-mlir",
        "compiler-lowering",
        "Buddy MLIR lowering failed without a more specific known signature.",
        0.76,
        "compiler",
        (r"buddy-opt", r"(?:error:|failed)"),
        (r".*buddy-opt.*", r".*error:.*"),
        (
            "Capture the failing MLIR operation and complete buddy-opt command.",
            "Add a specific diagnosis rule after compiler triage confirms the root cause.",
        ),
    ),
    DiagnosisRule(
        "mlir-translation-generic",
        "mlir-translate",
        "llvm-ir",
        "compiler-lowering",
        "MLIR-to-LLVM IR translation failed without a more specific known signature.",
        0.75,
        "compiler",
        (r"mlir-translate", r"(?:error:|failed)"),
        (r".*mlir-translate.*", r".*error:.*"),
        (
            "Capture ll.mlir and the complete mlir-translate invocation.",
            "Identify the first operation that still lacks LLVM translation support.",
        ),
    ),
    DiagnosisRule(
        "llvm-codegen-generic",
        "llc",
        "riscv-object",
        "compiler-codegen",
        "LLVM RISC-V object generation failed.",
        0.78,
        "compiler",
        (r"(?:^|[\s/])llc(?:[\s:]|$)", r"(?:error:|failed)"),
        (r".*llc.*", r".*error:.*"),
        (
            "Preserve the LLVM IR and exact llc target flags.",
            "Confirm whether the failure is an unsupported instruction or malformed IR.",
        ),
    ),
    DiagnosisRule(
        "linker-failed",
        "link",
        "link-load",
        "linker",
        "RISC-V object linking or loading failed.",
        0.96,
        "environment",
        (r"(?:undefined reference|linker command failed|ld returned|can't link)",),
        (r".*(?:undefined reference|linker command failed|ld returned|can't link).*",),
        (
            "Identify the missing symbol or incompatible object in the link command.",
            "Verify runtime libraries and ABI flags before changing operator source.",
        ),
    ),
    DiagnosisRule(
        "illegal-instruction",
        "target-capability",
        "hardware-capability",
        "target-capability",
        "The generated program used an instruction unavailable on the execution target.",
        0.99,
        "hardware",
        (r"illegal instruction",),
        (r".*illegal instruction.*",),
        (
            "Compare generated ISA extensions with the server CPU capabilities.",
            "Retest with corrected target flags or report a code-generation defect.",
        ),
    ),
    DiagnosisRule(
        "runtime-crash",
        "runtime",
        "runtime",
        "runtime",
        "The compiled operator crashed during execution.",
        0.96,
        "operator-or-runtime",
        (r"(?:segmentation fault|core dumped)",),
        (r".*(?:segmentation fault|core dumped).*",),
        (
            "Reproduce with the smallest failing shape and retain the generated artifacts.",
            "Inspect bounds, masks, pointers, and runtime ABI before proposing a repair.",
        ),
    ),
    DiagnosisRule(
        "numerical-mismatch",
        "correctness",
        "correctness",
        "correctness",
        "The operator result differs from the PyTorch reference.",
        0.98,
        "operator-source",
        (r"(?:assert_close|mismatched elements)",),
        (r".*(?:assert_close|mismatched elements).*",),
        (
            "Compare the first mismatched values, dtype, shape, and configured tolerances.",
            "Repair the implementation without modifying the acceptance test or tolerance.",
        ),
    ),
    DiagnosisRule(
        "subprocess-compilation-failed",
        "compilation",
        "compilation",
        "compiler",
        "A compiler subprocess returned a nonzero exit code.",
        0.62,
        "human-triage",
        (r"subprocess\.CalledProcessError",),
        (r".*subprocess\.CalledProcessError.*", r".*error:.*"),
        (
            "Locate the earlier compiler error and failed command in the full log.",
            "Do not edit source until the concrete compiler stage is identified.",
        ),
    ),
    DiagnosisRule(
        "pytest-failed",
        "pytest",
        "runtime",
        "test",
        "Pytest reported one or more failing tests without a known lower-level signature.",
        0.55,
        "operator-or-test",
        (r"(?:^FAILED\s|\b\d+ failed\b)",),
        (r"^FAILED\s.*", r".*\b\d+ failed\b.*"),
        (
            "Inspect the first test failure and its traceback in the full log.",
            "Determine whether the test, implementation, or runtime is responsible.",
        ),
    ),
)


def _clean_line(line: str) -> str:
    return " ".join(ANSI_ESCAPE_RE.sub("", line).strip().split())


def _dedupe_key(line: str) -> str:
    normalized = TEMP_PATH_RE.sub("/tmp/<tmp>", line.lower())
    error_index = normalized.find("error:")
    return normalized[error_index:] if error_index >= 0 else normalized


def _evidence(
    lines: list[str],
    patterns: Iterable[str],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for line_number, raw_line in enumerate(lines, start=1):
            line = _clean_line(raw_line)
            if not line or not compiled.search(line):
                continue
            key = _dedupe_key(line)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "line_number": line_number,
                    "text": line[:500],
                    "matched_pattern": pattern,
                }
            )
            if len(result) >= max_items:
                return result
    return result


def _fallback_evidence(lines: list[str], max_items: int) -> list[dict[str, Any]]:
    nonempty = [
        (index, _clean_line(line))
        for index, line in enumerate(lines, start=1)
        if _clean_line(line)
    ]
    return [
        {"line_number": index, "text": line[:500], "matched_pattern": "fallback-tail"}
        for index, line in nonempty[-max_items:]
    ]


def _failed_event(stage_events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    return next((event for event in stage_events if event.get("status") == "failed"), None)


def _render_command(command: Any) -> str | None:
    if isinstance(command, str):
        return command or None
    if isinstance(command, list):
        return " ".join(shlex.quote(str(item)) for item in command)
    return None


def _event_stage(event: dict[str, Any]) -> tuple[str, str] | None:
    stage = str(event.get("stage", ""))
    for prefix, failure_stage, pipeline_stage in EVENT_STAGE_MAP:
        if stage.startswith(prefix):
            return failure_stage, str(event.get("pipeline_stage") or pipeline_stage)
    return None


def diagnose_log(
    text: str,
    exit_code: int | None,
    *,
    stage_events: Iterable[dict[str, Any]] = (),
    validation_command: str | None = None,
    max_evidence: int = 8,
) -> dict[str, Any]:
    """Return a serializable diagnosis with cited log evidence."""

    if max_evidence < 1:
        raise ValueError("max_evidence must be positive")
    events = list(stage_events)
    failed_event = _failed_event(events)
    failed_command = _render_command((failed_event or {}).get("command")) or validation_command

    if exit_code is None:
        return {
            "schema_version": 1,
            "status": "planned",
            "rule_id": None,
            "category": None,
            "failure_stage": None,
            "pipeline_stage": None,
            "summary": None,
            "confidence": 1.0,
            "repair_scope": "none",
            "failed_command": failed_command,
            "evidence": [],
            "recommended_actions": ["Run the approved validation command before diagnosing."],
        }
    if exit_code == 0:
        return {
            "schema_version": 1,
            "status": "passed",
            "rule_id": None,
            "category": None,
            "failure_stage": None,
            "pipeline_stage": None,
            "summary": None,
            "confidence": 1.0,
            "repair_scope": "none",
            "failed_command": None,
            "evidence": [],
            "recommended_actions": ["Preserve the passing receipt as validation evidence."],
        }

    lines = text.splitlines()
    if exit_code == 124:
        evidence = _evidence(lines, (r".*TIMEOUT.*",), max_items=max_evidence)
        return {
            "schema_version": 1,
            "status": "failed",
            "rule_id": "validation-timeout",
            "category": "timeout",
            "failure_stage": "timeout",
            "pipeline_stage": "execution",
            "summary": "The validation command exceeded its configured timeout.",
            "confidence": 1.0,
            "repair_scope": "retry-policy",
            "failed_command": failed_command,
            "evidence": evidence or _fallback_evidence(lines, max_evidence),
            "recommended_actions": [
                "Check whether compilation is making progress before increasing the timeout.",
                "Retry with a bounded timeout and preserve partial output.",
            ],
        }

    rule = next((candidate for candidate in RULES if candidate.matches(text)), None)
    if rule is None:
        observed_stage = _event_stage(failed_event or {})
        if observed_stage is not None:
            failure_stage, pipeline_stage = observed_stage
            return {
                "schema_version": 1,
                "status": "failed",
                "rule_id": "failed-compiler-stage-event",
                "category": "compiler-lowering",
                "failure_stage": failure_stage,
                "pipeline_stage": pipeline_stage,
                "summary": f"The observed {failure_stage} compiler stage failed.",
                "confidence": 0.7,
                "repair_scope": "compiler",
                "failed_command": failed_command,
                "evidence": _fallback_evidence(lines, max_evidence),
                "recommended_actions": [
                    "Preserve the stage input, output, and exact failed command.",
                    "Inspect the first unsupported operation before changing operator source.",
                ],
            }
        return {
            "schema_version": 1,
            "status": "failed",
            "rule_id": "unknown-nonzero-exit",
            "category": "unknown",
            "failure_stage": "unknown",
            "pipeline_stage": "unknown",
            "summary": "The command returned a nonzero exit code without a known signature.",
            "confidence": 0.2,
            "repair_scope": "human-triage",
            "failed_command": failed_command,
            "evidence": _fallback_evidence(lines, max_evidence),
            "recommended_actions": [
                "Collect the complete log and compiler stage events.",
                "Add a reviewed diagnosis rule only after the root cause is confirmed.",
            ],
        }

    evidence = _evidence(lines, rule.evidence_patterns, max_items=max_evidence)
    if not evidence:
        evidence = _fallback_evidence(lines, max_evidence)
    payload = asdict(rule)
    payload.pop("match_patterns")
    payload.pop("evidence_patterns")
    payload["status"] = "failed"
    payload["schema_version"] = 1
    payload["failed_command"] = failed_command
    payload["evidence"] = evidence
    payload["recommended_actions"] = list(rule.recommended_actions)
    return payload
