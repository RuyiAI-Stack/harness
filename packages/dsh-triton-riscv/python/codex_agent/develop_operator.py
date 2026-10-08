#!/usr/bin/env python3
"""Generate, validate, diagnose, and optionally repair one Triton operator."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .adapters import CodexCliModelGateway
from .core import ModelRequest, WorkflowPhase, WorkflowSession
from .discover_operators import discover_operators
from .embeddings import build_embedding_provider
from .memory import (
    MemoryQuery,
    MemoryStore,
    add_embedding_arguments,
    ingest_results,
    normalized_error_signature as memory_error_signature,
    records_from_run,
    render_memory_context,
)
from .operator_agent import run_preflight
from .pipeline_observer import (
    build_pipeline_report,
    collect_artifacts,
    collect_environment,
    load_stage_events,
    render_pipeline_markdown,
)
from .validate_operator import classify_log, extract_error_excerpt, slugify


NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REMOTE_HOST_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
TORCH_SYMBOL_RE = re.compile(r"\b(torch(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
OPERATOR_ROOT = Path("python/examples/flaggems")
DEFAULT_RESULTS_DIR = Path("agent-results/development")
REPAIRABLE_STAGES = {
    "import",
    "ttir",
    "triton-frontend",
    "runtime",
    "correctness",
    "pytest",
    "unknown",
}
COMPILER_STAGES = {
    "compilation",
    "triton-shared-opt",
    "linalg-mlir",
    "buddy-opt",
    "llvm-mlir",
    "mlir-translate",
    "llvm-ir",
    "llc",
    "riscv-object",
}
EXTERNAL_FAILURE_STAGES = {
    "environment",
    "build",
    "timeout",
    "target-capability",
    "hardware-capability",
    "link",
    "link-load",
    "selection",
}
PIPELINE_STAGE_ORDER = (
    "ttir",
    "linalg-mlir",
    "llvm-mlir",
    "llvm-ir",
    "riscv-object",
    "link-load",
    "hardware-capability",
    "runtime",
    "correctness",
)
STAGE_ALIASES = {
    "triton-frontend": "ttir",
    "triton-shared-opt": "linalg-mlir",
    "buddy-opt": "llvm-mlir",
    "mlir-translate": "llvm-ir",
    "llc": "riscv-object",
    "link": "link-load",
    "target-capability": "hardware-capability",
}
VALIDATION_START = "<!-- autonomous-validation:start -->"
VALIDATION_END = "<!-- autonomous-validation:end -->"


@dataclass(frozen=True)
class OperatorSpec:
    schema_version: int
    name: str
    semantics: str
    pytorch_reference: str
    inputs: list[dict[str, str]]
    output: str
    shape_cases: list[list[int]]
    input_shape_cases: list[dict[str, list[int]]]
    dtypes: list[str]
    tolerances: dict[str, float]
    backward: bool
    implementation_file: str
    test_file: str
    reference_operators: list[str]
    notes: str


@dataclass
class ContractAudit:
    status: str
    errors: list[str]
    warnings: list[str]
    test_sha256: str | None


@dataclass(frozen=True)
class RepairDecision:
    action: str
    stage: str | None
    category: str
    strategy: str
    reason: str


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extract_pytest_summary(text: str) -> str | None:
    summaries = re.findall(
        r"(?:\d+ (?:passed|failed|skipped|xfailed|xpassed))(?:, \d+ "
        r"(?:passed|failed|skipped|xfailed|xpassed))* in [0-9.]+s",
        text,
    )
    return summaries[-1] if summaries else None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes the repository: {relative_path}") from exc
    return path


def validate_operator_path(path: str, *, test_file: bool) -> None:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"operator path must be repository-relative: {path}")
    if candidate.suffix != ".py" or not candidate.is_relative_to(OPERATOR_ROOT):
        raise ValueError(f"operator path must be a Python file under {OPERATOR_ROOT}: {path}")
    if test_file and not candidate.name.startswith("test_"):
        raise ValueError(f"test file must start with test_: {path}")
    if not test_file and candidate.name.startswith("test_"):
        raise ValueError(f"implementation file cannot start with test_: {path}")


def load_operator_spec(path: Path, repo_root: Path) -> OperatorSpec:
    return parse_operator_spec(json.loads(path.read_text(encoding="utf-8")), repo_root)


def parse_operator_spec(data: dict, repo_root: Path) -> OperatorSpec:
    """Share contract validation between CLI files and typed plugin tools."""
    if not isinstance(data, dict):
        raise ValueError("operator spec must be a JSON object")
    required = {
        "schema_version",
        "name",
        "semantics",
        "pytorch_reference",
        "inputs",
        "output",
        "shape_cases",
        "dtypes",
        "tolerances",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"operator spec is missing fields: {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError("only operator spec schema_version 1 is supported")
    name = data["name"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid operator name: {name!r}")
    if not isinstance(data["semantics"], str) or len(data["semantics"].strip()) < 10:
        raise ValueError("semantics must be a concrete description")
    if not isinstance(data["pytorch_reference"], str) or "torch." not in data["pytorch_reference"]:
        raise ValueError("pytorch_reference must contain a torch expression")
    inputs = data["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must contain at least one input contract")
    input_names: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"name", "description"}:
            raise ValueError("each input requires only name and description")
        if not NAME_RE.fullmatch(str(item["name"])):
            raise ValueError(f"invalid input name: {item['name']!r}")
        if item["name"] in input_names:
            raise ValueError(f"duplicate input name: {item['name']}")
        input_names.add(item["name"])
    shape_cases = data["shape_cases"]
    if (
        not isinstance(shape_cases, list)
        or not shape_cases
        or any(
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            for shape in shape_cases
        )
    ):
        raise ValueError("shape_cases must be non-empty lists of positive integers")
    dtypes = data["dtypes"]
    if not isinstance(dtypes, list) or not dtypes or any(
        not isinstance(dtype, str) or not dtype.startswith("torch.") for dtype in dtypes
    ):
        raise ValueError("dtypes must contain one or more torch dtype names")
    input_shape_cases = data.get("input_shape_cases", [])
    if not isinstance(input_shape_cases, list):
        raise ValueError("input_shape_cases must be a list")
    for case in input_shape_cases:
        if not isinstance(case, dict) or set(case) != input_names:
            raise ValueError(
                "each input_shape_cases entry must provide every named input"
            )
        if any(
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            for shape in case.values()
        ):
            raise ValueError("input_shape_cases must use positive integer shapes")
    tolerances = data["tolerances"]
    if set(tolerances) != {"rtol", "atol"} or any(
        not isinstance(tolerances[key], (int, float)) or tolerances[key] < 0
        for key in ("rtol", "atol")
    ):
        raise ValueError("tolerances must contain non-negative rtol and atol")
    implementation_file = data.get(
        "implementation_file",
        (OPERATOR_ROOT / f"{name}.py").as_posix(),
    )
    test_file = data.get(
        "test_file",
        (OPERATOR_ROOT / f"test_{name}.py").as_posix(),
    )
    validate_operator_path(implementation_file, test_file=False)
    validate_operator_path(test_file, test_file=True)
    resolve_repo_path(repo_root, implementation_file)
    resolve_repo_path(repo_root, test_file)
    references = data.get("reference_operators", [])
    if not isinstance(references, list) or any(not isinstance(item, str) for item in references):
        raise ValueError("reference_operators must be a list of operator names")
    return OperatorSpec(
        schema_version=1,
        name=name,
        semantics=data["semantics"].strip(),
        pytorch_reference=data["pytorch_reference"].strip(),
        inputs=inputs,
        output=str(data["output"]).strip(),
        shape_cases=shape_cases,
        input_shape_cases=input_shape_cases,
        dtypes=dtypes,
        tolerances={key: float(tolerances[key]) for key in ("rtol", "atol")},
        backward=bool(data.get("backward", False)),
        implementation_file=implementation_file,
        test_file=test_file,
        reference_operators=references,
        notes=str(data.get("notes", "")).strip(),
    )


def operator_tokens(value: str) -> set[str]:
    ignored = {"and", "operator", "torch", "tensor", "compute", "elementwise"}
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9]*", value.lower())
        if token not in ignored and len(token) > 1
    }


def choose_references(spec: OperatorSpec, inventory: dict, limit: int = 4) -> list[dict]:
    operators = inventory.get("operators", [])
    explicit = {name: index for index, name in enumerate(spec.reference_operators)}
    query_tokens = operator_tokens(
        f"{spec.name} {spec.semantics} {spec.pytorch_reference}"
    )
    fused_suffix = spec.name.split("_", 1)[1] if "_" in spec.name else ""
    scored: list[tuple[int, str, dict]] = []
    for operator in operators:
        name = operator["name"]
        if name == spec.name:
            continue
        score = 0
        if name in explicit:
            score += 100 - explicit[name]
        candidate_text = " ".join(
            [name, *operator.get("torch_references", []), *operator.get("tl_ops", [])]
        )
        score += 6 * len(query_tokens & operator_tokens(candidate_text))
        if fused_suffix and name.endswith(fused_suffix):
            score += 12
        if spec.backward and any("backward" in function for function in operator.get("public_functions", [])):
            score += 3
        if score:
            scored.append((-score, name, operator))
    return [item[2] for item in sorted(scored)[:limit]]


def render_task(
    spec: OperatorSpec,
    references: list[dict],
    memory_context: str | None = None,
) -> str:
    inputs = "\n".join(
        f"- `{item['name']}`: {item['description']}" for item in spec.inputs
    )
    reference_lines = "\n".join(
        f"- `{item['implementation_file']}` ({item['name']}; tl ops: "
        f"{', '.join(item.get('tl_ops', [])) or 'none detected'})"
        for item in references
    ) or "- No close repository reference was discovered; inspect nearby FlagGems operators."
    shapes = ", ".join(str(tuple(shape)) for shape in spec.shape_cases)
    input_shape_cases = (
        json.dumps(spec.input_shape_cases, sort_keys=True)
        if spec.input_shape_cases
        else "none"
    )
    dtypes = ", ".join(spec.dtypes)
    backward = "required" if spec.backward else "not required"
    return f"""# Operator Development Task: {spec.name}

## Immutable Contract

- Semantics: {spec.semantics}
- PyTorch reference: `{spec.pytorch_reference}`
- Output: {spec.output}
- Shape cases: {shapes}
- Per-input shape cases: `{input_shape_cases}`
- Dtypes: {dtypes}
- Maximum tolerance: rtol={spec.tolerances['rtol']}, atol={spec.tolerances['atol']}
- Backward validation: {backward}

### Inputs

{inputs}

## Automatically Selected References

{reference_lines}

## Retrieved Historical Evidence

{memory_context or "No relevant verified historical memory was retrieved."}

## Allowed Files

- Implementation: `{spec.implementation_file}`
- Test: `{spec.test_file}`

Do not modify the contract, reference expression, shape/dtype coverage, or
tolerances merely to make validation pass. During repair iterations, the test
file is locked and only the implementation may be changed.

## Required Work

1. Inspect the selected references and `docs/05-Operator Migration.md`.
2. Implement a clear Triton kernel and Python wrapper matching the contract.
3. Add pytest coverage that computes the PyTorch reference independently and
   compares it with `torch.testing.assert_close`.
4. Include every required shape and dtype and the backward path when required.
5. Keep pointer arithmetic, masks, casts, and boundary behavior explicit.
6. Do not use autotuning in the first implementation.

## Validation

```sh
source scripts/triton-riscv-env.sh
python -m pytest -q {spec.test_file} -s
```

The orchestration agent runs this command in the configured RISC-V environment,
classifies failures, and permits only bounded semantic-preserving repairs.

## Notes

{spec.notes or "No additional notes."}
"""


def collect_literal_shapes(tree: ast.AST) -> set[tuple[int, ...]]:
    shapes: set[tuple[int, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        values: list[int] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, int):
                break
            values.append(element.value)
        else:
            if values:
                shapes.add(tuple(values))
    return shapes


def audit_test_contract(spec: OperatorSpec, repo_root: Path) -> ContractAudit:
    test_path = resolve_repo_path(repo_root, spec.test_file)
    errors: list[str] = []
    warnings: list[str] = []
    if not test_path.exists():
        return ContractAudit("failed", [f"missing test file: {spec.test_file}"], [], None)
    try:
        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        return ContractAudit("failed", [f"test file cannot be parsed: {exc}"], [], None)

    module_name = Path(spec.implementation_file).stem
    imports_implementation = any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.split(".")[-1] == module_name
        for node in ast.walk(tree)
    )
    if not imports_implementation:
        errors.append(f"test does not import the {module_name} implementation module")
    test_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]
    if not test_functions:
        errors.append("test file defines no pytest test functions")
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if "torch.testing.assert_close" not in calls:
        errors.append("test must compare results with torch.testing.assert_close")
    source_without_space = "".join(source.split())
    required_symbols = sorted(set(TORCH_SYMBOL_RE.findall(spec.pytorch_reference)))
    for symbol in required_symbols:
        if symbol not in source_without_space:
            errors.append(f"test is missing PyTorch reference symbol {symbol}")
    unparsed = ast.unparse(tree)
    for dtype in spec.dtypes:
        if dtype not in unparsed:
            errors.append(f"test is missing required dtype {dtype}")
    literal_shapes = collect_literal_shapes(tree)
    for shape in spec.shape_cases:
        if tuple(shape) not in literal_shapes:
            errors.append(f"test is missing required shape {tuple(shape)}")
    for case in spec.input_shape_cases:
        for input_name, shape in case.items():
            if tuple(shape) not in literal_shapes:
                errors.append(
                    f"test is missing {input_name} shape {tuple(shape)} from an input-shape case"
                )
    if spec.backward and not any(
        "backward" in node.name for node in test_functions
    ):
        errors.append("operator requires a backward pytest case")
    if spec.backward and ".backward(" not in source_without_space:
        warnings.append("backward test does not visibly call the PyTorch backward reference")
    assert_close_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.testing.assert_close"
    ]
    for call in assert_close_calls:
        keywords = {item.arg: item.value for item in call.keywords if item.arg}
        for tolerance in ("rtol", "atol"):
            value = keywords.get(tolerance)
            if value is None:
                warnings.append(
                    f"an assert_close call does not set {tolerance} explicitly"
                )
            elif isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                if float(value.value) > spec.tolerances[tolerance]:
                    errors.append(
                        f"assert_close {tolerance}={value.value} exceeds the contract maximum "
                        f"{spec.tolerances[tolerance]}"
                    )
            elif isinstance(value, ast.Name):
                warnings.append(
                    f"assert_close {tolerance} uses variable {value.id}; review its parameter values"
                )
    return ContractAudit(
        "passed" if not errors else "failed",
        errors,
        warnings,
        file_sha256(test_path),
    )


def collect_workspace_state(repo_root: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    state: dict[str, str] = {}
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", "surrogateescape")
        path = repo_root / relative
        state[relative] = file_sha256(path) if path.is_file() else "<missing>"
    return state


def unauthorized_changes(
    before: dict[str, str],
    after: dict[str, str],
    allowed_paths: set[str],
) -> list[str]:
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path) and path not in allowed_paths
    )


def snapshot_workspace_bytes(
    repo_root: Path,
    state: dict[str, str],
) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for relative in state:
        path = repo_root / relative
        if path.is_file():
            snapshot[relative] = path.read_bytes()
    return snapshot


def restore_workspace_paths(
    repo_root: Path,
    before_state: dict[str, str],
    before_contents: dict[str, bytes],
    paths: list[str],
) -> list[str]:
    failed: list[str] = []
    for relative in paths:
        path = repo_root / relative
        try:
            if relative in before_contents:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(before_contents[relative])
            elif relative not in before_state and path.exists():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                else:
                    failed.append(relative)
            else:
                failed.append(relative)
        except OSError:
            failed.append(relative)
    return failed


def snapshot_files(repo_root: Path, paths: list[str]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for relative in paths:
        path = resolve_repo_path(repo_root, relative)
        snapshot[relative] = path.read_text(encoding="utf-8") if path.exists() else None
    return snapshot


def restore_files(repo_root: Path, snapshot: dict[str, str | None]) -> None:
    for relative, content in snapshot.items():
        path = resolve_repo_path(repo_root, relative)
        if content is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_patch(
    path: Path,
    before: dict[str, str | None],
    after: dict[str, str | None],
) -> None:
    lines: list[str] = []
    for relative in sorted(before.keys() | after.keys()):
        lines.extend(
            difflib.unified_diff(
                (before.get(relative) or "").splitlines(keepends=True),
                (after.get(relative) or "").splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    path.write_text("".join(lines), encoding="utf-8")


def run_codex(
    repo_root: Path,
    prompt: str,
    run_dir: Path,
    label: str,
    timeout_seconds: int,
) -> dict:
    gateway = CodexCliModelGateway()
    return gateway.invoke(
        ModelRequest(
            prompt=prompt,
            repo_root=repo_root,
            run_dir=run_dir,
            label=label,
            timeout_seconds=timeout_seconds,
        )
    ).as_dict()


def generation_prompt(
    spec: OperatorSpec,
    task_path: Path,
    references: list[dict],
    memory_context: str | None = None,
) -> str:
    references_text = "\n".join(
        f"- {item['implementation_file']}" for item in references
    )
    return f"""Implement the Triton-RISCV operator described in {task_path.as_posix()}.

Read the task, docs/05-Operator Migration.md, and these selected references:
{references_text or '- inspect nearby FlagGems operators'}

Relevant version-scoped historical evidence:
{memory_context or 'No sufficiently relevant verified memory was retrieved.'}

You may create or edit only:
- {spec.implementation_file}
- {spec.test_file}

Do not edit the task, workflow, reference expression, expected values, shapes,
dtypes, or tolerances. The pytest test must independently compute
`{spec.pytorch_reference}` and use torch.testing.assert_close. Do not run git
commands or commit. The RISC-V validation environment is remote, so finish with
local syntax/static checks only and let the orchestration agent run pytest.
"""


def contract_repair_prompt(spec: OperatorSpec, audit: ContractAudit) -> str:
    errors = "\n".join(f"- {item}" for item in audit.errors)
    return f"""The generated test for {spec.name} failed the immutable contract audit:
{errors}

Correct the implementation/test package without changing the operator contract.
You may edit only {spec.implementation_file} and {spec.test_file}. The test must
compute `{spec.pytorch_reference}` independently, cover all required shapes and
dtypes, and use torch.testing.assert_close. Do not reduce coverage or loosen the
specified maximum tolerances. Do not run git commands or commit.
"""


def failure_stage(validation: dict) -> str | None:
    return validation.get("first_failure_stage") or validation.get("failure_stage")


def plan_repair(
    validation: dict,
    allow_compiler_workaround: bool,
) -> RepairDecision:
    stage = failure_stage(validation)
    if validation.get("status") == "passed":
        return RepairDecision(
            "stop",
            stage,
            "complete",
            "none",
            "the locked acceptance test passed",
        )
    if stage in EXTERNAL_FAILURE_STAGES:
        return RepairDecision(
            "stop",
            stage,
            "external-blocker",
            "preserve implementation and report the external failure",
            f"{stage} failures are not safely repairable by changing operator code",
        )
    if stage in COMPILER_STAGES or STAGE_ALIASES.get(stage or "") in {
        "linalg-mlir",
        "llvm-mlir",
        "llvm-ir",
        "riscv-object",
    }:
        if not allow_compiler_workaround:
            return RepairDecision(
                "stop",
                stage,
                "compiler-blocker",
                "preserve implementation and report the compiler limitation",
                "implementation-only compiler workarounds were disabled",
            )
        return RepairDecision(
            "repair",
            stage,
            "compiler-workaround",
            (
                "rewrite only the operator implementation with equivalent, already "
                "supported Triton primitives; do not modify compiler sources"
            ),
            "the failure may be avoided by an equivalent implementation pattern",
        )
    strategies = {
        "import": "repair module imports, wrapper names, and public signatures",
        "ttir": "replace unavailable Triton APIs and correct frontend typing or shapes",
        "triton-frontend": "replace unavailable Triton APIs and correct frontend typing or shapes",
        "runtime": "repair pointer arithmetic, masks, strides, launch grid, and boundary handling",
        "correctness": "repair the formula, broadcasting, dtype conversion, indexing, or backward derivative",
        "pytest": "repair the implementation API or behavior exposed by the pytest failure",
        "unknown": "make the smallest implementation-only change supported by the failure evidence",
    }
    if stage in REPAIRABLE_STAGES or stage is None:
        return RepairDecision(
            "repair",
            stage,
            "implementation",
            strategies.get(stage or "unknown", strategies["unknown"]),
            "the evidence points to code owned by the operator implementation",
        )
    return RepairDecision(
        "stop",
        stage,
        "unclassified-blocker",
        "preserve implementation and request manual classification",
        "the failure stage is not covered by the automatic repair policy",
    )


def may_repair(stage: str | None, allow_compiler_workaround: bool) -> bool:
    validation = {
        "status": "failed",
        "failure_stage": stage,
        "first_failure_stage": None,
    }
    return plan_repair(validation, allow_compiler_workaround).action == "repair"


def normalized_error_signature(validation: dict) -> str:
    evidence = "\n".join(validation.get("error_excerpt", []))
    evidence = re.sub(r"/tmp/tmp[^/\s'\"]+", "/tmp/<tmp>", evidence)
    payload = {
        "stage": failure_stage(validation),
        "reason": validation.get("likely_reason"),
        "evidence": evidence,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def compare_validations(previous: dict, candidate: dict) -> tuple[str, bool, str]:
    if candidate.get("status") == "passed":
        return "passed", True, "the locked acceptance test passed"
    previous_stage = STAGE_ALIASES.get(
        failure_stage(previous) or "",
        failure_stage(previous),
    )
    candidate_stage = STAGE_ALIASES.get(
        failure_stage(candidate) or "",
        failure_stage(candidate),
    )
    positions = {stage: index for index, stage in enumerate(PIPELINE_STAGE_ORDER)}
    previous_position = positions.get(previous_stage)
    candidate_position = positions.get(candidate_stage)
    if (
        previous_position is not None
        and candidate_position is not None
        and candidate_position > previous_position
    ):
        return (
            "progressed",
            True,
            f"the first failure moved from {previous_stage} to {candidate_stage}",
        )
    if (
        previous_position is not None
        and candidate_position is not None
        and candidate_position < previous_position
    ):
        return (
            "regressed",
            False,
            f"the first failure moved backward from {previous_stage} to {candidate_stage}",
        )
    if normalized_error_signature(previous) == normalized_error_signature(candidate):
        return "ineffective", False, "the failure stage and normalized evidence did not change"
    return (
        "changed-evidence",
        True,
        "the failure evidence changed without moving to an earlier pipeline stage",
    )


def repair_prompt(
    spec: OperatorSpec,
    validation: dict,
    decision: RepairDecision,
    repair_history: list[dict] | None = None,
    memory_context: str | None = None,
    working_memory: dict | None = None,
) -> str:
    excerpt = "\n".join(validation.get("error_excerpt", []))
    stage_lines = "\n".join(
        f"- {item.get('id')}: {item.get('status')}"
        for item in validation.get("stages", [])
        if item.get("status") not in {"not-run", "not-established"}
    )
    previous_attempts = "\n".join(
        f"- attempt {item['attempt']}: {item['outcome']} ({item['reason']})"
        for item in (repair_history or [])
    )
    return f"""Repair the {spec.name} Triton implementation using this validation evidence.

First failure stage: {failure_stage(validation)}
Likely reason: {validation.get('likely_reason')}
Repair category: {decision.category}
Required strategy: {decision.strategy}
Error excerpt:
{excerpt}

Observed pipeline stages:
{stage_lines or '- no fresh stage artifacts were available; use the classified log evidence'}

Previous repair outcomes:
{previous_attempts or '- none'}

Current working memory:
{json.dumps(working_memory or {}, indent=2, sort_keys=True)}

Relevant version-scoped historical evidence:
{memory_context or 'No sufficiently relevant verified memory was retrieved.'}

The test file is a locked acceptance test. Edit only
{spec.implementation_file}. Do not edit {spec.test_file}, the PyTorch reference,
shapes, dtypes, expected values, or tolerances. Preserve the semantics
`{spec.pytorch_reference}` and make the smallest evidence-based implementation
change. For compiler workarounds, use an equivalent operator formulation built
from patterns already working in the selected repository references; never edit
the compiler or backend. Do not run git commands or commit. Do not report the
repair as successful because the orchestration agent will rerun the locked test.
"""


def spec_memory_query(
    spec: OperatorSpec,
    references: list[dict],
    *,
    validation: dict | None = None,
    environment: dict | None = None,
) -> MemoryQuery:
    tl_ops = sorted(
        {
            operation
            for reference in references
            for operation in reference.get("tl_ops", [])
        }
    )
    stage = failure_stage(validation) if validation else None
    error_signature = None
    if validation:
        error_signature = memory_error_signature(
            stage,
            validation.get("likely_reason"),
            validation.get("error_excerpt", []),
        )
    environment = environment or {}
    execution = environment.get("execution", {})
    triton = environment.get("triton", {})
    environment_query = {
        "architecture": environment.get("architecture"),
        "execution_mode": execution.get("mode") if isinstance(execution, dict) else None,
        "triton": triton.get("version") if isinstance(triton, dict) else triton,
        "llvm": environment.get("llvm_version"),
        "buddy": environment.get("buddy_version"),
    }
    return MemoryQuery(
        operator=spec.name,
        semantics=spec.semantics,
        pytorch_reference=spec.pytorch_reference,
        tl_ops=tl_ops,
        failure_stage=stage,
        error_signature=error_signature,
        environment={key: value for key, value in environment_query.items() if value},
    )


def build_working_memory(
    spec: OperatorSpec,
    validation: dict,
    repair_history: list[dict],
    locked_test_hash: str,
    remaining_repairs: int,
) -> dict:
    return {
        "operator": spec.name,
        "immutable_reference": spec.pytorch_reference,
        "allowed_file": spec.implementation_file,
        "locked_test_file": spec.test_file,
        "locked_test_sha256": locked_test_hash,
        "current_failure_stage": failure_stage(validation),
        "current_reason": validation.get("likely_reason"),
        "current_error_excerpt": validation.get("error_excerpt", [])[:4],
        "completed_repairs": len(repair_history),
        "remaining_repairs": remaining_repairs,
        "previous_outcomes": [
            {
                "attempt": item.get("attempt"),
                "outcome": item.get("outcome"),
                "accepted": item.get("accepted"),
                "reason": item.get("reason"),
            }
            for item in repair_history[-5:]
        ],
    }


def run_ssh(host: str, command: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    if not REMOTE_HOST_RE.fullmatch(host):
        raise ValueError(f"invalid SSH host: {host!r}")
    return subprocess.run(
        ["ssh", host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def remote_environment_prefix(remote_root: str) -> str:
    root = shlex.quote(remote_root)
    return (
        f"cd {root} && "
        "source .venv/bin/activate && "
        "source scripts/triton-riscv-env.sh"
    )


def run_remote_preflight(host: str, remote_root: str) -> dict:
    checks = (
        "python -c 'import triton; print(triton.__version__)' && "
        "command -v triton-shared-opt && command -v buddy-opt && uname -m"
    )
    start = time.monotonic()
    try:
        completed = run_ssh(
            host,
            f"{remote_environment_prefix(remote_root)} && {checks}",
            90,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "exit_code": 124,
            "reason": "remote environment check timed out",
            "output": exc.stdout or "",
        }
    output = completed.stdout.strip()
    lines = output.splitlines()
    architecture = lines[-1] if completed.returncode == 0 and lines else None
    native_riscv = architecture in {"riscv", "riscv64"}
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "reason": None if completed.returncode == 0 else "remote environment check failed",
        "duration_seconds": round(time.monotonic() - start, 3),
        "output": output,
        "architecture": architecture,
        "execution": {
            "mode": "native-riscv" if native_riscv else "non-riscv-host",
            "native_riscv": native_riscv,
            "hardware_execution_established": native_riscv,
        },
    }


def sync_remote_files(
    spec: OperatorSpec,
    repo_root: Path,
    host: str,
    remote_root: str,
) -> None:
    for relative in (spec.implementation_file, spec.test_file):
        local_path = resolve_repo_path(repo_root, relative)
        if not local_path.exists():
            raise FileNotFoundError(f"cannot sync missing file: {relative}")
        remote_path = f"{remote_root.rstrip('/')}/{relative}"
        parent = str(Path(remote_path).parent)
        mkdir_result = run_ssh(host, f"mkdir -p {shlex.quote(parent)}", 30)
        if mkdir_result.returncode != 0:
            raise RuntimeError(f"remote mkdir failed: {mkdir_result.stdout.strip()}")
        completed = subprocess.run(
            ["scp", local_path.as_posix(), f"{host}:{remote_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"scp failed for {relative}: {completed.stdout.strip()}")


def run_validation(
    spec: OperatorSpec,
    repo_root: Path,
    run_dir: Path,
    iteration: int,
    timeout_seconds: int,
    remote_host: str | None,
    remote_root: str | None,
    source_env: bool,
    environment: dict,
    capture_pipeline: bool,
    fresh_compile: bool,
) -> dict:
    command = f"python -m pytest -q {shlex.quote(spec.test_file)} -s"
    dump_dir = run_dir / f"validation-{iteration}-artifacts"
    artifact_sync_error: str | None = None
    start = time.monotonic()
    if remote_host:
        assert remote_root is not None
        sync_remote_files(spec, repo_root, remote_host, remote_root)
        remote_run = (
            f"{remote_root.rstrip('/')}/agent-results/development-runs/"
            f"{slugify(spec.name)}-{time.time_ns()}-{iteration}"
        )
        remote_dump = f"{remote_run}/artifacts"
        setup = [f"mkdir -p {shlex.quote(remote_dump)}"]
        if capture_pipeline:
            setup.append(
                f"export TRITON_SHARED_DUMP_PATH={shlex.quote(remote_dump)}"
            )
        if fresh_compile:
            setup.append(
                f"export TRITON_CACHE_DIR={shlex.quote(remote_run + '/cache')}"
            )
        shell_command = " && ".join(
            [remote_environment_prefix(remote_root), *setup, command]
        )
        try:
            completed = run_ssh(remote_host, shell_command, timeout_seconds)
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_seconds} seconds\n"
            exit_code = 124
        display_command = f"ssh {remote_host} {shlex.quote(shell_command)}"
        if capture_pipeline:
            dump_dir.mkdir(parents=True, exist_ok=True)
            try:
                copied = subprocess.run(
                    ["scp", "-r", f"{remote_host}:{remote_dump}/.", dump_dir.as_posix()],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if copied.returncode != 0:
                    artifact_sync_error = copied.stdout.strip()
            except subprocess.TimeoutExpired:
                artifact_sync_error = "remote artifact copy timed out"
        try:
            run_ssh(remote_host, f"rm -rf {shlex.quote(remote_run)}", 60)
        except subprocess.TimeoutExpired:
            pass
    else:
        shell_command = command
        setup = []
        if capture_pipeline:
            setup.append(
                f"export TRITON_SHARED_DUMP_PATH={shlex.quote(dump_dir.as_posix())}"
            )
        if fresh_compile:
            setup.append(
                f"export TRITON_CACHE_DIR={shlex.quote((run_dir / f'cache-{iteration}').as_posix())}"
            )
        if setup:
            shell_command = " && ".join([*setup, shell_command])
        if source_env:
            shell_command = f"source scripts/triton-riscv-env.sh && {shell_command}"
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
            output = completed.stdout
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_seconds} seconds\n"
            exit_code = 124
        display_command = shell_command
    log_path = run_dir / f"validation-{iteration}.log"
    log_path.write_text(output, encoding="utf-8")
    status, failure_stage, likely_reason = classify_log(output, exit_code)
    operator_context = {
        "name": spec.name,
        "test_contract": {
            "pytorch_reference": [spec.pytorch_reference],
            "numerical_assertion": True,
            "backward": spec.backward,
            "broadcast": bool(spec.input_shape_cases),
            "dtypes": spec.dtypes,
        },
    }
    artifacts = collect_artifacts(dump_dir) if capture_pipeline else []
    stage_events = load_stage_events(dump_dir) if capture_pipeline else []
    pipeline_report = build_pipeline_report(
        operator=operator_context,
        validation_status=status,
        failure_stage=failure_stage,
        likely_reason=likely_reason,
        exit_code=exit_code,
        artifacts=artifacts,
        stage_events=stage_events,
        environment=environment,
    )
    pipeline_path = run_dir / f"validation-{iteration}-pipeline.json"
    write_json(pipeline_path, pipeline_report)
    report_path = run_dir / f"validation-{iteration}-report.md"
    report_path.write_text(
        render_pipeline_markdown(pipeline_report),
        encoding="utf-8",
    )
    result = {
        "operator": spec.name,
        "iteration": iteration,
        "command": display_command,
        "status": status,
        "exit_code": exit_code,
        "failure_stage": failure_stage,
        "likely_reason": likely_reason,
        "error_excerpt": extract_error_excerpt(output) if status == "failed" else [],
        "test_summary": extract_pytest_summary(output),
        "duration_seconds": round(time.monotonic() - start, 3),
        "log_path": log_path.as_posix(),
        "first_failure_stage": pipeline_report["first_failure_stage"],
        "execution_mode": pipeline_report["execution_mode"],
        "correctness": pipeline_report["correctness"],
        "stages": pipeline_report["stages"],
        "pipeline_report_path": pipeline_path.as_posix(),
        "pipeline_markdown_path": report_path.as_posix(),
        "artifact_dir": dump_dir.as_posix() if capture_pipeline else None,
        "artifact_sync_error": artifact_sync_error,
    }
    write_json(run_dir / f"validation-{iteration}.json", result)
    return result


def update_task_validation_record(task_path: Path, final: dict) -> None:
    content = task_path.read_text(encoding="utf-8")
    if VALIDATION_START in content and VALIDATION_END in content:
        prefix = content.split(VALIDATION_START, 1)[0].rstrip()
    else:
        prefix = content.rstrip()
    lines = [
        VALIDATION_START,
        "",
        "## Autonomous Validation Record",
        "",
        f"- Final status: `{final['status']}`",
        f"- Stop reason: {final.get('stop_reason') or 'acceptance test passed'}",
        f"- Acceptance-test SHA-256: `{final['locked_test_sha256']}`",
        f"- Repair attempts: {final['repair_attempts']}",
        f"- Accepted repairs: {final.get('accepted_repairs', 0)}",
        f"- Rejected repairs: {final.get('rejected_repairs', 0)}",
        "",
        "| Attempt | Status | Failure stage | Test summary |",
        "| --- | --- | --- | --- |",
    ]
    for validation in final["validations"]:
        lines.append(
            f"| {validation['iteration']} | {validation['status']} | "
            f"{validation.get('failure_stage') or ''} | "
            f"{validation.get('test_summary') or ''} |"
        )
    failure_excerpts = [
        line
        for validation in final["validations"]
        for line in validation.get("error_excerpt", [])[:1]
    ]
    if failure_excerpts:
        lines.extend(["", "### Failure Evidence", ""])
        lines.extend(f"- `{line}`" for line in failure_excerpts)
    lines.extend(
        [
            "",
            "Full logs, Codex transcripts, and per-iteration patches are stored "
            "under the run directory shown by the command output.",
            "",
            VALIDATION_END,
        ]
    )
    task_path.write_text(prefix + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def render_repair_summary(final: dict) -> str:
    lines = [
        f"# Autonomous Repair: {final['operator']}",
        "",
        f"- Final status: {final['status']}",
        f"- Final validation: {final.get('validation_status', 'unknown')}",
        f"- Stop reason: {final.get('stop_reason') or 'none'}",
        f"- Locked test SHA-256: `{final['locked_test_sha256']}`",
        f"- Repair attempts: {final['repair_attempts']}",
        f"- Accepted repairs: {final.get('accepted_repairs', 0)}",
        f"- Rejected repairs: {final.get('rejected_repairs', 0)}",
        "",
        "## Validation Attempts",
        "",
        "| Attempt | Result | First failure | Correctness | Summary |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for item in final["validations"]:
        lines.append(
            f"| {item['iteration']} | {item['status']} | "
            f"{failure_stage(item) or ''} | {item.get('correctness') or ''} | "
            f"{item.get('test_summary') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Repair Decisions",
            "",
            "| Repair | Category | Outcome | Accepted | Reason |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for item in final.get("repair_history", []):
        lines.append(
            f"| {item['attempt']} | {item['decision']['category']} | "
            f"{item['outcome']} | {str(item['accepted']).lower()} | "
            f"{item['reason']} |"
        )
    if not final.get("repair_history"):
        lines.append("|  |  | no repair needed |  |  |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop and validate one operator from a structured specification."
    )
    parser.add_argument("--spec", required=True, help="Operator specification JSON.")
    parser.add_argument("--repo-root", default=".", help="Local repository root.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR.as_posix())
    parser.add_argument("--remote-host", default=None, help="SSH host used for RISC-V validation.")
    parser.add_argument("--remote-root", default=None, help="Repository root on the remote host.")
    parser.add_argument("--source-env", action="store_true", help="Source the environment for local validation.")
    parser.add_argument("--prepare-only", action="store_true", help="Generate context without invoking Codex.")
    parser.add_argument("--allow-existing", action="store_true", help="Allow completion of existing implementation/test files.")
    parser.add_argument("--force-task", action="store_true", help="Overwrite an existing generated task file.")
    parser.add_argument("--max-generation-attempts", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--codex-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--allow-compiler-workaround",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Try implementation-only equivalent rewrites for backend compiler failures "
            "without editing compiler sources (enabled by default)."
        ),
    )
    parser.add_argument(
        "--capture-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture per-stage IR, command, and object evidence for every validation.",
    )
    parser.add_argument(
        "--fresh-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an isolated Triton cache for each validation attempt.",
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrieve and record evidence-gated operator memories.",
    )
    parser.add_argument("--memory-db", default="agent-results/memory.sqlite3")
    parser.add_argument("--memory-limit", type=int, default=5)
    parser.add_argument(
        "--memory-auto-ingest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Import completed runs into memory before retrieval.",
    )
    add_embedding_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_generation_attempts < 1 or args.max_repairs < 0:
        raise SystemExit("generation attempts must be >= 1 and repairs must be >= 0")
    if args.memory_limit < 0 or args.memory_limit > 20:
        raise SystemExit("--memory-limit must be between 0 and 20")
    if bool(args.remote_host) != bool(args.remote_root):
        raise SystemExit("--remote-host and --remote-root must be supplied together")
    repo_root = Path(args.repo_root).resolve()
    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = repo_root / spec_path
    try:
        spec = load_operator_spec(spec_path, repo_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid operator spec: {exc}") from exc

    implementation_path = resolve_repo_path(repo_root, spec.implementation_file)
    test_path = resolve_repo_path(repo_root, spec.test_file)
    if not args.allow_existing and (implementation_path.exists() or test_path.exists()):
        raise SystemExit("operator files already exist; pass --allow-existing to complete them")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    run_dir = results_root / f"{timestamp}-{slugify(spec.name)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    inventory = discover_operators(repo_root)
    references = choose_references(spec, inventory)
    memory_store: MemoryStore | None = None
    memory_warnings: list[str] = []
    memory_ingest = None
    generation_memories: list[dict] = []
    if args.memory:
        memory_path = Path(args.memory_db)
        if not memory_path.is_absolute():
            memory_path = repo_root / memory_path
        try:
            provider = build_embedding_provider(
                args.embedding_provider,
                model=args.embedding_model,
                base_url=args.embedding_base_url,
                api_key_env=args.embedding_api_key_env,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            provider = None
            memory_warnings.append(
                f"embedding provider unavailable; using lexical retrieval: {exc}"
            )
        try:
            memory_store = MemoryStore(memory_path, provider)
            if args.memory_auto_ingest:
                memory_ingest = ingest_results(memory_store, results_root)
            generation_memories = memory_store.retrieve(
                spec_memory_query(spec, references),
                args.memory_limit,
            )
        except Exception as exc:
            memory_warnings.append(f"memory unavailable; continuing without it: {exc}")
            if memory_store is not None:
                memory_store.close()
            memory_store = None
            generation_memories = []
    generation_memory_context = render_memory_context(generation_memories)

    task_path = repo_root / "tasks" / "operators" / f"{spec.name}.md"
    if task_path.exists() and not args.force_task:
        raise SystemExit(f"task already exists: {task_path}; pass --force-task to replace it")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        render_task(spec, references, generation_memory_context),
        encoding="utf-8",
    )
    write_json(run_dir / "operator-spec.json", asdict(spec))
    write_json(run_dir / "references.json", references)
    write_json(run_dir / "retrieved-memory-generation.json", generation_memories)
    write_json(
        run_dir / "memory-status.json",
        {
            "enabled": args.memory,
            "database": args.memory_db if args.memory else None,
            "embedding_provider": args.embedding_provider,
            "auto_ingest": memory_ingest,
            "retrieved": len(generation_memories),
            "warnings": memory_warnings,
        },
    )

    if args.remote_host:
        preflight = run_remote_preflight(args.remote_host, args.remote_root)
        environment = {
            "status": "ready" if preflight["status"] == "passed" else "incomplete",
            "architecture": preflight.get("architecture"),
            "execution": preflight.get("execution", {}),
            "python": {},
            "triton": {},
        }
    else:
        preflight = run_preflight(repo_root, args.source_env, False)
        environment = collect_environment(
            repo_root,
            source_env=args.source_env,
            dry_run=False,
        )
    write_json(run_dir / "preflight.json", preflight)
    write_json(run_dir / "environment.json", environment)
    prepared = {
        "operator": spec.name,
        "task_path": task_path.relative_to(repo_root).as_posix(),
        "references": [item["name"] for item in references],
        "preflight": preflight,
        "run_dir": run_dir.as_posix(),
        "retrieved_memories": [item["id"] for item in generation_memories],
        "memory_warnings": memory_warnings,
    }
    print(json.dumps(prepared, indent=2, sort_keys=True))
    if args.prepare_only:
        if memory_store is not None:
            memory_store.close()
        return 0
    if preflight["status"] != "passed":
        write_json(run_dir / "final-result.json", {**prepared, "status": "environment-failed"})
        if memory_store is not None:
            memory_store.close()
        return 2

    allowed_generation = {spec.implementation_file, spec.test_file}
    file_paths = [spec.implementation_file, spec.test_file]
    previous_snapshot = snapshot_files(repo_root, file_paths)
    contract_audit: ContractAudit | None = None
    for attempt in range(1, args.max_generation_attempts + 1):
        before_state = collect_workspace_state(repo_root)
        before_workspace = snapshot_workspace_bytes(repo_root, before_state)
        before_generation_snapshot = snapshot_files(repo_root, file_paths)
        prompt = (
            generation_prompt(
                spec,
                task_path.relative_to(repo_root),
                references,
                generation_memory_context,
            )
            if attempt == 1
            else contract_repair_prompt(spec, contract_audit)
        )
        codex_result = run_codex(
            repo_root,
            prompt,
            run_dir,
            f"codex-generation-{attempt}",
            args.codex_timeout_seconds,
        )
        write_json(run_dir / f"codex-generation-{attempt}.json", codex_result)
        if codex_result["status"] != "passed":
            after_state = collect_workspace_state(repo_root)
            changed = unauthorized_changes(before_state, after_state, set())
            restore_files(repo_root, before_generation_snapshot)
            restore_workspace_paths(repo_root, before_state, before_workspace, changed)
            write_json(run_dir / "final-result.json", {**prepared, "status": "generation-failed", "codex": codex_result})
            if memory_store is not None:
                memory_store.close()
            return 3
        after_state = collect_workspace_state(repo_root)
        unauthorized = unauthorized_changes(
            before_state,
            after_state,
            allowed_generation,
        )
        if unauthorized:
            restore_files(repo_root, before_generation_snapshot)
            restore_failures = restore_workspace_paths(
                repo_root,
                before_state,
                before_workspace,
                unauthorized,
            )
            write_json(
                run_dir / "final-result.json",
                {
                    **prepared,
                    "status": "generation-safety-blocked",
                    "unauthorized_changes": unauthorized,
                    "restore_failures": restore_failures,
                },
            )
            if memory_store is not None:
                memory_store.close()
            return 3
        current_snapshot = snapshot_files(repo_root, file_paths)
        write_patch(run_dir / f"generation-{attempt}.patch", previous_snapshot, current_snapshot)
        previous_snapshot = current_snapshot
        contract_audit = audit_test_contract(spec, repo_root)
        write_json(run_dir / f"contract-audit-{attempt}.json", asdict(contract_audit))
        if contract_audit.status == "passed":
            break
    assert contract_audit is not None
    if contract_audit.status != "passed":
        write_json(run_dir / "final-result.json", {**prepared, "status": "test-contract-failed", "contract_audit": asdict(contract_audit)})
        if memory_store is not None:
            memory_store.close()
        return 4

    locked_test_hash = contract_audit.test_sha256
    assert locked_test_hash is not None
    validations: list[dict] = []
    validation = run_validation(
        spec,
        repo_root,
        run_dir,
        1,
        args.timeout_seconds,
        args.remote_host,
        args.remote_root,
        args.source_env,
        environment,
        args.capture_pipeline,
        args.fresh_compile,
    )
    validations.append(validation)
    repair_history: list[dict] = []
    seen_implementation_hashes = {file_sha256(implementation_path)}
    stop_reason: str | None = None
    terminal_status: str | None = None
    validation_iteration = 1
    memory_retrievals: list[dict] = []
    for repair_index in range(1, args.max_repairs + 1):
        if validation["status"] == "passed":
            break
        decision = plan_repair(validation, args.allow_compiler_workaround)
        write_json(run_dir / f"repair-decision-{repair_index}.json", asdict(decision))
        if decision.action != "repair":
            stop_reason = decision.reason
            terminal_status = "blocked"
            break
        before_state = collect_workspace_state(repo_root)
        before_workspace = snapshot_workspace_bytes(repo_root, before_state)
        before_snapshot = snapshot_files(repo_root, file_paths)
        before_implementation_hash = file_sha256(implementation_path)
        working_memory = build_working_memory(
            spec,
            validation,
            repair_history,
            locked_test_hash,
            args.max_repairs - repair_index + 1,
        )
        write_json(
            run_dir / f"working-memory-repair-{repair_index}.json",
            working_memory,
        )
        repair_memories: list[dict] = []
        if memory_store is not None:
            try:
                repair_memories = memory_store.retrieve(
                    spec_memory_query(
                        spec,
                        references,
                        validation=validation,
                        environment=environment,
                    ),
                    args.memory_limit,
                )
            except Exception as exc:
                memory_warnings.append(
                    f"repair {repair_index} memory retrieval failed: {exc}"
                )
        write_json(
            run_dir / f"retrieved-memory-repair-{repair_index}.json",
            repair_memories,
        )
        memory_retrievals.append(
            {
                "repair": repair_index,
                "failure_stage": failure_stage(validation),
                "memory_ids": [item["id"] for item in repair_memories],
            }
        )
        codex_result = run_codex(
            repo_root,
            repair_prompt(
                spec,
                validation,
                decision,
                repair_history,
                render_memory_context(repair_memories),
                working_memory,
            ),
            run_dir,
            f"codex-repair-{repair_index}",
            args.codex_timeout_seconds,
        )
        write_json(run_dir / f"codex-repair-{repair_index}.json", codex_result)
        if codex_result["status"] != "passed":
            after_state = collect_workspace_state(repo_root)
            changed = unauthorized_changes(before_state, after_state, set())
            restore_files(repo_root, before_snapshot)
            restore_failures = restore_workspace_paths(
                repo_root,
                before_state,
                before_workspace,
                [relative for relative in changed if relative not in before_snapshot],
            )
            repair_history.append(
                {
                    "attempt": repair_index,
                    "decision": asdict(decision),
                    "outcome": "codex-failed",
                    "accepted": False,
                    "reason": codex_result.get("reason") or "Codex invocation failed",
                    "changed_files": changed,
                    "restore_failures": restore_failures,
                    "before_implementation_sha256": before_implementation_hash,
                    "after_implementation_sha256": None,
                    "candidate_validation_iteration": None,
                }
            )
            write_json(run_dir / "repair-history.json", repair_history)
            if restore_failures:
                stop_reason = "Codex failed after changes that could not be restored"
                terminal_status = "safety-blocked"
                break
            continue

        after_state = collect_workspace_state(repo_root)
        changed_outside_implementation = unauthorized_changes(
            before_state,
            after_state,
            {spec.implementation_file},
        )
        if changed_outside_implementation:
            restore_files(repo_root, before_snapshot)
            unrestored_existing = restore_workspace_paths(
                repo_root,
                before_state,
                before_workspace,
                [
                    relative
                    for relative in changed_outside_implementation
                    if relative not in before_snapshot
                ],
            )
            outcome = "locked-test-modified" if (
                spec.test_file in changed_outside_implementation
                and not unrestored_existing
            ) else "scope-violation"
            reason = (
                "the locked acceptance test changed and was restored"
                if outcome == "locked-test-modified"
                else "Codex changed files outside the implementation allowlist"
            )
            repair_history.append(
                {
                    "attempt": repair_index,
                    "decision": asdict(decision),
                    "outcome": outcome,
                    "accepted": False,
                    "reason": reason,
                    "unauthorized_changes": changed_outside_implementation,
                    "unrestored_existing_changes": unrestored_existing,
                    "before_implementation_sha256": before_implementation_hash,
                    "after_implementation_sha256": None,
                    "candidate_validation_iteration": None,
                }
            )
            write_json(run_dir / "repair-history.json", repair_history)
            if unrestored_existing:
                stop_reason = (
                    "automatic repair changed pre-existing files outside the allowlist; "
                    "manual review is required"
                )
                terminal_status = "safety-blocked"
                break
            continue

        if not test_path.exists() or file_sha256(test_path) != locked_test_hash:
            restore_files(repo_root, before_snapshot)
            repair_history.append(
                {
                    "attempt": repair_index,
                    "decision": asdict(decision),
                    "outcome": "locked-test-modified",
                    "accepted": False,
                    "reason": "the locked acceptance test changed and was restored",
                    "before_implementation_sha256": before_implementation_hash,
                    "after_implementation_sha256": None,
                    "candidate_validation_iteration": None,
                }
            )
            write_json(run_dir / "repair-history.json", repair_history)
            continue

        after_snapshot = snapshot_files(repo_root, file_paths)
        write_patch(run_dir / f"repair-{repair_index}.patch", before_snapshot, after_snapshot)
        if not implementation_path.exists():
            restore_files(repo_root, before_snapshot)
            repair_history.append(
                {
                    "attempt": repair_index,
                    "decision": asdict(decision),
                    "outcome": "implementation-deleted",
                    "accepted": False,
                    "reason": "the implementation file was deleted and restored",
                    "before_implementation_sha256": before_implementation_hash,
                    "after_implementation_sha256": None,
                    "candidate_validation_iteration": None,
                }
            )
            write_json(run_dir / "repair-history.json", repair_history)
            continue
        after_implementation_hash = file_sha256(implementation_path)
        if after_implementation_hash == before_implementation_hash:
            restore_files(repo_root, before_snapshot)
            repair_history.append(
                {
                    "attempt": repair_index,
                    "decision": asdict(decision),
                    "outcome": "no-change",
                    "accepted": False,
                    "reason": "Codex did not change the implementation",
                    "before_implementation_sha256": before_implementation_hash,
                    "after_implementation_sha256": after_implementation_hash,
                    "candidate_validation_iteration": None,
                }
            )
            write_json(run_dir / "repair-history.json", repair_history)
            continue
        if after_implementation_hash in seen_implementation_hashes:
            restore_files(repo_root, before_snapshot)
            repair_history.append(
                {
                    "attempt": repair_index,
                    "decision": asdict(decision),
                    "outcome": "repeated-implementation",
                    "accepted": False,
                    "reason": "the repair repeated an implementation already tested",
                    "before_implementation_sha256": before_implementation_hash,
                    "after_implementation_sha256": after_implementation_hash,
                    "candidate_validation_iteration": None,
                }
            )
            write_json(run_dir / "repair-history.json", repair_history)
            continue

        validation_iteration += 1
        candidate_validation = run_validation(
            spec,
            repo_root,
            run_dir,
            validation_iteration,
            args.timeout_seconds,
            args.remote_host,
            args.remote_root,
            args.source_env,
            environment,
            args.capture_pipeline,
            args.fresh_compile,
        )
        validations.append(candidate_validation)
        outcome, accepted, reason = compare_validations(
            validation,
            candidate_validation,
        )
        repair_history.append(
            {
                "attempt": repair_index,
                "decision": asdict(decision),
                "outcome": outcome,
                "accepted": accepted,
                "reason": reason,
                "before_implementation_sha256": before_implementation_hash,
                "after_implementation_sha256": after_implementation_hash,
                "candidate_validation_iteration": validation_iteration,
            }
        )
        if accepted:
            seen_implementation_hashes.add(after_implementation_hash)
            validation = candidate_validation
            if memory_store is not None:
                for item in repair_memories:
                    memory_store.mark_useful(item["id"])
        else:
            restore_files(repo_root, before_snapshot)
        write_json(run_dir / "repair-history.json", repair_history)

    if validation["status"] == "passed":
        terminal_status = "passed"
        stop_reason = "the locked acceptance test passed"
    elif terminal_status is None:
        final_decision = plan_repair(validation, args.allow_compiler_workaround)
        if final_decision.action == "stop":
            terminal_status = "blocked"
            stop_reason = final_decision.reason
        else:
            terminal_status = "repair-exhausted"
            stop_reason = f"the maximum of {args.max_repairs} repair attempts was reached"
    test_hash_verified = test_path.exists() and file_sha256(test_path) == locked_test_hash
    final = {
        **prepared,
        "status": terminal_status,
        "validation_status": validation["status"],
        "stop_reason": stop_reason,
        "final_failure_stage": failure_stage(validation),
        "contract_audit": asdict(contract_audit),
        "locked_test_sha256": locked_test_hash,
        "locked_test_verified": test_hash_verified,
        "implementation_sha256": file_sha256(implementation_path),
        "validations": validations,
        "repair_attempts": len(repair_history),
        "accepted_repairs": sum(item["accepted"] for item in repair_history),
        "rejected_repairs": sum(not item["accepted"] for item in repair_history),
        "repair_history": repair_history,
        "memory_retrievals": memory_retrievals,
        "memory_warnings": memory_warnings,
    }
    update_task_validation_record(task_path, final)
    write_json(run_dir / "final-result.json", final)
    (run_dir / "repair-summary.md").write_text(
        render_repair_summary(final),
        encoding="utf-8",
    )
    if memory_store is not None:
        for record in records_from_run(run_dir):
            memory_store.add(record)
        write_json(run_dir / "memory-stats.json", memory_store.stats())
        memory_store.close()
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if terminal_status == "passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())
