#!/usr/bin/env python3
"""Discover Triton-RISCV validation targets.

The discovery output is intentionally mechanical and machine-readable. Later
agent stages can use it to choose targets, run tests, classify failures, and
summarize project coverage.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


SKIP_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "agent-results",
    "build",
    "united-interns",
}
IMPLEMENTATION_ROOTS = (
    Path("backend"),
    Path("codex_agent"),
    Path("python/examples"),
    Path("lib"),
    Path("include/triton-shared"),
    Path("tools"),
)


TL_OP_RE = re.compile(r"\btl\.([A-Za-z_][A-Za-z0-9_]*)")
RUN_RE = re.compile(r"^\s*(?://|;)\s*RUN:\s*(.*)$")
PASS_RE = re.compile(r"--([A-Za-z0-9][A-Za-z0-9_-]*)")
CMAKE_CALL_RE = re.compile(
    r"\b("
    r"add_custom_target|add_executable|add_library|add_lit_testsuite|"
    r"add_llvm_executable|add_triton_library|add_triton_plugin|"
    r"add_mlir_[A-Za-z0-9_]+"
    r")\s*\(",
    re.IGNORECASE,
)
CMAKE_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[^\s()]+')
CMAKE_SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".td")


@dataclass
class PytestTarget:
    kind: str
    path: str
    command: str
    implementation_files: list[str] = field(default_factory=list)
    test_functions: list[str] = field(default_factory=list)
    triton_kernels: list[str] = field(default_factory=list)
    tl_ops: list[str] = field(default_factory=list)
    parametrize_count: int = 0
    likely_area: str = "unknown"


@dataclass
class LitTarget:
    kind: str
    path: str
    command: str
    run_lines: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    dialect_keywords: list[str] = field(default_factory=list)
    likely_area: str = "unknown"


@dataclass
class BuildTarget:
    kind: str
    path: str
    name: str
    command: str
    target_type: str
    source_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    likely_area: str = "build"


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_skip(path: Path, repo_root: Path) -> bool:
    try:
        parts = path.relative_to(repo_root).parts
    except ValueError:
        return True
    return any(part in SKIP_DIRECTORY_NAMES for part in parts)


def has_triton_jit_decorator(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text == "triton.jit" or text.startswith("triton.jit("):
            return True
    return False


def count_parametrize_decorators(node: ast.FunctionDef) -> int:
    count = 0
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text.startswith("pytest.mark.parametrize"):
            count += 1
    return count


def python_module_file(repo_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    module_file = repo_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = repo_root / relative / "__init__.py"
    return package_file if package_file.is_file() else None


def local_imported_python_files(
    tree: ast.AST,
    source_path: Path,
    repo_root: Path,
) -> list[Path]:
    files: list[Path] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = python_module_file(repo_root, alias.name)
                if candidate is not None:
                    files.append(candidate)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = source_path.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                candidates = [base / f"{node.module.replace('.', '/')}.py"]
            else:
                candidates = [base / f"{alias.name}.py" for alias in node.names]
            files.extend(candidate for candidate in candidates if candidate.is_file())
        elif node.module:
            candidate = python_module_file(repo_root, node.module)
            if candidate is not None:
                files.append(candidate)
    return unique_sorted_path(files)


def unique_sorted_path(values: Iterable[Path]) -> list[Path]:
    return sorted(set(values), key=lambda path: path.as_posix())


def infer_pytest_area(path: Path, tl_ops: list[str]) -> str:
    name = path.stem.removeprefix("test_")
    path_text = path.as_posix()
    op_set = set(tl_ops)

    if "flaggems" in path_text:
        return "operator-migration"
    if "dot" in op_set or "matmul" in name or "mm" in name:
        return "matrix"
    if {"sum", "max", "min"} & op_set or "reduce" in name or "norm" in name:
        return "reduction"
    if {"load", "store"} & op_set and ("gather" in name or "scatter" in name):
        return "memory-indexing"
    if {"exp", "erf", "sqrt", "rsqrt", "log"} & op_set:
        return "math"
    if "mask" in name or "where" in op_set:
        return "masking"
    return "general"


def discover_pytest_target(path: Path, repo_root: Path) -> PytestTarget | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    implementation_paths = local_imported_python_files(tree, path, repo_root)
    implementation_sources: list[str] = []
    for implementation_path in implementation_paths:
        try:
            implementation_sources.append(implementation_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue

    test_functions: list[str] = []
    triton_kernels: list[str] = []
    parametrize_count = 0

    trees = [tree]
    for implementation_source in implementation_sources:
        try:
            trees.append(ast.parse(implementation_source))
        except SyntaxError:
            continue

    for index, current_tree in enumerate(trees):
        for node in ast.walk(current_tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if index == 0 and node.name.startswith("test_"):
                test_functions.append(node.name)
                parametrize_count += count_parametrize_decorators(node)
            if has_triton_jit_decorator(node):
                triton_kernels.append(node.name)

    if not test_functions and not triton_kernels:
        return None

    relative_path = rel(path, repo_root)
    combined_source = "\n".join([source, *implementation_sources])
    tl_ops = unique_sorted(TL_OP_RE.findall(combined_source))

    return PytestTarget(
        kind="pytest",
        path=relative_path,
        command=f"python -m pytest -q {relative_path} -s",
        implementation_files=[rel(item, repo_root) for item in implementation_paths],
        test_functions=unique_sorted(test_functions),
        triton_kernels=unique_sorted(triton_kernels),
        tl_ops=tl_ops,
        parametrize_count=parametrize_count,
        likely_area=infer_pytest_area(path, tl_ops),
    )


def infer_lit_area(path: Path, passes: list[str], dialects: list[str]) -> str:
    path_text = path.as_posix().lower()
    pass_text = " ".join(passes)

    if "sanitizer" in path_text:
        return "sanitizer"
    if "tritontostructured" in path_text or "triton-to-structured" in pass_text:
        return "triton-to-structured"
    if "structuredtomemref" in path_text or "structured-to-memref" in pass_text:
        return "structured-to-memref"
    if "tritontolinalg" in path_text or "triton-to-linalg" in pass_text:
        return "triton-to-linalg"
    if "tritonarithtolinalg" in path_text or "triton-arith-to-linalg" in pass_text:
        return "triton-arith-to-linalg"
    if "tritontoptr" in path_text or "triton-to-ptr" in pass_text:
        return "triton-to-ptr"
    if "memref" in dialects:
        return "memref"
    return "general"


def discover_lit_target(path: Path, repo_root: Path) -> LitTarget | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    run_lines = [match.group(1).strip() for line in source.splitlines() if (match := RUN_RE.match(line))]
    if not run_lines:
        return None

    passes = unique_sorted(PASS_RE.findall("\n".join(run_lines)))
    dialect_keywords = unique_sorted(
        keyword
        for keyword in [
            "tt",
            "tts",
            "tptr",
            "linalg",
            "memref",
            "scf",
            "arith",
            "bufferization",
            "llvm",
        ]
        if re.search(rf"\b{re.escape(keyword)}\.", source)
    )
    relative_path = rel(path, repo_root)

    return LitTarget(
        kind="lit",
        path=relative_path,
        command=f"llvm-lit -sv {relative_path}",
        run_lines=run_lines,
        passes=passes,
        dialect_keywords=dialect_keywords,
        likely_area=infer_lit_area(path, passes, dialect_keywords),
    )


def cmake_calls(source: str) -> Iterable[tuple[str, str]]:
    for match in CMAKE_CALL_RE.finditer(source):
        depth = 1
        index = match.end()
        quoted = False
        escaped = False
        while index < len(source) and depth:
            character = source[index]
            if escaped:
                escaped = False
            elif character == "\\" and quoted:
                escaped = True
            elif character == '"':
                quoted = not quoted
            elif not quoted and character == "(":
                depth += 1
            elif not quoted and character == ")":
                depth -= 1
            index += 1
        if depth == 0:
            yield match.group(1), source[match.end() : index - 1]


def cmake_target_type(function_name: str) -> str:
    lowered = function_name.lower()
    if lowered == "add_lit_testsuite":
        return "test-suite"
    if lowered == "add_custom_target":
        return "custom"
    if "executable" in lowered:
        return "executable"
    if "plugin" in lowered:
        return "plugin"
    return "library"


def resolve_cmake_source(
    token: str,
    cmake_path: Path,
    repo_root: Path,
) -> str | None:
    token = token.strip('"')
    token = token.replace("${CMAKE_CURRENT_SOURCE_DIR}/", "")
    token = token.replace("${CMAKE_CURRENT_LIST_DIR}/", "")
    if not token.lower().endswith(CMAKE_SOURCE_SUFFIXES):
        return None
    if "${" in token or "$<" in token:
        return None
    candidate = (cmake_path.parent / token).resolve()
    try:
        return candidate.relative_to(repo_root).as_posix()
    except ValueError:
        return None


def cmake_dependencies(tokens: list[str]) -> list[str]:
    dependencies: list[str] = []
    keywords = {
        "ARGS",
        "DEPENDS",
        "LINK_LIBS",
        "PARTIAL_SOURCES_INTENDED",
        "SOURCES",
    }
    active = False
    for token in tokens:
        normalized = token.strip('"')
        upper = normalized.upper()
        if upper in {"DEPENDS", "LINK_LIBS"}:
            active = True
            continue
        if upper in keywords:
            active = False
            continue
        if active and not normalized.startswith(("${", "$<")):
            dependencies.append(normalized)
    return unique_sorted(dependencies)


def discover_cmake_targets(path: Path, repo_root: Path) -> list[BuildTarget]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    targets: list[BuildTarget] = []
    relative_path = rel(path, repo_root)
    for function_name, arguments in cmake_calls(source):
        tokens = [token.strip('"') for token in CMAKE_TOKEN_RE.findall(arguments)]
        if not tokens or tokens[0].startswith(("${", "$<")):
            continue
        name = tokens[0]
        target_type = cmake_target_type(function_name)
        sources = unique_sorted(
            source_path
            for token in tokens[1:]
            if (source_path := resolve_cmake_source(token, path, repo_root))
        )
        targets.append(
            BuildTarget(
                kind="build",
                path=relative_path,
                name=name,
                command=f"cmake --build <build-dir> --target {name}",
                target_type=target_type,
                source_files=sources,
                dependencies=cmake_dependencies(tokens[1:]),
                likely_area="testing" if target_type == "test-suite" else "build",
            )
        )
    return targets


def pytest_environment() -> dict:
    return {
        "tools": ["python", "pytest", "triton"],
        "source_env": True,
        "execution": "native-riscv-required-for-hardware-validation",
    }


def lit_environment(target: LitTarget) -> dict:
    tools = ["llvm-lit"]
    run_text = " ".join(target.run_lines)
    for tool in ["triton-shared-opt", "FileCheck", "mlir-opt", "llc"]:
        if tool in run_text:
            tools.append(tool)
    return {
        "tools": unique_sorted(tools),
        "source_env": True,
        "execution": "configured-toolchain",
    }


def unified_targets(
    pytest_targets: list[PytestTarget],
    lit_targets: list[LitTarget],
    build_targets: list[BuildTarget],
) -> list[dict]:
    targets: list[dict] = []
    for target in pytest_targets:
        targets.append(
            {
                "id": f"pytest::{target.path}",
                "kind": target.kind,
                "name": Path(target.path).stem,
                "path": target.path,
                "command": target.command,
                "source_files": target.implementation_files,
                "dependencies": target.implementation_files,
                "environment": pytest_environment(),
                "likely_area": target.likely_area,
            }
        )
    for target in lit_targets:
        targets.append(
            {
                "id": f"lit::{target.path}",
                "kind": target.kind,
                "name": Path(target.path).stem,
                "path": target.path,
                "command": target.command,
                "source_files": [target.path],
                "dependencies": target.passes,
                "environment": lit_environment(target),
                "likely_area": target.likely_area,
            }
        )
    for target in build_targets:
        targets.append(
            {
                "id": f"cmake::{target.name}@{target.path}",
                "kind": target.kind,
                "name": target.name,
                "path": target.path,
                "command": target.command,
                "source_files": target.source_files,
                "dependencies": target.dependencies,
                "environment": {
                    "tools": ["cmake", "ninja"],
                    "source_env": False,
                    "execution": "configured-build-tree",
                },
                "likely_area": target.likely_area,
                "target_type": target.target_type,
            }
        )
    return sorted(targets, key=lambda item: item["id"])


def implementation_files(repo_root: Path) -> list[str]:
    files: list[str] = []
    for relative_root in IMPLEMENTATION_ROOTS:
        source_root = repo_root / relative_root
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if not path.is_file() or should_skip(path, repo_root):
                continue
            if path.name.startswith("test_") or path.name in {"__init__.py", "conftest.py"}:
                continue
            if path.suffix not in {".py", ".cpp", ".cc", ".c", ".td"}:
                continue
            files.append(rel(path, repo_root))
    root_plugin = repo_root / "triton_shared.cc"
    if root_plugin.is_file():
        files.append(rel(root_plugin, repo_root))
    return unique_sorted(files)


def lit_source_prefix(test_path: str) -> str | None:
    parts = Path(test_path).parts
    if len(parts) < 3 or parts[0] != "test":
        return None
    if parts[1] not in {"Conversion", "Sanitizer", "Transform"}:
        return None
    return Path("lib", parts[1], parts[2]).as_posix() + "/"


def build_coverage(
    repo_root: Path,
    pytest_targets: list[PytestTarget],
    lit_targets: list[LitTarget],
    build_targets: list[BuildTarget],
) -> dict:
    test_evidence: dict[str, set[str]] = {}
    build_evidence: dict[str, set[str]] = {}
    candidates = implementation_files(repo_root)

    for target in pytest_targets:
        for source in target.implementation_files:
            test_evidence.setdefault(source, set()).add(target.path)

    for target in lit_targets:
        prefix = lit_source_prefix(target.path)
        if prefix is None:
            continue
        for source in candidates:
            if source.startswith(prefix):
                test_evidence.setdefault(source, set()).add(target.path)

    for target in build_targets:
        for source in target.source_files:
            build_evidence.setdefault(source, set()).add(target.name)

    mappings: list[dict] = []
    gaps: list[dict] = []
    status_counts = {"tested": 0, "build-only": 0, "unmapped": 0}
    for source in candidates:
        tests = sorted(test_evidence.get(source, set()))
        builds = sorted(build_evidence.get(source, set()))
        if tests:
            status = "tested"
            reason = "mapped to at least one pytest or lit target"
        elif builds:
            status = "build-only"
            reason = "included in a build target but no behavioral test was mapped"
        else:
            status = "unmapped"
            reason = "no pytest, lit, or direct CMake source mapping was found"
        status_counts[status] += 1
        mapping = {
            "source": source,
            "status": status,
            "test_targets": tests,
            "build_targets": builds,
        }
        mappings.append(mapping)
        if status != "tested":
            gaps.append(
                {
                    **mapping,
                    "reason": reason,
                    "suggested_action": (
                        "add or map a behavioral test"
                        if status == "build-only"
                        else "review the source and add a validation target"
                    ),
                }
            )
    return {
        "method": "heuristic source/import/directory mapping",
        "summary": {
            "implementation_files": len(candidates),
            **status_counts,
            "coverage_gaps": len(gaps),
        },
        "mappings": mappings,
        "gaps": gaps,
    }


def discover(repo_root: Path) -> dict:
    repo_root = repo_root.resolve()
    pytest_targets: list[PytestTarget] = []
    lit_targets: list[LitTarget] = []
    build_targets: list[BuildTarget] = []

    for path in sorted(repo_root.rglob("test_*.py")):
        if should_skip(path, repo_root):
            continue
        target = discover_pytest_target(path, repo_root)
        if target is not None:
            pytest_targets.append(target)

    lit_paths = [*repo_root.rglob("*.mlir"), *repo_root.rglob("*.ll")]
    for path in sorted(lit_paths):
        if should_skip(path, repo_root):
            continue
        target = discover_lit_target(path, repo_root)
        if target is not None:
            lit_targets.append(target)

    for path in sorted(repo_root.rglob("CMakeLists.txt")):
        if should_skip(path, repo_root):
            continue
        build_targets.extend(discover_cmake_targets(path, repo_root))

    area_counts: dict[str, int] = {}
    for target in [*pytest_targets, *lit_targets, *build_targets]:
        area_counts[target.likely_area] = area_counts.get(target.likely_area, 0) + 1

    targets = unified_targets(pytest_targets, lit_targets, build_targets)
    coverage = build_coverage(repo_root, pytest_targets, lit_targets, build_targets)

    return {
        "schema_version": 2,
        "repo_root": repo_root.as_posix(),
        "summary": {
            "pytest_targets": len(pytest_targets),
            "lit_targets": len(lit_targets),
            "build_targets": len(build_targets),
            "test_suite_targets": sum(
                target.target_type == "test-suite" for target in build_targets
            ),
            "total_targets": len(targets),
            "area_counts": dict(sorted(area_counts.items())),
            "coverage": coverage["summary"],
        },
        "targets": targets,
        "pytest_targets": [asdict(target) for target in pytest_targets],
        "lit_targets": [asdict(target) for target in lit_targets],
        "build_targets": [asdict(target) for target in build_targets],
        "coverage": coverage,
    }


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(result: dict) -> str:
    summary = result["summary"]
    coverage = summary["coverage"]
    lines = [
        "# Triton-RISCV Project Validation Inventory",
        "",
        "## Summary",
        "",
        f"- Pytest targets: {summary['pytest_targets']}",
        f"- Lit targets: {summary['lit_targets']}",
        f"- CMake/Ninja targets: {summary['build_targets']}",
        f"- Total targets: {summary['total_targets']}",
        f"- Implementation files reviewed: {coverage['implementation_files']}",
        f"- Mapped to tests: {coverage['tested']}",
        f"- Build-only: {coverage['build-only']}",
        f"- Unmapped: {coverage['unmapped']}",
        "",
        "Coverage mapping is heuristic. A mapped file has related validation evidence, "
        "not a proof of complete behavioral coverage.",
        "",
        "## Validation Targets",
        "",
        "| ID | Kind | Area | Command |",
        "| --- | --- | --- | --- |",
    ]
    for target in result["targets"]:
        lines.append(
            f"| {markdown_cell(target['id'])} | {target['kind']} | "
            f"{target['likely_area']} | `{markdown_cell(target['command'])}` |"
        )
    lines.extend(
        [
            "",
            "## Coverage Gaps",
            "",
            "| Source | Status | Reason | Suggested action |",
            "| --- | --- | --- | --- |",
        ]
    )
    for gap in result["coverage"]["gaps"]:
        lines.append(
            f"| {markdown_cell(gap['source'])} | {gap['status']} | "
            f"{markdown_cell(gap['reason'])} | {markdown_cell(gap['suggested_action'])} |"
        )
    if not result["coverage"]["gaps"]:
        lines.append("|  |  | No heuristic coverage gaps found. |  |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Triton-RISCV pytest, lit, CMake/Ninja, and coverage targets."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write discovery JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optionally write a Markdown inventory and coverage-gap report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    result = discover(repo_root)
    content = json.dumps(result, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + "\n", encoding="utf-8")
        try:
            display_path = output_path.relative_to(repo_root)
        except ValueError:
            display_path = output_path
        print(f"wrote {display_path}")
    else:
        print(content)

    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_markdown(result), encoding="utf-8")
        try:
            display_path = report_path.relative_to(repo_root)
        except ValueError:
            display_path = report_path
        print(f"wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
