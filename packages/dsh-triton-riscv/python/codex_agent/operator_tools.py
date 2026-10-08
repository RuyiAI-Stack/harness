"""Typed, side-effect-free Triton-RISCV domain tools."""

from __future__ import annotations

from dataclasses import asdict
from difflib import get_close_matches
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Literal

from pydantic import BaseModel, Field

from codex_agent.discover_operators import (
    OPERATOR_ROOT,
    discover_operator,
    is_valid_operator_name,
)


class OperatorFileResult(BaseModel):
    path: str
    sha256: str
    start_line: int
    end_line: int
    total_lines: int
    next_line: int | None
    content: str


def read_operator_file(repo_root: Path, path: str, start_line: int = 1,
                       max_lines: int = 200) -> OperatorFileResult:
    """Bounded source access without executing code or following symlinks."""
    relative = PurePosixPath(path)
    parts = path.split("/")
    if (relative.is_absolute() or any(p in {"", ".", ".."} for p in parts)
            or "\\" in path or "\x00" in path):
        raise ValueError("path must be a normalized repository-relative path")
    if not ((relative.parent == PurePosixPath(OPERATOR_ROOT) and relative.suffix == ".py")
            or (relative.parent == PurePosixPath("tasks/operators") and relative.suffix == ".md")):
        raise ValueError("Only FlagGems source/tests and tasks/operators Markdown are readable")
    if (type(start_line) is not int or start_line < 1
            or type(max_lines) is not int or not 1 <= max_lines <= 400):
        raise ValueError("start_line must be positive; max_lines must be in 1..400")
    # Traverse with directory descriptors to avoid check-then-open symlink races.
    directory = os.open(repo_root.resolve(), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ValueError("Source must be a regular file no larger than 1 MiB")
            data = source.read(1024 * 1024 + 1)
    finally:
        os.close(directory)
    if len(data) > 1024 * 1024 or b"\x00" in data:
        raise ValueError("Source exceeds the size limit or contains binary data")
    lines = data.decode("utf-8").splitlines(keepends=True)
    if start_line > len(lines) + 1:
        raise ValueError("start_line is past the end of the file")
    selected, size = [], 0
    for line in lines[start_line - 1:start_line - 1 + max_lines]:
        length = len(line.encode("utf-8"))
        if size + length > 32768:
            if not selected:
                raise ValueError("A single source line exceeds the 32 KiB output budget")
            break
        selected.append(line)
        size += length
    end = start_line + len(selected) - 1
    return OperatorFileResult(path=path, sha256=hashlib.sha256(data).hexdigest(),
        start_line=start_line, end_line=end, total_lines=len(lines),
        next_line=end + 1 if end < len(lines) else None, content="".join(selected))


class OperatorEvidence(BaseModel):
    """Repository evidence returned for one discovered operator."""

    name: str
    visibility: str
    implementation_file: str
    test_files: list[str]
    test_nodes: list[str]
    validation_command: str
    triton_kernels: list[str]
    public_functions: list[str]
    tl_ops: list[str]
    torch_references: list[str]
    parametrize: list[str]
    test_contract: dict[str, Any] = Field(default_factory=dict)
    risk_hints: list[str]


class DiscoverOperatorResult(BaseModel):
    """Stable structured output for the discover_operator tool."""

    status: Literal["found", "not_found", "not_operator"]
    query: str
    message: str
    operator: OperatorEvidence | None = None
    suggestions: list[str] = Field(default_factory=list)


def _operator_names(repo_root: Path) -> list[str]:
    root = repo_root / OPERATOR_ROOT
    if not root.is_dir():
        return []
    return sorted(
        path.stem
        for path in root.glob("*.py")
        if path.name != "__init__.py"
        and not path.name.startswith("test_")
        and is_valid_operator_name(path.stem)
    )


def _suggestions(repo_root: Path, query: str) -> list[str]:
    names = _operator_names(repo_root)
    contains = [name for name in names if query.lower() in name.lower()]
    if contains:
        return contains[:5]
    return get_close_matches(query, names, n=5, cutoff=0.4)


def discover_operator_evidence(
    repo_root: Path,
    operator_name: str,
) -> DiscoverOperatorResult:
    """Return implementation and test evidence without executing repository code."""

    query = operator_name.strip()
    if not query:
        raise ValueError("operator_name cannot be empty")
    if not is_valid_operator_name(query):
        raise ValueError(
            "operator_name must be a Python identifier containing only letters, "
            "numbers, and underscores"
        )

    resolved_root = repo_root.resolve()
    implementation = resolved_root / OPERATOR_ROOT / f"{query}.py"
    if not implementation.is_file():
        return DiscoverOperatorResult(
            status="not_found",
            query=query,
            message=f"No operator implementation file was found for {query}.",
            suggestions=_suggestions(resolved_root, query),
        )

    target = discover_operator(implementation, resolved_root)
    if target is None:
        return DiscoverOperatorResult(
            status="not_operator",
            query=query,
            message=(
                f"{implementation.relative_to(resolved_root)} exists but contains "
                "no discoverable Triton JIT kernel."
            ),
        )

    return DiscoverOperatorResult(
        status="found",
        query=query,
        message=f"Discovered repository evidence for {query}.",
        operator=OperatorEvidence.model_validate(asdict(target)),
    )
