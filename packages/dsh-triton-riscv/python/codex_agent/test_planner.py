"""Diff-aware test planning for the Triton Shared project.

The planner consumes ``agent-results/project-targets.json`` (or the output of
``discover.py``), maps changed paths to affected targets, and emits a stable
plan suitable for CI or the web control plane.  Planning is read-only: commands
are returned for a runner to execute.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_TARGETS = Path("agent-results/project-targets.json")


@dataclass(frozen=True)
class DiffFile:
    path: str
    status: str = "M"


@dataclass
class PlannedTarget:
    id: str
    name: str
    kind: str
    path: str
    command: str
    likely_area: str = "general"
    dependencies: list[str] = field(default_factory=list)
    retries: int = 0
    timeout_seconds: int = 300
    reasons: list[str] = field(default_factory=list)
    regression: bool = False


@dataclass
class TestPlan:
    tier: str
    changed_files: list[str]
    affected_targets: list[str]
    targets: list[PlannedTarget]
    parallel_groups: list[list[str]]
    dependency_impact: dict[str, list[str]]
    retries: int
    timeout_seconds: int
    history_regressions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_project_targets(path: str | Path = DEFAULT_TARGETS) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("targets", [])) or list(data.get("pytest_targets", [])) + list(data.get("lit_targets", [])) + list(data.get("build_targets", []))


def read_git_diff(repo_root: str | Path = ".", base: str | None = None, *, include_untracked: bool = False) -> list[DiffFile]:
    """Return changed paths from git, including staged and unstaged changes.

    New files are included when ``include_untracked`` is enabled, which is
    useful for CI on a worktree where a change has not been staged yet.
    """
    root = Path(repo_root)
    commands = [["git", "diff", "--name-status"], ["git", "diff", "--cached", "--name-status"]]
    if base:
        commands = [["git", "diff", "--name-status", f"{base}...HEAD"], ["git", "diff", "--name-status", base]]
    seen: dict[str, DiffFile] = {}
    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, cwd=root, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                status, path = parts[0], parts[-1]
                # Rename/copy status is e.g. R100; keep the destination path.
                seen[path] = DiffFile(path=path, status=status)
    if include_untracked:
        try:
            out = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root, text=True, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if line.startswith("?? "):
                    path = line[3:].strip()
                    if path:
                        seen[path] = DiffFile(path=path, status="A")
        except (OSError, subprocess.CalledProcessError):
            pass
    # New, untracked files are part of a worktree diff from a CI user's point
    # of view and should influence planning as added files.
    if not include_untracked:
        return list(seen.values())
    try:
        out = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        for path in out.splitlines():
            if path:
                seen.setdefault(path, DiffFile(path=path, status="A"))
    except (OSError, subprocess.CalledProcessError):
        pass
    return list(seen.values())


def _normal(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _target_id(target: Mapping) -> str:
    return str(target.get("id") or f"{target.get('kind', 'target')}::{target.get('path', target.get('name', ''))}")


def _matches(path: str, target: Mapping) -> bool:
    path = _normal(path)
    candidates = [target.get("path", ""), *(target.get("source_files") or [])]
    for candidate in candidates:
        candidate = _normal(str(candidate))
        if not candidate:
            continue
        if path == candidate or path.startswith(candidate.rstrip("/") + "/") or path.endswith("/" + candidate):
            return True
    # A changed source in a target's directory generally indicates impact.
    tpath = _normal(str(target.get("path", "")))
    return bool(tpath and "/" in tpath and path.startswith(tpath.rsplit("/", 1)[0] + "/"))


def _history_ids(history: Iterable[Mapping] | None) -> set[str]:
    ids: set[str] = set()
    for item in history or ():
        status = str(item.get("status", item.get("outcome", ""))).lower()
        duration = item.get("duration_seconds")
        baseline = item.get("baseline_duration_seconds", item.get("baseline_duration"))
        perf_regression = False
        try:
            perf_regression = baseline is not None and duration is not None and float(duration) > float(baseline) * 1.5
        except (TypeError, ValueError):
            pass
        if status in {"failed", "fail", "failure", "error", "timeout", "timed_out", "timed-out", "flaky", "regression"} or perf_regression or item.get("exit_code", 0) not in (0, None):
            value = item.get("id") or item.get("target_id") or item.get("path")
            if value:
                ids.add(str(value))
    return ids


def _tier(changed: Sequence[str], impacted: Sequence[Mapping], requested: str | None) -> str:
    if requested and requested != "auto":
        if requested not in {"quick", "standard", "full"}:
            raise ValueError("tier must be quick, standard, full, or auto")
        return requested
    broad = any(Path(p).name in {"CMakeLists.txt", "pyproject.toml", "setup.py", "requirements.txt", "project-targets.json"} or p.startswith((".github/", "cmake/")) for p in changed)
    if broad or len(changed) > 20:
        return "full"
    return "quick" if len(impacted) <= 5 else "standard"


def plan_tests(
    changed_files: Sequence[str | DiffFile],
    targets: Sequence[Mapping] | Mapping | None = None,
    *,
    tier: str | None = "auto",
    history: Iterable[Mapping] | None = None,
    max_targets: int | None = None,
) -> TestPlan:
    """Build a deterministic plan from paths and discovered target metadata."""
    if targets is None:
        target_data = load_project_targets()
    elif isinstance(targets, Mapping):
        target_data = list(targets.get("targets", [])) or list(targets.get("pytest_targets", [])) + list(targets.get("lit_targets", [])) + list(targets.get("build_targets", []))
    else:
        target_data = list(targets)
    paths = [x.path if isinstance(x, DiffFile) else str(x) for x in changed_files]
    impacted = [t for t in target_data if any(_matches(p, t) for p in paths)]
    by_name = {_target_id(t): t for t in target_data}
    direct_ids = {_target_id(t) for t in impacted}
    # Include transitive dependents when a build target changes.
    closure = set(direct_ids)
    changed = True
    while changed:
        changed = False
        selected_names = {str(by_name[d].get("name")) for d in closure if d in by_name}
        for tid, target in by_name.items():
            deps = set(map(str, target.get("dependencies") or []))
            dep_names = {str(by_name[d].get("name")) for d in by_name if d in closure}
            # Include both prerequisites and dependents so a changed leaf gets
            # a buildable chain and a changed library exercises its consumers.
            if tid not in closure and (deps & closure or deps & dep_names or str(target.get("name")) in selected_names):
                closure.add(tid); changed = True
            if tid in closure:
                for candidate, value in by_name.items():
                    if str(value.get("name")) in deps and candidate not in closure:
                        closure.add(candidate); changed = True
    history_ids = _history_ids(history)
    history_matches = {tid for tid, t in by_name.items() if tid in history_ids or str(t.get("path")) in history_ids or str(t.get("name")) in history_ids}
    level = _tier(paths, impacted, tier)
    if level == "full":
        selected_ids = set(by_name)
    elif level == "standard":
        areas = {str(t.get("likely_area", "general")) for t in impacted}
        selected_ids = closure | {tid for tid, t in by_name.items() if t.get("likely_area", "general") in areas and t.get("kind") in {"pytest", "lit"}}
    else:
        selected_ids = closure
    selected_ids |= history_matches
    if not selected_ids and target_data:
        # Empty diffs still run a tiny smoke set for useful CI feedback.
        selected_ids = {_target_id(t) for t in target_data[:3]}
    ordered = [t for t in target_data if _target_id(t) in selected_ids]
    if max_targets is not None:
        ordered = ordered[:max_targets]
    retries, timeout = ({"quick": (0, 180), "standard": (1, 600), "full": (2, 1800)})[level]
    planned: list[PlannedTarget] = []
    for t in ordered:
        tid = _target_id(t)
        reasons = []
        if tid in direct_ids: reasons.append("changed source/path")
        if tid in closure - direct_ids: reasons.append("dependency impact")
        regression = tid in history_matches
        if regression: reasons.append("historical regression")
        planned.append(PlannedTarget(tid, str(t.get("name", tid)), str(t.get("kind", "test")), str(t.get("path", "")), str(t.get("command", "")), str(t.get("likely_area", "general")), list(map(str, t.get("dependencies") or [])), int(t.get("retries", retries)), int(t.get("timeout_seconds", timeout)), reasons, regression))
    # Group independent targets by area; dependency-heavy targets run later.
    groups: dict[str, list[str]] = {}
    name_to_id = {str(v.get("name")): k for k, v in by_name.items()}
    for t in planned:
        dep_ids = {name_to_id.get(dep, dep) for dep in t.dependencies} & selected_ids
        key = "deps:" + ",".join(sorted(dep_ids)) if dep_ids else "area:" + t.likely_area
        groups.setdefault(key, []).append(t.id)
    return TestPlan(level, paths, sorted(direct_ids), planned, list(groups.values()), {k: list(map(str, v.get("dependencies") or [])) for k, v in by_name.items() if k in selected_ids}, retries, timeout, sorted(history_matches & selected_ids))


def build_test_plan(*args, **kwargs) -> TestPlan:
    return plan_tests(*args, **kwargs)


def plan_from_git(repo_root: str | Path = ".", targets_path: str | Path = DEFAULT_TARGETS, *, base: str | None = None, tier: str | None = "auto", history: Iterable[Mapping] | None = None, max_targets: int | None = None) -> TestPlan:
    """Convenience API that reads git changes and project target metadata."""
    return plan_tests(read_git_diff(repo_root, base, include_untracked=True), load_project_targets(targets_path), tier=tier, history=history, max_targets=max_targets)


def detect_changed_files(repo_root: str | Path = ".", base: str | None = None) -> list[str]:
    return [item.path for item in read_git_diff(repo_root, base)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--base")
    parser.add_argument("--tier", choices=["auto", "quick", "standard", "full"], default="auto")
    parser.add_argument("--changed", nargs="*", help="changed paths (defaults to git diff)")
    parser.add_argument("--history", type=Path)
    args = parser.parse_args(argv)
    changes = args.changed if args.changed else read_git_diff(args.repo_root, args.base)
    history = json.loads(args.history.read_text()) if args.history else None
    plan = plan_tests(changes, load_project_targets(args.targets), tier=args.tier, history=history)
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
