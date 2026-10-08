"""Project-level failure grouping and guarded repair planning.

This module lifts the operator diagnosis primitives to repository scope.  It is
fully deterministic: logs are normalized, failures are grouped by root-cause
fingerprint, and proposed edits are emitted for review without touching source.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

_TEMP = re.compile(r"(?:/tmp/tmp[^/\s:'\"]+|/private/var/folders/[^/\s:'\"]+)")
_LINE = re.compile(r"(?P<file>[A-Za-z0-9_.<>/-]+):(?P<line>\d+)")
_WS = re.compile(r"\s+")

@dataclass(frozen=True)
class ProjectFailure:
    fingerprint: str
    summary: str
    category: str
    stage: str | None
    occurrences: int
    tests: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class RepairPlan:
    fingerprint: str
    scope: str
    files: list[str]
    rationale: str
    patch: str = ""
    requires_review: bool = True


def normalize_log(text: str) -> str:
    """Remove volatile paths, line numbers and whitespace for stable grouping."""
    value = _TEMP.sub("<tmp>", text)
    value = _LINE.sub(lambda m: f"{m.group('file')}:<line>", value)
    return _WS.sub(" ", value).strip()


def fingerprint(text: str) -> str:
    normalized = normalize_log(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _classify(text: str) -> tuple[str, str | None, str]:
    low = text.lower()
    if "modulenotfounderror" in low or "importerror" in low:
        return "environment", "environment", "dependency/import failure"
    if "compilationerror" in low or "triton" in low and "error" in low:
        return "operator", "triton-frontend", "frontend compilation failure"
    if "linalg" in low or "dialect" in low or "buddy-opt" in low:
        return "compiler", "buddy-opt", "compiler lowering failure"
    if "cmake" in low or "ninja" in low or "linker" in low:
        return "build", "build", "build configuration failure"
    if "assert" in low or "mismatch" in low or "expected" in low and "actual" in low:
        return "runtime", "correctness", "numerical/runtime failure"
    return "unknown", None, "unclassified failure"


def analyze_failures(records: Iterable[Mapping[str, object]]) -> list[ProjectFailure]:
    """Group validation records by normalized root-cause fingerprint.

    Records may contain ``log``/``output``, ``test``/``target`` and
    ``implementation_file``/``source_file`` fields.  This loose contract makes
    the function usable with pytest, lit and build result JSONL files.
    """
    groups: dict[str, dict] = {}
    for record in records:
        raw = str(record.get("log") or record.get("output") or record.get("error") or "")
        if not raw:
            continue
        fp = fingerprint(raw)
        category, stage, summary = _classify(raw)
        item = groups.setdefault(fp, {"summary": summary, "category": category, "stage": stage,
                                      "occurrences": 0, "tests": set(), "source_files": set(), "evidence": []})
        item["occurrences"] += 1
        for key in ("test", "target", "name"):
            if record.get(key): item["tests"].add(str(record[key]))
        for key in ("implementation_file", "source_file", "file"):
            if record.get(key): item["source_files"].add(str(record[key]))
        excerpt = normalize_log(raw)[:500]
        if excerpt and excerpt not in item["evidence"]: item["evidence"].append(excerpt)
    return [ProjectFailure(fingerprint=k, tests=sorted(v.pop("tests")), source_files=sorted(v.pop("source_files")), evidence=v.pop("evidence"), **v) for k,v in sorted(groups.items())]


def plan_repairs(failures: Iterable[ProjectFailure], *, max_files: int = 3) -> list[RepairPlan]:
    """Create review-gated plans with conservative file scopes."""
    plans = []
    for failure in failures:
        files = failure.source_files[:max_files]
        low_risk = failure.category in {"operator", "runtime"} and len(files) <= 1
        scope = "operator-source" if low_risk else "project-review"
        plans.append(RepairPlan(failure.fingerprint, scope, files,
            f"{failure.summary}; affects {failure.occurrences} target(s): {', '.join(failure.tests[:5]) or 'unknown'}",
            requires_review=not low_risk))
    return plans


def write_report(path: Path, failures: Iterable[ProjectFailure], plans: Iterable[RepairPlan] = ()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "failures": [asdict(x) for x in failures], "repair_plans": [asdict(x) for x in plans]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
