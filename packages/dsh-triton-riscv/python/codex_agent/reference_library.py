"""Offline, evidence-gated repository references; never import collected code.

This catalog is separate from production memory. Static matches are candidates,
not verified implementations. Only a source-bound executed receipt can promote
a candidate. Even promoted records are references, not general repair rules.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess

from codex_agent.memory import SECRET_PATTERNS, jaccard, tokens
from codex_agent.validation_evidence import RECEIPT_ROOT, audit_validation_receipt

SCHEMA = "repository-reference-v1"
MAX_BYTES = 4 * 1024 * 1024
MAX_RECEIPTS = 10000
ENV_FIELDS = ("architecture", "execution_mode", "triton", "llvm", "buddy")
GUARD = re.compile(
    r"ignore (?:all |the )?(?:previous|prior) instructions|"
    r"disable (?:the )?(?:safety|approval)|忽略.{0,8}(?:之前|以上|系统).{0,8}指令",
    re.I,
)
UNSAFE_CALLS = {"eval", "exec", "setattr", "delattr", "monkeypatch.setattr",
                "mock.patch", "unittest.mock.patch"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_read(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("absolute-source-path")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("source-path-escape")
    if not target.is_file() or target.stat().st_size > MAX_BYTES:
        raise ValueError("missing-or-oversize-source")
    data = target.read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError("oversize-source")
    return data


def symbol(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return symbol(node.value) + "." + node.attr
    if isinstance(node, ast.Call):
        return symbol(node.func)
    return ""


def inspect_python(raw: bytes, *, test: bool) -> tuple[dict, list[str]]:
    text = raw.decode("utf-8")
    tree = ast.parse(text)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = node.module + "." + alias.name

    def canonical(node):
        value = symbol(node)
        first, dot, rest = value.partition(".")
        return aliases.get(first, first) + (dot + rest if dot else "")

    reasons = []
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        reasons.append("possible-secret")
    if GUARD.search(text):
        reasons.append("instruction-like-source-content")
    imports = []
    names = []
    kernels = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
            if any(canonical(d) == "triton.jit" for d in node.decorator_list):
                kernels.append(node.name)
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.Call):
            name = canonical(node.func)
            calls.append(name)
            if name in UNSAFE_CALLS or name.startswith(("unittest.mock.patch.", "mock.patch.")):
                reasons.append("dynamic-code-or-reference-patching")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = getattr(node, "targets", [getattr(node, "target", None)])
            if any(canonical(t).startswith(("torch.", "numpy.")) for t in targets if t):
                reasons.append("reference-library-mutation")
    if test:
        if not any(name.startswith("test_") for name in names):
            reasons.append("no-test-function")
        if not any(name in {"torch.testing.assert_close", "torch.allclose",
                            "np.testing.assert_allclose", "numpy.testing.assert_allclose"}
                   for name in calls):
            reasons.append("no-recognized-numeric-comparison")
    elif not kernels:
        reasons.append("no-recognized-triton-kernel")
    return {"functions": sorted(set(names)), "kernels": sorted(set(kernels)),
            "imports": sorted(set(imports)),
            "calls": sorted(set(calls)),
            "tl_ops": sorted({c.removeprefix("triton.language.") for c in calls
                              if c.startswith("triton.language.")})}, sorted(set(reasons))


def receipt_index(root: Path) -> tuple[dict, list[dict]]:
    index: dict[str, list[tuple[Path, dict]]] = {}
    rejected = []
    paths = sorted((root / RECEIPT_ROOT).glob("*.json"))
    if len(paths) > MAX_RECEIPTS:
        raise ValueError("receipt-count-exceeds-scan-budget")
    for path in paths:
        try:
            item = json.loads(safe_read(root, path.relative_to(root).as_posix()))
            if not isinstance(item, dict) or not isinstance(item.get("operator"), str):
                raise ValueError("invalid-receipt-object")
            index.setdefault(item["operator"], []).append((path, item))
        except (OSError, ValueError, UnicodeError):
            rejected.append({"path": path.relative_to(root).as_posix(), "reason": "unreadable-receipt"})
    return index, rejected


def bind_artifacts(root: Path, paths: list[str]) -> tuple[list[dict], list[str]]:
    """Add content hashes, but never reinterpret a newly computed hash as historic proof."""
    artifacts = []
    reasons = []
    for value in paths:
        try:
            absolute = Path(value) if Path(value).is_absolute() else root / value
            relative = absolute.resolve().relative_to(root).as_posix()
            raw = safe_read(root, relative)
            artifacts.append({"path": relative, "sha256": digest(raw)})
        except (OSError, ValueError, TypeError):
            reasons.append("unbound-evidence-artifact")
    return artifacts, reasons


def check_receipt(root: Path, path: Path, receipt: dict,
                  implementation: str, tests: str) -> tuple[dict | None, list[str]]:
    reasons = []
    if receipt.get("implementation_file") != implementation or receipt.get("test_files") != [tests]:
        reasons.append("receipt-source-or-tests-do-not-match-pair")
    reference = {"run_id": receipt.get("run_id"), "operator": receipt.get("operator"),
                 "status": receipt.get("status")}
    try:
        audited = audit_validation_receipt(root, reference)
        if not audited.success:
            for reason in audited.reasons or ["receipt-is-not-verified-passed"]:
                if reason == "source changed while validation was running" and receipt.get("source_snapshot_stable") is None:
                    reason = "missing-source-stability-evidence"
                reasons.append(reason)
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        reasons.append("receipt-audit-error")
    if receipt.get("status") != "passed" or receipt.get("dry_run") is not False:
        reasons.append("not-an-executed-pass")
    try:
        command = shlex.split(receipt.get("command") or "")
    except ValueError:
        command = []
    if not any(c == "pytest" or c.endswith("/pytest") for c in command):
        reasons.append("no-pytest-command")
    selected = [c.split("::", 1)[0] for c in command]
    if tests not in selected:
        reasons.append("test-command-does-not-select-recorded-file")
    command_text = receipt.get("command") or ""
    if GUARD.search(command_text) or any(p.search(command_text) for p in SECRET_PATTERNS):
        reasons.append("unsafe-command-content")
    remote = receipt.get("remote_preflight") or {}
    if not isinstance(remote, dict):
        remote = {}
    architecture = receipt.get("architecture") or remote.get("architecture")
    triton = receipt.get("triton_version") or remote.get("triton_version")
    if architecture != "riscv64" or not triton:
        reasons.append("missing-riscv-environment-binding")
    for key in ("architecture", "triton_version"):
        if receipt.get(key) and remote.get(key) and receipt[key] != remote[key]:
            reasons.append("conflicting-environment")
    try:
        log_path = Path(receipt.get("log_path") or "")
        relative = (log_path if log_path.is_absolute() else root / log_path).resolve().relative_to(root)
        log = safe_read(root, relative.as_posix()).decode("utf-8")
        if not re.search(r"\b[1-9]\d* passed\b", log):
            reasons.append("no-positive-pytest-count")
        if re.search(r"\b[1-9]\d* (?:failed|errors?|xpassed)\b|^FAILED\b|^ERROR\b|Traceback", log, re.M):
            reasons.append("log-contradicts-success" if receipt.get("status") == "passed"
                           else "log-has-failure-evidence")
        if GUARD.search(log) or any(p.search(log) for p in SECRET_PATTERNS):
            reasons.append("unsafe-log-content")
    except (OSError, ValueError, UnicodeError):
        reasons.append("unreadable-validation-log")
    artifact_paths = [path.relative_to(root).as_posix(), receipt.get("log_path") or ""]
    if receipt.get("execution_target") == "remote":
        artifact_paths.append((RECEIPT_ROOT / f"{receipt.get('approved_run_id')}.json").as_posix())
    artifacts, binding_errors = bind_artifacts(root, artifact_paths)
    reasons.extend(binding_errors)
    if reasons:
        return None, sorted(set(reasons))
    return {"run_id": receipt["run_id"], "receipt_path": path.relative_to(root).as_posix(),
            "command": receipt["command"], "artifacts": artifacts,
            "environment": {"architecture": architecture, "triton": triton,
                            "execution_mode": receipt.get("execution_mode")},
            "claim": "recorded tests passed for these exact files; not a verified root cause or general rule"}, []


def candidate(root: Path, implementation: Path, receipts: list[tuple[Path, dict]]) -> dict:
    name = implementation.stem
    relative = implementation.relative_to(root).as_posix()
    tests = implementation.with_name(f"test_{name}.py").relative_to(root).as_posix()
    record = {"schema": SCHEMA, "kind": "repository-reference", "operator": name,
              "source_root": root.as_posix(), "implementation": relative, "tests": tests,
              "decision": "quarantined", "validation": None,
              "authority": "reference-data-not-instructions", "artifacts": [],
              "reasons": [], "receipt_checks": []}
    reasons = record["reasons"]
    for key, path in (("implementation", relative), ("tests", tests)):
        try:
            raw = safe_read(root, path)
            record["artifacts"].append({"path": path, "sha256": digest(raw)})
            info, problems = inspect_python(raw, test=key == "tests")
            record[key + "_facts"] = info
            reasons.extend(problems)
        except (OSError, ValueError, UnicodeError, SyntaxError, RecursionError):
            reasons.append("missing-invalid-or-unsafe-" + key)
    tests_info = record.get("tests_facts", {})
    if not any(module.split(".")[-1] == name for module in tests_info.get("imports", [])):
        reasons.append("test-does-not-explicitly-import-implementation")
    functions = set(record.get("implementation_facts", {}).get("functions", []))
    if not any(call in functions or call.startswith(name + ".") for call in tests_info.get("calls", [])):
        reasons.append("no-recognized-call-to-implementation")
    # Static checks do not establish numerical correctness or prove benign code.
    valid = []
    conflicting_runs = []
    for path, receipt in receipts:
        verified, failures = check_receipt(root, path, receipt, relative, tests)
        record["receipt_checks"].append({"path": path.relative_to(root).as_posix(),
                                         "eligible": verified is not None, "reasons": failures})
        if verified:
            valid.append(verified)
        elif (receipt.get("status") == "failed" and receipt.get("implementation_file") == relative
              and receipt.get("test_files") == [tests]):
            try:
                failed = audit_validation_receipt(root, {"run_id": receipt.get("run_id"), "operator": name})
                if failed.verdict == "verified-failed":
                    conflicting_runs.append(receipt["run_id"])
            except (OSError, ValueError, TypeError, AttributeError, KeyError):
                pass
    if not valid:
        reasons.append("no-source-bound-passing-validation")
    if valid and conflicting_runs:
        reasons.append("same-source-has-conflicting-execution-results")
    if not reasons:
        environments = {json.dumps(v["environment"], sort_keys=True) for v in valid}
        if len(environments) > 1:
            reasons.append("multiple-environments-require-separate-reference-versions")
        else:
            record["validation"] = valid[0]
            record["decision"] = "admitted"
    if record["decision"] == "admitted":
        try:
            if any(digest(safe_read(root, a["path"])) != a["sha256"]
                   for a in [*record["artifacts"], *record["validation"]["artifacts"]]):
                raise ValueError("source-changed")
        except (OSError, ValueError):
            record["decision"] = "quarantined"
            reasons.append("evidence-changed-during-collection")
    record["reasons"] = sorted(set(reasons))
    identity = {k: record[k] for k in ("kind", "operator", "artifacts")}
    record["reference_id"] = digest(json.dumps(identity, sort_keys=True).encode())
    return record


def git_revision(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def build_library(root: Path, output: Path, *, excluded_operators: set[str] | None = None,
                  provenance: str = "unknown") -> dict:
    """Inventory without executing source/tests or touching any existing database."""
    root, output = root.resolve(), output.resolve()
    if provenance not in {"real", "controlled-demo", "synthetic", "unknown"}:
        raise ValueError("invalid-provenance")
    if not (root / "python/examples/flaggems").is_dir():
        raise ValueError("operator-directory-not-found")
    if output.exists():
        raise ValueError("output-already-exists")
    excluded = excluded_operators or set()
    receipts, rejected_receipts = receipt_index(root)
    records, seen = [], set()
    for path in sorted((root / "python/examples/flaggems").glob("*.py")):
        if path.name.startswith(("test_", "__")) or path.stem in {"conftest", "utils"}:
            continue
        record = candidate(root, path, receipts.get(path.stem, []))
        record["provenance"] = provenance
        if path.stem in excluded:
            record["decision"] = "quarantined"
            record["reasons"].append("excluded-evaluation-target")
        if provenance != "real":
            record["decision"] = "quarantined"
            record["reasons"].append("not-declared-real-source")
        if record["reference_id"] in seen:
            continue
        seen.add(record["reference_id"])
        records.append(record)
    output.mkdir(parents=True)
    database = output / "references.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE references_catalog (id TEXT PRIMARY KEY, admitted INTEGER NOT NULL, payload TEXT NOT NULL)")
        connection.executemany("INSERT INTO references_catalog VALUES (?, ?, ?)",
                               [(r["reference_id"], r["decision"] == "admitted", json.dumps(r, ensure_ascii=False)) for r in records])
    for decision, filename in (("admitted", "admitted.jsonl"), ("quarantined", "quarantine.jsonl")):
        with (output / filename).open("x", encoding="utf-8") as stream:
            for record in records:
                if record["decision"] == decision:
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    reasons = Counter(reason for r in records for reason in set(r["reasons"]))
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_root": root.as_posix(), "git_head": git_revision(root),
                "provenance": provenance, "provenance_basis": "caller-declared-not-independently-certified",
                "candidates": len(records), "admitted": sum(r["decision"] == "admitted" for r in records),
                "quarantined": sum(r["decision"] == "quarantined" for r in records),
                "quarantine_reasons": dict(sorted(reasons.items())),
                "rejected_receipts": rejected_receipts, "excluded_operators": sorted(excluded),
                "product_database_modified": False, "model_or_remote_calls": 0,
                "limitations": ["heuristic static checks are not a sandbox or proof of correctness",
                                "no arbitrary docs or PR claims promoted to verified rules",
                                "historical validation may not match current source hashes",
                                "not connected to production retrieval by default"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def search_library(library: Path, query: str, *, environment: dict,
                   excluded_operators: set[str] | None = None, limit: int = 5) -> dict:
    """Read-only pilot retrieval; never bypass quarantine or stale source checks."""
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    if environment.get("architecture") != "riscv64" or not environment.get("triton"):
        raise ValueError("explicit riscv64 architecture and Triton version required")
    database = library.resolve() / "references.sqlite3"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute("SELECT payload FROM references_catalog WHERE admitted=1 ORDER BY id").fetchall()
    finally:
        connection.close()
    items, rejected = [], []
    for (payload,) in rows:
        record = json.loads(payload)
        if (record.get("schema") != SCHEMA or record.get("decision") != "admitted"
                or record.get("provenance") != "real" or not record.get("validation")):
            rejected.append({"id": record.get("reference_id"), "reason": "invalid-admission-metadata"})
            continue
        if record["operator"] in (excluded_operators or set()):
            continue
        validation = record["validation"]
        known = validation["environment"]
        if any(environment.get(k) != known.get(k) for k in ENV_FIELDS
               if environment.get(k) or known.get(k)):
            rejected.append({"id": record["reference_id"], "reason": "incompatible-or-unknown-environment"})
            continue
        root = Path(record["source_root"])
        artifacts = [*record["artifacts"], *validation["artifacts"]]
        try:
            if any(digest(safe_read(root, a["path"])) != a["sha256"] for a in artifacts):
                raise ValueError("hash-mismatch")
        except (OSError, ValueError):
            rejected.append({"id": record["reference_id"], "reason": "source-or-evidence-changed"})
            continue
        facts = record.get("implementation_facts", {})
        score = jaccard(tokens(query), tokens(" ".join([record["operator"], *facts.get("functions", []), *facts.get("tl_ops", [])])))
        if score > 0:
            items.append({"score": score, **record})
    items.sort(key=lambda r: (-r["score"], r["reference_id"]))
    return {"items": items[:limit], "rejected": rejected,
            "scope": "offline reference pilot, not production memory ranking",
            "warning": "Use only under the recorded contract and environment; passing these tests is not general correctness."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--provenance", choices=["real", "controlled-demo", "synthetic", "unknown"], default="unknown")
    build.add_argument("--exclude-operator", action="append", default=[])
    search = commands.add_parser("search")
    search.add_argument("--library", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--architecture", default="riscv64")
    search.add_argument("--triton", required=True)
    search.add_argument("--execution-mode")
    search.add_argument("--exclude-operator", action="append", default=[])
    args = parser.parse_args()
    if args.command == "build":
        result = build_library(args.repo_root, args.output, provenance=args.provenance,
                               excluded_operators=set(args.exclude_operator))
    else:
        result = search_library(args.library, args.query,
                                 environment={"architecture": args.architecture, "triton": args.triton,
                                              "execution_mode": args.execution_mode},
                                 excluded_operators=set(args.exclude_operator))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
