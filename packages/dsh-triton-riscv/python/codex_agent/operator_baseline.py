#!/usr/bin/env python3
"""Build and execute a deterministic operator validation baseline queue.

The queue intentionally records *plans* separately from validation receipts. A
 dry run can never become a pass; only an executed pytest command with exit code
 zero receives ``passed`` status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .summarize_operator_results import latest_by_operator, load_result_files, read_results

DEFAULT_INVENTORY = Path("agent-results/operators.json")
DEFAULT_RESULTS_DIR = Path("agent-results")
DEFAULT_QUEUE = Path("agent-results/operator-baseline-queue.json")


@dataclass
class BaselineItem:
    id: str
    operator: str
    visibility: str
    implementation_file: str
    command: str
    selection_reason: list[str] = field(default_factory=list)
    risk_hints: list[str] = field(default_factory=list)
    state: str = "pending"  # pending|planned|running|passed|failed|blocked|skipped
    attempts: int = 0
    last_exit_code: int | None = None
    last_duration_seconds: float | None = None
    last_log_path: str | None = None
    last_run_at: str | None = None


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")


def _item_id(operator: str) -> str:
    return "op-" + hashlib.sha1(operator.encode("utf-8")).hexdigest()[:12]


def inventory_fingerprint(inventory: dict) -> str:
    payload = json.dumps(inventory.get("operators", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _risk_bucket(operator: dict) -> str:
    ops = set(operator.get("tl_ops", []))
    hints = " ".join(operator.get("risk_hints", [])).lower()
    if "dot" in ops:
        return "dot"
    if ops & {"exp", "log", "sqrt", "rsqrt", "erf"}:
        return "math"
    if "where" in ops:
        return "mask"
    if ops & {"load", "store"}:
        return "memory"
    if "backward" in hints or operator.get("test_contract", {}).get("backward"):
        return "backward"
    return "general"


def _score(operator: dict) -> tuple[int, str]:
    """Stable priority score: public and broad/risky operators first."""
    score = 0
    if operator.get("visibility", "public") == "public":
        score += 20
    contract = operator.get("test_contract", {})
    score += min(int(contract.get("selected_test_count", 0)), 5)
    score += 4 if contract.get("pytorch_reference") else 0
    score += 3 if contract.get("numerical_assertion") else 0
    score += min(len(operator.get("risk_hints", [])), 3) * 2
    return score, operator.get("name", "")


def load_latest_results(results_dir: Path) -> dict[str, dict]:
    try:
        paths = load_result_files(results_dir, [])
        return {item.get("operator"): item for item in latest_by_operator(read_results(paths)) if item.get("operator")}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def select_representative_operators(
    inventory: dict,
    results: dict[str, dict] | None = None,
    *,
    limit: int = 12,
) -> list[dict]:
    """Select unvalidated operators with deterministic risk/visibility coverage."""
    results = results or {}
    candidates = []
    for op in inventory.get("operators", []):
        result = results.get(op.get("name"))
        if result and result.get("status") in {"passed", "failed", "skipped"}:
            continue
        if not op.get("validation_command"):
            continue
        candidates.append(op)
    candidates.sort(key=lambda item: (-_score(item)[0], _score(item)[1]))
    chosen: list[dict] = []
    risk_buckets: set[str] = set()
    visibilities: set[str] = set()
    # Greedily maximize diversity across both risk families and public/internal
    # visibility. Ties use the stable risk score and name ordering above.
    while candidates and len(chosen) < limit:
        ranked = sorted(
            candidates,
            key=lambda op: (
                -((1 if _risk_bucket(op) not in risk_buckets else 0) +
                  (1 if op.get("visibility", "public") not in visibilities else 0)),
                -_score(op)[0],
                _score(op)[1],
            ),
        )
        op = ranked[0]
        candidates.remove(op)
        chosen.append(op)
        risk_buckets.add(_risk_bucket(op))
        visibilities.add(op.get("visibility", "public"))
    return chosen


def build_queue(
    inventory: dict,
    results_dir: Path,
    *,
    limit: int = 12,
    existing: dict | None = None,
) -> dict:
    latest = load_latest_results(results_dir)
    existing_items = {item.get("operator"): item for item in (existing or {}).get("items", [])}
    selected = select_representative_operators(inventory, latest, limit=limit)
    items: list[dict] = []
    for op in selected:
        old = existing_items.get(op.get("name"), {})
        reasons = [f"risk-bucket:{_risk_bucket(op)}", f"visibility:{op.get('visibility', 'public')}"]
        if op.get("test_contract", {}).get("pytorch_reference"):
            reasons.append("has-independent-pytorch-reference")
        item = BaselineItem(
            id=old.get("id", _item_id(op["name"])),
            operator=op["name"],
            visibility=op.get("visibility", "public"),
            implementation_file=op.get("implementation_file", ""),
            command=op.get("validation_command", ""),
            selection_reason=old.get("selection_reason", reasons),
            risk_hints=op.get("risk_hints", []),
            state=old.get("state", "pending"),
            attempts=int(old.get("attempts", 0)),
            last_exit_code=old.get("last_exit_code"),
            last_duration_seconds=old.get("last_duration_seconds"),
            last_log_path=old.get("last_log_path"),
            last_run_at=old.get("last_run_at"),
        )
        # Preserve terminal evidence from an existing queue; do not overwrite it.
        items.append(asdict(item))
    return {
        "schema_version": 1,
        "queue_id": (existing or {}).get("queue_id") or f"baseline-{time.strftime('%Y%m%d-%H%M%S')}",
        "created_at": (existing or {}).get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inventory_fingerprint": inventory_fingerprint(inventory),
        "selection": {"limit": limit, "strategy": "risk-bucket-and-visibility", "reproducible": True},
        "items": items,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_baseline_markdown(queue: dict) -> str:
    """Render a reviewable queue report without implying unexecuted success."""
    lines = [
        "# Operator Validation Baseline Queue", "",
        f"- Queue: `{queue.get('queue_id', '')}`",
        f"- Inventory fingerprint: `{queue.get('inventory_fingerprint', '')}`",
        f"- Reproducible strategy: {queue.get('selection', {}).get('strategy', '')}", "",
        "| Operator | Visibility | State | Attempts | Command |", "| --- | --- | --- | ---: | --- |",
    ]
    for item in queue.get("items", []):
        command = str(item.get("command", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{item.get('operator', '')}` | {item.get('visibility', '')} | **{item.get('state', 'pending')}** | {item.get('attempts', 0)} | `{command}` |")
    lines += ["", "A `planned` item has not executed. Only a real subprocess exit code of 0 is recorded as `passed`."]
    return "\n".join(lines) + "\n"


def execute_queue(queue: dict, repo_root: Path, results_dir: Path, *, dry_run: bool = True, timeout_seconds: int = 900, checkpoint_path: Path | None = None) -> dict:
    """Execute pending queue items and persist per-item receipts.

    Results are append-only JSONL receipts. Queue state is updated after every
    item, so interrupted runs can be resumed without losing prior evidence.
    """
    receipt_path = results_dir / "operator-baseline-results.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    for item in queue.get("items", []):
        if item.get("state") in {"passed", "failed", "blocked", "skipped"}:
            continue
        command = item.get("command", "")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["last_run_at"] = now
        if not command:
            item["state"] = "blocked"
            item["blocked_reason"] = "no-validation-command"
            if checkpoint_path:
                _write_json(checkpoint_path, queue)
            continue
        if dry_run:
            item["state"] = "planned"
            item["last_exit_code"] = None
            if checkpoint_path:
                _write_json(checkpoint_path, queue)
            continue
        item["state"] = "running"
        start = time.monotonic()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = results_dir / "logs" / f"baseline-{timestamp}-{_slug(item['operator'])}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(["bash", "-lc", command], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout_seconds, check=False)
            output, code = completed.stdout, completed.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            output += f"\nTIMEOUT after {timeout_seconds} seconds\n"
            code = 124
        duration = round(time.monotonic() - start, 3)
        log_path.write_text(output, encoding="utf-8")
        item.update({"state": "passed" if code == 0 else "failed", "last_exit_code": code, "last_duration_seconds": duration, "last_log_path": log_path.as_posix()})
        receipt = {"schema_version": 1, "operator": item["operator"], "status": item["state"], "exit_code": code, "duration_seconds": duration, "log_path": log_path.as_posix(), "command": command, "execution_mode": "baseline", "recorded_at": now}
        with receipt_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        if checkpoint_path:
            _write_json(checkpoint_path, queue)
    return queue


def write_baseline_queue(inventory_path: Path, results_dir: Path, queue_path: Path, *, limit: int = 12, resume: bool = True, dry_run: bool = True, timeout_seconds: int = 900) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    existing = None
    if resume and queue_path.exists():
        try:
            existing = json.loads(queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
    queue = build_queue(inventory, results_dir, limit=limit, existing=existing)
    queue = execute_queue(queue, inventory_path.parent.parent if inventory_path.parent.name == "agent-results" else Path.cwd(), results_dir, dry_run=dry_run, timeout_seconds=timeout_seconds, checkpoint_path=queue_path)
    _write_json(queue_path, queue)
    markdown_path = queue_path.with_suffix(".md")
    markdown_path.write_text(render_baseline_markdown(queue), encoding="utf-8")
    return queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/resume a deterministic operator validation baseline queue.")
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY.as_posix())
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR.as_posix())
    parser.add_argument("--queue", default=DEFAULT_QUEUE.as_posix())
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run real pytest commands; default is dry-run planning.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.inventory).resolve().parent.parent
    queue = write_baseline_queue(Path(args.inventory), Path(args.results_dir), Path(args.queue), limit=max(0, args.limit), resume=not args.no_resume, dry_run=not args.execute, timeout_seconds=args.timeout_seconds)
    print(json.dumps({"queue_id": queue["queue_id"], "items": len(queue["items"]), "states": {state: sum(1 for item in queue["items"] if item.get("state") == state) for state in {item.get("state") for item in queue["items"]}}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
