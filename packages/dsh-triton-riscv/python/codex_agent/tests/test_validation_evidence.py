from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.platform.executor import HarnessRunExecutor
from codex_agent.validation_evidence import (
    RECEIPT_ROOT,
    audit_validation_receipt,
    capture_source_snapshot,
    file_sha256,
)


class ValidationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        operator_root = self.root / "python/examples/flaggems"
        operator_root.mkdir(parents=True)
        self.implementation = operator_root / "demo.py"
        self.test_file = operator_root / "test_demo.py"
        self.implementation.write_text("def demo(x):\n    return x\n", encoding="utf-8")
        self.test_file.write_text("def test_demo():\n    assert True\n", encoding="utf-8")
        self.log = self.root / "agent-results/operator-lifecycle/validation/logs/demo.log"
        self.log.parent.mkdir(parents=True)
        self.log.write_text("1 passed in 0.10s\n", encoding="utf-8")
        (self.root / RECEIPT_ROOT).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def source_snapshot(self) -> dict[str, str]:
        return capture_source_snapshot(
            self.root,
            "python/examples/flaggems/demo.py",
            ["python/examples/flaggems/test_demo.py"],
        )

    def write_receipt(
        self,
        *,
        run_id: str = "run-20260904-120000-abcdef12",
        status: str = "passed",
        exit_code: int | None = 0,
        dry_run: bool = False,
        execution_target: str = "local",
        approved_run_id: str | None = None,
    ) -> dict:
        receipt = {
            "run_id": run_id,
            "operator": "demo",
            "status": status,
            "exit_code": exit_code,
            "dry_run": dry_run,
            "command": "python -m pytest -q test_demo.py",
            "implementation_file": "python/examples/flaggems/demo.py",
            "test_files": ["python/examples/flaggems/test_demo.py"],
            "log_path": self.log.as_posix() if not dry_run else None,
            "log_sha256": file_sha256(self.log) if not dry_run else None,
            "source_snapshot": self.source_snapshot,
            "source_snapshot_stable": True,
            "execution_target": execution_target,
            "remote_preflight": None,
        }
        if approved_run_id:
            receipt["approved_run_id"] = approved_run_id
        path = self.root / RECEIPT_ROOT / f"{run_id}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return {
            "run_id": run_id,
            "operator": "demo",
            "status": status,
            "receipt_path": path.relative_to(self.root).as_posix(),
        }

    def test_accepts_live_pass_with_matching_artifacts(self) -> None:
        evidence = audit_validation_receipt(self.root, self.write_receipt())

        self.assertEqual(evidence.verdict, "verified-passed")
        self.assertTrue(evidence.trusted)
        self.assertTrue(evidence.success)

    def test_rejects_model_status_that_disagrees_with_receipt(self) -> None:
        reference = self.write_receipt(status="failed", exit_code=1)
        reference["status"] = "passed"

        evidence = audit_validation_receipt(self.root, reference)

        self.assertEqual(evidence.verdict, "invalid")
        self.assertIn("status does not match", " ".join(evidence.reasons))

    def test_rejects_source_changes_after_validation(self) -> None:
        reference = self.write_receipt()
        self.implementation.write_text("def demo(x):\n    return x + 1\n", encoding="utf-8")

        evidence = audit_validation_receipt(self.root, reference)

        self.assertEqual(evidence.verdict, "invalid")
        self.assertIn("changed after validation", " ".join(evidence.reasons))

    def test_rejects_tampered_log(self) -> None:
        reference = self.write_receipt()
        self.log.write_text("20 passed in 0.01s\n", encoding="utf-8")

        evidence = audit_validation_receipt(self.root, reference)

        self.assertEqual(evidence.verdict, "invalid")
        self.assertIn("log hash", " ".join(evidence.reasons))

    def test_planned_receipt_is_trusted_but_not_successful(self) -> None:
        evidence = audit_validation_receipt(
            self.root,
            self.write_receipt(
                status="planned",
                exit_code=None,
                dry_run=True,
            ),
        )

        self.assertEqual(evidence.verdict, "planned")
        self.assertTrue(evidence.trusted)
        self.assertFalse(evidence.success)

    def test_remote_pass_requires_approved_plan_link(self) -> None:
        reference = self.write_receipt(execution_target="remote")
        receipt_path = self.root / reference["receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["remote_preflight"] = {
            "status": "passed",
            "architecture": "riscv64",
        }
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        evidence = audit_validation_receipt(self.root, reference)

        self.assertEqual(evidence.verdict, "invalid")
        self.assertIn("no approved plan", " ".join(evidence.reasons))

    def test_executor_extracts_only_structured_receipt_references(self) -> None:
        reference = self.write_receipt()
        notification = {"payload": {"tool_result": json.dumps(reference)}}

        self.assertEqual(
            HarnessRunExecutor._validation_receipt_references(notification),
            [reference],
        )
        self.assertEqual(
            HarnessRunExecutor._validation_receipt_references(
                {"payload": {"text": "the model says all tests passed"}}
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
