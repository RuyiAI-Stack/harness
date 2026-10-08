from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from codex_agent.operator_agent import main, prune_old_logs, run_preflight, select_operators
from codex_agent.operator_status import build_status_report


class OperatorAgentTests(unittest.TestCase):
    def test_selects_only_unvalidated_public_operators(self) -> None:
        operators = [
            {"name": "_internal", "visibility": "internal"},
            {"name": "add", "visibility": "public"},
            {"name": "mul", "visibility": "public"},
        ]
        latest = {"add": {"status": "passed"}}

        selected = select_operators(
            operators,
            latest,
            names=[],
            visibility="public",
            selection="unvalidated",
            contains=None,
            offset=0,
            limit=20,
        )

        self.assertEqual([item["name"] for item in selected], ["mul"])

    def test_status_report_does_not_overclaim_correctness(self) -> None:
        inventory = {
            "operators": [
                {
                    "name": "add",
                    "visibility": "public",
                    "implementation_file": "add.py",
                    "test_files": ["test_add.py"],
                    "test_nodes": ["test_add.py::test_add"],
                    "validation_command": "pytest test_add.py",
                },
                {
                    "name": "mul",
                    "visibility": "public",
                    "implementation_file": "mul.py",
                    "test_files": ["test_mul.py"],
                    "test_nodes": ["test_mul.py::test_mul"],
                    "validation_command": "pytest test_mul.py",
                },
            ]
        }
        report = build_status_report(
            inventory,
            [{"operator": "add", "status": "passed"}],
        )

        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["public_validated"], 1)
        self.assertEqual(report["summary"]["unvalidated"], 1)
        self.assertEqual(report["operators"][0]["correctness"], "basic-tests-passed")
        self.assertEqual(report["operators"][1]["correctness"], "not-established")

    def test_log_retention_protects_latest_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            log_dir = results_dir / "logs"
            log_dir.mkdir()
            old_log = log_dir / "old.log"
            latest_log = log_dir / "latest.log"
            old_log.write_text("old", encoding="utf-8")
            latest_log.write_text("latest", encoding="utf-8")

            removed = prune_old_logs(
                results_dir,
                keep_logs=1,
                latest={"add": {"log_path": latest_log.as_posix()}},
            )

            self.assertEqual(removed, [old_log.resolve().as_posix()])
            self.assertFalse(old_log.exists())
            self.assertTrue(latest_log.exists())

    def test_preflight_dry_run_and_failure(self) -> None:
        self.assertEqual(run_preflight(Path("."), True, True)["reason"], "dry-run")

        completed = type("Completed", (), {"returncode": 1, "stdout": "missing tool"})()
        with patch("codex_agent.operator_agent.subprocess.run", return_value=completed):
            result = run_preflight(Path("."), False, False)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["output"], "missing tool")

    def test_agent_main_dry_run_discovers_and_reports_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            operator_dir = root / "python/examples/flaggems"
            operator_dir.mkdir(parents=True)
            (operator_dir / "add.py").write_text(
                "import triton\n@triton.jit\ndef add_kernel():\n    pass\n",
                encoding="utf-8",
            )
            (operator_dir / "test_add.py").write_text(
                "from .add import add_kernel\ndef test_add():\n    pass\n",
                encoding="utf-8",
            )
            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "operator-agent",
                    "--repo-root",
                    temp_dir,
                    "--results-dir",
                    "results",
                    "--operator",
                    "add",
                    "--dry-run",
                ],
            ), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            run_report = json.loads((root / "results/operator-agent-last-run.json").read_text())
            self.assertEqual(run_report["selected_operators"], ["add"])
            self.assertEqual(run_report["attempts_written"], 1)
            self.assertEqual(run_report["preflight"]["status"], "skipped")
            self.assertTrue((root / "results/operators.json").exists())
            self.assertTrue((root / "results/operator-status.md").exists())

    def test_select_operator_reports_unknown_explicit_name(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unknown operators"):
            select_operators(
                [{"name": "add"}],
                {},
                names=["missing"],
                visibility="all",
                selection="all",
                contains=None,
                offset=0,
                limit=None,
            )


if __name__ == "__main__":
    unittest.main()
