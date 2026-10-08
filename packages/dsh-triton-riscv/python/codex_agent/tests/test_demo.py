from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.demo import development_cases, render_report


class DemoTests(unittest.TestCase):
    def test_selects_strongest_case_per_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, status, repairs in (
                ("run-failed", "repair-exhausted", 1),
                ("run-passed", "passed", 1),
            ):
                run_dir = root / name
                run_dir.mkdir()
                (run_dir / "final-result.json").write_text(
                    json.dumps(
                        {
                            "operator": "tanh_and_mul",
                            "status": status,
                            "repair_attempts": repairs,
                            "validations": [
                                {
                                    "iteration": 1,
                                    "status": "failed",
                                    "failure_stage": "mlir-translate",
                                },
                                {"iteration": 2, "status": "passed"}
                                if status == "passed"
                                else {"iteration": 2, "status": "failed"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            cases = development_cases(root)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["status"], "passed")
        self.assertEqual(cases[0]["initial_failure_stage"], "mlir-translate")

    def test_empty_report_does_not_claim_a_historical_success(self) -> None:
        result = {
            "generated_at": "2026-08-26T00:00:00+0800",
            "result_path": "demo-result.json",
            "agent_tests": {
                "tests": 1,
                "status": "passed",
                "log_path": "agent-tests.log",
            },
            "development_cases": [],
            "memory": {
                "stats": {"active": 0},
                "results": [],
                "database": "memory.sqlite3",
            },
            "project_discovery": {
                "total_targets": 0,
                "inventory_path": "targets.json",
                "coverage": {},
            },
        }

        report = render_report(result)

        self.assertIn("当前没有可用的历史算子开发证据", report)
        self.assertNotIn("`tanh_and_mul` 是目前最完整", report)


if __name__ == "__main__":
    unittest.main()
