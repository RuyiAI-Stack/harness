from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.operator_status import (
    build_status_report,
    operator_status,
    render_status_markdown,
    write_status_reports,
)


class OperatorStatusTests(unittest.TestCase):
    def test_maps_result_and_inventory_states(self) -> None:
        operator = {"validation_command": "pytest test.py"}
        self.assertEqual(operator_status(operator, {"status": "passed"}), ("passed", "basic-tests-passed"))
        self.assertEqual(operator_status(operator, {"status": "failed"}), ("failed", "basic-tests-failed"))
        self.assertEqual(operator_status(operator, {"status": "skipped"}), ("skipped", "not-runnable"))
        self.assertEqual(operator_status(operator, {"status": "planned"}), ("planned", "not-established"))
        self.assertEqual(operator_status(operator, None), ("unvalidated", "not-established"))
        self.assertEqual(operator_status({"validation_command": ""}, None), ("unverified", "not-established"))

    def test_writes_json_and_markdown_from_latest_results(self) -> None:
        inventory = {
            "operators": [
                {
                    "name": "add",
                    "visibility": "public",
                    "implementation_file": "add.py",
                    "test_nodes": ["test_add.py::test_add"],
                    "validation_command": "pytest test_add.py",
                },
                {
                    "name": "_internal",
                    "implementation_file": "internal.py",
                    "validation_command": "",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory_path = root / "operators.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            (root / "operator-validation-one.jsonl").write_text(
                json.dumps({"operator": "add", "status": "failed", "failure_stage": "pytest"}) + "\n"
                + json.dumps({"operator": "add", "status": "passed"}) + "\n",
                encoding="utf-8",
            )
            json_output = root / "status.json"
            markdown_output = root / "status.md"
            report = write_status_reports(
                inventory_path,
                root,
                json_output,
                markdown_output,
            )

            self.assertEqual(report["summary"]["operators"], 2)
            self.assertEqual(report["summary"]["validated"], 1)
            self.assertEqual(report["summary"]["validation_percent"], 50.0)
            self.assertEqual(report["summary"]["unverified"], 1)
            self.assertEqual(json.loads(json_output.read_text())["summary"]["passed"], 1)
            markdown = markdown_output.read_text()
            self.assertIn("basic tests passed", markdown)
            self.assertIn("| add | public | 1 | passed |", markdown)

    def test_markdown_escapes_table_values(self) -> None:
        report = build_status_report(
            {
                "operators": [
                    {
                        "name": "pipe|operator",
                        "visibility": "public",
                        "implementation_file": "op.py",
                        "validation_command": "pytest",
                    }
                ]
            },
            [{"operator": "pipe|operator", "status": "failed", "likely_reason": "a|b"}],
        )
        markdown = render_status_markdown(report)
        self.assertIn("pipe\\|operator", markdown)
        self.assertIn("a\\|b", markdown)


if __name__ == "__main__":
    unittest.main()
