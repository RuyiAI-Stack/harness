from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codex_agent.operator_baseline import (
    build_queue,
    execute_queue,
    render_baseline_markdown,
    select_representative_operators,
)


class OperatorBaselineTests(unittest.TestCase):
    def inventory(self):
        def op(name, visibility, tl_ops, command="printf ok"):
            return {"name": name, "visibility": visibility, "implementation_file": f"{name}.py", "validation_command": command, "tl_ops": tl_ops, "risk_hints": [], "test_contract": {"selected_test_count": 1, "pytorch_reference": True}}
        return {"operators": [op("exp_op", "public", ["exp"]), op("dot_op", "public", ["dot"]), op("mem_op", "internal", ["load", "store"]), op("plain", "public", ["add"])]}

    def test_selection_covers_buckets_deterministically(self):
        selected = select_representative_operators(self.inventory(), limit=3)
        self.assertEqual({item["name"] for item in selected}, {"dot_op", "exp_op", "mem_op"})

    def test_dry_run_never_records_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue = build_queue(self.inventory(), root, limit=2)
            result = execute_queue(queue, root, root / "results", dry_run=True)
            self.assertTrue(all(item["state"] == "planned" for item in result["items"]))
            self.assertFalse((root / "results" / "operator-baseline-results.jsonl").exists())

    def test_live_execution_records_receipt_and_resume_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue = build_queue(self.inventory(), root, limit=1)
            result = execute_queue(queue, root, root / "results", dry_run=False)
            self.assertEqual(result["items"][0]["state"], "passed")
            receipt = json.loads((root / "results" / "operator-baseline-results.jsonl").read_text().splitlines()[0])
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["execution_mode"], "baseline")

    def test_markdown_labels_planned_state(self):
        queue = build_queue(self.inventory(), Path("."), limit=1)
        text = render_baseline_markdown(queue)
        self.assertIn("planned", text)
        self.assertIn("Only a real subprocess exit code of 0", text)


if __name__ == "__main__":
    unittest.main()
