from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_agent import project_tools as jobs


class ProjectToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        test = self.root / "python/examples/test_local.py"
        test.parent.mkdir(parents=True)
        test.write_text("def test_ok():\n    assert 1 == 1\n")
        self.test_file = test
        self.target = "pytest::python/examples/test_local.py"

    def test_inventory_pagination_and_strict_target_selection(self):
        result = jobs.inspect_project(self.root, kind="project", limit=1)
        self.assertEqual(result["items"][0]["id"], self.target)
        with self.assertRaises(ValueError):
            jobs.prepare_validation_job(self.root, ["echo unsafe"], kind="project", source_env=False)
        with self.assertRaises(ValueError):
            jobs.prepare_validation_job(self.root, [self.target] * 2, kind="project")

    def test_real_project_execution_is_approved_and_consumed_once(self):
        with patch.dict(os.environ, {"TRITON_RISCV_REQUIRE_REMOTE": "0", "TRITON_RISCV_ALLOW_VALIDATION": "1"}):
            job = jobs.prepare_validation_job(self.root, [self.target], kind="project", source_env=False)
            with self.assertRaises(PermissionError):
                jobs.execute_validation_job(self.root, job["job_id"])
            jobs.decide_validation_job(self.root, job["job_id"], approve=True, reviewer="native-harness:test")
            result = jobs.execute_validation_job(self.root, job["job_id"])
            self.assertEqual(result["status"], "passed", result)
            self.assertTrue(Path(result["results"][0]["log_path"]).is_file())
            with patch("codex_agent.project_tools.run_target") as runner:
                self.assertEqual(jobs.execute_validation_job(self.root, job["job_id"]), result)
                runner.assert_not_called()

    def test_source_change_is_rejected_before_execution(self):
        with patch.dict(os.environ, {"TRITON_RISCV_REQUIRE_REMOTE": "0", "TRITON_RISCV_ALLOW_VALIDATION": "1"}):
            job = jobs.prepare_validation_job(self.root, [self.target], kind="project", source_env=False)
            jobs.decide_validation_job(self.root, job["job_id"], approve=True, reviewer="native-harness:test")
            self.test_file.write_text("def test_bad():\n    assert False\n")
            with self.assertRaisesRegex(PermissionError, "source changed"):
                jobs.execute_validation_job(self.root, job["job_id"])

    def test_failed_project_is_not_reported_as_passed(self):
        self.test_file.write_text("def test_bad():\n    assert False\n")
        with patch.dict(os.environ, {"TRITON_RISCV_REQUIRE_REMOTE": "0", "TRITON_RISCV_ALLOW_VALIDATION": "1"}):
            job = jobs.prepare_validation_job(self.root, [self.target], kind="project", source_env=False)
            jobs.decide_validation_job(self.root, job["job_id"], approve=True, reviewer="native-harness:test")
            result = jobs.execute_validation_job(self.root, job["job_id"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["results"][0]["exit_code"], 1)

    def test_remote_only_environment_rejects_local_project(self):
        with patch.dict(os.environ, {"TRITON_RISCV_REQUIRE_REMOTE": "1"}):
            with self.assertRaises(PermissionError):
                jobs.prepare_validation_job(self.root, [self.target], kind="project", source_env=False)


if __name__ == "__main__":
    unittest.main()
