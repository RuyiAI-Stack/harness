from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from codex_agent.run_validation import (
    build_shell_command,
    load_targets,
    main,
    run_target,
    slugify,
    target_matches,
)


class RunValidationTests(unittest.TestCase):
    def test_loads_filters_and_formats_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "targets.json"
            path.write_text(
                json.dumps(
                    {
                        "pytest_targets": [{"kind": "pytest", "path": "python/test_add.py"}],
                        "lit_targets": [{"kind": "lit", "path": "test/add.mlir"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(load_targets(path)), 2)
        self.assertEqual(slugify("test/a.py::test x"), "test__a_py__test_x")
        self.assertEqual(
            build_shell_command("pytest test.py", True),
            "source scripts/triton-riscv-env.sh && pytest test.py",
        )
        self.assertTrue(
            target_matches(
                {"kind": "pytest", "likely_area": "math", "path": "test_exp.py"},
                kind="pytest",
                area="math",
                path_contains="exp",
            )
        )
        self.assertFalse(
            target_matches(
                {"kind": "lit", "likely_area": "general", "path": "test.mlir"},
                kind="pytest",
                area=None,
                path_contains=None,
            )
        )

    def test_run_target_records_success_and_dry_run(self) -> None:
        target = {
            "kind": "pytest",
            "path": "python/examples/test_smoke.py",
            "command": "printf validation-ok",
            "likely_area": "general",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = run_target(
                target,
                repo_root=root,
                results_dir=root / "results",
                source_env=False,
                dry_run=False,
                timeout_seconds=5,
            )
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(Path(result.log_path).read_text(), "validation-ok")
            planned = run_target(
                target,
                repo_root=root,
                results_dir=root / "results",
                source_env=True,
                dry_run=True,
                timeout_seconds=5,
            )
            self.assertIsNone(planned.exit_code)
            self.assertIsNone(planned.log_path)
            self.assertTrue(planned.command.startswith("source scripts/triton-riscv-env.sh"))

    def test_run_target_converts_timeout_to_result_and_log(self) -> None:
        target = {
            "kind": "lit",
            "path": "test/slow.mlir",
            "command": "slow-command",
            "likely_area": "general",
        }
        timeout = subprocess.TimeoutExpired(
            cmd="slow-command",
            timeout=1,
            output=b"partial output",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "codex_agent.run_validation.run_bounded",
            side_effect=timeout,
        ):
            root = Path(temp_dir)
            result = run_target(
                target,
                repo_root=root,
                results_dir=root / "results",
                source_env=False,
                dry_run=False,
                timeout_seconds=1,
            )
            self.assertEqual(result.exit_code, 124)
            log = Path(result.log_path).read_text()
            self.assertIn("partial output", log)
            self.assertIn("TIMEOUT after 1 seconds", log)

    def test_cli_filters_targets_and_writes_result_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as output_dir:
            repo_root = Path(repo_dir)
            targets_path = repo_root / "targets.json"
            targets_path.write_text(
                json.dumps(
                    {
                        "pytest_targets": [
                            {
                                "kind": "pytest",
                                "path": "python/test_add.py",
                                "command": "printf ok",
                                "likely_area": "general",
                            },
                            {
                                "kind": "pytest",
                                "path": "python/test_mul.py",
                                "command": "printf ok",
                                "likely_area": "general",
                            },
                        ],
                        "lit_targets": [],
                    }
                ),
                encoding="utf-8",
            )
            stdout = StringIO()
            with patch(
                "sys.argv",
                [
                    "run-validation",
                    "--repo-root",
                    repo_dir,
                    "--targets",
                    targets_path.as_posix(),
                    "--results-dir",
                    output_dir,
                    "--path-contains",
                    "add",
                    "--dry-run",
                ],
            ), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            result_files = list(Path(output_dir).glob("validation-*.jsonl"))
            self.assertEqual(len(result_files), 1)
            records = result_files[0].read_text().splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["path"], "python/test_add.py")
            self.assertIn(result_files[0].as_posix(), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
