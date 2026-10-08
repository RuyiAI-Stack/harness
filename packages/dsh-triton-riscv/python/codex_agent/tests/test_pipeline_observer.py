from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent.pipeline_observer import (
    build_pipeline_report,
    collect_artifacts,
    collect_environment,
    load_stage_events,
    render_pipeline_markdown,
)


class PipelineObserverTests(unittest.TestCase):
    def test_collects_and_classifies_pipeline_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_dir = Path(temp_dir)
            for name in ("tt.mlir", "ttshared.mlir", "ll.mlir", "ll.ir", "kernel.o"):
                (dump_dir / name).write_bytes(name.encode("utf-8"))

            artifacts = collect_artifacts(dump_dir)

            self.assertEqual(len(artifacts), 5)
            self.assertEqual(
                {item["stage"] for item in artifacts},
                {"ttir", "linalg-mlir", "llvm-mlir", "llvm-ir", "riscv-object"},
            )
            self.assertTrue(all(len(item["sha256"]) == 64 for item in artifacts))

    def test_identifies_riscv_elf_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_dir = Path(temp_dir)
            header = bytearray(64)
            header[:4] = b"\x7fELF"
            header[4] = 2
            header[5] = 1
            header[18:20] = (243).to_bytes(2, "little")
            (dump_dir / "kernel.o").write_bytes(header)

            artifact = collect_artifacts(dump_dir)[0]

        self.assertEqual(artifact["machine"], "RISC-V")
        self.assertTrue(artifact["is_riscv"])

    def test_reports_first_failed_stage_and_stops_later_stages(self) -> None:
        report = build_pipeline_report(
            operator={"name": "gelu", "test_contract": {"numerical_assertion": True}},
            validation_status="failed",
            failure_stage="mlir-translate",
            likely_reason="unsupported linalg operation",
            exit_code=1,
            artifacts=[
                {"path": "tt.mlir", "stage": "ttir"},
                {"path": "ttshared.mlir", "stage": "linalg-mlir"},
                {"path": "ll.mlir", "stage": "llvm-mlir"},
            ],
            stage_events=[],
            environment={"execution": {"mode": "native-riscv"}},
        )

        statuses = {item["id"]: item["status"] for item in report["stages"]}
        self.assertEqual(report["first_failure_stage"], "llvm-ir")
        self.assertEqual(statuses["llvm-mlir"], "passed")
        self.assertEqual(statuses["llvm-ir"], "failed")
        self.assertEqual(statuses["runtime"], "not-run")

    def test_success_requires_numerical_assertion_for_correctness_claim(self) -> None:
        without_assertion = build_pipeline_report(
            operator={"name": "smoke", "test_contract": {}},
            validation_status="passed",
            failure_stage=None,
            likely_reason=None,
            exit_code=0,
            artifacts=[],
            stage_events=[],
            environment={"execution": {"mode": "native-riscv"}},
        )
        with_assertion = build_pipeline_report(
            operator={"name": "add", "test_contract": {"numerical_assertion": True}},
            validation_status="passed",
            failure_stage=None,
            likely_reason=None,
            exit_code=0,
            artifacts=[],
            stage_events=[],
            environment={"execution": {"mode": "native-riscv"}},
        )

        self.assertEqual(without_assertion["correctness"], "not-established")
        self.assertEqual(with_assertion["correctness"], "basic-tests-passed")
        self.assertIn("passed-cached-or-inferred", render_pipeline_markdown(with_assertion))

    def test_loads_stage_events_and_records_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dump_dir = Path(temp_dir)
            (dump_dir / "stage-events.jsonl").write_text(
                '{"stage":"triton-shared-opt","command":["opt","in.mlir"],'
                '"duration_seconds":1.25,"exit_code":0,"status":"passed"}\n',
                encoding="utf-8",
            )
            events = load_stage_events(dump_dir)

        self.assertEqual(events[0]["pipeline_stage"], "linalg-mlir")
        self.assertEqual(events[0]["command"], ["opt", "in.mlir"])

    def test_environment_snapshot_distinguishes_native_riscv(self) -> None:
        def fake_probe(_root, command, **_kwargs):
            if command == "uname -m":
                output = "riscv64"
            elif command == "uname -srm":
                output = "Linux 6.0 riscv64"
            elif "platform.python_version" in command:
                output = "3.11.6\n/venv/bin/python"
            elif "import triton" in command:
                output = "3.4.0"
            elif command.startswith("command -v"):
                output = "/tools/" + command.split()[-1].strip('"').split("}")[-1]
            else:
                output = "LLVM version 22.0.0git"
            return {"command": command, "exit_code": 0, "output": output}

        with patch("codex_agent.pipeline_observer._probe", side_effect=fake_probe):
            report = collect_environment(Path("."), source_env=True)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["execution"]["mode"], "native-riscv")
        self.assertTrue(report["execution"]["hardware_execution_established"])

    def test_non_riscv_success_does_not_claim_hardware_validation(self) -> None:
        report = build_pipeline_report(
            operator={"name": "add", "test_contract": {"numerical_assertion": True}},
            validation_status="passed",
            failure_stage=None,
            likely_reason=None,
            exit_code=0,
            artifacts=[],
            stage_events=[],
            environment={
                "architecture": "arm64",
                "execution": {
                    "mode": "non-riscv-host",
                    "hardware_execution_established": False,
                },
            },
        )

        stages = {item["id"]: item["status"] for item in report["stages"]}
        self.assertEqual(stages["hardware-capability"], "not-established")

    def test_markdown_renders_structured_failure_evidence(self) -> None:
        report = build_pipeline_report(
            operator={"name": "gelu", "test_contract": {"numerical_assertion": True}},
            validation_status="failed",
            failure_stage="mlir-translate",
            likely_reason="unlowered linalg operation",
            exit_code=1,
            artifacts=[],
            diagnosis={
                "status": "failed",
                "rule_id": "unlowered-linalg-translation",
                "category": "compiler-capability",
                "confidence": 0.99,
                "repair_scope": "compiler",
                "failed_command": "mlir-translate --mlir-to-llvmir ll.mlir",
                "evidence": [
                    {"line_number": 42, "text": "error: linalg.generic remained"}
                ],
                "recommended_actions": ["Inspect ll.mlir."],
            },
        )

        markdown = render_pipeline_markdown(report)
        self.assertIn("unlowered-linalg-translation", markdown)
        self.assertIn("Line 42", markdown)
        self.assertIn("mlir-translate --mlir-to-llvmir ll.mlir", markdown)
        self.assertIn("Inspect ll.mlir.", markdown)


if __name__ == "__main__":
    unittest.main()
