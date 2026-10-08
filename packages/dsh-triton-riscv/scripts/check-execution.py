"""Run independent execution-safety checks locally; never invoke SSH or an LLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


GROUPS = {
    "resource-capabilities": ["test_resource_probe.py", "test_remote_executor.py"],
    "retention-preview": ["test_retention.py", "test_remote_maintenance.py"],
    "capacity-recovery": ["test_capacity_recovery.py", "test_remote_capacity.py", "test_remote_maintenance.py"],
    "batch-resume": ["test_batch_recovery.py", "test_project_tools.py"],
    "read-reconnect": ["test_remote_retry.py", "test_remote_executor.py"],
    "remote-stop": ["test_remote_control.py", "test_remote_process.py", "test_remote_job.py", "test_native_bridge.py"],
    "capacity": ["test_remote_capacity.py", "test_remote_job.py", "test_remote_executor.py"],
    "cleanup": ["test_remote_maintenance.py"],
    "status": ["test_mcp_tools.py"],
    "recovery-and-approval": ["test_execution_guard.py", "test_operator_lifecycle.py", "test_native_bridge.py"],
}


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=package / ".venv/bin/python")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing results are never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("RISCV_", "TRITON_RISCV_"))}
    environment["TRITON_RISCV_EMBEDDING_PROVIDER"] = "none"
    report = {"scope": "local-tests-only", "model_calls": 0, "remote_calls": 0, "modules": []}
    for name, files in GROUPS.items():
        paths = [f"codex_agent/tests/{file}" for file in files]
        # Resolving the venv's symlink would select the base interpreter and lose
        # its installed dependencies. Keep the entrypoint inside the venv.
        command = [str(args.python.absolute()), "-m", "pytest", "-q", *paths, f"--junitxml={output / (name + '.xml')}"]
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=package / "python", env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
            text, code = completed.stdout, completed.returncode
        except subprocess.TimeoutExpired as error:
            text, code = (error.stdout or b"") + b"\nacceptance timeout\n", 124
        (output / (name + ".log")).write_bytes(text)
        item = {"module": name, "command": command, "exit_code": code,
                "duration_seconds": round(time.monotonic() - started, 3),
                "status": "passed" if code == 0 else "failed",
                "test_files_sha256": {path: hashlib.sha256((package / "python" / path).read_bytes()).hexdigest() for path in paths}}
        try:
            suites = ET.parse(output / (name + ".xml")).getroot().iter("testsuite")
            rows = list(suites)
            item["junit"] = {key: sum(int(row.get(key, "0")) for row in rows)
                             for key in ("tests", "failures", "errors", "skipped")}
            if not rows or item["junit"]["failures"] or item["junit"]["errors"]:
                item["status"] = "failed"
        except (OSError, ET.ParseError):
            item["status"] = "failed"
            item["evidence_error"] = "missing or invalid JUnit result"
        report["modules"].append(item)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {item['status']} ({item['duration_seconds']}s)", flush=True)
    print(f"Saved logs and per-module results: {output}")
    return int(any(item["status"] != "passed" for item in report["modules"]))


if __name__ == "__main__":
    raise SystemExit(main())
