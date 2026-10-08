"""Opt-in Linux isolation acceptance. Uses only disposable canary files/sockets.

Run with the trusted environment activated, --repo for its runtime, and --output
inside a designated temporary experiment directory. No real credentials are read.
This test bypasses static screening to exercise the OS layer independently.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
import uuid


CHECKS = r'''
import ctypes, errno, json, os, socket, subprocess
from pathlib import Path
settings = json.loads(Path("/work/settings.json").read_text())
results = []
def check(name, function):
    try:
        function()
        results.append({"name": name, "passed": True})
    except Exception as error:
        results.append({"name": name, "passed": False, "error": str(error)})
def require(value):
    assert value
def denied(function):
    try:
        function()
    except OSError as error:
        assert error.errno in {errno.ENOENT, errno.EACCES, errno.EPERM, errno.EROFS, errno.ENETUNREACH, errno.ECONNREFUSED}
    else:
        raise AssertionError("forbidden operation unexpectedly succeeded")
check("input_read_allowed", lambda: require(Path("/work/sentinel").read_text() == "unchanged"))
check("scratch_write_allowed", lambda: Path("/tmp/scratch").write_text("allowed"))
check("outside_read_blocked", lambda: denied(lambda: Path(settings["outside"]).read_bytes()))
check("outside_write_blocked", lambda: denied(lambda: Path(settings["outside"]).write_text("wrong")))
check("input_write_blocked", lambda: denied(lambda: Path("/work/sentinel").write_text("wrong")))
check("root_write_blocked", lambda: denied(lambda: Path("/root-canary").write_text("wrong")))
check("runtime_write_blocked", lambda: denied(lambda: Path(settings["runtime"]).write_text("wrong")))
check("symlink_escape_blocked", lambda: denied(lambda: Path("/work/escape").read_bytes()))
check("host_process_hidden", lambda: denied(lambda: Path(f'/proc/{settings["parent_pid"]}/cmdline').read_bytes()))
check("credential_environment_removed", lambda: require(not {"ISRC_API_KEY", "SSH_AUTH_SOCK", "PRIVATE_CANARY", "TRITON_RISCV_ALLOW_VALIDATION"} & set(os.environ)))
def tcp():
    with socket.socket() as connection:
        connection.settimeout(1)
        denied(lambda: connection.connect(("127.0.0.1", settings["port"])))
check("host_tcp_blocked", tcp)
def unix(address):
    with socket.socket(socket.AF_UNIX) as connection:
        denied(lambda: connection.connect(address))
check("host_unix_path_blocked", lambda: unix(settings["unix"]))
check("host_abstract_socket_blocked", lambda: unix("\0" + settings["abstract"]))
def caps():
    fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    require(int(fields["CapEff"].strip(), 16) == 0)
    require(int(fields["CapBnd"].strip(), 16) == 0)
    require(fields["NoNewPrivs"].strip() == "1")
check("capabilities_removed", caps)
def remount():
    Path("/tmp/remount-target").mkdir()
    result = subprocess.run(["mount", "--bind", "/usr", "/tmp/remount-target"], capture_output=True)
    require(result.returncode != 0)
check("remount_blocked", remount)
def compiler():
    Path("/tmp/sample.c").write_text("int answer(void) { return 42; }\n")
    compiled = subprocess.run(["gcc", "-shared", "-fPIC", "/tmp/sample.c", "-o", "/tmp/sample.so"], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    require(ctypes.CDLL("/tmp/sample.so").answer() == 42)
check("compiler_and_shared_library_allowed", compiler)
print("ACCEPTANCE_JSON=" + json.dumps(results), flush=True)
raise SystemExit(0 if all(item["passed"] for item in results) else 1)
'''


TIMEOUT_CHECK = r'''
import ctypes, os, sys, time
libc = ctypes.CDLL(None)
libc.prctl(15, sys.argv[1].encode(), 0, 0, 0)
if os.fork() == 0:
    os.setsid()
    libc.prctl(15, sys.argv[2].encode(), 0, 0, 0)
    print("DETACHED_CHILD_READY", flush=True)
time.sleep(30)
'''


def marked_processes(markers):
    found = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            name = (path / "comm").read_text().strip()
            if name in markers:
                found[int(path.name)] = name
        except (OSError, ValueError):
            continue
    return found


def timeout_acceptance(args):
    markers = ["tsi" + uuid.uuid4().hex[:10], "tsc" + uuid.uuid4().hex[:10]]
    report = {"kind": "timeout-descendant-cleanup", "real_operator_test": False}
    with tempfile.TemporaryDirectory(prefix="timeout-", dir=args.output) as temporary:
        stage = Path(temporary)
        (stage / "workspace").mkdir()
        (stage / "workspace/timeout_check.py").write_text(TIMEOUT_CHECK)
        command = ["timeout", "--signal=TERM", "--kill-after=2s", "3s", sys.executable,
                   "-I", str(args.launcher.resolve()), "--repo", str(args.repo.resolve()),
                   "--stage", str(stage), "--", "python", "/work/timeout_check.py", *markers]
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seen = set()
        try:
            while process.poll() is None and time.monotonic() - started < 10:
                seen.update(marked_processes(markers).values())
                time.sleep(0.1)
            output, _ = process.communicate(timeout=5)
            (args.output / "timeout.log").write_text(output)
            deadline = time.monotonic() + 3
            survivors = marked_processes(markers)
            while survivors and time.monotonic() < deadline:
                time.sleep(0.1)
                survivors = marked_processes(markers)
            report.update(exit_code=process.returncode, duration_seconds=round(time.monotonic()-started, 3),
                          parent_and_detached_child_observed=set(markers) <= seen,
                          survivors=survivors)
            report["passed"] = (report["parent_and_detached_child_observed"] and not survivors
                                and process.returncode in {124, 137, -signal.SIGKILL})
        finally:
            # Only our random, same-user fixture processes are eligible for cleanup.
            for pid in marked_processes(markers):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    (args.output / "timeout-result.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"kind": "OS-isolation-canary-acceptance", "real_operator_test": False, "checks": []}
    with tempfile.TemporaryDirectory(prefix="canaries-", dir=args.output) as temporary:
        stage = Path(temporary)
        work = stage / "workspace"
        work.mkdir()
        (work / "sentinel").write_text("unchanged")
        outside = stage / "outside-canary"
        outside.write_text("fixture-not-a-secret")
        (work / "escape").symlink_to(outside)
        runtime = stage / "runtime"
        runtime.mkdir()
        (runtime / "readonly-canary").write_text("unchanged")
        with socket.socket() as tcp, socket.socket(socket.AF_UNIX) as unix, socket.socket(socket.AF_UNIX) as abstract:
            tcp.bind(("127.0.0.1", 0)); tcp.listen(2)
            unix_name = str(stage / "host.sock")
            abstract_name = "triton-acceptance-" + uuid.uuid4().hex
            unix.bind(unix_name); unix.listen(1)
            abstract.bind("\0" + abstract_name); abstract.listen(1)
            # Positive control proves the host endpoint exists before testing denial.
            with socket.create_connection(tcp.getsockname(), timeout=1):
                connection, _ = tcp.accept()
                connection.close()
            (work / "settings.json").write_text(json.dumps({
                "outside": str(outside), "runtime": str(runtime / "readonly-canary"),
                "parent_pid": os.getpid(), "port": tcp.getsockname()[1], "unix": unix_name, "abstract": abstract_name,
            }))
            (work / "checks.py").write_text(CHECKS)
            environment = {**os.environ, "BUILD_DIR": str(runtime), "ISRC_API_KEY": "fixture-not-a-key",
                           "PRIVATE_CANARY": "fixture", "SSH_AUTH_SOCK": unix_name}
            command = [sys.executable, "-I", str(args.launcher.resolve()), "--repo", str(args.repo.resolve()),
                       "--stage", str(stage), "--", "python", "/work/checks.py"]
            started = time.monotonic()
            result = subprocess.run(["timeout", "--kill-after=3s", "45s", *command],
                env=environment, capture_output=True, text=True, timeout=55)
            report.update(exit_code=result.returncode, duration_seconds=round(time.monotonic()-started, 3))
            (args.output / "canary.log").write_text(result.stdout + result.stderr)
            for line in result.stdout.splitlines():
                if line.startswith("ACCEPTANCE_JSON="):
                    report["checks"] = json.loads(line.split("=", 1)[1])
            report["outside_unchanged"] = outside.read_text() == "fixture-not-a-secret"
            report["input_unchanged"] = (work / "sentinel").read_text() == "unchanged"
            report["runtime_unchanged"] = (runtime / "readonly-canary").read_text() == "unchanged"
    report["passed"] = bool(report["checks"]) and report["exit_code"] == 0 and all(
        item["passed"] for item in report["checks"]) and all(report[key] for key in (
            "outside_unchanged", "input_unchanged", "runtime_unchanged"))
    (args.output / "canary-result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        return 1
    timeout_report = timeout_acceptance(args)
    print(json.dumps(timeout_report, indent=2))
    return 0 if timeout_report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
