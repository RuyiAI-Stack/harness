import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from codex_agent import linux_sandbox as sandbox


class LinuxSandboxTests(unittest.TestCase):
    def test_policy_is_allowlisted_not_ambient_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/bin/python").write_text("fixture")
            (root / "stage/workspace").mkdir(parents=True)
            env = {"ISRC_API_KEY": "fake-secret", "SSH_AUTH_SOCK": "/fake",
                   "PYTHONPATH": "/untrusted", "LD_PRELOAD": "/untrusted.so", "TRITON_INTERPRET": "1",
                   "TRITON_RISCV_ALLOW_VALIDATION": "1", "HTTP_PROXY": "http://invalid"}
            with patch.object(sandbox, "namespaces", return_value={"pid": "before"}):
                policy = sandbox.build_policy(root, root / "stage", env, ["python", "-m", "pytest"])
            self.assertFalse(set(env) & set(policy["environment"]) - {"PYTHONPATH"})
            self.assertNotIn("untrusted", policy["environment"]["PYTHONPATH"])
            self.assertEqual(policy["environment"]["HOME"], "/tmp/home")
            self.assertEqual(policy["mounts"], [str((root / ".venv").resolve())])
            self.assertNotIn(str(root.resolve()), policy["mounts"])
            self.assertEqual(policy["command"][0], str(root.resolve() / ".venv/bin/python"))

    def test_broad_or_escaping_runtime_grants_are_rejected(self):
        for value in ("/", "/tmp", "/etc", "/home", str(Path.home()), "relative", "/usr/../"):
            with self.subTest(value=value), self.assertRaises((ValueError, FileNotFoundError)):
                sandbox.allowed_runtime(value)

    def test_unsupported_platform_and_missing_tools_fail_closed(self):
        with patch.object(sandbox.platform, "system", return_value="Darwin"):
            self.assertEqual(sandbox.main([]), sandbox.EXIT_SETUP_FAILED)
        with patch.object(sandbox.platform, "system", return_value="Linux"), patch.object(
            sandbox.shutil, "which", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "missing isolation tools"):
                sandbox.installed_tools()

    def test_same_namespace_refuses_to_mount_anything(self):
        with patch.object(sandbox, "installed_tools", return_value={}), patch.object(
            sandbox, "namespaces", return_value={"pid": "same"}
        ), patch.object(sandbox, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "not established"):
                sandbox.enter({"before": {"pid": "same"}})
            run.assert_not_called()

    def test_launcher_digest_is_full_sha256(self):
        self.assertEqual(len(sandbox.launcher_digest()), 64)

    def test_root_mount_is_readonly_before_capability_drop_and_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            (stage / "workspace").mkdir()
            policy = {"before": {"pid": "before"}, "stage": str(stage), "mounts": [],
                      "environment": {"PATH": sandbox.SYSTEM_PATH}, "command": ["python", "test.py"]}
            tools = {name: "/usr/bin/" + name for name in sandbox.TOOLS}
            with patch.object(sandbox, "installed_tools", return_value=tools), patch.object(
                sandbox, "namespaces", return_value={"pid": "after"}
            ), patch.object(sandbox.os, "getpid", return_value=1), patch.object(
                sandbox, "_run"
            ) as run, patch.object(sandbox, "_exec") as execute, patch.object(sandbox.resource, "setrlimit"):
                sandbox.enter(policy)
            run.assert_called_with("/usr/bin/mount", "-o", "remount,bind,ro,nosuid,nodev", str(stage / "rootfs"))
            argv, env = execute.call_args.args
            self.assertEqual(argv[:3], [tools["chroot"], str(stage / "rootfs"), tools["setpriv"]])
            self.assertIn("--bounding-set=-all", argv)
            self.assertIn("--no-new-privs", argv)
            self.assertEqual(env, policy["environment"])

    def test_mount_failure_never_executes_candidate(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            sandbox, "installed_tools", return_value={"mount": "/usr/bin/mount"}
        ), patch.object(sandbox, "namespaces", return_value={"pid": "after"}), patch.object(
            sandbox.os, "getpid", return_value=1
        ), patch.object(sandbox, "_run", side_effect=subprocess.CalledProcessError(32, "mount")), patch.object(
            sandbox, "_exec"
        ) as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                sandbox.enter({"before": {"pid": "before"}, "stage": temp})
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
