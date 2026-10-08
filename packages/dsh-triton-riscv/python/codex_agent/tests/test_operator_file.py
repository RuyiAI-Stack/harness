import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from codex_agent.operator_tools import read_operator_file


class OperatorFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.name = "python/examples/flaggems/demo.py"
        self.file = self.root / self.name
        self.file.parent.mkdir(parents=True)

    def test_source_pagination_hash_and_eof(self):
        data = "raise RuntimeError('must not execute')\n# reference\npass\n"
        self.file.write_text(data)
        result = read_operator_file(self.root, self.name, max_lines=2)
        self.assertEqual(result.content, "".join(data.splitlines(keepends=True)[:2]))
        self.assertEqual(result.sha256, hashlib.sha256(data.encode()).hexdigest())
        self.assertEqual(result.next_line, 3)
        self.assertIsNone(read_operator_file(self.root, self.name, 3).next_line)
        self.assertEqual(read_operator_file(self.root, self.name, 4).content, "")

    def test_tasks_are_readable_but_private_and_external_paths_are_not(self):
        task = self.root / "tasks/operators/demo.md"
        task.parent.mkdir(parents=True)
        task.write_text("# contract\n")
        self.assertEqual(read_operator_file(self.root, "tasks/operators/demo.md").content, "# contract\n")
        for path in ["../secret.py", "/tmp/secret.py", "agent-results/private.json",
                     ".env", "python/examples/flaggems/../secret.py", "tasks/operators/x.json",
                     "python//examples/flaggems/demo.py", "python/examples/flaggems/x/y.py"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                read_operator_file(self.root, path)

    def test_symlinked_file_and_directory_and_fifo_are_rejected(self):
        secret = self.root / "secret.py"
        secret.write_text("secret\n")
        self.file.symlink_to(secret)
        with self.assertRaises(OSError):
            read_operator_file(self.root, self.name)
        self.file.unlink()
        os.mkfifo(self.file)
        with self.assertRaises(ValueError):
            read_operator_file(self.root, self.name)
        self.file.unlink()
        self.file.parent.rmdir()
        self.file.parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            read_operator_file(self.root, "python/examples/flaggems/secret.py")

    def test_limits_binary_and_invalid_utf8(self):
        for data in [b"a" * (1024 * 1024 + 1), b"a\x00b", b"\xff"]:
            self.file.write_bytes(data)
            with self.assertRaises(ValueError):
                read_operator_file(self.root, self.name)
        self.file.write_text("x" * 32769)
        with self.assertRaisesRegex(ValueError, "single source line"):
            read_operator_file(self.root, self.name)
        for start, maximum in [(0, 2), (1, 401), (1, 0), (True, 2)]:
            with self.assertRaises(ValueError):
                read_operator_file(self.root, self.name, start, maximum)

    def test_byte_budget_preserves_complete_unicode_lines(self):
        line = "\u4e2d" * 4000 + "\n"
        self.file.write_text(line * 4)
        result = read_operator_file(self.root, self.name)
        self.assertEqual(result.content, line * 2)
        self.assertEqual(result.next_line, 3)
        self.assertLessEqual(len(result.content.encode("utf-8")), 32768)
