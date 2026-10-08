import tempfile
import unittest
from pathlib import Path

from codex_agent.source_risk import assert_source_screened, screen_operator_files, source_findings
from codex_agent.tests.test_operator_development import IMPLEMENTATION, TEST_SOURCE


class SourceRiskTests(unittest.TestCase):
    def test_normal_operator_and_tests_are_accepted(self):
        assert_source_screened(IMPLEMENTATION)
        assert_source_screened(TEST_SOURCE)
        assert_source_screened('text = "os.system(\\\"echo nope\\\")"\n# eval is a word here\n')

    def test_direct_operations_and_simple_aliases_are_flagged_without_execution(self):
        sources = [
            'import subprocess as sp\nsp.run(["true"])',
            'from os import remove as delete\ndelete("file")',
            'import os\nrun = os.system\nrun("true")',
            'from pathlib import Path as P\np = P("file")\np.write_text("x")',
            'import builtins as b\nb.eval("1")',
            'import requests\nrequests.get("https://example.invalid")',
            'import os\nvalue = os.getenv("SERVICE_TOKEN")',
            'import os\nvalue = os.environ["SECRET"]',
            'import importlib\nimportlib.import_module("dynamic")',
        ]
        for source in sources:
            with self.subTest(source=source):
                findings = source_findings(source, "candidate.py")
                self.assertTrue(findings)
                self.assertTrue(all(item["path"] == "candidate.py" and item["line"] > 0 for item in findings))
                with self.assertRaisesRegex(ValueError, "NOT proof"):
                    assert_source_screened(source)

    def test_file_screen_is_bounded_and_root_confined(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "operator.py"
            source.write_text(IMPLEMENTATION)
            screen_operator_files(root, ["operator.py", "operator.py"])
            source.write_text("x" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "1 MiB"):
                screen_operator_files(root, ["operator.py"])
            with self.assertRaises(ValueError):
                screen_operator_files(root, ["../outside.py"])


if __name__ == "__main__":
    unittest.main()
