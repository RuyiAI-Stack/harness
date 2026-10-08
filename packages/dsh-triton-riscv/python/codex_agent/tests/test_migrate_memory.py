from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
import unittest

from codex_agent.memory import MemoryRecord, MemoryStore
from codex_agent.migrate_memory import migrate_memory


class MemoryMigrationTests(unittest.TestCase):
    def test_backup_keeps_original_and_evidence_without_inventing_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "old"
            run = root / "agent-results/development/example"
            run.mkdir(parents=True)
            log = run / "validation.log"
            log.write_text("1 failed\n")
            (run / ".env").write_text("DO_NOT_COPY=this\n")
            db = root / "agent-results/memory.sqlite3"
            with MemoryStore(db) as store:
                store.add(MemoryRecord(memory_type="failure-diagnosis", operator="demo", semantics="square",
                    pytorch_reference="x*x", summary="Failure, cause unknown", outcome="failed", confidence_grade="C",
                    source_run="agent-results/development/example", evidence={"log_path": str(log),
                    "patch_path": "agent-results/missing.diff", "recommended_actions": ["inspect source"]}))
            before = hashlib.sha256(db.read_bytes()).hexdigest()
            out = Path(temporary) / "new"
            result = migrate_memory(db, root, out)
            self.assertTrue(result["source_file_unchanged"])
            self.assertEqual(before, hashlib.sha256(db.read_bytes()).hexdigest())
            self.assertEqual(len(result["records"]), 1)
            self.assertEqual(len(result["gaps"]), 1)
            with sqlite3.connect(out / "memory.sqlite3") as con:
                row = con.execute("SELECT outcome,evidence_json,source_run FROM memories").fetchone()
            self.assertEqual(row[0], "failed")
            evidence = json.loads(row[1])
            self.assertEqual(Path(evidence["log_path"]).read_text(), "1 failed\n")
            self.assertEqual(evidence["recommended_actions"], ["inspect source"])
            self.assertNotIn("applied_action", evidence)
            self.assertTrue(Path(row[2]).is_dir())
            self.assertFalse(list(out.rglob(".env")))
            with self.assertRaises(FileExistsError):
                migrate_memory(db, root, out)


if __name__ == "__main__":
    unittest.main()
