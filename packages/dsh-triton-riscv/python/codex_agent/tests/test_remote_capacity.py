import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from codex_agent import remote_capacity as capacity


CHILD = """
import json, sys, time
from pathlib import Path
from codex_agent.remote_capacity import CapacityPool
pool = CapacityPool(Path(sys.argv[1]), int(sys.argv[2]))
with pool.acquire(sys.argv[3], wait_seconds=3) as slot:
    print(json.dumps({'event': 'enter', 'time': time.monotonic(), **slot}), flush=True)
    time.sleep(float(sys.argv[4]))
    print(json.dumps({'event': 'exit', 'time': time.monotonic()}), flush=True)
"""


class RemoteCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "slots"
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        self.temporary.cleanup()

    def child(self, name, seconds, limit=2):
        child = subprocess.Popen([sys.executable, "-c", CHILD, str(self.root), str(limit), name, str(seconds)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(child)
        return child

    def test_four_real_processes_never_exceed_two_slots(self):
        children = [self.child(str(index), 0.35) for index in range(4)]
        events = []
        waits = []
        for child in children:
            output, error = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 0, error)
            rows = [json.loads(line) for line in output.splitlines()]
            self.assertEqual([row["event"] for row in rows], ["enter", "exit"])
            waits.append(rows[0]["queue_seconds"])
            events.extend(rows)
        active = maximum = 0
        for row in sorted(events, key=lambda item: item["time"]):
            active += 1 if row["event"] == "enter" else -1
            maximum = max(maximum, active)
        self.assertEqual((maximum, active), (2, 0))
        self.assertGreater(max(waits), 0.1)
        self.assertEqual([row["state"] for row in capacity.CapacityPool(self.root).snapshot()], ["free", "free"])

    def test_busy_queue_expires_without_entering_and_normal_release_reuses_slot(self):
        pool = capacity.CapacityPool(self.root, limit=1)
        with pool.acquire("first"):
            with self.assertRaises(capacity.CapacityUnavailable):
                with pool.acquire("second", wait_seconds=0.15):
                    self.fail("busy pool entered")
            self.assertEqual(pool.snapshot()[0]["state"], "occupied")
        with pool.acquire("third", wait_seconds=0) as slot:
            self.assertEqual(slot["slot"], 0)

    def test_killed_owner_fences_slot_instead_of_assuming_test_has_stopped(self):
        child = self.child("lost-supervisor", 10, limit=1)
        first = json.loads(child.stdout.readline())
        self.assertEqual(first["event"], "enter")
        pool = capacity.CapacityPool(self.root, limit=1)
        self.assertEqual(pool.snapshot()[0]["state"], "occupied")
        child.kill()
        child.communicate(timeout=5)
        state = pool.snapshot()[0]
        self.assertEqual(state["state"], "fenced")
        self.assertEqual(state["reservation"]["job_id"], "lost-supervisor")
        with self.assertRaises(capacity.CapacityUnavailable):
            with pool.acquire("retry", wait_seconds=0):
                self.fail("unknown execution released capacity")

    def test_exception_keeps_reservation_and_does_not_delete_lock_inode(self):
        pool = capacity.CapacityPool(self.root, limit=1)
        with self.assertRaisesRegex(RuntimeError, "receipt"):
            with pool.acquire("unknown"):
                inode = (self.root / "slot-0.lock").stat().st_ino
                raise RuntimeError("receipt could not be committed")
        self.assertEqual(pool.snapshot()[0]["state"], "fenced")
        self.assertEqual((self.root / "slot-0.lock").stat().st_ino, inode)

    def test_conflicting_policy_and_corrupt_state_fail_closed(self):
        pool = capacity.CapacityPool(self.root, limit=1)
        with self.assertRaisesRegex(ValueError, "policy mismatch"):
            capacity.CapacityPool(self.root, limit=2)
        (self.root / "slot-0.json").write_text("broken")
        with self.assertRaises(ValueError):
            with pool.acquire("new", wait_seconds=0):
                self.fail("corrupt state accepted")

    def test_symlinks_and_shared_permissions_are_rejected(self):
        target = Path(self.temporary.name) / "target"
        target.mkdir(mode=0o700)
        self.root.symlink_to(target)
        with self.assertRaises(ValueError):
            capacity.CapacityPool(self.root)
        self.root.unlink()
        self.root.mkdir(mode=0o755)
        with self.assertRaises(ValueError):
            capacity.CapacityPool(self.root)
        self.root.chmod(0o700)
        pool = capacity.CapacityPool(self.root)
        (self.root / "slot-0.lock").symlink_to(target / "untouched")
        with self.assertRaises(OSError):
            with pool.acquire("blocked"):
                self.fail("symlink lock accepted")
        self.assertFalse((target / "untouched").exists())

    def test_fixed_server_policy_is_not_read_from_model_or_environment(self):
        self.assertEqual(capacity.MAX_ACTIVE_JOBS, 2)
        self.assertEqual(capacity.QUEUE_TIMEOUT_SECONDS, 30)
        with self.assertRaises(ValueError):
            capacity.CapacityPool(self.root, limit=True)


if __name__ == "__main__":
    unittest.main()
