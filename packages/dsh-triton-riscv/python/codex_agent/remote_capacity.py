"""Server-side admission for trusted supervisors, not a machine resource quota.

Slot files are never unlinked: replacing lock inodes would split the lock domain.
A durable reservation survives supervisor death and fails closed until inspected.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid

MAX_ACTIVE_JOBS = 2
QUEUE_TIMEOUT_SECONDS = 30
POLICY_VERSION = 1


class CapacityUnavailable(RuntimeError):
    pass


class AdmissionCancelled(RuntimeError):
    pass


def _write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def _open_lock(path, *, create=True):
    fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0) | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
            info.st_mode & 0o077 or info.st_nlink != 1):
        os.close(fd)
        raise ValueError("capacity lock must be a private, owned regular file")
    return os.fdopen(fd, "r+")


def _read(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "r") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > 65536):
            raise ValueError("capacity state must be a small, private, owned regular file")
        text = handle.read(65537)
        if len(text) > 65536:
            raise ValueError("capacity state exceeds size limit")
        return json.loads(text)


class CapacityPool:
    """Independent processes share a fixed policy and fail-closed reservations."""

    def __init__(self, root: Path, limit: int = MAX_ACTIVE_JOBS):
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("invalid capacity limit")
        self.root, self.limit = root, limit
        root.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("capacity directory must be private, owned and not a symlink")
        with _open_lock(root / "policy.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            expected = {"version": POLICY_VERSION, "limit": limit}
            current = _read(root / "policy.json")
            if current is None and not (root / "policy.json").exists():
                _write(root / "policy.json", expected)
            elif current != expected:
                raise ValueError("capacity policy mismatch; drain jobs before changing policy")

    @contextmanager
    def acquire(self, job_id: str, wait_seconds: float = QUEUE_TIMEOUT_SECONDS, *, cancelled=None,
                request_digest=None):
        if not 0 <= wait_seconds <= 60:
            raise ValueError("invalid capacity wait budget")
        started = time.monotonic()
        selected = None
        while selected is None:
            if cancelled and cancelled():
                raise AdmissionCancelled("cancelled before admission")
            for index in range(self.limit):
                lock = _open_lock(self.root / f"slot-{index}.lock")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                record = self.root / f"slot-{index}.json"
                try:
                    if _read(record) is not None:
                        lock.close()
                        continue
                    reservation = {"job_id": job_id, "reserved_at": time.time(),
                                   "request_digest": request_digest, "reservation_id": uuid.uuid4().hex}
                    _write(record, reservation)
                except BaseException:
                    lock.close()
                    raise
                selected = lock, record, index
                break
            if selected is None:
                remaining = wait_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise CapacityUnavailable("all validation slots are busy or fenced; no test was started")
                time.sleep(min(0.1, remaining))
        lock, record, index = selected
        try:
            yield {"slot": index, "limit": self.limit,
                   "reservation_id": reservation["reservation_id"],
                   "queue_seconds": round(time.monotonic() - started, 3)}
        except BaseException:
            # Do not infer child termination from loss of the supervisor.
            raise
        else:
            _write(record, None)
        finally:
            lock.close()

    def snapshot(self):
        result = []
        for index in range(self.limit):
            with _open_lock(self.root / f"slot-{index}.lock") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    status = "occupied"
                    # The writer may be between locking and reserving.
                    record = _read(self.root / f"slot-{index}.json")
                else:
                    record = _read(self.root / f"slot-{index}.json")
                    status = "fenced" if record is not None else "free"
                result.append({"slot": index, "state": status, "reservation": record})
        return result


def server_pool():
    # Shared across checkouts and clients for this Unix account. Never a tool argument.
    return CapacityPool(server_root())


def server_root():
    return Path(f"/tmp/triton-riscv-capacity-{os.getuid()}")


def _existing_policy(root):
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("capacity directory must be private, owned and not a symlink")
    policy = _read(root / "policy.json")
    if (not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION or
            type(policy.get("limit")) is not int or not 1 <= policy["limit"] <= 8):
        raise ValueError("invalid existing capacity policy")
    return policy


def inspect_existing(root):
    """Read only, including on an uninitialized pool: no mkdir or lock creation."""
    if not root.exists() and not root.is_symlink():
        return []
    policy = _existing_policy(root)
    rows = []
    for index in range(policy["limit"]):
        path = root / f"slot-{index}.json"
        try:
            lock = _open_lock(root / f"slot-{index}.lock", create=False)
        except FileNotFoundError:
            if path.exists() or path.is_symlink():
                raise ValueError("reservation without a persistent slot lock")
            rows.append({"slot": index, "state": "uninitialized", "reservation": None})
            continue
        with lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = False
            except BlockingIOError:
                busy = True
            record = _read(path)
            if record is not None and not isinstance(record, dict):
                raise ValueError("invalid capacity reservation")
            rows.append({"slot": index, "state": "occupied" if busy else "fenced" if record is not None else "free",
                         "reservation": record})
    return rows


def reservation_token(slot, record):
    return hashlib.sha256(json.dumps({"slot": slot, "reservation": record}, sort_keys=True).encode()).hexdigest()


def inspect_job_capacity(root, job_id, request_digest, verify_terminal):
    rows = [row for row in inspect_existing(root) if (row["reservation"] or {}).get("job_id") == job_id]
    report = {"state": "no-reservation" if not rows else "blocked", "releasable": False, "slots": rows}
    if not rows:
        return report
    if len(rows) != 1:
        return {**report, "reason": "multiple reservations require host inspection"}
    row = rows[0]
    try:
        with _open_lock(root / f"slot-{row['slot']}.lock", create=False) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record = _read(root / f"slot-{row['slot']}.json")
            if record != row["reservation"]:
                raise ValueError("reservation changed; inspect again")
            _check_identity(record, job_id, request_digest)
            evidence = verify_terminal(row["slot"], record)
        return {**report, "state": "releasable", "releasable": True, "evidence": evidence,
                "release_token": reservation_token(row["slot"], record)}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {**report, "reason": str(error)}


def _check_identity(record, job_id, request_digest):
    if (not isinstance(record, dict) or record.get("job_id") != job_id or
            not request_digest or record.get("request_digest") != request_digest or
            not record.get("reservation_id")):
        raise ValueError("reservation identity missing or changed; legacy reservations require host inspection")


def release_job_capacity(root, job_id, request_digest, token, verify_terminal):
    """Explicit compare-and-release, never delete a lock or act on process age."""
    if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("invalid release token")
    policy = _existing_policy(root)
    audit_path = root / f"release-{token}.json"
    def read_audit():
        previous = _read(audit_path)
        if previous is not None and (not isinstance(previous, dict) or previous.get("job_id") != job_id or
                previous.get("request_digest") != request_digest or previous.get("release_token") != token or
                previous.get("state") not in {"prepared", "released"} or
                type(previous.get("slot")) is not int or not 0 <= previous["slot"] < policy["limit"]):
            raise ValueError("release audit identity mismatch")
        return previous
    previous = read_audit()
    if previous is not None:
        slot = previous["slot"]
    else:
        rows = [row for row in inspect_existing(root)
                if row["reservation"] and reservation_token(row["slot"], row["reservation"]) == token]
        if len(rows) != 1:
            raise ValueError("reservation changed; inspect and confirm again")
        slot = rows[0]["slot"]
    with _open_lock(root / f"slot-{slot}.lock", create=False) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = _read(root / f"slot-{slot}.json")
        # Re-read under the slot lock: another release may have just committed.
        previous = read_audit()
        if previous and (current is None or reservation_token(slot, current) != token):
            if previous["state"] != "released":
                _write(audit_path, {**previous, "state": "released", "reconciled_at": time.time()})
            return {"state": "already-released", "slot": slot, "release_token": token}
        if current is None or reservation_token(slot, current) != token:
            raise ValueError("reservation changed; inspect and confirm again")
        _check_identity(current, job_id, request_digest)
        evidence = verify_terminal(slot, current)
        audit = {"job_id": job_id, "request_digest": request_digest, "release_token": token,
                 "slot": slot, "reservation": current, "evidence": evidence, "prepared_at": time.time()}
        _write(audit_path, {**audit, "state": "prepared"})
        _write(root / f"slot-{slot}.json", None)
        _write(audit_path, {**audit, "state": "released", "released_at": time.time()})
    return {"state": "released", "slot": slot, "release_token": token, "evidence": evidence}
