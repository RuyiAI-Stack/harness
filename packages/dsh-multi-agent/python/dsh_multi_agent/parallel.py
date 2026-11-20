"""Bounded fan-out; authorization and task-specific execution belong to callers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile


@contextmanager
def local_replica_slot():
    """Four replicas per OS user across local workers; not a distributed quota."""
    folder = Path(tempfile.gettempdir()) / f"dsh-replicas-{os.getuid()}"
    folder.mkdir(mode=0o700, exist_ok=True)
    if folder.is_symlink() or folder.stat().st_uid != os.getuid():
        raise PermissionError("Unsafe replica quota directory")
    acquired = None
    for i in range(4):
        handle = (folder / f"slot-{i}").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = handle
            break
        except BlockingIOError:
            handle.close()
    if acquired is None:
        raise RuntimeError("Local replica capacity exhausted; no model call made")
    try:
        yield
    finally:
        fcntl.flock(acquired, fcntl.LOCK_UN)
        acquired.close()


def run_parallel(tasks, run, *, cancelled=lambda: False, on_result=lambda row: None,
                 max_parallel=2):
    if not 1 <= max_parallel <= 2 or not 1 <= len(tasks) <= 4:
        raise ValueError("At most four tasks and two concurrent replicas are allowed")
    identifiers = [task["task_id"] for task in tasks]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Replica task IDs must be nonempty and unique")

    def execute(task):
        if cancelled():
            return {"task_id": task["task_id"], "status": "cancelled"}
        try:
            return {**run(task, cancelled), "task_id": task["task_id"]}
        except Exception as error:
            # Arbitrary provider errors can contain secrets. Retain the category only.
            return {"task_id": task["task_id"], "status": "failed", "error_type": type(error).__name__}

    results = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            results[row["task_id"]] = row
            on_result(row)
    return [results[task["task_id"]] for task in tasks]
