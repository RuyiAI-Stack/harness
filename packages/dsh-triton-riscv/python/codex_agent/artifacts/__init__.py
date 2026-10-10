"""Content-addressed task artifacts; the current adapter uses the workspace filesystem."""
import hashlib
import os
from pathlib import Path
import re
import tempfile

from codex_agent.paths import state_root
from codex_agent.tasks.schema import artifacts
from codex_agent.tasks.store import digest


class ArtifactStore:
    def __init__(self, root):
        self.folder = (state_root(root) / "task-artifacts").resolve()

    def put(self, jobs, job_id, owner, name, data):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name) or len(data) > 16 * 1024 * 1024:
            raise ValueError("invalid artifact name or artifact exceeds 16 MiB")
        checksum = hashlib.sha256(data).hexdigest()
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder / checksum
        with tempfile.NamedTemporaryFile(dir=self.folder, delete=False) as stream:
            temp = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp, target)
            directory = os.open(self.folder, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            with jobs.db.transaction() as conn:
                job = jobs.require_owner(conn, job_id, owner)
                identifier = digest([job_id, job["attempt"], name])
                old = jobs.db.rows(artifacts, artifacts.c.id == identifier, connection=conn)
                if old:
                    if old[0]["sha256"] != checksum:
                        raise ValueError("artifact identity already has different bytes")
                else:
                    jobs.db.insert(conn, artifacts, dict(id=identifier, job_id=job_id, attempt=job["attempt"],
                        sha256=checksum, size=len(data), name=name, storage_key=checksum))
            return identifier
        finally:
            temp.unlink(missing_ok=True)

    def read(self, record):
        key = record["storage_key"]
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("invalid artifact storage key")
        path = self.folder / key
        if path.is_symlink() or path.stat().st_size != record["size"]:
            raise ValueError("artifact changed")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("artifact checksum mismatch")
        return data
