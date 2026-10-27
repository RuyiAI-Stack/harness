"""Private workspaces and bounded local services behind authenticated Unix sockets."""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time

import httpx
from sqlalchemy import select

from codex_agent.runtime_config import CONFIG_ENV, runtime_config
from .guard import GATEWAY_ENV, MARKER, OWNER_ENV
from .schema import users, workspaces
from .store import AccountError, digest


def freeze_template(source, destination):
    """Export only reviewed operator inputs, never user state, .git, or credentials."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Choose a new template directory outside the source checkout")
    files = subprocess.run(["git", "-C", str(source), "ls-files", "-z", "--",
        "python/examples/flaggems", "tasks/operators", "scripts", "pytest.ini"],
        check=True, capture_output=True, timeout=30).stdout.decode().split("\0")
    files = [f for f in files if f]
    if not files:
        raise ValueError("No tracked operator sources found")
    contents = {}
    for name in files:
        path = source / name
        if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(source) is False:
            raise ValueError("Template entries must be regular in-checkout files")
        if any(p.startswith(".") or p in {"__pycache__", "agent-results", "node_modules"} for p in Path(name).parts):
            continue
        if path.stat().st_size > 4 * 1024 * 1024 or sum(len(v) for v in contents.values()) > 128 * 1024 * 1024:
            raise ValueError("Template exceeds export limits")
        contents[name] = path.read_bytes()
    destination.mkdir(mode=0o700, parents=True)
    for name, content in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (destination / "template-manifest.json").write_text(json.dumps(
        {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}, sort_keys=True))


class Workbenches:
    def __init__(self, accounts):
        self.accounts, self.config = accounts, accounts.config
        self.children, self.lock = {}, threading.RLock()
        self.socket_dir = None
        self.file_lock = None
        self.config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.config.data_dir.is_symlink() or self.config.data_dir.stat().st_mode & 0o077:
            raise ValueError("Account data_dir must be private (chmod 700), not a symlink")
        manifest = self.config.template / "template-manifest.json"
        self.template_manifest = json.loads(manifest.read_text())
        if not isinstance(self.template_manifest, dict) or not self.template_manifest:
            raise ValueError("Prepare a reviewed template before starting accounts")
        for name, checksum in self.template_manifest.items():
            path = self.config.template / name
            if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                    or not path.resolve().is_relative_to(self.config.template.resolve())
                    or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum):
                raise ValueError("Template content does not match its manifest")

    def start(self):
        # One gateway owns these private services. Do not duplicate workers with --workers N.
        self.file_lock = (self.config.data_dir / "gateway.lock").open("a")
        try:
            fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file_lock.close()
            raise RuntimeError("Another account gateway owns this data directory") from None
        # macOS TMPDIR paths can already exceed AF_UNIX's 104-byte pathname limit.
        self.socket_dir = tempfile.TemporaryDirectory(prefix="triton-accounts-", dir="/tmp")

    def provision(self, uid):
        root = (self.config.data_dir / "workspaces" / uid).resolve()
        workspace_id = digest(str(root.resolve()))
        with self.accounts.engine.begin() as conn:
            user = conn.execute(select(users).where(users.c.id == uid).with_for_update()).mappings().first()
            if not user or user["status"] != "active":
                raise AccountError(403, "Account unavailable")
            existing = conn.execute(select(workspaces).where(workspaces.c.user_id == uid)).mappings().first()
            if existing:
                if existing["root"] != str(root) or existing["status"] != "ready":
                    raise AccountError(409, "Workspace requires administrator reconciliation")
                if root.is_symlink() or (root / MARKER).read_text().strip() != uid:
                    raise AccountError(409, "Workspace identity mismatch")
                return root
            if root.exists():
                # A previous interrupted provisioning is not silently overwritten or adopted.
                raise AccountError(409, "Incomplete workspace; contact the administrator")
            root.mkdir(parents=True, mode=0o700)
            for name, checksum in self.template_manifest.items():
                source = self.config.template / name
                if source.is_symlink() or not source.resolve().is_relative_to(self.config.template.resolve()):
                    raise AccountError(409, "Template changed")
                body = source.read_bytes()
                if hashlib.sha256(body).hexdigest() != checksum:
                    raise AccountError(409, "Template changed")
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            (root / MARKER).write_text(uid)
            (root / MARKER).chmod(0o400)
            conn.execute(workspaces.insert().values(workspace_id=workspace_id, user_id=uid, root=str(root), status="ready"))
            self.accounts.log(conn, "workspace-created", uid)
        return root

    def environment(self, root, uid, key):
        cfg = runtime_config()
        if any((cfg.permissions.development, cfg.permissions.repair, cfg.permissions.validation)) and not cfg.remote.required:
            raise AccountError(503, "Account execution requires a configured, isolated remote validator")
        child = cfg.model_copy(update={"repoRoot": str(root), "stateDir": str(root / "agent-results")})
        env = dict(os.environ)
        for name in ("DSH_HOME", "DSH_SESSION_ROOT", "TRITON_RISCV_CORDIS_CONFIG", "PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        env.update({CONFIG_ENV: child.model_dump_json(), OWNER_ENV: uid, GATEWAY_ENV: key,
            "TRITON_RISCV_REPO_ROOT": str(root), "TRITON_RISCV_STATE_DIR": str(root / "agent-results"),
            "DSH_SESSION_ROOT": str(root / "agent-results/deepseek-harness"),
            "TRITON_RISCV_REQUIRE_APPROVED_VALIDATION": "1"})
        return env

    def get(self, uid):
        with self.lock:
            if uid in self.children:
                entry = self.children[uid]
                if any(p.poll() is not None for p in entry["processes"]):
                    self.stop(uid)
                    raise AccountError(503, "Workspace service stopped; retry to recover")
                return entry
            if len(self.children) >= self.config.max_active:
                raise AccountError(503, "Active workspace capacity reached; contact the administrator")
            if self.socket_dir is None:
                raise AccountError(503, "Account gateway has not started")
            root = self.provision(uid)
            key = secrets.token_urlsafe(32)
            env = self.environment(root, uid, key)
            socket = Path(self.socket_dir.name) / (uid + ".sock")
            log_path = self.config.data_dir / (uid + ".service.log")
            log = log_path.open("a")
            log_path.chmod(0o600)
            entry = {"root": root, "socket": socket, "key": key, "processes": [], "slots": threading.BoundedSemaphore(8)}
            try:
                args = [sys.executable, "-I", "-m", "uvicorn", "codex_agent.platform.api:create_app", "--factory",
                        "--uds", str(socket), "--no-access-log", "--no-proxy-headers"]
                entry["processes"].append(subprocess.Popen(args, cwd=root, env=env, stdout=log, stderr=log, start_new_session=True))
                deadline = time.monotonic() + 30
                with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket)), base_url="http://private", timeout=1) as client:
                    while time.monotonic() < deadline:
                        if entry["processes"][0].poll() is not None:
                            raise AccountError(503, "Workspace failed to start; inspect the private service log")
                        try:
                            if client.get("/api/health", headers={"x-triton-gateway": key}).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(.1)
                    else:
                        raise AccountError(503, "Workspace startup timed out")
                cfg = runtime_config()
                roles = ["relay", "recover"] if cfg.queue.enabled or cfg.indexing.enabled else []
                if cfg.queue.enabled:
                    roles += ["agent", "validation", "memory"]
                if cfg.indexing.enabled:
                    roles += ["index"]
                for role in roles:
                    entry["processes"].append(subprocess.Popen([sys.executable, "-I", "-m", "codex_agent.workers",
                        role, "--workspace", str(root)], cwd=root, env=env, stdout=log, stderr=log, start_new_session=True))
                self.children[uid] = entry
                return entry
            except BaseException:
                self.children[uid] = entry
                self.stop(uid)
                raise
            finally:
                log.close()

    def stop(self, uid):
        with self.lock:
            entry = self.children.pop(uid, None)
            if entry is None:
                return
            for process in reversed(entry["processes"]):
                if process.poll() is None:
                    process.terminate()
            for process in reversed(entry["processes"]):
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            entry["socket"].unlink(missing_ok=True)

    async def monitor(self):
        def active_accounts():
            with self.accounts.engine.connect() as conn:
                return set(conn.execute(select(users.c.id).where(users.c.status == "active")).scalars())

        while True:
            await asyncio.sleep(2)
            try:
                active = await asyncio.to_thread(active_accounts)
                for uid in list(self.children):
                    if uid not in active:
                        await asyncio.to_thread(self.stop, uid)
            except asyncio.CancelledError:
                raise
            except Exception:
                # DB unavailability must not leave already-admitted work running unchecked.
                for uid in list(self.children):
                    await asyncio.to_thread(self.stop, uid)

    def close(self):
        for uid in list(self.children):
            self.stop(uid)
        if self.socket_dir:
            self.socket_dir.cleanup()
        if self.file_lock:
            self.file_lock.close()
