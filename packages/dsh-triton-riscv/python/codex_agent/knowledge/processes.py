"""Gateway-owned shared index workers; no user source or model-run permissions."""
import os
import signal
import subprocess
import sys

from codex_agent.runtime_config import CONFIG_ENV, Permissions, runtime_config
from .vectors import ROOT_ENV


class KnowledgeWorkers:
    def __init__(self, config):
        self.config = config
        self.processes = []

    def start(self):
        cfg = runtime_config()
        if cfg.storage.urlEnv.startswith("TRITON_RAG_HISTORY_"):
            raise ValueError("The history credential prefix is reserved for gateway reads")
        root = self.config.data_dir / "knowledge"
        if self.config.knowledge_enabled:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        from .store import KnowledgeStore
        KnowledgeStore(root).configure_publishers(self.config.admin_user_ids, enabled=self.config.knowledge_enabled)
        if not self.config.knowledge_enabled:
            return
        cfg = cfg.model_copy(update={"repoRoot": str(root), "stateDir": str(root / "agent-results"),
                                     "permissions": Permissions(),
                                     "queue": cfg.queue.model_copy(update={"enabled": True})})
        env = dict(os.environ)
        for name in tuple(env):
            if name.startswith("TRITON_RAG_HISTORY_"):
                env.pop(name)
        for key in ("TRITON_WORKBENCH_OWNER", "TRITON_WORKBENCH_GATEWAY_KEY", ROOT_ENV,
                    "PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        env[CONFIG_ENV] = cfg.model_dump_json()
        log_path = self.config.data_dir / "knowledge.service.log"
        try:
            with log_path.open("a") as log:
                log_path.chmod(0o600)
                for role in ("relay", "recover", "index"):
                    self.processes.append(subprocess.Popen([sys.executable, "-I", "-m", "codex_agent.workers",
                        role, "--workspace", str(root)], env=env, cwd=root, stdout=log, stderr=log,
                        start_new_session=True))
        except BaseException:
            self.close()
            raise

    def monitor(self):
        if self.processes and any(p.poll() is not None for p in self.processes):
            self.close()
            self.start()

    def close(self):
        processes, self.processes = self.processes, []
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
