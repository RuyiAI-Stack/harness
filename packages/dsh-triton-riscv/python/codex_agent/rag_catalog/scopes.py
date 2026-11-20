"""The current admin catalog manages shared knowledge, never account memory."""
import hashlib
from pathlib import Path


def authorized_scopes(config, accounts, user):
    if not config.knowledge_enabled:
        return []
    root = (Path(config.data_dir) / "knowledge").resolve()
    return [dict(id=hashlib.sha256(str(root).encode()).hexdigest(), label="共享知识库",
                 root=root, kind="knowledge_document", writable=True)]
