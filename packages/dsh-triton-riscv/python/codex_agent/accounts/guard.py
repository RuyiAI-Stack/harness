"""Trusted process identity, never a browser field or an LLM tool argument."""
import hashlib
import os
from pathlib import Path
import re

from sqlalchemy import select

OWNER_ENV = "TRITON_WORKBENCH_OWNER"
GATEWAY_ENV = "TRITON_WORKBENCH_GATEWAY_KEY"
MARKER = ".triton-account"


def owner_id():
    value = os.environ.get(OWNER_ENV, "")
    if value and not re.fullmatch(r"[a-f0-9]{32}", value):
        raise PermissionError("Invalid account process identity")
    return value


def check_workspace(root, connection=None):
    root = Path(root).resolve()
    owner = owner_id()
    marker = root / MARKER
    if not owner and not marker.exists():
        return
    if not owner or marker.is_symlink() or not marker.is_file() or marker.read_text().strip() != owner:
        raise PermissionError("Account workspace requires its authenticated service identity")
    from .schema import users, workspaces
    workspace_id = hashlib.sha256(str(root).encode()).hexdigest()
    query = select(users.c.id).join(workspaces, workspaces.c.user_id == users.c.id).where(
        users.c.id == owner, users.c.status == "active", workspaces.c.workspace_id == workspace_id,
        workspaces.c.root == str(root), workspaces.c.status == "ready")
    if connection is None:
        from codex_agent.storage.database import engine
        with engine().connect() as conn:
            allowed = conn.execute(query).scalar_one_or_none()
    else:
        allowed = connection.execute(query).scalar_one_or_none()
    if not allowed:
        raise PermissionError("Account workspace unavailable or access revoked")


def reviewer_identity(requested):
    owner = owner_id()
    return "account:" + owner if owner else requested
