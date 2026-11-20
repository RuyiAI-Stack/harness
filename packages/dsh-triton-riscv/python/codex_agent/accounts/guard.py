"""Trusted process identity, never a browser field or an LLM tool argument."""
import hashlib
import os
from pathlib import Path
import re
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import select

OWNER_ENV = "TRITON_WORKBENCH_OWNER"
GATEWAY_ENV = "TRITON_WORKBENCH_GATEWAY_KEY"
MARKER = ".triton-account"
_admin_scope = ContextVar("explicit_admin_workspace", default=None)


@contextmanager
def admin_workspace(root, actor_id):
    """Gateway-only capability after route authorization, never a model argument.

    The exact root is bound to one request context. Do not impersonate users by
    changing process-wide environment variables while other requests run.
    """
    if owner_id() or not re.fullmatch(r"[a-f0-9]{32}", actor_id):
        raise PermissionError("Only the authenticated gateway can administer a workspace")
    token = _admin_scope.set((str(Path(root).resolve()), actor_id))
    try:
        yield
    finally:
        _admin_scope.reset(token)


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
    admitted = _admin_scope.get()
    if not owner and admitted and admitted[0] == str(root):
        from .schema import users, workspaces
        workspace_id = hashlib.sha256(str(root).encode()).hexdigest()
        query = select(users.c.id).join(workspaces, workspaces.c.user_id == users.c.id).where(
            users.c.status == "active", workspaces.c.workspace_id == workspace_id,
            workspaces.c.root == str(root), workspaces.c.status == "ready")
        actor = select(users.c.id).where(users.c.id == admitted[1], users.c.status == "active")
        if marker.is_symlink() or not marker.is_file():
            raise PermissionError("Workspace marker unavailable")
        def verify(conn):
            uid = conn.execute(query).scalar_one_or_none()
            return uid and marker.read_text().strip() == uid and conn.execute(actor).scalar_one_or_none()
        if connection is None:
            from codex_agent.storage.database import engine
            with engine().connect() as conn:
                allowed = verify(conn)
        else:
            allowed = verify(connection)
        if not allowed:
            raise PermissionError("Workspace unavailable or administrator revoked")
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
