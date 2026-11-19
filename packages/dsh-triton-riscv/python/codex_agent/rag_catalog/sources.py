"""Explicit historical database sources. Reads never register, migrate or index."""
from functools import lru_cache
import hashlib
import hmac
import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from codex_agent.accounts.guard import MARKER
from codex_agent.accounts.schema import users, workspaces as account_workspaces
from codex_agent.storage.database import StorageError
from codex_agent.storage.schema import workspaces


class HistoryUnavailable(StorageError):
    pass


class HistoryForbidden(PermissionError):
    pass


def configured_sources(config):
    return {"default_source_id": "current", "origin_filters": [
        {"id": "uploaded", "label": "新增 / 上传"},
        *[{"id": "history:" + s.id, "label": s.label.split(" · ", 1)[0]}
          for s in config.rag_history_sources],
    ], "sources": [
        {"id": "current", "label": "共享知识库", "read_only": False, "status": "configured", "kind": "current"},
        *[{"id": s.id, "label": s.label, "read_only": True, "status": "configured", "kind": "memory_case"}
          for s in config.rag_history_sources],
    ]}


def source_config(config, source_id):
    found = next((s for s in config.rag_history_sources if s.id == source_id), None)
    if found is None:
        raise KeyError("Source is not registered")
    return found


def source_signing_key(master, source_id, connection_identity=""):
    # Do not reuse a token across databases even when roots/source IDs match.
    binding = "rag-source:" + source_id + ":" + hashlib.sha256(connection_identity.encode()).hexdigest()
    return hmac.new(master, binding.encode(), hashlib.sha256).digest()


def _readonly_session(connection, _record=None, _proxy=None):
    # Reapply on every checkout; a pooled connection never inherits writable mode.
    connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute("SET SESSION TRANSACTION READ ONLY")


@lru_cache(maxsize=16)
def history_engine(raw, pid):
    del pid  # part of the cache identity, so a fork never reuses a parent pool.
    try:
        url = make_url(raw)
        if url.drivername != "mysql+pymysql" or not url.database:
            raise ValueError()
        if set(url.query) - {"unix_socket", "ssl_ca", "ssl_cert", "ssl_key", "ssl_check_hostname"}:
            raise ValueError()
        engine = create_engine(url, pool_size=2, max_overflow=0, pool_timeout=5,
            pool_pre_ping=True, pool_recycle=1800, isolation_level="READ COMMITTED", hide_parameters=True,
            connect_args={"charset": "utf8mb4", "connect_timeout": 5, "read_timeout": 10,
                "write_timeout": 10, "local_infile": False,
                "init_command": "SET SESSION TRANSACTION READ ONLY"})
        event.listen(engine, "checkout", _readonly_session)
        return engine
    except (ValueError, TypeError, SQLAlchemyError):
        raise HistoryUnavailable("Historical source connection configuration is unavailable") from None


def historical_scopes(source, engine):
    """Validate every explicitly configured workspace; never silently omit one."""
    with engine.connect() as conn:
        rows = list(conn.execute(select(workspaces).where(
            workspaces.c.workspace_id.in_(source.workspace_ids))).mappings())
        if {r["workspace_id"] for r in rows} != set(source.workspace_ids):
            raise HistoryUnavailable("A configured historical workspace is missing")
        inspector = inspect(conn)
        account_table = inspector.has_table(account_workspaces.name)
        user_table = inspector.has_table(users.name)
        if account_table != user_table:
            raise HistoryUnavailable("Historical account metadata is incomplete")
        linked = {}
        if account_table:
            linked = {r["workspace_id"]: dict(r) for r in conn.execute(
                select(account_workspaces.c.workspace_id, account_workspaces.c.user_id,
                    account_workspaces.c.root, account_workspaces.c.status,
                    users.c.status.label("owner_status"))
                .outerjoin(users, users.c.id == account_workspaces.c.user_id)
                .where(account_workspaces.c.workspace_id.in_(source.workspace_ids))).mappings()}
        scopes = []
        for row in rows:
            stored_root = Path(row["root"])
            if not stored_root.is_absolute():
                raise HistoryUnavailable("Historical workspace identity does not match")
            root = stored_root.resolve()
            if hashlib.sha256(str(root).encode()).hexdigest() != row["workspace_id"]:
                raise HistoryUnavailable("Historical workspace identity does not match")
            marker = root / MARKER
            account = linked.get(row["workspace_id"])
            if account is not None or marker.exists() or marker.is_symlink():
                if not source.owner_user_id or account is None:
                    raise HistoryForbidden("Historical account owner must be explicitly granted")
                if (account["user_id"] != source.owner_user_id or account["owner_status"] != "active"
                        or account["status"] != "ready" or account["root"] != str(root)):
                    raise HistoryForbidden("Historical account scope unavailable or revoked")
                if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 128:
                    raise HistoryForbidden("Historical account marker unavailable")
                if marker.read_text().strip() != source.owner_user_id:
                    raise HistoryForbidden("Historical account marker does not match")
            elif source.owner_user_id:
                # An asserted owner without an account registry cannot be verified.
                raise HistoryForbidden("Historical account owner cannot be verified")
            scopes.append({"id": row["workspace_id"], "root": root, "label": source.label + " · " + row["workspace_id"][:8],
                           "kind": "memory_case", "writable": False})
        return scopes


def open_history(config, source_id, master):
    source = source_config(config, source_id)
    raw = os.environ.get(source.url_env, "")
    if not raw:
        raise HistoryUnavailable("Historical source environment is not configured")
    try:
        engine = history_engine(raw, os.getpid())
        scopes = historical_scopes(source, engine)
        return source, engine, scopes, source_signing_key(master, source.id, raw)
    except HistoryForbidden:
        raise
    except (SQLAlchemyError, OSError, UnicodeError):
        raise HistoryUnavailable("Historical source is unreachable or its metadata cannot be read") from None
