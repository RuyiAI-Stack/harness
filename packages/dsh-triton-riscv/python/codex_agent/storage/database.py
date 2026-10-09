"""Bounded MySQL pools and workspace-scoped SQLAlchemy Core operations."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import os
from pathlib import Path

from sqlalchemy import create_engine, select, update, delete, text
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from codex_agent.runtime_config import runtime_config
from .schema import counters, workspaces


class StorageError(RuntimeError):
    pass


_read_connection = ContextVar("cache_read_connection", default=None)


@lru_cache(maxsize=8)
def _engine(raw, pool_size, overflow, timeout, pid):
    try:
        url = make_url(raw)
        if url.drivername != "mysql+pymysql" or not url.database:
            raise ValueError()
        return create_engine(url, pool_size=pool_size, max_overflow=overflow,
                             pool_timeout=timeout, pool_pre_ping=True, pool_recycle=1800,
                             isolation_level="READ COMMITTED", hide_parameters=True,
                             connect_args={"charset": "utf8mb4", "connect_timeout": 5,
                                           "read_timeout": 30, "write_timeout": 30})
    except (ValueError, SQLAlchemyError):
        raise StorageError("Invalid MySQL URL; use mysql+pymysql and a database name") from None


def engine():
    cfg = runtime_config().storage
    raw = os.environ.get(cfg.urlEnv, "")
    if not raw:
        raise StorageError(f"Set {cfg.urlEnv} to the MySQL connection URL, then run codex_agent.storage upgrade")
    return _engine(raw, cfg.poolSize, cfg.maxOverflow, cfg.poolTimeout, os.getpid())


class WorkspaceDatabase:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.workspace_id = hashlib.sha256(str(self.root).encode()).hexdigest()
        self.engine = engine()
        self.task_guard = None
        self.cache_namespace = hashlib.sha256(
            self.engine.url.render_as_string(hide_password=True).encode()).hexdigest()[:24]

    def predicate(self, table, *conditions):
        return (table.c.workspace_id == self.workspace_id, *conditions)

    @contextmanager
    def transaction(self):
        borrowed = _read_connection.get()
        if borrowed and borrowed[0] is self.engine:
            try:
                yield borrowed[1]
            except SQLAlchemyError:
                raise StorageError("MySQL cached read failed; check service and schema") from None
            return
        try:
            with self.engine.begin() as connection:
                connection.info["cache_changes"] = set()
                try:
                    if self.task_guard is not None:
                        self.task_guard(connection)
                    yield connection
                    for workspace, domain in sorted(connection.info["cache_changes"]):
                        stmt = insert(counters).values(workspace_id=workspace,
                            name="cache-version:" + domain, value=1)
                        connection.execute(stmt.on_duplicate_key_update(value=counters.c.value + 1))
                finally:
                    connection.info.pop("cache_changes", None)
        except SQLAlchemyError:
            # Driver exceptions may include a username, host, SQL parameters or credentials.
            raise StorageError("MySQL operation failed; check service, schema migration and constraints") from None

    def register(self):
        with self.transaction() as conn:
            stmt = insert(workspaces).values(workspace_id=self.workspace_id, root=str(self.root))
            conn.execute(stmt.on_duplicate_key_update(root=stmt.inserted.root))

    def rows(self, table, *conditions, order=(), connection=None, limit=None, lock=False):
        stmt = select(table).where(*self.predicate(table, *conditions)).order_by(*order)
        if limit is not None:
            stmt = stmt.limit(limit)
        if lock:
            stmt = stmt.with_for_update()
        if connection is None:
            with self.transaction() as conn:
                return [dict(row) for row in conn.execute(stmt).mappings()]
        return [dict(row) for row in connection.execute(stmt).mappings()]

    def insert(self, conn, table, values):
        conn.execute(table.insert().values(**{**values, "workspace_id": self.workspace_id}))
        self._changed(conn, table)

    def update(self, conn, table, values, *conditions):
        result = conn.execute(update(table).where(*self.predicate(table, *conditions)).values(**values))
        if result.rowcount and set(values) != {"last_used_at"}:
            self._changed(conn, table)
        return result

    def delete(self, conn, table, *conditions):
        result = conn.execute(delete(table).where(*self.predicate(table, *conditions)))
        if result.rowcount:
            self._changed(conn, table)
        return result

    def _changed(self, conn, table):
        domain = {"memories": "memory", "memory_chunks": "memory",
                  "references_catalog": "references"}.get(table.name)
        if domain:
            if "cache_changes" not in conn.info:
                raise StorageError("Cached data writes require WorkspaceDatabase.transaction")
            conn.info["cache_changes"].add((self.workspace_id, domain))

    def revision(self, domain):
        rows = self.rows(counters, counters.c.name == "cache-version:" + domain)
        return rows[0]["value"] if rows else 0

    @contextmanager
    def cache_read_slot(self, limit):
        """Server-wide, connection-owned slots survive Redis loss, not process loss."""
        from .cache import CacheBusy
        acquired = None
        token = None
        try:
            with self.engine.connect() as conn:
                try:
                    for slot in range(limit):
                        name = f"triton-cache:{self.cache_namespace}:{slot}"
                        if conn.execute(text("SELECT GET_LOCK(:name, 0)"), {"name": name}).scalar() == 1:
                            acquired = name
                            break
                    if acquired is None:
                        raise CacheBusy("Historical retrieval is busy; retry later")
                    token = _read_connection.set((self.engine, conn))
                    yield
                finally:
                    if token is not None:
                        _read_connection.reset(token)
                    if acquired:
                        conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": acquired})
        except SQLAlchemyError:
            raise StorageError("MySQL cache read admission failed") from None

    def lock(self, conn, name):
        stmt = insert(counters).values(workspace_id=self.workspace_id, name=name, value=0)
        conn.execute(stmt.on_duplicate_key_update(value=counters.c.value))
        return conn.execute(select(counters.c.value).where(
            *self.predicate(counters, counters.c.name == name)).with_for_update()).scalar_one()

    def allocate(self, conn, name):
        value = self.lock(conn, name) + 1
        self.update(conn, counters, {"value": value}, counters.c.name == name)
        return value
