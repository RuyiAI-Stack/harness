"""MySQL authority for desired index versions and transactional indexing jobs."""
import hashlib
import json
from pathlib import Path

from sqlalchemy import select, func, exists, and_

from codex_agent.runtime_config import runtime_config
from codex_agent.storage.database import WorkspaceDatabase
from codex_agent.storage.schema import memories, memory_indexes as indexes, memory_index_entries as entries
from codex_agent.tasks.store import JobStore, Conflict, encode, digest
from codex_agent.tasks.schema import jobs
from codex_agent.memory_chunking import ChunkPolicy
from codex_agent.memory_lineage import SPLITTER_VERSION


class IndexStore:
    def __init__(self, root, *, database=None):
        self.db = database or WorkspaceDatabase(root)
        if database is None:
            self.db.register()
        self.jobs = JobStore(root, database=self.db)

    def prepare(self, conn):
        cfg = runtime_config().indexing
        if not cfg.enabled:
            raise ValueError("indexing is disabled")
        definition = cfg.model_dump(exclude={"enabled", "generation"})
        rows = self.db.rows(indexes, indexes.c.generation == cfg.generation, connection=conn)
        path = cfg.embedding.tokenizerJson
        # Subsequent source writes do not repeatedly read a potentially large tokenizer.
        # Workers verify the frozen checksum before using it.
        definition["tokenizer_sha256"] = (json.loads(rows[0]["config_json"])["tokenizer_sha256"] if rows
            else hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)
        definition["splitter"] = SPLITTER_VERSION
        definition["maxChars"] = ChunkPolicy().max_chars
        fingerprint = digest(definition)
        if rows:
            if rows[0]["fingerprint"] != fingerprint:
                raise Conflict("index definition changed; choose a new generation")
            if rows[0]["status"] == "retired":
                raise Conflict("generation is retired; choose a new generation")
            return rows[0]
        row = dict(generation=cfg.generation, status="staging", config_json=encode(definition),
                   fingerprint=fingerprint, collection="triton_" + digest([
                       self.db.cache_namespace, self.db.workspace_id, cfg.generation, fingerprint]))
        self.db.insert(conn, indexes, row)
        return row

    def sync(self, conn, memory_id, *, retry=False, generation=None):
        """Caller holds memory-write. No network or model work inside this transaction."""
        self.prepare(conn)
        rows = self.db.rows(memories, memories.c.id == memory_id, connection=conn)
        row = rows[0] if rows else None
        desired = "present" if row and row["active"] else "deleted"
        version = row["case_version"] if row else None
        if desired == "present" and not version:
            raise ValueError("snapshot required before indexing")
        submitted = 0
        conditions = (indexes.c.generation == generation,) if generation else ()
        for index in self.db.rows(indexes, *conditions, connection=conn):
            if index["status"] == "retired" and desired == "present":
                continue
            generation = index["generation"]
            old = self.db.rows(entries, entries.c.generation == generation,
                               entries.c.memory_id == memory_id, connection=conn)
            if old and old[0]["desired"] == desired and old[0]["case_version"] == version:
                previous = self.jobs._row(conn, jobs, old[0]["job_id"])
                if not retry or previous["status"] not in {"failed", "dead", "cancelled"}:
                    continue
            if not old and desired == "deleted":
                continue
            revision = old[0]["revision"] + 1 if old else 1
            payload = dict(generation=generation, memory_id=memory_id, revision=revision,
                           case_version=version, desired=desired, index_fingerprint=index["fingerprint"])
            job = self.jobs.submit("index", payload, "index:" + digest(payload), connection=conn)
            value = dict(revision=revision, case_version=version, desired=desired,
                         status="pending", job_id=job["id"], point_count=0)
            if old:
                self.db.update(conn, entries, value, entries.c.generation == generation,
                               entries.c.memory_id == memory_id)
            else:
                self.db.insert(conn, entries, dict(generation=generation, memory_id=memory_id, **value))
            submitted += 1
        return submitted

    def backfill_batch(self, *, cursor=None, page_size=100, retry=False):
        """Bounded enqueue only; replay is safe, workers perform model/network work.

        The cursor freezes ID upper bounds, not source contents. Each page indexes
        the latest source revision; concurrent writes still use incremental sync.
        """
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("page_size must be an integer between 1 and 100")
        from codex_agent.memory import MemoryStore
        memory = MemoryStore(self.db.root)
        with self.db.transaction() as conn:
            self.db.lock(conn, "memory-write")
            index = self.prepare(conn)
            scope = dict(schema=1, workspace=self.db.workspace_id, database=self.db.cache_namespace,
                         generation=index["generation"], fingerprint=index["fingerprint"], retry=retry)
            if cursor is None:
                source_until = conn.execute(select(func.max(memories.c.id)).where(
                    *self.db.predicate(memories))).scalar_one() or 0
                orphan_until = conn.execute(select(func.max(entries.c.memory_id)).where(
                    *self.db.predicate(entries, entries.c.generation == index["generation"]))).scalar_one() or 0
                state = dict(**scope, phase="sources", after=0,
                             source_until=source_until, orphan_until=orphan_until)
            else:
                state = dict(cursor) if isinstance(cursor, dict) else {}
                fields = {"phase", "after", "source_until", "orphan_until"}
                if (set(state) != set(scope) | fields
                    or any(state.get(k) != v or type(state.get(k)) is not type(v) for k, v in scope.items())
                    or state.get("phase") not in {"sources", "orphans"}
                    or any(type(state.get(k)) is not int or state[k] < 0
                           for k in ("after", "source_until", "orphan_until"))):
                    raise ValueError("invalid cursor or different workspace/index/retry policy")
                if state["after"] > state["source_until" if state["phase"] == "sources" else "orphan_until"]:
                    raise ValueError("cursor exceeds its upper bound")
            phase = state["phase"]
            if phase == "sources":
                rows = self.db.rows(memories, memories.c.id > state["after"],
                                    memories.c.id <= state["source_until"], order=(memories.c.id,),
                                    limit=page_size, connection=conn)
                identifiers = [row["id"] for row in rows]
                for row in rows:
                    memory._snapshot_case(conn, row)
            else:
                missing = ~exists(select(memories.c.id).where(
                    memories.c.workspace_id == entries.c.workspace_id, memories.c.id == entries.c.memory_id))
                identifiers = conn.execute(select(entries.c.memory_id).where(*self.db.predicate(entries,
                    entries.c.generation == index["generation"], entries.c.memory_id > state["after"],
                    entries.c.memory_id <= state["orphan_until"], missing))
                    .order_by(entries.c.memory_id).limit(page_size)).scalars().all()
            submitted = sum(self.sync(conn, mid, retry=retry, generation=index["generation"])
                            for mid in identifiers)
            if identifiers:
                state["after"] = identifiers[-1]
            done = phase == "orphans" and len(identifiers) < page_size
            if phase == "sources" and len(identifiers) < page_size:
                state.update(phase="orphans", after=0)
        return dict(generation=index["generation"], phase=phase, scanned=len(identifiers),
                    submitted=submitted, done=done, next_cursor=None if done else state,
                    retrieval_changed=False)

    def backfill(self, *, retry=False, page_size=100):
        from codex_agent.memory import MemoryStore
        memory = MemoryStore(self.db.root)
        total, cursor = 0, 0
        with self.db.transaction() as conn:
            self.db.lock(conn, "memory-write")
            self.prepare(conn)
        while True:
            with self.db.transaction() as conn:
                self.db.lock(conn, "memory-write")
                rows = self.db.rows(memories, memories.c.id > cursor, order=(memories.c.id,),
                                    limit=page_size, connection=conn)
                for row in rows:
                    memory._snapshot_case(conn, row)
                    total += self.sync(conn, row["id"], retry=retry)
                if rows:
                    cursor = rows[-1]["id"]
            if len(rows) < page_size:
                break
        # A source deleted outside MemoryStore still has a ledger entry to clean up.
        cursor = 0
        while True:
            with self.db.transaction() as conn:
                self.db.lock(conn, "memory-write")
                missing = ~exists(select(memories.c.id).where(
                    memories.c.workspace_id == entries.c.workspace_id, memories.c.id == entries.c.memory_id))
                ids = conn.execute(select(entries.c.memory_id).where(*self.db.predicate(entries,
                    entries.c.memory_id > cursor, missing)).distinct().order_by(entries.c.memory_id).limit(page_size)).scalars().all()
                for identifier in ids:
                    total += self.sync(conn, identifier, retry=retry)
                if ids:
                    cursor = ids[-1]
            if len(ids) < page_size:
                break
        return {"submitted": total, "retrieval_changed": False}

    def missing_count(self, conn, generation):
        ready = exists(select(entries.c.memory_id).where(
            entries.c.workspace_id == memories.c.workspace_id,
            entries.c.generation == generation, entries.c.memory_id == memories.c.id,
            entries.c.case_version == memories.c.case_version,
            entries.c.desired == "present", entries.c.status == "ready"))
        return conn.execute(select(func.count()).select_from(memories).where(
            *self.db.predicate(memories, memories.c.active == 1, ~ready))).scalar_one()

    def activate(self, generation):
        with self.db.transaction() as conn:
            self.db.lock(conn, "memory-write")
            rows = self.db.rows(indexes, indexes.c.generation == generation, connection=conn)
            if not rows or rows[0]["status"] == "retired":
                raise ValueError("unknown or retired generation")
            if self.missing_count(conn, generation):
                raise Conflict("current active cases are not fully indexed; inspect status and backfill")
            self.db.update(conn, indexes, {"status": "staging"}, indexes.c.status == "active")
            self.db.update(conn, indexes, {"status": "active"}, indexes.c.generation == generation)
        return {"active_generation": generation, "retrieval_changed": False}

    def retire(self, generation):
        with self.db.transaction() as conn:
            self.db.lock(conn, "memory-write")
            rows = self.db.rows(indexes, indexes.c.generation == generation, connection=conn)
            if not rows:
                raise ValueError("unknown index generation")
            if rows[0]["status"] == "active" or generation == runtime_config().indexing.generation:
                raise Conflict("activate/configure the replacement before retiring this generation")
            self.db.update(conn, indexes, {"status": "retired"}, indexes.c.generation == generation)
        return {"retired_generation": generation, "vectors_and_audit_preserved": True}

    def status(self):
        result = []
        with self.db.transaction() as conn:
            for index in self.db.rows(indexes, connection=conn):
                states = conn.execute(select(entries.c.status, jobs.c.status.label("job_status"), func.count().label("count"))
                    .select_from(entries.join(jobs, and_(entries.c.workspace_id == jobs.c.workspace_id,
                                                        entries.c.job_id == jobs.c.id)))
                    .where(*self.db.predicate(entries, entries.c.generation == index["generation"]))
                    .group_by(entries.c.status, jobs.c.status)).mappings().all()
                result.append({"generation": index["generation"], "status": index["status"],
                    "collection": index["collection"], "missing_active_cases": self.missing_count(conn, index["generation"]),
                    "states": [dict(row) for row in states]})
        return {"indexes": result, "retrieval_changed": False}


def schedule(db, conn, memory_id):
    if runtime_config().indexing.enabled:
        IndexStore(db.root, database=db).sync(conn, memory_id)
