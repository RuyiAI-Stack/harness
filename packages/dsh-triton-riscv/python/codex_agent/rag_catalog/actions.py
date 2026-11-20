"""Explicit memory lifecycle/correction writes, with durable idempotent receipts."""
import hashlib
import json
import secrets

from sqlalchemy import select

from codex_agent.accounts.guard import admin_workspace
from codex_agent.accounts.schema import users
from codex_agent.memory import MemoryStore, normalize_text
from codex_agent.platform.store import timestamp
from codex_agent.storage.schema import meta, memories
from codex_agent.tasks.store import Conflict, digest, encode


def operation_key(actor, request_id):
    return "rag-op:" + hashlib.sha256(encode([actor, request_id]).encode()).hexdigest()


def operation(catalog, actor, request_id, item_id):
    scope, kind, source_id = catalog.resolve(item_id)
    if kind != "memory_case":
        raise ValueError("Memory receipt requires a memory item")
    with catalog.engine.connect() as conn:
        rows = conn.execute(select(meta.c.value).where(meta.c.workspace_id == scope["id"],
            meta.c.key == operation_key(actor, request_id)).limit(2)).scalars().all()
    if len(rows) != 1:
        return {"status": "unknown"}
    record = json.loads(rows[0])
    if record["source_id"] != source_id:
        return {"status": "unknown"}
    return {"status": "committed", "item_id": catalog.item_id(record["scope_id"], "memory_case", record["source_id"]),
            "action": record["action"]}


def corrections(catalog, item_id):
    scope, kind, source_id = catalog.resolve(item_id)
    if kind != "memory_case":
        return []
    with catalog.engine.connect() as conn:
        rows = conn.execute(select(meta.c.value).where(meta.c.workspace_id == scope["id"],
            meta.c.key.startswith("rag-note:" + source_id + ":", autoescape=True)).order_by(meta.c.key).limit(50)).scalars()
        return [json.loads(value) for value in rows]


def execute(catalog, item_id, value, actor):
    scope, kind, source_id = catalog.resolve(item_id)
    if kind != "memory_case" or not scope["writable"]:
        raise PermissionError("This source does not permit memory actions")
    # Bind to the durable identity rather than a process-local signed item token.
    fingerprint = digest([scope["id"], kind, source_id, value.model_dump()])
    key = operation_key(actor, value.request_id)
    with admin_workspace(scope["root"], actor):
        with MemoryStore(scope["root"]) as store:
            db = store.db
            with db.transaction() as conn:
                if not conn.execute(select(users.c.id).where(users.c.id == actor, users.c.status == "active")).scalar_one_or_none():
                    raise PermissionError("Administrator access revoked")
                db.lock(conn, "memory-write")
                prior = db.rows(meta, meta.c.key == key, connection=conn)
                if prior:
                    if json.loads(prior[0]["value"])["fingerprint"] != fingerprint:
                        raise Conflict("Request ID was already used with a different action")
                    return
                if catalog.revision_for(conn, item_id) != value.expected_revision:
                    raise Conflict("资料已更新，请重新读取后再操作")
                rows = db.rows(memories, memories.c.id == int(source_id), connection=conn, lock=True)
                if not rows:
                    raise KeyError("Memory not found")
                row = rows[0]
                if value.action == "archive":
                    if not row["active"]:
                        raise Conflict("此经验已停用，请重新读取")
                    db.update(conn, memories, dict(active=0, archive_reason=normalize_text(value.reason),
                        updated_at=timestamp()), memories.c.id == int(source_id))
                    store._queue_index(conn, int(source_id))
                elif value.action == "correct":
                    previous = db.rows(meta, meta.c.key.startswith("rag-note:" + source_id + ":", autoescape=True),
                                       connection=conn, limit=50)
                    if len(previous) >= 50:
                        raise Conflict("该资料的纠错说明已达上限，请先审阅现有说明")
                    note = dict(id=secrets.token_hex(16), content=value.content.strip(), created_at=timestamp(),
                                actor_id=actor, source_version=row["case_version"], state="pending_review")
                    db.insert(conn, meta, dict(key="rag-note:" + source_id + ":" + note["id"], value=encode(note)))
                else:
                    raise ValueError("Unsupported memory action")
                record = dict(fingerprint=fingerprint, actor_id=actor, scope_id=scope["id"], source_id=source_id,
                              action=value.action, created_at=timestamp())
                db.insert(conn, meta, dict(key=key, value=encode(record)))
