"""Recycle-bin authority and generation fences for in-flight index/parse jobs."""
from sqlalchemy import Column, ForeignKeyConstraint, Integer, String
from codex_agent.storage.schema import table

controls = table("knowledge_document_controls", Column("document_id", String(32), primary_key=True),
    Column("revision", Integer, nullable=False), Column("deleted_at", String(40)),
    Column("actor_id", String(64), nullable=False),
    ForeignKeyConstraint(["workspace_id", "document_id"],
                         ["knowledge_documents.workspace_id", "knowledge_documents.id"]))


class DocumentUnavailable(KeyError):
    pass


def control(store, conn, identifier):
    return next(iter(store.rows(conn, controls, controls.c.document_id == identifier)),
                dict(revision=0, deleted_at=None))


def require_active(store, conn, identifier, revision=None):
    current = control(store, conn, identifier)
    if current["deleted_at"] or (revision is not None and current["revision"] != revision):
        raise DocumentUnavailable("Document was deleted or its lifecycle changed")
    return current["revision"]
