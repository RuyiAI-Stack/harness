"""Versioned document authority. Vectors never grant publication permission."""
from sqlalchemy import Column, ForeignKeyConstraint, Index, Integer, String
from codex_agent.storage.schema import table, text, timestamp

documents = table("knowledge_documents", Column("id", String(32), primary_key=True),
    Column("latest_version", Integer, nullable=False), Column("published_version", Integer),
    timestamp("created_at"), timestamp("updated_at"))
versions = table("knowledge_versions", Column("document_id", String(32), primary_key=True),
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("title", String(200), nullable=False), text("source"), text("content"),
    Column("source_hash", String(64), nullable=False), Column("state", String(24), nullable=False),
    text("definition_json"), Column("fingerprint", String(64)), Column("job_id", String(64)),
    Column("chunk_count", Integer, nullable=False), Column("error", String(255)), timestamp("created_at"),
    ForeignKeyConstraint(["workspace_id", "document_id"],
                         ["knowledge_documents.workspace_id", "knowledge_documents.id"]))
chunks = table("knowledge_chunks", Column("document_id", String(32), primary_key=True),
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("id", String(64), primary_key=True), Column("position", Integer, nullable=False), text("text"),
    ForeignKeyConstraint(["workspace_id", "document_id", "version"],
                         ["knowledge_versions.workspace_id", "knowledge_versions.document_id", "knowledge_versions.version"]))
requests = table("knowledge_requests", Column("id", String(64), primary_key=True),
    Column("fingerprint", String(64), nullable=False), Column("document_id", String(32), nullable=False),
    Column("version", Integer, nullable=False))
audit = table("knowledge_audit", Column("id", String(32), primary_key=True),
    Column("actor_id", String(64), nullable=False), Column("action", String(40), nullable=False),
    Column("document_id", String(32)), Column("version", Integer), timestamp("created_at"))
Index("idx_knowledge_published", documents.c.workspace_id, documents.c.published_version)
TABLES = (documents, versions, chunks, requests, audit)
