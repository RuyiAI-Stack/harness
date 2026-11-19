"""Immutable links from historical sources to editable shared documents."""
from sqlalchemy import Column, ForeignKeyConstraint, String
from codex_agent.storage.schema import table, timestamp

source_links = table("knowledge_source_links",
    Column("origin_key", String(64), primary_key=True),
    Column("source_alias", String(48), nullable=False),
    Column("source_scope", String(64), nullable=False),
    Column("source_kind", String(32), nullable=False),
    Column("source_id", String(128), nullable=False),
    Column("source_version", String(128), nullable=False),
    Column("source_revision", String(256), nullable=False),
    Column("source_content_hash", String(64), nullable=False),
    Column("document_id", String(32), nullable=False),
    Column("actor_id", String(64), nullable=False),
    timestamp("created_at"),
    ForeignKeyConstraint(["workspace_id", "document_id"],
        ["knowledge_documents.workspace_id", "knowledge_documents.id"]))
