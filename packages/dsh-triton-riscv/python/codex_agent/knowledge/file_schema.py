"""File provenance is separate from immutable document versions and vector data."""
from sqlalchemy import Column, ForeignKeyConstraint, Integer, String
from codex_agent.storage.schema import table, text

sources = table("knowledge_file_sources", Column("document_id", String(32), primary_key=True),
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("file_id", String(32), nullable=False), Column("filename", String(240), nullable=False),
    Column("format", String(8), nullable=False), Column("sha256", String(64), nullable=False),
    Column("parser_version", String(40), nullable=False), Column("reviewed", Integer, nullable=False),
    text("sections_json"), text("warnings_json"),
    ForeignKeyConstraint(["workspace_id", "document_id", "version"],
        ["knowledge_versions.workspace_id", "knowledge_versions.document_id", "knowledge_versions.version"]))
locations = table("knowledge_chunk_locations", Column("document_id", String(32), primary_key=True),
    Column("version", Integer, primary_key=True, autoincrement=False),
    Column("chunk_id", String(64), primary_key=True), text("locator"),
    ForeignKeyConstraint(["workspace_id", "document_id", "version"],
        ["knowledge_versions.workspace_id", "knowledge_versions.document_id", "knowledge_versions.version"]))
TABLES = (sources, locations)
