"""Current MySQL schema. Every domain key is scoped to a canonical workspace."""
from sqlalchemy import (BigInteger, Column, Float, ForeignKeyConstraint, Index,
                        Integer, MetaData, String, Table, UniqueConstraint)
from sqlalchemy.dialects.mysql import LONGTEXT

metadata = MetaData()


def table(name, *columns):
    return Table(name, metadata, Column("workspace_id", String(64), primary_key=True),
                 *columns, mysql_engine="InnoDB", mysql_charset="utf8mb4",
                 mysql_collate="utf8mb4_bin")


def text(name, nullable=False, default=None):
    return Column(name, LONGTEXT, nullable=nullable, server_default=default)


def timestamp(name):
    # Preserve historical timestamp text verbatim, including its timezone.
    return Column(name, String(40), nullable=False)


workspaces = table("workspaces", text("root"))
counters = table("counters", Column("name", String(100), primary_key=True),
                 Column("value", BigInteger, nullable=False))
meta = table("metadata", Column("key", String(100), primary_key=True), text("value"))
sessions = table("sessions", Column("id", String(64), primary_key=True),
                 text("title"), Column("status", String(64), nullable=False),
                 timestamp("created_at"), timestamp("updated_at"),
                 Column("pinned", Integer, nullable=False, server_default="0"))
messages = table("messages", Column("id", String(64), primary_key=True),
                 Column("session_id", String(64), nullable=False),
                 Column("ordinal", BigInteger, nullable=False),
                 Column("role", String(64), nullable=False), text("content"),
                 text("metadata_json"), timestamp("created_at"),
                 ForeignKeyConstraint(["workspace_id", "session_id"],
                                      ["sessions.workspace_id", "sessions.id"], ondelete="CASCADE"))
runs = table("runs", Column("id", String(64), primary_key=True),
             Column("session_id", String(64), nullable=False), text("intent"),
             Column("operator", String(255)), Column("status", String(64), nullable=False),
             Column("phase", String(64), nullable=False), text("request_json"), text("result_json"),
             timestamp("created_at"), timestamp("updated_at"),
             Column("next_sequence", BigInteger, nullable=False, server_default="0"),
             ForeignKeyConstraint(["workspace_id", "session_id"],
                                  ["sessions.workspace_id", "sessions.id"], ondelete="CASCADE"))
events = table("events", Column("id", BigInteger, primary_key=True, autoincrement=False),
               Column("run_id", String(64), nullable=False),
               Column("sequence", BigInteger, nullable=False),
               Column("event_type", String(100), nullable=False), text("payload_json"),
               timestamp("created_at"),
               UniqueConstraint("workspace_id", "run_id", "sequence"),
               ForeignKeyConstraint(["workspace_id", "run_id"],
                                    ["runs.workspace_id", "runs.id"], ondelete="CASCADE"))
checkpoints = table("context_checkpoints", Column("session_id", String(64), primary_key=True),
                    text("summary"), Column("through_message_id", String(64)),
                    text("source_ids_json"), text("metrics_json"), timestamp("updated_at"),
                    ForeignKeyConstraint(["workspace_id", "session_id"],
                                         ["sessions.workspace_id", "sessions.id"], ondelete="CASCADE"))
memories = table("memories", Column("id", BigInteger, primary_key=True, autoincrement=False),
                 Column("fingerprint", String(64), nullable=False),
                 Column("memory_type", String(100), nullable=False),
                 Column("operator", String(255), nullable=False), text("semantics"),
                 text("pytorch_reference"), text("tl_ops_json"), Column("failure_stage", String(100)),
                 text("error_signature", nullable=True), text("environment_json"), text("summary"),
                 text("searchable_text"), Column("outcome", String(64), nullable=False),
                 text("evidence_json"), Column("confidence_grade", String(8), nullable=False),
                 Column("confidence", Float(53), nullable=False), text("source_run"),
                 Column("test_sha256", String(64)),
                 Column("active", Integer, nullable=False, server_default="1"),
                 text("archive_reason", nullable=True), timestamp("created_at"), timestamp("updated_at"),
                 Column("last_used_at", String(40)),
                 Column("useful_count", BigInteger, nullable=False, server_default="0"),
                 text("embedding_json", nullable=True), Column("embedding_provider", String(255)),
                 Column("embedding_model", String(255)), Column("embedding_dim", Integer),
                 Column("case_version", String(64)),
                 UniqueConstraint("workspace_id", "fingerprint"))
memory_versions = table("memory_versions",
               Column("memory_id", BigInteger, primary_key=True, autoincrement=False),
               Column("case_version", String(64), primary_key=True), text("payload_json"),
               timestamp("captured_at"),
               ForeignKeyConstraint(["workspace_id", "memory_id"],
                                    ["memories.workspace_id", "memories.id"], ondelete="CASCADE"))
chunks = table("memory_chunks", Column("memory_id", BigInteger, primary_key=True, autoincrement=False),
               Column("kind", String(64), primary_key=True),
               Column("position", Integer, primary_key=True, autoincrement=False), text("text"),
               text("source_field"), text("embedding_json", nullable=True),
               Column("embedding_provider", String(255)), Column("embedding_model", String(255)),
               Column("embedding_dim", Integer),
               Column("case_version", String(64)), Column("chunk_id", String(64)),
               Column("unit_id", String(64)), Column("part_index", Integer),
               Column("part_count", Integer), text("provenance_json", nullable=True),
               ForeignKeyConstraint(["workspace_id", "memory_id", "case_version"],
                                    ["memory_versions.workspace_id", "memory_versions.memory_id",
                                     "memory_versions.case_version"], name="fk_chunks_version", ondelete="CASCADE"),
               ForeignKeyConstraint(["workspace_id", "memory_id"],
                                    ["memories.workspace_id", "memories.id"], ondelete="CASCADE"))
references = table("references_catalog", Column("id", String(64), primary_key=True),
                   Column("admitted", Integer, nullable=False), text("payload"))

Index("idx_messages_session", messages.c.workspace_id, messages.c.session_id, messages.c.ordinal)
Index("idx_runs_session", runs.c.workspace_id, runs.c.session_id, runs.c.created_at)
Index("idx_memories_active", memories.c.workspace_id, memories.c.active)
Index("idx_memories_operator", memories.c.workspace_id, memories.c.operator)
Index("idx_memories_stage", memories.c.workspace_id, memories.c.failure_stage)
Index("idx_chunks_unit", chunks.c.workspace_id, chunks.c.memory_id, chunks.c.case_version, chunks.c.unit_id)
Index("idx_chunks_identity", chunks.c.workspace_id, chunks.c.chunk_id, unique=True)

memory_indexes = table("memory_indexes", Column("generation", String(64), primary_key=True),
    Column("status", String(24), nullable=False), text("config_json"),
    Column("fingerprint", String(64), nullable=False), Column("collection", String(100), nullable=False))
memory_index_entries = table("memory_index_entries",
    Column("generation", String(64), primary_key=True),
    Column("memory_id", BigInteger, primary_key=True, autoincrement=False),
    Column("revision", BigInteger, nullable=False), Column("case_version", String(64)),
    Column("desired", String(16), nullable=False), Column("status", String(24), nullable=False),
    Column("job_id", String(64), nullable=False), Column("point_count", Integer, nullable=False),
    ForeignKeyConstraint(["workspace_id", "generation"],
                         ["memory_indexes.workspace_id", "memory_indexes.generation"]))
# Versioned indexed text remains auditable even after the source case is archived/deleted.
memory_index_chunks = table("memory_index_chunks",
    Column("generation", String(64), primary_key=True),
    Column("memory_id", BigInteger, primary_key=True, autoincrement=False),
    Column("revision", BigInteger, primary_key=True, autoincrement=False),
    Column("chunk_id", String(64), primary_key=True), Column("point_id", String(36), nullable=False),
    Column("case_version", String(64), nullable=False), text("text"), text("provenance_json"),
    ForeignKeyConstraint(["workspace_id", "generation"],
                         ["memory_indexes.workspace_id", "memory_indexes.generation"]))

TABLES = {t.name: t for t in metadata.tables.values()}
