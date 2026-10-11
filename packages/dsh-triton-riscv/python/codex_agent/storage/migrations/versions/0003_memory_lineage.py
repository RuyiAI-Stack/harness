"""Add nullable lineage and immutable source snapshots; no automatic data rewrite."""
from alembic import op
from sqlalchemy import inspect

revision = "0003_memory_lineage"
down_revision = "0002_tasks"

COLUMNS = {
    "memories": {"case_version": "VARCHAR(64)"},
    "memory_chunks": {"case_version": "VARCHAR(64)", "chunk_id": "VARCHAR(64)",
                      "unit_id": "VARCHAR(64)", "part_index": "INTEGER",
                      "part_count": "INTEGER", "provenance_json": "LONGTEXT"},
}


def upgrade():
    # MySQL DDL is not transactional: a retry must tolerate partially added columns.
    for table, columns in COLUMNS.items():
        existing = {c["name"] for c in inspect(op.get_bind()).get_columns(table)}
        for name, kind in columns.items():
            if name not in existing:
                op.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind} NULL")
    op.execute("""CREATE TABLE IF NOT EXISTS memory_versions (
        workspace_id VARCHAR(64) NOT NULL, memory_id BIGINT NOT NULL,
        case_version VARCHAR(64) NOT NULL, payload_json LONGTEXT NOT NULL,
        captured_at VARCHAR(40) NOT NULL,
        PRIMARY KEY (workspace_id, memory_id, case_version),
        FOREIGN KEY (workspace_id, memory_id) REFERENCES memories (workspace_id, id) ON DELETE CASCADE
    ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""")
    existing = {i["name"] for i in inspect(op.get_bind()).get_indexes("memory_chunks")}
    for name, sql in {
        "idx_chunks_identity": "CREATE UNIQUE INDEX idx_chunks_identity ON memory_chunks (workspace_id, chunk_id)",
        "idx_chunks_unit": "CREATE INDEX idx_chunks_unit ON memory_chunks (workspace_id, memory_id, case_version, unit_id)",
    }.items():
        if name not in existing:
            op.execute(sql)
    if "fk_chunks_version" not in {f["name"] for f in inspect(op.get_bind()).get_foreign_keys("memory_chunks")}:
        op.execute("""ALTER TABLE memory_chunks ADD CONSTRAINT fk_chunks_version
            FOREIGN KEY (workspace_id, memory_id, case_version)
            REFERENCES memory_versions (workspace_id, memory_id, case_version) ON DELETE CASCADE""")


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
