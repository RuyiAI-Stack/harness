"""Asynchronous index ledger, frozen at revision 0004; no historical backfill."""
from alembic import op

revision = "0004_memory_indexes"
down_revision = "0003_memory_lineage"


def upgrade():
    for ddl in (
        """memory_indexes (
            workspace_id VARCHAR(64) NOT NULL, generation VARCHAR(64) NOT NULL,
            status VARCHAR(24) NOT NULL, config_json LONGTEXT NOT NULL,
            fingerprint VARCHAR(64) NOT NULL, collection VARCHAR(100) NOT NULL,
            PRIMARY KEY (workspace_id, generation))""",
        """memory_index_entries (
            workspace_id VARCHAR(64) NOT NULL, generation VARCHAR(64) NOT NULL,
            memory_id BIGINT NOT NULL, revision BIGINT NOT NULL, case_version VARCHAR(64),
            desired VARCHAR(16) NOT NULL, status VARCHAR(24) NOT NULL, job_id VARCHAR(64) NOT NULL,
            point_count INTEGER NOT NULL, PRIMARY KEY (workspace_id, generation, memory_id),
            FOREIGN KEY (workspace_id, generation) REFERENCES memory_indexes (workspace_id, generation))""",
        """memory_index_chunks (
            workspace_id VARCHAR(64) NOT NULL, generation VARCHAR(64) NOT NULL,
            memory_id BIGINT NOT NULL, revision BIGINT NOT NULL, chunk_id VARCHAR(64) NOT NULL,
            point_id VARCHAR(36) NOT NULL, case_version VARCHAR(64) NOT NULL,
            text LONGTEXT NOT NULL, provenance_json LONGTEXT NOT NULL,
            PRIMARY KEY (workspace_id, generation, memory_id, revision, chunk_id),
            FOREIGN KEY (workspace_id, generation) REFERENCES memory_indexes (workspace_id, generation))""",
    ):
        op.execute("CREATE TABLE IF NOT EXISTS " + ddl + " ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin")


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
