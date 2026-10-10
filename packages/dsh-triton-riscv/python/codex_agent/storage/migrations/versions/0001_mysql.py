"""Initial workspace-isolated MySQL schema, frozen at revision 0001."""
from alembic import op
from sqlalchemy import inspect

revision = "0001_mysql"
down_revision = None

TABLE_DDL = (
    """CREATE TABLE IF NOT EXISTS counters (
    workspace_id VARCHAR(64) NOT NULL,
    name VARCHAR(100) NOT NULL,
    value BIGINT NOT NULL,
    PRIMARY KEY (workspace_id, name)
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS memories (
    workspace_id VARCHAR(64) NOT NULL,
    id BIGINT NOT NULL,
    fingerprint VARCHAR(64) NOT NULL,
    memory_type VARCHAR(100) NOT NULL,
    operator VARCHAR(255) NOT NULL,
    semantics LONGTEXT NOT NULL,
    pytorch_reference LONGTEXT NOT NULL,
    tl_ops_json LONGTEXT NOT NULL,
    failure_stage VARCHAR(100),
    error_signature LONGTEXT,
    environment_json LONGTEXT NOT NULL,
    summary LONGTEXT NOT NULL,
    searchable_text LONGTEXT NOT NULL,
    outcome VARCHAR(64) NOT NULL,
    evidence_json LONGTEXT NOT NULL,
    confidence_grade VARCHAR(8) NOT NULL,
    confidence FLOAT(53) NOT NULL,
    source_run LONGTEXT NOT NULL,
    test_sha256 VARCHAR(64),
    active INTEGER NOT NULL DEFAULT '1',
    archive_reason LONGTEXT,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    last_used_at VARCHAR(40),
    useful_count BIGINT NOT NULL DEFAULT '0',
    embedding_json LONGTEXT,
    embedding_provider VARCHAR(255),
    embedding_model VARCHAR(255),
    embedding_dim INTEGER,
    PRIMARY KEY (workspace_id, id),
    UNIQUE (workspace_id, fingerprint)
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS metadata (
    workspace_id VARCHAR(64) NOT NULL,
    `key` VARCHAR(100) NOT NULL,
    value LONGTEXT NOT NULL,
    PRIMARY KEY (workspace_id, `key`)
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS references_catalog (
    workspace_id VARCHAR(64) NOT NULL,
    id VARCHAR(64) NOT NULL,
    admitted INTEGER NOT NULL,
    payload LONGTEXT NOT NULL,
    PRIMARY KEY (workspace_id, id)
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS sessions (
    workspace_id VARCHAR(64) NOT NULL,
    id VARCHAR(64) NOT NULL,
    title LONGTEXT NOT NULL,
    status VARCHAR(64) NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    pinned INTEGER NOT NULL DEFAULT '0',
    PRIMARY KEY (workspace_id, id)
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id VARCHAR(64) NOT NULL,
    root LONGTEXT NOT NULL,
    PRIMARY KEY (workspace_id)
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS context_checkpoints (
    workspace_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    summary LONGTEXT NOT NULL,
    through_message_id VARCHAR(64),
    source_ids_json LONGTEXT NOT NULL,
    metrics_json LONGTEXT NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    PRIMARY KEY (workspace_id, session_id),
    FOREIGN KEY(workspace_id, session_id) REFERENCES sessions (workspace_id, id) ON DELETE CASCADE
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS memory_chunks (
    workspace_id VARCHAR(64) NOT NULL,
    memory_id BIGINT NOT NULL,
    kind VARCHAR(64) NOT NULL,
    position INTEGER NOT NULL,
    text LONGTEXT NOT NULL,
    source_field LONGTEXT NOT NULL,
    embedding_json LONGTEXT,
    embedding_provider VARCHAR(255),
    embedding_model VARCHAR(255),
    embedding_dim INTEGER,
    PRIMARY KEY (workspace_id, memory_id, kind, position),
    FOREIGN KEY(workspace_id, memory_id) REFERENCES memories (workspace_id, id) ON DELETE CASCADE
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS messages (
    workspace_id VARCHAR(64) NOT NULL,
    id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    ordinal BIGINT NOT NULL,
    `role` VARCHAR(64) NOT NULL,
    content LONGTEXT NOT NULL,
    metadata_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    PRIMARY KEY (workspace_id, id),
    FOREIGN KEY(workspace_id, session_id) REFERENCES sessions (workspace_id, id) ON DELETE CASCADE
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS runs (
    workspace_id VARCHAR(64) NOT NULL,
    id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    intent LONGTEXT NOT NULL,
    operator VARCHAR(255),
    status VARCHAR(64) NOT NULL,
    phase VARCHAR(64) NOT NULL,
    request_json LONGTEXT NOT NULL,
    result_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    next_sequence BIGINT NOT NULL DEFAULT '0',
    PRIMARY KEY (workspace_id, id),
    FOREIGN KEY(workspace_id, session_id) REFERENCES sessions (workspace_id, id) ON DELETE CASCADE
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS events (
    workspace_id VARCHAR(64) NOT NULL,
    id BIGINT NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    sequence BIGINT NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payload_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    PRIMARY KEY (workspace_id, id),
    UNIQUE (workspace_id, run_id, sequence),
    FOREIGN KEY(workspace_id, run_id) REFERENCES runs (workspace_id, id) ON DELETE CASCADE
)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin""",
)
INDEX_DDL = (
    ("memories", "idx_memories_active", "CREATE INDEX idx_memories_active ON memories (workspace_id, active)"),
    ("memories", "idx_memories_operator", "CREATE INDEX idx_memories_operator ON memories (workspace_id, operator)"),
    ("memories", "idx_memories_stage", "CREATE INDEX idx_memories_stage ON memories (workspace_id, failure_stage)"),
    ("messages", "idx_messages_session", "CREATE INDEX idx_messages_session ON messages (workspace_id, session_id, ordinal)"),
    ("runs", "idx_runs_session", "CREATE INDEX idx_runs_session ON runs (workspace_id, session_id, created_at)"),
)


def upgrade():
    # MySQL DDL commits independently; retry completes a partially applied revision.
    for statement in TABLE_DDL:
        op.execute(statement)
    inspector = inspect(op.get_bind())
    for table, name, statement in INDEX_DDL:
        if name not in {item["name"] for item in inspector.get_indexes(table)}:
            op.execute(statement)


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
