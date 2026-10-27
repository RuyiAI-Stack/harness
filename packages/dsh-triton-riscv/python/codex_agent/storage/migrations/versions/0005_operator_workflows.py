"""Add dormant workflow checkpoints; no changes to existing jobs or source data."""
from alembic import op
from sqlalchemy import inspect

revision = "0005_operator_workflows"
down_revision = "0004_memory_indexes"

TABLE_DDL = (
    """operator_workflows (
        workspace_id VARCHAR(64) NOT NULL, id VARCHAR(64) NOT NULL,
        session_id VARCHAR(64) NOT NULL, kind VARCHAR(24) NOT NULL, operator VARCHAR(128) NOT NULL,
        state VARCHAR(32) NOT NULL, revision BIGINT NOT NULL, input_epoch BIGINT NOT NULL,
        policy_version VARCHAR(32) NOT NULL, request_key VARCHAR(64) NOT NULL,
        fingerprint VARCHAR(64) NOT NULL, request_json LONGTEXT NOT NULL,
        versions_json LONGTEXT NOT NULL, result_json LONGTEXT NOT NULL,
        repair_count INTEGER NOT NULL, max_repairs INTEGER NOT NULL,
        created_ms BIGINT NOT NULL, updated_ms BIGINT NOT NULL,
        PRIMARY KEY (workspace_id, id), UNIQUE (workspace_id, request_key))""",
    """operator_workflow_steps (
        workspace_id VARCHAR(64) NOT NULL, id VARCHAR(64) NOT NULL, task_id VARCHAR(64) NOT NULL,
        role VARCHAR(24) NOT NULL, state VARCHAR(32) NOT NULL, status VARCHAR(32) NOT NULL,
        input_epoch BIGINT NOT NULL, created_revision BIGINT NOT NULL,
        run_id VARCHAR(64), job_id VARCHAR(64), input_json LONGTEXT NOT NULL,
        result_json LONGTEXT NOT NULL, created_ms BIGINT NOT NULL, finished_ms BIGINT,
        PRIMARY KEY (workspace_id, id),
        FOREIGN KEY (workspace_id, task_id) REFERENCES operator_workflows (workspace_id, id))""",
    """operator_workflow_events (
        workspace_id VARCHAR(64) NOT NULL, task_id VARCHAR(64) NOT NULL, revision BIGINT NOT NULL,
        event_key VARCHAR(64) NOT NULL, fingerprint VARCHAR(64) NOT NULL, operation VARCHAR(32) NOT NULL,
        payload_json LONGTEXT NOT NULL, snapshot_json LONGTEXT NOT NULL, created_ms BIGINT NOT NULL,
        PRIMARY KEY (workspace_id, task_id, revision), UNIQUE (workspace_id, task_id, event_key),
        FOREIGN KEY (workspace_id, task_id) REFERENCES operator_workflows (workspace_id, id))""",
)
INDEX_DDL = (
    ("operator_workflows", "idx_operator_workflow_session", "workspace_id, session_id, created_ms"),
    ("operator_workflows", "idx_operator_workflow_pending", "workspace_id, state, updated_ms"),
    ("operator_workflow_steps", "idx_operator_workflow_steps", "workspace_id, task_id, created_revision"),
)


def upgrade():
    for statement in TABLE_DDL:
        op.execute("CREATE TABLE IF NOT EXISTS " + statement + " ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin")
    inspector = inspect(op.get_bind())
    for table, name, columns in INDEX_DDL:
        if name not in {item["name"] for item in inspector.get_indexes(table)}:
            op.execute(f"CREATE INDEX {name} ON {table} ({columns})")


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
