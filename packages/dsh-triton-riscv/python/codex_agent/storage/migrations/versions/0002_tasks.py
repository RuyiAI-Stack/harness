"""Transactional jobs and delivery ledger, frozen at revision 0002."""
from alembic import op
from sqlalchemy import inspect

revision = "0002_tasks"
down_revision = "0001_mysql"
TABLE_DDL = [
    "\nCREATE TABLE IF NOT EXISTS jobs (\n\tworkspace_id VARCHAR(64) NOT NULL, \n\tid VARCHAR(64) NOT NULL, \n\tkind VARCHAR(24) NOT NULL, \n\tstatus VARCHAR(32) NOT NULL, \n\tidempotency_key VARCHAR(64) NOT NULL, \n\tfingerprint VARCHAR(64) NOT NULL, \n\tpayload_json LONGTEXT NOT NULL, \n\tresult_json LONGTEXT NOT NULL, \n\trun_id VARCHAR(64), \n\texecution_host VARCHAR(255) NOT NULL, \n\tattempt INTEGER NOT NULL, \n\tgeneration INTEGER NOT NULL, \n\towner VARCHAR(64), \n\tlease_until BIGINT NOT NULL, \n\teffects_started INTEGER NOT NULL, \n\tcancel_requested INTEGER NOT NULL, \n\tcreated_ms BIGINT NOT NULL, \n\tupdated_ms BIGINT NOT NULL, \n\tPRIMARY KEY (workspace_id, id), \n\tUNIQUE (workspace_id, idempotency_key)\n)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin\n\n",
    "\nCREATE TABLE IF NOT EXISTS job_attempts (\n\tworkspace_id VARCHAR(64) NOT NULL, \n\tjob_id VARCHAR(64) NOT NULL, \n\tnumber INTEGER NOT NULL, \n\towner VARCHAR(64) NOT NULL, \n\tstatus VARCHAR(32) NOT NULL, \n\tstarted_ms BIGINT NOT NULL, \n\tfinished_ms BIGINT, \n\tresult_json LONGTEXT NOT NULL, \n\tPRIMARY KEY (workspace_id, job_id, number)\n)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin\n\n",
    "\nCREATE TABLE IF NOT EXISTS outbox_events (\n\tworkspace_id VARCHAR(64) NOT NULL, \n\tid VARCHAR(64) NOT NULL, \n\tjob_id VARCHAR(64) NOT NULL, \n\tgeneration INTEGER NOT NULL, \n\tkind VARCHAR(24) NOT NULL, \n\tstatus VARCHAR(24) NOT NULL, \n\tattempts INTEGER NOT NULL, \n\towner VARCHAR(64), \n\tlease_until BIGINT NOT NULL, \n\tavailable_ms BIGINT NOT NULL, \n\tcreated_ms BIGINT NOT NULL, \n\tpublished_ms BIGINT, \n\tlast_error VARCHAR(255), \n\tPRIMARY KEY (workspace_id, id), \n\tUNIQUE (workspace_id, job_id, generation)\n)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin\n\n",
    "\nCREATE TABLE IF NOT EXISTS inbox_events (\n\tworkspace_id VARCHAR(64) NOT NULL, \n\tevent_id VARCHAR(64) NOT NULL, \n\tjob_id VARCHAR(64) NOT NULL, \n\tstatus VARCHAR(24) NOT NULL, \n\treceived_ms BIGINT NOT NULL, \n\tfinished_ms BIGINT, \n\tPRIMARY KEY (workspace_id, event_id)\n)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin\n\n",
    "\nCREATE TABLE IF NOT EXISTS task_artifacts (\n\tworkspace_id VARCHAR(64) NOT NULL, \n\tid VARCHAR(64) NOT NULL, \n\tjob_id VARCHAR(64) NOT NULL, \n\tattempt INTEGER NOT NULL, \n\tsha256 VARCHAR(64) NOT NULL, \n\tsize BIGINT NOT NULL, \n\tname VARCHAR(120) NOT NULL, \n\tstorage_key LONGTEXT NOT NULL, \n\tPRIMARY KEY (workspace_id, id)\n)ENGINE=InnoDB CHARSET=utf8mb4 COLLATE utf8mb4_bin\n\n"
]
INDEX_DDL = [
    [
        "jobs",
        "idx_job_recovery",
        "CREATE INDEX idx_job_recovery ON jobs (workspace_id, status, lease_until)"
    ],
    [
        "outbox_events",
        "idx_outbox_ready",
        "CREATE INDEX idx_outbox_ready ON outbox_events (workspace_id, status, available_ms)"
    ]
]


def upgrade():
    for statement in TABLE_DDL:
        op.execute(statement)
    inspector = inspect(op.get_bind())
    for table, name, statement in INDEX_DDL:
        if name not in {item["name"] for item in inspector.get_indexes(table)}:
            op.execute(statement)


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
