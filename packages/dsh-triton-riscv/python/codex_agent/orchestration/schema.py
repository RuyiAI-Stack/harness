"""Workflow records are separate from one-shot runs and queue delivery jobs."""
from sqlalchemy import BigInteger, Column, ForeignKeyConstraint, Index, Integer, String, UniqueConstraint

from codex_agent.storage.schema import table, text


tasks = table("operator_workflows", Column("id", String(64), primary_key=True),
    Column("session_id", String(64), nullable=False), Column("kind", String(24), nullable=False),
    Column("operator", String(128), nullable=False), Column("state", String(32), nullable=False),
    Column("revision", BigInteger, nullable=False), Column("input_epoch", BigInteger, nullable=False),
    Column("policy_version", String(32), nullable=False),
    Column("request_key", String(64), nullable=False), Column("fingerprint", String(64), nullable=False),
    text("request_json"), text("versions_json"), text("result_json"),
    Column("repair_count", Integer, nullable=False), Column("max_repairs", Integer, nullable=False),
    Column("created_ms", BigInteger, nullable=False), Column("updated_ms", BigInteger, nullable=False),
    UniqueConstraint("workspace_id", "request_key"))

steps = table("operator_workflow_steps", Column("id", String(64), primary_key=True),
    Column("task_id", String(64), nullable=False), Column("role", String(24), nullable=False),
    Column("state", String(32), nullable=False), Column("status", String(32), nullable=False),
    Column("input_epoch", BigInteger, nullable=False), Column("created_revision", BigInteger, nullable=False),
    Column("run_id", String(64)), Column("job_id", String(64)),
    text("input_json"), text("result_json"),
    Column("created_ms", BigInteger, nullable=False), Column("finished_ms", BigInteger),
    ForeignKeyConstraint(["workspace_id", "task_id"],
                         ["operator_workflows.workspace_id", "operator_workflows.id"]))

events = table("operator_workflow_events", Column("task_id", String(64), primary_key=True),
    Column("revision", BigInteger, primary_key=True, autoincrement=False),
    Column("event_key", String(64), nullable=False), Column("fingerprint", String(64), nullable=False),
    Column("operation", String(32), nullable=False), text("payload_json"), text("snapshot_json"),
    Column("created_ms", BigInteger, nullable=False),
    UniqueConstraint("workspace_id", "task_id", "event_key"),
    ForeignKeyConstraint(["workspace_id", "task_id"],
                         ["operator_workflows.workspace_id", "operator_workflows.id"]))

Index("idx_operator_workflow_session", tasks.c.workspace_id, tasks.c.session_id, tasks.c.created_ms)
Index("idx_operator_workflow_pending", tasks.c.workspace_id, tasks.c.state, tasks.c.updated_ms)
Index("idx_operator_workflow_steps", steps.c.workspace_id, steps.c.task_id, steps.c.created_revision)
