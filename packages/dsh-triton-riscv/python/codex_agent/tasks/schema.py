"""Queue schema, kept independent of conversation and memory models."""
from sqlalchemy import BigInteger, Column, Index, Integer, String, UniqueConstraint
from codex_agent.storage.schema import table, text

jobs = table("jobs", Column("id", String(64), primary_key=True),
    Column("kind", String(24), nullable=False), Column("status", String(32), nullable=False),
    Column("idempotency_key", String(64), nullable=False), Column("fingerprint", String(64), nullable=False),
    text("payload_json"), text("result_json"), Column("run_id", String(64)),
    Column("execution_host", String(255), nullable=False),
    Column("attempt", Integer, nullable=False), Column("generation", Integer, nullable=False),
    Column("owner", String(64)), Column("lease_until", BigInteger, nullable=False),
    Column("effects_started", Integer, nullable=False), Column("cancel_requested", Integer, nullable=False),
    Column("created_ms", BigInteger, nullable=False), Column("updated_ms", BigInteger, nullable=False),
    UniqueConstraint("workspace_id", "idempotency_key"))
attempts = table("job_attempts", Column("job_id", String(64), primary_key=True),
    Column("number", Integer, primary_key=True, autoincrement=False),
    Column("owner", String(64), nullable=False), Column("status", String(32), nullable=False),
    Column("started_ms", BigInteger, nullable=False), Column("finished_ms", BigInteger), text("result_json"))
outbox = table("outbox_events", Column("id", String(64), primary_key=True),
    Column("job_id", String(64), nullable=False), Column("generation", Integer, nullable=False),
    Column("kind", String(24), nullable=False), Column("status", String(24), nullable=False),
    Column("attempts", Integer, nullable=False), Column("owner", String(64)),
    Column("lease_until", BigInteger, nullable=False), Column("available_ms", BigInteger, nullable=False),
    Column("created_ms", BigInteger, nullable=False), Column("published_ms", BigInteger),
    Column("last_error", String(255)), UniqueConstraint("workspace_id", "job_id", "generation"))
inbox = table("inbox_events", Column("event_id", String(64), primary_key=True),
    Column("job_id", String(64), nullable=False), Column("status", String(24), nullable=False),
    Column("received_ms", BigInteger, nullable=False), Column("finished_ms", BigInteger))
artifacts = table("task_artifacts", Column("id", String(64), primary_key=True),
    Column("job_id", String(64), nullable=False), Column("attempt", Integer, nullable=False),
    Column("sha256", String(64), nullable=False), Column("size", BigInteger, nullable=False),
    Column("name", String(120), nullable=False), text("storage_key"))
Index("idx_job_recovery", jobs.c.workspace_id, jobs.c.status, jobs.c.lease_until)
Index("idx_outbox_ready", outbox.c.workspace_id, outbox.c.status, outbox.c.available_ms)

QUEUE_TABLES = (jobs, attempts, outbox, inbox, artifacts)
