"""Transactional task/outbox records. AMQP contains references, never authority."""
from contextlib import nullcontext
import hashlib
import json
import secrets
import socket

from sqlalchemy import and_, or_, select, func

from codex_agent.runtime_config import runtime_config
from codex_agent.storage.database import WorkspaceDatabase
from codex_agent.platform.store import PlatformStore, timestamp
from codex_agent.storage.schema import runs
from .schema import jobs, attempts, outbox, inbox, artifacts


TERMINAL = {"succeeded", "failed", "cancelled", "dead"}


class Conflict(ValueError):
    pass


class LeaseLost(RuntimeError):
    pass


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def now_ms(conn):
    # One authority for leases even when clients have different wall clocks.
    return int(conn.execute(select(func.unix_timestamp(func.current_timestamp(3)) * 1000)).scalar_one())


class JobStore:
    def __init__(self, root):
        self.db = WorkspaceDatabase(root)
        self.db.register()
        self.platform = PlatformStore(root)
        self.config = runtime_config().queue

    def _row(self, conn, table, identifier, *, lock=False):
        rows = self.db.rows(table, table.c.id == identifier, connection=conn, lock=lock)
        if not rows:
            raise KeyError("unknown task record")
        return rows[0]

    def get(self, identifier):
        with self.db.transaction() as conn:
            row = self._row(conn, jobs, identifier)
        return self.public(row)

    @staticmethod
    def public(row):
        result = {k: v for k, v in row.items() if k not in {"owner", "workspace_id", "idempotency_key"}}
        for field in ("payload", "result"):
            result[field] = json.loads(result.pop(field + "_json"))
        return result

    def find_key(self, conn, key):
        return next(iter(self.db.rows(jobs, jobs.c.idempotency_key == digest(key), connection=conn)), None)

    def submit(self, kind, payload, key, *, run_id=None, connection=None):
        if kind not in {"agent", "validation", "memory"}:
            raise ValueError("unsupported job kind")
        if not isinstance(key, str) or not 1 <= len(key) <= 160:
            raise ValueError("idempotency key must contain 1..160 characters")
        serialized = encode(payload)
        if len(serialized.encode()) > 32768:
            raise ValueError("task payload exceeds 32 KiB")
        fingerprint = digest([kind, payload])
        with nullcontext(connection) if connection is not None else self.db.transaction() as conn:
            self.db.lock(conn, "task-submit")
            previous = self.find_key(conn, key)
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise Conflict("idempotency key reused with different task input")
                return self.public(previous)
            count = conn.execute(select(func.count()).select_from(jobs).where(
                *self.db.predicate(jobs, jobs.c.status.not_in(TERMINAL)))).scalar_one()
            if kind != "memory" and count >= self.config.maxPending:
                raise Conflict("workspace task capacity reached")
            latest = conn.execute(select(func.max(jobs.c.created_ms)).where(*self.db.predicate(jobs))).scalar_one()
            now, identifier = max(now_ms(conn), (latest or 0) + 1), secrets.token_hex(16)
            value = dict(id=identifier, kind=kind, status="queued", idempotency_key=digest(key),
                fingerprint=fingerprint, payload_json=serialized, result_json="{}", run_id=run_id,
                execution_host=socket.gethostname(), attempt=0, generation=1, owner=None,
                lease_until=0, effects_started=0, cancel_requested=0, created_ms=now, updated_ms=now)
            self.db.insert(conn, jobs, value)
            self._enqueue(conn, value, now)
            return self.public(value)

    def _enqueue(self, conn, job, at):
        self.db.insert(conn, outbox, dict(id=secrets.token_hex(16), job_id=job["id"],
            generation=job["generation"], kind=job["kind"], status="pending", attempts=0,
            owner=None, lease_until=0, available_ms=at, created_ms=now_ms(conn),
            published_ms=None, last_error=None))

    def claim_publication(self):
        with self.db.transaction() as conn:
            now = now_ms(conn)
            stmt = select(outbox).where(*self.db.predicate(outbox,
                outbox.c.available_ms <= now, or_(outbox.c.status == "pending",
                    and_(outbox.c.status == "publishing", outbox.c.lease_until < now))))
            row = conn.execute(stmt.order_by(outbox.c.created_ms, outbox.c.id)
                .limit(1).with_for_update(skip_locked=True)).mappings().first()
            if row is None:
                return None
            owner = secrets.token_hex(16)
            self.db.update(conn, outbox, dict(status="publishing", owner=owner,
                lease_until=now + 30000, attempts=row["attempts"] + 1), outbox.c.id == row["id"])
            return {**dict(row), "owner": owner}

    def publication_result(self, event, error=None):
        with self.db.transaction() as conn:
            row = self._row(conn, outbox, event["id"], lock=True)
            now = now_ms(conn)
            if row["owner"] != event["owner"] or row["status"] != "publishing" or row["lease_until"] < now:
                raise LeaseLost("outbox lease expired")
            # Keep failed deliveries visible and retryable; never discard on broker downtime.
            values = dict(owner=None, lease_until=0, status="pending" if error else "published",
                last_error=error, available_ms=now + min(60000, 1000 * 2 ** min(row["attempts"], 6)),
                published_ms=None if error else now)
            self.db.update(conn, outbox, values, outbox.c.id == row["id"])

    def envelope(self, event):
        return {"version": 1, "workspace_id": self.db.workspace_id, "event_id": event["id"],
                "job_id": event["job_id"], "kind": event["kind"], "generation": event["generation"]}

    def claim(self, envelope, kind):
        if (not isinstance(envelope, dict) or set(envelope) !=
                {"version", "workspace_id", "event_id", "job_id", "kind", "generation"}
                or type(envelope["version"]) is not int or envelope["version"] != 1
                or envelope["workspace_id"] != self.db.workspace_id or envelope["kind"] != kind):
            raise ValueError("invalid task envelope")
        with self.db.transaction() as conn:
            self.db.lock(conn, "task-claim")
            event = self._row(conn, outbox, envelope["event_id"])
            if self.envelope(event) != envelope:
                raise ValueError("message does not match the durable outbox")
            job = self._row(conn, jobs, envelope["job_id"], lock=True)
            if job["execution_host"] != socket.gethostname():
                self.db.update(conn, jobs, {"result_json": encode({"reason": "worker host mismatch", "expected_host": job["execution_host"]})}, jobs.c.id == job["id"])
                if job["status"] == "queued" and job["generation"] == envelope["generation"]:
                    self._retry_queued(conn, job, 30000)
                return None
            if job["status"] in TERMINAL or job["generation"] != envelope["generation"]:
                return None
            if job["status"] != "queued":
                return None  # Recovery, not redelivery, owns an expired/uncertain attempt.
            blockers = self.db.rows(jobs, jobs.c.id != job["id"],
                jobs.c.kind.in_(["agent", "validation"]),
                or_(jobs.c.status.in_(["running", "needs-reconciliation"]),
                    and_(jobs.c.status == "queued", or_(jobs.c.created_ms < job["created_ms"],
                        and_(jobs.c.created_ms == job["created_ms"], jobs.c.id < job["id"])))), connection=conn)
            if blockers and kind != "memory":
                self._retry_queued(conn, job, 1000)
                return None
            now, owner = now_ms(conn), secrets.token_hex(16)
            values = dict(status="running", owner=owner, attempt=job["attempt"] + 1,
                lease_until=now + self.config.leaseSeconds * 1000, updated_ms=now)
            self.db.update(conn, jobs, values, jobs.c.id == job["id"])
            self.db.insert(conn, attempts, dict(job_id=job["id"], number=values["attempt"],
                owner=owner, status="running", started_ms=now, finished_ms=None, result_json="{}"))
            if not self.db.rows(inbox, inbox.c.event_id == event["id"], connection=conn):
                self.db.insert(conn, inbox, dict(event_id=event["id"], job_id=job["id"],
                    status="received", received_ms=now, finished_ms=None))
            return {**job, **values}

    def _retry_queued(self, conn, job, delay):
        new = {**job, "generation": job["generation"] + 1}
        self.db.update(conn, jobs, dict(status="queued", generation=new["generation"], owner=None,
            lease_until=0, effects_started=0, updated_ms=now_ms(conn)), jobs.c.id == job["id"])
        self._enqueue(conn, new, now_ms(conn) + delay)

    def require_owner(self, conn, identifier, owner):
        row = self._row(conn, jobs, identifier, lock=True)
        if row["status"] != "running" or row["owner"] != owner or row["lease_until"] < now_ms(conn):
            raise LeaseLost("task lease lost; do not repeat external effects")
        return row

    def heartbeat(self, identifier, owner):
        with self.db.transaction() as conn:
            row = self.require_owner(conn, identifier, owner)
            self.db.update(conn, jobs, dict(lease_until=now_ms(conn) + self.config.leaseSeconds * 1000,
                updated_ms=now_ms(conn)), jobs.c.id == identifier)
            return bool(row["cancel_requested"])

    def begin_effects(self, identifier, owner):
        with self.db.transaction() as conn:
            row = self.require_owner(conn, identifier, owner)
            if row["cancel_requested"]:
                raise LeaseLost("task cancellation requested before effects")
            self.db.update(conn, jobs, {"effects_started": 1}, jobs.c.id == identifier)

    def finish(self, identifier, owner, status, result, *, followup=None):
        if status not in TERMINAL | {"needs-reconciliation"}:
            raise ValueError("invalid completion status")
        with self.db.transaction() as conn:
            job = self.require_owner(conn, identifier, owner)
            now = now_ms(conn)
            self.db.update(conn, jobs, dict(status=status, result_json=encode(result),
                owner=None, lease_until=0, updated_ms=now), jobs.c.id == identifier)
            self.db.update(conn, attempts, dict(status=status, result_json=encode(result), finished_ms=now),
                attempts.c.job_id == identifier, attempts.c.number == job["attempt"])
            self.db.update(conn, inbox, dict(status="completed", finished_ms=now), inbox.c.job_id == identifier)
            self._run_status(conn, job, status, result)
            if followup:
                self.submit("memory", followup, "memory:" + identifier, connection=conn)

    def _run_status(self, conn, job, status, result):
        if not job["run_id"]:
            return
        mapped = {"succeeded": "completed", "cancelled": "cancelled", "queued": "queued",
                  "running": "running"}.get(status, "failed")
        current = self.db.rows(runs, runs.c.id == job["run_id"], connection=conn, lock=True)
        if current and (current[0]["status"] not in {"completed", "failed", "cancelled"}
                        or current[0]["phase"] == "needs-reconciliation"):
            self.db.update(conn, runs, dict(status=mapped, phase=status, result_json=encode(result), updated_at=timestamp()),
                runs.c.id == job["run_id"])
            self.platform._add_event(conn, job["run_id"], mapped,
                {"job_id": job["id"], "phase": status, "message": result.get("reason", status)})

    def cancel(self, identifier):
        with self.db.transaction() as conn:
            job = self._row(conn, jobs, identifier, lock=True)
            if job["status"] in TERMINAL:
                return self.public(job)
            status = "cancelled" if job["status"] == "queued" else job["status"]
            self.db.update(conn, jobs, dict(cancel_requested=1, status=status, updated_ms=now_ms(conn)), jobs.c.id == identifier)
            if status == "cancelled":
                self._run_status(conn, job, status, {"reason": "cancelled before execution"})
        return self.get(identifier)

    def recover(self):
        recovered = []
        with self.db.transaction() as conn:
            self.db.lock(conn, "task-claim")
            now = now_ms(conn)
            rows = conn.execute(select(jobs).where(*self.db.predicate(jobs,
                jobs.c.status == "running", jobs.c.lease_until < now))
                .limit(50).with_for_update(skip_locked=True)).mappings().all()
            for row in rows:
                job = dict(row)
                status = "needs-reconciliation" if job["effects_started"] and job["kind"] != "memory" else "queued"
                persisted = self.db.rows(runs, runs.c.id == job["run_id"], connection=conn) if job["run_id"] else []
                completed = persisted and persisted[0]["status"] in {"completed", "failed", "cancelled"}
                if completed:
                    status = {"completed": "succeeded", "failed": "failed", "cancelled": "cancelled"}[persisted[0]["status"]]
                if status == "queued" and job["cancel_requested"]:
                    status = "cancelled"
                if status == "queued" and job["attempt"] >= self.config.maxAttempts:
                    status = "dead"
                reason = {"reason": "worker lease expired", "external_outcome": "unknown" if job["effects_started"] else "not-started"}
                if completed:
                    reason = json.loads(persisted[0]["result_json"])
                self.db.update(conn, attempts, dict(status="interrupted", finished_ms=now, result_json=encode(reason)),
                    attempts.c.job_id == job["id"], attempts.c.number == job["attempt"])
                if status == "queued":
                    self._retry_queued(conn, job, min(30000, 1000 * 2 ** job["attempt"]))
                else:
                    self.db.update(conn, jobs, dict(status=status, owner=None, lease_until=0,
                        result_json=encode(reason), updated_ms=now), jobs.c.id == job["id"])
                self._run_status(conn, job, status, reason)
                recovered.append({"job_id": job["id"], "status": status})
        return recovered

    def reconcile(self, identifier, note):
        if not note.strip():
            raise ValueError("record how external execution was checked and stopped")
        with self.db.transaction() as conn:
            job = self._row(conn, jobs, identifier, lock=True)
            if job["status"] != "needs-reconciliation":
                raise Conflict("only uncertain tasks require manual reconciliation")
            result = {"reason": "manually closed without automatic retry", "operator_note": note[:2000],
                      "verification": "operator-attested-not-machine-verified"}
            self.db.update(conn, jobs, dict(status="failed", result_json=encode(result), updated_ms=now_ms(conn)), jobs.c.id == identifier)
            self._run_status(conn, job, "failed", result)
        return self.get(identifier)

    def detail(self, identifier):
        job = self.get(identifier)
        job["attempts"] = [{k: v for k, v in r.items() if k not in {"workspace_id", "owner"}}
            for r in self.db.rows(attempts, attempts.c.job_id == identifier, order=(attempts.c.number,))]
        job["deliveries"] = [{k: r[k] for k in ("id", "generation", "status", "attempts", "last_error", "published_ms")}
            for r in self.db.rows(outbox, outbox.c.job_id == identifier, order=(outbox.c.generation,))]
        job["artifacts"] = self.db.rows(artifacts, artifacts.c.job_id == identifier)
        with self.db.transaction() as conn:
            followup = self.find_key(conn, "memory:" + identifier)
        job["followups"] = [self.public(followup)] if followup else []
        return job
