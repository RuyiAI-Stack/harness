"""Transactional workflow checkpoints. No model, queue dispatch or file writes.

This is an internal server API, not an MCP tool. Callers supply observed input
digests; they are version bindings, not evidence that code is correct or approved.
"""
import json
import uuid
from contextlib import nullcontext

from codex_agent.storage.database import WorkspaceDatabase
from codex_agent.storage.schema import runs, sessions
from codex_agent.tasks.schema import jobs
from codex_agent.tasks.store import Conflict, digest, encode, now_ms
from .models import (InputVersions, NEXT_STATE, PROTECTED_STATES, REQUIRED_VERSIONS,
                     ROLES_BY_STATE, StepRequest, StepResult, TaskRequest, TERMINAL, WAITING)
from .schema import events, steps, tasks


def key_digest(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError("request/event key must contain 1..160 nonblank characters")
    return digest(value)


def bounded(value):
    serialized = encode(value)
    if len(serialized.encode()) > 65536:
        raise ValueError("workflow payload exceeds 64 KiB; use evidence references")
    return serialized


class WorkflowStore:
    def __init__(self, root, *, database=None):
        self.db = database or WorkspaceDatabase(root)
        if database is None:
            self.db.register()

    def _task(self, conn, identifier, *, lock=False):
        rows = self.db.rows(tasks, tasks.c.id == identifier, connection=conn, lock=lock)
        if not rows:
            raise KeyError("unknown workflow in this workspace")
        return rows[0]

    def _snapshot(self, conn, row):
        value = {k: v for k, v in row.items() if k not in {"workspace_id", "request_key", "fingerprint"}}
        for field in ("request", "versions", "result"):
            value[field] = json.loads(value.pop(field + "_json"))
        value["steps"] = []
        for step in self.db.rows(steps, steps.c.task_id == row["id"],
                                 order=(steps.c.created_revision,), connection=conn):
            item = {k: v for k, v in step.items() if k != "workspace_id"}
            for field in ("input", "result"):
                item[field] = json.loads(item.pop(field + "_json"))
            value["steps"].append(item)
        # Reading a checkpoint never means it is safe to repeat an unfinished tool.
        value["recovery_action"] = (
            "inspect_lifecycle_artifacts" if value["result"].get("lifecycle", {}).get("operation") else
            "terminal" if row["state"] in TERMINAL else
            "wait" if row["state"] in WAITING else
            "inspect_unfinished_steps" if any(s["status"] == "running" for s in value["steps"]) else
            "continue_from_checkpoint")
        return value

    def get(self, identifier):
        with self.db.transaction() as conn:
            return self._snapshot(conn, self._task(conn, identifier, lock=True))

    def for_session(self, session_id):
        with self.db.transaction() as conn:
            if not self.db.rows(sessions, sessions.c.id == session_id, connection=conn):
                raise KeyError("unknown session")
            return [self._snapshot(conn, row) for row in self.db.rows(tasks, tasks.c.session_id == session_id,
                    order=(tasks.c.created_ms.desc(), tasks.c.id.desc()), limit=50, connection=conn)]

    def pending(self, *, limit=50):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        return [{k: r[k] for k in ("id", "session_id", "kind", "operator", "state", "revision")}
                for r in self.db.rows(tasks, tasks.c.state.not_in(TERMINAL),
                    order=(tasks.c.updated_ms, tasks.c.id), limit=limit)]

    def history(self, identifier, *, after=0, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("invalid event cursor or limit")
        with self.db.transaction() as conn:
            self._task(conn, identifier)
            return [{"revision": r["revision"], "operation": r["operation"],
                     "payload": json.loads(r["payload_json"]), "snapshot": json.loads(r["snapshot_json"]),
                     "created_ms": r["created_ms"]}
                    for r in self.db.rows(events, events.c.task_id == identifier,
                        events.c.revision > after, order=(events.c.revision,), limit=limit, connection=conn)]

    def _event(self, conn, row, event_key, fingerprint, operation, payload):
        snapshot = {k: row[k] for k in ("state", "revision", "input_epoch", "repair_count", "policy_version")}
        snapshot["versions"] = json.loads(row["versions_json"])
        self.db.insert(conn, events, dict(task_id=row["id"], revision=row["revision"],
            event_key=event_key, fingerprint=fingerprint, operation=operation,
            payload_json=bounded(payload), snapshot_json=encode(snapshot), created_ms=row["updated_ms"]))

    def create(self, request, *, request_id, context=None, connection=None):
        request = TaskRequest.model_validate(request).model_dump(mode="json")
        serialized, key = bounded(request), key_digest(request_id)
        payload = request if context is None else {"request": request, "context": context}
        bounded(payload)
        fingerprint = digest(payload)
        with (self.db.transaction() if connection is None else nullcontext(connection)) as conn:
            self.db.lock(conn, "operator-workflow-create")
            previous = self.db.rows(tasks, tasks.c.request_key == key, connection=conn, lock=True)
            if previous:
                if previous[0]["fingerprint"] != fingerprint:
                    raise Conflict("request key reused with different workflow input")
                return self._snapshot(conn, previous[0])
            if not self.db.rows(sessions, sessions.c.id == request["session_id"], connection=conn, lock=True):
                raise KeyError("unknown session in this workspace")
            now = now_ms(conn)
            row = dict(id=uuid.uuid4().hex, session_id=request["session_id"], kind=request["kind"],
                operator=request["operator"], state="preparing", revision=1, input_epoch=1,
                policy_version="operator-workflow-v1", request_key=key, fingerprint=fingerprint,
                request_json=serialized, versions_json=encode(request["versions"]), result_json="{}",
                repair_count=0, max_repairs=request["max_repairs"], created_ms=now, updated_ms=now)
            self.db.insert(conn, tasks, row)
            self._event(conn, row, digest(["created", row["id"]]), fingerprint, "created", payload)
            return self._snapshot(conn, row)

    def _mutate(self, identifier, expected_revision, event_id, operation, payload, change, *, connection=None):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        key = key_digest(event_id)
        bounded(payload)
        fingerprint = digest([operation, expected_revision, payload])
        with (self.db.transaction() if connection is None else nullcontext(connection)) as conn:
            row = self._task(conn, identifier, lock=True)
            replay = self.db.rows(events, events.c.task_id == identifier, events.c.event_key == key, connection=conn)
            if replay:
                if replay[0]["fingerprint"] != fingerprint:
                    raise Conflict("event key reused with different workflow input")
                return self._snapshot(conn, row)
            if row["revision"] != expected_revision:
                raise Conflict("workflow revision changed; reload before deciding again")
            if row["state"] in TERMINAL:
                raise Conflict("workflow is closed")
            values, detail = change(conn, row)
            row = {**row, **values, "revision": row["revision"] + 1, "updated_ms": now_ms(conn)}
            self.db.update(conn, tasks, {**values, "revision": row["revision"], "updated_ms": row["updated_ms"]},
                           tasks.c.id == identifier)
            self._event(conn, row, key, fingerprint, operation, {**payload, **detail})
            return self._snapshot(conn, row)

    def start_step(self, identifier, request, *, expected_revision, event_id):
        request = StepRequest.model_validate(request).model_dump(mode="json")

        def change(conn, row):
            if request["role"] not in ROLES_BY_STATE.get(row["state"], set()):
                raise Conflict("role is not allowed in the current workflow state")
            if self.db.rows(steps, steps.c.task_id == identifier, steps.c.status == "running", connection=conn):
                raise Conflict("finish or invalidate the current step first")
            run_id, job_id = request["run_id"], request["job_id"]
            if run_id:
                self.db.lock(conn, "workflow-run:" + digest(run_id))
                linked_runs = self.db.rows(runs, runs.c.id == run_id,
                    runs.c.session_id == row["session_id"], connection=conn)
                if (not linked_runs or linked_runs[0]["operator"] not in {None, row["operator"]}
                        or self.db.rows(steps, steps.c.run_id == run_id,
                                        steps.c.task_id != identifier, connection=conn)):
                    raise Conflict("run does not belong to this workflow, session and operator")
            if job_id:
                linked = self.db.rows(jobs, jobs.c.id == job_id, connection=conn)
                if not run_id or not linked or linked[0]["kind"] != "agent" or linked[0]["run_id"] != run_id:
                    raise Conflict("job does not belong to this agent run")
            step_id = uuid.uuid4().hex
            self.db.insert(conn, steps, dict(id=step_id, task_id=identifier, role=request["role"],
                state=row["state"], status="running", input_epoch=row["input_epoch"],
                created_revision=row["revision"] + 1, run_id=run_id, job_id=job_id,
                input_json=bounded({**request, "versions": json.loads(row["versions_json"])}), result_json="{}",
                created_ms=now_ms(conn), finished_ms=None))
            return {}, {"step_id": step_id}

        return self._mutate(identifier, expected_revision, event_id, "step_started", request, change)

    def finish_step(self, identifier, step_id, result, *, expected_revision, event_id):
        result = StepResult.model_validate(result).model_dump(mode="json")

        def change(conn, row):
            found = self.db.rows(steps, steps.c.id == step_id, steps.c.task_id == identifier, connection=conn)
            if not found:
                raise Conflict("step does not belong to this workflow")
            step = found[0]
            if step["status"] != "running" or step["state"] != row["state"] or step["input_epoch"] != row["input_epoch"]:
                raise Conflict("step is finished or its inputs have been superseded")
            self.db.update(conn, steps, dict(status=result["status"], result_json=bounded(result),
                finished_ms=now_ms(conn)), steps.c.id == step_id)
            return {}, {}

        return self._mutate(identifier, expected_revision, event_id, "step_finished",
                            {"step_id": step_id, "result": result}, change)

    def advance(self, identifier, target, versions, *, expected_revision, event_id):
        versions = InputVersions.model_validate(versions).model_dump(mode="json")
        if not isinstance(target, str):
            raise ValueError("target must be a workflow state")

        def change(conn, row):
            if target in PROTECTED_STATES:
                raise PermissionError("approval and validation evidence adapter is not connected; no execution or success transition")
            if NEXT_STATE.get(row["state"]) != target:
                raise Conflict("invalid workflow transition")
            current = self.db.rows(steps, steps.c.task_id == identifier,
                steps.c.input_epoch == row["input_epoch"], steps.c.state == row["state"],
                order=(steps.c.created_revision.desc(),), connection=conn)
            if not current or current[0]["status"] != "completed":
                raise Conflict("the latest step must complete before advancing")
            previous = json.loads(row["versions_json"])
            produced_field = {"preparing": "contract_sha256", "planning_tests": "tests_sha256",
                              "implementing": "source_sha256"}.get(row["state"])
            if any(versions[k] != v for k, v in previous.items() if k != produced_field):
                raise Conflict("upstream inputs changed; revise inputs before proceeding")
            if any(versions[k] is None for k in REQUIRED_VERSIONS[target]):
                raise Conflict("required input versions are missing")
            return {"state": target, "versions_json": encode(versions), "input_epoch": row["input_epoch"] + 1}, {}

        return self._mutate(identifier, expected_revision, event_id, "advanced",
                            {"target": target, "versions": versions}, change)

    def revise_inputs(self, identifier, versions, *, reason, expected_revision, event_id):
        versions = InputVersions.model_validate(versions).model_dump(mode="json")
        reason = self._reason(reason)

        def change(conn, row):
            if row["state"] == "waiting_validation" or json.loads(row["result_json"]).get("lifecycle"):
                raise Conflict("reconcile pending validation before changing inputs")
            previous = json.loads(row["versions_json"])
            changed = {k for k in previous if previous[k] != versions[k]}
            if not changed:
                raise Conflict("input versions are unchanged")
            state = ("preparing" if "contract_sha256" in changed or versions["contract_sha256"] is None else
                     "planning_tests" if "tests_sha256" in changed or versions["tests_sha256"] is None else
                     "implementing")
            # Retain old results for audit, but never count them as current-stage evidence.
            self.db.update(conn, steps, {"status": "superseded"}, steps.c.task_id == identifier,
                           steps.c.status != "superseded")
            return {"state": state, "versions_json": encode(versions),
                    "input_epoch": row["input_epoch"] + 1}, {}

        return self._mutate(identifier, expected_revision, event_id, "inputs_revised",
                            {"versions": versions, "reason": reason}, change)

    def rework(self, identifier, *, expected_revision, event_id, findings):
        def change(conn, row):
            if row["state"] != "reviewing":
                raise Conflict("only reviewed candidates can be reworked")
            result = json.loads(row["result_json"])
            count = result.get("rework_count", 0)
            if count >= row["max_repairs"] or not findings:
                raise Conflict("review rework budget exhausted or feedback missing")
            result.update(rework_count=count + 1, review_feedback=findings)
            return {"state": "implementing", "input_epoch": row["input_epoch"] + 1,
                    "result_json": bounded(result)}, {}
        return self._mutate(identifier, expected_revision, event_id, "review_rework", {"findings": findings}, change)

    @staticmethod
    def _reason(reason):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("a nonblank reason of at most 2000 characters is required")
        return reason.strip()

    def close(self, identifier, status, *, reason, expected_revision, event_id):
        if status not in {"cancelled", "blocked", "failed"}:
            raise ValueError("only cancelled, blocked or failed can be recorded without execution evidence")
        reason = self._reason(reason)

        def change(conn, row):
            active = self.db.rows(steps, steps.c.task_id == identifier, steps.c.status == "running", connection=conn)
            if active or row["state"] == "waiting_validation" or json.loads(row["result_json"]).get("lifecycle"):
                raise Conflict("unfinished work must be reconciled before closing the workflow")
            return {"state": status, "result_json": encode({**json.loads(row["result_json"]), "reason": reason})}, {}

        return self._mutate(identifier, expected_revision, event_id, "closed",
                            {"status": status, "reason": reason}, change)
