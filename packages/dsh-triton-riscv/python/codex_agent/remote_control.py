"""Host-only, approval-bound remote cancellation. Not a model-facing tool."""
import time
import subprocess

from codex_agent.execution_guard import atomic_json, inspect_execution, journal_path
from codex_agent.process_control import execution_budget
from codex_agent.remote_executor import RemoteValidationConfig, _job_rpc


def request_remote_stop(root, kind, identifier):
    journal = inspect_execution(root, kind, identifier)
    if not journal:
        return {"status": "cancel_requested", "remote_shutdown_confirmed": False,
                "reason": "no durable remote handle yet"}
    if journal.get("kind") != kind or journal.get("id") != identifier:
        raise PermissionError("cancellation journal identity mismatch")
    if journal["state"] == "completed":
        return {"status": "already_completed", "remote_shutdown_confirmed": False,
                "reason": "completed result is unchanged"}
    job = journal.get("execution", {}).get("remote_job") if kind == "validation" else None
    if not job:
        return {"status": "cancel_requested", "remote_shutdown_confirmed": False,
                "reason": "automatic remote stop currently supports one tracked validation only"}
    config = RemoteValidationConfig.from_env()
    result = {"status": "cancel_requested", "run_id": identifier, "job_id": job["job_id"],
              "remote_shutdown_confirmed": False, "requested_at": time.time()}
    if not config or config.host != job.get("host") or config.repository != job.get("repository"):
        result["reason"] = "remote target changed; no request sent"
    else:
        try:
            # Independent of the just-created local cancellation marker. The
            # caller verified the approving native session before entering here.
            with execution_budget(10):
                value = _job_rpc(config, job, "cancel")
                deadline = time.monotonic() + 3
                while value["state"] in {"accepted", "queued", "running"} and time.monotonic() < deadline:
                    time.sleep(0.1)
                    value = _job_rpc(config, job, "inspect")
                result["remote_state"] = value["state"]
                result["terminal"] = value if value["state"] == "completed" else None
                confirmed = value.get("cancellation", {}).get("confirmed") is True
                if value["state"] == "completed" and confirmed:
                    result.update(status="remote_stopped", remote_shutdown_confirmed=True)
                elif value["state"] == "completed":
                    result.update(status="already_completed", reason="original result won the cancellation race")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result["reason"] = f"remote cancellation not confirmed: {type(error).__name__}"
    # Keep this separate: neither a lost cancel reply nor a successful stop may
    # fabricate a completed validation receipt or overwrite its execution journal.
    atomic_json(journal_path(root, kind, identifier).with_suffix(".remote-cancel.json"), result)
    return result
