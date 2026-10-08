"""Read-only Linux resource observations, NOT a quota installer or enforcer.

This stdlib-only file also runs over SSH on stdin. Never write to a cgroup:
the current group can be shared with unrelated SSH sessions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath


PREFIX = "RESOURCE_CAPABILITIES="
REQUIRE_ENV = "TRITON_RISCV_REQUIRE_TASK_QUOTAS"
MAX_BYTES = 1024 * 1024


def quota_policy_error(environ=None):
    env = os.environ if environ is None else environ
    if "TRITON_RISCV_CONFIG" in env:
        # Only the local host calls this policy helper. The remote probe below
        # remains standalone stdlib code and does not load plugin dependencies.
        from codex_agent.runtime_config import runtime_config
        value = "1" if runtime_config(env).remote.requireTaskQuotas else "0"
    else:
        value = env.get(REQUIRE_ENV, "0").strip().lower()
    if value not in {"", "0", "false", "1", "true"}:
        return f"{REQUIRE_ENV} must be 0 or 1; new validation was not started"
    if value in {"1", "true"}:
        return ("Task-wide CPU and memory quotas are required, but this executor has "
                "no verified per-task cgroup enforcer. New validation was not started; "
                "ask the host administrator for a dedicated delegated resource group "
                "and integrate/test enforcement before using strict mode.")
    return None


def unknown(reason="not-probed"):
    return {"version": 1, "probe_status": "unknown", "cgroup_mode": "unknown",
            "cpu_controller_visible": None, "memory_controller_visible": None,
            "memberships": [], "mounts": [], "current_group": None,
            "errors": [reason], "quota_enforcement": "not-implemented",
            "aggregate_cpu_quota": False, "aggregate_memory_quota": False}


def _read(path):
    with path.open("r", encoding="utf-8") as stream:
        text = stream.read(MAX_BYTES + 1)
    if len(text) > MAX_BYTES:
        raise ValueError("probe input exceeded size limit")
    return text


def _unescape(value):
    for escaped, literal in ((r"\040", " "), (r"\011", "\t"),
                             (r"\012", "\n"), (r"\134", "\\")):
        value = value.replace(escaped, literal)
    return value


def probe(proc=Path("/proc")):
    result = unknown()
    errors = result["errors"] = []
    inputs = {}
    for name in ("self/cgroup", "self/mountinfo", "cgroups"):
        try:
            inputs[name] = _read(proc / name)
        except (OSError, ValueError, UnicodeError) as error:
            errors.append(f"{name}: {type(error).__name__}")
    try:
        memberships = []
        for line in inputs.get("self/cgroup", "").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if not hierarchy.isdigit() or not path.startswith("/") or ".." in PurePosixPath(path).parts:
                raise ValueError("invalid membership")
            memberships.append({"hierarchy": int(hierarchy),
                                "controllers": controllers.split(",") if controllers else [], "path": path})
        result["memberships"] = memberships
        if not memberships:
            errors.append("no cgroup membership evidence")
        mounts = []
        for line in inputs.get("self/mountinfo", "").splitlines():
            before, after = line.split(" - ", 1)
            tail = after.split()
            if tail[0] not in {"cgroup", "cgroup2"}:
                continue
            fields = before.split()
            mounts.append({"type": tail[0], "root": _unescape(fields[3]),
                           "path": _unescape(fields[4]), "options": tail[2].split(",")})
        result["mounts"] = mounts
        versions = {item["type"] for item in mounts}
        result["cgroup_mode"] = ("hybrid" if len(versions) == 2 else
                                  "v2" if "cgroup2" in versions else
                                  "v1" if "cgroup" in versions else "none")
        enabled = set()
        controller_rows = 0
        for line in inputs.get("cgroups", "").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, hierarchy, count, active = line.split()
            controller_rows += 1
            if active == "1":
                enabled.add(name)
        if not controller_rows:
            errors.append("no controller evidence")
        # /proc/cgroups is v1-oriented. For v2 use the mounted current group's
        # visible controllers instead; neither existence nor W_OK proves delegation.
        if "cgroup2" in versions:
            for member in memberships:
                if member["hierarchy"] != 0 or member["controllers"]:
                    continue
                for mount in mounts:
                    if mount["type"] != "cgroup2":
                        continue
                    base = PurePosixPath(mount["root"])
                    try:
                        relative = PurePosixPath(member["path"]).relative_to(base)
                    except ValueError:
                        continue
                    group = Path(mount["path"]) / relative
                    visible = _read(group / "cgroup.controllers").split()
                    enabled.update(visible)
                    result["current_group"] = {
                        "path": str(group), "controllers": visible,
                        "directory_writable_observed": os.access(group, os.W_OK | os.X_OK),
                        "delegation_verified": False,
                    }
                    break
            if result["current_group"] is None:
                errors.append("v2 current group could not be resolved")
        if not errors:
            result["cpu_controller_visible"] = "cpu" in enabled
            result["memory_controller_visible"] = "memory" in enabled
            result["probe_status"] = "observed"
    except (OSError, ValueError, IndexError, UnicodeError) as error:
        errors.append(f"incomplete cgroup evidence: {type(error).__name__}")
    return result


def decode(output):
    rows = [line[len(PREFIX):] for line in output.splitlines() if line.startswith(PREFIX)]
    if len(rows) != 1 or len(rows[0]) > MAX_BYTES:
        return unknown("missing, duplicate or oversized resource evidence")
    try:
        result = json.loads(rows[0])
        if (not isinstance(result, dict) or result.get("version") != 1 or
                result.get("probe_status") not in {"observed", "unknown"} or
                any(type(result.get(key)) not in (bool, type(None)) for key in
                    ("cpu_controller_visible", "memory_controller_visible"))):
            raise ValueError("invalid resource evidence")
        # Probe output can never certify enforcement, even on a writable v2 mount.
        result.update(quota_enforcement="not-implemented", aggregate_cpu_quota=False,
                      aggregate_memory_quota=False)
        return result
    except (ValueError, TypeError):
        return unknown("invalid resource evidence")


if __name__ == "__main__":
    print(PREFIX + json.dumps(probe(), sort_keys=True))
