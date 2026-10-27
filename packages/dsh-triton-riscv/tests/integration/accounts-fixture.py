"""Exercise production APIs against disposable services; Vitest owns assertions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import select

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.guard import check_workspace, reviewer_identity
from codex_agent.accounts.schema import users
from codex_agent.accounts.store import Accounts, AccountError, digest
from codex_agent.accounts.workspaces import Workbenches
from codex_agent.artifacts import ArtifactStore
from codex_agent.harness.sdk_profile import kernel, stage_composition, write_profile
from codex_agent.memory import MemoryRecord, MemoryStore
from codex_agent.platform.store import PlatformStore
from codex_agent.platform.api import create_app
from codex_agent.orchestration.store import WorkflowStore
from codex_agent.storage.cache import ReadCache
from codex_agent.tasks.store import JobStore


@contextmanager
def bound(manager, uid):
    root = manager.provision(uid)
    with patch.dict(os.environ, manager.environment(root, uid, "fixture-gateway-key"), clear=True):
        yield root


def rejected(call):
    try:
        call()
        return False
    except (PermissionError, ValueError, KeyError, AccountError):
        return True


def run():
    result = {}
    with tempfile.TemporaryDirectory(prefix="triton-identity-") as directory:
        base = Path(directory)
        template = base / "template"
        operator = template / "python/examples/flaggems/fixture.py"
        operator.parent.mkdir(parents=True)
        operator.write_text("def fixture(x):\n    return x\n")
        relative = str(operator.relative_to(template))
        checksum = hashlib.sha256(operator.read_bytes()).hexdigest()
        (template / "template-manifest.json").write_text(json.dumps({relative: checksum}))
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps({"schemaVersion": 1, "repoRoot": str(template), "stateDir": str(base / "unused")})
        config = AccountConfig(origin="http://127.0.0.1:8766", data_dir=base / "state",
                               template=template, allow_local_http=True, max_accounts=50)
        accounts = Accounts(config)
        manager = Workbenches(accounts)
        gateway = create_gateway(config, accounts=accounts, workbenches=manager)
        headers = {"Origin": config.origin, "Content-Type": "application/json"}
        password = secrets.token_urlsafe(20)
        names = ["alice_" + secrets.token_hex(4), "bob_" + secrets.token_hex(4)]
        with TestClient(gateway, base_url=config.origin) as client:
            result["anonymous"] = client.get("/api/sessions").status_code
            result["bad_host"] = client.get("/api/auth/me", headers={"Host": "evil.invalid"}).status_code
            result["bad_origin"] = client.post("/api/auth/login", json={}, headers={"Origin": "https://evil.invalid"}).status_code
            invite = accounts.invite()
            result["registration"] = client.post("/api/auth/register", headers=headers, json={
                "username": names[0], "password": password, "invitation": invite}).status_code
            result["invite_replay"] = client.post("/api/auth/register", headers=headers, json={
                "username": "other_" + secrets.token_hex(4), "password": password, "invitation": invite}).status_code
            result["spoof_registration"] = client.post("/api/auth/register", headers=headers, json={
                "username": "spoof", "password": password, "invitation": accounts.invite(), "user_id": "chosen"}).status_code
            bob = accounts.register(names[1], password, accounts.invite())
            shared_invite = accounts.invite()
            def signup(index):
                try:
                    accounts.register("race_" + secrets.token_hex(4), password, shared_invite)
                    return 201
                except AccountError as error:
                    return error.status
            with ThreadPoolExecutor(max_workers=4) as pool:
                result["concurrent_invite"] = sorted(pool.map(signup, range(4)))
            response = client.post("/api/auth/login", json={"username": names[0], "password": password}, headers=headers)
            alice = response.json()["user"]
            token_a = client.cookies.get(config.cookie_name)
            cookie_a = {"Cookie": config.cookie_name + "=" + token_a}
            result["cookie_flags"] = {"httponly": "HttpOnly" in response.headers["set-cookie"],
                                      "strict": "SameSite=strict" in response.headers["set-cookie"]}
            csrf_a = {**headers, "X-CSRF-Token": alice["csrf_token"], **cookie_a}
            result["invalid_credentials"] = [client.post("/api/auth/login", headers=headers, json={
                "username": name, "password": password + "wrong"}).status_code
                for name in (names[0], "missing_" + secrets.token_hex(4))]
            result["missing_csrf"] = client.post("/api/sessions", json={"title": "bad"}, headers=headers).status_code
            response = client.post("/api/sessions", json={"title": "alice-private"}, headers=csrf_a)
            if response.status_code != 200:
                logs = [p.read_text()[-5000:] for p in config.data_dir.glob("*.service.log")]
                raise RuntimeError(json.dumps({"response": response.text, "private_logs": logs}))
            session_a = response.json()["id"]
            root_a = manager.provision(alice["id"])
            with bound(manager, alice["id"]) as root:
                store = PlatformStore(root)
                message = store.add_message(session_a, "user", "private-content-a")
                store.save_context_checkpoint(session_a, summary="private-checkpoint-a",
                    through_message_id=message["id"], source_ids=[message["id"]], metrics={})
                run_a = store.create_run(session_a, "fixture", "fixture", {}, status="completed")["id"]
                workflow_a = WorkflowStore(root).create(dict(session_id=session_a, kind="create", operator="fixture", requirement="identity"), request_id="fixture-workflow")["id"]
                jobs_a = JobStore(root)
                job_a = jobs_a.submit("memory", {"fixture": True}, "same-request")
                envelope_a = jobs_a.envelope(jobs_a.claim_publication())
                claimed = jobs_a.claim(envelope_a, "memory")
                artifact_a = ArtifactStore(root).put(jobs_a, job_a["id"], claimed["owner"], "private.txt", b"private-artifact-a")
                saved_config = os.environ["TRITON_RISCV_CONFIG"]
                cfg = json.loads(saved_config)
                cfg["queue"]["enabled"] = True
                os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
                with TestClient(create_app(root)) as private:
                    result["owner_artifact"] = private.get(f"/api/jobs/{job_a['id']}/artifacts/{artifact_a}", headers={"x-triton-gateway": "fixture-gateway-key"}).text
                os.environ["TRITON_RISCV_CONFIG"] = saved_config
                memory = MemoryStore(root)
                memory.add(MemoryRecord(memory_type="failure-diagnosis", operator="fixture", semantics="private-a",
                    pytorch_reference="identity", summary="private-memory-a", outcome="failed",
                    confidence_grade="C", source_run="fixture-a"))
                result["memory_a"] = len(memory._stored())
                cache_a = ReadCache(jobs_a.db).key("detail", {"id": 1}, 1)
                result["reviewer_bound"] = reviewer_identity("forged-reviewer") == "account:" + alice["id"]
                profile = base / "profile"
                write_profile(profile, kernel())
                import codex_agent.harness
                stage_composition(Path(codex_agent.harness.__file__).parent / "cordis.yml", profile)
                composition = (profile / "profiles/triton/domain.yml").read_text()
                result["restricted_tools"] = all(x not in composition for x in ("id: bash", "id: fs-local", "id: agent-instructions"))
            result["marker_blocks_legacy"] = rejected(lambda: check_workspace(root_a))
            entry = manager.get(alice["id"])
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(entry["socket"])), base_url="http://private") as raw:
                result["private_bypass"] = raw.get("/api/sessions").status_code
            result["owner_read"] = client.get("/api/sessions/" + session_a, headers=cookie_a).status_code
            result["owner_sse"] = client.get(f"/api/runs/{run_a}/events/stream", headers=cookie_a).status_code
            token_b, user_b = accounts.login(names[1], password)
            cookie_b = {"Cookie": config.cookie_name + "=" + token_b}
            csrf_b = {**headers, **cookie_b, "X-CSRF-Token": user_b["csrf_token"]}
            result["bob_sessions"] = client.get("/api/sessions", headers=cookie_b).json()
            # Simulate an old A tab whose browser cookie was replaced by B's login.
            stale = {**csrf_b, "X-Triton-Account": alice["id"]}
            result["stale_tab"] = [
                client.get("/api/sessions", headers=stale).status_code,
                client.post("/api/sessions", headers=stale, json={"title": "must-not-create"}).status_code,
                client.get(f"/api/runs/{run_a}/events/stream?account_id={alice['id']}", headers=cookie_b).status_code,
                client.post("/api/auth/logout", headers=stale, json={}).status_code,
            ]
            result["stale_tab_no_effects"] = (client.get("/api/sessions", headers=cookie_b).json() == []
                and client.get("/api/auth/me", headers=cookie_b).json()["user"]["id"] == bob["id"])
            result["bound_owner_read"] = client.get("/api/sessions/" + session_a,
                headers={**cookie_a, "X-Triton-Account": alice["id"]}).status_code
            paths = [f"sessions/{session_a}", f"sessions/{session_a}/approvals", f"runs/{run_a}",
                     f"runs/{run_a}/events", f"runs/{run_a}/events/stream",
                     f"operator-workflows/{workflow_a}", f"operator-workflows/{workflow_a}/usage"]
            result["cross_reads"] = [client.get("/api/" + path, headers={**cookie_b, "X-User-ID": alice["id"]}).status_code for path in paths]
            result["cross_delete"] = client.delete("/api/sessions/" + session_a, headers=csrf_b).status_code
            result["cross_pin"] = client.patch("/api/sessions/" + session_a, headers=csrf_b, json={"pinned": True}).status_code
            with bound(manager, bob["id"]) as root_b:
                jobs_b = JobStore(root_b)
                job_b = jobs_b.submit("memory", {"fixture": True}, "same-request")
                result["distinct_jobs"] = job_b["id"] != job_a["id"]
                result["foreign_envelope"] = rejected(lambda: jobs_b.claim(envelope_a, "memory"))
                result["foreign_job"] = rejected(lambda: jobs_b.get(job_a["id"]))
                saved_config = os.environ["TRITON_RISCV_CONFIG"]
                cfg = json.loads(saved_config)
                cfg["queue"]["enabled"] = True
                os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
                with TestClient(create_app(root_b)) as private:
                    result["foreign_artifact_http"] = private.get(f"/api/jobs/{job_a['id']}/artifacts/{artifact_a}", headers={"x-triton-gateway": "fixture-gateway-key"}).status_code
                os.environ["TRITON_RISCV_CONFIG"] = saved_config
                result["checkpoint_b"] = PlatformStore(root_b).get_context_checkpoint(session_a)
                result["memory_b"] = len(MemoryStore(root_b)._stored())
                result["cache_isolated"] = ReadCache(jobs_b.db).key("detail", {"id": 1}, 1) != cache_a
                result["artifact_not_in_b"] = not (ArtifactStore(root_b).folder / hashlib.sha256(b"private-artifact-a").hexdigest()).exists()
                result["wrong_workspace"] = rejected(lambda: check_workspace(root_a))
                (root_b / relative).write_text("bob's change")
            result["files_isolated"] = hashlib.sha256((root_a / relative).read_bytes()).hexdigest() == checksum
            result["template_unchanged"] = hashlib.sha256(operator.read_bytes()).hexdigest() == checksum
            key = accounts.prefix + "session:" + digest(token_a)
            accounts.redis.expire(key, 10)
            client.get("/api/auth/me", headers=cookie_a)
            result["poll_no_renewal"] = accounts.redis.ttl(key) <= 10
            result["logout"] = client.post("/api/auth/logout", json={}, headers=csrf_a).status_code
            result["logout_replay"] = client.get("/api/sessions", headers=cookie_a).status_code
            token_a, user_a = accounts.login(names[0], password)
            second_token, _ = accounts.login(names[0], password)
            accounts.change_password(user_a, password, password + "new")
            result["password_revokes_all"] = all(rejected(lambda t=t: accounts.authenticate(t)) for t in (token_a, second_token))
            result["old_password_rejected"] = rejected(lambda: accounts.login(names[0], password))
            new_token, _ = accounts.login(names[0], password + "new")
            accounts.redis.delete(accounts.prefix + "session:" + digest(new_token))
            result["expired"] = rejected(lambda: accounts.authenticate(new_token))
            tokens = [accounts.login(names[0], password + "new")[0] for _ in range(config.max_sessions + 1)]
            result["session_cap"] = sum(not rejected(lambda t=t: accounts.authenticate(t)) for t in tokens) == config.max_sessions
            accounts.set_status(names[0], False)
            result["disabled_session"] = rejected(lambda: accounts.authenticate(tokens[-1]))
            result["disabled_login"] = rejected(lambda: accounts.login(names[0], password + "new"))
            with patch.dict(os.environ, manager.environment(root_a, alice["id"], "fixture-key"), clear=True):
                result["disabled_worker"] = rejected(lambda: jobs_a.heartbeat(job_a["id"], claimed["owner"]))
            original_client = httpx.AsyncClient
            class RevokingStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b"data: before-logout\n\n"
                    accounts.logout(token_b, bob["id"])
                    yield b"data: must-not-arrive\n\n"
            def stream_client(**kwargs):
                kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200,
                    headers={"content-type": "text/event-stream"}, stream=RevokingStream()))
                return original_client(**kwargs)
            with patch("codex_agent.accounts.api.httpx.AsyncClient", side_effect=stream_client):
                result["revoked_stream"] = client.get("/api/fixture-stream", headers=cookie_b).text
            token_b, _ = accounts.login(names[1], password)
            cookie_b = {"Cookie": config.cookie_name + "=" + token_b}
            with patch.object(accounts.redis, "get", side_effect=RedisConnectionError("fixture unavailable")):
                result["redis_outage"] = client.get("/api/sessions", headers=cookie_b).status_code
            with accounts.engine.connect() as conn:
                hashed = conn.execute(select(users.c.password_hash).where(users.c.id == bob["id"])).scalar_one()
            result["argon2id"] = hashed.startswith("$argon2id$") and password not in hashed
            with patch.object(accounts.redis, "config_get", return_value={"maxmemory-policy": "allkeys-lfu"}):
                result["lfu_rejected"] = rejected(accounts.health)
            rate_key = secrets.token_hex(16)
            accounts.limit("fixture", rate_key, 1, 30)
            result["rate_limited"] = rejected(lambda: accounts.limit("fixture", rate_key, 1, 30))
            result["https_required"] = rejected(lambda: config.model_validate({**config.model_dump(), "origin": "http://public.example"}))
            result["secure_cookie"] = config.model_copy(update={"origin": "https://app.example"}).cookie_name.startswith("__Host-")
            if os.environ.get("TRITON_TEST_AMQP_URL"):
                worker_user = accounts.register("worker_" + secrets.token_hex(4), password, accounts.invite())
                cfg = json.loads(os.environ["TRITON_RISCV_CONFIG"])
                cfg["queue"] = {"enabled": True}
                with patch.dict(os.environ, {"TRITON_RISCV_CONFIG": json.dumps(cfg), "TRITON_AMQP_URL": os.environ["TRITON_TEST_AMQP_URL"]}):
                    entry = manager.get(worker_user["id"])
                    time.sleep(1)
                    result["account_workers"] = len(entry["processes"]) == 6 and all(p.poll() is None for p in entry["processes"])
                    accounts.set_status(worker_user["username"], False)
                    for _ in range(60):
                        if all(p.poll() is not None for p in entry["processes"]):
                            break
                        time.sleep(.1)
                    result["disable_stops_workers"] = all(p.poll() is not None for p in entry["processes"])
    return result


print(json.dumps(run()))
