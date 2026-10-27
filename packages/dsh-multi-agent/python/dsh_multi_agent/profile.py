"""Explicit SDK composition without parent profiles, tools or workspace context."""
import json
import re
from pathlib import Path


def kernel():
    rows = [
        ("timer", "cordis-plugin-timer"), ("llm", "dsh-llm"),
        ("session", "dsh-session"), ("session-projection", "dsh-session-projection"),
        ("tools", "dsh-tools"), ("agent", "dsh-agent"),
        ("llm-retry", "dsh-llm-retry"), ("jobs", "dsh-jobs-local"),
        ("invariants", "dsh-invariants"), ("session-invariant", "dsh-session/invariant"),
        ("agent-invariant", "dsh-agent/invariant"), ("scope-invariant", "dsh-scope/invariant"),
        ("agent-loop-invariant", "dsh-agent-loop/invariant"),
    ]
    return [{"id": identifier, "name": "@deepseek-ai/" + name} for identifier, name in rows] + [
        {"id": "system-prompt", "name": "@deepseek-ai/dsh-system-prompt",
         "config": {"includeHarnessIdentity": False, "includeRuntimeContext": False}},
        {"id": "agent-loop", "name": "@deepseek-ai/dsh-agent-loop", "config": {"agents": []}},
    ]


def write_profile(home, entries, *, name="isolated-task"):
    """The caller owns a fresh home; profile names cannot escape that home."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid SDK profile name")
    profile = Path(home) / "profiles" / name
    profile.mkdir(parents=True, mode=0o700)
    manifest = {"name": name + "-sdk-profile", "private": True,
                "dsh": {"profile": {"bundles": [], "patchReload": "startup"}}}
    (profile / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    (profile / "cordis.patch.yml").write_text(json.dumps([{"insert": entries}]), encoding="utf-8")
    return {"dsh_home": str(home), "profile": name}


def composition(settings, policy, session_root, *, policy_identity=None):
    return kernel() + [
        {"id": "sdk-jsonrpc-server", "name": "@deepseek-ai/dsh-sdk-jsonrpc-server"},
        {"id": "role-model", "name": "@deepseek-ai/dsh-llm-pi-ai", "config": {"providers": {
            settings.provider: {"apiKeyEnv": settings.api_key_env, "api": "openai-responses",
                "baseURL": settings.base_url, "models": [{"id": settings.model, "name": settings.model}]}}}},
        {"id": "sessions", "name": "@deepseek-ai/dsh-session-persistence-jsonl", "config": {"root": str(session_root)}},
        {"id": "checkpoints", "name": "@deepseek-ai/dsh-session-checkpoint-policy"},
        {"id": "role-policy", "name": str(Path(__file__).with_name("role-policy.mjs")),
         "config": {"systemPrompt": policy, "identity": policy_identity or {}}},
    ]
