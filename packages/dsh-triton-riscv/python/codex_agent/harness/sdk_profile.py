"""Explicit SDK composition; never inherit the native UI's profile or home."""
from pathlib import Path
from dsh_multi_agent.profile import kernel, write_profile as _write_profile


def write_profile(home, entries):
    """Preserve the domain profile name while using the shared SDK writer."""
    return _write_profile(home, entries, name="triton")


def stage_composition(source, home):
    """Anchor bare host packages in the profile, retaining local plugin paths/JS tags."""
    import yaml

    class Expression(str):
        pass

    class Loader(yaml.SafeLoader):
        pass

    class Dumper(yaml.SafeDumper):
        pass

    Loader.add_constructor("tag:yaml.org,2002:js", lambda loader, node: Expression(loader.construct_scalar(node)))
    Dumper.add_representer(Expression, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:js", value))
    source = Path(source).resolve()
    entries = yaml.load(source.read_text(encoding="utf-8"), Loader=Loader)
    if not isinstance(entries, list):
        raise ValueError("SDK composition must be a list of service entries")
    from codex_agent.accounts.guard import owner_id
    if owner_id():
        # Multi-user tools are typed domain capabilities, never a shared-OS shell.
        allowed = {"sdk-jsonrpc-server", "llm-isrc", "sessions", "session-checkpoints",
                   "triton-riscv-domain-policy", "mcp-triton-riscv"}
        entries = [row for row in entries if isinstance(row, dict) and row.get("id") in allowed]
        if {row["id"] for row in entries} != allowed:
            raise ValueError("Account mode requires the bundled restricted Harness composition")

    def relocate(rows):
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("SDK service entries must be objects")
            name = row.get("name")
            if type(name) is str and name.startswith(("./", "../")):
                row["name"] = str((source.parent / name).resolve())
            if isinstance(row.get("group"), list):
                relocate(row["group"])

    relocate(entries)
    target = Path(home) / "profiles/triton/domain.yml"
    target.write_text(yaml.dump(entries, Dumper=Dumper, sort_keys=False), encoding="utf-8")
