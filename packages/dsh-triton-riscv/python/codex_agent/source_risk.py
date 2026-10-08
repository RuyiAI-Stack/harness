"""Conservative static screening for operator candidates, NOT a Python sandbox."""
from __future__ import annotations

import ast
from pathlib import Path


BLOCKED_MODULES = {"subprocess", "socket", "requests", "httpx", "urllib", "http", "ftplib",
                   "ctypes", "cffi", "pickle", "dill", "multiprocessing"}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "open", "builtins.open",
                 "builtins.eval", "builtins.exec", "builtins.compile", "builtins.__import__",
                 "importlib.import_module", "importlib.util.spec_from_file_location",
                 "os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
                 "os.rename", "os.replace", "os.chmod", "os.chown", "os.kill", "os.killpg",
                 "os.fork", "os.open", "os.write", "shutil.rmtree", "shutil.move",
                 "torch.save", "torch.load", "numpy.save", "numpy.load"}
PATH_METHODS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "unlink",
                "rmdir", "rename", "replace", "chmod", "touch", "mkdir"}


def source_findings(source: str, path: str = "<candidate>") -> list[dict]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    findings: list[dict] = []

    def add(node, reason):
        findings.append({"path": path, "line": node.lineno, "reason": reason})

    def symbol(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return symbol(node.value) + "." + node.attr
        if isinstance(node, ast.Call):
            return symbol(node.func)
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    add(node, "external-process/network/dynamic-runtime import: " + alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = node.module + "." + alias.name
            if node.module.split(".")[0] in BLOCKED_MODULES:
                add(node, "external-process/network/dynamic-runtime import: " + node.module)
    # Follow simple aliases, including p = Path(...), without claiming general
    # data-flow analysis. Indirect imports, dependencies and obfuscation can evade this.
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = symbol(node.value)
                if value.startswith(("pathlib.", "os.", "shutil.", "builtins.", "importlib.")) or value in BLOCKED_CALLS:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = value
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = symbol(node.func)
            if (name in BLOCKED_CALLS or name.startswith(("os.exec", "os.spawn"))
                    or (name.startswith("pathlib.") and name.rsplit(".", 1)[-1] in PATH_METHODS)):
                add(node, "file/process/dynamic-code operation: " + name)
            if name in {"os.getenv", "os.environ.get"} and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and any(
                        word in value.value.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    add(node, "credential environment access")
        elif isinstance(node, ast.Subscript) and symbol(node.value) == "os.environ":
            add(node, "environment access requires manual review")
    return [
        {"path": p, "line": line, "reason": reason}
        for p, line, reason in sorted({(f["path"], f["line"], f["reason"]) for f in findings})]


def assert_source_screened(source: str, path: str = "<candidate>") -> None:
    findings = source_findings(source, path)
    if findings:
        details = "; ".join(f"{f['path']}:{f['line']} {f['reason']}" for f in findings[:12])
        raise ValueError("Operator risk screening blocked execution/proposal: " + details +
                         ". Inspect these operations; passing static screening is NOT proof of safe execution.")


def screen_operator_files(root: Path, files: list[str]) -> None:
    for relative in dict.fromkeys(files):
        path = (root.resolve() / relative).resolve()
        path.relative_to(root.resolve())
        with path.open("rb") as handle:
            raw = handle.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("operator risk screening file exceeds 1 MiB")
        assert_source_screened(raw.decode("utf-8"), relative)
