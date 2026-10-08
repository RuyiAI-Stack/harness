"""Copy a historical database and referenced evidence without opening the original MemoryStore."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from codex_agent.memory import MemoryStore

ALLOWED_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".diff", ".patch", ".py", ".mlir", ".ll"}
PATH_KEYS = {"source", "source_run", "path", "log_path", "receipt_path", "patch_path", "final_result_path", "task_file"}


def migrate_memory(source: Path, source_root: Path, destination: Path) -> dict:
    source, source_root, destination = source.resolve(), source_root.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError("destination must be new; no history will be overwritten")
    if not source.is_file():
        raise FileNotFoundError(source)
    source.relative_to(source_root)
    destination.mkdir(parents=True, mode=0o700)
    database = destination / "memory.sqlite3"
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"source_database": str(source), "source_root": str(source_root),
                "source_database_sha256_before": original_hash, "records": [], "files": [], "gaps": [],
                "note": "Historical labels preserved, not newly audited. Missing evidence is not a verified result."}
    # SQLite backup includes committed WAL state; plain file copying may lose it.
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(database) as copied:
            original.backup(copied)
    database.chmod(0o600)
    copied_files = {}
    total_bytes = 0

    def archive(value: str) -> str:
        nonlocal total_bytes
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = source_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(source_root):
            manifest["gaps"].append({"source": value, "reason": "outside approved source root"})
            return value
        if not candidate.exists():
            if value.startswith(("agent-results/", "/")):
                manifest["gaps"].append({"source": value, "reason": "missing"})
            return value
        paths = sorted(candidate.rglob("*")) if candidate.is_dir() else [candidate]
        for item in paths:
            if not item.is_file() or item.suffix not in ALLOWED_SUFFIXES or item.is_symlink():
                continue
            resolved = item.resolve()
            if not resolved.is_relative_to(source_root) or any(p.startswith(".") for p in item.relative_to(source_root).parts):
                continue
            if str(item) in copied_files:
                continue
            size = item.stat().st_size
            if size > 16 * 1024**2 or total_bytes + size > 100 * 1024**2:
                manifest["gaps"].append({"source": str(item), "reason": "archive size budget exceeded"})
                continue
            target = destination / "evidence" / item.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
            target.chmod(0o600)
            total_bytes += size
            entry = {"source": str(item), "copy": str(target), "bytes": size,
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            manifest["files"].append(entry)
            copied_files[str(item)] = str(target)
        if candidate.is_dir() and (destination / "evidence" / candidate.relative_to(source_root)).exists():
            return str(destination / "evidence" / candidate.relative_to(source_root))
        return copied_files.get(str(candidate), value)

    def relocate(value, key=""):
        if isinstance(value, dict):
            return {k: relocate(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v, key) for v in value]
        if isinstance(value, str) and key in PATH_KEYS and value:
            return archive(value)
        return value

    # Only the independent copy may undergo schema upgrades or evidence relocation.
    with MemoryStore(database) as store:
        rows = store.connection.execute("SELECT id, source_run, evidence_json FROM memories").fetchall()
        for row in rows:
            evidence = relocate(json.loads(row["evidence_json"]))
            original_run = row["source_run"]
            new_run = archive(original_run) if original_run.startswith(("agent-results/", "/")) else original_run
            manifest["records"].append({"memory_id": row["id"], "original_source_run": original_run, "source_run": new_run})
            store.connection.execute("UPDATE memories SET source_run=?, evidence_json=? WHERE id=?",
                (new_run, json.dumps(evidence, ensure_ascii=False), row["id"]))
        # Rebuild only this copy's chunks after changing provenance; old vectors are stale.
        store.connection.execute("DELETE FROM memory_chunks")
        store.connection.execute("UPDATE memories SET embedding_json=NULL, embedding_provider=NULL, embedding_model=NULL, embedding_dim=NULL")
        store.connection.commit()
    with MemoryStore(database) as store:
        manifest["stats"] = store.stats()
    manifest["source_database_sha256_after"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["source_file_unchanged"] = manifest["source_database_sha256_after"] == original_hash
    manifest["database"] = str(database)
    manifest["archived_bytes"] = total_bytes
    (destination / "migration-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    result = migrate_memory(args.source, args.source_root, args.destination)
    print(json.dumps({"database": result["database"], "records": len(result["records"]),
                      "archived_files": len(result["files"]), "gaps": len(result["gaps"]),
                      "source_file_unchanged": result["source_file_unchanged"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
