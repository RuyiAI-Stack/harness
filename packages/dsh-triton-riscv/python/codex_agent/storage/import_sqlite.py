"""Import a stopped legacy SQLite workspace into a new MySQL workspace.

Sources are opened read-only in a consistent SQLite read transaction. The MySQL
import commits once, refuses existing workspaces, and never rewrites evidence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from codex_agent.storage.database import WorkspaceDatabase
from codex_agent.storage.schema import TABLES, workspaces, counters, meta
from codex_agent.memory import MemoryStore, SCHEMA_VERSION, memory_chunks


@contextmanager
def snapshot(path):
    source = Path(path).resolve(strict=True)
    connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def read_rows(connection, name):
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
        return None
    if name == "messages":
        return [dict(row) for row in connection.execute('SELECT *, rowid AS ordinal FROM messages ORDER BY rowid')]
    # Names come exclusively from the fixed schema, never a caller SQL fragment.
    return [dict(row) for row in connection.execute('SELECT * FROM "' + name + '"')]


def import_sqlite(workspace: Path, *, platform=None, memory=None, references=None):
    sources = {"platform": platform, "memory": memory, "references": references}
    if not any(sources.values()):
        raise ValueError("supply at least one SQLite source")
    groups = {"platform": ("sessions", "messages", "runs", "events", "context_checkpoints"),
              "memory": ("memories", "memory_chunks"),
              "references": ("references_catalog",)}
    data, provenance = {}, []
    old_version = None
    with ExitStack() as stack:
        for kind, source in sources.items():
            if source is None:
                continue
            source = Path(source).resolve(strict=True)
            conn = stack.enter_context(snapshot(source))
            if kind == "memory":
                versions = read_rows(conn, "metadata") or []
                old_version = next((int(r["value"]) for r in versions if r["key"] == "schema_version"), None)
                if old_version is not None and old_version > SCHEMA_VERSION:
                    raise ValueError("source memory schema is newer than this importer")
            for name in groups[kind]:
                rows = read_rows(conn, name)
                if rows is None and name not in {"context_checkpoints", "memory_chunks"}:
                    raise ValueError("missing required source table: " + name)
                data[name] = rows or []
            provenance.append({"kind": kind, "path": str(source),
                               "main_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                               "access": "read-only snapshot; committed WAL included"})
        if "memories" in data:
            if old_version != SCHEMA_VERSION:
                data["memory_chunks"] = []
            indexed = {row["memory_id"] for row in data["memory_chunks"]}
            for row in data["memories"]:
                if row["id"] in indexed:
                    continue
                row.setdefault("embedding_json", None)
                item = MemoryStore._row_dict(row)
                for part in memory_chunks(item):
                    data["memory_chunks"].append(dict(memory_id=row["id"], kind=part.kind,
                        position=part.position, text=part.text, source_field=part.source_field))
                # Parent IDs/provenance stay unchanged. Old vectors describe a different layout.
                for key in ("embedding_json", "embedding_provider", "embedding_model", "embedding_dim"):
                    row[key] = None
        db = WorkspaceDatabase(workspace)
        with db.transaction() as target:
            if db.rows(workspaces, connection=target, lock=True):
                raise ValueError("target workspace already exists; import into a new workspace, never overwrite")
            db.insert(target, workspaces, {"root": str(db.root)})
            event_next = {}
            for row in data.get("events", []):
                event_next[row["run_id"]] = max(event_next.get(row["run_id"], 0), row["sequence"] + 1)
            for row in data.get("runs", []):
                row["next_sequence"] = event_next.get(row["id"], 0)
            for name, rows in data.items():
                table = TABLES[name]
                allowed = set(table.c.keys()) - {"workspace_id"}
                for source_row in rows:
                    row = dict(source_row)
                    if set(row) - allowed:
                        raise ValueError("unrecognized source columns in " + name)
                    if name in {"memories", "memory_chunks"}:
                        for key in ("embedding_json", "embedding_provider", "embedding_model", "embedding_dim"):
                            row.setdefault(key, None)
                    if name == "memory_chunks":
                        row.setdefault("source_field", "")
                    if name == "sessions":
                        row.setdefault("pinned", 0)
                    db.insert(target, table, row)
                actual = db.rows(table, connection=target)
                if len(actual) != len(rows):
                    raise ValueError("import count mismatch in " + name)
                # Verify every supplied field, not only counts or passed labels.
                keys = [c.name for c in table.primary_key if c.name != "workspace_id"]
                indexed = {tuple(r[k] for k in keys): r for r in actual}
                for row in rows:
                    copy = indexed[tuple(row[k] for k in keys)]
                    if any(copy[k] != v for k, v in row.items()):
                        raise ValueError("import field mismatch in " + name)
            for name, source, field in (("messages", "messages", "ordinal"),
                                        ("events", "events", "id"), ("memories", "memories", "id")):
                db.insert(target, counters, {"name": name,
                    "value": max((r[field] for r in data.get(source, [])), default=0)})
            if memory:
                db.insert(target, meta, {"key": "schema_version", "value": str(SCHEMA_VERSION)})
            receipt = {"workspace": str(db.root), "workspace_id": db.workspace_id,
                       "sources": provenance, "counts": {k: len(v) for k, v in data.items()},
                       "old_memory_version": old_version, "evidence_paths_rewritten": False,
                       "labels_reaudited": False}
            db.insert(target, meta, {"key": "sqlite_import", "value": json.dumps(receipt, ensure_ascii=False)})
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    for key in ("platform", "memory", "references"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    print(json.dumps(import_sqlite(args.workspace, platform=args.platform, memory=args.memory,
                                   references=args.references), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
