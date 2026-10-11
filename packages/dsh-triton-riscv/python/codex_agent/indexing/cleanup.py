"""Repeatable garbage collection of obsolete vector revisions; preserves SQL audit text."""
import json

from codex_agent.storage.schema import memory_indexes as indexes, memory_index_entries as entries
from .qdrant import Qdrant


def cleanup(store, generation, *, vector_factory=Qdrant):
    index = store.db.rows(indexes, indexes.c.generation == generation)
    if not index:
        raise ValueError("unknown index generation")
    vectors = vector_factory(json.loads(index[0]["config_json"]), index[0]["collection"])
    cursor, count = 0, 0
    while True:
        rows = store.db.rows(entries, entries.c.generation == generation, entries.c.memory_id > cursor,
                             order=(entries.c.memory_id,), limit=100)
        for row in rows:
            vectors.delete_revision(store.db.workspace_id, row["memory_id"], row["revision"],
                                    older=True, include=row["desired"] == "deleted")
            count += 1
        if len(rows) < 100:
            break
        cursor = rows[-1]["memory_id"]
    return {"cases_checked": count, "generation": generation, "sql_audit_preserved": True}
