"""In-memory SQL probe; all assertions are in the Vitest caller."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles

from codex_agent.indexing import retrieval
from codex_agent.indexing.qdrant import Qdrant
from codex_agent.memory import MemoryQuery, MemoryStore
from codex_agent.memory_lineage import case_payload, case_identity, digest, SPLITTER_VERSION
from codex_agent.storage.schema import (memories, memory_indexes as indexes,
    memory_index_entries as entries, memory_index_chunks as chunks)


@compiles(LONGTEXT, "sqlite")
def text_type(element, compiler, **kwargs):
    return "TEXT"


class Database:
    workspace_id = "workspace-a"

    def __init__(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.statements = []
        for table in (memories, indexes, entries, chunks):
            table.create(self.engine)

    def predicate(self, table, *conditions):
        return (table.c.workspace_id == self.workspace_id, *conditions)

    def rows(self, table, *conditions, limit=None):
        statement = select(table).where(*self.predicate(table, *conditions)).limit(limit)
        with self.transaction() as conn:
            return [dict(row) for row in conn.execute(statement).mappings()]

    def transaction(self):
        return self.engine.begin()


data = json.load(sys.stdin)
db = Database()
store = MemoryStore.__new__(MemoryStore)
store.db = db
definition = dict(splitter=SPLITTER_VERSION, dimensions=4, embedding=dict(provider="fixture", model="fixture-model"))
index = dict(workspace_id=db.workspace_id, generation="existing-gen", fingerprint=digest(definition),
             status="active", config_json=json.dumps(definition), collection="existing_collection")
row = {c.name: None if c.nullable else "" for c in memories.columns}
row.update(workspace_id=db.workspace_id, id=1, active=1, fingerprint="source-fp", memory_type="implementation",
    operator="tanh_and_mul", summary="unverified original evidence", semantics="exponential formulation",
    confidence=0.5, confidence_grade="C", useful_count=0, source_run="original-run", outcome="failed",
    tl_ops_json="[]", environment_json="{}", evidence_json="{}", embedding_json=None)
row["case_version"] = digest(case_payload(store._row_dict(row)))
chunk_text = "exact original SQL evidence"
provenance = dict(case_id=case_identity(db.workspace_id, 1), source_field="summary",
    text_sha256=hashlib.sha256(chunk_text.encode()).hexdigest(), source={"state": "reported-not-proven"})
record = dict(workspace_id=db.workspace_id, generation=index["generation"], memory_id=1, revision=1,
    case_version=row["case_version"], chunk_id="existing-chunk", point_id="existing-point", text=chunk_text,
    provenance_json=json.dumps(provenance))
with db.transaction() as conn:
    conn.execute(memories.insert().values(row))
    conn.execute(indexes.insert().values(index))
    conn.execute(entries.insert().values(workspace_id=db.workspace_id, generation=index["generation"], memory_id=1,
        revision=1, case_version=row["case_version"], desired="present", status="ready", job_id="fixture", point_count=1))
    conn.execute(chunks.insert().values(record))
    conn.execute(memories.insert().values({**row, "workspace_id": "workspace-b", "summary": "foreign SQL evidence"}))
    conn.execute(indexes.insert().values({**index, "workspace_id": "workspace-b"}))
    conn.execute(entries.insert().values(workspace_id="workspace-b", generation=index["generation"], memory_id=1,
        revision=1, case_version=row["case_version"], desired="present", status="ready", job_id="foreign", point_count=1))
    conn.execute(chunks.insert().values({**record, "workspace_id": "workspace-b", "text": "foreign SQL evidence"}))

hit = {"id": record["point_id"], "score": 0.9, "payload": {"workspace_id": db.workspace_id,
    "generation": index["generation"], "index_fingerprint": index["fingerprint"], "memory_id": 1,
    "revision": 1, "chunk_id": record["chunk_id"], "case_version": row["case_version"], "text": "FORGED VECTOR TEXT"}}
scenario = data.get("scenario", "valid")
if scenario == "forged":
    hit["payload"][data["field"]] = data["value"]
if scenario in {"pending", "archived", "changed-parent", "changed-chunk"}:
    table, values = {
        "pending": (entries, {"status": "pending"}), "archived": (memories, {"active": 0}),
        "changed-parent": (memories, {"summary": "unversioned edit"}),
        "changed-chunk": (chunks, {"text": "unversioned chunk edit"}),
    }[scenario]
    with db.transaction() as conn:
        conn.execute(table.update().where(table.c.workspace_id == db.workspace_id).values(values))


class Provider:
    name = "fixture"
    model = "fixture-model"
    max_input_tokens = 2048

    def count_tokens(self, text):
        return len(text)

    def embed(self, texts):
        return [data.get("vector", [1, 0, 0, 0]) for _ in texts]


class Transport:
    calls = []

    def query(self, vector, workspace, generation, fingerprint, *, limit, offset=0):
        self.calls.append(dict(workspace=workspace, generation=generation, fingerprint=fingerprint, limit=limit, offset=offset))
        if scenario == "generation-switch":
            with db.transaction() as conn:
                conn.execute(indexes.update().where(indexes.c.workspace_id == db.workspace_id).values(status="retired"))
        if scenario == "archived-between-pages" and offset:
            with db.transaction() as conn:
                conn.execute(memories.update().where(memories.c.workspace_id == db.workspace_id).values(active=0))
        if scenario in {"budget", "archived-between-pages"}:
            return [copy.deepcopy(hit) for _ in range(limit)]
        return [copy.deepcopy(hit)]


service = Transport()
trace = {}
result = {}
try:
    with patch.object(retrieval, "runtime_config", return_value=SimpleNamespace(indexing=SimpleNamespace(enabled=True))):
        result["items"] = retrieval.retrieve(store, MemoryQuery("tanh_and_mul", "exponential formulation", ""),
            limit=data.get("limit", 5), trace=trace, exclude_source_runs=data.get("exclude", []),
            provider_factory=lambda _: Provider(), vector_factory=lambda *_: service)
except Exception as error:
    result["error"] = type(error).__name__
result.update(trace=trace, calls=service.calls, transport_module=Qdrant.__mro__[1].__module__)
print(json.dumps(result))
