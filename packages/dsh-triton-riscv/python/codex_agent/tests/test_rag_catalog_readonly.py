"""SQL-backed catalog checks in an ephemeral database; no runtime/embedding calls."""
import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text, event, union_all
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles

from codex_agent.rag_catalog.store import CatalogStore
from codex_agent.knowledge.schema import documents, versions, chunks
from codex_agent.knowledge.file_schema import sources, locations
from codex_agent.knowledge.lifecycle import controls
from codex_agent.storage.schema import (metadata, memories, memory_versions, counters,
    chunks as memory_chunks, memory_indexes, memory_index_entries, memory_index_chunks)
from codex_agent.tasks.schema import jobs
from codex_agent.tasks.store import Conflict
from codex_agent.storage.database import StorageError
from codex_agent.memory_lineage import SPLITTER_VERSION, digest


@compiles(LONGTEXT, "sqlite")
def sqlite_text(_type, _compiler, **_kw):
    return "TEXT"


FP = "f" * 64
CASE = "1" * 64
OLD_CASE = "2" * 64
KEY = b"isolated-test-catalog-private-key"
INDEX_DEFINITION = {"splitter": SPLITTER_VERSION}


@pytest.fixture
def fixture(tmp_path):
    db = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(db, tables=[documents, versions, chunks, sources, locations, controls,
        jobs, memories, memory_versions, memory_chunks, memory_indexes, memory_index_entries, memory_index_chunks, counters])
    def scope(name, kind):
        root = (tmp_path / name).resolve()
        return dict(id=hashlib.sha256(str(root).encode()).hexdigest(), root=root, label=name, kind=kind, writable=True)
    shared = scope("shared", "knowledge_document")
    team = scope("team", "memory_case")
    denied = scope("denied", "memory_case")
    stamp = "2026-11-11T12:00:00Z"
    def add_doc(conn, identifier, *, latest=1, published=1, body="测试正文", state="ready"):
        conn.execute(documents.insert().values(workspace_id=shared["id"], id=identifier,
            latest_version=latest, published_version=published, created_at=stamp, updated_at=stamp))
        for version in range(1, latest + 1):
            conn.execute(versions.insert().values(workspace_id=shared["id"], document_id=identifier,
                version=version, title="同名资料", source="测试来源", content=body if version == latest else "旧版正文 保留原句",
                source_hash=str(version) * 64, state=state if version == latest else "ready", definition_json="{}",
                fingerprint=FP, job_id=None, chunk_count=1, error=None, created_at=stamp))
            conn.execute(chunks.insert().values(workspace_id=shared["id"], document_id=identifier,
                version=version, id=f"chunk-{identifier}-{version}", position=0, text=f"片段 {version}"))
    def add_memory(conn, sid, identifier, *, active=1):
        conn.execute(memories.insert().values(workspace_id=sid, id=identifier, fingerprint=hashlib.sha256(str(identifier).encode()).hexdigest(),
            memory_type="failure-diagnosis", operator="同名资料", semantics="输入步长异常",
            pytorch_reference="reference(x)", tl_ops_json="[]", failure_stage="compile", error_signature="shape",
            environment_json='{"api_key":"NEVER-RETURN"}', summary="已检查输入步长", searchable_text="正文包含输入步长异常",
            outcome="failed", evidence_json='{"summary":"记录证据","embedding":[1234],"api_key":"NEVER-RETURN"}',
            confidence_grade="C", confidence=.5, source_run="/private/task/record-a", test_sha256=None,
            active=active, archive_reason=None, created_at=stamp, updated_at=stamp, last_used_at=None, useful_count=0,
            embedding_json="[999999]", embedding_provider="test", embedding_model="test", embedding_dim=1, case_version=CASE))
        for version in [CASE, OLD_CASE]:
            payload = dict(semantics="旧记忆" if version == OLD_CASE else "当前记忆", summary="原始记录",
                           outcome="failed", source_run="/private/task/record-a", evidence={"summary":"快照", "embedding":[999999]})
            conn.execute(memory_versions.insert().values(workspace_id=sid, memory_id=identifier,
                case_version=version, payload_json=json.dumps(payload), captured_at=stamp))
        conn.execute(memory_chunks.insert().values(workspace_id=sid, memory_id=identifier, kind="summary", position=0,
            text="当前记忆片段", source_field="summary", embedding_json="[999999]", embedding_provider="test",
            embedding_model="test", embedding_dim=1, case_version=CASE, chunk_id=f"m-{identifier}", unit_id="unit",
            part_index=0, part_count=1, provenance_json="{}"))
    with db.begin() as conn:
        add_doc(conn, "a" * 32, latest=2, published=1, body="最新草稿 100% 覆盖", state="draft")
        add_doc(conn, "b" * 32, published=None, body="待处理文本", state="pending")
        add_doc(conn, "c" * 32, published=None, body="回收站内容")
        conn.execute(controls.insert().values(workspace_id=shared["id"], document_id="c" * 32,
            revision=1, deleted_at=stamp, actor_id="admin"))
        add_memory(conn, team["id"], 1)
        add_memory(conn, team["id"], 2, active=0)
        add_memory(conn, denied["id"], 1)
        conn.execute(sources.insert().values(workspace_id=shared["id"], document_id="a" * 32, version=1,
            file_id="e" * 32, filename="missing-original.pdf", format="pdf", sha256="9" * 64,
            parser_version="fixture", reviewed=1, sections_json='[{"id":"s1","locator":"第1页","text":"旧版正文 保留原句"}]', warnings_json="[]"))
        for value in [shared, team, denied]:
            conn.execute(counters.insert().values(workspace_id=value["id"], name="cache-version:rag-catalog", value=1))
    yield dict(db=db, shared=shared, team=team, denied=denied)
    db.dispose()


def store(f, **kw):
    return CatalogStore([f["shared"], f["team"]], cursor_key=KEY, database_engine=f["db"],
                        memory_mode=kw.pop("memory_mode", "legacy"), knowledge_fingerprint=FP, **kw)


def item(s, kind, source):
    return next(v for v in s.catalog()["items"] if v["source_kind"] == kind and v["source_id"] == str(source))


def test_union_lists_both_sources_and_never_leaks_vectors_or_denied_scope(fixture):
    result = store(fixture).catalog()
    assert result["total"] == 4
    assert {x["source_kind"] for x in result["items"]} == {"memory_case", "knowledge_document"}
    serialized = json.dumps(result)
    assert "999999" not in serialized and "embedding" not in serialized and "NEVER-RETURN" not in serialized
    assert fixture["denied"]["id"] not in serialized
    assert "private/task" not in serialized


def test_mysql_compile_uses_bound_scope_and_literal_query(fixture):
    s = store(fixture)
    view = union_all(s._knowledge_query(fixture["shared"]), s._memory_query(fixture["team"])).subquery()
    statement = view.select().where(*s._conditions(view, "100%_", "all", "all", "all", "active")).limit(26)
    compiled = statement.compile(dialect=mysql.dialect())
    assert fixture["shared"]["id"] in compiled.params.values()
    assert fixture["team"]["id"] in compiled.params.values()
    assert "100/%/_" in compiled.params.values()
    assert "embedding_json" not in str(compiled)


def test_current_memory_detail_never_selects_embedding_or_environment_columns(fixture):
    observed = []
    def record(_conn, _cursor, statement, _params, _context, _executemany):
        observed.append(statement)
    event.listen(fixture["db"], "before_cursor_execute", record)
    try:
        s = store(fixture)
        s.detail(item(s, "memory_case", 1)["item_id"])
    finally:
        event.remove(fixture["db"], "before_cursor_execute", record)
    assert all("embedding_json" not in q and "environment_json" not in q for q in observed)


def test_stale_and_ready_versions_cannot_be_indexed_in_place(fixture):
    s = store(fixture)
    original = item(s, "knowledge_document", "a" * 32)
    assert "index" in original["capabilities"]
    for state in ["stale", "ready"]:
        with fixture["db"].begin() as conn:
            conn.execute(versions.update().where(versions.c.workspace_id == fixture["shared"]["id"],
                versions.c.document_id == "a" * 32, versions.c.version == 2).values(state=state))
        current = item(s, "knowledge_document", "a" * 32)
        assert current["index_state"] == state
        assert "index" not in current["capabilities"]
        assert current["serving_version"] == "1" and current["retrieval_state"] == "eligible"


def test_malformed_source_metadata_fails_explicitly_without_claiming_empty_success(fixture):
    s = store(fixture)
    source_id = item(s, "memory_case", 1)["item_id"]
    with fixture["db"].begin() as conn:
        conn.execute(memories.update().where(memories.c.workspace_id == fixture["team"]["id"],
            memories.c.id == 1).values(evidence_json="not-json"))
    with pytest.raises(StorageError, match="metadata is malformed"):
        s.detail(source_id)


def test_body_search_includes_latest_and_serving_and_escapes_wildcards(fixture):
    s = store(fixture)
    assert s.catalog(q="保留原句")["total"] == 1
    assert s.catalog(q="输入步长")["total"] == 2
    assert s.catalog(q="%")["total"] == 1
    assert s.catalog(q="_")["total"] == 0


def test_same_title_does_not_merge_sources(fixture):
    s = store(fixture)
    assert s.catalog(q="同名资料")["total"] == 4
    ids = [x["item_id"] for x in s.catalog()["items"]]
    assert len(set(ids)) == 4


def test_detail_defaults_serving_preserves_draft_and_exact_old_file(fixture):
    s = store(fixture)
    d = item(s, "knowledge_document", "a" * 32)
    result = s.detail(d["item_id"])
    assert result["latest_version"] == "2" and result["serving_version"] == "1"
    assert result["selected_version"] == "1" and result["content"] == "旧版正文 保留原句"
    assert result["source_file"]["download_available"] is False
    assert s.detail(d["item_id"], "2")["content"] == "最新草稿 100% 覆盖"
    assert result["sections"][0]["locator"] == "第1页"


def test_memory_detail_and_historical_snapshot_are_white_listed(fixture):
    s = store(fixture)
    d = item(s, "memory_case", 1)
    result = s.detail(d["item_id"])
    assert "输入步长异常" in result["content"]
    assert result["provenance"]["source_run"] == "record-a"
    payload = json.dumps(result)
    assert "NEVER-RETURN" not in payload and "999999" not in payload and '"embedding"' not in payload
    assert "旧记忆" in s.detail(d["item_id"], OLD_CASE)["content"]


def test_item_scope_access_cannot_be_forged_or_carried_between_grants(fixture):
    s = store(fixture)
    d = item(s, "memory_case", 1)
    narrow = CatalogStore([fixture["shared"]], cursor_key=KEY, database_engine=fixture["db"])
    with pytest.raises(KeyError): narrow.detail(d["item_id"])
    with pytest.raises(KeyError): s.detail(d["item_id"][:-3] + "bad")
    with pytest.raises(ValueError): s.catalog(scope_id=fixture["denied"]["id"])


def test_keyset_pages_no_duplicates_and_changed_revision_stops(fixture):
    s = store(fixture)
    first = s.catalog(limit=1)
    second = s.catalog(limit=1, cursor=first["next_cursor"])
    assert first["items"][0]["item_id"] != second["items"][0]["item_id"]
    with fixture["db"].begin() as conn:
        conn.execute(counters.update().where(counters.c.workspace_id == fixture["team"]["id"]).values(value=2))
    with pytest.raises(Conflict): s.catalog(limit=1, cursor=first["next_cursor"])


def test_cursor_bound_to_query_scope_and_page_size(fixture):
    s = store(fixture)
    first = s.catalog(limit=1)
    with pytest.raises(Conflict): s.catalog(limit=2, cursor=first["next_cursor"])
    with pytest.raises(Conflict): s.catalog(limit=1, q="changed", cursor=first["next_cursor"])
    with pytest.raises(ValueError): s.catalog(cursor="bad")


def test_summary_eligible_can_overlap_pending_and_never_counts_trash(fixture):
    summary = store(fixture).summary()
    assert summary["total"] == 4 and summary["eligible"] == 2
    assert summary["pending"] >= 2
    assert store(fixture).catalog(view="trash")["total"] == 1


def test_archived_memory_stays_readable_but_not_eligible_or_archivable(fixture):
    s = store(fixture)
    d = item(s, "memory_case", 2)
    assert d["lifecycle_state"] == "archived" and d["serving_version"] is None
    assert "archive" not in d["capabilities"]
    assert s.detail(d["item_id"])["content"]


def test_qdrant_mode_requires_matching_active_generation(fixture):
    s = store(fixture, memory_mode="qdrant")
    assert item(s, "memory_case", 1)["retrieval_state"] == "unavailable"
    with fixture["db"].begin() as conn:
        conn.execute(memory_indexes.insert().values(workspace_id=fixture["team"]["id"], generation="g1",
            status="active", config_json=json.dumps(INDEX_DEFINITION), fingerprint=digest(INDEX_DEFINITION), collection="fixture"))
        conn.execute(memory_index_entries.insert().values(workspace_id=fixture["team"]["id"], generation="g1",
            memory_id=1, revision=1, case_version=CASE, desired="present", status="ready", job_id="job", point_count=1))
    assert item(s, "memory_case", 1)["retrieval_state"] == "eligible"
    with fixture["db"].begin() as conn:
        conn.execute(memory_index_entries.update().values(case_version=OLD_CASE))
    d = item(s, "memory_case", 1)
    assert d["index_state"] == "stale" and d["serving_version"] is None


@pytest.mark.parametrize("definition,fingerprint", [
    (INDEX_DEFINITION, "wrong-hash"),
    ({"splitter": "obsolete"}, digest({"splitter": "obsolete"})),
    ([], digest([])),
])
def test_qdrant_corrupt_or_incompatible_definition_remains_readable_but_ineligible(fixture, definition, fingerprint):
    with fixture["db"].begin() as conn:
        conn.execute(memory_indexes.insert().values(workspace_id=fixture["team"]["id"], generation="active-old",
            status="active", config_json=json.dumps(definition), fingerprint=fingerprint, collection="fixture"))
        conn.execute(memory_index_entries.insert().values(workspace_id=fixture["team"]["id"], generation="active-old",
            memory_id=1, revision=1, case_version=CASE, desired="present", status="ready", job_id="job", point_count=1))
    s = store(fixture, memory_mode="qdrant")
    result = item(s, "memory_case", 1)
    assert result["retrieval_state"] == "unavailable" and result["index_state"] == "unknown"
    assert result["serving_version"] is None and s.detail(result["item_id"])["content"]


def test_qdrant_gate_keeps_valid_old_active_generation_and_reads_once_per_scope(fixture, monkeypatch):
    from codex_agent.indexing import retrieval
    real_gate = retrieval._index
    calls = []
    def gate(reader, generation, allow_staging):
        calls.append((generation, allow_staging))
        return real_gate(reader, generation, allow_staging)
    monkeypatch.setattr(retrieval, "_index", gate)
    with fixture["db"].begin() as conn:
        for scope_id, generation, state, definition in [
            (fixture["team"]["id"], "old-active", "active", INDEX_DEFINITION),
            (fixture["team"]["id"], "new-staging", "staging", {"splitter": "future"}),
            (fixture["denied"]["id"], "another-active", "active", {"splitter": "foreign"}),
        ]:
            conn.execute(memory_indexes.insert().values(workspace_id=scope_id, generation=generation,
                status=state, config_json=json.dumps(definition), fingerprint=digest(definition), collection="fixture"))
        conn.execute(memory_index_entries.insert().values(workspace_id=fixture["team"]["id"], generation="old-active",
            memory_id=1, revision=1, case_version=CASE, desired="present", status="ready", job_id="job", point_count=1))
    result = store(fixture, memory_mode="qdrant").catalog()
    assert len(calls) == 1 and calls[0] == (None, False)
    active = next(row for row in result["items"] if row["source_kind"] == "memory_case" and row["source_id"] == "1")
    assert active["retrieval_state"] == "eligible" and active["serving_version"] == CASE


def test_unknown_runtime_mode_does_not_claim_eligible(fixture):
    assert item(store(fixture, memory_mode="unknown"), "memory_case", 1)["retrieval_state"] == "unknown"


def test_version_and_chunk_pagination_bind_exact_source(fixture):
    s = store(fixture)
    d = item(s, "knowledge_document", "a" * 32)
    vs = s.versions(d["item_id"], limit=1)
    assert vs["items"][0]["version"] == "2"
    assert s.versions(d["item_id"], limit=1, cursor=vs["next_cursor"])["items"][0]["is_serving"]
    assert s.chunks(d["item_id"], "1")["items"][0]["text"] == "片段 1"
    with pytest.raises(KeyError): s.chunks(d["item_id"], "3")


def test_failed_source_is_partial_not_successful_empty_catalog(fixture):
    with fixture["db"].begin() as conn:
        conn.execute(text("DROP TABLE memory_index_entries"))
    result = store(fixture).catalog()
    assert result["complete"] is False and result["total"] == 2
    assert any(h["status"] == "unavailable" for h in result["source_health"])


def test_missing_original_text_appears_in_reconciliation(fixture):
    with fixture["db"].begin() as conn:
        conn.execute(versions.update().where(versions.c.document_id == "b" * 32).values(content="", state="draft"))
    result = store(fixture).reconciliation()
    assert result["vector_scan_performed"] is False
    assert any(d["source_id"] == "b" * 32 and d["content_state"] == "missing" for d in result["items"])


def test_item_revision_can_be_rechecked_inside_locked_source_transaction(fixture):
    s = store(fixture)
    d = item(s, "memory_case", 1)
    with fixture["db"].begin() as conn:
        assert s.revision_for(conn, d["item_id"]) == d["revision"]
        conn.execute(memories.update().where(memories.c.workspace_id == fixture["team"]["id"], memories.c.id == 1).values(active=0))
        assert s.revision_for(conn, d["item_id"]) != d["revision"]


def test_missing_published_index_fingerprint_never_counts_as_eligible(fixture):
    with fixture["db"].begin() as conn:
        conn.execute(versions.update().where(versions.c.document_id == "a" * 32,
            versions.c.version == 1).values(fingerprint=None))
    d = item(store(fixture), "knowledge_document", "a" * 32)
    assert d["retrieval_state"] == "unavailable" and d["serving_version"] is None


def test_legacy_memory_without_snapshot_still_has_readable_version(fixture):
    s = store(fixture)
    with fixture["db"].begin() as conn:
        conn.execute(memory_versions.delete().where(memory_versions.c.workspace_id == fixture["team"]["id"]))
    d = item(s, "memory_case", 1)
    result = s.versions(d["item_id"])
    assert result["items"] == [{"version": CASE, "state": "recorded", "chunk_count": 0,
        "created_at": "2026-11-11T12:00:00Z", "is_latest": True, "is_serving": True}]
    assert "输入步长异常" in s.detail(d["item_id"], CASE)["content"]


def test_source_grant_without_write_has_no_mutation_capabilities(fixture):
    scopes = [{**fixture["shared"], "writable": False}, {**fixture["team"], "writable": False}]
    s = CatalogStore(scopes, cursor_key=KEY, database_engine=fixture["db"])
    assert all(set(row["capabilities"]) == {"view", "versions", "chunks"} for row in s.catalog()["items"])


def test_historical_memory_chunks_use_versioned_index_evidence(fixture):
    s = store(fixture)
    d = item(s, "memory_case", 1)
    with fixture["db"].begin() as conn:
        conn.execute(memory_index_chunks.insert().values(workspace_id=fixture["team"]["id"],
            generation="historical", memory_id=1, revision=2, chunk_id="old-text", point_id="point",
            case_version=OLD_CASE, text="保留的历史索引文字", provenance_json='{"position":3,"source_field":"summary"}'))
    result = s.chunks(d["item_id"], OLD_CASE)
    assert result["items"] == [dict(id="old-text", text="保留的历史索引文字", generation="historical", revision=2,
                                    position=3, locator="summary")]


def test_reconciliation_groups_missing_source_mappings_and_paginates_with_scope(fixture):
    with fixture["db"].begin() as conn:
        for sid, identifier in [(fixture["team"]["id"], 9), (fixture["team"]["id"], 10), (fixture["denied"]["id"], 777)]:
            for generation in ["g1", "g2"]:
                conn.execute(memory_index_entries.insert().values(workspace_id=sid, generation=generation,
                    memory_id=identifier, revision=1, case_version=CASE, desired="present", status="ready", job_id="job", point_count=2))
                for n in [1, 2]:
                    conn.execute(memory_index_chunks.insert().values(workspace_id=sid, generation=generation,
                        memory_id=identifier, revision=1, chunk_id=f"chunk-{n}", point_id=f"point-{n}",
                        case_version=CASE, text="INDEX-TEXT-NOT-ORIGINAL", provenance_json="{}"))
    s = store(fixture)
    first = s.reconciliation(limit=1)
    assert first["index_anomalies_total"] == 2
    assert first["index_scan_complete"] and first["complete"]
    a = first["index_anomalies"][0]
    assert a["index_entries"] == 2 and a["index_chunks"] == 4 and a["reason"] == "source_missing"
    assert "INDEX-TEXT-NOT-ORIGINAL" not in json.dumps(first) and '"777"' not in json.dumps(first)
    second = s.reconciliation(limit=1, index_cursor=first["index_anomalies_next_cursor"])
    assert second["index_anomalies"][0]["source_id"] != a["source_id"]
    assert second["index_anomalies_next_cursor"] is None
    assert s.reconciliation(scope_id=fixture["shared"]["id"])["index_anomalies_total"] == 0
    with pytest.raises(Conflict):
        s.reconciliation(limit=2, index_cursor=first["index_anomalies_next_cursor"])
    with fixture["db"].begin() as conn:
        conn.execute(counters.update().where(counters.c.workspace_id == fixture["team"]["id"]).values(value=3))
    with pytest.raises(Conflict):
        s.reconciliation(limit=1, index_cursor=first["index_anomalies_next_cursor"])


def test_missing_index_table_returns_partial_reconciliation(fixture):
    with fixture["db"].begin() as conn:
        conn.execute(text("DROP TABLE memory_index_chunks"))
    result = store(fixture).reconciliation()
    assert not result["index_scan_complete"] and not result["complete"]
    assert any(h["status"] == "unavailable" for h in result["index_source_health"])
