"""Catalog cursors survive worker heartbeats but observe index lifecycle changes."""
import pytest
from sqlalchemy import create_engine
from codex_agent.tests.test_rag_catalog_readonly import sqlite_text  # SQLite LONGTEXT compiler
from codex_agent.storage.database import WorkspaceDatabase
from codex_agent.tasks.schema import jobs


@pytest.mark.parametrize('kind,changes,invalidates', [
    ('index', {'lease_until': 9000, 'updated_ms': 5000}, False),
    ('index', {'effects_started': 1}, False),
    ('execute', {'status': 'succeeded', 'updated_ms': 5000}, False),
    ('index', {'status': 'succeeded', 'updated_ms': 5000}, True),
    ('index', {'cancel_requested': 1}, True),
])
def test_only_relevant_job_changes_invalidate_catalog(kind, changes, invalidates):
    engine = create_engine('sqlite://')
    jobs.create(engine)
    db = object.__new__(WorkspaceDatabase)
    db.workspace_id = 'a' * 64
    with engine.begin() as conn:
        conn.execute(jobs.insert().values(workspace_id=db.workspace_id, id='job', kind=kind,
            status='running', idempotency_key='key', fingerprint='f', payload_json='{}', result_json='{}',
            execution_host='local', attempt=1, generation=1, owner='worker', lease_until=1000,
            effects_started=0, cancel_requested=0, created_ms=1, updated_ms=1))
        conn.info['cache_changes'] = set()
        assert db.update(conn, jobs, changes, jobs.c.id == 'job').rowcount == 1
        assert bool(conn.info['cache_changes']) is invalidates
    engine.dispose()
