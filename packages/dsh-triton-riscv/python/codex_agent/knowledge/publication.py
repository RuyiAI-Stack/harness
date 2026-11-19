"""Confirmed save -> durable outbox -> indexed and published, with readback gates.

Only an explicit confirm creates authority to publish. Draft saves and uploads
never create intents. An old published version remains visible until replacement
is ready; cancellation, newer drafts and lifecycle changes revoke pending intents.
"""
import json

from sqlalchemy import case

from codex_agent.platform.store import timestamp
from codex_agent.tasks.schema import jobs
from codex_agent.tasks.store import Conflict, digest, now_ms
from .schema import documents, versions
from .publication_schema import intents
from .lifecycle import control, require_active
from .files import source_hash, source_row
from .publication_policy import require_publisher, set_policy


def request_key(actor, request_id):
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100 or '\0' in request_id:
        raise ValueError('Invalid publication request ID')
    return digest([actor, 'confirm', request_id])


def invalidate(store, db, conn, identifier, state):
    pending = store.rows(conn, intents, intents.c.document_id == identifier, intents.c.state == 'pending')
    if not pending:
        return
    db.update(conn, intents, dict(state=state, error='Publication confirmation was withdrawn or superseded.',
        updated_at=timestamp()), intents.c.document_id == identifier, intents.c.state == 'pending')
    db.update(conn, jobs, dict(cancel_requested=1, updated_ms=now_ms(conn),
        status=case((jobs.c.status == 'queued', 'cancelled'), else_=jobs.c.status)),
        jobs.c.id.in_([row['job_id'] for row in pending]), jobs.c.status.in_(['queued', 'running']))


def _sync(store, conn, doc, intent):
    state, error = intent['state'], intent['error']
    if state == 'pending':
        try:
            require_publisher(store, conn, intent['actor_id'])
        except Conflict:
            return dict(state='withdrawn', request_id=intent['request_id'], job_id=intent['job_id'],
                version=intent['version'], error='Publisher permission is missing or revoked.')
        lifecycle = control(store, conn, doc['id'])
        v = store.row(conn, versions, doc['id'], intent['version'])
        job = next(iter(store.rows(conn, jobs, jobs.c.id == intent['job_id'])), None)
        if lifecycle['deleted_at'] or lifecycle['revision'] != intent['catalog_revision']:
            state, error = 'withdrawn', 'Document lifecycle changed; confirmation is no longer active.'
        elif doc['latest_version'] != intent['version'] or v['job_id'] != intent['job_id']:
            state, error = 'superseded', 'A newer draft or index task replaced this confirmation.'
        elif not job or job['cancel_requested'] or job['status'] in {'failed', 'dead', 'cancelled', 'needs-reconciliation'}:
            state, error = 'failed', 'Indexing failed or stopped. The previously published version was retained.'
        else:
            from .vectors import index_definition
            try:
                valid_configuration = digest(index_definition()) == intent['fingerprint']
            except (ValueError, OSError):
                valid_configuration = False
            if not valid_configuration:
                state, error = 'failed', 'Index configuration changed; confirm a new version.'
            elif job['status'] == 'succeeded':
                state, error = 'failed', 'Index completion lacks a matching publication receipt.'
            else:
                state, error = ('indexing' if job['status'] == 'running' else 'pending'), None
    elif state == 'published' and doc['published_version'] != intent['version']:
        state, error = ('withdrawn' if doc['published_version'] is None else 'superseded'), 'This version is no longer published.'
    return dict(state=state, request_id=intent['request_id'], job_id=intent['job_id'],
                version=intent['version'], error=error)


def sync_for_document(store, conn, doc, actor):
    rows = store.rows(conn, intents, intents.c.actor_id == actor,
        intents.c.document_id == doc['id'], intents.c.version == doc['latest_version'])
    return _sync(store, conn, doc, rows[0]) if rows else None


def _receipt(store, conn, intent):
    doc = store.row(conn, documents, intent['document_id'], allow_deleted=True)
    return {**store.summary(conn, doc), 'sync': _sync(store, conn, doc, intent)}


def operation(store, request_id, actor):
    with store.engine.connect() as conn:
        found = store.rows(conn, intents, intents.c.id == request_key(actor, request_id), intents.c.actor_id == actor)
        if not found:
            raise KeyError('Unknown publication request')
        return _receipt(store, conn, found[0])


def confirm(store, value, actor, *, publisher_ids=None):
    key, payload_hash = request_key(actor, value.request_id), digest(value.model_dump(exclude={'request_id'}))
    db = store.writer()
    with db.transaction() as conn:
        db.lock(conn, 'knowledge-write')
        if publisher_ids is not None:
            set_policy(store, db, conn, publisher_ids)
        require_publisher(store, conn, actor)
        prior = store.rows(conn, intents, intents.c.id == key, intents.c.actor_id == actor)
        if prior:
            if prior[0]['request_fingerprint'] != payload_hash:
                raise Conflict('Confirmation request ID reused with different content')
            return _receipt(store, conn, prior[0])
        from .vectors import index_definition
        definition = index_definition()  # Validate before saving; no network/model call here.
        saved = store.save_transaction(value, actor, db, conn, operation='confirm')
        identifier, version = saved['id'], saved['latest_version']
        store.enqueue_transaction(identifier, version, 'confirm:' + key, actor,
            saved['catalog_revision'], definition, db, conn, publication_id=key)
        v = store.row(conn, versions, identifier, version)
        row = dict(id=key, actor_id=actor, request_id=value.request_id, request_fingerprint=payload_hash,
            document_id=identifier, version=version, catalog_revision=saved['catalog_revision'],
            source_hash=v['source_hash'], fingerprint=v['fingerprint'], job_id=v['job_id'], state='pending',
            error=None, created_at=timestamp(), updated_at=timestamp())
        db.insert(conn, intents, row)
        store.log(db, conn, actor, 'publication-confirmed', identifier, version)
        return _receipt(store, conn, row)


def require_confirmation(store, conn, job, payload):
    """Readback before every effect and final publish, never trust queue payload alone."""
    key = payload.get('publication_id')
    if key is None:
        return None  # Existing explicit index operation never auto-publishes.
    found = store.rows(conn, intents, intents.c.id == key)
    if not found:
        raise Conflict('Publication authority is missing')
    intent = found[0]
    if intent['state'] != 'pending' or job.get('cancel_requested'):
        raise Conflict('Publication authority was withdrawn or superseded')
    if (intent['job_id'] != job['id'] or any(intent[k] != payload.get(k) for k in
            ('document_id', 'version', 'catalog_revision', 'source_hash', 'fingerprint'))):
        raise Conflict('Publication authority does not match this index job')
    doc = store.row(conn, documents, intent['document_id'])
    require_active(store, conn, doc['id'], intent['catalog_revision'])
    if doc['latest_version'] != intent['version']:
        raise Conflict('A newer draft superseded this publication')
    require_publisher(store, conn, intent['actor_id'])
    v = store.row(conn, versions, doc['id'], intent['version'])
    origin = source_row(store, conn, doc['id'], intent['version'])
    sections = json.loads(origin['sections_json']) if origin else None
    if (v['job_id'] != job['id'] or v['source_hash'] != intent['source_hash']
            or v['fingerprint'] != intent['fingerprint']
            or source_hash(v['title'], v['source'], v['content'], sections) != intent['source_hash']
            or (origin and not origin['reviewed'])):
        raise Conflict('Confirmed source or index identity changed')
    from .vectors import index_definition
    if digest(index_definition()) != intent['fingerprint']:
        raise Conflict('Index configuration changed after confirmation')
    return intent


def finish_publication(store, db, conn, job, payload):
    intent = require_confirmation(store, conn, job, payload)
    if intent is None:
        return False
    version = store.row(conn, versions, intent['document_id'], intent['version'])
    if version['state'] != 'ready' or version['chunk_count'] < 1:
        raise Conflict('Confirmed index is not complete')
    db.update(conn, documents, dict(published_version=intent['version'], updated_at=timestamp()),
        documents.c.id == intent['document_id'])
    db.update(conn, intents, dict(state='published', error=None, updated_at=timestamp()), intents.c.id == intent['id'])
    store.log(db, conn, intent['actor_id'], 'confirmed-published', intent['document_id'], intent['version'])
    return True
