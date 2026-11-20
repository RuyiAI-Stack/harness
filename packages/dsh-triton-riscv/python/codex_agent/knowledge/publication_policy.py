"""Server-owned publisher policy shared by the gateway and durable index workers."""
import json
import re

from sqlalchemy import case, inspect, select

from codex_agent.accounts.schema import users
from codex_agent.platform.store import timestamp
from codex_agent.storage.schema import meta
from codex_agent.tasks.schema import jobs
from codex_agent.tasks.store import Conflict, encode, now_ms
from .publication_schema import intents

POLICY_KEY = 'knowledge-publisher-policy:v1'


def set_policy(store, db, conn, admin_user_ids, *, enabled=True):
    """Caller holds knowledge-write. Inputs come from gateway configuration, never HTTP JSON."""
    if (type(enabled) is not bool or not isinstance(admin_user_ids, (tuple, list))
            or len(admin_user_ids) > 100 or any(not isinstance(uid, str)
                or not re.fullmatch(r'[a-f0-9]{32}', uid) for uid in admin_user_ids)):
        raise ValueError('Invalid publisher configuration')
    permitted = sorted(set(admin_user_ids)) if enabled else []
    value = encode(dict(schema_version=1, enabled=enabled, admin_user_ids=permitted))
    found = store.rows(conn, meta, meta.c.key == POLICY_KEY)
    if not found:
        db.insert(conn, meta, dict(key=POLICY_KEY, value=value))
    elif found[0]['value'] != value:
        db.update(conn, meta, dict(value=value), meta.c.key == POLICY_KEY)
    # Revocation is irreversible for these confirmations. Re-adding a publisher
    # later does not resurrect work that was withdrawn during a gateway restart.
    active = set(conn.execute(select(users.c.id).where(users.c.id.in_(permitted),
        users.c.status == 'active')).scalars()) if permitted else set()
    revoked = [row for row in store.rows(conn, intents, intents.c.state == 'pending')
        if row['actor_id'] not in active]
    if revoked:
        db.update(conn, intents, dict(state='withdrawn', error='Publisher permission was revoked.',
            updated_at=timestamp()), intents.c.id.in_([row['id'] for row in revoked]))
        db.update(conn, jobs, dict(cancel_requested=1, updated_ms=now_ms(conn),
            status=case((jobs.c.status == 'queued', 'cancelled'), else_=jobs.c.status)),
            jobs.c.id.in_([row['job_id'] for row in revoked]), jobs.c.status.in_(['queued', 'running']))


def configure_publishers(store, admin_user_ids, *, enabled=True):
    # A pre-feature accounts-only deployment has no automatic publication
    # authority to revoke. Do not force creation of a knowledge workspace when
    # disabled; connection errors still propagate, never masquerade as absence.
    if enabled is False and not inspect(store.engine).has_table(intents.name):
        return
    db = store.writer()
    with db.transaction() as conn:
        db.lock(conn, 'knowledge-write')
        set_policy(store, db, conn, admin_user_ids, enabled=enabled)


def require_publisher(store, conn, actor):
    found = store.rows(conn, meta, meta.c.key == POLICY_KEY)
    try:
        policy = json.loads(found[0]['value']) if found else None
        if (not isinstance(policy, dict) or policy.get('schema_version') != 1
                or policy.get('enabled') is not True or not isinstance(policy.get('admin_user_ids'), list)
                or actor not in policy['admin_user_ids']):
            raise ValueError()
    except (ValueError, TypeError):
        raise Conflict('Publisher permission is missing or revoked') from None
    if not conn.execute(select(users.c.id).where(users.c.id == actor,
            users.c.status == 'active').with_for_update()).first():
        raise Conflict('The confirming administrator is no longer active')
