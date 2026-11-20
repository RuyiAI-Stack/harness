"""Historical source isolation/read-only admission; no private database or model."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig, RagHistorySource
from codex_agent.accounts.guard import MARKER
from codex_agent.accounts.schema import users, workspaces as account_workspaces
from codex_agent.rag_catalog import api, sources
from codex_agent.rag_catalog.store import CatalogStore
from codex_agent.storage.schema import workspaces, meta, counters
from codex_agent.tests.test_rag_catalog_api import Accounts, ADMIN, OTHER, ORIGIN, headers, action
from codex_agent.tests import test_rag_catalog_readonly as readonly
from codex_agent.tests.test_rag_catalog_readonly import KEY
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # API handlers execute reads in worker threads; share only this ephemeral DB.
    monkeypatch.setattr(readonly, "create_engine", lambda url: create_engine(
        url, poolclass=StaticPool, connect_args={"check_same_thread": False}))
    yield from readonly.fixture.__wrapped__(tmp_path)


def definition(f, **changes):
    return RagHistorySource(id='history', label='历史资料', url_env='TRITON_RAG_HISTORY_TEST_URL',
        workspace_ids=(f['team']['id'],), **changes)


@pytest.fixture
def history(fixture, monkeypatch, tmp_path):
    f = fixture
    workspaces.create(f['db'])
    meta.create(f['db'])
    with f['db'].begin() as conn:
        conn.execute(workspaces.insert().values(workspace_id=f['team']['id'], root=str(f['team']['root'])))
    monkeypatch.setenv('TRITON_RAG_HISTORY_TEST_URL', 'mysql+pymysql://unit:DO-NOT-LEAK@localhost/history')
    monkeypatch.setattr(sources, 'history_engine', lambda *_: f['db'])
    config = AccountConfig(origin=ORIGIN, data_dir=tmp_path/'accounts', template=tmp_path/'template',
        allow_local_http=True, admin_user_ids=(ADMIN,), rag_history_sources=(definition(f),))
    accounts = Accounts()
    # Current engine has no source tables: history must never fall back to it.
    accounts.engine = create_engine("sqlite+pysqlite:///:memory:",
        poolclass=StaticPool, connect_args={"check_same_thread": False})
    counters.create(accounts.engine)
    monkeypatch.setattr(api, 'authorized_scopes', lambda *_: [])
    monkeypatch.setattr(api, 'runtime_config', lambda: SimpleNamespace(indexing=SimpleNamespace(enabled=False),
        memory=SimpleNamespace(retrievalMode='legacy')))
    client = TestClient(create_gateway(config, accounts=accounts, workbenches=object()), base_url=ORIGIN)
    client.cookies.set(config.cookie_name, 'admin-session')
    yield SimpleNamespace(f=f, config=config, client=client)
    client.close()
    accounts.engine.dispose()


def test_sources_are_configuration_only_and_never_expose_credentials_or_roots(history, monkeypatch):
    monkeypatch.setattr(sources, 'history_engine', Mock(side_effect=AssertionError('must not connect')))
    result = history.client.get('/api/admin/rag/sources')
    assert result.status_code == 200
    assert result.json() == {'default_source_id':'current', 'origin_filters':[
        {'id':'uploaded','label':'新增 / 上传'}, {'id':'history:history','label':'历史资料'}], 'sources':[
        {'id':'current','label':'共享知识库','read_only':False,'status':'configured','kind':'current'},
        {'id':'history','label':'历史资料','read_only':True,'status':'configured','kind':'memory_case'}]}
    assert 'TRITON_RAG_HISTORY_TEST_URL' not in result.text and 'DO-NOT-LEAK' not in result.text
    history.client.cookies.set(history.config.cookie_name, 'other-session')
    assert history.client.get('/api/admin/rag/sources').status_code == 403


def test_historical_catalog_and_version_reads_use_selected_engine_and_are_read_only(history):
    c = history.client
    statements = []
    event.listen(history.f['db'], 'before_cursor_execute',
        lambda _c, _cu, statement, _p, _ctx, _many: statements.append(statement))
    listing = c.get('/api/admin/rag/catalog?source_id=history')
    assert listing.status_code == 200, listing.text
    items = listing.json()['items']
    assert len(items) == 2
    item = next(x for x in items if x['source_id'] == '1')
    assert item['capabilities'] == ['view', 'versions', 'chunks']
    base = '/api/admin/rag/catalog/' + item['item_id']
    detail = c.get(base, params={'source_id':'history'})
    assert detail.status_code == 200 and '输入步长异常' in detail.json()['content']
    assert detail.json()['corrections'] == []
    assert c.get(base, params={'source_id':'current'}).status_code == 404
    for path,params in [
        (base+'/versions', {}),
        (base+'/versions/'+item['latest_version'], {}),
        (base+'/chunks', {'version':item['latest_version']}),
        ('/api/admin/rag/catalog/summary', {}),
        ('/api/admin/rag/catalog/reconciliation', {}),
        ('/api/admin/rag/operations/unknown', {'item_id':item['item_id']}),
    ]:
        r = c.get(path, params={'source_id':'history', **params})
        assert r.status_code == 200, r.text
    assert c.get(base+'/original', params={'source_id':'history','version':item['latest_version']}).status_code == 404
    assert statements and all(s.lstrip().split()[0].upper() in {'SELECT', 'PRAGMA'} for s in statements)


def test_historical_writes_fail_before_opening_any_connection(history, monkeypatch):
    monkeypatch.setattr(api, 'open_history', Mock(side_effect=AssertionError('must not connect')))
    result = history.client.post('/api/admin/rag/catalog/item/actions?source_id=history',json=action(),headers=headers())
    assert result.status_code == 403 and '只读' in result.text


def test_unregistered_source_or_missing_scope_does_not_return_empty_success(history):
    assert history.client.get('/api/admin/rag/catalog?source_id=unregistered').status_code == 404
    with history.f['db'].begin() as conn:
        conn.execute(workspaces.delete())
    response = history.client.get('/api/admin/rag/catalog?source_id=history')
    assert response.status_code == 503
    assert 'DO-NOT-LEAK' not in response.text


def test_sources_tokens_cannot_cross_aliases_or_database_reconfiguration(fixture):
    f=fixture
    first = CatalogStore([f['team']], cursor_key=sources.source_signing_key(KEY,'old','db-a'),database_engine=f['db'])
    token=first.item_id(f['team']['id'],'memory_case','1')
    for alias,db in [('other','db-a'),('old','db-b'),('current','db-a')]:
        second=CatalogStore([f['team']],cursor_key=sources.source_signing_key(KEY,alias,db),database_engine=f['db'])
        with pytest.raises(KeyError):second.resolve(token)


def add_owner(f, *, marker=True, status='active'):
    users.create(f['db'])
    account_workspaces.create(f['db'])
    root=f['team']['root'];root.mkdir(parents=True)
    if marker:(root/MARKER).write_text(ADMIN)
    with f['db'].begin() as conn:
        conn.execute(users.insert().values(id=ADMIN,username='owner',password_hash='unused',status=status,auth_version=1,created_ms=1))
        conn.execute(account_workspaces.insert().values(workspace_id=f['team']['id'],user_id=ADMIN,root=str(root),status='ready'))


def test_private_history_requires_explicit_owner_even_for_current_admin(history):
    f=history.f;add_owner(f)
    response = history.client.get('/api/admin/rag/catalog?source_id=history')
    assert response.status_code == 403
    assert response.json()['error_code'] == 'rag_source_forbidden'
    original = history.client.get('/api/admin/rag/catalog/unknown/original',
        params={'source_id':'history', 'version':'1'})
    assert original.status_code == 403 and original.json()['error_code'] == 'rag_source_forbidden'
    allowed=sources.historical_scopes(definition(f,owner_user_id=ADMIN),f['db'])
    assert len(allowed)==1 and allowed[0]['writable'] is False
    with pytest.raises(PermissionError):sources.historical_scopes(definition(f,owner_user_id=OTHER),f['db'])


def test_revoked_or_missing_owner_marker_blocks_history(history):
    f=history.f;add_owner(f)
    source=definition(f,owner_user_id=ADMIN)
    with f['db'].begin() as conn:conn.execute(users.update().values(status='disabled'))
    with pytest.raises(PermissionError):sources.historical_scopes(source,f['db'])
    with f['db'].begin() as conn:conn.execute(users.update().values(status='active'))
    (f['team']['root']/MARKER).unlink()
    with pytest.raises(PermissionError):sources.historical_scopes(source,f['db'])


def test_marker_without_account_registry_is_not_legacy(history):
    f=history.f;root=f['team']['root'];root.mkdir(parents=True);(root/MARKER).write_text(ADMIN)
    with pytest.raises(PermissionError):sources.historical_scopes(definition(f),f['db'])


def test_unknown_owner_claim_missing_env_and_bad_root_fail_closed(history, monkeypatch):
    f=history.f
    with pytest.raises(PermissionError):sources.historical_scopes(definition(f,owner_user_id=ADMIN),f['db'])
    with f['db'].begin() as conn:conn.execute(workspaces.update().values(root='/different/root'))
    with pytest.raises(sources.HistoryUnavailable):sources.historical_scopes(definition(f),f['db'])
    monkeypatch.delenv('TRITON_RAG_HISTORY_TEST_URL')
    assert history.client.get('/api/admin/rag/catalog?source_id=history').status_code == 503


@pytest.mark.parametrize('changes', [{'id':'current'}, {'id':'../escape'}, {'label':' '},
    {'url_env':'mysql://secret'}, {'url_env':'TRITON_MYSQL_URL'}, {'url_env':'OTHER_DB_URL'}, {'workspace_ids':()}, {'workspace_ids':('not-a-hash',)},
    {'workspace_ids':('a'*64,)*2}, {'owner_user_id':'unknown'}, {'memory_mode':'pretend-ready'}])
def test_history_config_rejects_ambiguous_or_unbounded_sources(changes):
    values=dict(id='history',label='History',url_env='TRITON_RAG_HISTORY_TEST_URL',workspace_ids=('a'*64,))
    values.update(changes)
    with pytest.raises(ValueError):RagHistorySource(**values)


def test_duplicate_source_aliases_rejected(tmp_path):
    value=RagHistorySource(id='history',label='History',url_env='TRITON_RAG_HISTORY_TEST_URL',workspace_ids=('a'*64,))
    with pytest.raises(ValueError):AccountConfig(origin=ORIGIN,data_dir=tmp_path/'data',template=tmp_path/'template',
        allow_local_http=True,rag_history_sources=(value,value))


def test_engine_is_bounded_and_enforces_readonly_on_initial_and_reused_connections(monkeypatch):
    # Verify actual construction options and the executable checkout hook separately.
    sources.history_engine.cache_clear()
    captured={};engine=object()
    def factory(url,**kwargs):captured.update(kwargs);return engine
    monkeypatch.setattr(sources,'create_engine',factory)
    listener=Mock();monkeypatch.setattr(sources.event,'listen',listener)
    assert sources.history_engine('mysql+pymysql://readonly@localhost/history',1234) is engine
    assert captured['max_overflow']==0 and captured['pool_timeout']==5 and captured['pool_size']==2
    assert captured['connect_args']['init_command']=='SET SESSION TRANSACTION READ ONLY'
    listener.assert_called_once_with(engine,'checkout',sources._readonly_session)
    cursor=Mock();cursor.__enter__=Mock(return_value=cursor);cursor.__exit__=Mock(return_value=False)
    conn=Mock();conn.cursor.return_value=cursor
    sources._readonly_session(conn)
    conn.rollback.assert_called_once();cursor.execute.assert_called_once_with('SET SESSION TRANSACTION READ ONLY')
    sources.history_engine.cache_clear()


@pytest.mark.parametrize('raw',['sqlite:///private.db','not-a-url','mysql+pymysql://u:DO-NOT-LEAK@localhost/db?local_infile=true'])
def test_invalid_connection_settings_are_sanitized(raw):
    with pytest.raises(sources.HistoryUnavailable) as error:sources.history_engine(raw,4321)
    assert 'DO-NOT-LEAK' not in str(error.value)
