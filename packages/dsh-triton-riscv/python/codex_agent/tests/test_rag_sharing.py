"""Shared copies are editable drafts; original history remains untouched."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib

import pytest
from sqlalchemy import select

from codex_agent.accounts.config import AccountConfig, RagHistorySource
from codex_agent.knowledge.source_links import source_links
from codex_agent.knowledge.schema import documents, versions
from codex_agent.rag_catalog import sharing
from codex_agent.tasks.schema import jobs, outbox
from codex_agent.tasks.store import Conflict
from codex_agent.tests.test_knowledge_publication import env, value
from codex_agent.tests.test_rag_catalog_api import ADMIN, OTHER, ORIGIN
from codex_agent.tests.test_rag_history_sources import history, fixture


@pytest.fixture
def shared(env, monkeypatch, tmp_path):
    source_links.create(env.db.engine)
    root = tmp_path / 'historical'
    scope = {'id': hashlib.sha256(str(root).encode()).hexdigest(), 'root': root,
             'kind': 'memory_case', 'writable': False}
    source = RagHistorySource(id='history', label='明确授权的历史来源',
        url_env='TRITON_RAG_HISTORY_SHARE_URL', workspace_ids=(scope['id'],), allow_share=True)
    config = AccountConfig(origin=ORIGIN, data_dir=env.db.root.parent, template=tmp_path/'template',
        allow_local_http=True, knowledge_enabled=True, admin_user_ids=(ADMIN, OTHER),
        rag_history_sources=(source,))
    monkeypatch.setenv(source.url_env, 'mysql+pymysql://reader:PRIVATE@localhost/history')
    monkeypatch.setattr(sharing, 'historical_scopes', lambda *args: [scope])
    original = dict(title='历史经验', selected_title='历史经验', content='真实原始记录，不是已重新验证的结论',
        content_state='available', lifecycle_state='active', latest_version='v1',
        selected_version='v1', revision='revision1', capabilities=['view','versions','chunks'])
    class Catalog:
        engine = object()
        def resolve(self, _item): return scope, 'memory_case', '1'
        def detail(self, _item, version=None): return deepcopy(original)
    return SimpleNamespace(env=env, config=config, source=source, catalog=Catalog(), original=original)


def request(rid='share-request'):
    return sharing.ShareInput(request_id=rid, expected_revision='revision1')


def import_one(s, rid='share-request', actor=ADMIN):
    return sharing.share(s.config, 'history', s.catalog, 'signed-item', request(rid), actor)


def test_import_is_one_draft_with_provenance_and_no_index_jobs(shared):
    s=shared;before=deepcopy(s.original)
    result=import_one(s)
    assert result['published_version'] is None and result['state']=='draft'
    assert '待审核' in result['source']
    with s.env.db.engine.connect() as conn:
        assert len(s.env.store.rows(conn,documents))==1
        assert len(s.env.store.rows(conn,versions))==1
        assert len(s.env.store.rows(conn,source_links))==1
        assert not s.env.store.rows(conn,jobs) and not s.env.store.rows(conn,outbox)
    assert s.original==before
    detail=sharing.decorate(s.config,'history',s.catalog,'signed-item',deepcopy(s.original))
    assert detail['shared_document_id']==result['id']
    assert 'share_to_shared' in detail['capabilities']


def test_duplicate_requests_and_actors_do_not_duplicate_or_reset_edits(shared):
    s=shared;first=import_one(s)
    assert import_one(s)['id']==first['id']
    assert import_one(s,'another',OTHER)['id']==first['id']
    edit=value('edit',document_id=first['id'],expected_version=1).model_copy(update={'content':'管理员修正后的版本'})
    changed=s.env.store.save(edit,ADMIN)
    assert import_one(s,'after-edit')['latest_version']==changed['latest_version']==2
    assert s.env.store.detail(first['id'])['content']=='管理员修正后的版本'
    s.env.store.set_deleted(first['id'],2,0,ADMIN,deleted=True)
    assert import_one(s,'after-delete')['state']=='deleted'


def test_link_failure_rolls_back_original_copy_and_receipt(shared, monkeypatch):
    s=shared;insert=s.env.db.insert
    def fail(conn, table, values):
        if table is source_links: raise RuntimeError('Interrupted mapping')
        return insert(conn,table,values)
    monkeypatch.setattr(s.env.db,'insert',fail)
    with pytest.raises(RuntimeError): import_one(s)
    with s.env.db.engine.connect() as conn:
        assert not s.env.store.rows(conn,documents)
        assert not s.env.store.rows(conn,versions)
        assert not s.env.store.rows(conn,source_links)


def test_grant_revocation_and_revision_change_block_import(shared, monkeypatch):
    s=shared
    denied=s.config.model_copy(update={'rag_history_sources':(s.source.model_copy(update={'allow_share':False}),)})
    with pytest.raises(PermissionError): sharing.share(denied,'history',s.catalog,'item',request(),ADMIN)
    s.original['revision']='changed'
    with pytest.raises(Conflict):import_one(s)
    s.original['revision']='revision1'
    monkeypatch.setattr(sharing,'historical_scopes',lambda *_:(_ for _ in ()).throw(PermissionError('owner revoked')))
    with pytest.raises(PermissionError):import_one(s)
    with s.env.db.engine.connect() as conn:assert not s.env.store.rows(conn,documents)


def test_read_grant_does_not_grant_sharing_and_current_source_cannot_share(history):
    c=history.client
    from codex_agent.tests.test_rag_catalog_api import headers
    for source in ['history','current']:
        response=c.post('/api/admin/rag/catalog/anything/share',params={'source_id':source},
            json={'request_id':'request','expected_revision':'r'},headers=headers())
        assert response.status_code==403
    response=c.post('/api/admin/rag/catalog/anything/share?source_id=history',
        json={'request_id':'request','expected_revision':'r','root':'/private/other'},headers=headers())
    assert response.status_code==422


def test_unknown_body_and_old_version_do_not_get_share_capability(shared):
    s=shared
    old=deepcopy(s.original);old['selected_version']='old'
    assert 'share_to_shared' not in sharing.decorate(s.config,'history',s.catalog,'item',old)['capabilities']
    s.original['content']=''
    with pytest.raises(Conflict):import_one(s)


def test_same_request_cannot_be_reused_for_different_source_item(shared):
    s=shared;import_one(s)
    scope,kind,_=s.catalog.resolve('item')
    s.catalog.resolve=lambda _: (scope,kind,'2')
    with pytest.raises(Conflict):import_one(s)


def test_connection_credential_rotation_preserves_link_identity(shared, monkeypatch):
    s=shared;first=import_one(s)
    monkeypatch.setenv(s.source.url_env,'mysql+pymysql://different-reader:ROTATED@localhost/history')
    assert import_one(s,'rotated')['id']==first['id']


def test_current_knowledge_detail_retains_source_alias_after_cleanup_read(shared, monkeypatch):
    from fastapi.testclient import TestClient
    from codex_agent.accounts.api import create_gateway
    from codex_agent.rag_catalog import api
    from codex_agent.storage.schema import counters
    from codex_agent.tests.test_rag_catalog_api import Accounts
    s=shared
    counters.create(s.env.db.engine)
    saved=s.env.store.save(value(),ADMIN)
    accounts=Accounts();accounts.engine=s.env.db.engine
    scope={'id':s.env.store.scope,'root':s.env.db.root,'kind':'knowledge_document',
           'label':'共享资料库','writable':True}
    monkeypatch.setattr(api,'authorized_scopes',lambda *_:[scope])
    monkeypatch.setattr(api,'runtime_config',lambda:SimpleNamespace(
        indexing=SimpleNamespace(enabled=False),memory=SimpleNamespace(retrievalMode='legacy')))
    client=TestClient(create_gateway(s.config,accounts=accounts,workbenches=object()),base_url=ORIGIN)
    client.cookies.set(s.config.cookie_name,'admin-session')
    listing=client.get('/api/admin/rag/catalog?source_id=current')
    assert listing.status_code==200
    item=listing.json()['items'][0]
    response=client.get('/api/admin/rag/catalog/'+item['item_id'],params={'source_id':'current'})
    assert response.status_code==200,response.text
    assert response.json()['source_id']==saved['id']
    assert response.json()['content']=='Old original fact'
    assert 'share_to_shared' not in response.json()['capabilities']
    client.close()
