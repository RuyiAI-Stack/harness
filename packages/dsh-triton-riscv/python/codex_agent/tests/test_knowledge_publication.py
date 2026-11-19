"""Atomic confirm/index/publish and revocation using SQLite + fake vector effects."""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from codex_agent.accounts.schema import users
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.api import create_gateway
from codex_agent.knowledge import store as stores, vectors, publication, publication_policy, api
from codex_agent.knowledge.schema import TABLES, documents, versions, chunks
from codex_agent.knowledge.file_schema import sources, locations
from codex_agent.knowledge.lifecycle import controls
from codex_agent.knowledge.publication_schema import intents
from codex_agent.runtime_config import RuntimeConfig, Indexing
from codex_agent.storage.schema import meta
from codex_agent.tasks import store as task_store
from codex_agent.tasks.schema import QUEUE_TABLES, jobs, outbox
from codex_agent.tests.test_rag_catalog_api import Accounts, ADMIN, OTHER, ORIGIN, headers


@compiles(LONGTEXT, 'sqlite')
def sqlitetext(_type, _compiler, **_kw):return 'TEXT'


class DB:
    def __init__(self, root):
        self.root=root;self.workspace_id=hashlib.sha256(str(root).encode()).hexdigest()
        self.engine=create_engine('sqlite://',poolclass=StaticPool,connect_args={'check_same_thread':False})
        for table in [*TABLES,sources,locations,controls,intents,meta,*QUEUE_TABLES,users]:table.create(self.engine)
    @contextmanager
    def transaction(self):
        with self.engine.begin() as conn:yield conn
    def lock(self, conn, name):assert name in {'knowledge-write','task-submit','task-claim'}
    def predicate(self, table, *where):return (table.c.workspace_id==self.workspace_id,*where)
    def rows(self, table,*where,connection=None,order=(),limit=None,lock=False):
        if connection is None:
            with self.engine.connect() as conn:return self.rows(table,*where,connection=conn,order=order,limit=limit)
        stmt=select(table).where(*self.predicate(table,*where)).order_by(*order)
        if limit is not None:stmt=stmt.limit(limit)
        return [dict(r) for r in connection.execute(stmt).mappings()]
    def insert(self,conn,table,value):conn.execute(table.insert().values(workspace_id=self.workspace_id,**value))
    def update(self,conn,table,value,*where):return conn.execute(table.update().where(*self.predicate(table,*where)).values(**value))
    def delete(self,conn,table,*where):return conn.execute(table.delete().where(*self.predicate(table,*where)))


@pytest.fixture
def env(tmp_path,monkeypatch):
    db=DB((tmp_path/'knowledge').resolve())
    monkeypatch.setattr(stores,'engine',lambda:db.engine)
    library=stores.KnowledgeStore(db.root)
    monkeypatch.setattr(stores.KnowledgeStore,'writer',lambda _:db)
    for module in [task_store,stores,publication,publication_policy]:
        if hasattr(module,'now_ms'):monkeypatch.setattr(module,'now_ms',lambda _:1000)
    definition={'generation':'g1','maxTokens':500,'overlapTokens':0,'batchSize':100,
        'splitter':vectors.SPLITTER,'dimensions':2}
    monkeypatch.setattr(vectors,'index_definition',lambda:dict(definition))
    monkeypatch.setattr(vectors,'runtime_config',lambda:RuntimeConfig(indexing=Indexing(enabled=True)))
    with db.engine.begin() as conn:
        for uid in [ADMIN,OTHER]:conn.execute(users.insert().values(id=uid,username=uid,password_hash='unused',status='active',auth_version=1,created_ms=1))
    library.configure_publishers((ADMIN,OTHER))
    index=Index()
    yield SimpleNamespace(db=db,store=library,definition=definition,index=index,monkeypatch=monkeypatch)
    db.engine.dispose()


def value(request='request1',**kw):return stores.DocumentInput(title='Shared fact',source='reviewed',content='Old original fact',request_id=request,**kw)


def claim(env,identifier):
    j=task_store.JobStore(env.db.root,database=env.db)
    with env.db.engine.connect() as conn:event=env.db.rows(outbox,outbox.c.job_id==identifier,connection=conn)[0]
    job=j.claim(j.envelope(event),'index')
    return j,job


class Provider:
    max_input_tokens=500
    def count_tokens(self,text):return len(text)


class Index:
    def __init__(self):self.hits=[];self.after=None;self.failure=None;self.effects=0
    def index_chunks(self,rows,provider):
        self.effects+=1
        if self.failure:raise self.failure
        for r in rows:
            self.hits.append({'score':1,'payload':{'document_id':r.document_id,'version':r.version,
                'source_version':r.source_version,'chunk_id':r.chunk_id}})
        if self.after:self.after()
    def embed(self,*_):return [[1,2]]
    def search(self,*_,limit,offset):return self.hits[offset:offset+limit]


def execute(env,receipt):
    j,job=claim(env,receipt['sync']['job_id'])
    vectors.execute_knowledge_index(j,job,job['owner'],provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)


def test_confirmation_atomically_creates_source_intent_job_and_outbox(env):
    result=env.store.confirm(value(),ADMIN)
    assert result['sync']['state']=='pending' and result['state']=='pending'
    with env.db.engine.connect() as conn:
        for table in [documents,versions,intents,jobs,outbox]:assert len(env.db.rows(table,connection=conn))==1
    again=env.store.confirm(value(),ADMIN)
    assert again==result
    with pytest.raises(task_store.Conflict):env.store.confirm(value().model_copy(update={'content':'changed'}),ADMIN)
    with pytest.raises(KeyError):env.store.operation('request1',OTHER)
    assert env.store.detail(result['id'],ADMIN)['sync']['request_id']=='request1'
    assert env.store.detail(result['id'],OTHER)['sync'] is None


def test_enqueue_failure_rolls_back_draft_intent_and_request(env,monkeypatch):
    def fail(*_args,**_kwargs):raise RuntimeError('enqueue unavailable')
    monkeypatch.setattr(task_store.JobStore,'submit',fail)
    with pytest.raises(RuntimeError):env.store.confirm(value(),ADMIN)
    with env.db.engine.connect() as conn:
        for table in [*TABLES,intents,jobs,outbox]:assert not env.db.rows(table,connection=conn)


def test_success_publishes_only_after_index_then_is_visible_for_both_accounts(env):
    result=env.store.confirm(value(),ADMIN)
    assert vectors.search(env.db.root,'fact',provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)['status']=='empty'
    execute(env,result)
    receipt=env.store.operation('request1',ADMIN)
    assert receipt['sync']['state']=='published' and receipt['published_version']==1
    # Two independent consumer calls use the shared library root, no owner-local copy.
    for _account in [ADMIN,OTHER]:
        found=vectors.search(env.db.root,'fact',provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)
        assert found['items'][0]['version']==1 and found['items'][0]['text']=='Old original fact'
    assert env.store.confirm(value(),ADMIN)['sync']['state']=='published'
    with env.db.engine.connect() as conn:assert len(env.db.rows(jobs,connection=conn))==1


def test_replacement_failure_keeps_previously_published_version(env):
    first=env.store.confirm(value(),ADMIN);execute(env,first)
    new=value('replace',document_id=first['id'],expected_version=1).model_copy(update={'content':'New fact'})
    second=env.store.confirm(new,ADMIN)
    env.index.failure=TimeoutError('remote uncertain')
    j,job=claim(env,second['sync']['job_id'])
    with pytest.raises(TimeoutError):vectors.execute_knowledge_index(j,job,job['owner'],provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)
    with env.db.transaction() as conn:env.db.update(conn,jobs,{'status':'failed'},jobs.c.id==job['id'])
    receipt=env.store.operation('replace',ADMIN)
    assert receipt['sync']['state']=='failed' and receipt['published_version']==1
    assert env.store.published_document(first['id'],1)['content']=='Old original fact'


@pytest.mark.parametrize('change',['draft','withdraw','delete','revision','configuration','source','job','actor'])
def test_change_during_remote_index_never_publishes_stale_confirmation(env,change):
    result=env.store.confirm(value(),ADMIN)
    def alter():
        if change=='draft':env.store.save(value('new-draft',document_id=result['id'],expected_version=1),ADMIN)
        elif change=='withdraw':env.store.publish(result['id'],None,ADMIN)
        elif change=='delete':env.store.set_deleted(result['id'],1,0,ADMIN,deleted=True)
        elif change=='configuration':env.definition['generation']='g2'
        else:
            with env.db.transaction() as conn:
                if change=='revision':env.db.insert(conn,controls,{'document_id':result['id'],'revision':1,'deleted_at':None,'actor_id':ADMIN})
                elif change=='source':env.db.update(conn,versions,{'content':'tampered'},versions.c.document_id==result['id'])
                elif change=='job':env.db.update(conn,versions,{'job_id':'replacement'},versions.c.document_id==result['id'])
                elif change=='actor':conn.execute(users.update().where(users.c.id==ADMIN).values(status='disabled'))
    env.index.after=alter
    with pytest.raises((task_store.Conflict,task_store.LeaseLost,KeyError)):execute(env,result)
    with env.db.engine.connect() as conn:
        assert env.store.row(conn,documents,result['id'],allow_deleted=True)['published_version'] is None
        assert env.store.rows(conn,intents)[0]['state']!='published'


def test_manual_index_and_ordinary_save_never_create_publication_authority(env):
    saved=env.store.save(value(),ADMIN)
    env.store.enqueue(saved['id'],1,'manual-index',ADMIN)
    with env.db.engine.connect() as conn:
        assert not env.db.rows(intents,connection=conn)
        job=env.db.rows(jobs,connection=conn)[0]
    j,claimed=claim(env,job['id'])
    vectors.execute_knowledge_index(j,claimed,claimed['owner'],provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)
    assert env.store.detail(saved['id'])['published_version'] is None
    env.store.publish(saved['id'],1,ADMIN)
    assert env.store.detail(saved['id'])['published_version']==1


def test_successful_replacement_hides_old_vector_hits(env):
    first=env.store.confirm(value(),ADMIN);execute(env,first)
    second=env.store.confirm(value('replace',document_id=first['id'],expected_version=1).model_copy(update={'content':'New verified fact'}),ADMIN)
    execute(env,second)
    result=vectors.search(env.db.root,'fact',provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)
    assert {r['version'] for r in result['items']}=={2}
    assert all(r['text']=='New verified fact' for r in result['items'])
    with pytest.raises(KeyError):env.store.published_document(first['id'],1)


def test_interrupted_remote_upsert_can_retry_same_job_without_duplicate_version(env):
    result=env.store.confirm(value(),ADMIN)
    j,job=claim(env,result['sync']['job_id'])
    env.index.failure=TimeoutError('uncertain vector ack')
    with pytest.raises(TimeoutError):vectors.execute_knowledge_index(j,job,job['owner'],provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)
    assert env.store.operation('request1',ADMIN)['published_version'] is None
    j.retry_index(job['id'],job['owner'],{'reason':'uncertain vector ack'})
    with env.db.engine.connect() as conn:
        event=max(env.db.rows(outbox,outbox.c.job_id==job['id'],connection=conn),key=lambda row:row['generation'])
    retried=j.claim(j.envelope(event),'index')
    assert retried['owner']!=job['owner'] and retried['attempt']==2
    env.index.failure=None
    vectors.execute_knowledge_index(j,retried,retried['owner'],provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index)
    assert env.store.operation('request1',ADMIN)['sync']['state']=='published'
    with env.db.engine.connect() as conn:
        assert len(env.db.rows(versions,connection=conn))==1
        assert len(env.db.rows(intents,connection=conn))==1


@pytest.fixture
def client(env,tmp_path,monkeypatch):
    config=AccountConfig(origin=ORIGIN,data_dir=tmp_path/'accounts',template=tmp_path/'template',
        allow_local_http=True,knowledge_enabled=True,admin_user_ids=(ADMIN,))
    monkeypatch.setattr(api,'KnowledgeStore',lambda _:env.store)
    monkeypatch.setattr(api,'search',lambda _root,q,limit: vectors.search(env.db.root,q,limit=limit,
        provider_factory=lambda _:Provider(),index_factory=lambda *_:env.index))
    c=TestClient(create_gateway(config,accounts=Accounts(),workbenches=object()),base_url=ORIGIN)
    c.cookies.set(config.cookie_name,'admin-session')
    yield c,config
    c.close()


def test_api_confirmation_receipt_restores_poll_and_is_not_a_document_route(env,client):
    c,config=client
    response=c.post('/api/admin/knowledge/confirm',json=value().model_dump(),headers=headers())
    assert response.status_code==202,response.text
    result=response.json()
    read=c.get('/api/admin/knowledge/operations/request1')
    assert read.status_code==200 and read.json()['sync']['state']=='pending'
    assert c.get('/api/admin/knowledge/'+result['id']).json()['sync']['request_id']=='request1'
    assert c.get('/api/admin/knowledge').json()['documents'][0]['sync']['request_id']=='request1'
    execute(env,result)
    for cookie,uid in [('admin-session',ADMIN),('other-session',OTHER)]:
        c.cookies.set(config.cookie_name,cookie)
        response=c.post('/api/knowledge/search',json={'query':'fact'},headers=headers())
        assert response.status_code==200,response.text
        assert response.json()['items'][0]['version']==1
    assert c.get('/api/admin/knowledge/operations/request1').status_code==403
    assert c.post('/api/admin/knowledge/confirm',json=value().model_dump(),headers=headers()).status_code==403


def test_api_timeout_after_commit_is_resolved_by_actor_receipt_and_same_id_retry(env,client,monkeypatch):
    c,_=client
    original=env.store.confirm
    def late_commit(*args,**kwargs):
        original(*args,**kwargs)
        raise TimeoutError('response lost after commit')
    monkeypatch.setattr(env.store,'confirm',late_commit)
    response=c.post('/api/admin/knowledge/confirm',json=value().model_dump(),headers=headers())
    assert response.status_code==503 and 'receipt' in response.json()['detail']
    committed=c.get('/api/admin/knowledge/operations/request1')
    assert committed.status_code==200 and committed.json()['sync']['state']=='pending'
    monkeypatch.setattr(env.store,'confirm',original)
    retry=c.post('/api/admin/knowledge/confirm',json=value().model_dump(),headers=headers())
    assert retry.status_code==202 and retry.json()['sync']==committed.json()['sync']
    with env.db.engine.connect() as conn:
        for table in [documents,versions,intents,jobs,outbox]:assert len(env.db.rows(table,connection=conn))==1


def test_stale_ordinary_save_replay_does_not_revoke_new_confirmation(env):
    draft=env.store.save(value('draft'),ADMIN)
    confirmed=env.store.confirm(value('confirmed',document_id=draft['id'],expected_version=1),ADMIN)
    with pytest.raises(task_store.Conflict):env.store.save(value('draft'),ADMIN)
    assert env.store.operation('confirmed',ADMIN)['sync']['state']=='pending'
    execute(env,confirmed)
    assert env.store.detail(draft['id'])['published_version']==2


def test_invalid_configuration_creates_no_confirmation_or_draft(env,monkeypatch):
    def invalid():raise ValueError('index disabled')
    monkeypatch.setattr(vectors,'index_definition',invalid)
    with pytest.raises(ValueError):env.store.confirm(value(),ADMIN)
    with env.db.engine.connect() as conn:
        for table in [*TABLES,intents,jobs,outbox]:assert not env.db.rows(table,connection=conn)


def test_withdrawn_confirmation_replay_cannot_restart_indexing(env):
    confirmed=env.store.confirm(value(),ADMIN)
    env.store.publish(confirmed['id'],None,ADMIN)
    replay=env.store.confirm(value(),ADMIN)
    assert replay['sync']['state']=='withdrawn'
    with env.db.engine.connect() as conn:
        assert len(env.db.rows(jobs,connection=conn))==1
        assert env.db.rows(jobs,connection=conn)[0]['status']=='cancelled'


def test_missing_server_policy_cannot_create_publication_authority(env):
    with env.db.transaction() as conn:env.db.delete(conn,meta,meta.c.key==publication_policy.POLICY_KEY)
    with pytest.raises(task_store.Conflict):env.store.confirm(value(),ADMIN)
    with env.db.engine.connect() as conn:
        for table in [documents,versions,intents,jobs,outbox]:assert not env.db.rows(table,connection=conn)


def test_admin_removal_revokes_pending_work_without_disabling_account(env):
    first=env.store.confirm(value(),ADMIN)
    other=env.store.confirm(value('other'),OTHER)
    env.store.configure_publishers((OTHER,))
    with env.db.engine.connect() as conn:
        assert conn.execute(select(users.c.status).where(users.c.id==ADMIN)).scalar_one()=='active'
        revoked=env.db.rows(jobs,jobs.c.id==first['sync']['job_id'],connection=conn)[0]
        assert revoked['cancel_requested']==1 and revoked['status']=='cancelled'
    assert env.store.operation('request1',ADMIN)['sync']['state']=='withdrawn'
    assert env.store.operation('other',OTHER)['sync']['state']=='pending'
    env.store.configure_publishers((ADMIN,OTHER))
    assert env.store.confirm(value(),ADMIN)['sync']['state']=='withdrawn'
    execute(env,other)
    assert env.store.detail(other['id'])['published_version']==1


def test_publisher_removed_during_vector_effect_cannot_commit_and_regrant_does_not_revive(env):
    result=env.store.confirm(value(),ADMIN)
    env.index.after=lambda:env.store.configure_publishers((OTHER,))
    with pytest.raises(task_store.Conflict):execute(env,result)
    env.store.configure_publishers((ADMIN,OTHER))
    assert env.store.operation('request1',ADMIN)['sync']['state']=='withdrawn'
    assert env.store.detail(result['id'])['published_version'] is None


def test_disabling_knowledge_withdraws_all_pending_publications(env):
    result=env.store.confirm(value(),ADMIN)
    env.store.configure_publishers((ADMIN,OTHER),enabled=False)
    with pytest.raises(task_store.Conflict):env.store.confirm(value('blocked'),ADMIN)
    assert env.store.operation('request1',ADMIN)['sync']['state']=='withdrawn'
    env.store.configure_publishers((ADMIN,OTHER))
    assert env.store.confirm(value(),ADMIN)['sync']['state']=='withdrawn'


def test_confirm_route_accepts_same_large_review_limit_as_save(env,client,monkeypatch):
    c,_=client
    captured=[]
    monkeypatch.setattr(env.store,'confirm',lambda value,actor,**kw:captured.append(value) or {'ok':True})
    payload=dict(title='Large reviewed source',source='',content='',document_id='c'*32,expected_version=1,
        request_id='large-review',sections=[dict(id=f's{i+1}',text='学'*14000) for i in range(50)])
    raw=json.dumps(payload,ensure_ascii=False).encode()
    assert 2*1024*1024 < len(raw) < 8*1024*1024
    response=c.post('/api/admin/knowledge/confirm',content=raw,headers=headers(**{'content-type':'application/json'}))
    assert response.status_code==202,response.text
    assert len(captured)==1 and len(captured[0].sections)==50
    response=c.post('/api/admin/knowledge/confirm',content=b'{}',headers=headers(**{
        'content-type':'application/json','content-length':str(8*1024*1024+1)}))
    assert response.status_code==413 and len(captured)==1


def test_disabled_accounts_only_schema_has_no_authority_to_revoke(tmp_path,monkeypatch):
    engine=create_engine('sqlite://')
    users.create(engine)
    monkeypatch.setattr(stores,'engine',lambda:engine)
    library=stores.KnowledgeStore(tmp_path/'never-enabled-knowledge')
    def no_writer():raise AssertionError('must not register or migrate an unused knowledge workspace')
    monkeypatch.setattr(library,'writer',no_writer)
    library.configure_publishers((ADMIN,),enabled=False)
    assert not library.root.exists()
    engine.dispose()
