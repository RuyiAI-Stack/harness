"""Isolated reconciliation tests; fake read-only connectors, no embeddings/network."""
import copy
import hashlib
import json
from pathlib import Path
import time

import pytest

from codex_agent.memory_lineage import CASE_FIELDS, case_identity, digest
from codex_agent.rag_catalog.audit import (AuditBlocked, ReadOnlyVector,
    SQLReadSource, load_manifest, run_audit)


def fixture_data(tmp_path):
    root = str((tmp_path / 'workspace').resolve())
    scope = hashlib.sha256(root.encode()).hexdigest()
    payload = dict.fromkeys(CASE_FIELDS)
    payload.update(memory_type='experience', operator='test', semantics='PRIVATE-BODY',
        summary='PRIVATE-SUMMARY', pytorch_reference='', tl_ops=[], environment={'api_key': 'NEVER-OUTPUT'}, evidence={},
        outcome='passed', confidence_grade='A', source_run='fixture')
    version = digest(payload)
    parent = {**payload, 'id': 1, 'workspace_id': scope, 'case_version': version,
        'searchable_text': 'PRIVATE-BODY', 'active': 1,
        'tl_ops_json': '[]', 'environment_json': json.dumps(payload['environment']), 'evidence_json': '{}'}
    entry = dict(workspace_id=scope, generation='g', memory_id=1, revision=1,
        case_version=version, status='ready', desired='present', point_count=1)
    chunk = dict(workspace_id=scope, generation='g', memory_id=1, revision=1,
        case_version=version, chunk_id='a' * 64, point_id='00000000-0000-0000-0000-000000000001',
        text='PRIVATE-CHUNK', provenance_json=json.dumps({'case_id': case_identity(scope, 1),
            'text_sha256': hashlib.sha256(b'PRIVATE-CHUNK').hexdigest()}))
    definition = {'splitter': 'structural-v1', 'qdrantUrl': 'http://127.0.0.1:16333', 'apiKeyEnv': 'TRITON_QDRANT_API_KEY', 'dimensions': 1024, 'timeoutSeconds': 5}
    index = dict(workspace_id=scope, generation='g', fingerprint=digest(definition),
        collection='registered_collection', definition=definition)
    point = dict(id=chunk['point_id'], payload={k:chunk[k] for k in
        ('workspace_id','generation','memory_id','revision','case_version','chunk_id')})
    point['payload']['index_fingerprint'] = index['fingerprint']
    manifest = {'schema_version': 1, 'vector_services': [{'url': 'http://127.0.0.1:16333', 'api_key_env': 'TRITON_QDRANT_API_KEY'}], 'sources': [{'alias':'history','url_env':'TEST_HISTORY_URL',
        'scopes':[{'id':scope, 'root':root, 'kind':'memory_case'}]}]}
    return dict(scope=scope, parent=parent, entry=entry, chunk=chunk, index=index,
        point=point, manifest=manifest, version=version)


class FakeSource:
    def __init__(self, data):
        self.data = data
        self.rows = {'sources':[data['parent']], 'entries':[data['entry']], 'chunks':[data['chunk']]}
        self.version_binding = 'initial'
        self.revoked = False
        self.snapshot_missing = set()
        self.timeout_phase = None
        self.page_calls = []
        self.binding_calls = 0

    def grants(self):
        if self.revoked:
            raise PermissionError('PRIVATE CREDENTIAL MUST NOT APPEAR')
        return 'granted'

    def binding(self, deadline):
        self.binding_calls += 1
        return dict(grant=self.grants(), versions=self.version_binding), [self.data['index']]

    def page(self, scope, phase, cursor, limit):
        self.page_calls.append((scope, phase, cursor, limit))
        assert scope == self.data['scope'] and limit <= 100
        if self.timeout_phase == phase:
            raise TimeoutError('PRIVATE CREDENTIAL MUST NOT APPEAR')
        offset = (cursor or [0])[0]
        rows = self.rows[phase][offset:offset + limit]
        return rows, [offset + len(rows)] if offset + len(rows) < len(self.rows[phase]) else None

    def parent(self, scope, identifier):
        assert scope == self.data['scope']
        return next((r for r in self.rows['sources'] if r['id'] == identifier), None)

    def snapshot(self, scope, identifier, version):
        if (identifier, version) in self.snapshot_missing:
            return 'missing'
        return 'valid' if identifier == 1 and version == self.data['version'] else 'missing'

    def entry(self, scope, generation, identifier):
        return next((r for r in self.rows['entries'] if r['memory_id'] == identifier and r['generation'] == generation), None)

    def chunk_count(self, scope, entry):
        return sum(all(c[k] == entry[k] for k in ('memory_id','generation','revision','case_version')) for c in self.rows['chunks'])

    def vector_chunk(self, scope, generation, point_id):
        return [r for r in self.rows['chunks'] if r['point_id'] == point_id and r['generation'] == generation]


class FakeVector:
    def __init__(self, data):
        self.data = data
        self.rows = [data['point']]
        self.calls = []
        self.fail = False

    def points(self, ids, deadline):
        self.calls.append(('points', ids))
        if self.fail:
            raise TimeoutError('SECRET')
        return [p for p in self.rows if p['id'] in ids]

    def page(self, scope, cursor, limit, deadline):
        self.calls.append(('page',scope,cursor,limit))
        assert scope == self.data['scope'] and limit <= 100
        offset = int(cursor or 0)
        rows = self.rows[offset:offset+limit]
        return rows, offset+len(rows) if offset+len(rows)<len(self.rows) else None


@pytest.fixture
def data(tmp_path):
    return fixture_data(tmp_path)


def execute(data, tmp_path, source, vector, **kw):
    return run_audit(data['manifest'], tmp_path / 'audit', source_factory=lambda _: source,
        vector_factory=lambda _: vector, **kw)


def codes(result):
    return {row['code'] for row in result['findings'].values()}


def test_complete_bidirectional_audit_no_body_or_vectors(data, tmp_path):
    source, vector = FakeSource(data), FakeVector(data)
    state = execute(data, tmp_path, source, vector)
    assert state['status'] == 'complete' and not state['findings']
    assert state['counts'] == dict(sources=1, entries=1, chunks=1, vector_points=1)
    stored = (tmp_path/'audit/checkpoint.json').read_text()
    assert all(value not in stored for value in ('PRIVATE-BODY','PRIVATE-SUMMARY','PRIVATE-CHUNK','NEVER-OUTPUT','api_key'))
    assert (tmp_path/'audit/checkpoint.json').stat().st_mode & 0o777 == 0o600
    assert source.binding_calls == 2


def test_missing_source_and_historical_snapshot_are_distinct(data, tmp_path):
    source, vector = FakeSource(data), FakeVector(data)
    old = copy.deepcopy(data['point'])
    old['id'] = '00000000-0000-0000-0000-000000000002'
    old['payload']['case_version'] = 'b' * 64
    orphan = copy.deepcopy(data['point'])
    orphan['id'] = '00000000-0000-0000-0000-000000000003'
    orphan['payload']['memory_id'] = 99
    vector.rows.extend([old, orphan])
    state = execute(data, tmp_path, source, vector)
    assert state['status'] == 'complete'
    assert {'point_source_missing','point_snapshot_missing','point_version_not_current',
        'point_chunk_missing_or_ambiguous'} <= codes(state)
    old_issues = [f for f in state['findings'].values() if f['version'] == 'b'*64]
    assert any(f['code']=='point_snapshot_missing' for f in old_issues)
    assert not any(f['code']=='point_source_missing' for f in old_issues)


def test_ready_sql_chunk_without_vector_and_corrupt_text(data, tmp_path):
    source, vector = FakeSource(data), FakeVector(data)
    vector.rows=[]
    source.rows['chunks'][0]['text']='CHANGED PRIVATE BODY'
    state=execute(data,tmp_path,source,vector)
    assert {'ready_chunk_vector_missing','chunk_hash_mismatch'} <= codes(state)


def test_page_budget_resume_no_duplicates(data, tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    state=execute(data,tmp_path,source,vector,max_pages=1)
    assert state['status']=='partial' and state['pages_completed']==1
    assert state['task_index']==1
    state=execute(data,tmp_path,source,vector,resume=True)
    assert state['status']=='complete' and state['counts']['sources']==1 and state['pages_completed']==4


def test_timeout_keeps_last_checkpoint_and_resume(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    source.timeout_phase='entries'
    state=execute(data,tmp_path,source,vector)
    assert state['status']=='partial' and state['task_index']==1 and state['pages_completed']==1
    assert state['counts']['entries']==0 and not state['findings']
    assert 'PRIVATE' not in json.dumps(state)
    source.timeout_phase=None
    state=execute(data,tmp_path,source,vector,resume=True)
    assert state['status']=='complete' and state['counts']['sources']==1


def test_manifest_changed_resume_rejected_before_connection(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    execute(data,tmp_path,source,vector,max_pages=1)
    data['manifest']['sources'][0]['url_env']='OTHER_HISTORY_URL'
    called=[]
    state=run_audit(data['manifest'],tmp_path/'audit',resume=True,
        source_factory=lambda _: called.append(True))
    assert state['status']=='stale' and not called


def test_source_changed_or_revoked_on_resume(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    execute(data,tmp_path,source,vector,max_pages=1)
    source.version_binding='changed'
    state=execute(data,tmp_path,source,vector,resume=True)
    assert state['status']=='stale'
    other=tmp_path/'other'
    source.version_binding='initial'
    execute(data,other,source,vector,max_pages=1)
    source.revoked=True
    state=execute(data,other,source,vector,resume=True)
    assert state['status']=='blocked' and 'PRIVATE' not in json.dumps(state)


def test_deadline_checkpoint_and_no_resume_wrong_scope(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    original=source.page
    def slow(*args):
        result=original(*args)
        time.sleep(.02)
        return result
    source.page=slow
    state=execute(data,tmp_path,source,vector,deadline_seconds=.01)
    assert state['status']=='partial' and state['pages_completed']==0
    assert state['counts']['sources']==0
    data['manifest']['sources'][0]['scopes'][0]['root']='/different'
    with pytest.raises(AuditBlocked):
        execute(data,tmp_path,source,vector,resume=True)


def test_readonly_vector_transport_bounded_identity_only(data):
    class Transport:
        timeout=20
        def __init__(self,*args):
            self.calls=[]
        def request(self,method,path,value):
            self.calls.append((method,path,value))
            assert method=='POST' and path in {'/points','/points/scroll'}
            assert value['with_vector'] is False
            assert set(value['with_payload'])=={'workspace_id','generation','memory_id','revision','case_version','chunk_id','index_fingerprint'}
            assert 'text' not in value['with_payload']
            return [] if path=='/points' else {'points':[],'next_page_offset':None}
        def ensure(self):
            pytest.fail('read audit must not ensure a collection')
    reader=ReadOnlyVector(data['index'],factory=Transport,allowed_services=data['manifest']['vector_services'])
    reader.points([data['point']['id']],time.monotonic()+5)
    reader.page(data['scope'],None,100,time.monotonic()+5)
    assert reader.transport.timeout<=5
    filters=reader.transport.calls[-1][2]['filter']['must']
    assert {r['key'] for r in filters}=={'workspace_id','generation'}


def test_nonadvancing_vector_cursor_blocks(data):
    class Transport:
        timeout=1
        def __init__(self,*args): pass
        def request(self,*args): return {'points':[],'next_page_offset':'same'}
    reader=ReadOnlyVector(data['index'],factory=Transport,allowed_services=data['manifest']['vector_services'])
    with pytest.raises(AuditBlocked,match='advance'):
        reader.page(data['scope'],'same',10,time.monotonic()+1)


def test_sql_adapter_real_keyset_queries_and_readonly_surface(data,tmp_path):
    from sqlalchemy import create_engine,event
    from sqlalchemy.dialects.mysql import LONGTEXT
    from sqlalchemy.ext.compiler import compiles
    from codex_agent.storage.schema import (metadata,workspaces,memories,memory_versions,
        chunks,memory_indexes,memory_index_entries,memory_index_chunks)
    @compiles(LONGTEXT,'sqlite')
    def as_text(_type,_compiler,**kw): return 'TEXT'
    engine=create_engine('sqlite+pysqlite:///:memory:')
    tables=[workspaces,memories,memory_versions,chunks,memory_indexes,memory_index_entries,memory_index_chunks]
    metadata.create_all(engine,tables=tables)
    stamp='2026-11-13T00:00:00Z'
    parent={k:v for k,v in data['parent'].items() if k in memories.c}
    parent.update(fingerprint='f'*64,confidence=1.0,created_at=stamp,updated_at=stamp,
        useful_count=0,test_sha256=None)
    snapshot={k:data['parent'].get(k) for k in CASE_FIELDS}
    with engine.begin() as conn:
        conn.execute(workspaces.insert().values(workspace_id=data['scope'],root=data['manifest']['sources'][0]['scopes'][0]['root']))
        conn.execute(memories.insert().values(**parent))
        conn.execute(memory_versions.insert().values(workspace_id=data['scope'],memory_id=1,
            case_version=data['version'],payload_json=json.dumps(snapshot),captured_at=stamp))
        conn.execute(memory_indexes.insert().values(workspace_id=data['scope'],generation='g',status='staging',
            fingerprint=data['index']['fingerprint'],collection='registered_collection',
            config_json=json.dumps(data['index']['definition'])))
        conn.execute(memory_index_entries.insert().values(**data['entry'],job_id='fixture'))
        conn.execute(memory_index_chunks.insert().values(**data['chunk']))
    statements=[]
    @event.listens_for(engine,'before_cursor_execute')
    def assert_read(_conn,_cur,statement,parameters,context,executemany):
        statements.append(statement)
        assert statement.lstrip().upper().startswith(('SELECT','PRAGMA'))
    reader=SQLReadSource(data['manifest']['sources'][0],engine=engine)
    result=execute(data,tmp_path,reader,FakeVector(data))
    assert result['status']=='complete' and not result['findings']
    assert any('memory_index_chunks' in sql for sql in statements)
    assert all('embedding_json' not in sql for sql in statements)
    page,cursor=reader.page(data['scope'],'sources',None,1)
    assert len(page)==1 and cursor==[1]
    page,cursor=reader.page(data['scope'],'sources',cursor,1)
    assert page==[] and cursor is None
    engine.dispose()


def test_grants_change_after_page_discards_uncommitted_page(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    original=source.page
    def revoke(*args):
        rows=original(*args)
        source.revoked=True
        return rows
    source.page=revoke
    state=execute(data,tmp_path,source,vector)
    assert state['status']=='blocked' and state['pages_completed']==0
    assert state['task_index']==0 and state['counts']['sources']==0


def test_tampered_resume_task_cannot_read_ungranted_scope(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    execute(data,tmp_path,source,vector,max_pages=1)
    path=tmp_path/'audit/checkpoint.json'
    state=json.loads(path.read_text())
    state['tasks'][state['task_index']]['scope']='0'*64
    path.write_text(json.dumps(state))
    source.page_calls.clear()
    state=execute(data,tmp_path,source,vector,resume=True)
    assert state['status']=='blocked' and state['reason']=='checkpoint_task_not_granted'
    assert not source.page_calls


def test_oversized_checkpoint_and_lock_symlink_rejected(data,tmp_path,monkeypatch):
    import codex_agent.rag_catalog.audit as audit
    source,vector=FakeSource(data),FakeVector(data)
    execute(data,tmp_path,source,vector,max_pages=1)
    monkeypatch.setattr(audit,'MAX_CHECKPOINT_BYTES',1)
    with pytest.raises(AuditBlocked,match='size'):
        execute(data,tmp_path,source,vector,resume=True)
    other=tmp_path/'symlink'
    (other/'audit').mkdir(parents=True)
    (other/'audit/audit.lock').symlink_to(tmp_path/'real-lock')
    with pytest.raises(AuditBlocked,match='symlink'):
        execute(data,other,source,vector)


def test_change_during_run_is_stale_not_complete(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    original=source.page
    def changed(*args):
        rows=original(*args)
        source.version_binding='changed-during-read'
        return rows
    source.page=changed
    state=execute(data,tmp_path,source,vector)
    assert state['status']=='stale' and state['reason']=='source_changed_during_audit'


def test_anomalies_deduplicate_by_source_scope_version_and_reason(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    for i in (2,3):
        point=copy.deepcopy(data['point'])
        point['id']=f'00000000-0000-0000-0000-{i:012d}'
        point['payload']['memory_id']=99
        vector.rows.append(point)
    state=execute(data,tmp_path,source,vector)
    missing=[f for f in state['findings'].values() if f['code']=='point_source_missing']
    assert len(missing)==1 and missing[0]['occurrences']==2 and len(missing[0]['examples'])==2


def test_findings_budget_preserves_previous_page(data,tmp_path,monkeypatch):
    import codex_agent.rag_catalog.audit as audit
    source,vector=FakeSource(data),FakeVector(data)
    source.rows['sources'][0]['searchable_text']=''
    source.rows['sources'][0]['case_version']=None
    monkeypatch.setattr(audit,'MAX_FINDINGS',1)
    state=execute(data,tmp_path,source,vector)
    assert state['status']=='blocked' and state['reason'].startswith('finding_budget')
    assert state['pages_completed']==0 and not state['findings']


@pytest.mark.parametrize('field,value', [('qdrantUrl','https://ungranted.example'),('apiKeyEnv','UNRELATED_PRIVATE_TOKEN')])
def test_sql_definition_cannot_choose_endpoint_or_secret(data,tmp_path,field,value):
    source,vector=FakeSource(data),FakeVector(data)
    data['index']['definition'][field]=value
    data['index']['fingerprint']=digest(data['index']['definition'])
    constructed=[]
    state=run_audit(data['manifest'],tmp_path/'audit',source_factory=lambda _:source,
        vector_factory=lambda index:constructed.append(index))
    assert state['status']=='blocked' and state['reason']=='vector_endpoint_or_key_not_granted'
    assert not constructed and not source.page_calls
    with pytest.raises(AuditBlocked,match='not_granted'):
        ReadOnlyVector(data['index'],factory=lambda *a:constructed.append(a),
            allowed_services=data['manifest']['vector_services'])
    assert not constructed


def test_missing_network_grant_blocks_before_transport(data,tmp_path):
    source,vector=FakeSource(data),FakeVector(data)
    data['manifest'].pop('vector_services')
    result=execute(data,tmp_path,source,vector)
    assert result['status']=='blocked' and not vector.calls
