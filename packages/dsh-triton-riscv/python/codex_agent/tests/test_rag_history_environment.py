"""Historical DB credentials belong to the gateway, not account or index workers."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from codex_agent.accounts import workspaces
from codex_agent.accounts.config import AccountConfig, RagHistorySource
from codex_agent.accounts.store import AccountError
from codex_agent.knowledge import processes, store
from codex_agent.runtime_config import RuntimeConfig, Storage


@pytest.fixture
def environment(tmp_path, monkeypatch):
    history_env = 'TRITON_RAG_HISTORY_DAILY_URL'
    inherited = {
        history_env: 'mysql+pymysql://private-history@localhost/daily',
        'TRITON_RAG_HISTORY_UNREGISTERED_URL': 'mysql+pymysql://private-history@localhost/other',
        'TRITON_MYSQL_URL': 'mysql+pymysql://current@localhost/current',
        'CUSTOM_MAIN_DB_URL': 'mysql+pymysql://current@localhost/custom',
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    config = AccountConfig(origin='http://127.0.0.1:8766', allow_local_http=True,
        data_dir=tmp_path/'accounts', template=tmp_path/'template', knowledge_enabled=True,
        rag_history_sources=(RagHistorySource(id='history',label='Historical snapshot',
            url_env=history_env, workspace_ids=('a'*64,)),))
    cfg = RuntimeConfig(storage=Storage(urlEnv='CUSTOM_MAIN_DB_URL'))
    monkeypatch.setattr(workspaces, 'runtime_config', lambda: cfg)
    monkeypatch.setattr(processes, 'runtime_config', lambda: cfg)
    return SimpleNamespace(config=config, cfg=cfg, inherited=inherited)


def assert_private(child, original):
    assert not any(key.startswith('TRITON_RAG_HISTORY_') for key in child)
    assert child['TRITON_MYSQL_URL'] == original['TRITON_MYSQL_URL']
    assert child['CUSTOM_MAIN_DB_URL'] == original['CUSTOM_MAIN_DB_URL']
    for key, value in original.items():
        assert os.environ[key] == value  # Parent gateway retains its own credentials.


def test_account_environment_strips_history_credentials_only_from_child(environment, tmp_path):
    workbench = object.__new__(workspaces.Workbenches)
    workbench.config = environment.config
    env = workbench.environment(tmp_path/'workspace', 'a'*32, 'gateway-test')
    assert_private(env, environment.inherited)


def test_shared_workers_receive_no_history_credentials(environment, monkeypatch):
    launches = []
    configured = []
    monkeypatch.setattr(store, 'KnowledgeStore', lambda root: SimpleNamespace(
        configure_publishers=lambda ids,enabled: configured.append((ids,enabled))))
    def launch(*args, **kwargs):
        assert configured == [(environment.config.admin_user_ids,True)]
        launches.append(kwargs['env'])
        return Mock()
    monkeypatch.setattr(processes.subprocess, 'Popen', launch)
    worker = processes.KnowledgeWorkers(environment.config)
    worker.start()
    assert len(launches) == 3
    for env in launches:
        assert_private(env, environment.inherited)


def test_reserved_history_prefix_cannot_be_used_as_main_storage(environment, monkeypatch, tmp_path):
    cfg = RuntimeConfig(storage=Storage(urlEnv='TRITON_RAG_HISTORY_MAIN_URL'))
    monkeypatch.setattr(workspaces, 'runtime_config', lambda: cfg)
    monkeypatch.setattr(processes, 'runtime_config', lambda: cfg)
    workbench = object.__new__(workspaces.Workbenches)
    workbench.config = environment.config
    with pytest.raises(AccountError, match='reserved'):
        workbench.environment(tmp_path/'workspace', 'a'*32, 'gateway-test')
    popen = Mock(side_effect=AssertionError('must not launch a worker'))
    monkeypatch.setattr(processes.subprocess, 'Popen', popen)
    with pytest.raises(ValueError, match='reserved'):
        processes.KnowledgeWorkers(environment.config).start()
    popen.assert_not_called()


def test_disabled_shared_library_revokes_publishers_without_starting_workers(environment,monkeypatch):
    configured=[]
    monkeypatch.setattr(store,'KnowledgeStore',lambda root:SimpleNamespace(
        configure_publishers=lambda ids,enabled:configured.append((ids,enabled))))
    popen=Mock(side_effect=AssertionError('disabled library must not start workers'))
    monkeypatch.setattr(processes.subprocess,'Popen',popen)
    config=environment.config.model_copy(update={'knowledge_enabled':False})
    processes.KnowledgeWorkers(config).start()
    assert configured==[(config.admin_user_ids,False)]
    popen.assert_not_called()


def test_policy_sync_failure_prevents_starting_workers(environment,monkeypatch):
    def unavailable(*args,**kwargs):raise RuntimeError('policy unavailable')
    monkeypatch.setattr(store,'KnowledgeStore',lambda root:SimpleNamespace(configure_publishers=unavailable))
    popen=Mock(side_effect=AssertionError('missing policy must not start workers'))
    monkeypatch.setattr(processes.subprocess,'Popen',popen)
    with pytest.raises(RuntimeError,match='policy unavailable'):processes.KnowledgeWorkers(environment.config).start()
    popen.assert_not_called()
