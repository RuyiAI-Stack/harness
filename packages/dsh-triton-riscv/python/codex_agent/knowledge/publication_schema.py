"""Durable confirmation authority, separate from mutable job delivery state."""
from sqlalchemy import Column, ForeignKeyConstraint, Index, Integer, String
from codex_agent.storage.schema import table, timestamp

intents = table('knowledge_publication_intents',
    Column('id', String(64), primary_key=True),
    Column('actor_id', String(64), nullable=False),
    Column('request_id', String(100), nullable=False),
    Column('request_fingerprint', String(64), nullable=False),
    Column('document_id', String(32), nullable=False),
    Column('version', Integer, nullable=False),
    Column('catalog_revision', Integer, nullable=False),
    Column('source_hash', String(64), nullable=False),
    Column('fingerprint', String(64), nullable=False),
    Column('job_id', String(64), nullable=False),
    Column('state', String(24), nullable=False),
    Column('error', String(255)), timestamp('created_at'), timestamp('updated_at'),
    ForeignKeyConstraint(['workspace_id', 'document_id', 'version'],
        ['knowledge_versions.workspace_id', 'knowledge_versions.document_id', 'knowledge_versions.version']))
Index('idx_knowledge_publication_document', intents.c.workspace_id, intents.c.document_id, intents.c.version)
