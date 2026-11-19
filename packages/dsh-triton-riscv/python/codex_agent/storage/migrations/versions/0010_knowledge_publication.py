"""Persist explicit confirmation-to-publication authority with job identity."""
from alembic import op

revision = '0010_knowledge_publication'
down_revision = '0009_knowledge_management'


def upgrade():
    from codex_agent.knowledge.schema import TABLES as _parents
    from codex_agent.knowledge.publication_schema import intents
    intents.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError('Destructive downgrade disabled; restore a verified backup instead')
