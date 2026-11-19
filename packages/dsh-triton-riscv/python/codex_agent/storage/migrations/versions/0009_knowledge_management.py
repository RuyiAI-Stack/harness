"""Add reversible deletion without changing original document contents."""
from alembic import op

revision = "0009_knowledge_management"
down_revision = "0008_knowledge_files"


def upgrade():
    from codex_agent.knowledge.schema import TABLES as _parents
    from codex_agent.knowledge.lifecycle import controls
    controls.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
