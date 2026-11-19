"""Add shared document knowledge without changing private operator memories."""
from alembic import op

revision = "0007_knowledge"
down_revision = "0006_accounts"


def upgrade():
    from codex_agent.knowledge.schema import TABLES
    for table in TABLES:
        table.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
