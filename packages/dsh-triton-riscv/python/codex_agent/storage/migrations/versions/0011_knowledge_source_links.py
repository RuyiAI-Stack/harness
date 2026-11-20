"""Track shared copies without mutating historical source databases."""
from alembic import op

revision = "0011_knowledge_source_links"
down_revision = "0010_knowledge_publication"


def upgrade():
    from codex_agent.knowledge.schema import TABLES as _parents
    from codex_agent.knowledge.source_links import source_links
    source_links.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
