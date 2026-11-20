"""Add private PDF/DOCX provenance and chunk source locations."""
from alembic import op

revision = "0008_knowledge_files"
down_revision = "0007_knowledge"


def upgrade():
    from codex_agent.knowledge.schema import TABLES as _parents
    from codex_agent.knowledge.file_schema import TABLES
    for table in TABLES:
        table.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
