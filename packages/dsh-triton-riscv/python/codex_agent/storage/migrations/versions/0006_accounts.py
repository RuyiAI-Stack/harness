"""Add account identities without assigning or changing historical workspaces."""
from alembic import op

revision = "0006_accounts"
down_revision = "0005_operator_workflows"


def upgrade():
    tables = (
        """account_users (id VARCHAR(32) PRIMARY KEY, username VARCHAR(64) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL, status VARCHAR(16) NOT NULL,
        auth_version BIGINT NOT NULL, created_ms BIGINT NOT NULL)""",
        """account_invitations (digest VARCHAR(64) PRIMARY KEY, expires_ms BIGINT NOT NULL,
        used_by VARCHAR(32), FOREIGN KEY (used_by) REFERENCES account_users(id))""",
        """account_workspaces (workspace_id VARCHAR(64) PRIMARY KEY, user_id VARCHAR(32) NOT NULL UNIQUE,
        root VARCHAR(1024) NOT NULL, status VARCHAR(16) NOT NULL,
        FOREIGN KEY (user_id) REFERENCES account_users(id))""",
        """account_audit (id VARCHAR(32) PRIMARY KEY, user_id VARCHAR(32),
        action VARCHAR(48) NOT NULL, created_ms BIGINT NOT NULL)""",
    )
    for ddl in tables:
        op.execute("CREATE TABLE IF NOT EXISTS " + ddl + " ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_bin")


def downgrade():
    raise RuntimeError("Destructive downgrade disabled; restore a verified backup instead")
