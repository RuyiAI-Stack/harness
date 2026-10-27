"""Global identity tables; private domain rows retain their workspace keys."""
from sqlalchemy import BigInteger, Column, ForeignKey, MetaData, String, Table

metadata = MetaData()
options = dict(mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin")
users = Table("account_users", metadata,
    Column("id", String(32), primary_key=True),
    Column("username", String(64), nullable=False, unique=True),
    Column("password_hash", String(255), nullable=False),
    Column("status", String(16), nullable=False),
    Column("auth_version", BigInteger, nullable=False),
    Column("created_ms", BigInteger, nullable=False), **options)
invitations = Table("account_invitations", metadata,
    Column("digest", String(64), primary_key=True),
    Column("expires_ms", BigInteger, nullable=False),
    Column("used_by", String(32), ForeignKey("account_users.id")), **options)
workspaces = Table("account_workspaces", metadata,
    Column("workspace_id", String(64), primary_key=True),
    Column("user_id", String(32), ForeignKey("account_users.id"), nullable=False, unique=True),
    Column("root", String(1024), nullable=False),
    Column("status", String(16), nullable=False), **options)
audit = Table("account_audit", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", String(32)),
    Column("action", String(48), nullable=False),
    Column("created_ms", BigInteger, nullable=False), **options)
