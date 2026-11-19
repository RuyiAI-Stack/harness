"""Public gateway settings; credentials stay in declared environment variables."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import re


class RagHistorySource(BaseModel):
    """Server-owned, read-only grants to a historical memory database."""
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")
    label: str = Field(min_length=1, max_length=120)
    # Reserved exclusively for gateway history reads; never inherited by workers.
    url_env: str = Field(pattern=r"^TRITON_RAG_HISTORY_[A-Z][A-Z0-9_]*$")
    workspace_ids: tuple[str, ...]
    memory_mode: Literal["unknown", "qdrant"] = "unknown"
    owner_user_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    # Reading another source never implicitly authorizes publishing its content.
    allow_share: bool = False
    # Explicit administrator-owned endpoint for bounded read-only vector checks.
    vector_url: str | None = None
    vector_api_key_env: str | None = Field(default=None, pattern=r"^TRITON_RAG_HISTORY_[A-Z][A-Z0-9_]*$")

    @model_validator(mode="after")
    def check_source(self):
        if self.id == "current" or not self.label.strip() or "\0" in self.label:
            raise ValueError("Invalid historical source identity")
        if (not 1 <= len(self.workspace_ids) <= 100 or len(set(self.workspace_ids)) != len(self.workspace_ids)
                or any(not re.fullmatch(r"[a-f0-9]{64}", wid) for wid in self.workspace_ids)):
            raise ValueError("Explicit canonical workspace IDs are required")
        if self.vector_url:
            url = urlsplit(self.vector_url)
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                    or url.query or url.fragment or url.path not in {"", "/"}):
                raise ValueError("vector_url must be a credential-free HTTP(S) origin")
        return self


class AccountConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    origin: str
    data_dir: Path
    template: Path
    listen_port: int = Field(default=8766, ge=1024, le=65535)
    redis_url_env: str = Field(default="TRITON_AUTH_REDIS_URL", pattern=r"^[A-Z][A-Z0-9_]*$")
    idle_seconds: int = Field(default=1800, ge=60, le=86400)
    absolute_seconds: int = Field(default=43200, ge=300, le=604800)
    max_accounts: int = Field(default=16, ge=1, le=1000)
    max_active: int = Field(default=4, ge=1, le=32)
    max_sessions: int = Field(default=5, ge=1, le=20)
    allow_local_http: bool = False
    admin_user_ids: tuple[str, ...] = ()
    knowledge_enabled: bool = False
    rag_catalog_enabled: bool = True
    # Accepted for old config files only; the current catalog is shared-only.
    # Historical memory access must use explicit read-only rag_history_sources.
    rag_workspace_ids: tuple[str, ...] = ()
    rag_history_sources: tuple[RagHistorySource, ...] = ()

    @model_validator(mode="after")
    def check(self):
        url = urlsplit(self.origin)
        local = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.path or url.query or url.fragment
                or (url.scheme != "https" and not (local and self.allow_local_http))):
            raise ValueError("origin must be an HTTPS origin, or explicitly enabled loopback HTTP")
        if not self.data_dir.is_absolute() or not self.template.is_absolute():
            raise ValueError("data_dir and template must be absolute paths")
        if self.idle_seconds > self.absolute_seconds:
            raise ValueError("idle lifetime must not exceed absolute lifetime")
        if self.data_dir.resolve() == self.template.resolve() or self.data_dir.resolve().is_relative_to(self.template.resolve()):
            raise ValueError("account state must be outside the template")
        if len(self.admin_user_ids) > 100 or any(not re.fullmatch(r"[a-f0-9]{32}", uid) for uid in self.admin_user_ids):
            raise ValueError("admin_user_ids must contain explicit stable local account IDs")
        if len(self.rag_workspace_ids) > 100 or any(not re.fullmatch(r"[a-f0-9]{64}", wid) for wid in self.rag_workspace_ids):
            raise ValueError("rag_workspace_ids must contain explicit canonical workspace IDs")
        if len(self.rag_history_sources) > 16 or len({s.id for s in self.rag_history_sources}) != len(self.rag_history_sources):
            raise ValueError("Historical source IDs must be unique, with at most 16 sources")
        return self

    @property
    def secure_cookie(self):
        return self.origin.startswith("https://")

    @property
    def cookie_name(self):
        return "__Host-triton_session" if self.secure_cookie else "triton_local_session"


def load_config(path):
    raw = Path(path).read_bytes()
    if len(raw) > 16384:
        raise ValueError("account configuration too large")
    return AccountConfig.model_validate(json.loads(raw))
