"""Public gateway settings; credentials stay in declared environment variables."""
import json
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
