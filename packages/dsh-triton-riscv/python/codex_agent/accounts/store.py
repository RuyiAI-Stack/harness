"""Invite-only accounts and revocable, opaque Redis login sessions."""
import hashlib
import json
import os
import re
import secrets
import time
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from redis import Redis
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from codex_agent.storage.database import engine
from .schema import audit, invitations, users


class AccountError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def now_ms():
    return int(time.time() * 1000)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def password_ok(password):
    if not isinstance(password, str) or not 12 <= len(password) <= 128 or len(password.encode()) > 512:
        raise AccountError(400, "Password must contain 12 to 128 characters")


class Accounts:
    def __init__(self, config):
        self.config, self.engine = config, engine()
        self.hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
        self.dummy = self.hasher.hash(secrets.token_urlsafe(32))
        url = os.environ.get(config.redis_url_env, "")
        if not url:
            raise ValueError("Set the dedicated authentication Redis URL")
        from codex_agent.runtime_config import runtime_config
        if url == os.environ.get(runtime_config().cache.urlEnv):
            raise ValueError("Authentication Redis must be separate from the application cache")
        self.redis = Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        self.prefix = "triton-auth:" + digest(config.origin + str(config.data_dir.resolve()))[:24] + ":"

    def health(self):
        self.redis.ping()
        # Failing to verify the eviction policy is a deployment error, not a reason to bypass it.
        if self.redis.config_get("maxmemory-policy").get("maxmemory-policy") != "noeviction":
            raise AccountError(503, "Authentication Redis requires maxmemory-policy=noeviction")
        with self.engine.connect() as conn:
            conn.execute(select(users.c.id).limit(1))

    def close(self):
        self.redis.close()

    def log(self, conn, action, user_id=None):
        conn.execute(audit.insert().values(id=uuid.uuid4().hex, user_id=user_id, action=action, created_ms=now_ms()))

    def limit(self, kind, subject, maximum=10, seconds=300):
        key = self.prefix + "rate:" + kind + ":" + digest(subject)
        count = self.redis.eval("local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],ARGV[1]) end; return n", 1, key, seconds)
        if count > maximum:
            raise AccountError(429, "Too many requests; try again later")

    def invite(self, seconds=86400):
        token = secrets.token_urlsafe(32)
        with self.engine.begin() as conn:
            conn.execute(invitations.insert().values(digest=digest(token), expires_ms=now_ms() + seconds * 1000))
            self.log(conn, "invitation-created")
        return token

    def register(self, username, password, invitation):
        username = username.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,31}", username):
            raise AccountError(400, "Username must contain 3 to 32 lowercase letters, digits, underscores or hyphens")
        password_ok(password)
        hashed = self.hasher.hash(password)
        uid = uuid.uuid4().hex
        try:
            with self.engine.connect() as conn:
                # Serialize signup capacity without relying on an eventually consistent cache.
                lock = "triton-account-register:" + digest(str(self.engine.url))[:24]
                if conn.exec_driver_sql("SELECT GET_LOCK(%s, 5)", (lock,)).scalar() != 1:
                    raise AccountError(503, "Registration busy; retry later")
                try:
                    invite = conn.execute(select(invitations).where(invitations.c.digest == digest(invitation)).with_for_update()).mappings().first()
                    if not invite or invite["used_by"] or invite["expires_ms"] <= now_ms():
                        raise AccountError(400, "Invitation invalid or expired")
                    if conn.execute(select(func.count()).select_from(users)).scalar_one() >= self.config.max_accounts:
                        raise AccountError(409, "Account capacity reached")
                    conn.execute(users.insert().values(id=uid, username=username, password_hash=hashed,
                        status="active", auth_version=1, created_ms=now_ms()))
                    conn.execute(update(invitations).where(invitations.c.digest == digest(invitation)).values(used_by=uid))
                    self.log(conn, "registered", uid)
                    # Commit before releasing the connection-owned admission lock.
                    conn.commit()
                finally:
                    if conn.in_transaction():
                        conn.rollback()
                    conn.exec_driver_sql("SELECT RELEASE_LOCK(%s)", (lock,))
                    conn.commit()
        except IntegrityError:
            raise AccountError(409, "Account could not be registered") from None
        return {"id": uid, "username": username, "auth_version": 1, "status": "active"}

    def verify(self, username, password):
        username = username.strip().lower()
        with self.engine.connect() as conn:
            user = conn.execute(select(users).where(users.c.username == username)).mappings().first()
        try:
            self.hasher.verify(user["password_hash"] if user else self.dummy, password)
        except (VerificationError, InvalidHashError):
            user = None
        if not user or user["status"] != "active":
            raise AccountError(401, "Invalid username or password")
        if self.hasher.check_needs_rehash(user["password_hash"]):
            with self.engine.begin() as conn:
                conn.execute(update(users).where(users.c.id == user["id"], users.c.password_hash == user["password_hash"])
                    .values(password_hash=self.hasher.hash(password)))
        return dict(user)

    def login(self, username, password):
        user = self.verify(username, password)
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        data = {"user_id": user["id"], "version": user["auth_version"], "csrf": csrf,
                "expires": int(time.time()) + self.config.absolute_seconds}
        key = self.prefix + "session:" + digest(token)
        index = self.prefix + "user:" + user["id"]
        # Hashes, never raw cookie tokens, are stored in Redis; sessions per account are bounded.
        self.redis.eval("""for _,key in ipairs(redis.call('ZRANGE',KEYS[2],0,-1)) do
            local raw=redis.call('GET',key);
            if not raw or cjson.decode(raw).version~=tonumber(ARGV[6]) then
              redis.call('DEL',key); redis.call('ZREM',KEYS[2],key); end; end;
          while redis.call('ZCARD',KEYS[2])>=tonumber(ARGV[4]) do
            local old=redis.call('ZPOPMIN',KEYS[2],1); redis.call('DEL',old[1]); end;
          redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[2]);
          redis.call('ZADD',KEYS[2],ARGV[3],KEYS[1]); redis.call('EXPIRE',KEYS[2],ARGV[5]); return 1""",
          2, key, index, json.dumps(data), self.config.idle_seconds, now_ms(),
          self.config.max_sessions, self.config.absolute_seconds, user["auth_version"])
        with self.engine.begin() as conn:
            self.log(conn, "login", user["id"])
        return token, {"id": user["id"], "username": user["username"], "csrf_token": csrf}

    def authenticate(self, token, *, touch=True):
        if not token or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            raise AccountError(401, "Please sign in")
        key = self.prefix + "session:" + digest(token)
        raw = self.redis.get(key)
        if not raw:
            raise AccountError(401, "Session expired; please sign in")
        try:
            data = json.loads(raw)
            remaining = int(data["expires"]) - int(time.time())
            with self.engine.connect() as conn:
                user = conn.execute(select(users).where(users.c.id == data["user_id"])).mappings().first()
            if remaining <= 0 or not user or user["status"] != "active" or data["version"] != user["auth_version"]:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            self.redis.delete(key)
            raise AccountError(401, "Session revoked; please sign in") from None
        if touch:
            # EXPIRE does not recreate a key concurrently removed by logout.
            if not self.redis.expire(key, min(remaining, self.config.idle_seconds)):
                raise AccountError(401, "Session revoked; please sign in")
        return {"id": user["id"], "username": user["username"], "csrf_token": data["csrf"]}

    def logout(self, token, uid):
        key = self.prefix + "session:" + digest(token)
        with self.redis.pipeline() as pipe:
            pipe.delete(key).zrem(self.prefix + "user:" + uid, key).execute()
        with self.engine.begin() as conn:
            self.log(conn, "logout", uid)

    def change_password(self, user, old, new):
        password_ok(new)
        checked = self.verify(user["username"], old)
        hashed = self.hasher.hash(new)
        with self.engine.begin() as conn:
            result = conn.execute(update(users).where(users.c.id == user["id"],
                users.c.auth_version == checked["auth_version"], users.c.status == "active")
                .values(password_hash=hashed, auth_version=users.c.auth_version + 1))
            if result.rowcount != 1:
                raise AccountError(409, "Account changed; sign in again")
            self.log(conn, "password-changed", user["id"])

    def set_status(self, username, active):
        with self.engine.begin() as conn:
            user = conn.execute(select(users).where(users.c.username == username)).mappings().first()
            if not user:
                raise AccountError(404, "Unknown account")
            conn.execute(update(users).where(users.c.id == user["id"]).values(
                status="active" if active else "disabled", auth_version=users.c.auth_version + 1))
            self.log(conn, "enabled" if active else "disabled", user["id"])
