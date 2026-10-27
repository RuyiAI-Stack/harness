"""Authenticated same-origin gateway. Private APIs are never bound to TCP."""
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from .store import AccountError, Accounts, digest
from .workspaces import Workbenches


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class Registration(Credentials):
    invitation: str = Field(min_length=1, max_length=128)


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    old_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


async def limited_body(request, limit):
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > limit:
            raise AccountError(413, "Request too large")
    return bytes(body)


def create_gateway(config, *, accounts=None, workbenches=None):
    accounts = accounts or Accounts(config)
    workbenches = workbenches or Workbenches(accounts)

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.to_thread(accounts.health)
        workbenches.start()
        monitor = asyncio.create_task(workbenches.monitor())
        try:
            yield
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            await asyncio.to_thread(workbenches.close)
            accounts.close()

    app = FastAPI(title="Triton-RISCV Accounts", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.accounts, app.state.workbenches = accounts, workbenches

    @app.exception_handler(AccountError)
    async def account_error(_request, error):
        return JSONResponse({"detail": error.message}, status_code=error.status, headers={"Cache-Control": "no-store"})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _error):
        return JSONResponse({"detail": "Invalid request fields"}, status_code=422)

    @app.middleware("http")
    async def boundary(request, call_next):
        try:
            if request.headers.get("host", "").lower() != urlsplit(config.origin).netloc.lower():
                raise AccountError(400, "Unexpected host")
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                if request.headers.get("origin") != config.origin:
                    raise AccountError(403, "Same-origin request required")
                if request.headers.get("content-type", "").split(";")[0] != "application/json":
                    raise AccountError(415, "JSON request required")
            if request.url.path.startswith("/api/"):
                if request.headers.get("sec-fetch-site") == "cross-site":
                    raise AccountError(403, "Cross-site request rejected")
                length = request.headers.get("content-length")
                if length and (not length.isdigit() or int(length) > 2 * 1024 * 1024):
                    raise AccountError(413, "Request too large")
            response = await call_next(request)
        except AccountError as error:
            response = JSONResponse({"detail": error.message}, status_code=error.status)
        except (RedisError, SQLAlchemyError, httpx.HTTPError):
            response = JSONResponse({"detail": "Account service unavailable; access remains locked"}, status_code=503)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
        if config.secure_cookie:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    async def user_for(request, *, mutation=False, touch=True):
        token = request.cookies.get(config.cookie_name)
        user = await asyncio.to_thread(accounts.authenticate, token, touch=touch)
        # A browser tab may still display A after another tab rotates the cookie to B.
        # This is a consistency check, never an alternative source of identity.
        expected = [request.headers.get("x-triton-account")]
        if request.url.path.endswith("/events/stream"):
            expected.extend(request.query_params.getlist("account_id"))
        if any(value is not None and value != user["id"] for value in expected):
            raise AccountError(401, "Account changed; reload before continuing")
        if mutation and not secrets.compare_digest(request.headers.get("x-csrf-token", ""), user["csrf_token"]):
            raise AccountError(403, "CSRF check failed")
        return token, user

    @app.get("/api/auth/me")
    async def me(request: Request):
        _, user = await user_for(request, touch=False)
        return {"enabled": True, "user": user}

    async def credentials(request, schema):
        ip = request.client.host if request.client else "unknown"
        # Ignore arbitrary X-Forwarded-For; proxy trust must be configured at deployment.
        await asyncio.to_thread(accounts.limit, "auth-ip", ip, 30, 300)
        try:
            body = schema.model_validate_json(await limited_body(request, 4096))
        except ValueError:
            raise AccountError(422, "Invalid account fields") from None
        await asyncio.to_thread(accounts.limit, "auth-name", body.username.strip().lower(), 10, 300)
        return body

    @app.post("/api/auth/register", status_code=201)
    async def register(request: Request):
        body = await credentials(request, Registration)
        await asyncio.to_thread(accounts.register, body.username, body.password, body.invitation)
        return {"registered": True}

    @app.post("/api/auth/login")
    async def login(request: Request):
        body = await credentials(request, Credentials)
        token, user = await asyncio.to_thread(accounts.login, body.username, body.password)
        old = request.cookies.get(config.cookie_name)
        if old and old != token:
            # Rotation must not accept a browser-supplied session ID as the new ID.
            await asyncio.to_thread(accounts.redis.delete, accounts.prefix + "session:" + digest(old))
        response = JSONResponse({"enabled": True, "user": user})
        response.set_cookie(config.cookie_name, token, max_age=config.absolute_seconds,
            httponly=True, secure=config.secure_cookie, samesite="strict", path="/")
        return response

    @app.post("/api/auth/logout")
    async def logout(request: Request):
        token, user = await user_for(request, mutation=True)
        await asyncio.to_thread(accounts.logout, token, user["id"])
        response = JSONResponse({"signed_out": True})
        response.delete_cookie(config.cookie_name, path="/", secure=config.secure_cookie, httponly=True, samesite="strict")
        return response

    @app.post("/api/auth/password")
    async def password(request: Request):
        _, user = await user_for(request, mutation=True)
        await asyncio.to_thread(accounts.limit, "password", user["id"], 5, 300)
        try:
            body = PasswordChange.model_validate_json(await limited_body(request, 4096))
        except ValueError:
            raise AccountError(422, "Invalid password fields") from None
        await asyncio.to_thread(accounts.change_password, user, body.old_password, body.new_password)
        response = JSONResponse({"signed_out": True})
        response.delete_cookie(config.cookie_name, path="/", secure=config.secure_cookie, httponly=True, samesite="strict")
        return response

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PATCH", "DELETE", "PUT"])
    async def proxy(path: str, request: Request):
        token, user = await user_for(request, mutation=request.method not in {"GET", "HEAD"})
        await asyncio.to_thread(accounts.limit, "api", user["id"], 600, 60)
        entry = await asyncio.to_thread(workbenches.get, user["id"])
        if not entry["slots"].acquire(blocking=False):
            raise AccountError(429, "Too many concurrent requests")
        client, upstream = None, None
        try:
            content = await limited_body(request, 2 * 1024 * 1024)
            client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(entry["socket"])),
                base_url="http://private", timeout=httpx.Timeout(60, connect=5))
            headers = {"x-triton-gateway": entry["key"], "content-type": "application/json"}
            # Never forward cookies, arbitrary identity headers, or a caller-selected destination.
            target = "/api/" + path
            if request.url.query:
                target += "?" + request.url.query
            upstream = await client.send(client.build_request(request.method, target, content=content, headers=headers), stream=True)
            response_headers = {k: v for k, v in upstream.headers.items() if k.lower() in {
                "content-type", "content-disposition"}}
            response_headers["Cache-Control"] = "no-store"
        except BaseException:
            if upstream:
                await upstream.aclose()
            if client:
                await client.aclose()
            entry["slots"].release()
            raise

        async def stream():
            try:
                async for chunk in upstream.aiter_bytes():
                    # Recheck existing streams too: logout/disable is not just a page redirect.
                    await asyncio.to_thread(accounts.authenticate, token, touch=False)
                    yield chunk
            except (AccountError, RedisError, SQLAlchemyError):
                return
            finally:
                await upstream.aclose()
                await client.aclose()
                entry["slots"].release()
        return StreamingResponse(stream(), status_code=upstream.status_code, headers=response_headers)

    dist = Path(__file__).parents[1] / "frontend/dist"
    if dist.is_dir():
        app.mount("/ui-assets", StaticFiles(directory=dist), name="ui-assets")

    @app.get("/")
    def index():
        if not (dist / "index.html").is_file():
            return JSONResponse({"detail": "Build the workbench frontend first"}, status_code=503)
        return FileResponse(dist / "index.html")

    return app
