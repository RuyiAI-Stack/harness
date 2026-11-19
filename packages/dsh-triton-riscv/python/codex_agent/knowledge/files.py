"""Private originals, durable parse requests and lease-fenced parse completion."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets

from sqlalchemy import func, select

from codex_agent.platform.store import timestamp
from codex_agent.tasks.store import Conflict, JobStore, digest, encode
from codex_agent.tasks.schema import jobs as job_table
from .file_schema import sources
from .parsing import ParseError, check_signature, filename_format, isolated_parse, parser_version
from .schema import documents, versions, requests
from .lifecycle import require_active


def source_hash(title, source, content, sections=None):
    return digest([title, source, content] if sections is None else [title, source, content, sections])


def source_row(store, conn, identifier, version):
    return next(iter(store.rows(conn, sources, sources.c.document_id == identifier,
                               sources.c.version == version)), None)


def file_path(store, identifier, *, preview=False):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Invalid original file identifier")
    directory = store.root / ("previews" if preview else "originals")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.resolve().parent != store.root:
        raise ValueError("Unsafe original file directory")
    path = directory / identifier
    if path.is_symlink():
        raise ValueError("Unsafe original file")
    return path


def preview_path(store, source):
    from .images import IMAGE_FORMATS
    if not source or source["format"] not in IMAGE_FORMATS:
        raise KeyError("No image preview")
    path = file_path(store, source["file_id"], preview=True)
    if not path.is_file():
        raise KeyError("Image preview is not ready")
    return path


def has_preview(store, source):
    try:
        preview_path(store, source)
        return True
    except (KeyError, OSError, ValueError):
        return False


def save_preview(store, source, encoded):
    from .images import MAX_PREVIEW
    data = base64.b64decode(encoded, validate=True)
    if len(data) > MAX_PREVIEW or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid image preview")
    path = file_path(store, source["file_id"], preview=True)
    existing_bytes = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file() and p != path)
    if existing_bytes + len(data) > 512 * 1024 * 1024:
        raise ParseError("Image-preview quota reached (512 MiB); contact the maintainer")
    temporary = path.with_name(secrets.token_hex(16))
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as dest:
            dest.write(data)
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def upload(store, data, filename, title, source, request_id, actor):
    fmt = filename_format(filename)
    check_signature(data, fmt)
    if (not 1 <= len(title.strip()) <= 200 or len(source) > 1000
            or not 1 <= len(request_id) <= 100 or "\0" in title + source):
        raise ParseError("Invalid title, source or upload request ID")
    sha = hashlib.sha256(data).hexdigest()
    fingerprint = digest([filename, title, source, sha])
    key = digest([actor, "upload", request_id])
    db = store.writer()
    jobs = JobStore(store.root, database=db)
    path = None
    try:
        with db.transaction() as conn:
            db.lock(conn, "knowledge-write")
            previous = store.rows(conn, requests, requests.c.id == key)
            if previous:
                if previous[0]["fingerprint"] != fingerprint:
                    raise Conflict("Upload request ID reused with different content")
                return store.summary(conn, store.row(conn, documents, previous[0]["document_id"]))
            active = conn.execute(select(func.count()).select_from(versions).join(job_table,
                (job_table.c.workspace_id == versions.c.workspace_id) & (job_table.c.id == versions.c.job_id)).where(
                versions.c.workspace_id == store.scope, versions.c.state == "parsing",
                job_table.c.status.in_(["queued", "running", "retrying"]))).scalar_one()
            if active >= 8:
                raise Conflict("Eight files are already parsing; wait before uploading another")
            identifier, fid = secrets.token_hex(16), secrets.token_hex(16)
            path = file_path(store, fid)
            originals = list(path.parent.iterdir())
            if len(originals) >= 2000 or sum(p.stat().st_size for p in originals if p.is_file()) + len(data) > 512 * 1024 * 1024:
                raise ParseError("Original-file quota reached (512 MiB / 2000 files); contact the maintainer")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as dest:
                dest.write(data)
                dest.flush()
                os.fsync(dest.fileno())
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            now = timestamp()
            db.insert(conn, documents, dict(id=identifier, latest_version=1, published_version=None,
                created_at=now, updated_at=now))
            job = jobs.submit("index", dict(domain="knowledge-parse", document_id=identifier,
                version=1, file_id=fid, sha256=sha, parser_version=parser_version(fmt), catalog_revision=0), "parse:" + key, connection=conn)
            db.insert(conn, versions, dict(document_id=identifier, version=1, title=title, source=source,
                content="", source_hash=source_hash(title, source, "", []), state="parsing",
                definition_json="{}", fingerprint=None, job_id=job["id"], chunk_count=0,
                error=None, created_at=now))
            db.insert(conn, sources, dict(document_id=identifier, version=1, file_id=fid,
                filename=filename, format=fmt, sha256=sha, parser_version=parser_version(fmt),
                reviewed=0, sections_json="[]", warnings_json="[]"))
            db.insert(conn, requests, dict(id=key, fingerprint=fingerprint, document_id=identifier, version=1))
            store.log(db, conn, actor, "file-uploaded", identifier, 1)
            result = store.summary(conn, store.row(conn, documents, identifier))
        return result
    except BaseException:
        # A lost connection at COMMIT is ambiguous. Keep the immutable blob rather
        # than delete the source of a possibly committed task; quotas bound orphans.
        raise


def execute_parse(jobs, job, owner):
    from .store import KnowledgeStore
    store, db = KnowledgeStore(jobs.db.root), jobs.db
    with db.transaction() as conn:
        current = jobs.require_owner(conn, job["id"], owner)
        payload = json.loads(current["payload_json"])
        if current["kind"] != "index" or payload.get("domain") != "knowledge-parse":
            raise ValueError("Not a document parse task")
        require_active(store, conn, payload["document_id"], payload.get("catalog_revision", 0))
        v = store.row(conn, versions, payload["document_id"], payload["version"])
        source = source_row(store, conn, payload["document_id"], payload["version"])
        if (not source or v["state"] != "parsing" or v["job_id"] != job["id"]
                or source["file_id"] != payload["file_id"] or source["sha256"] != payload["sha256"]
                or source["parser_version"] != parser_version(source["format"])
                or payload["parser_version"] != source["parser_version"]):
            raise ValueError("Parse task does not match stored original")
    jobs.begin_effects(job["id"], owner)
    result = isolated_parse(file_path(store, source["file_id"]), source["format"], source["sha256"])
    content = "\n\n".join(part["text"] for part in result["sections"])
    with db.transaction() as conn:
        db.lock(conn, "knowledge-write")
        jobs.require_owner(conn, job["id"], owner)
        require_active(store, conn, payload["document_id"], payload.get("catalog_revision", 0))
        latest = store.row(conn, versions, v["document_id"], v["version"])
        if latest["job_id"] != job["id"] or latest["state"] != "parsing":
            raise Conflict("Parse task was superseded")
        if result.get("preview"):
            try:
                save_preview(store, source, result.pop("preview"))
            except (OSError, ValueError) as error:
                result.update(state="parse_failed", sections=[], error=str(error) if isinstance(error, ParseError)
                    else "Unable to save image preview; contact the maintainer and upload again")
                content = ""
        db.update(conn, sources, dict(sections_json=encode(result["sections"]),
            warnings_json=encode(result["warnings"])), sources.c.document_id == v["document_id"],
            sources.c.version == v["version"])
        error = result.get("error")
        if result["state"] == "ocr_required":
            error = "No extractable PDF text. Scanned PDF OCR is not enabled; upload individual PNG/JPEG pages for OCR."
        db.update(conn, versions, dict(content=content, source_hash=source_hash(v["title"], v["source"], content,
            result["sections"]), state=result["state"], error=error),
            versions.c.document_id == v["document_id"], versions.c.version == v["version"])
        store.log(db, conn, "parse-worker", result["state"], v["document_id"], v["version"])
        jobs.finish(job["id"], owner, "succeeded" if result["state"] == "review" else "failed",
                    dict(parse_state=result["state"], sections=len(result["sections"]), error=error), connection=conn)
