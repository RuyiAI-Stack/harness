"""Read-only, account-scoped access to operator source and validation evidence."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

MAX_FILE_BYTES = 8 * 1024 * 1024
ALLOWED_FILES = (
    ("python/examples/flaggems/", {".py"}),
    ("tasks/operators/", {".md"}),
    ("agent-results/operator-development/requests/", {".json"}),
    ("agent-results/operator-development/proposals/", {".json"}),
    ("agent-results/operator-development/patches/", {".diff"}),
    ("agent-results/operator-lifecycle/receipts/", {".json"}),
    ("agent-results/operator-lifecycle/proposals/", {".json"}),
    ("agent-results/operator-lifecycle/patches/", {".diff"}),
    ("agent-results/operator-lifecycle/validation/logs/", {".log", ".txt"}),
)


def read_workspace_file(root: Path, value: str) -> tuple[str, bytes]:
    root_spelling = root.absolute()
    root = root.resolve()
    # Validate the original spelling before Path normalizes dot components.
    if not value or "\\" in value or any(ord(char) < 32 for char in value):
        raise HTTPException(404, "File unavailable in this workspace")
    if ".." in value.split("/"):
        raise HTTPException(404, "File unavailable in this workspace")
    path = Path(value)
    if path.is_absolute():
        for base in (root, root_spelling):
            if path.is_relative_to(base):
                path = path.relative_to(base)
                break
        else:
            raise HTTPException(404, "File unavailable in this workspace") from None
    relative = path.as_posix()
    if (any(part.startswith(".") for part in path.parts)
        or not any(relative.startswith(prefix) and path.suffix in suffixes
                   for prefix, suffixes in ALLOWED_FILES)):
        raise HTTPException(404, "File unavailable in this workspace")

    # Open each component relative to an already-open directory. Reject symlinks
    # throughout, including replacements made between validation and reading.
    directory = None
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in path.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise HTTPException(404, "File unavailable in this workspace")
            if info.st_size > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds the 8 MiB preview limit")
            content = file.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds the 8 MiB preview limit")
            content.decode("utf-8")
            if b"\x00" in content:
                raise ValueError("Binary content")
        return path.name, content
    except (OSError, ValueError):
        raise HTTPException(404, "File unavailable in this workspace") from None
    finally:
        if directory is not None:
            os.close(directory)


def router(root: Path) -> APIRouter:
    routes = APIRouter()

    @routes.get("/api/workspace/files")
    def workspace_file(path: str = Query(min_length=1, max_length=4096)):
        name, content = read_workspace_file(root, path)
        return Response(content, media_type="text/plain; charset=utf-8", headers={
            "Content-Disposition": "inline; filename*=UTF-8''" + quote(name, safe=""),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox; frame-ancestors 'none'",
        })

    return routes
