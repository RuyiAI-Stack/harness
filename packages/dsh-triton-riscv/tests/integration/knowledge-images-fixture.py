"""Image probes for Vitest. Temporary accounts and fixtures, no remote model calls."""
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from fastapi.testclient import TestClient
from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.store import Accounts
from codex_agent.knowledge import images
from codex_agent.knowledge.files import execute_parse, file_path
from codex_agent.knowledge.parsing import parse_bytes, isolated_parse, filename_format
from codex_agent.knowledge.processes import KnowledgeWorkers
from codex_agent.knowledge.store import KnowledgeStore
from codex_agent.knowledge.vectors import search, augment_chat, ROOT_ENV
from codex_agent.tasks.store import JobStore


TSV = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n" + (
    "5\t1\t1\t1\t1\t1\t40\t40\t300\t40\t95\tAtlas restore code OCR-731\n"
    "5\t1\t1\t1\t2\t1\t40\t100\t300\t40\t91\tKeep 37 days of backups\n"
    "5\t1\t2\t1\t1\t1\t40\t190\t300\t40\t45\t中文资料：恢复前检查备份\n")


def picture(fmt="PNG", *, blank=False, chinese=False):
    image = Image.new("RGB", (1200, 420), "white")
    if not blank:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=36)
        for y, text in ((40, "Atlas restore code OCR-731"), (100, "Keep 37 days of backups")):
            draw.text((40, y), text, fill="black", font=font)
        if chinese:
            font = ImageFont.truetype(os.environ["TRITON_TEST_CJK_FONT"], 40)
            draw.text((40, 190), "中文资料：恢复前检查备份", fill="black", font=font)
    stream = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "private metadata must not appear in preview")
    image.save(stream, format=fmt, **({"pnginfo": metadata} if fmt == "PNG" else {"quality": 95}))
    return stream.getvalue()


class NoChildren:
    def start(self): pass
    def close(self): pass
    async def monitor(self):
        while True: await asyncio.sleep(60)


def run():
    r = {}
    data = picture()
    with patch.object(images, "recognize", return_value=TSV):
        r["formats"] = {}
        for fmt in ("png", "jpg", "jpeg"):
            parsed = parse_bytes(data if fmt == "png" else picture("JPEG"), fmt)
            r["formats"][fmt] = {k: v for k, v in parsed.items() if k != "preview"}
        preview = Image.open(io.BytesIO(base64.b64decode(parsed["preview"])))
        r["preview_clean"] = preview.format == "PNG" and not preview.info
        r["uppercase"] = filename_format("SCAN.JPEG")
        rotated = Image.new("RGB", (420, 1200), "white")
        exif = rotated.getexif(); exif[274] = 6
        stream = io.BytesIO(); rotated.save(stream, "JPEG", exif=exif)
        r["orientation"] = images.decode_image(stream.getvalue(), "jpg").size
    for name, value, fmt in (("signature", b"not png", "png"), ("truncated", data[:50], "png"),
            ("disguised", picture("JPEG"), "png")):
        try:
            parse_bytes(value, fmt); r[name] = "accepted"
        except ValueError: r[name] = "rejected"
    with patch.object(images, "MAX_PIXELS", 20):
        try:
            images.decode_image(data, "png"); r["pixels"] = "accepted"
        except ValueError: r["pixels"] = "rejected"
    stream = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(stream, "PNG", save_all=True,
        append_images=[Image.new("RGB", (10, 10), "black")], duration=100, loop=0)
    try: images.decode_image(stream.getvalue(), "png"); r["animation"] = "accepted"
    except ValueError: r["animation"] = "rejected"
    with patch.object(images, "recognize", return_value=TSV.splitlines()[0]):
        r["empty"] = parse_bytes(data, "png")["state"]
    with patch.object(images, "recognize", side_effect=subprocess.TimeoutExpired("ocr", 25)):
        timeout = parse_bytes(data, "png")
        r["timeout"] = {k: timeout[k] for k in ("state", "error")}
        r["timeout_preview"] = bool(timeout.get("preview"))
    with patch.object(images.shutil, "which", return_value=None):
        r["missing_engine"] = parse_bytes(data, "png")["error"]
    with patch.object(images.shutil, "which", return_value="/test/tesseract"), patch.object(
            images.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"eng\n")):
        r["missing_chinese"] = parse_bytes(data, "png")["error"]
    real = os.environ.get("TRITON_TEST_OCR") == "1"
    if real:
        for fmt in ("png", "jpg", "jpeg"):
            with tempfile.TemporaryDirectory() as temporary:
                raw = picture("PNG" if fmt == "png" else "JPEG", chinese=bool(os.environ.get("TRITON_TEST_CJK_FONT")))
                path = Path(temporary) / ("input." + fmt); path.write_bytes(raw)
                result = isolated_parse(path, fmt, hashlib.sha256(raw).hexdigest())
                r["real_" + fmt] = {k: v for k, v in result.items() if k != "preview"}
        r["real_blank"] = parse_bytes(picture(blank=True), "png")["state"]

    with tempfile.TemporaryDirectory(prefix="knowledge-images-") as temporary:
        base = Path(temporary); template = base / "template"; template.mkdir()
        cfg = dict(schemaVersion=1, repoRoot=str(template), stateDir=str(base / "state"),
            queue={"urlEnv": "TRITON_TEST_AMQP_URL"}, indexing=dict(enabled=True,
                modelRevision="local-all-minilm-v2", dimensions=384,
                embedding=dict(provider="ollama", model="all-minilm:v2", tokenBudget=180,
                    baseUrl=os.environ.get("TRITON_TEST_EMBEDDING_URL", "http://127.0.0.1:1")),
                qdrantUrl=os.environ["TRITON_TEST_QDRANT_URL"]))
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
        config = AccountConfig(origin="http://127.0.0.1:8766", data_dir=base / "accounts",
            template=template, allow_local_http=True, knowledge_enabled=True, rag_catalog_enabled=True)
        accounts = Accounts(config); password = secrets.token_urlsafe(20)
        admin = accounts.register("ia_" + secrets.token_hex(4), password, accounts.invite())
        member = accounts.register("iu_" + secrets.token_hex(4), password, accounts.invite())
        config = config.model_copy(update={"admin_user_ids": (admin["id"],)})
        gateway = create_gateway(config, accounts=accounts, workbenches=NoChildren())
        root = config.data_dir / "knowledge"; root.mkdir(parents=True)
        store, jobs = KnowledgeStore(root), JobStore(root)
        with TestClient(gateway, base_url=config.origin) as client:
            def headers(user):
                token, identity = accounts.login(user["username"], password)
                return {"Origin": config.origin, "X-CSRF-Token": identity["csrf_token"],
                    "Cookie": config.cookie_name + "=" + token}
            a, u = headers(admin), headers(member)
            binary = {**a, "Content-Type": "application/octet-stream"}
            body = {**a, "Content-Type": "application/json"}
            path = "/api/admin/knowledge/files?filename=receipt.png&request_id=image-first"
            r["anonymous"] = client.post(path, headers={"Origin": config.origin, "Content-Type": "application/octet-stream"}, content=data).status_code
            r["member"] = client.post(path, headers={**u, "Content-Type": "application/octet-stream"}, content=data).status_code
            r["csrf"] = client.post(path, headers={k: v for k, v in binary.items() if k != "X-CSRF-Token"}, content=data).status_code
            r["oversize"] = client.post(path, headers={**binary, "Content-Length": str(13 * 1024 * 1024)}, content=b"x").status_code
            first = client.post(path, headers=binary, content=data); identifier = first.json()["id"]
            r["upload"] = first.status_code
            r["duplicate"] = client.post(path, headers=binary, content=data).json()["id"] == identifier
            r["replay_conflict"] = client.post(path, headers=binary, content=picture(blank=True)).status_code
            parsed = images.parse_image(data, "png") if real else None
            if not real:
                with patch.object(images, "recognize", return_value=TSV): parsed = parse_bytes(data, "png")
            event = jobs.claim_publication(); job = jobs.claim(jobs.envelope(event), "index"); jobs.publication_result(event)
            with patch("codex_agent.knowledge.files.isolated_parse", return_value=parsed):
                execute_parse(jobs, job, job["owner"])
            detail = store.detail(identifier)
            r["before_review"] = dict(state=detail["state"], chunks=detail["chunks"], published=detail["published_version"])
            r["early_index"] = client.post(f"/api/admin/knowledge/{identifier}/index", headers=body,
                json={"version": 1, "request_id": "early"}).status_code
            r["early_publish"] = client.post(f"/api/admin/knowledge/{identifier}/publish", headers=body,
                json={"version": 1}).status_code
            purl = f"/api/admin/knowledge/{identifier}/preview?version=1"
            response = client.get(purl, headers=a)
            r["preview"] = dict(status=response.status_code, mime=response.headers.get("content-type"),
                cache=response.headers.get("cache-control"), clean=not Image.open(io.BytesIO(response.content)).info)
            r["member_preview"] = client.get(purl, headers=u).status_code
            r["wrong_version"] = client.get(purl.replace("version=1", "version=999"), headers=a).status_code
            r["original_exact"] = client.get(f"/api/admin/knowledge/{identifier}/original", headers=a).content == data
            catalog = client.get("/api/admin/rag/catalog", headers=a).json()
            item = next(x for x in catalog["items"] if x["source_id"] == identifier)
            cpath = f"/api/admin/rag/catalog/{item['item_id']}/preview?version=1"
            r["catalog_preview"] = client.get(cpath, headers=a).status_code
            r["catalog_member"] = client.get(cpath, headers=u).status_code
            r["catalog_wrong_version"] = client.get(cpath.replace("version=1", "version=999"), headers=a).status_code
            review = dict(title="Atlas recovery screenshot", source="Synthetic image fixture, not business evidence",
                document_id=identifier, expected_version=1, request_id="image-review",
                sections=[dict(id=s["id"], text=s["text"]) for s in detail["sections"]])
            review["sections"][0]["text"] += "\nReviewed recovery code: REVIEW-842."
            r["forged_sections"] = client.post("/api/admin/knowledge", headers=body,
                json={**review, "sections": [{"id": "s999", "text": "forged"}]}).status_code
            saved = client.post("/api/admin/knowledge", headers=body, json=review)
            r["saved"] = saved.status_code
            r["edited"] = store.detail(identifier)["sections"][0]
            r["version_retained"] = client.get(purl, headers=a).status_code
            r["stale_edit"] = client.post("/api/admin/knowledge", headers=body,
                json={**review, "request_id": "stale"}).status_code
            failed = client.post(path.replace("image-first", "preview-failure"), headers=binary, content=data).json()
            event = jobs.claim_publication(); job = jobs.claim(jobs.envelope(event), "index"); jobs.publication_result(event)
            with patch.object(images, "recognize", return_value=TSV): failed_parse = parse_bytes(data, "png")
            with patch("codex_agent.knowledge.files.isolated_parse", return_value=failed_parse), patch(
                    "codex_agent.knowledge.files.save_preview", side_effect=OSError("disk unavailable")):
                execute_parse(jobs, job, job["owner"])
            failure = store.detail(failed["id"])
            r["preview_failure"] = dict(state=failure["state"], error=failure["error"],
                published=failure["published_version"], chunks=failure["chunks"])
            if real and os.environ.get("TRITON_TEST_EMBEDDING_URL"):
                workers = KnowledgeWorkers(config); workers.start()
                try:
                    # This upload is parsed by the real RabbitMQ worker, without mocks.
                    live = client.post(path.replace("image-first", "async-image"), headers=binary, content=data).json()
                    def wait(identifier, predicate):
                        for _ in range(500):
                            current = store.detail(identifier, admin["id"])
                            if predicate(current): return current
                            time.sleep(.15)
                        raise RuntimeError("Timed out waiting for image lifecycle")
                    live = wait(live["id"], lambda d: d["state"] != "parsing")
                    r["async_parse"] = live["state"]
                    confirmation = dict(title="Atlas live OCR fixture", source="Synthetic acceptance fixture",
                        document_id=live["id"], expected_version=1, request_id="confirm-image",
                        sections=[dict(id=s["id"], text=s["text"]) for s in live["sections"]])
                    confirmed = client.post("/api/admin/knowledge/confirm", headers=body, json=confirmation)
                    r["confirmed"] = confirmed.status_code
                    ready = wait(live["id"], lambda d: d.get("sync", {}).get("state") in {"published", "failed"})
                    r["async_publish"] = dict(state=ready["state"], published=ready["published_version"],
                        sync=ready.get("sync"), chunks=ready["chunks"])
                    os.environ[ROOT_ENV] = str(root)
                    try:
                        task, evidence = augment_chat("What is the Atlas restore code?", "Atlas restore code")
                        r["chat_context"] = task
                        r["retrieval"] = evidence
                    finally: os.environ.pop(ROOT_ENV, None)
                    trash = client.post(f"/api/admin/knowledge/{live['id']}/delete", headers=body,
                        json={"version": 2, "catalog_revision": 0})
                    r["delete"] = trash.status_code
                    r["after_delete"] = search(root, "Atlas restore code")
                    restored = client.post(f"/api/admin/knowledge/{live['id']}/restore", headers=body,
                        json={"version": 2, "catalog_revision": 1}).json()
                    r["restored_unpublished"] = restored.get("published_version") is None
                finally: workers.close()
    return r


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--fixtures-only":
        target = Path(sys.argv[2]); target.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "jpg", "jpeg"):
            (target / ("atlas." + suffix)).write_bytes(picture("PNG" if suffix == "png" else "JPEG", chinese=True))
        (target / "blank.png").write_bytes(picture(blank=True))
        print(json.dumps({"directory": str(target)}))
    else:
        result = run()
        if os.environ.get("TRITON_IMAGE_TEST_RESULT"):
            Path(os.environ["TRITON_IMAGE_TEST_RESULT"]).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False))
