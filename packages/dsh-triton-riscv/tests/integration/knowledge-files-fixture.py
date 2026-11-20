"""Document ingestion probes. Vitest owns assertions; all accounts/data are disposable."""
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
import zipfile
from unittest.mock import patch

from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, NumberObject
from fastapi.testclient import TestClient

from codex_agent.accounts.api import create_gateway
from codex_agent.accounts.config import AccountConfig
from codex_agent.accounts.store import Accounts
from codex_agent.knowledge.files import execute_parse, file_path
from codex_agent.knowledge.parsing import parse_bytes, ParseError, isolated_parse, MAX_SECTION
from codex_agent.knowledge.processes import KnowledgeWorkers
from codex_agent.knowledge.store import KnowledgeStore
from codex_agent.knowledge.vectors import execute_knowledge_index, search, augment_chat, ROOT_ENV
from codex_agent.knowledge.file_schema import sources
from codex_agent.tasks.store import JobStore
from codex_agent.workers.handlers import execute_job


def pdf_bytes(texts=("Atlas retrieval fixture", "Atlas access code is PDF-428."), password=None):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
                DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            content = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content.set_data(f"BT /F1 12 Tf 72 700 Td ({escaped}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(content)
        else:
            image = DecodedStreamObject()
            image.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"), NameObject("/BitsPerComponent"): NumberObject(8)})
            image.set_data(b"\x00\x00\x00")
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"):
                DictionaryObject({NameObject("/Im1"): writer._add_object(image)})})
            content = DecodedStreamObject()
            content.set_data(b"q 200 0 0 200 72 500 cm /Im1 Do Q")
            page[NameObject("/Contents")] = writer._add_object(content)
    if password:
        writer.encrypt(password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def docx_bytes(long=False):
    doc = Document()
    doc.add_heading("Atlas document retrieval", level=1)
    doc.add_paragraph("Atlas restoration code is DOCX-529.")
    doc.add_paragraph("中文资料：恢复前应检查备份完整性。")
    table = doc.add_table(rows=2, cols=2)
    for row, values in zip(table.rows, [("Service", "Retention"), ("Atlas", "37 days")]):
        for cell, text in zip(row.cells, values):
            cell.text = text
    if long:
        doc.add_paragraph("无空格测试" * 15000)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def archive_with(data, name, content):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as old, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as new:
        for entry in old.infolist():
            if entry.filename != name:
                new.writestr(entry.filename, old.read(entry))
        new.writestr(name, content)
    return out.getvalue()


class NoChildren:
    def start(self): pass
    def close(self): pass
    async def monitor(self):
        while True:
            await asyncio.sleep(60)


class Provider:
    max_input_tokens = 180
    def count_tokens(self, text): return len(text)
    def embed(self, texts): return [[1., 0., 0., 0.] for _ in texts]


def run():
    result = {}
    pdf, docx = pdf_bytes(), docx_bytes()
    parsed = parse_bytes(pdf, "pdf")
    result["pdf"] = dict(pages=len(parsed["sections"]), second=parsed["sections"][1])
    parsed = parse_bytes(docx, "docx")
    result["docx"] = dict(sections=parsed["sections"], warnings=parsed["warnings"])
    long = parse_bytes(docx_bytes(long=True), "docx")
    result["long"] = dict(chars=sum(len(s["text"]) for s in long["sections"]),
        largest=max(len(s["text"]) for s in long["sections"]),
        distinct_ids=len({s["id"] for s in long["sections"]}) == len(long["sections"]))
    result["scan"] = parse_bytes(pdf_bytes(("",)), "pdf")["state"]
    result["mixed"] = parse_bytes(pdf_bytes(("visible text", "")), "pdf")
    invalids = {"encrypted": (pdf_bytes(password="fixture"), "pdf"),
        "signature": (b"not a PDF", "pdf"), "corrupt": (b"%PDF-1.7 broken", "pdf"),
        "macro": (archive_with(docx, "word/vbaProject.bin", b"ignored"), "docx"),
        "zip_bomb": (archive_with(docx, "word/large.xml", b"x" * 2_000_000), "docx"),
        "traversal": (archive_with(docx, "../outside.xml", b"<x/>"), "docx"),
        "entity": (archive_with(docx, "word/document.xml", b'<!DOCTYPE x [<!ENTITY a SYSTEM "file:///etc/passwd">]><x>&a;</x>'), "docx"),
        "linked_media": (archive_with(docx, "word/_rels/document.xml.rels",
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="external" Type="image" Target="http://127.0.0.1:1/private" TargetMode="External"/></Relationships>'), "docx")}
    for name, (data, fmt) in invalids.items():
        try:
            parse_bytes(data, fmt)
            result[name] = "accepted"
        except Exception as error:
            result[name] = type(error).__name__
    result["timeout"] = None
    with patch("codex_agent.knowledge.parsing._parse_process", side_effect=__import__("subprocess").TimeoutExpired("parser", 45)):
        result["timeout"] = isolated_parse(Path("/tmp/file"), "pdf", "hash")["state"]
    with tempfile.TemporaryDirectory(prefix="knowledge-files-") as temporary:
        base = Path(temporary)
        template = base / "template"
        template.mkdir()
        wrong_hash = base / "wrong.pdf"
        wrong_hash.write_bytes(pdf)
        result["hash_mismatch"] = isolated_parse(wrong_hash, "pdf", "0" * 64)
        cfg = dict(schemaVersion=1, repoRoot=str(template), stateDir=str(base / "state"), indexing={"enabled": True,
            "modelRevision": "file-fixture", "dimensions": 4,
            "embedding": {"provider": "ollama", "model": "fixture", "baseUrl": "http://127.0.0.1:1", "tokenBudget": 180},
            "qdrantUrl": os.environ["TRITON_TEST_QDRANT_URL"]})
        os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
        config = AccountConfig(origin="http://127.0.0.1:8766", data_dir=base / "accounts",
            template=template, allow_local_http=True, knowledge_enabled=True)
        accounts = Accounts(config)
        password = secrets.token_urlsafe(20)
        admin = accounts.register("fa_" + secrets.token_hex(4), password, accounts.invite())
        member = accounts.register("fu_" + secrets.token_hex(4), password, accounts.invite())
        config = config.model_copy(update={"admin_user_ids": (admin["id"],)})
        gateway = create_gateway(config, accounts=accounts, workbenches=NoChildren())
        root = config.data_dir / "knowledge"
        root.mkdir(parents=True)
        store, jobs = KnowledgeStore(root), JobStore(root)
        with TestClient(gateway, base_url=config.origin) as client:
            token, identity = accounts.login(admin["username"], password)
            headers = {"Origin": config.origin, "X-CSRF-Token": identity["csrf_token"],
                "Cookie": config.cookie_name + "=" + token}
            binary = {**headers, "Content-Type": "application/octet-stream"}
            json_headers = {**headers, "Content-Type": "application/json"}
            user_token, user = accounts.login(member["username"], password)
            user_headers = {"Origin": config.origin, "Cookie": config.cookie_name + "=" + user_token,
                "X-CSRF-Token": user["csrf_token"], "Content-Type": "application/octet-stream"}
            path = "/api/admin/knowledge/files?filename=sample.pdf&request_id=first"
            result["anonymous"] = client.post(path, headers={"Origin": config.origin, "Content-Type": "application/octet-stream"}, content=pdf).status_code
            result["ordinary"] = client.post(path, headers=user_headers, content=pdf).status_code
            result["csrf"] = client.post(path, headers={k: v for k, v in binary.items() if k != "X-CSRF-Token"}, content=pdf).status_code
            result["origin"] = client.post(path, headers={**binary, "Origin": "http://bad.invalid"}, content=pdf).status_code
            result["oversize"] = client.post(path, headers={**binary, "Content-Length": str(13 * 1024 * 1024)}, content=b"x").status_code
            result["bad_type"] = client.post(path, headers=json_headers, content=pdf).status_code
            result["other_route_binary"] = client.post("/api/admin/knowledge", headers=binary, content=pdf).status_code
            result["extension"] = client.post(path.replace("sample.pdf", "sample.doc"), headers=binary, content=pdf).status_code
            result["forged"] = client.post(path, headers=binary, content=docx).status_code
            first = client.post(path, headers=binary, content=pdf)
            result["upload"] = first.status_code
            doc = first.json()
            identifier = doc["id"]
            result["initial_state"] = doc["state"]
            result["duplicate"] = client.post(path, headers=binary, content=pdf).json()["id"] == identifier
            result["conflict"] = client.post(path, headers=binary, content=pdf_bytes(("changed",))).status_code
            event = jobs.claim_publication()
            job = jobs.claim(jobs.envelope(event), "index")
            jobs.publication_result(event)
            execute_job(root, job["id"], job["owner"])
            detail = client.get(f"/api/admin/knowledge/{identifier}", headers=headers).json()
            result["parsed_state"] = detail["state"]
            result["file_source"] = detail["source_file"]
            result["no_chunks_before_review"] = detail["chunks"] == []
            result["early_index"] = client.post(f"/api/admin/knowledge/{identifier}/index", headers=json_headers,
                json={"version": 1, "request_id": "early"}).status_code
            original = client.get(f"/api/admin/knowledge/{identifier}/original", headers=headers)
            result["original"] = dict(equal=original.content == pdf, attachment="attachment" in original.headers.get("content-disposition", ""))
            result["ordinary_original"] = client.get(f"/api/admin/knowledge/{identifier}/original", headers=user_headers).status_code
            section_edits = [{"id": s["id"], "text": s["text"]} for s in detail["sections"]]
            section_edits[1]["text"] += " Reviewed by administrator."
            review = dict(title=detail["title"], source="fixture source", sections=section_edits,
                document_id=identifier, expected_version=1, request_id="review")
            result["forged_sections"] = client.post("/api/admin/knowledge", headers=json_headers,
                json={**review, "sections": section_edits[::-1]}).status_code
            saved = client.post("/api/admin/knowledge", headers=json_headers, json=review)
            result["review"] = saved.json()
            result["review_replay"] = client.post("/api/admin/knowledge", headers=json_headers, json=review).json()["latest_version"]
            result["stale_review"] = client.post("/api/admin/knowledge", headers=json_headers,
                json={**review, "request_id": "stale"}).status_code
            client.post(f"/api/admin/knowledge/{identifier}/index", headers=json_headers,
                json={"version": 2, "request_id": "index"})
            event = jobs.claim_publication()
            job = jobs.claim(jobs.envelope(event), "index")
            jobs.publication_result(event)
            execute_knowledge_index(jobs, job, job["owner"], provider_factory=lambda d: Provider())
            result["indexed"] = store.detail(identifier)
            result["published"] = client.post(f"/api/admin/knowledge/{identifier}/publish", headers=json_headers,
                json={"version": 2}).json()["published_version"]
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                os.environ[ROOT_ENV] = str(root)
                task, evidence = augment_chat("Original user question", "Atlas access code")
                result["context"] = task
                result["hit_count"] = len(evidence["items"])
                os.environ.pop(ROOT_ENV)
            client.post(f"/api/admin/knowledge/{identifier}/unpublish", headers=json_headers, json={})
            with patch("codex_agent.knowledge.vectors.cached_provider", return_value=Provider()):
                result["unpublished"] = search(root, "Atlas")["status"]
            if os.environ.get("TRITON_TEST_EMBEDDING_URL"):
                # Real background parsing works even before any embedding configuration is enabled.
                cfg["indexing"]["enabled"] = False
                cfg["queue"] = {"urlEnv": "TRITON_TEST_AMQP_URL"}
                cfg["repoRoot"] = str(root)
                os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
                workers = KnowledgeWorkers(config)
                workers.start()
                try:
                    uploaded = client.post("/api/admin/knowledge/files?filename=table.docx&request_id=docx", headers=binary, content=docx).json()
                    for _ in range(200):
                        document = store.detail(uploaded["id"])
                        if document["state"] != "parsing": break
                        time.sleep(.15)
                    result["async_docx"] = document["state"]
                    reviewed = client.post("/api/admin/knowledge", headers=json_headers, json=dict(title="Atlas restoration",
                        source="DOCX integration fixture", request_id="docx-review", document_id=uploaded["id"], expected_version=1,
                        sections=[{"id": s["id"], "text": s["text"]} for s in document["sections"]])).json()
                finally:
                    workers.close()
                cfg["indexing"].update(enabled=True, dimensions=384, modelRevision="local-all-minilm-v2",
                    embedding={"provider": "ollama", "model": "all-minilm:v2", "tokenBudget": 180,
                        "baseUrl": os.environ["TRITON_TEST_EMBEDDING_URL"]})
                os.environ["TRITON_RISCV_CONFIG"] = json.dumps(cfg)
                workers.start()
                try:
                    client.post(f"/api/admin/knowledge/{uploaded['id']}/index", headers=json_headers,
                        json={"version": reviewed["latest_version"], "request_id": "real"})
                    for _ in range(400):
                        document = store.detail(uploaded["id"])
                        if document["state"] != "pending": break
                        time.sleep(.15)
                    result["async_index"] = document["state"]
                    client.post(f"/api/admin/knowledge/{uploaded['id']}/publish", headers=json_headers,
                        json={"version": reviewed["latest_version"]})
                    found = search(root, "Atlas restoration code", limit=5)
                    result["real_retrieval"] = found
                finally:
                    workers.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fixtures-only":
        target = Path(sys.argv[2])
        target.mkdir(parents=True, exist_ok=True)
        (target / "atlas.pdf").write_bytes(pdf_bytes())
        (target / "atlas.docx").write_bytes(docx_bytes())
        (target / "scan.pdf").write_bytes(pdf_bytes(("",)))
        print(json.dumps({"directory": str(target)}))
    else:
        print(json.dumps(run(), ensure_ascii=False))
