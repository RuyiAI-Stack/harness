"""Verify scoped vector counts, never infer live Qdrant state from SQL alone."""
from datetime import datetime, timezone
import json
import os
import re
import time

import httpx
from sqlalchemy import and_, select

from codex_agent.knowledge.schema import versions
from codex_agent.runtime_config import runtime_config
from codex_agent.storage.schema import memory_indexes, memory_index_entries
from codex_agent.tasks.store import digest
from .sources import source_config


def inspect_vectors(config, source_id, catalog, item_id, version):
    detail = catalog.detail(item_id, version=version)
    scope, kind, identifier = catalog.resolve(item_id)
    selected = detail["selected_version"]
    cfg = runtime_config().indexing
    target = cfg.qdrantUrl if source_id == "current" else source_config(config, source_id).vector_url
    records = []
    with catalog.engine.connect() as conn:
        if kind == "knowledge_document":
            row = conn.execute(select(versions).where(versions.c.workspace_id == scope["id"],
                versions.c.document_id == identifier, versions.c.version == int(selected))).mappings().one()
            definition = json.loads(row["definition_json"])
            if definition and row["fingerprint"]:
                records.append(dict(definition=definition, fingerprint=row["fingerprint"],
                    collection="knowledge_" + digest([scope["id"], row["fingerprint"]]),
                    generation=definition.get("generation"), expected=row["chunk_count"],
                    fields={"document_id": identifier, "version": int(selected), "source_version": row["source_hash"]}))
        else:
            i, e = memory_indexes.c, memory_index_entries.c
            rows = conn.execute(select(memory_indexes.c.config_json, i.fingerprint, i.collection,
                i.generation, e.revision, e.point_count).select_from(memory_indexes.join(memory_index_entries,
                and_(i.workspace_id == e.workspace_id, i.generation == e.generation))).where(
                e.workspace_id == scope["id"], e.memory_id == int(identifier), e.case_version == selected)
                .order_by(i.generation).limit(9)).mappings()
            for row in rows:
                records.append(dict(definition=json.loads(row["config_json"]), fingerprint=row["fingerprint"],
                    collection=row["collection"], generation=row["generation"], expected=row["point_count"],
                    fields={"memory_id": int(identifier), "revision": row["revision"], "case_version": selected}))
    result = dict(version=selected, checked_at=datetime.now(timezone.utc).isoformat(),
                  complete=len(records) <= 8, checks=[], vector_scan_performed=False)
    deadline = time.monotonic() + 6
    for record in records[:8]:
        if time.monotonic() >= deadline:
            result["complete"] = False
            break
        definition = record["definition"]
        check = dict(generation=record["generation"], expected_count=record["expected"], actual_count=None,
                     dimensions=definition.get("dimensions"), state="not_checked")
        if (digest(definition) != record["fingerprint"]
                or not re.fullmatch(r"[A-Za-z0-9_-]+", record["collection"])):
            check["reason"] = "索引定义不完整，不能据此访问向量服务"
        elif not target or str(definition.get("qdrantUrl", "")).rstrip("/") != target.rstrip("/"):
            check["reason"] = "该历史索引的向量服务未配置或地址已变化，未确认实际向量"
        else:
            fields = dict(workspace_id=scope["id"], generation=record["generation"],
                          index_fingerprint=record["fingerprint"], **record["fields"])
            headers = {}
            key_env = cfg.apiKeyEnv if source_id == "current" else source_config(config, source_id).vector_api_key_env
            key = os.environ.get(key_env, "") if key_env else ""
            if key:
                headers["api-key"] = key
            try:
                # Never follow a stored URL or redirect to an unconfigured service.
                with httpx.Client(timeout=2, trust_env=False, follow_redirects=False) as client:
                    response = client.post(target.rstrip("/") + "/collections/" + record["collection"] + "/points/count",
                        headers=headers, json={"exact": True, "filter": {"must": [
                            {"key": k, "match": {"value": v}} for k, v in fields.items()]}})
                if response.status_code == 404:
                    count = 0
                else:
                    response.raise_for_status()
                    if len(response.content) > 65536:
                        raise ValueError("Oversized response")
                    body = response.json()
                    if body.get("status") != "ok":
                        raise ValueError("Unconfirmed count")
                    count = body.get("result", {}).get("count")
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid count")
                result["vector_scan_performed"] = True
                check.update(actual_count=count, state="present" if count > 0 and count == record["expected"]
                             else "missing" if count == 0 else "mismatch")
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                check.update(state="unavailable", reason="向量服务不可达或响应无效，不能认定向量不存在")
        result["checks"].append(check)
    if not records:
        result["reason"] = "该版本没有索引记录，不能仅凭原文判断向量是否存在"
    return result
