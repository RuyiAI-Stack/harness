"""Deterministic, source-bound evidence assembly. No inference of repair causality."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

SCHEMA = "evidence-chain-v1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def source_key(path: str) -> str:
    # Historical exports replace the checkout prefix, never the relative artifact path.
    value = str(path).replace("\\", "/")
    for prefix in ("agent-results/", "python/", "tasks/"):
        if prefix in value:
            return prefix + value.split(prefix, 1)[1]
    return value


def instant(value: str | None):
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return date if date.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class EvidenceSource:
    path: str
    content: str
    pointer: str = ""
    lines: tuple[int, int] | None = None
    available_at: str | None = None
    time_basis: str = "unknown"
    source_sha256: str | None = None

    def object(self) -> dict:
        if self.pointer:
            return {}
        try:
            value = json.loads(self.content)
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}


def evidence_item(source: EvidenceSource, kind: str, state: str, value: Any,
                  run_id: str, *, pointer: str = "", proposal_id: str | None = None,
                  attempt: Any = None, relation: str = "receipt-field") -> dict:
    content = encode(value)
    full_hash = digest(content)
    truncated = len(content) > 8000
    if truncated:
        content = content[:8000] + "\n[excerpt truncated; see source]"
    location = pointer or source.pointer
    identity = encode([source_key(source.path), location, source.lines, kind, run_id, proposal_id, attempt])
    return {"evidence_id": digest(identity)[:24], "kind": kind, "state": state,
            "run_id": run_id, "proposal_id": proposal_id, "attempt": attempt,
            "relation": relation, "source": source.path, "pointer": location,
            "lines": list(source.lines) if source.lines else None,
            "source_sha256": source.source_sha256,
            "content_sha256": digest(source.content), "value_sha256": full_hash,
            "available_at": source.available_at, "time_basis": source.time_basis,
            "text": content, "truncated": truncated}


def log_fragments(source: EvidenceSource) -> list[EvidenceSource]:
    lines = source.content.splitlines()
    hits = [i for i, line in enumerate(lines) if re.search(
        r"error:|Error|Traceback|CalledProcessError|FAILED|\b\d+ (?:passed|failed|skipped)\b", line)]
    if not hits:
        return []
    selected = sorted(set(hits[:2] + hits[-2:]))
    spans = []
    for index in selected:
        start, end = max(0, index - 1), min(len(lines), index + 3)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start, end))
    offset = source.lines[0] - 1 if source.lines else 0
    return [replace(source, content="\n".join(lines[a:b]), lines=(a+offset+1, b+offset)) for a,b in spans]


def assemble_receipt_evidence(receipt: dict, sources: list[EvidenceSource],
                              *, as_of: str | None = None) -> dict:
    """Accept explicit IDs/paths only; unknown/future sources fail closed for as-of."""
    rid = str(receipt.get("run_id") or "unknown")
    items, gaps, conflicts = [], [], []
    allowed = []
    cutoff = instant(as_of)
    if as_of and cutoff is None:
        raise ValueError("as_of must be a timezone-aware timestamp")
    for source in sources:
        stamp = instant(source.available_at)
        if cutoff and (stamp is None or stamp > cutoff):
            gaps.append({"source": source.path, "reason": "unknown-or-future-availability"})
        else:
            allowed.append(source)
    anchors = [s for s in allowed if s.object().get("run_id") == rid
               and "dry_run" in s.object() and not s.object().get("proposal_id")]
    # Current schemas can include proposal_id on validation receipts.
    if not anchors:
        anchors = [s for s in allowed if s.object().get("run_id") == rid and "dry_run" in s.object()]
    if not anchors:
        return {"schema": SCHEMA, "identity": rid, "items": [], "conflicts": [],
                "gaps": gaps + [{"reason":"receipt-source-not-available"}],
                "remote_version_binding":"unknown", "repair_causality":"not-established"}
    anchor = anchors[0]
    for other in anchors[1:]:
        if other.object() != anchor.object():
            conflicts.append({"field":"receipt", "sources":[anchor.path, other.path], "reason":"conflicting-receipt-snapshots"})
    # The supplied receipt must correspond to the retained source, not arbitrary caller data.
    if any(anchor.object().get(k) != receipt.get(k) for k in ("run_id","status","exit_code","operator","dry_run")):
        raise ValueError("receipt and evidence source disagree")
    for kind, pointer, value, state in [
        ("validation", "", {k:receipt.get(k) for k in (
            "run_id","status","dry_run","command","exit_code","test_summary","correctness",
            "test_files","source_sha256","test_sha256","execution_target")}, "reported-result"),
        ("error", "/error_excerpt", receipt.get("error_excerpt"), "observed-error"),
        ("diagnosis", "/diagnosis" if receipt.get("diagnosis") else "/likely_reason",
         receipt.get("diagnosis") or receipt.get("likely_reason"), "reported-not-proven"),
        ("environment", "/remote_preflight", receipt.get("remote_preflight"), "reported-environment"),
    ]:
        if value:
            items.append(evidence_item(anchor, kind, state, value, rid, pointer=pointer))
    diagnosis = receipt.get("diagnosis") if isinstance(receipt.get("diagnosis"), dict) else {}
    if diagnosis.get("recommended_actions"):
        items.append(evidence_item(anchor, "recommendation", "not-executed", diagnosis["recommended_actions"], rid,
                                   pointer="/diagnosis/recommended_actions"))
    remote = receipt.get("remote_preflight") if isinstance(receipt.get("remote_preflight"), dict) else {}
    for key in ("architecture", "triton_version"):
        if receipt.get(key) and remote.get(key) and receipt[key] != remote[key]:
            conflicts.append({"field":key, "source":anchor.path, "top_level":receipt[key], "remote_preflight":remote[key]})
    log_key = source_key(receipt.get("log_path") or "")
    logs = [s for s in allowed if log_key and source_key(s.path) == log_key and not s.pointer]
    for log in logs:
        fragments = log_fragments(log)
        for frag in fragments:
            items.append(evidence_item(frag, "log", "observed-log", frag.content, rid, relation="receipt.log_path"))
        summaries = re.findall(r"(?m)^.*\b\d+ (?:passed|failed|skipped)\b.*$", log.content)
        for summary in summaries[-1:]:
            items.append(evidence_item(log, "test_summary", "observed-log", summary, rid, relation="receipt.log_path"))
            if receipt.get("status") == "passed" and re.search(r"\b[1-9]\d* failed\b", summary):
                conflicts.append({"field":"status", "receipt_source":anchor.path, "log_source":log.path,
                                  "receipt_status":"passed", "log_summary":summary})
    if log_key and not logs:
        gaps.append({"source":receipt.get("log_path"), "reason":"referenced-log-unavailable"})
    for proposal_source in allowed:
        proposal = proposal_source.object()
        pid = proposal.get("proposal_id")
        if not pid or "dry_run" in proposal:
            continue
        if proposal.get("run_id") != rid and proposal.get("validation_run_id") != rid:
            continue
        if proposal.get("operator") != receipt.get("operator"):
            conflicts.append({"field":"operator", "source":proposal_source.path, "reason":"linked-proposal-operator-mismatch"})
            continue
        relation = "proposal.run_id" if proposal.get("run_id") == rid else "proposal.validation_run_id"
        applied = proposal.get("status") == "applied" and bool(proposal.get("applied_at"))
        state = "host-recorded-applied" if applied else "proposed-not-applied"
        value = {k:proposal.get(k) for k in ("proposal_id","run_id","validation_run_id","status","created_at",
                    "applied_at","applied_source_sha256","source_sha256","test_sha256","repair_attempt")}
        items.append(evidence_item(proposal_source, "action", state, value, rid, proposal_id=pid,
                                   attempt=proposal.get("repair_attempt"), relation=relation))
        for source in allowed:
            same_doc = source_key(source.path) == source_key(proposal_source.path)
            is_patch = source_key(source.path) == source_key(str(Path(proposal_source.path).parent.parent / "patches" / f"{pid}.diff"))
            if same_doc and source.pointer == "/rationale":
                items.append(evidence_item(source, "rationale", "hypothesis-not-proven", source.content, rid, proposal_id=pid, relation=relation))
            elif is_patch and not source.pointer:
                items.append(evidence_item(source, "patch", state, source.content, rid, proposal_id=pid, relation="proposal-id-patch-path"))
        for key, kind, field_state in [("rationale","rationale","hypothesis-not-proven"),("diff","patch",state)]:
            if proposal.get(key):
                items.append(evidence_item(proposal_source, kind, field_state, proposal[key], rid, pointer="/"+key, proposal_id=pid, relation=relation))
        if not proposal.get("validation_run_id"):
            gaps.append({"source":proposal_source.path,"proposal_id":pid,"reason":"no-explicit-followup-validation-binding"})
    unique = {e["evidence_id"]:e for e in items}
    for index, conflict in enumerate(conflicts):
        entry = evidence_item(anchor, "conflict", "unresolved", conflict, rid, pointer=f"#conflict-{index}")
        unique[entry["evidence_id"]] = entry
    for index, gap in enumerate(gaps):
        entry = evidence_item(anchor, "gap", "unknown", gap, rid, pointer=f"#gap-{index}")
        unique[entry["evidence_id"]] = entry
    return {"schema":SCHEMA, "identity":rid, "items":list(unique.values()), "gaps":gaps,
            "conflicts":conflicts, "remote_version_binding":"unknown", "repair_causality":"not-established"}


def collect_lifecycle_sources(root: Path, receipt: dict) -> list[EvidenceSource]:
    """Read bounded local artifacts; never follow references outside the checkout."""
    root = root.resolve()
    sources = []
    rid = receipt.get("run_id", "unknown")
    stamp = receipt.get("completed_at") or receipt.get("created_at")
    path = receipt.get("receipt_path") or f"agent-results/operator-lifecycle/receipts/{rid}.json"
    sources.append(EvidenceSource(str(path), encode(receipt), available_at=stamp, time_basis="recorded-host-time"))

    def read(path, available_at=None):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.stat().st_size > MAX_SOURCE_BYTES:
            return None
        raw = candidate.read_bytes()
        return EvidenceSource(str(candidate), raw.decode("utf-8", errors="replace"),
                              available_at=available_at, time_basis="host-recorded-not-signed",
                              source_sha256=hashlib.sha256(raw).hexdigest())

    if receipt.get("log_path"):
        source = read(receipt["log_path"], stamp)
        if source:
            sources.append(source)
    for path in sorted((root / "agent-results/operator-lifecycle/proposals").glob("*.json")):
        source = read(path)
        if not source:
            continue
        obj = source.object()
        if rid not in (obj.get("run_id"), obj.get("validation_run_id")):
            continue
        source = replace(source, available_at=obj.get("applied_at") or obj.get("created_at"))
        sources.append(source)
        pid = obj.get("proposal_id")
        if isinstance(pid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            patch = read(path.parent.parent / "patches" / f"{pid}.diff", source.available_at)
            if patch:
                sources.append(patch)
    return sources


def enrich_development(record, run_dir: Path, final: dict, spec: dict):
    """Attach versioned development facts without interpreting patch existence as apply."""
    rid = str(run_dir)
    def source(name, obj=None):
        p = run_dir / name
        content = encode(obj) if obj is not None else p.read_text(encoding="utf-8", errors="replace")
        return EvidenceSource(str(p), content, source_sha256=digest(content))
    final_source = source("final-result.json")
    spec_source = source("operator-spec.json")
    items = [evidence_item(spec_source, "contract", "recorded-contract", spec, rid)]
    attempt_number = record.evidence.get("attempt")
    iteration = record.evidence.get("iteration")
    if attempt_number is not None:
        iteration = record.evidence.get("candidate_validation_iteration")
    selected = [v for v in final.get("validations", []) if isinstance(v, dict) and (
        v.get("iteration") == iteration if iteration is not None else v.get("status") == record.outcome)]
    if attempt_number is not None and iteration is None:
        selected = []
    if record.memory_type == "successful-run":
        selected = selected[-1:]
    for validation in selected:
        attempt = validation.get("iteration")
        items.append(evidence_item(final_source, "validation", "reported-result", validation, rid,
                                   pointer=f"/validations/{final.get('validations', []).index(validation)}", attempt=attempt))
        path = validation.get("log_path")
        # Use an explicit receipt path, or the development writer's documented iteration filename.
        candidate = run_dir / f"validation-{attempt}.log" if attempt is not None else None
        if path:
            expected = Path(str(path)).name
            # Archived checkout prefixes differ; the run directory component must still agree.
            candidate = run_dir / expected if Path(str(path)).parent.name == run_dir.name else None
        if candidate and candidate.is_file() and candidate.stat().st_size <= MAX_SOURCE_BYTES:
            log = source(candidate.name)
            for frag in log_fragments(log):
                items.append(evidence_item(frag, "log", "observed-log", frag.content, rid, attempt=attempt,
                                           relation="development-validation-iteration"))
    for patch in sorted(run_dir.glob("repair-*.patch")):
        if patch.stat().st_size > MAX_SOURCE_BYTES:
            continue
        match = re.fullmatch(r"repair-(\d+)\.patch", patch.name)
        if not match:
            continue
        attempt = int(match[1])
        if attempt_number is not None and attempt != attempt_number:
            continue
        history = next((a for a in final.get("repair_history", []) if a.get("attempt") == attempt), {})
        if attempt_number is None and (history or record.memory_type != "successful-run"):
            continue
        state = "host-recorded-applied" if history.get("accepted") else "saved-patch-application-unknown"
        items.append(evidence_item(source(patch.name), "patch", state, source(patch.name).content, rid,
                                   attempt=attempt, relation="development-repair-attempt"))
    chain = {"schema":SCHEMA,"identity":f"{rid}:{record.memory_type}:iteration={iteration}:attempt={attempt_number}","items":items,
             "gaps":[],"conflicts":[],"remote_version_binding":"unknown","repair_causality":"not-established"}
    return replace(record, evidence={**record.evidence, "chain":chain})
