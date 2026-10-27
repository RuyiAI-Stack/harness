"""Observed Harness usage, never estimates from text length or model output."""
from datetime import datetime, timezone
from time import monotonic


FIELDS = {
    "uncached_input_tokens": "inputTokens", "output_tokens": "outputTokens",
    "cache_read_tokens": "cacheReadTokens", "cache_write_tokens": "cacheWriteTokens",
    "reasoning_tokens": "reasoningTokens", "total_tokens": "totalTokens",
}
TOKEN_FIELDS = (*FIELDS, "input_tokens")


def count(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def normalize_usage(value):
    value = value if isinstance(value, dict) else {}
    result = {key: value.get(name) if count(value.get(name)) else None for key, name in FIELDS.items()}
    known = [result[key] for key in ("uncached_input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")]
    result["known_tokens"] = sum(v for v in known if v is not None) if any(v is not None for v in known) else None
    total, output = result["total_tokens"], result["output_tokens"]
    invalid = any(name in value and not count(value[name]) for name in FIELDS.values())
    if result["reasoning_tokens"] is not None and output is not None and result["reasoning_tokens"] > output:
        invalid = True
    if total is not None and (total < (result["known_tokens"] or 0)
                             or (all(v is not None for v in known) and total != sum(known))):
        invalid = True
    if invalid:
        result["total_tokens"] = None
    elif total is None and all(v is not None for v in known):
        result["total_tokens"] = sum(known)
    result["input_tokens"] = (result["total_tokens"] - output
                              if result["total_tokens"] is not None and output is not None else None)
    result["usage_status"] = "invalid" if invalid else "complete" if result["input_tokens"] is not None else "partial" if value else "unknown"
    return result


def summarize(rows):
    """Keep known subtotals separate; one unknown call makes the exact sum unknown."""
    result = {}
    for field in (*TOKEN_FIELDS, "known_tokens"):
        values = [row.get(field) for row in rows]
        valid = [v for v in values if count(v)]
        result[field] = sum(valid) if values and len(valid) == len(values) else None
        if field == "known_tokens":
            result[field] = sum(valid) if valid else None
    result["unknown_usage_count"] = sum(row.get("usage_status") != "complete" for row in rows)
    result["usage_status"] = ("complete" if rows and not result["unknown_usage_count"] else
                              "partial" if result["known_tokens"] is not None else "unknown")
    if result["usage_status"] != "complete":
        result["total_tokens"] = None
        result["input_tokens"] = None
    return result


class RunUsage:
    """Fold snapshots by attempt, not by notification; callback and result overlap."""
    def __init__(self, session_id, *, provider=None, model=None, prompt_characters=0):
        self.session_id, self.provider, self.model = session_id, provider, model
        self.prompt_characters = prompt_characters
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = monotonic()
        self.events, self.issues = {}, set()

    def observe(self, notification):
        if not isinstance(notification, dict):
            return
        if "method" in notification:
            payload = notification.get("payload", {})
            if notification["method"] != "session.event" or not isinstance(payload, dict) or payload.get("sessionId") != self.session_id:
                return
            event = payload.get("event")
        else:
            event = notification
        if not isinstance(event, dict) or event.get("type") not in {
            "step/start", "step/end", "assistant/chunk", "assistant/message",
            "assistant/attempt", "llm/retry", "llm/retry-started",
        }:
            return
        data = event.get("data")
        if not isinstance(data, dict):
            self.issues.add("malformed_event")
            return
        chunk = data.get("chunk", {})
        if event["type"] == "assistant/chunk" and (not isinstance(chunk, dict) or chunk.get("type") != "usage"):
            return
        if not count(event.get("seq")) or not count(data.get("turn")) or not count(data.get("step")):
            self.issues.add("missing_event_identity")
            return
        usage = data.get("usage") or (chunk.get("usage") if isinstance(chunk, dict) else None)
        stream = data.get("stream", [])
        if usage is None and isinstance(stream, list):
            for item in stream:
                part = item.get("chunk", {}) if isinstance(item, dict) else {}
                if isinstance(part, dict) and part.get("type") == "usage":
                    usage = part.get("usage")
        source = data.get("message", {})
        source = source.get("source", {}) if isinstance(source, dict) else {}
        value = {"type": event["type"], "turn": data["turn"], "step": data["step"],
                 "usage": {name: usage[name] for name in FIELDS.values() if name in usage}
                 if isinstance(usage, dict) else None,
                 "provider": source.get("provider") if isinstance(source, dict) else None,
                 "model": source.get("model") if isinstance(source, dict) else None}
        seq = event["seq"]
        if seq in self.events and self.events[seq] != value:
            self.issues.add("conflicting_event_identity")
        elif len(self.events) >= 10000 and seq not in self.events:
            self.issues.add("event_limit_exceeded")
        else:
            self.events[seq] = value

    def finish(self, outcome):
        attempts, active, issues = [], None, set(self.issues)
        for event in (self.events[key] for key in sorted(self.events)):
            kind, key = event["type"], (event["turn"], event["step"])
            if kind in {"step/start", "llm/retry-started"}:
                if active is not None and not active["closed"]:
                    issues.add("unfinished_attempt")
                if kind == "llm/retry-started" and (active is None or not active["closed"]
                        or active["outcome"] != "failed_attempt" or (active["turn"], active["step"]) != key):
                    issues.add("missing_retry_boundary")
                active = {"turn": key[0], "step": key[1], "attempt": len(attempts) + 1,
                          "usage": None, "closed": False, "provider": self.provider, "model": self.model,
                          "outcome": "unknown", "start_observed": True}
                attempts.append(active)
                continue
            if active is None or (active["turn"], active["step"]) != key:
                issues.add("missing_attempt_start")
                active = {"turn": key[0], "step": key[1], "attempt": len(attempts) + 1,
                          "usage": None, "closed": False, "provider": self.provider, "model": self.model,
                          "outcome": "unknown", "start_observed": False}
                attempts.append(active)
            if active["closed"] and kind in {"assistant/chunk", "assistant/message", "assistant/attempt"}:
                issues.add("usage_after_attempt_closed")
            if event["usage"] is not None:
                active["usage"] = event["usage"]
            for field in ("provider", "model"):
                if isinstance(event[field], str) and event[field]:
                    active[field] = event[field]
            if kind in {"assistant/message", "assistant/attempt", "llm/retry", "step/end"}:
                active["closed"] = True
                if kind == "assistant/message":
                    active["outcome"] = "response_received"
                elif kind in {"assistant/attempt", "llm/retry"}:
                    active["outcome"] = "failed_attempt"
        rows = []
        for attempt in attempts:
            raw_usage = attempt.pop("usage")
            # pi-ai initializes absent provider usage to zeros. It cannot prove
            # a free request, including when a rejected request reports zero.
            if raw_usage and all(count(value) and value == 0 for value in raw_usage.values()):
                raw_usage = None
                issues.add("ambiguous_zero_usage")
            usage = normalize_usage(raw_usage)
            if not attempt["closed"] or not attempt["start_observed"]:
                usage.update(total_tokens=None, input_tokens=None, usage_status="partial" if usage["known_tokens"] is not None else "unknown")
            rows.append({**attempt, **usage})
        totals = summarize(rows)
        if issues:
            totals.update(total_tokens=None, input_tokens=None,
                          usage_status="partial" if totals["known_tokens"] is not None else "unknown")
        return {"schema_version": 1, "session_id": self.session_id, "provider": self.provider,
                "model": self.model, "outcome": outcome, "started_at": self.started_at,
                "elapsed_ms": round((monotonic() - self.started) * 1000, 3),
                "prompt_characters": self.prompt_characters, "attempt_count": len(rows),
                "attempts": rows, "totals": totals, "issues": sorted(issues),
                "cost": None, "cost_status": "pricing_not_configured"}
