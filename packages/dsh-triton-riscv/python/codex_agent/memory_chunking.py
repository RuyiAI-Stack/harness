"""Source-aware chunks for operator case evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, NamedTuple


class MemoryChunk(NamedTuple):
    kind: str
    position: int
    text: str
    source_field: str


@dataclass(frozen=True)
class ChunkPolicy:
    max_chars: int = 600
    max_tokens: int = 180
    overlap_tokens: int = 24

    def __post_init__(self) -> None:
        if self.max_chars < 1 or self.max_tokens < 1 or self.overlap_tokens < 0:
            raise ValueError("chunk limits must be positive and overlap nonnegative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap must be smaller than the token budget")


def chunk_title(operator: str, kind: str, source_field: str) -> str:
    return f"operator: {operator}\nsection: {kind}\nsource: {source_field}\n"


def embedding_input(operator: str, chunk: MemoryChunk) -> str:
    return chunk_title(operator, chunk.kind, chunk.source_field) + chunk.text


def _boundaries(text: str, level: int) -> list[str]:
    if level == 0:
        positions = [match.end() for match in re.finditer(r"\n[ \t]*\n", text)]
    elif level == 1:
        positions = [index + 1 for index, char in enumerate(text) if char == "\n"]
    elif level == 2:
        positions = [match.end() for match in re.finditer(r"[。！？.!?](?=\s|[^\x00-\x7f])", text)]
    else:
        positions = [match.end() for match in re.finditer(r"\s+", text)]
    positions = [position for position in positions if 0 < position < len(text)]
    return [text[start:end] for start, end in zip([0, *positions], [*positions, len(text)])]


def split_chunk_text(
    value: str,
    *,
    title: str = "",
    token_count: Callable[[str], int] | None = None,
    policy: ChunkPolicy = ChunkPolicy(),
) -> list[str]:
    """Preserve complete short blocks; recursively split oversized blocks."""
    text = value.strip()
    if not text:
        return []

    def fits(part: str) -> bool:
        return len(part) <= policy.max_chars and (
            token_count is None or token_count(title + part) <= policy.max_tokens
        )

    if not fits(""):
        raise ValueError("embedding title exceeds the configured token budget")

    if fits(text):
        return [text]
    paragraphs = _boundaries(text, 0)
    if len(paragraphs) > 1:
        return [
            chunk
            for paragraph in paragraphs
            for chunk in split_chunk_text(
                paragraph, title=title, token_count=token_count, policy=policy,
            )
        ]

    def room_for_overlap(part: str) -> bool:
        return len(part) <= policy.max_chars - policy.overlap_tokens and (
            token_count is None
            or token_count(title + part) <= policy.max_tokens - policy.overlap_tokens
        )

    def split(part: str, level: int) -> list[str]:
        if room_for_overlap(part):
            return [part]
        if level < 4:
            units = _boundaries(part, level)
            if len(units) > 1:
                result: list[str] = []
                current = ""
                for unit in units:
                    for piece in split(unit, level + 1):
                        if current and not room_for_overlap(current + piece):
                            result.append(current.strip())
                            current = ""
                        current += piece
                if current.strip():
                    result.append(current.strip())
                return result
            return split(part, level + 1)
        result = []
        start = 0
        while start < len(part):
            low, high = 1, min(len(part) - start, policy.max_chars)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if room_for_overlap(part[start:start + middle]):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if not best:
                raise ValueError("one character exceeds the embedding token budget")
            result.append(part[start:start + best])
            start += best
        return result

    pieces = split(text, 0)
    if len(pieces) < 2 or not policy.overlap_tokens:
        return pieces
    # Overlap is drawn only from the preceding piece of this same source field.
    overlapped = [pieces[0]]
    for previous, piece in zip(pieces, pieces[1:]):
        suffix = ""
        for index in range(1, len(previous) + 1):
            candidate = previous[-index:]
            if token_count is None:
                within_overlap = index <= policy.overlap_tokens
            else:
                within_overlap = (
                    token_count(title + candidate) - token_count(title)
                    <= policy.overlap_tokens
                )
            if not within_overlap or not fits(candidate + piece):
                break
            suffix = candidate
        if suffix and fits(suffix + piece):
            overlapped.append(suffix + piece)
        else:
            overlapped.append(piece)
    return overlapped


def case_fields(item: dict) -> list[tuple[str, str, str]]:
    """Return the normalized, source-bound text fields used by both chunkers."""
    evidence = item.get("evidence") or {}
    fields: list[tuple[str, str, str]] = []

    def add(kind: str, source: str, value: object, label: str) -> None:
        if value is not None and str(value).strip():
            fields.append((kind, source, f"{label}: {value}"))

    add("contract", "operator", item.get("operator"), "Operator")
    add("contract", "semantics", item.get("semantics"), "Semantics")
    add("contract", "pytorch_reference", item.get("pytorch_reference"), "PyTorch reference")
    if item.get("tl_ops"):
        add("contract", "tl_ops", ", ".join(item["tl_ops"]), "Triton operations")

    episode = evidence.get("repair_episode")
    if isinstance(episode, dict):
        failure = episode.get("initial_failure") or {}
        diagnosis = episode.get("diagnosis") or {}
        repair = episode.get("repair") or {}
        validation = episode.get("validation") or {}
        add("diagnosis", "failure_stage", failure.get("stage") or item.get("failure_stage"), "Failure stage")
        add("diagnosis", "evidence.repair_episode.initial_failure.command", failure.get("command"), "Failed command")
        for index, excerpt in enumerate(failure.get("error_excerpt") or []):
            add(
                "diagnosis",
                f"evidence.repair_episode.initial_failure.error_excerpt[{index}]",
                excerpt,
                "Observed error",
            )
        add(
            "diagnosis",
            "evidence.repair_episode.diagnosis.verified_cause",
            diagnosis.get("verified_cause"),
            "Cause verified by applied patch and passing regression",
        )
        add(
            "outcome",
            "evidence.repair_episode.repair.applied_action",
            repair.get("applied_action"),
            "Applied action",
        )
        add(
            "outcome",
            "evidence.repair_episode.repair.patch",
            repair.get("patch"),
            "Applied patch",
        )
        add(
            "outcome",
            "evidence.repair_episode.validation.summary",
            validation.get("summary"),
            "Validation result",
        )
        add(
            "outcome",
            "evidence.repair_episode.validation.regression_command",
            validation.get("regression_command"),
            "Regression command",
        )
    else:
        add("diagnosis", "failure_stage", item.get("failure_stage"), "Failure stage")
        for index, excerpt in enumerate(evidence.get("error_excerpt") or []):
            add("diagnosis", f"evidence.error_excerpt[{index}]", excerpt, "Observed error")
        if item.get("outcome") != "passed":
            add("diagnosis", "summary", item.get("summary"), "Reported cause (not verified)")

        add("outcome", "evidence.attempted_action", evidence.get("attempted_action"), "Attempted action")
        add("outcome", "evidence.applied_action", evidence.get("applied_action"), "Applied action")
        add(
            "outcome", "evidence.patch_excerpt", evidence.get("patch_excerpt"),
            "Applied patch" if evidence.get("accepted") else "Attempted patch (not verified as applied)",
        )
        add("outcome", "evidence.test_summary", evidence.get("test_summary"), "Validation result")
        add("outcome", "evidence.correctness", evidence.get("correctness"), "Correctness result")

    for index, action in enumerate(evidence.get("recommended_actions") or []):
        add("outcome", f"evidence.recommended_actions[{index}]", action, "Recommended action (not executed)")
    add("outcome", "outcome", item.get("outcome"), "Recorded outcome")
    if item.get("outcome") == "passed":
        add("outcome", "summary", item.get("summary"), "Run summary")

    chain = evidence.get("chain") or {}
    for entry in chain.get("items", []):
        kind = "diagnosis" if entry.get("kind") in {"error", "diagnosis", "log"} else (
            "contract" if entry.get("kind") in {"contract", "environment"} else "outcome")
        identity = entry.get("evidence_id", "unknown")
        binding = {key:entry.get(key) for key in ("run_id", "proposal_id", "attempt", "state", "source", "pointer", "lines")}
        add(kind, f"evidence.chain.{identity}", entry.get("text"), json.dumps(binding, ensure_ascii=False))

    return fields


def case_chunks(
    item: dict,
    *,
    token_count: Callable[[str], int] | None = None,
    policy: ChunkPolicy = ChunkPolicy(),
) -> list[MemoryChunk]:
    operator = str(item.get("operator") or "unknown")
    fields = case_fields(item)

    chunks: list[MemoryChunk] = []
    positions: dict[str, int] = {}
    for kind, source, content in fields:
        title = chunk_title(operator, kind, source)
        segments = [content]
        if source in {
            "evidence.patch_excerpt",
            "evidence.repair_episode.repair.patch",
        } and (
            len(content) > policy.max_chars
            or (token_count is not None and token_count(title + content) > policy.max_tokens)
        ):
            segments = re.split(r"(?=^diff --git |^@@ )", content, flags=re.MULTILINE)
            if len(segments) > 1 and segments[0].strip():
                segments[1] = segments[0] + segments[1]
                segments = segments[1:]
        for segment in segments:
            for part in split_chunk_text(segment, title=title, token_count=token_count, policy=policy):
                position = positions.get(kind, 0)
                chunks.append(MemoryChunk(kind, position, part, source))
                positions[kind] = position + 1
    return chunks
