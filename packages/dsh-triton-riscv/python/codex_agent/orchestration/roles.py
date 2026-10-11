"""Typed, proposal-only role inputs and outputs. IDs are attached by the caller."""
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator
from codex_agent.operator_development import NewOperatorSpec

from .models import Contract, Identifier, InputVersions, Role, Sha256, ROLES_BY_STATE, REQUIRED_VERSIONS

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Code = Annotated[str, StringConstraints(min_length=1, max_length=48000)]
PathText = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Notes = Annotated[list[Text], Field(max_length=32)]


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def operator_paths(operator):
    return (f"python/examples/flaggems/{operator}.py", f"python/examples/flaggems/test_{operator}.py")


class Evidence(Contract):
    id: Identifier
    kind: Literal["requirement", "contract", "log", "historical", "environment"]
    source: Text
    text: Text
    verified: bool = False


class FileSnapshot(Contract):
    path: PathText
    content: Code
    sha256: Sha256

    @model_validator(mode="after")
    def consistent(self):
        if sha256(self.content) != self.sha256:
            raise ValueError("file content does not match its digest")
        return self


class CandidateFile(Contract):
    path: PathText
    base_sha256: Sha256 | None
    content: Code


class RoleRequest(Contract):
    task_id: Identifier
    step_id: Identifier
    input_epoch: int = Field(ge=1)
    role: Role
    phase: Literal["preparing", "planning_tests", "implementing", "reviewing"]
    kind: Literal["create", "repair"]
    operator: Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]
    requirement: Text
    versions: InputVersions
    evidence: list[Evidence] = Field(default_factory=list, max_length=32)
    files: list[FileSnapshot] = Field(default_factory=list, max_length=2)
    candidate: CandidateFile | None = None
    review_baseline: FileSnapshot | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.role not in ROLES_BY_STATE[self.phase]:
            raise ValueError("role is not allowed in this phase")
        for field in REQUIRED_VERSIONS.get(self.phase, set()):
            if getattr(self.versions, field) is None:
                raise ValueError("required input version is missing")
        if len({e.id for e in self.evidence}) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        paths = operator_paths(self.operator)
        if len({f.path for f in self.files}) != len(self.files) or any(f.path not in paths for f in self.files):
            raise ValueError("file snapshots must belong to the current operator")
        for file in self.files:
            expected = self.versions.source_sha256 if file.path == paths[0] else self.versions.tests_sha256
            if expected is None or expected != file.sha256:
                raise ValueError("file snapshot is not bound to the current input version")
        if self.review_baseline is not None:
            if self.phase != "reviewing" or self.kind != "repair" or self.review_baseline.path != paths[0]:
                raise ValueError("review baseline is only the original repair implementation")
        if self.candidate is not None:
            if self.phase != "reviewing":
                raise ValueError("candidate implementation is only shown during review, not test planning")
            if self.kind == "repair" and self.review_baseline is None:
                raise ValueError("repair review requires the original source snapshot")
            expected_base = self.review_baseline.sha256 if self.review_baseline else None
            current = next((f for f in self.files if f.path == paths[0]), None)
            if (self.candidate.path != paths[0] or self.candidate.base_sha256 != expected_base
                    or current is None or self.candidate.content != current.content):
                raise ValueError("review candidate must match the proposed snapshot and original write-base")
        return self


class CommonResult(Contract):
    status: Literal["completed", "insufficient_evidence", "failed"]
    summary: Text
    evidence_refs: list[Identifier] = Field(default_factory=list, max_length=32)
    unknowns: Notes = Field(default_factory=list)


class MainResult(CommonResult):
    decision: Literal["request_semantics", "plan_tests", "ask_user", "blocked"]
    questions: Notes = Field(default_factory=list)

    @model_validator(mode="after")
    def clarification(self):
        if self.decision == "ask_user" and not self.questions:
            raise ValueError("ask_user requires specific questions")
        return self


class OperatorContract(Contract):
    formula: Text | None
    inputs: Notes
    outputs: Notes
    dtypes: Notes
    shapes: Notes
    boundaries: Notes
    tolerances: Text | None


class SemanticResult(CommonResult):
    contract: OperatorContract
    specification: NewOperatorSpec | None = None


class ImplementationResult(CommonResult):
    candidate: CandidateFile | None


class ReviewResult(CommonResult):
    verdict: Literal["tests_proposed", "acceptable", "changes_requested", "insufficient_evidence"]
    findings: Notes = Field(default_factory=list)
    test_candidate: CandidateFile | None = None


RESULT_TYPES = {"main": MainResult, "semantic": SemanticResult,
                "implementer": ImplementationResult, "reviewer": ReviewResult}


def validate_candidate(candidate, request, allowed_path):
    if candidate.path != allowed_path:
        raise ValueError("candidate path is outside this role's allowed file")
    current = next((f for f in request.files if f.path == allowed_path), None)
    digest = (request.versions.source_sha256 if allowed_path == operator_paths(request.operator)[0]
              else request.versions.tests_sha256)
    if candidate.base_sha256 != digest or (digest is not None and current is None):
        raise ValueError("candidate must reference an observed current file or a declared new file")


def parse_result(request, text):
    if len(text.encode()) > 65536:
        raise ValueError("role output exceeds 64 KiB")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique_object)
    result = RESULT_TYPES[request.role].model_validate(value)
    if not set(result.evidence_refs) <= {e.id for e in request.evidence}:
        raise ValueError("output references evidence not supplied to this role")
    paths = operator_paths(request.operator)
    if isinstance(result, ImplementationResult):
        if result.candidate is not None:
            validate_candidate(result.candidate, request, paths[0])
        if result.status == "completed" and result.candidate is None:
            raise ValueError("completed implementation requires a candidate")
        if result.status == "completed" and not any(f.path == paths[1] for f in request.files):
            raise ValueError("implementation requires observed acceptance tests")
    if isinstance(result, ReviewResult):
        if request.phase == "planning_tests":
            if result.verdict == "acceptable":
                raise ValueError("test planning cannot accept an unseen implementation")
            if result.test_candidate is not None:
                validate_candidate(result.test_candidate, request, paths[1])
            if result.status == "completed" and (result.verdict != "tests_proposed" or result.test_candidate is None):
                raise ValueError("completed test planning requires proposed tests")
        elif result.test_candidate is not None or result.verdict == "tests_proposed":
            raise ValueError("review cannot replace previously fixed acceptance tests")
        elif result.verdict == "acceptable":
            seen = {f.path for f in request.files}
            if result.status != "completed" or paths[1] not in seen or (request.candidate is None and paths[0] not in seen):
                raise ValueError("acceptance requires observed implementation and tests, and a completed review")
    return result
