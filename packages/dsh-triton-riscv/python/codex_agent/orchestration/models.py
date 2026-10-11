"""Input contracts and pre-execution transitions for operator workflows."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Identifier = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Role = Literal["main", "semantic", "implementer", "reviewer"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class InputVersions(Contract):
    contract_sha256: Sha256 | None = None
    source_sha256: Sha256 | None = None
    tests_sha256: Sha256 | None = None


class TaskRequest(Contract):
    session_id: Identifier
    kind: Literal["create", "repair"]
    operator: Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]
    requirement: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16000)]
    versions: InputVersions = Field(default_factory=InputVersions)
    max_repairs: int = Field(default=3, ge=1, le=20)
    auto_repair: bool = False
    auto_publish: bool = False


class StepRequest(Contract):
    role: Role
    instruction: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    run_id: Identifier | None = None
    job_id: Identifier | None = None


class StepResult(Contract):
    status: Literal["completed", "failed", "insufficient_evidence"]
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    evidence_refs: list[Annotated[str, StringConstraints(min_length=1, max_length=512)]] = Field(
        default_factory=list, max_length=32)
    artifact_id: Sha256 | None = None


ROLES_BY_STATE = {
    "preparing": {"main", "semantic"},
    "planning_tests": {"reviewer"},
    "implementing": {"implementer"},
    "reviewing": {"reviewer"},
}
NEXT_STATE = {
    "preparing": "planning_tests",
    "planning_tests": "implementing",
    "implementing": "reviewing",
    "reviewing": "waiting_approval",
}
REQUIRED_VERSIONS = {
    "planning_tests": {"contract_sha256"},
    "implementing": {"contract_sha256", "tests_sha256"},
    "reviewing": {"contract_sha256", "tests_sha256", "source_sha256"},
    "waiting_approval": {"contract_sha256", "tests_sha256", "source_sha256"},
}
WAITING = {"waiting_approval", "waiting_validation_approval", "waiting_validation", "applying"}
TERMINAL = {"succeeded", "failed", "cancelled", "blocked", "needs_reconciliation"}
# Execution transitions belong to the evidence/approval adapter, not agent text.
PROTECTED_STATES = {"waiting_validation", "repairing", "succeeded"}
