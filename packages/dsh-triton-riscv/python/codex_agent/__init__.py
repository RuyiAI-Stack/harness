"""Autonomous discovery and validation tools for Triton-RISCV."""

from .model_router import ModelRouter, RoutingDecision, TelemetryEvent, TelemetryRecorder
from .test_planner import TestPlan, PlannedTarget, plan_tests, plan_from_git
from .project_diagnosis import ProjectFailure, RepairPlan, analyze_failures, plan_repairs
from .project_repair import ProjectPatchProposal, propose_project_patch, review_project_patch, apply_project_patch
from .operator_baseline import (
    BaselineItem,
    build_queue,
    select_representative_operators,
    write_baseline_queue,
)
from .operator_development import (
    DevelopmentPlanResult as DevelopmentPreparation,
    DevelopmentProposalResult as ImplementationProposal,
    apply_operator_implementation,
    decide_operator_development_proposal as decide_implementation_proposal,
    get_operator_development_proposal as get_implementation_proposal,
    prepare_operator_development,
    propose_operator_implementation,
)

__all__ = [
    "ModelRouter", "RoutingDecision", "TelemetryEvent", "TelemetryRecorder",
    "TestPlan", "PlannedTarget", "plan_tests", "plan_from_git",
    "ProjectFailure", "RepairPlan", "analyze_failures", "plan_repairs",
    "ProjectPatchProposal", "propose_project_patch", "review_project_patch", "apply_project_patch",
    "BaselineItem", "build_queue", "select_representative_operators", "write_baseline_queue",
    "DevelopmentPreparation", "ImplementationProposal", "prepare_operator_development",
    "propose_operator_implementation", "get_implementation_proposal",
    "decide_implementation_proposal", "apply_operator_implementation",
]
