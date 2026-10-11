"""Stage-specific, bounded historical evidence; retrieval never authorizes work."""
from codex_agent.memory_api import retrieve_operator_memory
from .roles import Evidence


def role_evidence(root, task, seed, role, phase):
    if role == "main":
        return [], {"status": "not_requested", "reason": "routing uses current requirements"}
    stage = ("validation" if role == "reviewer" else
             "repair" if task["kind"] == "repair" else "generation")
    try:
        result = retrieve_operator_memory(root, task["operator"], run_id=seed.failure_run_id,
            semantics=task["request"]["requirement"],
            pytorch_reference=seed.specification.pytorch_reference if seed.specification else "",
            limit=3, task_stage=stage, exclude_source_runs=tuple(seed.exclude_source_runs))
        text = result.context_excerpt or ""
        trace = {"status": result.status, "stage": stage, "case_ids": [i.memory_id for i in result.items],
                 "excluded_source_runs": seed.exclude_source_runs, "context_chars": len(text)}
        if result.status != "found" or not text:
            return [], trace
        if len(text) > 6000:
            return [], {**trace, "status": "omitted", "reason": "context exceeds 6000-character budget"}
        return [Evidence(id="workflow_history", kind="historical", source="current configured RAG; see role artifact",
                         text=text, verified=False)], trace
    except Exception as error:
        return [], {"status": "unavailable", "stage": stage, "error_type": type(error).__name__}
