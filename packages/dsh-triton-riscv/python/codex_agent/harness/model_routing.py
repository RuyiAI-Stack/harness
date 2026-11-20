"""Operator-specific task facts for the reusable host model router."""
from dataclasses import asdict, replace
import re

from dsh_multi_agent.routing import RoutingPolicy
from codex_agent.runtime_config import runtime_config


def policy():
    return RoutingPolicy(**runtime_config().modelRouting.model_dump())


def routing_status(settings):
    cfg = policy()
    return {**asdict(cfg), "strongModel": cfg.strongModel or settings.model}


def classify(request, role=None):
    # Ambiguity and execution stay strong. Text is never interpreted as a model ID.
    if role in {"implementer", "reviewer", "main"}:
        return False, "code_or_review_role"
    text = request.casefold()
    # A bounded source noun is not an instruction to implement a feature.
    # Keep "实现文件上传" and other action phrases on the strong route.
    text = re.sub(r"实现(?:文件|路径|位置|源码)(?=$|[\s、，,。：:；;和与及/])", "源码资料", text)
    if re.search(r"修复|修改|生成|开发|实现|审查|验证|执行|运行|失败|报错|推导|正确性|并行|子智能体|"
                 r"\b(fix|repair|debug|implement|create|generate|review|verify|validate|execute|run|failure|"
                 r"error|derive|parallel|delegate|prove|write|delete|deploy)\b", text):
        return False, "code_execution_or_complex_task"
    if re.search(r"列出|整理|提取|摘录|汇总|摘要|\b(list|extract|summarize|summarise|inventory)\b", text):
        return True, "bounded_information_task"
    return False, "unknown_task"


def select_route(settings, request, prompt, *, role=None):
    low_risk, reason = classify(request, role)
    return policy().select(provider=settings.provider,
                           default_model=settings.routing_base_model or settings.model,
                           input_bytes=len(prompt.encode("utf-8")), low_risk=low_risk, reason=reason)


def routed_settings(settings, route):
    return replace(settings, model=route["model"],
                   routing_base_model=settings.routing_base_model or settings.model,
                   request_timeout_seconds=min(settings.request_timeout_seconds, route["timeout_seconds"]))


def replica_prompt(spec, task):
    import json
    return json.dumps({"goal": task["goal"], "background": spec["context"],
                       "original_user_request": spec["user_request"]}, ensure_ascii=False)


def replica_route(settings, spec, task):
    # A harmless-looking child goal cannot downgrade a high-risk parent request.
    return select_route(settings, spec["user_request"] + "\n" + task["goal"], replica_prompt(spec, task))
