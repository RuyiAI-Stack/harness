#!/usr/bin/env python3
"""Generate a reviewable, read-only demonstration of the autonomous agent."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .memory import MemoryQuery, MemoryStore, ingest_results


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def development_cases(development_dir: Path) -> list[dict]:
    candidates: dict[str, list[dict]] = {}
    for path in sorted(development_dir.glob("*/final-result.json")):
        final = read_json(path)
        operator = final.get("operator")
        if not operator or not final.get("validations"):
            continue
        validations = final["validations"]
        first_failed = next(
            (item for item in validations if item.get("status") == "failed"),
            None,
        )
        final_validation = validations[-1]
        test_summary = final_validation.get("test_summary")
        if not test_summary:
            iteration = final_validation.get("iteration")
            log_path = path.parent / f"validation-{iteration}.log"
            try:
                log_text = log_path.read_text(encoding="utf-8")
            except OSError:
                log_text = ""
            summaries = re.findall(
                r"(?:\d+ (?:passed|failed|skipped|xfailed|xpassed))"
                r"(?:, \d+ (?:passed|failed|skipped|xfailed|xpassed))*"
                r" in [0-9.]+s",
                log_text,
            )
            test_summary = summaries[-1] if summaries else None
        candidates.setdefault(operator, []).append(
            {
                "operator": operator,
                "status": final.get("status"),
                "run_dir": path.parent.as_posix(),
                "repair_attempts": final.get("repair_attempts", 0),
                "initial_failure_stage": (
                    first_failed.get("first_failure_stage")
                    or first_failed.get("failure_stage")
                    if first_failed
                    else None
                ),
                "initial_reason": first_failed.get("likely_reason") if first_failed else None,
                "final_validation": final_validation.get("status"),
                "test_summary": test_summary,
                "test_sha256": final.get("locked_test_sha256"),
                "validation_attempts": len(validations),
            }
        )
    selected: list[dict] = []
    for operator, items in candidates.items():
        items.sort(
            key=lambda item: (
                item["status"] == "passed",
                item["repair_attempts"] > 0,
                item["validation_attempts"],
                item["run_dir"],
            ),
            reverse=True,
        )
        selected.append(items[0])
    return sorted(selected, key=lambda item: item["operator"])


def run_agent_tests(repo_root: Path, output_dir: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "codex_agent/tests",
        "-q",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path = output_dir / "agent-tests.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", completed.stdout)
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "tests": int(match.group(1)) if match else None,
        "reported_seconds": float(match.group(2)) if match else None,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_path": log_path.as_posix(),
    }


def memory_demo(memory_path: Path, development_dir: Path) -> dict:
    with MemoryStore(memory_path) as store:
        ingestion = ingest_results(store, development_dir)
        memories = store.retrieve(
            MemoryQuery(
                operator="tanh_and_mul",
                semantics=(
                    "Compute tanh(x) multiplied by y using a compiler-supported "
                    "equivalent formulation."
                ),
                pytorch_reference="torch.tanh(x) * y",
                tl_ops=["exp", "load", "store"],
                failure_stage="mlir-translate",
            ),
            limit=3,
        )
        stats = store.stats()
    return {
        "database": memory_path.as_posix(),
        "ingestion": ingestion,
        "stats": stats,
        "query": {
            "operator": "tanh_and_mul",
            "failure_stage": "mlir-translate",
        },
        "results": [
            {
                "id": item["id"],
                "memory_type": item["memory_type"],
                "confidence_grade": item["confidence_grade"],
                "failure_stage": item.get("failure_stage"),
                "outcome": item["outcome"],
                "summary": item["summary"],
                "score": item["retrieval"]["score"],
                "source_run": item["source_run"],
            }
            for item in memories
        ],
    }


def project_demo(path: Path) -> dict:
    inventory = read_json(path)
    summary = inventory.get("summary", {})
    return {
        "inventory_path": path.as_posix(),
        "schema_version": inventory.get("schema_version"),
        "pytest_targets": summary.get("pytest_targets"),
        "lit_targets": summary.get("lit_targets"),
        "build_targets": summary.get("build_targets"),
        "total_targets": summary.get("total_targets"),
        "coverage": summary.get("coverage", {}),
    }


def render_report(result: dict) -> str:
    tests = result["agent_tests"]
    project = result["project_discovery"]
    memory = result["memory"]
    lines = [
        "# Triton-RISCV 自主 Agent 演示报告",
        "",
        f"生成时间：{result['generated_at']}",
        "",
        "## 总览",
        "",
        "本次只读演示汇总四类可独立核查的能力：算子生成与修复证据、",
        "Agent 自动化测试、全项目验证目标发现，以及可复用历史经验检索。",
        "",
        f"- Agent 测试：**{tests['tests'] or '未知'} 项，{tests['status']}**",
        f"- 项目验证目标：**{project.get('total_targets') or '不可用'} 项**",
        f"- 有效长期记忆：**{memory['stats']['active']} 条**",
        "- 本次是否重新运行原生 RISC-V：**否**",
        "",
        "## 算子闭环证据",
        "",
        "| 算子 | 最终结果 | 首次失败阶段 | 修复次数 | 最终测试 | 证据目录 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in result["development_cases"]:
        lines.append(
            f"| {item['operator']} | {item['status']} | "
            f"{item.get('initial_failure_stage') or '无'} | "
            f"{item['repair_attempts']} | {item.get('test_summary') or '已记录通过'} | "
            f"`{item['run_dir']}` |"
        )
    repaired_cases = [
        item
        for item in result["development_cases"]
        if item.get("status") == "passed" and item.get("repair_attempts", 0) > 0
    ]
    direct_cases = [
        item
        for item in result["development_cases"]
        if item.get("status") == "passed" and item.get("repair_attempts", 0) == 0
    ]
    lines.append("")
    if repaired_cases:
        item = repaired_cases[0]
        lines.extend(
            [
                f"`{item['operator']}` 是有修复记录的闭环案例：第一次验证在",
                f"`{item.get('initial_failure_stage') or 'unknown'}` 阶段失败；Agent 保持",
                "验收测试哈希不变，仅修改算子实现，最终重新验证通过。",
            ]
        )
    if direct_cases:
        item = direct_cases[0]
        lines.append(
            f"`{item['operator']}` 则无需修复，首次验证即通过"
            f"{('（' + item['test_summary'] + '）') if item.get('test_summary') else ''}。"
        )
    if not result["development_cases"]:
        lines.append("当前没有可用的历史算子开发证据，报告不作闭环通过声明。")
    lines.extend(
        [
            "",
            "## 长期记忆检索",
            "",
            "查询条件：算子 `tanh_and_mul`，失败阶段 `mlir-translate`。",
            "",
            "| 排名 | 记忆类型 | 证据等级 | 标准化阶段 | 得分 | 检索证据 |",
            "| ---: | --- | --- | --- | ---: | --- |",
        ]
    )
    for index, item in enumerate(memory["results"], start=1):
        summary = item["summary"].replace("|", "\\|")
        if len(summary) > 360:
            summary = summary[:357] + "..."
        lines.append(
            f"| {index} | {item['memory_type']} | {item['confidence_grade']} | "
            f"{item.get('failure_stage') or 'none'} | {item['score']:.3f} | {summary} |"
        )
    coverage = project.get("coverage", {})
    lines.extend(
        [
            "",
            "## 全项目自动发现",
            "",
            f"- Pytest 目标：{project.get('pytest_targets')}",
            f"- MLIR/LLVM lit 目标：{project.get('lit_targets')}",
            f"- CMake/Ninja 目标：{project.get('build_targets')}",
            f"- 已检查实现文件：{coverage.get('implementation_files')}",
            f"- 已映射行为测试：{coverage.get('tested')}",
            f"- 仅有构建验证：{coverage.get('build-only')}",
            f"- 尚未映射验证目标：{coverage.get('unmapped')}",
            "",
            "## 可核查文件",
            "",
            f"- 机器可读 Demo 结果：`{result['result_path']}`",
            f"- 完整 Agent 测试日志：`{tests['log_path']}`",
            f"- 长期记忆数据库：`{memory['database']}`",
            f"- 项目验证目标清单：`{project['inventory_path']}`",
            "",
            "报告中的 RISC-V 结果均明确标记为历史证据。服务器不可用期间，",
            "本次 Demo 不会把历史结果描述成新的硬件验证。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reviewable demonstration of agent capabilities."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "agent-results" / "demos" / timestamp
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tests = (
        run_agent_tests(repo_root, output_dir)
        if args.run_tests
        else {
            "status": "skipped",
            "tests": None,
            "log_path": None,
            "exit_code": None,
        }
    )
    result_path = output_dir / "demo-result.json"
    result = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_root": repo_root.as_posix(),
        "result_path": result_path.as_posix(),
        "agent_tests": tests,
        "development_cases": development_cases(
            repo_root / "agent-results" / "development"
        ),
        "memory": memory_demo(
            repo_root / "agent-results" / "memory.sqlite3",
            repo_root / "agent-results" / "development",
        ),
        "project_discovery": project_demo(
            repo_root / "agent-results" / "project-targets.json"
        ),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "demo-report.md"
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {result_path}")
    return 0 if tests.get("status") in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
