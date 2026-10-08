"""Evaluate deterministic failure diagnosis against curated log cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from codex_agent.failure_diagnosis import diagnose_log


DEFAULT_CASES = Path(__file__).parent / "tests/fixtures/diagnosis_cases.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("diagnosis fixture must contain a cases list")
    return cases


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        diagnosis = diagnose_log(
            str(case.get("log", "")),
            case.get("exit_code"),
            stage_events=case.get("stage_events", []),
            validation_command=case.get("validation_command"),
        )
        evidence_text = "\n".join(item["text"] for item in diagnosis["evidence"])
        result = {
            "id": case["id"],
            "expected_rule_id": case["expected_rule_id"],
            "actual_rule_id": diagnosis["rule_id"],
            "expected_stage": case["expected_stage"],
            "actual_stage": diagnosis["failure_stage"],
            "expected_category": case["expected_category"],
            "actual_category": diagnosis["category"],
            "rule_correct": diagnosis["rule_id"] == case["expected_rule_id"],
            "stage_correct": diagnosis["failure_stage"] == case["expected_stage"],
            "category_correct": diagnosis["category"] == case["expected_category"],
            "evidence_found": str(case["evidence_contains"]).lower() in evidence_text.lower(),
            "confidence": diagnosis["confidence"],
        }
        results.append(result)

    total = len(results)

    def accuracy(key: str) -> float:
        if not total:
            return 0.0
        return sum(bool(item[key]) for item in results) / total

    return {
        "schema_version": 1,
        "case_count": total,
        "rule_accuracy": accuracy("rule_correct"),
        "stage_accuracy": accuracy("stage_correct"),
        "category_accuracy": accuracy("category_correct"),
        "evidence_recall": accuracy("evidence_found"),
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    def marker(value: bool) -> str:
        return "pass" if value else "fail"

    lines = [
        "# Failure Diagnosis Evaluation",
        "",
        f"- Cases: {report['case_count']}",
        f"- Rule accuracy: {report['rule_accuracy']:.1%}",
        f"- Stage accuracy: {report['stage_accuracy']:.1%}",
        f"- Category accuracy: {report['category_accuracy']:.1%}",
        f"- Evidence recall: {report['evidence_recall']:.1%}",
        "",
        "| Case | Rule | Stage | Category | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {marker(item['rule_correct'])} | "
            f"{marker(item['stage_correct'])} | {marker(item['category_correct'])} | "
            f"{marker(item['evidence_found'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_cases(load_cases(args.cases))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0 if all(
        report[key] == 1.0
        for key in ("rule_accuracy", "stage_accuracy", "category_accuracy", "evidence_recall")
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
