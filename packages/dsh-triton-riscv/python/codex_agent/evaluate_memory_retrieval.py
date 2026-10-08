"""Evaluate evidence-memory retrieval against a curated regression fixture."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from codex_agent.memory import MemoryQuery, MemoryRecord, MemoryStore


DEFAULT_FIXTURE = Path(__file__).parent / "tests/fixtures/memory_retrieval_cases.json"


def load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("corpus"), list) or not isinstance(
        payload.get("queries"), list
    ):
        raise ValueError("fixture requires corpus and queries lists")
    return payload


def evaluate_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        with MemoryStore(Path(temporary) / "memory.sqlite3") as store:
            source_to_id: dict[str, int] = {}
            for item in payload["corpus"]:
                memory_id, _ = store.add(MemoryRecord(**item))
                source_to_id[item["source_run"]] = memory_id

            results = []
            for case in payload["queries"]:
                retrieved = store.retrieve(
                    MemoryQuery(**case["query"]),
                    limit=int(case.get("limit", 3)),
                )
                expected_id = source_to_id[case["expected_source_run"]]
                ranked_ids = [item["id"] for item in retrieved]
                rank = (
                    ranked_ids.index(expected_id) + 1
                    if expected_id in ranked_ids
                    else None
                )
                results.append(
                    {
                        "id": case["id"],
                        "expected_source_run": case["expected_source_run"],
                        "rank": rank,
                        "top_source_runs": [
                            item["source_run"] for item in retrieved
                        ],
                    }
                )

    count = len(results)
    hit_at_1 = sum(item["rank"] == 1 for item in results) / count if count else 0.0
    hit_at_3 = sum(
        item["rank"] is not None and item["rank"] <= 3 for item in results
    ) / count if count else 0.0
    mrr = sum(
        1.0 / item["rank"] if item["rank"] is not None else 0.0
        for item in results
    ) / count if count else 0.0
    return {
        "query_count": count,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mean_reciprocal_rank": mrr,
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Memory Retrieval Evaluation",
        "",
        f"- Queries: {report['query_count']}",
        f"- Hit@1: {report['hit_at_1']:.1%}",
        f"- Hit@3: {report['hit_at_3']:.1%}",
        f"- Mean reciprocal rank: {report['mean_reciprocal_rank']:.3f}",
        "",
        "| Query | Expected rank | Top sources |",
        "| --- | ---: | --- |",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {item['rank'] or 'miss'} | "
            f"{', '.join(item['top_source_runs'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_fixture(load_fixture(args.fixture))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "query_count",
        "hit_at_1",
        "hit_at_3",
        "mean_reciprocal_rank",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
