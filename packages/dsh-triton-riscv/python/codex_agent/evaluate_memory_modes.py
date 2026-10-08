"""Compare lexical, chunk-embedding, and weighted-fusion memory retrieval."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from codex_agent.embeddings import EmbeddingProvider, build_embedding_provider
from codex_agent.memory import (
    FUSION_LEXICAL_WEIGHT,
    MemoryQuery,
    MemoryRecord,
    MemoryStore,
)


DEFAULT_FIXTURE = Path(__file__).parent / "tests/fixtures/memory_retrieval_cases.json"
MODES = ("jaccard", "embedding", "fusion")


def load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("corpus"), list) or not isinstance(payload.get("queries"), list):
        raise ValueError("fixture requires corpus and queries lists")
    return payload


def _relevance(case: dict[str, Any]) -> dict[str, int]:
    labels = case.get("relevance")
    if labels is None:
        labels = {case["expected_source_run"]: 3}
    if not isinstance(labels, dict) or not labels or any(
        not isinstance(score, int) or score < 0 or score > 3
        for score in labels.values()
    ):
        raise ValueError(f"invalid relevance labels for {case.get('id')}")
    return labels


def _metrics(cases: list[dict[str, Any]], limit: int) -> dict[str, float]:
    count = len(cases)
    if not count:
        return {name: 0.0 for name in ("hit_at_1", "recall_at_k", "mrr", "ndcg_at_k", "unsafe_at_k")}
    totals = {name: 0.0 for name in ("hit_at_1", "recall_at_k", "mrr", "ndcg_at_k", "unsafe_at_k")}
    for case in cases:
        labels = case["relevance"]
        ranked = case["top_source_runs"][:limit]
        relevant = {source for source, grade in labels.items() if grade >= 2}
        totals["hit_at_1"] += bool(ranked and ranked[0] in relevant)
        totals["recall_at_k"] += len(relevant.intersection(ranked)) / len(relevant) if relevant else 0.0
        totals["mrr"] += next(
            (1.0 / rank for rank, source in enumerate(ranked, 1) if source in relevant), 0.0
        )
        gains = [2 ** labels.get(source, 0) - 1 for source in ranked]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_gains = sorted((2 ** grade - 1 for grade in labels.values()), reverse=True)[:limit]
        ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
        totals["ndcg_at_k"] += dcg / ideal if ideal else 0.0
        totals["unsafe_at_k"] += bool(set(ranked) & set(case["unsafe_source_runs"]))
    return {name: round(total / count, 6) for name, total in totals.items()}


def evaluate_modes(
    payload: dict[str, Any],
    embedding_provider: EmbeddingProvider,
    *,
    limit: int = 5,
    lexical_weight: float = FUSION_LEXICAL_WEIGHT,
) -> dict[str, Any]:
    if embedding_provider is None:
        raise ValueError("a real embedding provider is required for the three-way comparison")
    if limit < 1 or not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("limit must be positive and lexical_weight must be between 0 and 1")
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "memory.sqlite3"
        with MemoryStore(database) as store:
            sources = set()
            for item in payload["corpus"]:
                if item["source_run"] in sources:
                    raise ValueError(f"duplicate source_run: {item['source_run']}")
                sources.add(item["source_run"])
                store.add(MemoryRecord(**item))
        with MemoryStore(database, embedding_provider) as store:
            backfill = store.embed_missing()
            results: dict[str, Any] = {}
            for mode in MODES:
                cases = []
                durations_ms = []
                for case in payload["queries"]:
                    labels = _relevance(case)
                    if not set(labels) <= sources:
                        raise ValueError(f"unknown relevance source in {case['id']}")
                    started = time.perf_counter()
                    retrieved = store.retrieve(
                        MemoryQuery(**case["query"]),
                        limit=limit,
                        score_mode=mode,
                        lexical_weight=lexical_weight,
                        exclude_source_runs=case.get("exclude_source_runs", ()),
                    )
                    durations_ms.append((time.perf_counter() - started) * 1000)
                    cases.append({
                        "id": case["id"],
                        "relevance": labels,
                        "unsafe_source_runs": case.get("unsafe_source_runs", []),
                        "top_source_runs": [item["source_run"] for item in retrieved],
                        "matches": [
                            {"source_run": item["source_run"], **item["retrieval"]}
                            for item in retrieved
                        ],
                    })
                ordered_durations = sorted(durations_ms)
                results[mode] = {
                    "metrics": _metrics(cases, limit),
                    "latency_ms": {
                        "median": round(statistics.median(ordered_durations), 3) if ordered_durations else 0.0,
                        "p95": round(ordered_durations[math.ceil(0.95 * len(ordered_durations)) - 1], 3) if ordered_durations else 0.0,
                    },
                    "queries": cases,
                }
            return {
                "dataset_description": payload.get("description", "Curated retrieval fixture"),
                "corpus_count": store.stats()["active"],
                "query_count": len(payload["queries"]),
                "top_k": limit,
                "fusion_lexical_weight": lexical_weight,
                "embedding_provider": embedding_provider.name,
                "embedding_model": embedding_provider.model,
                "chunking": {"sections": ["contract", "diagnosis", "outcome"], "aggregation": "max compatible chunk similarity"},
                "embedding_backfill": backfill,
                "modes": results,
            }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Memory Retrieval Ablation",
        "",
        f"- Dataset: {report['dataset_description']}",
        f"- Corpus: {report['corpus_count']} cases; queries: {report['query_count']}",
        f"- Embedding: {report['embedding_provider']}/{report['embedding_model']}",
        f"- Top-k: {report['top_k']}; fusion keyword weight: {report['fusion_lexical_weight']:.2f}",
        "- Fusion uses weighted reciprocal ranks (k=60), not uncalibrated raw cosine/Jaccard scores.",
        "",
        "| Mode | Hit@1 | Recall@k | MRR | nDCG@k | Unsafe@k | Median ms | P95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        metrics = report["modes"][mode]["metrics"]
        latency = report["modes"][mode]["latency_ms"]
        lines.append(
            f"| {mode} | {metrics['hit_at_1']:.1%} | {metrics['recall_at_k']:.1%} | "
            f"{metrics['mrr']:.3f} | {metrics['ndcg_at_k']:.3f} | {metrics['unsafe_at_k']:.1%} | "
            f"{latency['median']:.1f} | {latency['p95']:.1f} |"
        )
    lines.extend(["", "## Query rankings", ""])
    for index, first in enumerate(report["modes"][MODES[0]]["queries"]):
        lines.append(f"### {first['id']}")
        lines.append(
            "Labels: " + ", ".join(
                f"{source}={grade}" for source, grade in first["relevance"].items()
            )
        )
        if first["unsafe_source_runs"]:
            lines.append("Known incompatible: " + ", ".join(first["unsafe_source_runs"]))
        for mode in MODES:
            case = report["modes"][mode]["queries"][index]
            lines.append(f"- {mode}: {', '.join(case['top_source_runs']) or '(empty)'}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--embedding-provider", choices=("sentence-transformers", "openai-compatible", "ollama"), required=True)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-api-key-env", default="AGENT_EMBEDDING_API_KEY")
    parser.add_argument("--embedding-tokenizer-json")
    parser.add_argument("--embedding-token-budget", type=int)
    parser.add_argument("--lexical-weight", type=float, default=FUSION_LEXICAL_WEIGHT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    provider = build_embedding_provider(
        args.embedding_provider,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key_env=args.embedding_api_key_env,
        tokenizer_json=args.embedding_tokenizer_json,
        token_budget=args.embedding_token_budget,
    )
    report = evaluate_modes(
        load_fixture(args.fixture), provider,
        limit=args.limit, lexical_weight=args.lexical_weight,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
