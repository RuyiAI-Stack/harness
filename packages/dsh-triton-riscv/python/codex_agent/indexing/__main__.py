"""Explicit index preparation/backfill/status. Never changes retrieval defaults."""
import argparse
import json
from pathlib import Path

from codex_agent.runtime_config import runtime_config
from .store import IndexStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "backfill", "backfill-batch", "status", "activate", "retire", "cleanup", "query"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--generation")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--task-stage", choices=["lookup", "generation", "diagnosis", "repair", "validation"], default="lookup")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cursor", help="JSON next_cursor returned by backfill-batch")
    args = parser.parse_args()
    cfg = runtime_config()
    if not cfg.indexing.enabled or Path(cfg.repoRoot).resolve() != args.workspace.resolve():
        parser.error("enable indexing and select the configured workspace")
    store = IndexStore(args.workspace)
    if args.command == "query":
        if not args.query:
            parser.error("query requires --query text")
        from codex_agent.memory import MemoryStore, MemoryQuery, render_memory_context
        from codex_agent.diagnostic_memory import _public_item
        from .retrieval import retrieve
        trace = {}
        store = MemoryStore(args.workspace)
        items = retrieve(store, MemoryQuery("", args.query, ""), args.limit,
                         generation=args.generation, allow_staging=bool(args.generation), trace=trace)
        from codex_agent.memory_expansion import expand_cases
        items = expand_cases(store, items, args.task_stage, trace=trace, allow_staging=bool(args.generation))
        public = [_public_item(item, args.query) for item in items]
        result = {"items": public, "trace": trace,
                  "context_excerpt": render_memory_context(public, query_text=args.query)}
    elif args.command == "prepare":
        with store.db.transaction() as conn:
            store.db.lock(conn, "memory-write")
            row = store.prepare(conn)
        result = {key: row[key] for key in ("generation", "status", "collection")}
    elif args.command == "backfill":
        result = store.backfill(retry=args.retry_failed)
    elif args.command == "backfill-batch":
        try:
            cursor = json.loads(args.cursor) if args.cursor else None
            result = store.backfill_batch(cursor=cursor, page_size=args.batch_size, retry=args.retry_failed)
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "activate":
        result = store.activate(args.generation or cfg.indexing.generation)
    elif args.command == "retire":
        if not args.generation:
            parser.error("retire requires an explicit --generation")
        result = store.retire(args.generation)
    elif args.command == "cleanup":
        from .cleanup import cleanup
        result = cleanup(store, args.generation or cfg.indexing.generation)
    else:
        result = store.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
