"""Server-local read-only audit CLI; never migrates, backfills or embeds."""
import argparse
import json
from .audit import AuditBlocked, load_manifest, run_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only historical RAG audit: no migration, backfill, embedding or publishing.")
    parser.add_argument("--manifest", required=True, help="Server-local JSON manifest; URLs are environment references only")
    parser.add_argument("--audit-dir", required=True, help="Local checkpoint/report directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--deadline-seconds", type=float, default=60)
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        state = run_audit(load_manifest(args.manifest), args.audit_dir, resume=args.resume,
            max_pages=args.max_pages, deadline_seconds=args.deadline_seconds, page_size=args.page_size)
        print(json.dumps({"read_only": True, "status": state["status"], "reason": state["reason"],
            "pages_completed": state["pages_completed"], "counts": state["counts"],
            "anomaly_groups": len(state["findings"]), "checkpoint": args.audit_dir + "/checkpoint.json"}, ensure_ascii=False))
        return 0 if state["status"] == "complete" else (2 if state["status"] == "partial" else 3)
    except (AuditBlocked, OSError, ValueError, TypeError):
        print(json.dumps({"read_only": True, "status": "blocked", "reason": "invalid_manifest_or_checkpoint"}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
