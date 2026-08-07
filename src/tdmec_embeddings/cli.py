"""Authorized embedding pilot CLI skeleton (no model download by default)."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "TDMEC embedding pilot scaffold. Refuses to download/run models "
            "unless --authorize-pilot is set."
        )
    )
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--run-id", required=True, help="Source preprocess run_id")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument(
        "--authorize-pilot",
        action="store_true",
        help="Required acknowledgment before any embedding model execution",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate DB connectivity and reader keys only (default safe mode)",
    )
    ap.add_argument(
        "--insert-smoke-vector",
        action="store_true",
        help="Insert a tiny fake vector into pgvector tables (no model)",
    )
    args = ap.parse_args(argv)

    from tdmec_db.connection import connect, resolve_database_url
    from tdmec_embeddings import reader, writer

    url = resolve_database_url(args.database_url)
    if not args.authorize_pilot and not args.dry_run and not args.insert_smoke_vector:
        print(
            "REFUSED: embedding execution requires --authorize-pilot "
            "(or use --dry-run / --insert-smoke-vector for DB-only checks).",
            file=sys.stderr,
        )
        return 2

    with connect(url) as conn:
        stats = reader.corpus_stats(conn, args.run_id)
        out = {
            "run_id": args.run_id,
            "model": args.model,
            "authorize_pilot": bool(args.authorize_pilot),
            "corpus": stats,
        }
        if args.insert_smoke_vector:
            out["smoke_vector"] = writer.insert_smoke_vector(conn, args.run_id)
        if args.authorize_pilot and not args.dry_run and not args.insert_smoke_vector:
            out["status"] = "AUTHORIZED_BUT_NOT_IMPLEMENTED"
            out["message"] = (
                "Model download/encode is intentionally unimplemented in this "
                "scaffold. Use --insert-smoke-vector for pgvector wiring checks."
            )
            print(json.dumps(out, indent=2))
            return 0
        out["status"] = "dry_run_ok" if args.dry_run or args.insert_smoke_vector else "ok"
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
