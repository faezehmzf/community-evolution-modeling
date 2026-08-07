"""Apply TDMEC SQL migrations and enable extensions."""
from __future__ import annotations

import argparse
from pathlib import Path

from tdmec_db.connection import connect, prepare_age_session, resolve_database_url


def _schema_sql() -> str:
    path = Path(__file__).with_name("schema.sql")
    return path.read_text(encoding="utf-8")


def migrate(database_url: str | None = None) -> None:
    url = resolve_database_url(database_url)
    with connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS age")
        conn.commit()
        prepare_age_session(conn)
        with conn.cursor() as cur:
            cur.execute(_schema_sql())
        conn.commit()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Apply TDMEC Postgres migrations")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args(argv)
    migrate(args.database_url)
    print("tdmec_db migrate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
