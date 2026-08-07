"""Database URL resolution and connections."""
from __future__ import annotations

import os
from typing import Optional

import psycopg


def resolve_database_url(explicit: Optional[str] = None) -> str:
    url = (
        explicit
        or os.environ.get("TDMEC_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )
    if not url:
        raise RuntimeError(
            "TDMEC_DATABASE_URL (or DATABASE_URL) is required for Postgres backends"
        )
    return url


def connect(database_url: Optional[str] = None, **kwargs) -> psycopg.Connection:
    url = resolve_database_url(database_url)
    conn = psycopg.connect(url, **kwargs)
    return conn


def prepare_age_session(conn: psycopg.Connection) -> None:
    """Load AGE and put ag_catalog on search_path for this session."""
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute('SET search_path = ag_catalog, "$user", public')
    conn.commit()
