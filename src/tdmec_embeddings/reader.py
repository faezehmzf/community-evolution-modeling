"""Read text units / event text from Postgres for embedding."""
from __future__ import annotations

from typing import Any, Dict

import psycopg


def corpus_stats(conn: psycopg.Connection, run_id: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM tdmec.node_text_units
            WHERE run_id=%s AND record_status='retained'
            """,
            (run_id,),
        )
        n_units = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT COUNT(*) FROM tdmec.events
            WHERE run_id=%s AND cleaned_text IS NOT NULL AND cleaned_text <> ''
            """,
            (run_id,),
        )
        n_events = int(cur.fetchone()[0])
        cur.execute(
            "SELECT age_graph_name FROM tdmec.runs WHERE run_id=%s",
            (run_id,),
        )
        row = cur.fetchone()
        age_graph = row[0] if row else None
    return {
        "node_text_units_retained": n_units,
        "events_with_cleaned_text": n_events,
        "age_graph_name": age_graph,
    }
