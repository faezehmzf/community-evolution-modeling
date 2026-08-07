"""SQL repositories for TDMEC Postgres tables."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
from psycopg.types.json import Json


def upsert_run(
    conn: psycopg.Connection,
    *,
    run_id: str,
    pipeline: str,
    config_hash: str,
    status: str = "UNVALIDATED",
    artifact_status: str = "PROVISIONAL",
    age_graph_name: Optional[str] = None,
    meta: Optional[dict] = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tdmec.runs
              (run_id, pipeline, config_hash, status, artifact_status, age_graph_name, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
              status = EXCLUDED.status,
              artifact_status = EXCLUDED.artifact_status,
              age_graph_name = COALESCE(EXCLUDED.age_graph_name, tdmec.runs.age_graph_name),
              meta = EXCLUDED.meta,
              updated_at = NOW()
            """,
            (
                run_id,
                pipeline,
                config_hash,
                status,
                artifact_status,
                age_graph_name,
                Json(meta or {}),
            ),
        )
    conn.commit()


def upsert_source_file(
    conn: psycopg.Connection,
    *,
    run_id: str,
    source_file: str,
    sha256: str,
    size_bytes: Optional[int] = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tdmec.source_files (run_id, source_file, sha256, size_bytes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (run_id, source_file) DO UPDATE SET
              sha256 = EXCLUDED.sha256,
              size_bytes = EXCLUDED.size_bytes
            """,
            (run_id, source_file, sha256, size_bytes),
        )
    conn.commit()


def chunk_done(conn: psycopg.Connection, run_id: str, source_file: str, chunk_idx: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM tdmec.file_chunks
            WHERE run_id=%s AND source_file=%s AND chunk_idx=%s
            """,
            (run_id, source_file, chunk_idx),
        )
        return cur.fetchone() is not None


def mark_chunk(
    conn: psycopg.Connection,
    *,
    run_id: str,
    source_file: str,
    chunk_idx: int,
    rows_inspected: int,
    events_inserted: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tdmec.file_chunks
              (run_id, source_file, chunk_idx, rows_inspected, events_inserted)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id, source_file, chunk_idx) DO UPDATE SET
              rows_inspected = EXCLUDED.rows_inspected,
              events_inserted = EXCLUDED.events_inserted,
              completed_at = NOW()
            """,
            (run_id, source_file, chunk_idx, rows_inspected, events_inserted),
        )
    conn.commit()


def load_nodes(conn: psycopg.Connection, rows: Sequence[Tuple[int, str]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tdmec.nodes (node_index, author_account_id)
            VALUES (%s, %s)
            ON CONFLICT (node_index) DO UPDATE SET
              author_account_id = EXCLUDED.author_account_id
            """,
            [(int(i), str(a)) for i, a in rows],
        )
    conn.commit()


def load_snapshots(conn: psycopg.Connection, rows: Sequence[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tdmec.snapshots
              (snapshot_id, quarter_label, start_utc, end_utc_exclusive, status)
            VALUES (%(snapshot_id)s, %(quarter_label)s, %(start_utc)s,
                    %(end_utc_exclusive)s, %(status)s)
            ON CONFLICT (snapshot_id) DO UPDATE SET
              quarter_label = EXCLUDED.quarter_label,
              start_utc = EXCLUDED.start_utc,
              end_utc_exclusive = EXCLUDED.end_utc_exclusive,
              status = EXCLUDED.status
            """,
            list(rows),
        )
    conn.commit()


def insert_events_ignore(
    conn: psycopg.Connection,
    run_id: str,
    events: Sequence[tuple],
) -> int:
    """events tuples: signature, snapshot_id, relation_id, source_idx, target_idx,
    cleaned_text, text_hash, text_quality, source_file, source_row_number.

    Returns number of rows attempted (exact ON CONFLICT insert count is not
    cheap at scale; progress logs use this as an upper bound).
    """
    if not events:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tdmec.events (
              run_id, signature, snapshot_id, relation_id, source_idx, target_idx,
              cleaned_text, text_hash, text_quality, source_file, source_row_number
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, signature) DO NOTHING
            """,
            [(run_id, *e) for e in events],
        )
    conn.commit()
    return len(events)


def insert_authored_tweets_ignore(
    conn: psycopg.Connection,
    run_id: str,
    tweets: Sequence[tuple],
) -> None:
    """tweets: source_file, source_row_number, snapshot_id, source_idx"""
    if not tweets:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tdmec.authored_tweets
              (run_id, source_file, source_row_number, snapshot_id, source_idx)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            [(run_id, *t) for t in tweets],
        )
    conn.commit()


def rebuild_edges_from_events(conn: psycopg.Connection, run_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tdmec.edges WHERE run_id = %s", (run_id,))
        cur.execute(
            """
            INSERT INTO tdmec.edges (
              run_id, snapshot_id, relation_id, src_index, dst_index,
              count_raw, weight_log1p
            )
            SELECT
              run_id, snapshot_id, relation_id, source_idx, target_idx,
              COUNT(*)::bigint,
              LN(1 + COUNT(*))::double precision
            FROM tdmec.events
            WHERE run_id = %s AND source_idx <> target_idx
            GROUP BY run_id, snapshot_id, relation_id, source_idx, target_idx
            """,
            (run_id,),
        )
        n = cur.rowcount
    conn.commit()
    return int(n or 0)


def fetch_edge_counts(conn: psycopg.Connection, run_id: str) -> Dict[Tuple[int, int, int, int], int]:
    out: Dict[Tuple[int, int, int, int], int] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT snapshot_id, relation_id, src_index, dst_index, count_raw
            FROM tdmec.edges WHERE run_id = %s
            """,
            (run_id,),
        )
        for sid, rid, src, dst, c in cur.fetchall():
            out[(int(sid), int(rid), int(src), int(dst))] = int(c)
    return out


def fetch_tweet_counts(conn: psycopg.Connection, run_id: str) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT snapshot_id, source_idx, COUNT(*)
            FROM tdmec.authored_tweets
            WHERE run_id = %s
            GROUP BY 1, 2
            """,
            (run_id,),
        )
        for sid, node, c in cur.fetchall():
            out[(int(sid), int(node))] = int(c)
    return out


def stream_events(conn: psycopg.Connection, run_id: str, batch_size: int = 100_000):
    with conn.cursor(name="tdmec_events_stream") as cur:
        cur.itersize = batch_size
        cur.execute(
            """
            SELECT snapshot_id, relation_id, source_idx, target_idx,
                   cleaned_text, text_hash, text_quality, source_file,
                   source_row_number, signature
            FROM tdmec.events
            WHERE run_id = %s
            ORDER BY snapshot_id, relation_id, source_idx, target_idx, signature
            """,
            (run_id,),
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield rows


def upsert_node_text_units(
    conn: psycopg.Connection,
    run_id: str,
    rows: Sequence[dict],
) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tdmec.node_text_units (
              run_id, tweet_id, node_index, snapshot_id, cleaned_text, text_normalized,
              text_hash, text_quality, record_status, source_file, source_row_number,
              artifact_status
            ) VALUES (
              %(run_id)s, %(tweet_id)s, %(node_index)s, %(snapshot_id)s, %(cleaned_text)s,
              %(text_normalized)s, %(text_hash)s, %(text_quality)s, %(record_status)s,
              %(source_file)s, %(source_row_number)s, %(artifact_status)s
            )
            ON CONFLICT (run_id, source_file, source_row_number) DO UPDATE SET
              tweet_id = EXCLUDED.tweet_id,
              node_index = EXCLUDED.node_index,
              snapshot_id = EXCLUDED.snapshot_id,
              cleaned_text = EXCLUDED.cleaned_text,
              text_normalized = EXCLUDED.text_normalized,
              text_hash = EXCLUDED.text_hash,
              text_quality = EXCLUDED.text_quality,
              record_status = EXCLUDED.record_status,
              artifact_status = EXCLUDED.artifact_status
            """,
            [{**r, "run_id": run_id, "artifact_status": r.get("artifact_status", "PROVISIONAL")} for r in rows],
        )
    conn.commit()


def list_nodes(conn: psycopg.Connection) -> List[Tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT node_index, author_account_id FROM tdmec.nodes ORDER BY node_index"
        )
        return [(int(a), str(b)) for a, b in cur.fetchall()]


def count_events(conn: psycopg.Connection, run_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tdmec.events WHERE run_id=%s", (run_id,))
        return int(cur.fetchone()[0])
