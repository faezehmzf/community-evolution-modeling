"""Env-gated Postgres + AGE + pgvector integration tests."""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TDMEC_DATABASE_URL"),
    reason="TDMEC_DATABASE_URL not set",
)


def test_migrate_and_extensions():
    from tdmec_db.connection import connect, prepare_age_session
    from tdmec_db.migrate import migrate

    migrate()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT extname FROM pg_extension WHERE extname IN ('age','vector') ORDER BY 1"
            )
            names = [r[0] for r in cur.fetchall()]
        assert names == ["age", "vector"]
        prepare_age_session(conn)


def test_pgvector_smoke_insert():
    from tdmec_embeddings.writer import insert_smoke_vector
    from tdmec_db.connection import connect

    run_id = f"smoke_vec_{uuid.uuid4().hex[:8]}"
    with connect() as conn:
        out = insert_smoke_vector(conn, run_id, dim=8)
    assert out["dim"] == 8
    assert len(out["embedding"]) == 8


def test_age_graph_roundtrip():
    from tdmec_db import age_graph, repositories as repo
    from tdmec_db.connection import connect

    run_id = f"age_{uuid.uuid4().hex[:8]}"
    gname = age_graph.sanitize_graph_name(run_id)
    with connect() as conn:
        repo.upsert_run(
            conn,
            run_id=run_id,
            pipeline="test",
            config_hash="t",
            age_graph_name=gname,
        )
        repo.load_nodes(conn, [(1, "acc_1"), (2, "acc_2")])
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tdmec.edges
                  (run_id, snapshot_id, relation_id, src_index, dst_index, count_raw, weight_log1p)
                VALUES (%s, 10, 2, 1, 2, 3, LN(1+3))
                ON CONFLICT DO NOTHING
                """,
                (run_id,),
            )
        conn.commit()
        age_graph.ensure_graph(conn, gname, reset=True)
        age_graph.upsert_account_vertices(conn, gname, [(1, "acc_1"), (2, "acc_2")])
        n = age_graph.sync_edges_from_sql(conn, run_id=run_id, graph_name=gname)
        assert n == 1
        hits = age_graph.neighbor_query(
            conn, gname, node_index=1, snapshot_id=10, relation="reply"
        )
        assert len(hits) >= 1
