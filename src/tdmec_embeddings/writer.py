"""Write embeddings into pgvector tables."""
from __future__ import annotations

from typing import Any, Dict

import psycopg
from pgvector.psycopg import register_vector


def insert_smoke_vector(conn: psycopg.Connection, run_id: str, dim: int = 8) -> Dict[str, Any]:
    """Insert a tiny fake vector to verify pgvector wiring (no model)."""
    register_vector(conn)
    vec = [float(i) / dim for i in range(dim)]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tdmec.runs (run_id, pipeline, config_hash, status, artifact_status)
            VALUES (%s, 'tdmec_embeddings_smoke', 'smoke', 'SMOKE', 'PROVISIONAL')
            ON CONFLICT (run_id) DO NOTHING
            """,
            (run_id,),
        )
        cur.execute(
            """
            INSERT INTO tdmec.node_tweet_embeddings
              (run_id, tweet_id, model_hash, dim, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id, tweet_id, model_hash) DO UPDATE SET
              embedding = EXCLUDED.embedding,
              dim = EXCLUDED.dim
            """,
            (run_id, "smoke_tweet_0", "smoke_model", dim, vec),
        )
        cur.execute(
            """
            SELECT dim, embedding FROM tdmec.node_tweet_embeddings
            WHERE run_id=%s AND tweet_id='smoke_tweet_0'
            """,
            (run_id,),
        )
        row = cur.fetchone()
    conn.commit()
    emb = row[1]
    if hasattr(emb, "to_list"):
        emb_list = emb.to_list()
    elif hasattr(emb, "tolist"):
        emb_list = emb.tolist()
    else:
        emb_list = [float(x) for x in list(emb)]
    return {"tweet_id": "smoke_tweet_0", "dim": int(row[0]), "embedding": emb_list}
