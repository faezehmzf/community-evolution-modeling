"""Apache AGE helpers: create graph, sync nodes/edges from SQL tables."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import psycopg

from tdmec_db.connection import prepare_age_session

RELATION_LABELS = {
    0: "mention",
    1: "retweet",
    2: "reply",
    3: "quote",
}


def sanitize_graph_name(run_id: str) -> str:
    """AGE graph names must be valid identifiers (max 63 chars)."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", run_id)
    if cleaned and cleaned[0].isdigit():
        cleaned = "g_" + cleaned
    return f"tdmec_g_{cleaned}"[:63]


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def _cypher_list_of_maps(rows: Sequence[Dict[str, Any]]) -> str:
    """Serialize Python dicts to an openCypher list-of-maps literal for AGE."""
    parts: List[str] = []
    for r in rows:
        fields: List[str] = []
        for k, v in r.items():
            if isinstance(v, str):
                fields.append(f"{k}: '{_esc(v)}'")
            elif isinstance(v, float):
                fields.append(f"{k}: {float(v)}")
            elif isinstance(v, bool):
                fields.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                fields.append(f"{k}: null")
            else:
                fields.append(f"{k}: {int(v)}")
        parts.append("{" + ", ".join(fields) + "}")
    return "[" + ", ".join(parts) + "]"


def ensure_graph(conn: psycopg.Connection, graph_name: str, *, reset: bool = False) -> None:
    prepare_age_session(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
            (graph_name,),
        )
        exists = cur.fetchone() is not None
        if exists and reset:
            cur.execute("SELECT drop_graph(%s, true)", (graph_name,))
            exists = False
        if not exists:
            cur.execute("SELECT create_graph(%s)", (graph_name,))
    conn.commit()


def upsert_account_vertices(
    conn: psycopg.Connection,
    graph_name: str,
    nodes: Sequence[Tuple[int, str]],
    *,
    batch_size: int = 1000,
) -> int:
    """Create/merge Account vertices for (node_index, author_account_id) via UNWIND."""
    prepare_age_session(conn)
    n = 0
    with conn.cursor() as cur:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            payload = [
                {"id": int(node_index), "aid": str(author_id)}
                for node_index, author_id in batch
            ]
            lit = _cypher_list_of_maps(payload)
            cur.execute(
                f"""
                SELECT * FROM cypher('{graph_name}', $$
                  UNWIND {lit} AS row
                  MERGE (a:Account {{id: row.id}})
                  SET a.author_account_id = row.aid
                  RETURN count(a)
                $$) AS (cnt agtype)
                """
            )
            n += len(batch)
            conn.commit()
    return n


def sync_edges_from_sql(
    conn: psycopg.Connection,
    *,
    run_id: str,
    graph_name: str,
    batch_size: int = 500,
) -> int:
    """Load SQL tdmec.edges for run_id into AGE as typed directed edges."""
    prepare_age_session(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT snapshot_id, relation_id, src_index, dst_index, count_raw, weight_log1p
            FROM tdmec.edges
            WHERE run_id = %s
            ORDER BY snapshot_id, relation_id, src_index, dst_index
            """,
            (run_id,),
        )
        rows = cur.fetchall()

    rid_lit = _esc(run_id)
    inserted = 0
    with conn.cursor() as cur:
        by_label: dict[str, list] = {lab: [] for lab in RELATION_LABELS.values()}
        for sid, rel_id, src, dst, count_raw, weight in rows:
            label = RELATION_LABELS.get(int(rel_id))
            if label is None:
                continue
            by_label[label].append(
                {
                    "src": int(src),
                    "dst": int(dst),
                    "sid": int(sid),
                    "count_raw": int(count_raw),
                    "weight_log1p": float(weight),
                }
            )

        for label, items in by_label.items():
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                lit = _cypher_list_of_maps(batch)
                cur.execute(
                    f"""
                    SELECT * FROM cypher('{graph_name}', $$
                      UNWIND {lit} AS row
                      MATCH (a:Account {{id: row.src}}), (b:Account {{id: row.dst}})
                      CREATE (a)-[e:{label} {{
                        snapshot_id: row.sid,
                        count_raw: row.count_raw,
                        weight_log1p: row.weight_log1p,
                        run_id: '{rid_lit}'
                      }}]->(b)
                      RETURN count(e)
                    $$) AS (cnt agtype)
                    """
                )
                inserted += len(batch)
                conn.commit()
    return inserted


def neighbor_query(
    conn: psycopg.Connection,
    graph_name: str,
    *,
    node_index: int,
    snapshot_id: int,
    relation: str = "reply",
) -> List[tuple]:
    """Return (dst_id, weight_log1p) for smoke tests / demos."""
    if relation not in RELATION_LABELS.values():
        raise ValueError(f"invalid relation label {relation}")
    prepare_age_session(conn)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT * FROM cypher('{graph_name}', $$
              MATCH (a:Account {{id: {int(node_index)}}})-[e:{relation}]->(b:Account)
              WHERE e.snapshot_id = {int(snapshot_id)}
              RETURN b.id, e.weight_log1p
            $$) AS (dst agtype, w agtype)
            """
        )
        return list(cur.fetchall())
