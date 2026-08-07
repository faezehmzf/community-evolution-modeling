"""Dataset A provisional graph pipeline (Postgres events → edges → AGE → X_struct)."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tdmec import constants as C
from tdmec_db import age_graph, repositories as repo
from tdmec_db.connection import connect, resolve_database_url
from tdmec_diagnostics.adapters import load_node_universe_lookup
from tdmec_discovery.cache import DownloadCache
from tdmec_discovery.hashing import sha256_file
from tdmec_discovery.sources import build_source
from tdmec_pilot.snapshots import boundary_table
from tdmec_pilot.storage import atomic_write_json

from .aggregate import edges_to_records
from .config import GraphConfig
from .events import RowAccounting, stream_workbook_events
from .features import build_structural_tensors
from .publish import GraphManifest, GraphRunLayout, make_run_id, write_parquet_atomic


class ConfigIncompatibleError(RuntimeError):
    pass


class GraphPipeline:
    def __init__(
        self,
        config: GraphConfig,
        *,
        dataset_a_source: str,
        output_root: str,
        node_index_map_path: str,
        cache_root: str = "/tmp/tdmec_cache",
        run_id: Optional[str] = None,
        input_files: Optional[List[str]] = None,
        verbose: bool = True,
        database_url: Optional[str] = None,
        sync_age: bool = True,
    ):
        self.cfg = config
        self.source_str = dataset_a_source
        self.output_root = output_root
        self.node_index_map_path = node_index_map_path
        self.cache_root = Path(cache_root)
        self.config_hash = config.config_hash()
        self.run_id = run_id or make_run_id(self.config_hash)
        self.layout = GraphRunLayout(output_root, self.run_id).ensure()
        self.manifest = GraphManifest(self.layout)
        self.input_files_filter = input_files
        self.verbose = verbose
        self.node_lookup = None
        self.logger = self._make_logger()
        self._local_inputs: Dict[str, Path] = {}
        self.database_url = resolve_database_url(database_url)
        self.sync_age = sync_age
        self.age_graph_name = age_graph.sanitize_graph_name(self.run_id)

    def _make_logger(self) -> logging.Logger:
        lg = logging.getLogger(f"tdmec_graph.{self.run_id}")
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        fh = logging.FileHandler(self.layout.root / "logs" / "graph.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
        if self.verbose:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(sh)
        return lg

    def _progress(self, msg: str) -> None:
        self.logger.info(msg)

    def _connect(self):
        return connect(self.database_url)

    # ------------------------------------------------------------------ setup
    def setup(self) -> dict:
        self.manifest.load_or_init(self.run_id, self.config_hash, self.cfg.canonical)
        if self.manifest.data.get("config_hash") not in (None, self.config_hash):
            if self.manifest.data.get("config_hash") != self.config_hash:
                raise ConfigIncompatibleError(
                    f"run config hash {self.manifest.data.get('config_hash')} "
                    f"!= current {self.config_hash}"
                )
        self.node_lookup = load_node_universe_lookup(
            self.node_index_map_path,
            expected_count=int(self.cfg.canonical["frozen_node_count"]),
        )
        info = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "n_nodes": self.node_lookup.n_nodes,
            "node_map_sha256": sha256_file(self.node_index_map_path),
            "artifact_status": "PROVISIONAL",
            "age_graph_name": self.age_graph_name,
            "database": "postgres+age",
        }
        with self._connect() as conn:
            repo.upsert_run(
                conn,
                run_id=self.run_id,
                pipeline="tdmec_graph",
                config_hash=self.config_hash,
                status="UNVALIDATED",
                artifact_status="PROVISIONAL",
                age_graph_name=self.age_graph_name,
                meta={"node_map_sha256": info["node_map_sha256"]},
            )
            # Load frozen node universe into SQL (shared across runs).
            node_rows = [
                (int(idx), str(aid))
                for aid, idx in self.node_lookup.mapping.items()
            ]
            repo.load_nodes(conn, node_rows)
            snap_rows = [
                {
                    "snapshot_id": b.snapshot_id,
                    "quarter_label": b.label,
                    "start_utc": b.start_utc,
                    "end_utc_exclusive": b.end_utc_exclusive,
                    "status": "provisional",
                }
                for b in boundary_table()
            ]
            repo.load_snapshots(conn, snap_rows)

        self.manifest.data["node_map_sha256"] = info["node_map_sha256"]
        self.manifest.data["age_graph_name"] = self.age_graph_name
        self.manifest.set_stage("setup", "complete", info)
        self.manifest.flush()
        self._progress(f"setup ok run_id={self.run_id} n_nodes={info['n_nodes']}")
        return info

    def verify_sources(self) -> dict:
        source = build_source(self.source_str)
        files = {f.name: f for f in source.list_files() if f.name.endswith(".xlsx")}
        names = sorted(files.keys())
        if self.input_files_filter:
            names = [n for n in names if n in set(self.input_files_filter)]
        if not names:
            raise FileNotFoundError("no Dataset A xlsx inputs matched")
        cache = DownloadCache(self.cache_root)
        result = {"files": {}, "n_files": len(names)}
        with self._connect() as conn:
            for name in names:
                rec = cache.get(source, files[name], compute_hash=True)
                prev = self.manifest.data.get("source_checksums", {}).get(name)
                if prev and prev != rec["sha256"]:
                    raise RuntimeError(f"input checksum changed for {name}")
                self.manifest.data.setdefault("source_checksums", {})[name] = rec["sha256"]
                self._local_inputs[name] = Path(rec["path"])
                repo.upsert_source_file(
                    conn,
                    run_id=self.run_id,
                    source_file=name,
                    sha256=rec["sha256"],
                    size_bytes=rec.get("size"),
                )
                result["files"][name] = {
                    "available": True,
                    "sha256": rec["sha256"],
                    "size": rec["size"],
                }
        self.manifest.set_stage("source_verification", "complete", {"n_files": len(names)})
        self.manifest.flush()
        self._progress(f"sources verified files={len(names)}")
        return result

    # ------------------------------------------------------------------ process
    def process_files(self) -> dict:
        total = RowAccounting()
        n_inserted_events = 0
        with self._connect() as conn:
            for fi, (name, path) in enumerate(sorted(self._local_inputs.items()), start=1):
                self._progress(
                    f"stage=events file={name} progress={fi}/{len(self._local_inputs)}"
                )
                for file_name, chunk_idx, results, acc in stream_workbook_events(
                    path,
                    node_lookup=self.node_lookup,
                    source_file_name=name,
                    chunk_size=self.cfg.chunk_size,
                ):
                    if repo.chunk_done(conn, self.run_id, file_name, chunk_idx):
                        self._progress(
                            f"resume skip file={file_name} chunk={chunk_idx}"
                        )
                        continue
                    event_rows = []
                    tweet_rows = []
                    for authored, events in results:
                        if authored is not None:
                            tweet_rows.append(
                                (
                                    authored.source_file,
                                    authored.source_row_number,
                                    authored.snapshot_id,
                                    authored.source_idx,
                                )
                            )
                        for ev in events:
                            event_rows.append(
                                (
                                    ev.signature,
                                    ev.snapshot_id,
                                    ev.relation_id,
                                    ev.source_idx,
                                    ev.target_idx,
                                    ev.cleaned_text,
                                    ev.text_hash,
                                    ev.text_quality,
                                    ev.source_file,
                                    ev.source_row_number,
                                )
                            )
                    if tweet_rows:
                        repo.insert_authored_tweets_ignore(conn, self.run_id, tweet_rows)
                    inserted = repo.insert_events_ignore(conn, self.run_id, event_rows)
                    n_inserted_events += inserted
                    repo.mark_chunk(
                        conn,
                        run_id=self.run_id,
                        source_file=file_name,
                        chunk_idx=chunk_idx,
                        rows_inspected=acc.rows_inspected,
                        events_inserted=inserted,
                    )
                    total.add(acc)
                    self._progress(
                        f"file={file_name} chunk={chunk_idx} "
                        f"rows={acc.rows_inspected} new_events={inserted}"
                    )

        accounting = {
            **total.as_dict(),
            "distinct_events_inserted": n_inserted_events,
        }
        self.manifest.data["accounting"] = accounting
        self.manifest.set_stage("process_files", "complete", accounting)
        self.manifest.flush()
        return accounting

    # ------------------------------------------------------------------ publish
    def publish(self) -> dict:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with self._connect() as conn:
            events_path = self.layout.root / "events" / "canonical_events.parquet"
            events_path.parent.mkdir(parents=True, exist_ok=True)
            cols = [
                "snapshot_id",
                "relation_id",
                "source_idx",
                "target_idx",
                "cleaned_text",
                "text_hash",
                "text_quality",
                "source_file",
                "source_row_number",
                "signature",
            ]
            schema = pa.schema(
                [
                    ("snapshot_id", pa.int32()),
                    ("relation_id", pa.int8()),
                    ("source_idx", pa.int32()),
                    ("target_idx", pa.int32()),
                    ("cleaned_text", pa.string()),
                    ("text_hash", pa.string()),
                    ("text_quality", pa.string()),
                    ("source_file", pa.string()),
                    ("source_row_number", pa.int64()),
                    ("signature", pa.string()),
                ]
            )
            tmp = events_path.with_suffix(".parquet.tmp")
            writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            n_events = 0
            for batch in repo.stream_events(conn, self.run_id):
                df = pd.DataFrame(batch, columns=cols)
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                writer.write_table(table)
                n_events += len(df)
            writer.close()
            tmp.replace(events_path)
            ev_sha = sha256_file(events_path)
            self.manifest.add_output(
                str(events_path.relative_to(self.layout.root)),
                events_path.stat().st_size,
                ev_sha,
                rows=n_events,
            )

            n_edges = repo.rebuild_edges_from_events(conn, self.run_id)
            edge_counts = repo.fetch_edge_counts(conn, self.run_id)
            edge_rows = edges_to_records(edge_counts)
            by_part: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
            for r in edge_rows:
                by_part[(r["snapshot_id"], r["relation_id"])].append(r)
            n_edges_files = 0
            for (sid, rid), rows in sorted(by_part.items()):
                part = self.layout.edge_part(sid, rid)
                df = pd.DataFrame(rows)
                sha = write_parquet_atomic(df, part)
                n_edges_files += len(df)
                self.manifest.add_output(
                    str(part.relative_to(self.layout.root)),
                    part.stat().st_size,
                    sha,
                    rows=len(df),
                )

            tweet_counts = repo.fetch_tweet_counts(conn, self.run_id)
            x_struct, mask = build_structural_tensors(edge_counts, tweet_counts)
            x_path = self.layout.root / "X_struct.npy"
            m_path = self.layout.root / "struct_active_mask.npy"
            np.save(x_path, x_struct)
            np.save(m_path, mask)
            self.manifest.add_output(
                "X_struct.npy", x_path.stat().st_size, sha256_file(x_path)
            )
            self.manifest.add_output(
                "struct_active_mask.npy", m_path.stat().st_size, sha256_file(m_path)
            )

            names_path = self.layout.root / "struct_feature_names.json"
            atomic_write_json(
                names_path,
                {
                    "schema_version": C.STRUCT_FEATURE_SCHEMA_VERSION,
                    "feature_names": list(C.STRUCT_FEATURE_NAMES),
                    "f_struct": C.F_STRUCT,
                },
            )
            self.manifest.add_output(
                "struct_feature_names.json",
                names_path.stat().st_size,
                sha256_file(names_path),
            )

            cal = [
                {
                    "snapshot_id": b.snapshot_id,
                    "quarter_label": b.label,
                    "start_boundary_utc": b.start_utc.isoformat(),
                    "end_boundary_utc": b.end_utc_exclusive.isoformat(),
                    "boundary_convention": C.BOUNDARY_CONVENTION,
                    "status": "provisional",
                }
                for b in boundary_table()
            ]
            cal_path = self.layout.root / "snapshot_calendar.json"
            atomic_write_json(
                cal_path,
                {
                    "frequency": C.SNAPSHOT_FREQUENCY,
                    "calendar_certification_status": "PROVISIONAL_DIAGNOSTIC_ONLY",
                    "snapshots": cal,
                },
            )
            self.manifest.add_output(
                "snapshot_calendar.json",
                cal_path.stat().st_size,
                sha256_file(cal_path),
            )

            dup_path = self.layout.root / "dataset_a_duplicate_report.parquet"
            dup_df = pd.DataFrame(
                [
                    {
                        "metric": "distinct_events",
                        "value": n_events,
                        "status": "PROVISIONAL",
                    },
                    {
                        "metric": "distinct_edges",
                        "value": n_edges_files,
                        "status": "PROVISIONAL",
                    },
                    {
                        "metric": "authored_tweets",
                        "value": sum(tweet_counts.values()),
                        "status": "PROVISIONAL",
                    },
                ]
            )
            dup_sha = write_parquet_atomic(dup_df, dup_path)
            self.manifest.add_output(
                "dataset_a_duplicate_report.parquet",
                dup_path.stat().st_size,
                dup_sha,
                rows=len(dup_df),
            )

            age_stats = {"synced": False, "n_vertices": 0, "n_edges": 0}
            if self.sync_age:
                self._progress(f"AGE sync graph={self.age_graph_name}")
                age_graph.ensure_graph(conn, self.age_graph_name, reset=True)
                nodes = repo.list_nodes(conn)
                n_v = age_graph.upsert_account_vertices(
                    conn, self.age_graph_name, nodes
                )
                n_e = age_graph.sync_edges_from_sql(
                    conn, run_id=self.run_id, graph_name=self.age_graph_name
                )
                age_stats = {
                    "synced": True,
                    "graph_name": self.age_graph_name,
                    "n_vertices": n_v,
                    "n_edges": n_e,
                }
                repo.upsert_run(
                    conn,
                    run_id=self.run_id,
                    pipeline="tdmec_graph",
                    config_hash=self.config_hash,
                    status="PUBLISHED",
                    artifact_status="PROVISIONAL",
                    age_graph_name=self.age_graph_name,
                    meta={"age": age_stats, "n_events": n_events, "n_edges": n_edges},
                )

            summary = {
                "n_events": n_events,
                "n_edges": n_edges_files,
                "n_edges_sql": n_edges,
                "n_active_node_slots": int(mask.sum()),
                "x_struct_shape": list(x_struct.shape),
                "mask_shape": list(mask.shape),
                "age": age_stats,
            }
            self.manifest.data["publish_summary"] = summary
            self.manifest.data["age_graph_name"] = self.age_graph_name
            self.manifest.set_stage("publish", "complete", summary)
            self.manifest.flush()
            self._progress(
                f"publish ok events={n_events} edges={n_edges_files} "
                f"active_slots={summary['n_active_node_slots']} age={age_stats}"
            )
            return summary

    def validate(self) -> dict:
        gates = {}
        x = np.load(self.layout.root / "X_struct.npy")
        mask = np.load(self.layout.root / "struct_active_mask.npy")
        gates["x_struct_shape"] = list(x.shape) == [
            C.PROVISIONAL_SNAPSHOT_COUNT,
            C.N_NODES,
            C.F_STRUCT,
        ]
        gates["mask_shape"] = list(mask.shape) == [
            C.PROVISIONAL_SNAPSHOT_COUNT,
            C.N_NODES,
        ]
        gates["x_struct_dtype"] = str(x.dtype) == "float32"
        gates["x_struct_finite"] = bool(np.isfinite(x).all())
        gates["inactive_rows_zero"] = bool(np.all(x[~mask] == 0))
        gates["n_nodes"] = self.node_lookup.n_nodes == C.N_NODES

        edge_parts = list(self.layout.root.glob("edges/**/*.parquet"))
        weight_ok = True
        self_loop = 0
        for p in edge_parts:
            df = pd.read_parquet(p)
            if len(df) == 0:
                continue
            if (df["src_index"] == df["dst_index"]).any():
                self_loop += int((df["src_index"] == df["dst_index"]).sum())
            expected = np.log1p(df["count_raw"].astype(float))
            if not np.allclose(df["weight_log1p"].astype(float), expected, atol=1e-6):
                weight_ok = False
        gates["weight_log1p_ok"] = weight_ok
        gates["no_self_loops"] = self_loop == 0
        gates["has_events"] = (
            self.manifest.data.get("publish_summary", {}).get("n_events", 0) > 0
        )
        gates["artifact_status_provisional"] = (
            self.manifest.data.get("artifact_status") == "PROVISIONAL"
        )
        gates["not_certified"] = self.manifest.data.get("certification_status") != "CERTIFIED"

        age_ok = True
        if self.sync_age:
            age_info = self.manifest.data.get("publish_summary", {}).get("age", {})
            age_ok = bool(age_info.get("synced")) and int(age_info.get("n_edges", 0)) > 0
            if age_ok:
                with self._connect() as conn:
                    # Spot-check: pick first SQL edge and Cypher-query it.
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT src_index, snapshot_id, relation_id, dst_index
                            FROM tdmec.edges WHERE run_id=%s LIMIT 1
                            """,
                            (self.run_id,),
                        )
                        row = cur.fetchone()
                    if row is None:
                        age_ok = False
                    else:
                        src, sid, rid, _dst = row
                        rel = age_graph.RELATION_LABELS[int(rid)]
                        hits = age_graph.neighbor_query(
                            conn,
                            self.age_graph_name,
                            node_index=int(src),
                            snapshot_id=int(sid),
                            relation=rel,
                        )
                        age_ok = len(hits) > 0
        gates["age_neighbor_query_ok"] = age_ok

        report = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "gates": {k: bool(v) for k, v in gates.items()},
            "all_passed": all(bool(v) for v in gates.values()),
            "publish_summary": self.manifest.data.get("publish_summary", {}),
            "accounting": self.manifest.data.get("accounting", {}),
        }
        atomic_write_json(self.layout.root / "validation_report.json", report)
        self.manifest.add_output(
            "validation_report.json",
            (self.layout.root / "validation_report.json").stat().st_size,
            sha256_file(self.layout.root / "validation_report.json"),
        )
        self.manifest.set_stage(
            "validate", "complete" if report["all_passed"] else "failed", report["gates"]
        )
        self.manifest.flush()
        self.manifest.write_checksums()
        self._progress(f"validate all_passed={report['all_passed']}")
        return report

    def cleanup_work_db(self) -> None:
        """No-op: working state lives in Postgres (kept for CLI compatibility)."""
        work = self.layout.root / "work"
        if work.is_dir() and not any(work.iterdir()):
            work.rmdir()

    def run(self, *, keep_work_db: bool = False) -> dict:
        self.setup()
        self.verify_sources()
        self.process_files()
        self.publish()
        report = self.validate()
        if report["all_passed"] and not keep_work_db:
            self.cleanup_work_db()
        return report
