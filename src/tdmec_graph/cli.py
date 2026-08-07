"""CLI for Dataset A provisional graph builders."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dataset A provisional graph builder")
    ap.add_argument("--config", default="configs/dataset_a_graph.yaml")
    ap.add_argument("--dataset-a-source", default=None)
    ap.add_argument("--node-index-map", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--input-file",
        action="append",
        default=None,
        help="Restrict to one or more workbook basenames (repeatable; smoke tests)",
    )
    ap.add_argument(
        "--keep-work-db",
        action="store_true",
        help="Deprecated no-op (Postgres retains run state).",
    )
    ap.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: TDMEC_DATABASE_URL / DATABASE_URL)",
    )
    ap.add_argument(
        "--skip-age-sync",
        action="store_true",
        help="Skip Apache AGE graph sync (SQL + file exports only)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from tdmec_graph.config import load_graph_config
    from tdmec_graph.pipeline import ConfigIncompatibleError, GraphPipeline

    cfg = load_graph_config(args.config)
    source = (
        args.dataset_a_source
        or os.environ.get("DATASET_A_SOURCE")
        or os.environ.get("TDMEC_DATASET_A_SOURCE")
        or cfg.runtime.get("dataset_a_source")
    )
    node_map = (
        args.node_index_map
        or os.environ.get("NODE_INDEX_MAP_PATH")
        or os.environ.get("TDMEC_NODE_INDEX_MAP")
        or cfg.runtime.get("node_index_map_path")
    )
    output_root = (
        args.output_root
        or os.environ.get("GRAPH_OUTPUT_ROOT")
        or os.environ.get("TDMEC_OUTPUT_ROOT")
        or cfg.runtime.get("output_root")
    )
    cache_root = (
        args.cache_root
        or os.environ.get("DISCOVERY_CACHE_ROOT")
        or os.environ.get("TDMEC_CACHE_ROOT")
        or cfg.runtime.get("cache_root", "/tmp/tdmec_cache")
    )
    missing = [
        k
        for k, v in {
            "--dataset-a-source": source,
            "--node-index-map": node_map,
            "--output-root": output_root,
        }.items()
        if not v
    ]
    if missing:
        ap.error(f"missing required settings: {missing}")

    try:
        pipe = GraphPipeline(
            cfg,
            dataset_a_source=source,
            output_root=output_root,
            node_index_map_path=node_map,
            cache_root=cache_root,
            run_id=args.run_id,
            input_files=args.input_file,
            verbose=not args.quiet,
            database_url=args.database_url,
            sync_age=not args.skip_age_sync,
        )
        report = pipe.run(keep_work_db=args.keep_work_db)
    except ConfigIncompatibleError as e:
        print(f"CONFIG_INCOMPATIBLE: {e}", file=sys.stderr)
        return 4

    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "all_passed": report["all_passed"],
                "gates": report["gates"],
                "publish_summary": report.get("publish_summary", {}),
            },
            indent=2,
        )
    )
    return 0 if report["all_passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
