"""CLI entry point for the Dataset B controlled preprocessing pilot.

Colab-independent: the same command runs on a normal Linux filesystem (pass a
local ``--output-root``) or on Colab (pass a mounted Drive path). It processes
exactly the two configured input files, supports resume, and never touches the
other 68 files, embeddings, or training.

Example:
    python scripts/run_dataset_b_pilot.py \
        --config configs/dataset_b_pilot.yaml \
        --dataset-b-source "$DATASET_B_SOURCE" \
        --node-index-map /path/to/node_index_map.parquet \
        --output-root /content/drive/MyDrive/TDMEC_PROJECT_OUTPUTS \
        --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmec_pilot.config import load_pilot_config  # noqa: E402
from tdmec_pilot.pipeline import ConfigIncompatibleError, PilotPipeline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dataset B controlled pilot")
    ap.add_argument("--config", default="configs/dataset_b_pilot.yaml")
    ap.add_argument("--dataset-b-source", default=None,
                    help="local dir | gdrive-anon:<id> | gdrive-api:<id> "
                         "(else DATASET_B_SOURCE env or config runtime.dataset_b_source)")
    ap.add_argument("--node-index-map", default=None,
                    help="path to node_index_map.parquet (else NODE_INDEX_MAP_PATH env "
                         "or config runtime.node_index_map_path)")
    ap.add_argument("--output-root", default=None,
                    help="persistent output root (else PILOT_OUTPUT_ROOT env "
                         "or config runtime.output_root)")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--run-id", default=None, help="reuse an existing run id to resume")
    ap.add_argument("--resume", action="store_true", help="resume the given --run-id")
    args = ap.parse_args(argv)

    cfg = load_pilot_config(args.config)

    source = (args.dataset_b_source or os.environ.get("DATASET_B_SOURCE")
              or cfg.runtime.get("dataset_b_source"))
    node_map = (args.node_index_map or os.environ.get("NODE_INDEX_MAP_PATH")
                or cfg.runtime.get("node_index_map_path"))
    output_root = (args.output_root or os.environ.get("PILOT_OUTPUT_ROOT")
                   or cfg.runtime.get("output_root"))
    cache_root = (args.cache_root or os.environ.get("DISCOVERY_CACHE_ROOT")
                  or cfg.runtime.get("cache_root", "/tmp/tdmec_cache"))

    missing = [k for k, v in {"--dataset-b-source": source, "--node-index-map": node_map,
                              "--output-root": output_root}.items() if not v]
    if missing:
        ap.error(f"missing required settings: {missing}")

    if args.resume and not args.run_id:
        ap.error("--resume requires --run-id")

    try:
        pipe = PilotPipeline(cfg, dataset_b_source=source, output_root=output_root,
                             node_index_map_path=node_map, cache_root=cache_root,
                             run_id=args.run_id)
        report = pipe.run()
    except ConfigIncompatibleError as e:
        print(f"CONFIG_INCOMPATIBLE: {e}", file=sys.stderr)
        sys.exit(4)

    print(json.dumps({"run_id": report["run_id"], "all_passed": report["all_passed"],
                      "gates": report["gates"], "accounting": report["accounting"]}, indent=2))
    sys.exit(0 if report["all_passed"] else 3)


if __name__ == "__main__":
    main()
