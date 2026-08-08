"""Transfer-ready CLI for mock, Qwen preflight, and bounded pilot modes.

IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .config import EmbeddingConfigError, load_embedding_config
from .pipeline import EmbeddingPipelineError, run_embedding_pipeline
from .sampling import SamplingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TDMEC file-backed embedding stage "
            "(IMPLEMENTED_NOT_EXECUTED / TRANSFER_PENDING_VALIDATION)"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--embedding-run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--node-source-root")
    parser.add_argument("--event-source-root")
    parser.add_argument("--node-source-run-id")
    parser.add_argument("--event-source-run-id")
    parser.add_argument("--max-node-rows", type=int)
    parser.add_argument("--max-event-rows", type=int)
    parser.add_argument("--input-batch-size", type=int)
    parser.add_argument("--output-shard-size", type=int)
    parser.add_argument(
        "--output-dimension",
        type=int,
        help="Override encoder output dimension (mock dimension or provisional D_text).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--replace-incomplete",
        action="store_true",
        help=(
            "Delete an interrupted/failed run directory with the same embedding_run_id "
            "and start clean. Refuses to delete COMPLETED outputs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rejected by policy; completed outputs must not be overwritten.",
    )
    parser.add_argument("--authorize-real-model", action="store_true")
    parser.add_argument("--preflight-report")
    parser.add_argument("--authorize-bounded-pilot", action="store_true")
    return parser


def _overrides(args: argparse.Namespace):
    config = load_embedding_config(args.config)
    node = replace(
        config.node_source,
        artifact_root=args.node_source_root or config.node_source.artifact_root,
        run_id=args.node_source_run_id or config.node_source.run_id,
    )
    event = replace(
        config.event_source,
        artifact_root=args.event_source_root or config.event_source.artifact_root,
        run_id=args.event_source_run_id or config.event_source.run_id,
    )
    encoder = config.encoder
    if args.output_dimension is not None:
        encoder = replace(encoder, output_dimension=int(args.output_dimension))
    config = replace(
        config,
        embedding_run_id=args.embedding_run_id or config.embedding_run_id,
        output_root=args.output_root or config.output_root,
        node_source=node,
        event_source=event,
        max_node_rows=args.max_node_rows or config.max_node_rows,
        max_event_rows=args.max_event_rows or config.max_event_rows,
        input_batch_size=args.input_batch_size or config.input_batch_size,
        output_shard_size=args.output_shard_size or config.output_shard_size,
        dry_run=bool(args.dry_run or config.dry_run),
        resume=bool(args.resume or config.resume),
        replace_incomplete=bool(
            getattr(args, "replace_incomplete", False) or config.replace_incomplete
        ),
        force=bool(args.force or config.force),
        encoder=encoder,
    )
    config.validate()
    return config


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _overrides(args)
        result = run_embedding_pipeline(
            config,
            authorize_real_model=args.authorize_real_model,
            authorize_bounded_pilot=args.authorize_bounded_pilot,
            preflight_report_path=args.preflight_report,
        )
    except (ValueError, OSError, EmbeddingPipelineError, EmbeddingConfigError, SamplingError) as exc:
        print(f"REFUSED_OR_FAILED: {exc}", file=sys.stderr)
        return 2
    # Privacy-safe: pipeline result contains hashes/counts only.
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
