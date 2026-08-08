"""CLI for bounded benchmark, resumable shard jobs, and short finalization."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .config import EmbeddingConfigError, load_embedding_config
from .embedding_benchmark import EmbeddingBenchmarkError, run_bounded_benchmark
from .model_preflight import ModelPreflightError, verify_preflight_report
from .pilot import PilotPipelineError, run_tdmec_embedding_pilot
from .staged import StagedEmbeddingError, run_embedding_shard, verify_all_shards_complete


def _authorize(args: argparse.Namespace, execution_mode: str) -> None:
    if not args.authorize_real_model:
        raise StagedEmbeddingError("real Qwen execution requires --authorize-real-model")
    if execution_mode == "qwen_bounded_pilot" and not args.authorize_bounded_pilot:
        raise StagedEmbeddingError(
            "bounded pilot execution requires --authorize-bounded-pilot"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TDMEC bounded embedding job controller")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--authorize-real-model", action="store_true")
    parser.add_argument("--authorize-bounded-pilot", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--embedding-run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--max-node-rows", type=int)
    parser.add_argument("--max-event-rows", type=int)
    parser.add_argument("--output-shard-size", type=int)
    sub = parser.add_subparsers(dest="command", required=True)

    shard = sub.add_parser("embed-shard")
    shard.add_argument("--resume", action="store_true", help="Explicitly preserve and verify completed shards.")
    shard.add_argument("--modality", choices=("node_text", "event_text"), required=True)
    shard.add_argument("--shard-id", type=int, required=True)
    shard.add_argument("--num-shards", type=int, required=True)
    shard.add_argument("--wall-clock-hours", type=float, default=9.0)
    shard.add_argument("--graceful-stop-minutes", type=float, default=20.0)
    shard.add_argument("--max-output-batches", type=int, help="Stage-D interruption test hook")
    shard.add_argument(
        "--repair-corrupt-shards",
        action="store_true",
        help="Quarantine only checksum-invalid incomplete shards, then recompute them.",
    )

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--rows-per-modality", type=int, default=256)
    benchmark.add_argument("--output", required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--skip-export", action="store_true")
    finalize.add_argument("--package-root")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_embedding_config(args.config)
        config = replace(
            config,
            embedding_run_id=args.embedding_run_id or config.embedding_run_id,
            output_root=args.output_root or config.output_root,
            max_node_rows=args.max_node_rows or config.max_node_rows,
            max_event_rows=args.max_event_rows or config.max_event_rows,
            output_shard_size=args.output_shard_size or config.output_shard_size,
        )
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise StagedEmbeddingError("batch size must be positive")
            config = replace(config, encoder=replace(config.encoder, batch_size=args.batch_size))
            config.validate()
        config.validate()
        _authorize(args, config.execution_mode)
        verify_preflight_report(args.preflight_report, config)
        if args.command == "embed-shard":
            if not args.resume:
                raise StagedEmbeddingError("embed-shard requires explicit --resume")
            config = replace(config, resume=True, replace_incomplete=False)
            result = run_embedding_shard(
                config,
                modality=args.modality,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                preflight_report_path=args.preflight_report,
                wall_clock_budget_seconds=args.wall_clock_hours * 3600.0,
                graceful_stop_seconds=args.graceful_stop_minutes * 60.0,
                max_output_batches=args.max_output_batches,
                repair_corrupt_shards=args.repair_corrupt_shards,
            )
        elif args.command == "benchmark":
            result = run_bounded_benchmark(
                config,
                preflight_report_path=args.preflight_report,
                rows_per_modality=args.rows_per_modality,
                output_path=args.output,
            )
        else:
            config = replace(config, resume=True, replace_incomplete=False)
            coverage = verify_all_shards_complete(config)
            result = run_tdmec_embedding_pilot(
                config,
                authorize_real_model=True,
                authorize_bounded_pilot=args.authorize_bounded_pilot,
                skip_export=args.skip_export,
                package_root=args.package_root,
                preflight_report_path=args.preflight_report,
            )
            result["pre_finalization_shard_coverage"] = coverage
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except (
        OSError,
        ValueError,
        EmbeddingConfigError,
        EmbeddingBenchmarkError,
        ModelPreflightError,
        PilotPipelineError,
        StagedEmbeddingError,
    ) as exc:
        print(f"REFUSED_OR_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
