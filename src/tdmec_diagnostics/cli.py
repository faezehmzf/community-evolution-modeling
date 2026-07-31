"""CLI entry point for Phase 2 diagnostics (Lightning AI Studio / controlled environments)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tdmec_diagnostics.adapters import AdapterConfigurationError
from tdmec_diagnostics.config import load_diagnostics_config
from tdmec_diagnostics.pipeline import run_diagnostics
from tdmec_diagnostics.workbook_io import UnsupportedSchemaError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tdmec-diagnostics",
        description=(
            "TDMEC Phase 2 privacy-safe data diagnostics. "
            "Produces evidence reports; does not certify QCAL-B01 / QDEDUP-B01. "
            "Never embeds credentials or private absolute paths in reports."
        ),
    )
    p.add_argument("--output-root", type=str, default="./artifacts")
    p.add_argument("--checkpoint-root", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument(
        "--resume-mode",
        type=str,
        choices=["resume", "restart"],
        default=None,
        help=(
            "resume (default) continues after the last transactionally "
            "committed source row; restart clears run state"
        ),
    )
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--provisional-start", type=str, default=None)
    p.add_argument("--provisional-end", type=str, default=None)
    p.add_argument("--source-format", type=str, choices=["xlsx", "synthetic"], default=None)
    p.add_argument("--dataset-a-adapter", type=str, default=None)
    p.add_argument("--dataset-b-adapter", type=str, default=None)
    p.add_argument(
        "--dataset-a-source",
        type=str,
        default=None,
        help="Source scheme for Dataset A (local:... | gdrive-anon:... | gdrive-api:...)",
    )
    p.add_argument("--dataset-b-source", type=str, default=None)
    p.add_argument(
        "--dataset-a-file",
        action="append",
        default=None,
        help="Explicit local Dataset A workbook (repeatable); bypasses source scheme",
    )
    p.add_argument("--dataset-b-file", action="append", default=None)
    p.add_argument(
        "--node-index-map",
        type=str,
        default=None,
        help="Path to frozen node_index_map.parquet (required for real mode)",
    )
    p.add_argument("--cache-root", type=str, default="/tmp/tdmec_cache")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_diagnostics_config(args.config)

    # Apply CLI overrides without mutating YAML unresolved markers.
    from dataclasses import replace

    overrides = {}
    if args.resume_mode is not None:
        overrides["resume_mode"] = args.resume_mode
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.provisional_start is not None:
        overrides["provisional_start_label"] = args.provisional_start
    if args.provisional_end is not None:
        overrides["provisional_end_label"] = args.provisional_end
    if args.source_format is not None:
        overrides["source_format"] = args.source_format
    if args.dataset_a_adapter is not None:
        overrides["dataset_a_adapter_id"] = args.dataset_a_adapter
    if args.dataset_b_adapter is not None:
        overrides["dataset_b_adapter_id"] = args.dataset_b_adapter
    if overrides:
        cfg = replace(cfg, **overrides)

    try:
        result = run_diagnostics(
            output_root=args.output_root,
            config=cfg,
            mode=args.mode,
            run_id=args.run_id,
            checkpoint_root=args.checkpoint_root,
            dataset_a_source=args.dataset_a_source,
            dataset_b_source=args.dataset_b_source,
            dataset_a_files=args.dataset_a_file,
            dataset_b_files=args.dataset_b_file,
            node_index_map=args.node_index_map,
            cache_root=args.cache_root,
        )
    except (
        AdapterConfigurationError,
        UnsupportedSchemaError,
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as exc:
        # Privacy-safe CLI errors: never print absolute paths from exceptions if present
        msg = str(exc)
        for token in ("/home/", "/Users/", "/workspace/", "/content/"):
            if token in msg:
                msg = "configuration/source error (details redacted for privacy)"
                break
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: configuration or runtime failure", file=sys.stderr)
        return 1

    summary = {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "layout": result.get("layout"),
        "complete": result.get("complete"),
        "real_data_executed": result.get("manifest", {}).get("real_data_executed"),
        "certification_claim": None,
    }
    # Ensure CLI stdout is privacy-safe
    out = json.dumps(summary, indent=2, sort_keys=True)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
