#!/usr/bin/env python3
"""Validate a TDMEC_INPUT package before model training.

Usage:
  python scripts/validate_tdmec_input.py --package-root /path/to/TDMEC_INPUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate TDMEC_INPUT package integrity")
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--expected-dimension", type=int)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args(argv)

    # Allow running from repo without install when PYTHONPATH=src
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from tdmec_embeddings.validation import validate_tdmec_input_package

    report = validate_tdmec_input_package(
        args.package_root,
        expected_dimension=args.expected_dimension,
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
