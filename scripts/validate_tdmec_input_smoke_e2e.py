#!/usr/bin/env python3
"""Validate TDMEC_INPUT_smoke_e2e (or any smoke-layout package) for DataLoader readiness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmec_embeddings.assemble_tdmec_input import validate_smoke_shapes  # noqa: E402
from tdmec_embeddings.validation import validate_tdmec_input_package  # noqa: E402
from tdmec.hashing import sha256_file  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate TDMEC_INPUT smoke E2E package")
    parser.add_argument(
        "--package-root",
        default=(
            "/teamspace/studios/this_studio/TDMEC_PROJECT_OUTPUTS/"
            "tdmec_input/TDMEC_INPUT_smoke_e2e"
        ),
    )
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.package_root)
    shape_report = validate_smoke_shapes(root)
    pkg_report = validate_tdmec_input_package(
        root,
        expected_dimension=shape_report["shapes"].get("D_text"),
        verify_checksums=not args.skip_checksums,
    )

    # Refresh checksum verification detail for the smoke-specific report
    if not args.skip_checksums and (root / "checksums" / "package_checksums.json").is_file():
        recorded = json.loads(
            (root / "checksums" / "package_checksums.json").read_text(encoding="utf-8")
        )
        mismatched = [
            rel
            for rel, digest in recorded.items()
            if rel != "checksums/package_checksums.json"
            and (not (root / rel).is_file() or sha256_file(root / rel) != digest)
        ]
        pkg_report.setdefault("checks", {})["checksum_mismatches"] = mismatched

    combined = {
        "package_root": root.as_posix(),
        "passed": bool(shape_report["passed"] and pkg_report["passed"]),
        "shape_validation": shape_report,
        "package_validation": {
            "passed": pkg_report["passed"],
            "failures": pkg_report.get("failures"),
            "shapes": pkg_report.get("checks", {}).get("text_embeddings", {}).get("shapes")
            if isinstance(pkg_report.get("checks", {}).get("text_embeddings"), dict)
            else None,
        },
    }
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0 if combined["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
