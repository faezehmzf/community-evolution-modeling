#!/usr/bin/env python3
"""Lightning AI Studio / controlled-environment Phase 2 diagnostics runner.

Default mode is synthetic fixtures. Real-data execution requires authorized
local Dataset A/B paths (or optional Drive adapters) supplied at runtime.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without editable install when src/ is on PYTHONPATH
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tdmec_diagnostics.cli import main as cli_main  # noqa: E402


def main() -> int:
    # Reuse CLI; keep a thin script entry for notebooks/docs.
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
