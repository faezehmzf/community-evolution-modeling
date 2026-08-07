#!/usr/bin/env python3
"""CLI: TDMEC-G smoke training on TDMEC_INPUT_smoke_e2e."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmec_model.train_g import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
