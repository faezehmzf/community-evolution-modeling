#!/usr/bin/env python3
"""Assemble TDMEC_INPUT_smoke_e2e from smoke graph + mock/smoke-64 pooled tensors."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmec_embeddings.assemble_tdmec_input import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
