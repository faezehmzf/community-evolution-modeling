"""TDMEC data-discovery tooling.

Platform-neutral, read-only discovery utilities for inspecting Dataset A and
Dataset B from a configurable source (local filesystem or Google Drive) without
mutating any source file.

The package is intentionally dependency-light at import time: heavy optional
dependencies (gdown, google-api-python-client, pandas, openpyxl) are imported
lazily inside the functions that need them so that, e.g., unit tests exercising
the local-filesystem adapter do not require network libraries.
"""

__version__ = "0.1.0"
