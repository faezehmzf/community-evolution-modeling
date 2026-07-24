"""Build the immutable frozen node-index map from Dataset A.

Canonical rule: the set of distinct author account ids (``user.id``) across all
Dataset A workbooks, sorted ascending as integers, is assigned node indices
0, 1, 2, … This map is immutable once published and is the single source of
truth for frozen-node reconciliation.

Read-only over Dataset A. Downloads each file one at a time and evicts it.

Usage:
    python scripts/build_node_index_map.py \
        --dataset-a-source "$DATASET_A_SOURCE" \
        --out artifacts/pilot/node_index_map.parquet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmec_discovery.cache import DownloadCache  # noqa: E402
from tdmec_discovery.sources import build_source  # noqa: E402
from tdmec_pilot.node_map import build_node_map_from_ids, save_node_map  # noqa: E402

_USER_ID_RE = re.compile(r"'id'\s*:\s*(\d+)")


def _extract_ids(path: str) -> set:
    from python_calamine import CalamineWorkbook

    wb = CalamineWorkbook.from_path(path)
    ws = wb.get_sheet_by_index(0)
    data = ws.to_python(skip_empty_area=True)
    header = [str(h) for h in data[0]]
    if "user" not in header:
        return set()
    ui = header.index("user")
    ids = set()
    for row in data[1:]:
        if ui < len(row) and isinstance(row[ui], str):
            m = _USER_ID_RE.search(row[ui])
            if m:
                ids.add(m.group(1))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-a-source", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-root", default="/tmp/tdmec_cache")
    ap.add_argument("--expected-count", type=int, default=16736)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    import os
    source_str = args.dataset_a_source or os.environ.get("DATASET_A_SOURCE")
    if not source_str:
        ap.error("provide --dataset-a-source or DATASET_A_SOURCE")

    source = build_source(source_str)
    cache = DownloadCache(args.cache_root)
    files = sorted((f for f in source.list_files() if f.ext == ".xlsx"),
                   key=lambda f: f.name)
    all_ids: set = set()
    for f in files:
        print(f"[node-map] {f.name} downloading...", flush=True)
        rec = cache.get(source, f, compute_hash=False)
        ids = _extract_ids(str(rec["path"]))
        all_ids |= ids
        if not args.keep_cache:
            cache.evict(f)
        print(f"[node-map] {f.name}: +{len(ids)} ids, cumulative {len(all_ids)}", flush=True)

    nm = build_node_map_from_ids(all_ids)
    save_node_map(nm, args.out)
    print(f"\nnode map: {len(nm)} authors -> indices {nm.min_index}..{nm.max_index}")
    if args.expected_count and len(nm) != args.expected_count:
        print(f"WARNING: expected {args.expected_count} authors, got {len(nm)}")
    else:
        print("count matches expected frozen population.")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
