"""Command-line entry point for the discovery tooling.

Examples
--------
    python -m tdmec_discovery list --dataset A
    python -m tdmec_discovery verify-access --out artifacts/discovery
    python -m tdmec_discovery inspect-file --dataset B --name statuses-0.xlsx
    python -m tdmec_discovery check-drive-write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .sources import build_source


def _resolve_source(cfg, dataset: str):
    src = cfg.dataset_a_source if dataset.upper() == "A" else cfg.dataset_b_source
    if not src:
        raise SystemExit(
            f"DATASET_{dataset.upper()}_SOURCE is not configured. "
            "Set it in the environment or config/discovery.local.env."
        )
    return build_source(src)


def cmd_list(args):
    cfg = load_config(args.env_file)
    source = _resolve_source(cfg, args.dataset)
    files = source.list_files()
    print(f"dataset={args.dataset} file_count={len(files)}")
    for f in files:
        print(f"  {f.relative_id}\text={f.ext}\tsize={f.size}")


def cmd_verify_access(args):
    from .verify import verify_access

    cfg = load_config(args.env_file)
    report = verify_access(cfg, out_dir=Path(args.out))
    print(json.dumps(report, indent=2))


def cmd_inspect_file(args):
    from .cache import DownloadCache
    from .xlsx_inspect import inspect_sheet

    cfg = load_config(args.env_file)
    source = _resolve_source(cfg, args.dataset)
    files = {f.name: f for f in source.list_files()}
    if args.name not in files:
        raise SystemExit(f"file not found in dataset {args.dataset}: {args.name}")
    cache = DownloadCache(cfg.cache_root)
    rec = cache.get(source, files[args.name])
    insp = inspect_sheet(rec["path"], dtype_scan_rows=args.scan_rows)
    print(json.dumps(insp.to_summary(include_samples=False), indent=2, default=str))
    if args.evict:
        cache.evict(files[args.name])


def cmd_check_drive_write(args):
    from .verify import verify_drive_write

    cfg = load_config(args.env_file)
    result = verify_drive_write(cfg)
    print(json.dumps(result, indent=2))
    if not result.get("write_access_confirmed"):
        sys.exit(3)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tdmec_discovery")
    p.add_argument("--env-file", default=None, help="optional dotenv file")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list files in a dataset source")
    pl.add_argument("--dataset", required=True, choices=["A", "B", "a", "b"])
    pl.set_defaults(func=cmd_list)

    pv = sub.add_parser("verify-access", help="Phase 0 byte-level access verification")
    pv.add_argument("--out", default="artifacts/discovery")
    pv.set_defaults(func=cmd_verify_access)

    pi = sub.add_parser("inspect-file", help="download + inspect one workbook")
    pi.add_argument("--dataset", required=True, choices=["A", "B", "a", "b"])
    pi.add_argument("--name", required=True)
    pi.add_argument("--scan-rows", type=int, default=5000)
    pi.add_argument("--evict", action="store_true", help="delete local copy after")
    pi.set_defaults(func=cmd_inspect_file)

    pc = sub.add_parser("check-drive-write", help="verify output Drive write access")
    pc.set_defaults(func=cmd_check_drive_write)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
