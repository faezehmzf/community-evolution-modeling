"""Driver for the TDMEC read-only discovery pass.

Runs Phases 2-5 using the reusable tooling:
  * builds an immutable source inventory,
  * fully processes all 12 Dataset A files (one at a time, evicting each) to
    extract the complete frozen account-id universe + edge/temporal evidence,
  * deep-inspects a distributed SAMPLE of Dataset B files (the preamble forbids a
    full Dataset B download during discovery) and inventories all 70 by size,
  * reconciles Dataset B author accounts against the frozen population.

Outputs (large / potentially sensitive) go under the git-ignored run directory
``artifacts/discovery/runs/<run_id>/``. Only sanitized JSON summaries are meant
to be committed. No raw tweet text is ever written.

Usage:
    PYTHONPATH=src python scripts/run_discovery.py --b-sample 0,1,2,10,20,30,34,35,49,60,68,69
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmec_discovery.analyze import deep_inspect  # noqa: E402
from tdmec_discovery.cache import DownloadCache  # noqa: E402
from tdmec_discovery.config import load_config  # noqa: E402
from tdmec_discovery.fields import ordered_signature, schema_signature  # noqa: E402
from tdmec_discovery.run import RunContext  # noqa: E402
from tdmec_discovery.sources import build_source  # noqa: E402


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default="config/discovery.local.env")
    ap.add_argument("--b-sample", default="0,1,2,10,20,30,34,35,49,60,68,69")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.env_file)
    ctx = RunContext.create(cfg, run_id=args.run_id)
    run_dir = ctx.output_dir
    logs = run_dir / "logs"
    print("run_id:", ctx.run_id)
    print("run_dir:", run_dir)
    _write_json(run_dir / "runtime_environment.json", ctx.manifest["runtime_environment"])

    cache = DownloadCache(cfg.cache_root)
    src_a = build_source(cfg.dataset_a_source)
    src_b = build_source(cfg.dataset_b_source)

    files_a = {f.name: f for f in src_a.list_files()}
    files_b = {f.name: f for f in src_b.list_files()}

    # ---- Dataset A: full pass over all 12 xlsx ---------------------------------
    a_xlsx = sorted(n for n in files_a if n.endswith(".xlsx"))
    a_records = []
    frozen_author_ids: set[str] = set()
    a_total_rows = 0
    for name in a_xlsx:
        print(f"[A] {name} downloading...", flush=True)
        rec = cache.get(src_a, files_a[name])
        print(f"[A] {name} inspecting ({rec['size']/1e6:.0f} MB)...", flush=True)
        d = deep_inspect(str(rec["path"]), extract_edges=True)
        ids = d.pop("_author_ids")
        frozen_author_ids |= ids
        a_total_rows += d["n_data_rows"]
        d.update({"file": name, "downloaded_size": rec["size"], "sha256": rec["sha256"]})
        a_records.append(d)
        _write_json(logs / f"A_{name}.json", d)
        if not args.keep_cache:
            cache.evict(files_a[name])
        print(f"[A] {name}: rows={d['n_data_rows']} authors_cum={len(frozen_author_ids)}", flush=True)

    _write_json(run_dir / "dataset_a_schema_registry.json", {
        "files": [{"file": r["file"], "columns": r["columns"],
                   "schema_sig": schema_signature(r["columns"]),
                   "n_data_rows": r["n_data_rows"]} for r in a_records],
    })

    # ---- Dataset B: distributed sample ---------------------------------------
    b_idx = [int(x) for x in args.b_sample.split(",") if x.strip() != ""]
    b_names = [f"statuses-{i}.xlsx" for i in b_idx]
    b_records = []
    b_sample_author_ids: set[str] = set()
    schema_families = {}
    for name in b_names:
        if name not in files_b:
            print(f"[B] {name} NOT FOUND, skipping", flush=True)
            continue
        print(f"[B] {name} downloading...", flush=True)
        rec = cache.get(src_b, files_b[name])
        print(f"[B] {name} inspecting ({rec['size']/1e6:.0f} MB)...", flush=True)
        d = deep_inspect(str(rec["path"]), extract_edges=False)
        ids = d.pop("_author_ids")
        b_sample_author_ids |= ids
        sig = schema_signature(d["columns"])
        osig = ordered_signature(d["columns"])
        d.update({"file": name, "downloaded_size": rec["size"], "sha256": rec["sha256"],
                  "schema_sig": sig, "ordered_sig": osig})
        fam = schema_families.setdefault(sig, {"columns": d["columns"], "ordered_sigs": set(),
                                               "files": []})
        fam["ordered_sigs"].add(osig)
        fam["files"].append(name)
        b_records.append(d)
        _write_json(logs / f"B_{name}.json", d)
        if not args.keep_cache:
            cache.evict(files_b[name])
        print(f"[B] {name}: rows={d['n_data_rows']} authors={d['approx_unique_author_ids']} "
              f"ts=[{d['ts_min']}..{d['ts_max']}] schema={sig}", flush=True)

    for sig, fam in schema_families.items():
        fam["ordered_sigs"] = sorted(fam["ordered_sigs"])
    _write_json(run_dir / "dataset_b_schema_registry.json", {
        "families": schema_families,
        "n_families": len(schema_families),
        "inspected_files": [r["file"] for r in b_records],
    })

    # ---- Reconciliation -------------------------------------------------------
    inter = frozen_author_ids & b_sample_author_ids
    recon = {
        "frozen_author_id_count": len(frozen_author_ids),
        "expected_frozen_from_summary": 16736,
        "b_sample_files": [r["file"] for r in b_records],
        "b_sample_unique_author_ids": len(b_sample_author_ids),
        "b_sample_authors_in_frozen": len(inter),
        "b_sample_authors_outside_frozen": len(b_sample_author_ids - frozen_author_ids),
        "frozen_accounts_seen_in_b_sample": len(inter),
        "frozen_accounts_not_seen_in_b_sample": len(frozen_author_ids - b_sample_author_ids),
        "join_key": "user.id (numeric author account id embedded in user blob)",
    }
    _write_json(run_dir / "account_reconciliation.json", recon)

    # ---- Consolidated summary for report authoring ---------------------------
    summary = {
        "run_id": ctx.run_id,
        "dataset_a": {
            "n_files_xlsx": len(a_xlsx),
            "total_rows": a_total_rows,
            "records": a_records,
        },
        "dataset_b": {
            "n_files_visible_xlsx": len([n for n in files_b if n.endswith(".xlsx")]),
            "n_files_inspected": len(b_records),
            "records": b_records,
            "schema_families": schema_families,
        },
        "reconciliation": recon,
    }
    _write_json(run_dir / "discovery_consolidated.json", summary)
    ctx.mark_stage("phase2_inventory", "partial-sample")
    ctx.mark_stage("phase3_dataset_a", "complete")
    ctx.mark_stage("phase4_dataset_b", "sample-complete")
    ctx.mark_stage("phase5_reconciliation", "sample-complete")
    ctx.write_manifest()
    ctx.write_checksums()
    print("\nDONE. Frozen authors:", len(frozen_author_ids),
          "| B sample authors:", len(b_sample_author_ids),
          "| overlap:", len(inter))


if __name__ == "__main__":
    main()
