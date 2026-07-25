"""Dataset B controlled pilot pipeline (stages P00-P12).

Colab-independent. Processes exactly the configured input files, one file and one
chunk at a time, with per-chunk checkpoints, atomic writes, resume, and hard
validation gates. Never downloads the other 68 files, never embeds, never trains.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tdmec_discovery.analyze import _read_sheet
from tdmec_discovery.hashing import sha256_file

from .config import PilotConfig
from .dedup import DuplicateTracker, content_hash
from .identifiers import normalize_tweet_id
from .node_map import NodeMap, load_node_map
from .schema import (NORMALIZED_COLUMNS, RECORD_STATUSES, records_to_frame,
                     validate_columns)
from .snapshots import assign_snapshot, boundary_table
from .storage import (Manifest, RunLayout, atomic_write_json, git_short_hash,
                      make_run_id, write_parquet_atomic)
from .text_quality import normalize_text
from .timestamps import parse_created_at
from .user_blob import parse_user_blob


def _to_int(v):
    if v is None or v == "":
        return None
    try:
        if isinstance(v, float):
            return int(v)
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


class ConfigIncompatibleError(RuntimeError):
    pass


class PilotPipeline:
    def __init__(self, config: PilotConfig, *, dataset_b_source: str,
                 output_root: str, node_index_map_path: str,
                 cache_root: str = "/tmp/tdmec_cache",
                 run_id: Optional[str] = None,
                 fail_after_chunk: Optional[int] = None):
        self.cfg = config
        self.source_str = dataset_b_source
        self.output_root = output_root
        self.node_index_map_path = node_index_map_path
        self.cache_root = Path(cache_root)
        self.config_hash = config.config_hash()
        self.run_id = run_id or make_run_id(self.config_hash)
        self.layout = RunLayout(output_root, self.run_id).ensure()
        self.manifest = Manifest(self.layout)
        self._fail_after_chunk = fail_after_chunk  # test hook: raise mid-run
        self.node_map: Optional[NodeMap] = None
        self.logger = self._make_logger()

    # ------------------------------------------------------------------ utils
    def _make_logger(self) -> logging.Logger:
        lg = logging.getLogger(f"pilot.{self.run_id}")
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        fh = logging.FileHandler(self.layout.root / "logs" / "pilot.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(sh)
        return lg

    # ------------------------------------------------------------------ P00
    def p00_runtime_and_storage(self) -> dict:
        import shutil
        from tdmec_pilot import __version__ as pilot_version

        self.manifest.load_or_init(self.run_id, self.config_hash, git_short_hash(),
                                   self.cfg.canonical)
        git_commit = git_short_hash()
        # compatibility (resume guard)
        compat = self.manifest.compatibility_check(self.config_hash, git_commit)
        if self.manifest.data.get("config_hash") != self.config_hash:
            raise ConfigIncompatibleError(
                f"existing run config hash {self.manifest.data.get('config_hash')} "
                f"!= current {self.config_hash}; refusing to mix incompatible configs")
        free_gb = shutil.disk_usage(self.output_root if Path(self.output_root).exists()
                                    else ".").free / 1e9
        info = {
            "pilot_version": pilot_version,
            "run_id": self.run_id, "config_hash": self.config_hash,
            "output_root_writable": self._writable(self.output_root),
            "cache_root_writable": self._writable(self.cache_root),
            "free_gb": round(free_gb, 1),
            "node_map_present": Path(self.node_index_map_path).is_file()
            if self.node_index_map_path else False,
            "compatibility": compat,
        }
        if not info["node_map_present"]:
            raise FileNotFoundError(f"node index map not found: {self.node_index_map_path}")
        self.node_map = load_node_map(
            self.node_index_map_path,
            expected_count=self.cfg.canonical["frozen_node_count"],
            index_min=self.cfg.canonical["valid_node_index_min"],
            index_max=self.cfg.canonical["valid_node_index_max"],
        )
        info["node_map_size"] = len(self.node_map)
        self.manifest.set_stage("P00_runtime_and_storage", "complete", info)
        self.manifest.flush()
        self.logger.info("P00 ok: %s", json.dumps(info))
        return info

    @staticmethod
    def _writable(root) -> bool:
        try:
            p = Path(root)
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("ok")
            t.unlink()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ P01
    def p01_source_verification(self) -> dict:
        from tdmec_discovery.cache import DownloadCache
        from tdmec_discovery.sources import build_source

        source = build_source(self.source_str)
        files = {f.name: f for f in source.list_files()}
        cache = DownloadCache(self.cache_root)
        result = {"files": {}}
        self._local_inputs: Dict[str, Path] = {}
        for name in self.cfg.input_files:
            if name not in files:
                result["files"][name] = {"available": False}
                continue
            rec = cache.get(source, files[name], compute_hash=True)
            prev = self.manifest.data.get("source_checksums", {}).get(name)
            if prev and prev != rec["sha256"]:
                raise RuntimeError(f"input checksum changed for {name}: {prev} -> {rec['sha256']}")
            self.manifest.data.setdefault("source_checksums", {})[name] = rec["sha256"]
            self._local_inputs[name] = Path(rec["path"])
            result["files"][name] = {"available": True, "sha256": rec["sha256"],
                                     "size": rec["size"]}
        missing = [n for n in self.cfg.input_files if not result["files"].get(n, {}).get("available")]
        result["all_available"] = not missing
        result["missing"] = missing
        atomic_write_json(self.layout.root / "source_validation.json", result)
        self.manifest.set_stage("P01_source_verification", "complete" if not missing else "failed", result)
        self.manifest.flush()
        self.logger.info("P01: %s", json.dumps({k: v.get("available") for k, v in result["files"].items()}))
        if missing:
            raise FileNotFoundError(f"required pilot input files missing: {missing}")
        return result

    # ------------------------------------------------------------------ P02
    def p02_schema_validation(self) -> dict:
        out = {"files": {}}
        self._headers: Dict[str, List[str]] = {}
        for name, path in self._local_inputs.items():
            sheet, header, _rows = _read_sheet(str(path))
            self._headers[name] = header
            cv = validate_columns(header, self.cfg.expected_columns)
            sheet_ok = sheet == self.cfg.expected_sheet
            out["files"][name] = {"sheet": sheet, "sheet_ok": sheet_ok,
                                  "columns_ok": cv["matches"], "missing": cv["missing"],
                                  "unexpected": cv["unexpected"]}
        out["all_valid"] = all(v["sheet_ok"] and v["columns_ok"] for v in out["files"].values())
        self.manifest.set_stage("P02_schema_validation", "complete" if out["all_valid"] else "failed", out)
        self.manifest.flush()
        self.logger.info("P02: all_valid=%s", out["all_valid"])
        if not out["all_valid"]:
            raise ValueError(f"Dataset B schema validation failed: {out}")
        return out

    # ------------------------------------------------ P03-P10 (chunk processing)
    def _transform_row(self, row, colidx, source_file, sheet, sheet_row_number) -> dict:
        tc = self.cfg.canonical["text_normalization"]

        def cell(c):
            i = colidx.get(c)
            return row[i] if i is not None and i < len(row) else None

        rec = {k: None for k in NORMALIZED_COLUMNS}
        rec.update({"source_file": source_file, "source_sheet": sheet,
                    "source_row_number": sheet_row_number,
                    "record_status": "rejected", "exclusion_reason": None,
                    "node_index": None, "snapshot_id": None})

        # P04 tweet id
        tid = normalize_tweet_id(cell("id"))
        if not tid.ok:
            rec["exclusion_reason"] = tid.error
            return rec
        rec["tweet_id"] = tid.value

        # P03 user blob
        user = parse_user_blob(cell("user"))
        if not user.ok:
            rec["exclusion_reason"] = user.error or "malformed_user_blob"
            return rec
        rec["author_account_id"] = user.account_id

        # P05 timestamp
        raw_ts = cell("created_at")
        rec["created_at_original"] = None if raw_ts is None else str(raw_ts)
        ts = parse_created_at(raw_ts)
        if not ts.ok:
            rec["exclusion_reason"] = ts.error
            return rec
        rec["created_at_utc"] = ts.utc

        # P06 text
        txt = normalize_text(cell("text"), unicode_form=tc.get("unicode_form", "NFC"),
                             normalize_newlines=tc.get("normalize_newlines", True),
                             strip_bom=tc.get("strip_bom", True))
        rec["text_raw"] = txt.text_raw
        rec["text_normalized"] = txt.text_normalized
        rec["text_quality"] = txt.quality
        rec["text_raw_len"] = txt.raw_len
        rec["text_normalized_len"] = txt.normalized_len

        # engagement
        for c in ["likes", "retweets", "reply_count", "quoted_count", "bookmarks", "views"]:
            rec[c] = _to_int(cell(c))

        # P08 frozen-node reconciliation
        node_index = self.node_map.get(user.account_id)
        if node_index is None:
            rec["record_status"] = "excluded"
            rec["exclusion_reason"] = "author_not_in_frozen_universe"
            return rec
        rec["node_index"] = node_index

        # P09 snapshot assignment
        sid = assign_snapshot(ts.utc)
        if sid is None:
            rec["record_status"] = "excluded"
            rec["exclusion_reason"] = "outside_canonical_snapshot_range"
            return rec
        rec["snapshot_id"] = sid

        rec["record_status"] = "retained"
        return rec

    def p10_process(self) -> dict:
        summary = {"files": {}}
        for name, path in self._local_inputs.items():
            summary["files"][name] = self._process_file(name, path)
        self.manifest.set_stage("P10_normalized_publication", "complete", summary)
        self.manifest.flush()
        return summary

    def _load_checkpoint(self, name: str) -> dict:
        p = self.layout.checkpoint_path(name)
        if p.is_file():
            return json.loads(p.read_text())
        return {"source_file": name, "chunks": {}}

    def _save_checkpoint(self, name: str, ck: dict) -> None:
        atomic_write_json(self.layout.checkpoint_path(name), ck)

    def _process_file(self, name: str, path: Path) -> dict:
        sheet, header, rows = _read_sheet(str(path))
        colidx = {c: i for i, c in enumerate(header)}
        chunk_size = self.cfg.chunk_size
        n = len(rows)
        n_chunks = (n + chunk_size - 1) // chunk_size
        ck = self._load_checkpoint(name)
        counts = {"rows_in": 0, "retained": 0, "excluded": 0, "rejected": 0, "chunks": n_chunks}
        new_chunks_written = 0
        for ci in range(n_chunks):
            key = str(ci)
            existing = ck["chunks"].get(key)
            npart = self.layout.normalized_dir(name) / f"part-{ci:05d}.parquet"
            xpart = self.layout.excluded_dir() / f"part-{Path(name).stem}-{ci:05d}.parquet"
            if existing and self._chunk_valid(existing, npart, xpart):
                self.logger.info("resume: skip %s chunk %d", name, ci)
                self._accumulate(counts, existing)
                continue
            start = ci * chunk_size
            chunk_rows = rows[start:start + chunk_size]
            recs = [self._transform_row(r, colidx, name, sheet, start + j + 2)
                    for j, r in enumerate(chunk_rows)]
            retained = [r for r in recs if r["record_status"] == "retained"]
            others = [r for r in recs if r["record_status"] != "retained"]
            n_sha = self._write_records(retained, npart)
            x_sha = self._write_records(others, xpart)
            entry = {"rows_in": len(recs), "retained": len(retained),
                     "excluded": sum(1 for r in others if r["record_status"] == "excluded"),
                     "rejected": sum(1 for r in others if r["record_status"] == "rejected"),
                     "normalized_part": str(npart.relative_to(self.layout.root)),
                     "excluded_part": str(xpart.relative_to(self.layout.root)),
                     "normalized_sha256": n_sha, "excluded_sha256": x_sha,
                     "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            ck["chunks"][key] = entry
            self._save_checkpoint(name, ck)
            self._accumulate(counts, entry)
            new_chunks_written += 1
            self.logger.info("wrote %s chunk %d: in=%d retained=%d",
                             name, ci, entry["rows_in"], entry["retained"])
            if self._fail_after_chunk is not None and new_chunks_written >= self._fail_after_chunk:
                raise KeyboardInterrupt(
                    f"test-hook interruption after {new_chunks_written} new chunk(s) in {name}")
        return counts

    @staticmethod
    def _accumulate(counts: dict, entry: dict) -> None:
        for k in ["rows_in", "retained", "excluded", "rejected"]:
            counts[k] += entry.get(k, 0)

    def _chunk_valid(self, entry: dict, npart: Path, xpart: Path) -> bool:
        try:
            return (npart.is_file() and xpart.is_file()
                    and sha256_file(npart) == entry.get("normalized_sha256")
                    and sha256_file(xpart) == entry.get("excluded_sha256"))
        except Exception:
            return False

    def _write_records(self, records: List[dict], path: Path) -> str:
        from .schema import parquet_schema

        df = records_to_frame(records)
        # cast nullable pandas to arrow-friendly via schema; ensure node/snapshot int32
        return write_parquet_atomic(df, path, schema=parquet_schema())

    # ------------------------------------------------ finalization (dedup/recon/stats)
    def finalize(self) -> dict:
        import pandas as pd

        parts = list(self.layout.root.glob("normalized_records/**/*.parquet")) + \
            list(self.layout.excluded_dir().glob("*.parquet"))
        frames = [pd.read_parquet(p) for p in parts if p.stat().st_size > 0]
        df = pd.concat(frames, ignore_index=True) if frames else records_to_frame([])

        # ---- duplicates (annotate; never drop) ----
        tracker = DuplicateTracker()
        valid_ids = df[df["tweet_id"].notna()]
        for _, r in valid_ids.iterrows():
            ch = content_hash((r["author_account_id"], r["created_at_original"], r["text_raw"],
                               r["likes"], r["retweets"], r["reply_count"],
                               r["quoted_count"], r["bookmarks"], r["views"]))
            tracker.add(str(r["tweet_id"]), str(r["source_file"]),
                        int(r["source_row_number"]), ch, str(r["record_status"]))
        dup_rows = tracker.duplicate_report()
        dup_df = pd.DataFrame([d.__dict__ for d in dup_rows]) if dup_rows else pd.DataFrame(
            columns=["tweet_id", "duplicate_type", "occurrence_count",
                     "canonical_source_file", "canonical_source_row_number", "source_locations"])
        dup_sha = write_parquet_atomic(dup_df, self.layout.root / "duplicate_records.parquet")

        # ---- account reconciliation ----
        auth = df[df["author_account_id"].notna()].copy()
        recon = (auth.groupby("author_account_id")
                 .agg(record_count=("tweet_id", "size"),
                      node_index=("node_index", "first"),
                      retained=("record_status", lambda s: (s == "retained").sum()))
                 .reset_index())
        recon["matched"] = recon["node_index"].notna()
        recon_sha = write_parquet_atomic(recon, self.layout.root / "account_reconciliation.parquet")
        unmatched = recon[~recon["matched"]][["author_account_id", "record_count"]]
        unmatched_sha = write_parquet_atomic(unmatched, self.layout.root / "unmatched_accounts.parquet")

        # ---- snapshot statistics ----
        snap = (df[df["record_status"] == "retained"]
                .groupby("snapshot_id").size().reset_index(name="retained_records"))
        snap_sha = write_parquet_atomic(snap, self.layout.root / "snapshot_statistics.parquet")

        # ---- text quality stats ----
        tq = {
            "by_quality": df["text_quality"].value_counts(dropna=False).to_dict(),
            "raw_len": {"min": int(df["text_raw_len"].min() or 0),
                        "max": int(df["text_raw_len"].max() or 0),
                        "mean": float(df["text_raw_len"].dropna().mean() or 0)},
            "normalized_len": {"min": int(df["text_normalized_len"].min() or 0),
                               "max": int(df["text_normalized_len"].max() or 0)},
        }
        tq = {k: ({str(kk): int(vv) for kk, vv in v.items()} if isinstance(v, dict) and k == "by_quality" else v)
              for k, v in tq.items()}
        atomic_write_json(self.layout.root / "text_quality_statistics.json", tq)

        result = {
            "total_records": int(len(df)),
            "retained": int((df["record_status"] == "retained").sum()),
            "excluded": int((df["record_status"] == "excluded").sum()),
            "rejected": int((df["record_status"] == "rejected").sum()),
            "duplicate_stats": tracker.stats(),
            "unique_authors": int(auth["author_account_id"].nunique()),
            "matched_authors": int(recon["matched"].sum()),
            "unmatched_authors": int((~recon["matched"]).sum()),
        }
        for rel, sha in [("duplicate_records.parquet", dup_sha),
                         ("account_reconciliation.parquet", recon_sha),
                         ("unmatched_accounts.parquet", unmatched_sha),
                         ("snapshot_statistics.parquet", snap_sha)]:
            p = self.layout.root / rel
            self.manifest.add_output(rel, p.stat().st_size, sha)
        # register part outputs
        for p in parts:
            self.manifest.add_output(str(p.relative_to(self.layout.root)), p.stat().st_size,
                                     sha256_file(p))
        self.manifest.set_stage("finalize", "complete", result)
        self.manifest.flush()
        return result

    # ------------------------------------------------------------------ P11
    def p11_validate(self, final: dict) -> dict:
        import pandas as pd

        gates = {}
        # inputs
        gates["both_inputs_available"] = all(
            self.manifest.data["stages"]["P01_source_verification"]["detail"]["files"][n]["available"]
            for n in self.cfg.input_files)
        gates["both_checksums_recorded"] = all(
            n in self.manifest.data.get("source_checksums", {}) for n in self.cfg.input_files)
        gates["schema_valid"] = self.manifest.data["stages"]["P02_schema_validation"]["detail"]["all_valid"]

        # read retained
        npats = list(self.layout.root.glob("normalized_records/**/*.parquet"))
        nframes = [pd.read_parquet(p) for p in npats if p.stat().st_size > 0]
        retained = pd.concat(nframes, ignore_index=True) if nframes else records_to_frame([])
        xpats = list(self.layout.excluded_dir().glob("*.parquet"))
        xframes = [pd.read_parquet(p) for p in xpats if p.stat().st_size > 0]
        excluded = pd.concat(xframes, ignore_index=True) if xframes else records_to_frame([])

        gates["tweet_ids_exact_strings"] = (
            retained["tweet_id"].map(lambda x: isinstance(x, str)).all() if len(retained) else True)
        gates["retained_all_have_node_index"] = bool(
            retained["node_index"].notna().all()) if len(retained) else True
        if len(retained):
            gates["node_index_in_range"] = bool(
                ((retained["node_index"] >= self.cfg.canonical["valid_node_index_min"]) &
                 (retained["node_index"] <= self.cfg.canonical["valid_node_index_max"])).all())
            gates["retained_snapshot_in_range"] = bool(
                ((retained["snapshot_id"] >= 0) & (retained["snapshot_id"] <= 34)).all())
        else:
            gates["node_index_in_range"] = True
            gates["retained_snapshot_in_range"] = True

        # row accounting
        rows_in = sum(v["rows_in"] for f in self.manifest.data["stages"]["P10_normalized_publication"]["detail"]["files"].values() for v in [f])
        retained_n = len(retained)
        excluded_n = int((excluded["record_status"] == "excluded").sum()) if len(excluded) else 0
        rejected_n = int((excluded["record_status"] == "rejected").sum()) if len(excluded) else 0
        gates["row_accounting_balances"] = (rows_in == retained_n + excluded_n + rejected_n)

        # parquet reopenable + checksum verify
        reopen_ok = True
        checksum_ok = True
        for rel, meta in self.manifest.data.get("outputs", {}).items():
            p = self.layout.root / rel
            try:
                pd.read_parquet(p)
            except Exception:
                reopen_ok = False
            if sha256_file(p) != meta["sha256"]:
                checksum_ok = False
        gates["parquet_reopenable"] = reopen_ok
        gates["output_checksums_verified"] = checksum_ok

        # raw source unchanged (re-hash cached inputs vs recorded)
        raw_ok = True
        from tdmec_discovery.hashing import sha256_file as _sh
        for name, p in self._local_inputs.items():
            if _sh(p) != self.manifest.data["source_checksums"][name]:
                raw_ok = False
        gates["raw_source_unchanged"] = raw_ok

        gates = {k: bool(v) for k, v in gates.items()}
        report = {
            "run_id": self.run_id, "config_hash": self.config_hash,
            "gates": gates, "all_passed": all(gates.values()),
            "accounting": {"rows_in": rows_in, "retained": retained_n,
                           "excluded": excluded_n, "rejected": rejected_n},
            "finalize": final,
            "snapshot_boundaries": [
                {"snapshot_id": b.snapshot_id, "label": b.label,
                 "start_utc": b.start_utc.isoformat(),
                 "end_utc_exclusive": b.end_utc_exclusive.isoformat()}
                for b in boundary_table()],
        }
        atomic_write_json(self.layout.root / "validation_report.json", report)
        self.manifest.set_stage("P11_output_validation",
                                "complete" if report["all_passed"] else "failed", gates)
        self.manifest.flush()
        self.manifest.write_checksums()
        self.logger.info("P11 gates: %s", json.dumps(gates))
        return report

    # ------------------------------------------------------------------ driver
    def run(self) -> dict:
        self.p00_runtime_and_storage()
        self.p01_source_verification()
        self.p02_schema_validation()
        self.p10_process()
        final = self.finalize()
        report = self.p11_validate(final)
        self.manifest.set_stage("P12_resume_verification", "n/a-single-pass",
                                {"note": "resume covered by checkpoint reuse + tests"})
        self.manifest.flush()
        return report
