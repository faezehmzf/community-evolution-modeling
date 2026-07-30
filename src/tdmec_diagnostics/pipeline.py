"""Resumable Phase 2 diagnostics pipeline with concrete A/B adapters."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tdmec import constants as C
from tdmec.hashing import sha256_file
from tdmec_diagnostics import constants as DC
from tdmec_diagnostics.adapters import (
    AdapterConfigurationError,
    NodeUniverseLookup,
    get_adapter,
    load_node_universe_lookup,
)
from tdmec_diagnostics.calendar_diag import CalendarAccumulator, human_calendar_summary
from tdmec_diagnostics.checkpoint import (
    ConfigIncompatibleError,
    DiagnosticsCheckpointStore,
    IncompleteRunError,
    InputChecksumDriftError,
)
from tdmec_diagnostics.config import DiagnosticsConfig, load_diagnostics_config
from tdmec_diagnostics.coverage_diag import CoverageAccumulator, human_coverage_summary
from tdmec_diagnostics.dedup_diag import DedupAccumulator, human_dedup_summary
from tdmec_diagnostics.fixtures import (
    SYN_END,
    SYN_FROZEN_NODES,
    SYN_NODE_UNIVERSE,
    SYN_START,
    build_synthetic_records,
    records_by_file,
)
from tdmec_diagnostics.io_utils import (
    DiagnosticsRunLayout,
    atomic_write_json,
    atomic_write_text,
    make_run_id,
    runtime_environment,
)
from tdmec_diagnostics.privacy import (
    assert_privacy_safe_mapping,
    privacy_safe_file_ref,
)
from tdmec_diagnostics.quarters import build_quarter_range, classify_timestamp
from tdmec_diagnostics.records import DiagnosticEventRecord
from tdmec_diagnostics.reports import (
    build_execution_manifest,
    build_run_summary,
    build_unresolved_evidence,
    build_warnings_report,
    human_run_summary,
    seal_report,
    scientific_content_hash,
)
from tdmec_diagnostics.status import finalize_run_status
from tdmec_diagnostics.text_length_diag import (
    TextLengthAccumulator,
    human_text_length_summary,
)
from tdmec_diagnostics.tokenizer import NullTokenizerProbe, WhitespaceTokenizerProbe
from tdmec_diagnostics.transaction_state import (
    TransactionalRunState,
    TransactionStateError,
)


@dataclass
class FileSpec:
    """One input file for diagnostics (privacy-safe name + optional local path)."""

    file_ref: str
    dataset: str  # A | B
    local_path: Optional[Path] = None
    checksum: Optional[str] = None
    records: Optional[Sequence[DiagnosticEventRecord]] = None


@dataclass
class DiagnosticsPipeline:
    config: DiagnosticsConfig
    output_root: Path
    run_id: Optional[str] = None
    checkpoint_root: Optional[Path] = None
    real_data_executed: bool = False
    real_data_status_message: str = (
        "Real data was not accessible or not authorized in this environment; "
        "diagnostics executed on synthetic fixtures only."
    )

    layout: Optional[DiagnosticsRunLayout] = field(default=None, init=False)
    checkpoint: Optional[DiagnosticsCheckpointStore] = field(default=None, init=False)
    transaction_state: Optional[TransactionalRunState] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        cfg_hash = self.config.config_hash()
        if self.run_id is None:
            self.run_id = make_run_id(cfg_hash)
        self.layout = DiagnosticsRunLayout(self.output_root, self.run_id).ensure()
        ckpt_dir = (
            Path(self.checkpoint_root)
            if self.checkpoint_root is not None
            else self.layout.checkpoint_dir
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = DiagnosticsCheckpointStore(
            root=ckpt_dir,
            config_hash=cfg_hash,
        )
        if self.config.resume_mode == "restart":
            self._restart_run_state()
        elif (
            self.config.enable_checkpoint
            and self.checkpoint.path.is_file()
            and not (
                self.checkpoint.root / TransactionalRunState.FILENAME
            ).is_file()
        ):
            self.checkpoint.load()

    def _restart_run_state(self) -> None:
        assert self.layout is not None and self.checkpoint is not None
        if self.checkpoint.path.is_file():
            self.checkpoint.path.unlink()
        state_path = self._accumulator_state_path()
        if state_path.is_file():
            state_path.unlink()
        transaction_path = (
            self.checkpoint.root / TransactionalRunState.FILENAME
        )
        transaction_path.unlink(missing_ok=True)
        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{transaction_path}{suffix}").unlink(missing_ok=True)
        for p in list(self.layout.reports_dir.glob("*.json")):
            p.unlink()
        for p in list(self.layout.human_dir.glob("*.md")):
            p.unlink()
        if self.layout.manifest_path.is_file():
            self.layout.manifest_path.unlink()
        self.checkpoint.files = {}

    def _boundaries(self):
        return build_quarter_range(
            self.config.provisional_start_label,
            self.config.provisional_end_label,
        )

    def _tokenizer(self):
        if self.config.enable_tokenizer_diagnostics:
            return WhitespaceTokenizerProbe()
        return NullTokenizerProbe()

    def _n_universe(self, override: Optional[int] = None) -> int:
        if override is not None:
            return int(override)
        if self.config.node_universe_size is not None:
            return int(self.config.node_universe_size)
        return int(C.N_NODES)

    def _accumulator_state_path(self) -> Path:
        assert self.checkpoint is not None
        return self.checkpoint.root / "accumulator_state.json"

    def _accumulator_payload(
        self,
        cal: CalendarAccumulator,
        dedup: DedupAccumulator,
        text: TextLengthAccumulator,
        cov: CoverageAccumulator,
    ) -> Dict[str, Any]:
        return {
            "schema_version": "tdmec-phase2-accumulator-state-v1",
            "config_hash": self.config.config_hash(),
            "calendar": cal.to_state(),
            "dedup": dedup.to_state(),
            "text_length": text.to_state(),
            "coverage": cov.to_state(),
        }

    def _publish_checkpoint_mirrors(
        self,
        checkpoint_payload: Dict[str, Any],
        accumulator_payload: Dict[str, Any],
    ) -> None:
        assert self.checkpoint is not None
        assert_privacy_safe_mapping(accumulator_payload)
        self.checkpoint.save(checkpoint_payload)
        atomic_write_json(self._accumulator_state_path(), accumulator_payload)

    def _commit_progress(
        self,
        cal: CalendarAccumulator,
        dedup: DedupAccumulator,
        text: TextLengthAccumulator,
        cov: CoverageAccumulator,
    ) -> None:
        assert self.checkpoint is not None
        checkpoint_payload = self.checkpoint.to_payload()
        accumulator_payload = self._accumulator_payload(cal, dedup, text, cov)
        assert_privacy_safe_mapping(accumulator_payload)
        if self.transaction_state is None:
            self._publish_checkpoint_mirrors(
                checkpoint_payload,
                accumulator_payload,
            )
            return
        self.transaction_state.commit_snapshot(
            checkpoint_payload,
            accumulator_payload,
        )
        self._publish_checkpoint_mirrors(
            checkpoint_payload,
            accumulator_payload,
        )

    def _begin_progress_transaction(self) -> None:
        if self.transaction_state is not None:
            self.transaction_state.begin()

    def _rollback_progress_transaction(self) -> None:
        if self.transaction_state is not None:
            self.transaction_state.rollback()

    def _load_transaction_state(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        assert self.checkpoint is not None
        self.transaction_state = TransactionalRunState(
            self.checkpoint.root,
            self.config.config_hash(),
        )
        if not self.transaction_state.has_snapshot:
            if self.checkpoint.path.is_file() or self._accumulator_state_path().is_file():
                raise TransactionStateError(
                    "unsealed legacy checkpoint state has no transactional "
                    "authority; restart with a new run or resume_mode='restart'"
                )
            return None
        checkpoint_payload, accumulator_payload, _generation = (
            self.transaction_state.load_snapshot()
        )
        self.checkpoint.load_payload(checkpoint_payload)
        return checkpoint_payload, accumulator_payload

    def _load_accumulators(
        self,
        *,
        boundaries,
        n_univ: int,
        frozen_nodes: Optional[set],
    ) -> Tuple[
        CalendarAccumulator,
        DedupAccumulator,
        TextLengthAccumulator,
        CoverageAccumulator,
        bool,
    ]:
        transaction_snapshot = (
            self._load_transaction_state()
            if self.config.enable_checkpoint
            else None
        )
        if transaction_snapshot is None:
            cal = CalendarAccumulator(boundaries=boundaries)
            dedup = DedupAccumulator(
                connection=(
                    self.transaction_state.connection
                    if self.transaction_state is not None
                    else None
                )
            )
            text = TextLengthAccumulator(
                quantiles=self.config.quantiles,
                candidate_max_lengths=self.config.candidate_max_lengths,
                tokenizer=self._tokenizer(),
            )
            cov = CoverageAccumulator(
                relations=self.config.relations,
                node_universe_size=n_univ,
                frozen_node_indices=frozen_nodes,
            )
            return cal, dedup, text, cov, False
        _checkpoint_payload, data = transaction_snapshot
        if data.get("config_hash") != self.config.config_hash():
            raise ConfigIncompatibleError(
                "accumulator state config_hash mismatch; use --resume-mode restart"
            )
        cal = CalendarAccumulator.from_state(data["calendar"], boundaries=boundaries)
        dedup = DedupAccumulator.from_state(
            data["dedup"],
            connection=(
                self.transaction_state.connection
                if self.transaction_state is not None
                else None
            ),
        )
        text = TextLengthAccumulator.from_state(
            data["text_length"],
            quantiles=self.config.quantiles,
            candidate_max_lengths=self.config.candidate_max_lengths,
            tokenizer=self._tokenizer(),
        )
        cov = CoverageAccumulator.from_state(
            data["coverage"],
            relations=self.config.relations,
            node_universe_size=n_univ,
            frozen_node_indices=frozen_nodes,
        )
        return cal, dedup, text, cov, True

    def _iter_source_row_chunks(
        self,
        records: Iterable[DiagnosticEventRecord],
    ) -> Iterable[Tuple[List[DiagnosticEventRecord], int]]:
        """Yield bounded chunks without splitting one source workbook row."""
        chunk: List[DiagnosticEventRecord] = []
        last_row_number = 1
        for record in records:
            row_number = int(record.source_row_number)
            if row_number < last_row_number:
                raise ValueError(
                    "adapter records must be ordered by source_row_number"
                )
            if (
                chunk
                and len(chunk) >= self.config.chunk_size
                and row_number != last_row_number
            ):
                yield chunk, last_row_number
                chunk = []
            chunk.append(record)
            last_row_number = row_number
        if chunk:
            yield chunk, last_row_number

    def _consume_chunk(
        self,
        chunk: Sequence[DiagnosticEventRecord],
        cal: CalendarAccumulator,
        dedup: DedupAccumulator,
        text: TextLengthAccumulator,
        cov: CoverageAccumulator,
        boundaries,
    ) -> Tuple[int, int]:
        accepted = rejected = 0
        for rec in chunk:
            _utc, sid, reason = classify_timestamp(
                rec.timestamp_raw, boundaries=boundaries
            )
            qlabel = rec.extra.get("quarter_label")
            if not qlabel and sid is not None:
                qlabel = boundaries[sid].label
            cal.observe(rec)
            dedup.observe(rec, quarter_label=qlabel)
            text.observe(rec, quarter_label=qlabel)
            cov.observe(rec, quarter_label=qlabel)
            if reason == DC.REASON_IN_RANGE:
                accepted += 1
            else:
                rejected += 1
        return accepted, rejected

    def run_on_files(
        self,
        files: Sequence[FileSpec],
        *,
        node_lookup: Optional[NodeUniverseLookup] = None,
        frozen_nodes: Optional[set] = None,
        node_universe_size: Optional[int] = None,
        require_complete: bool = True,
        interrupt_after_files: Optional[int] = None,
        interrupt_after_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process files with transactional chunk resume and no double counting."""
        assert self.layout is not None and self.checkpoint is not None
        boundaries = self._boundaries()
        cfg_hash = self.config.config_hash()
        n_univ = self._n_universe(node_universe_size)
        if frozen_nodes is None and node_lookup is not None:
            frozen_nodes = set(node_lookup.mapping.values())

        file_names = [f.file_ref for f in files]
        transaction_path = (
            self.checkpoint.root / TransactionalRunState.FILENAME
        )

        # A sealed run has no transaction database, so its JSON checkpoint is
        # authoritative and checksum drift must be rejected before returning.
        if self.config.enable_checkpoint and not transaction_path.is_file():
            for spec in files:
                cp = self.checkpoint.files.get(spec.file_ref)
                if (
                    cp is not None
                    and cp.source_checksum
                    and spec.checksum
                    and cp.source_checksum != spec.checksum
                ):
                    raise InputChecksumDriftError(
                        "source checksum drift for "
                        f"{privacy_safe_file_ref(spec.file_ref)}"
                    )

        # Idempotent sealed reload
        if (
            self.config.enable_checkpoint
            and not transaction_path.is_file()
            and self.checkpoint.is_complete(file_names)
            and self.layout.manifest_path.is_file()
            and all(
                self.layout.report_path(n).is_file()
                for n in (
                    DC.REPORT_CALENDAR,
                    DC.REPORT_DEDUP,
                    DC.REPORT_TEXT_LENGTH,
                    DC.REPORT_COVERAGE,
                    DC.REPORT_SUMMARY,
                )
            )
        ):
            return self._load_sealed_result()

        cal, dedup, text, cov, _loaded = self._load_accumulators(
            boundaries=boundaries, n_univ=n_univ, frozen_nodes=frozen_nodes
        )

        # Reject checksum drift against the authoritative transaction snapshot.
        if self.config.enable_checkpoint:
            for spec in files:
                cp = self.checkpoint.files.get(spec.file_ref)
                if (
                    cp is not None
                    and cp.source_checksum
                    and spec.checksum
                    and cp.source_checksum != spec.checksum
                ):
                    raise InputChecksumDriftError(
                        "source checksum drift for "
                        f"{privacy_safe_file_ref(spec.file_ref)}"
                    )

        source_meta: List[Dict[str, Any]] = []
        files_processed = 0
        chunks_processed = 0
        interrupted = False
        for spec in files:
            source_meta.append(
                {
                    "file_ref": privacy_safe_file_ref(spec.file_ref),
                    "dataset": spec.dataset,
                    "checksum": spec.checksum,
                }
            )
            cp = self.checkpoint.files.get(spec.file_ref)
            if (
                self.config.enable_checkpoint
                and cp is not None
                and cp.complete
                and self.config.resume_mode == "resume"
            ):
                # Completed files are not reprocessed (prevents double-counting).
                if spec.checksum and cp.source_checksum and spec.checksum != cp.source_checksum:
                    raise InputChecksumDriftError(
                        f"source checksum drift for {privacy_safe_file_ref(spec.file_ref)}"
                    )
                continue

            cp = self.checkpoint.files.get(spec.file_ref)
            start_after_source_row = (
                cp.last_source_row_number
                if (
                    self.config.enable_checkpoint
                    and cp is not None
                    and self.config.resume_mode == "resume"
                )
                else 0
            )
            if self.config.enable_checkpoint and cp is None:
                self.checkpoint.reset_file(
                    spec.file_ref,
                    source_checksum=spec.checksum,
                )

            # Stream only records after the last transactionally committed row.
            if spec.records is not None:
                record_iter: Iterable[DiagnosticEventRecord] = (
                    record
                    for record in spec.records
                    if int(record.source_row_number) > start_after_source_row
                )
            elif spec.local_path is not None and node_lookup is not None:
                adapter = get_adapter(
                    self.config.dataset_a_adapter_id
                    if spec.dataset.upper() == "A"
                    else self.config.dataset_b_adapter_id
                )
                record_iter = adapter(
                    spec.local_path,
                    node_lookup=node_lookup,
                    source_file_name=spec.file_ref,
                    start_after_source_row=start_after_source_row,
                )
            else:
                raise AdapterConfigurationError(
                    f"file {privacy_safe_file_ref(spec.file_ref)} has no records or path"
                )

            file_exhausted = True
            for chunk, last_source_row_number in self._iter_source_row_chunks(
                record_iter
            ):
                self._begin_progress_transaction()
                try:
                    accepted, rejected = self._consume_chunk(
                        chunk,
                        cal,
                        dedup,
                        text,
                        cov,
                        boundaries,
                    )
                    if self.config.enable_checkpoint:
                        current = self.checkpoint.files[spec.file_ref]
                        next_chunk_index = (
                            max(current.chunks_completed, default=-1) + 1
                        )
                        self.checkpoint.mark_chunk(
                            spec.file_ref,
                            next_chunk_index,
                            rows_inspected=len(chunk),
                            rows_accepted=accepted,
                            rows_rejected=rejected,
                            last_source_row_number=last_source_row_number,
                            source_checksum=spec.checksum,
                        )
                        self._commit_progress(cal, dedup, text, cov)
                    else:
                        self._rollback_progress_transaction()
                except Exception:
                    self._rollback_progress_transaction()
                    raise

                chunks_processed += 1
                if (
                    interrupt_after_chunks is not None
                    and chunks_processed >= interrupt_after_chunks
                ):
                    file_exhausted = False
                    interrupted = True
                    break

            if interrupted:
                break

            if file_exhausted and self.config.enable_checkpoint:
                self._begin_progress_transaction()
                try:
                    self.checkpoint.mark_file_complete(spec.file_ref)
                    self._commit_progress(cal, dedup, text, cov)
                except Exception:
                    self._rollback_progress_transaction()
                    raise

            files_processed += 1
            if interrupt_after_files is not None and files_processed >= interrupt_after_files:
                interrupted = True
                break

        complete = (
            self.checkpoint.is_complete(file_names)
            if self.config.enable_checkpoint
            else not interrupted
        )
        # When enable_checkpoint False, complete if all files visited
        if not self.config.enable_checkpoint:
            complete = not interrupted

        if require_complete and not complete:
            if self.transaction_state is not None:
                self.transaction_state.close()
                self.transaction_state = None
            else:
                dedup.close()
            raise IncompleteRunError(
                "diagnostics run incomplete; resume required before sealing reports"
            )

        result = self._seal_and_persist(
            cal=cal,
            dedup=dedup,
            text=text,
            cov=cov,
            boundaries=boundaries,
            cfg_hash=cfg_hash,
            source_meta=source_meta,
            complete=complete,
        )
        if self.transaction_state is not None:
            self.transaction_state.remove()
            self.transaction_state = None
        else:
            dedup.close()
        return result

    def run_on_records(
        self,
        records: Sequence[DiagnosticEventRecord],
        *,
        require_complete: bool = True,
        interrupt_after_files: Optional[int] = None,
        interrupt_after_chunks: Optional[int] = None,
        node_universe_size: Optional[int] = None,
        frozen_nodes: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Compatibility path: group in-memory records by source_file."""
        by_file = records_by_file(records)
        specs = [
            FileSpec(file_ref=name, dataset=rows[0].dataset, records=rows)
            for name, rows in sorted(by_file.items())
        ]
        return self.run_on_files(
            specs,
            frozen_nodes=frozen_nodes,
            node_universe_size=node_universe_size,
            require_complete=require_complete,
            interrupt_after_files=interrupt_after_files,
            interrupt_after_chunks=interrupt_after_chunks,
        )

    def _load_sealed_result(self) -> Dict[str, Any]:
        assert self.layout is not None
        reports = {}
        for n in (
            DC.REPORT_CALENDAR,
            DC.REPORT_DEDUP,
            DC.REPORT_TEXT_LENGTH,
            DC.REPORT_COVERAGE,
            DC.REPORT_WARNINGS,
            DC.REPORT_UNRESOLVED,
            DC.REPORT_SUMMARY,
        ):
            p = self.layout.report_path(n)
            if p.is_file():
                reports[n] = json.loads(p.read_text(encoding="utf-8"))
        manifest = json.loads(self.layout.manifest_path.read_text(encoding="utf-8"))
        return {
            "run_id": self.run_id,
            "status": manifest.get("processing_status"),
            "layout": str(self.layout.root),
            "reports": reports,
            "manifest": manifest,
            "complete": True,
            "resumed_from_sealed": True,
        }

    def _seal_and_persist(
        self,
        *,
        cal: CalendarAccumulator,
        dedup: DedupAccumulator,
        text: TextLengthAccumulator,
        cov: CoverageAccumulator,
        boundaries,
        cfg_hash: str,
        source_meta: List[Dict[str, Any]],
        complete: bool,
    ) -> Dict[str, Any]:
        assert self.layout is not None and self.checkpoint is not None
        cal_report = seal_report(cal.build_report(config_hash=cfg_hash))
        dedup_report = seal_report(dedup.build_report(config_hash=cfg_hash))
        text_report = seal_report(text.build_report(config_hash=cfg_hash))
        expected_snaps = [b.label for b in boundaries]
        cov_report = seal_report(
            cov.build_report(config_hash=cfg_hash, expected_snapshots=expected_snaps)
        )

        warnings: List[Dict[str, Any]] = list(cov_report.get("candidate_warnings") or [])
        if cal_report.get("internal_empty_quarters"):
            warnings.append(
                {
                    "code": "INTERNAL_EMPTY_QUARTERS",
                    "severity": "WARNING",
                    "message": "Internal empty quarters observed and preserved",
                    "count": len(cal_report["internal_empty_quarters"]),
                }
            )
        hard_failures: List[Dict[str, Any]] = []

        status = finalize_run_status(
            complete=complete,
            has_hard_failures=bool(hard_failures),
            has_review_flags=True,
        )
        component_status = (
            DC.DIAGNOSTIC_COMPLETE if complete and not hard_failures else DC.UNVALIDATED
        )

        def _restatus(rep: Dict[str, Any]) -> Dict[str, Any]:
            cleaned = {k: v for k, v in rep.items() if k != "scientific_content_hash"}
            cleaned["status"] = component_status
            return seal_report(cleaned)

        cal_report = _restatus(cal_report)
        dedup_report = _restatus(dedup_report)
        text_report = _restatus(text_report)
        cov_report = _restatus(cov_report)

        warn_report = seal_report(
            build_warnings_report(
                config_hash=cfg_hash,
                warnings=warnings,
                hard_failures=hard_failures,
                status=status,
            )
        )
        unresolved = seal_report(
            build_unresolved_evidence(
                config_hash=cfg_hash,
                calendar_report=cal_report,
                dedup_report=dedup_report,
                text_report=text_report,
                coverage_report=cov_report,
                status=DC.REVIEW_REQUIRED,
            )
        )
        summary = seal_report(
            build_run_summary(
                config_hash=cfg_hash,
                status=status,
                calendar_status=cal_report["status"],
                dedup_status=dedup_report["status"],
                text_status=text_report["status"],
                coverage_status=cov_report["status"],
                real_data_executed=self.real_data_executed,
                real_data_status_message=self.real_data_status_message,
            )
        )

        report_map = {
            DC.REPORT_CALENDAR: cal_report,
            DC.REPORT_DEDUP: dedup_report,
            DC.REPORT_TEXT_LENGTH: text_report,
            DC.REPORT_COVERAGE: cov_report,
            DC.REPORT_WARNINGS: warn_report,
            DC.REPORT_UNRESOLVED: unresolved,
            DC.REPORT_SUMMARY: summary,
        }
        report_hashes = {k: scientific_content_hash(v) for k, v in report_map.items()}

        manifest = build_execution_manifest(
            run_id=self.run_id or "",
            config_hash=cfg_hash,
            config_dict=self.config.scientific_dict(),
            source_files=source_meta,
            processing_status=status,
            rows_inspected=cal.rows_inspected,
            rows_accepted=cal.rows_accepted,
            rows_rejected=cal.rows_rejected,
            warning_counts=warn_report["warning_counts"],
            hard_failure_counts=warn_report["hard_failure_counts"],
            resume_state=self.checkpoint.resume_state(),
            report_hashes=report_hashes,
            runtime_environment=runtime_environment(),
            real_data_executed=self.real_data_executed,
        )
        assert_privacy_safe_mapping(
            {k: v for k, v in manifest.items() if k != "runtime_environment"}
        )

        for name, report in report_map.items():
            atomic_write_json(self.layout.report_path(name), report)
        atomic_write_json(self.layout.manifest_path, manifest)
        atomic_write_text(
            self.layout.human_path(DC.REPORT_CALENDAR), human_calendar_summary(cal_report)
        )
        atomic_write_text(
            self.layout.human_path(DC.REPORT_DEDUP), human_dedup_summary(dedup_report)
        )
        atomic_write_text(
            self.layout.human_path(DC.REPORT_TEXT_LENGTH),
            human_text_length_summary(text_report),
        )
        atomic_write_text(
            self.layout.human_path(DC.REPORT_COVERAGE),
            human_coverage_summary(cov_report),
        )
        atomic_write_text(
            self.layout.human_path(DC.REPORT_SUMMARY), human_run_summary(summary)
        )

        return {
            "run_id": self.run_id,
            "status": status,
            "layout": str(self.layout.root),
            "reports": report_map,
            "manifest": manifest,
            "complete": complete,
        }

    def run_synthetic(self, **kwargs: Any) -> Dict[str, Any]:
        cfg = self.config
        if (
            cfg.provisional_start_label == "2017-Q4"
            and cfg.provisional_end_label == "2026-Q2"
        ):
            cfg = replace(
                cfg,
                provisional_start_label=SYN_START,
                provisional_end_label=SYN_END,
                node_universe_size=len(SYN_NODE_UNIVERSE),
                dataset_a_source_scheme="synthetic",
                dataset_b_source_scheme="synthetic",
                source_format="synthetic",
            )
            self.config = cfg
            checkpoint_root = self.checkpoint.root  # type: ignore[union-attr]
            self.checkpoint = DiagnosticsCheckpointStore(
                root=checkpoint_root,
                config_hash=cfg.config_hash(),
            )
            if cfg.resume_mode == "restart":
                self._restart_run_state()
            elif (
                cfg.enable_checkpoint
                and self.checkpoint.path.is_file()
                and not (
                    self.checkpoint.root / TransactionalRunState.FILENAME
                ).is_file()
            ):
                self.checkpoint.load()
        self.real_data_executed = False
        self.real_data_status_message = (
            "Real data was not accessible or not authorized in this environment; "
            "diagnostics executed on synthetic fixtures only."
        )
        return self.run_on_records(
            build_synthetic_records(),
            node_universe_size=len(SYN_NODE_UNIVERSE),
            frozen_nodes=set(SYN_FROZEN_NODES),
            **kwargs,
        )

    def run_real(
        self,
        *,
        dataset_a_paths: Sequence[str | Path],
        dataset_b_paths: Sequence[str | Path],
        node_index_map_path: str | Path,
    ) -> Dict[str, Any]:
        """Execute diagnostics on local workbook paths via concrete adapters."""
        if not dataset_a_paths and not dataset_b_paths:
            raise AdapterConfigurationError(
                "real mode requires Dataset A and/or Dataset B source paths"
            )
        n_univ = self._n_universe()
        if n_univ != C.N_NODES and self.config.node_universe_size is None:
            n_univ = C.N_NODES
        lookup = load_node_universe_lookup(node_index_map_path, expected_count=n_univ)

        specs: List[FileSpec] = []
        for p in dataset_a_paths:
            path = Path(p)
            checksum = sha256_file(path) if path.is_file() else None
            specs.append(
                FileSpec(
                    file_ref=path.name,
                    dataset="A",
                    local_path=path,
                    checksum=checksum,
                )
            )
        for p in dataset_b_paths:
            path = Path(p)
            checksum = sha256_file(path) if path.is_file() else None
            specs.append(
                FileSpec(
                    file_ref=path.name,
                    dataset="B",
                    local_path=path,
                    checksum=checksum,
                )
            )
        self.real_data_executed = True
        self.real_data_status_message = (
            "Real-data adapters executed on configured local workbook paths. "
            "Results remain DIAGNOSTIC / REVIEW_REQUIRED — not CERTIFIED."
        )
        # Ensure production N unless explicitly overridden in config
        return self.run_on_files(
            specs,
            node_lookup=lookup,
            node_universe_size=n_univ,
            frozen_nodes=set(lookup.mapping.values()),
        )


def _resolve_source_to_local_paths(source: str, cache_root: Path) -> List[Path]:
    """Resolve a repository source scheme to local workbook paths (read-only)."""
    from tdmec_discovery.cache import DownloadCache
    from tdmec_discovery.sources import build_source

    if not source or source in {"unset", "none"}:
        raise AdapterConfigurationError(
            "source configuration is absent; set Dataset A/B source schemes and paths"
        )
    src = build_source(source)
    cache = DownloadCache(cache_root)
    paths: List[Path] = []
    for rf in src.list_files():
        name = rf.name
        if not str(name).lower().endswith(".xlsx"):
            continue
        rec = cache.get(src, rf, compute_hash=True)
        paths.append(Path(rec["path"]))
    if not paths:
        raise AdapterConfigurationError(
            "no .xlsx workbooks resolved from configured source"
        )
    return sorted(paths, key=lambda p: p.name)


def run_diagnostics(
    *,
    output_root: str | Path,
    config: Optional[DiagnosticsConfig] = None,
    config_path: Optional[str | Path] = None,
    mode: str = "synthetic",
    run_id: Optional[str] = None,
    checkpoint_root: Optional[str | Path] = None,
    dataset_a_source: Optional[str] = None,
    dataset_b_source: Optional[str] = None,
    node_index_map: Optional[str | Path] = None,
    cache_root: str | Path = "/tmp/tdmec_cache",
    dataset_a_files: Optional[Sequence[str | Path]] = None,
    dataset_b_files: Optional[Sequence[str | Path]] = None,
) -> Dict[str, Any]:
    """Colab/controlled-environment entry point."""
    cfg = config or load_diagnostics_config(config_path)
    if mode == "synthetic":
        if (
            cfg.provisional_start_label == "2017-Q4"
            and cfg.provisional_end_label == "2026-Q2"
        ):
            cfg = replace(
                cfg,
                provisional_start_label=SYN_START,
                provisional_end_label=SYN_END,
                node_universe_size=len(SYN_NODE_UNIVERSE),
                dataset_a_source_scheme="synthetic",
                dataset_b_source_scheme="synthetic",
                source_format="synthetic",
            )
        pipe = DiagnosticsPipeline(
            config=cfg,
            output_root=Path(output_root),
            run_id=run_id,
            checkpoint_root=(
                Path(checkpoint_root) if checkpoint_root else None
            ),
        )
        return pipe.run_synthetic()
    if mode == "real":
        a_paths: List[Path] = []
        b_paths: List[Path] = []
        if dataset_a_files:
            a_paths = [Path(p) for p in dataset_a_files]
        elif dataset_a_source:
            a_paths = _resolve_source_to_local_paths(
                dataset_a_source, Path(cache_root)
            )
        if dataset_b_files:
            b_paths = [Path(p) for p in dataset_b_files]
        elif dataset_b_source:
            b_paths = _resolve_source_to_local_paths(
                dataset_b_source, Path(cache_root)
            )
        if not a_paths and not b_paths:
            raise AdapterConfigurationError(
                "real mode requires --dataset-a-source and/or --dataset-b-source "
                "(or explicit --dataset-a-file / --dataset-b-file). "
                "No credentials are embedded; configure authorized sources externally."
            )
        if node_index_map is None:
            raise AdapterConfigurationError(
                "real mode requires --node-index-map pointing to the frozen "
                "node_index_map.parquet (N=16736)"
            )
        # Update scheme labels in a copy for reporting (paths stay out of hash)
        scheme_a = "local" if dataset_a_files else (
            str(dataset_a_source).split(":", 1)[0] if dataset_a_source else "unset"
        )
        scheme_b = "local" if dataset_b_files else (
            str(dataset_b_source).split(":", 1)[0] if dataset_b_source else "unset"
        )
        cfg = replace(
            cfg,
            dataset_a_source_scheme=scheme_a,
            dataset_b_source_scheme=scheme_b,
            source_format="xlsx",
        )
        pipe = DiagnosticsPipeline(
            config=cfg,
            output_root=Path(output_root),
            run_id=run_id,
            checkpoint_root=(
                Path(checkpoint_root) if checkpoint_root else None
            ),
        )
        return pipe.run_real(
            dataset_a_paths=a_paths,
            dataset_b_paths=b_paths,
            node_index_map_path=node_index_map,
        )
    raise ValueError(f"unknown diagnostics mode: {mode!r}")
