# Phase 2 Traceability

Maps Decision IDs to Phase 2 diagnostic tooling and tests. Method text is not
duplicated here. Phase 2 produces **evidence only**.

| Decision ID | Diagnostic focus | Primary modules | Primary tests |
|---|---|---|---|
| **QCAL-B01** | Timestamp/quarter evidence; candidate start/end/`T` | `quarters`, `calendar_diag` | `test_quarterly_assignment*`, `test_candidate_T*`, `test_timezone_boundary*` |
| **QCAL-B01-PROC** | Internal empty preserved; leading/trailing REVIEW_REQUIRED | `calendar_diag` | `test_internal_empty_quarter_preservation` |
| **QDEDUP-B01** | Candidate signatures; concordant/discordant | `dedup_diag` | `test_exact_duplicate*`, `test_cross_file*`, `test_cross_file_duplicates_three_files*` |
| **QDEDUP-B01-PROC** | Annotate-only; sources immutable | `dedup_diag`, adapters | `test_raw_source_immutability*`, adapter immutability tests |
| **Q-MISS** | Coverage rates; no invented thresholds | `coverage_diag` | `test_frozen_node*`, `test_production_n_*` |
| **QACT-01** | `model_active = struct OR node_text` | `coverage_diag.model_active_mask` | `test_coverage_category*`, `test_edge_text_alone*` |
| **QART-01-FRAME** | Manifest/warnings; no CERTIFIED | `reports`, `status`, `pipeline` | `test_diagnostic_status*`, `test_no_certified*` |
| **QEMB-LENGTH-DIAG** | Exact length-frequency quantiles | `text_length_diag`, `length_stats` | `test_text_length*`, `test_null_vs_empty*` |
| **QEMB-X01..X07** | Explicitly unresolved | `config`, unresolved report | config unresolved tests |
| **D2** | Frozen universe N=16736 production default | `coverage_diag`, adapters | `test_production_n_denominator_default` |
| **QSELF-01** | Self-loop candidates before exclusion | `coverage_diag`, A adapter | fixture + coverage tests |
| **QREL-01** | Invalid relation counting | `coverage_diag` | coverage tests |
| **Adapters A/B** | Documented schema streaming | `adapters`, `schema_contracts`, `workbook_io` | `test_dataset_*_adapter*`, `test_real_mode_via_adapter_files` |

**Privacy:** `privacy.py` — `test_privacy_safe_*`, CLI redaction.

**Streaming / resume:** `checkpoint`, `pipeline`, `transaction_state` —
source-row chunk resume, single-transaction checkpoint authority, bounded
dedup retained state, split-mirror recovery, sealed idempotence, explicit
checkpoint-root preservation, checksum/config invalidation, and no
loss/double-count.

**Phase 1 regression:** `test_phase1_regression_*` + full Phase 1 suite.
