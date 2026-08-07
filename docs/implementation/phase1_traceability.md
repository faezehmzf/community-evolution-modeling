# Phase 1 Traceability

Maps Decision IDs to Phase 1 code and tests. Method text is not duplicated here.

| Decision ID | Contract focus | Primary modules | Primary tests |
|---|---|---|---|
| **QREL-01** | Immutable relation map `mention=0…quote=3` | `tdmec.constants`, `RelationConfig`, `validate_relation_map` | `test_canonical_relation_map_passes`, `test_reordered_relation_map_fails`, `test_altered_relation_id_fails`, `test_relation_map_immutable_proxy`, `test_extra_and_missing_relations_fail` |
| **QSELF-01** | No canonical self-loops | `EdgeArtifactConfig`, `validate_no_self_loops` | `test_self_loop_fails`, `test_zero_sized_edge_structures_ok` |
| **QCAL-B01-PROC** | Provisional calendar; no CERTIFIED without approval | `CalendarConfig`, `SnapshotRegistrySchema`, `validate_provisional_calendar_not_certified` | `test_provisional_calendar_cannot_claim_certified`, `test_internal_empty_snapshot_representable` |
| **Q-WGT** | Edge key + `count_raw` + `weight_log1p=ln(1+c)` | `CanonicalEdgeRecord`, weight/key validators | `test_valid_count_raw_log1p_passes`, `test_invalid_log1p_fails`, `test_duplicate_edge_key_fails`, `test_nonpositive_count_raw_fails`, `test_weight_log1p_math` |
| **Q-FEAT** | `F_struct=17` ordered features + `struct_active_mask` | `StructuralFeatureConfig`, `StructuralArtifactSchema`, feature validators | `test_17_feature_schema_passes`, `test_wrong_feature_count_fails`, `test_feature_rename_fails`, `test_feature_reorder_fails`, `test_feature_order_hash_deterministic`, `test_canonical_17_feature_list_exact`, `test_struct_inactive_exact_zero` |
| **Q-TEXT** | Node/edge text artifacts; `D_text` unresolved | `TextArtifactConfig`, text artifact schemas | `test_unavailable_node_text_exact_zero`, `test_unavailable_edge_text_exact_zero`, `test_edge_text_order_mismatch_fails`, `test_unresolved_values_not_silently_finalized` |
| **Q-MISS** | Exact-zero unavailable vectors + masks/counts metadata | `MissingnessConfig`, `validate_exact_zero_when_unavailable` | `test_nonzero_unavailable_node_text_fails`, privacy tests, zero-sized edge text |
| **QACT-01** | `model_active = struct OR node_text` (not edge text) | `ActivityMaskConfig`, `validate_model_active_mask` | `test_valid_model_active_mask_passes`, `test_edge_text_alone_does_not_activate_model_active`, `test_node_text_alone_activates_model_active` |
| **QART-01-FRAME** | Certification, manifests, checksums, shards, hard vs warning | `CertificationConfig`, `ManifestSchema`, checksum/shard validators | `test_manifest_checksum_passes`, `test_modified_artifact_fails_checksum`, `test_invalid_certification_transition_fails`, `test_certification_gated_on_dependencies`, `test_missing_shard_fails`, `test_duplicate_shard_fails`, `test_hard_failures_and_warnings_distinct` |
| **QPROJ-01** (config) | `d_rel`, `d_sem=d_h` | `ModelDimensionConfig` | `test_d_sem_must_equal_d_h`, `test_unresolved_values_not_silently_finalized` |
| **QHP-01** (config) | `d_h=64` primary | `ModelDimensionConfig.d_h` | config defaults / hashing tests |
| **QHP-02** (config) | Train-time scaling must not overwrite raw `X_struct` | `StructuralFeatureConfig.overwrite_raw_with_training_scale=False` | schema construction invariants |
| **QHP-03** (config) | `K=10` primary | `ModelDimensionConfig.K` | config defaults |
| **QHP-04** (config) | KMeans++ init defaults recorded in training config only | `TrainingDefaultConfig` | config-only (no trainer) |
| **D2** | `N=16736`, indices `0..16735` | `NodeUniverseConfig`, node validators | `test_node_index_*`, `test_wrong_node_order_fails`, `test_node_order_hash_mismatch_fails` |

**Hashing / determinism:** `tdmec.hashing` — `test_configuration_hashing_deterministic`, `test_dictionary_ordering_does_not_alter_hash`, `test_nested_determinism_enum_set_dataclass_numpy`, `test_path_rejected_from_scientific_hash`.

**Privacy:** `test_validation_reports_do_not_expose_raw_private_text`, `test_privacy_rejects_external_id_and_email_and_path`.

**Packaging:** `test_package_import_smoke`.
