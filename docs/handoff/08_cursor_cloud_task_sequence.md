# 08 — Cursor Cloud Task Sequence

Ordered small tasks reflecting **current** artifact state. Do not start embeddings or training before prerequisites and open decisions are closed.

---

### T00 — Readiness gate
| Field | Content |
|---|---|
| Objective | Confirm docs hierarchy, D1/D2, no Proposed access needed |
| Fixed | D1, D2 |
| Prerequisites | Repo clone |
| Read | `docs/handoff/07`, `docs/method/03`, `09`, `10` |
| Modify | None (or checklist only) |
| Tests | None |
| Env | Cloud CPU |
| Forbidden | Data processing, training |
| Next | T01 |

### T01 — Enforce D2 node-map contract (code review + tests)
| Field | Content |
|---|---|
| Objective | Ensure builder cannot emit N≠16736 or expand from B |
| Fixed | D2 |
| Prerequisites | T00 |
| Read | `scripts/build_node_index_map.py`, `src/tdmec_pilot/node_map.py`, `docs/method/11` |
| Modify | Allowed after local implementation lands; Cloud may review/test |
| Tests | `test_node_map_d2.py` (length, deterministic order, reject expansion) |
| Real-data pilot | Optional dry-run counts only |
| Gates | Assert N==16736 |
| Forbidden | Publishing uncertified map as CERTIFIED without checksum |
| Next | T02 |

### T02 — Synthetic relation-event unit tests
| Field | Content |
|---|---|
| Objective | Lock 4 relations, direction, self-loop exclusion, safe parse |
| Fixed | R=4, directed, no self-loops |
| Prerequisites | T01 |
| Read | Method `04` A05; pilot/discovery parsers |
| Modify | Test files / future `dataset_a` module |
| Tests | Synthetic xlsx/rows |
| Forbidden | Full A processing |
| Next | T03 |

### T03 — Snapshot calendar decision binding
| Field | Content |
|---|---|
| Objective | Implement config-driven quarterly calendar; produce the 10-item coverage report; obtain user boundary confirmation |
| Fixed | **Quarterly frequency = USER_CONFIRMED (Q-CAL)**; monthly = sensitivity variant only; provisional pilot calendar 2017-Q4→2026-Q2 (35 bins); empty snapshots kept; out-of-range excluded with reason codes; invalid/epoch-outlier timestamps classified separately; calendar bounds **must be runtime config, never hard-coded** |
| Prerequisites | T01–T02 |
| Blocked if | Exact boundaries still `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS` → may run diagnostics on provisional calendar, but **must not label any full-data artifact `calendar-certified`** until user confirms boundaries after reviewing coverage report |
| Coverage report (10 items, A+B) | per-quarter row/event counts; unique active nodes/quarter; A edges per relation×quarter; B valid-text/quarter; before/after-range counts & %; excluded-tail timestamp distribution; empty/sparse snapshots; graph-only/text-only/graph+text coverage/snapshot; corruption/epoch evidence; extend/shorten consequences |
| Next | T04 |

### T04 — Dataset A graph pilot (controlled)
| Field | Content |
|---|---|
| Objective | Build edges for small shard set; validate schemas |
| Fixed | D2, relations, log1p; **Q-DEDUP** conflict-aware collapse (A composite signature, **float id forbidden as key**); `c` = distinct events after reconciliation |
| Prerequisites | T01–T03 |
| Outputs | Temp edges + validation report + duplicate diagnostic (candidate/concordant/discordant groups, rates by file/author/snapshot/relation, effect on edge counts) |
| Gates | No new nodes; self-loops 0; relation set exact; concordant collapsed, discordant retained+flagged; raw files unchanged |
| Env | Colab CPU + Drive |
| Forbidden | Declare CERTIFIED without checksum + review |
| Next | T05 |

### T05 — Dataset B full normalize (after pilot scale-up plan)
| Field | Content |
|---|---|
| Objective | Run 70-file normalization reconciled to frozen map |
| Prerequisites | T01, T03; pilot code as base |
| Fixed | **Q-DEDUP** B records via exact string IDs + conflict check; **Layer-2 in-field text cleaning** (conservative; original text kept private + separate cleaned field + hashes; embeddings use cleaned text) |
| Outputs | Partitioned parquet + coverage report + duplicate/in-field-cleaning diagnostic |
| Gates | unmatched→drop; N unchanged; exact tweet IDs preserved; concordant collapsed / discordant retained+flagged; meaningful repetition preserved; no full text in Git-visible manifests |
| Forbidden | Embeddings in this task |
| Next | T06 |

### T06 — Structural features + masks
| Field | Content |
|---|---|
| Objective | Build `X_struct[T,N,17]` (float32) + `struct_active_mask[T,N]` (bool) |
| Fixed (Q-FEAT) | **F_struct=17**; ordered schema: per relation {out_degree,in_degree,out_strength_log1p,in_strength_log1p} for mention(0–3)/retweet(4–7)/reply(8–11)/quote(12–15) + `16 tweet_count_log1p`. Degree=distinct neighbors (no log1p); strength=`log1p(Σ count_raw)`; tweet_count=`log1p(#distinct authored tweets)`. `active`/`n_active_relations` **not** features → separate `struct_active_mask`. Self-loops excluded before degree/strength; multi-target mention=one event/target; tweet_count counts once. No additional normalization here (**QHP-02** = train-time robust scaling only; do not standardize canonical artifacts). |
| Prerequisites | T04 (edges) + node map |
| Outputs | `X_struct`, `struct_active_mask`, ordered 17-name list, schema version, relation+snapshot maps, node-map hash, Q-DEDUP provenance, dtype/shape, validation summary |
| Gates | shapes `(T,16736,17)`/`(T,16736)`; finite; nonnegative; inactive→zero row+mask False; only-incoming→active; tweets-no-edges→active; multi-target changes edge features not tweet_count; self-loops don't affect degree/strength; deterministic ordering; identical inputs→identical hashes |
| Next | T07 |

### T07 — Embeddings (blocked until O-EMB)
| Field | Content |
|---|---|
| Objective | Offline embeddings — **T07a node text (Dataset B) FIRST**, then **T07b edge text (Dataset A) SECOND** (both required) |
| Fixed (Q-TEXT) | node: per-distinct-tweet embed → mean-pool `T_i^(t)` over in-snapshot B tweets → `X_node_text[T,N,D_text]`; edge: per-distinct-event embed → mean-pool `E_{i→j}^{(t,r)}` per canonical edge; **cleaned_text only**; strict in-snapshot; dedup before embed; cache keyed by content+provenance (not float A id); no A↔B tweet join |
| Prerequisites | **Q-EMB** (encoder/`D_text`) still open; **Q-MISS resolved** (M1 mask/zero semantics — `docs/method/17`); T05 (B corpus) for node; T04 (A edges+event text) for edge |
| Env | GPU Colab |
| Forbidden | Starting while Q-EMB unresolved; inventing alternate missing-text policies (follow Q-MISS M1); embedding raw (non-cleaned) text; concatenating snapshot tweets into one string |
| Next | T08 |

### T08 — Pack model-ready tensors
| Field | Content |
|---|---|
| Prerequisites | T04–T07 certified |
| Outputs | Parquet + `.pt`/safetensors + manifests |
| Gates | Shape checks; N=16736; checksums |
| Next | T09 |

### T09 — Model smoke (graph-only)
| Field | Content |
|---|---|
| Prerequisites | T08; Batch 4 primary pins (`d_h=64`, `K=10`) already confirmed |
| Forbidden | Full training claiming results |
| Next | T10 |

### T10 — Text-aware smoke + staged train
| Field | Content |
|---|---|
| Prerequisites | T07–T09 |
| Next | T11 |

### T11 — Evaluation + evolution
| Field | Content |
|---|---|
| Prerequisites | Trained checkpoints |
| Next | Done / thesis analysis |

---

## Dependency summary

```
T00 → T01 → T02 → T03(quarterly fixed; boundaries REVIEW_REQUIRED) → T04(Q-WGT) → T06(Q-FEAT: F_struct=17)
                     ↓
                    T05 (full B)
                     ↓
              T07(O-EMB) → T08 → T09 → T10 → T11
```

**Hard stops:** T07 without O-EMB; T08 without certified upstream; T10 without tensors.
