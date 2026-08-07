# 04 — Complete Data-to-Model Pipeline (Dataset A & B)

**Authority:** P-001 (scientific), verified data contracts, V3 when compatible with D1/D2.  
**N = 16,736** (D2).

Status tags per stage:
- `IMPLEMENTED_V3` — `temporal_graph_pipeline`
- `IMPLEMENTED_REPO` — current repo discovery/pilot
- `METHOD_ONLY` — specified, not implemented
- `OUTDATED` — conflicts with verified data or D2
- `USER_CONFIRMATION_REQUIRED`

---

## Part A — Dataset A pipeline

### A00 — Source inventory & access
| Field | Contract |
|---|---|
| Purpose | Locate 12 `.xlsx` + `extraction_summary.json` |
| Input | Drive/local Dataset A path |
| Output | Manifest with hashes |
| Status | `IMPLEMENTED_REPO` (discovery) |
| Artifact | Inventory docs; not model-ready |
| Env | Local/Cloud read |
| Consumer | All A stages |

### A01 — Schema validation
| Field | Contract |
|---|---|
| Purpose | Confirm 31-column schema |
| Input | A workbooks |
| Output | Schema registry |
| Status | `IMPLEMENTED_REPO` |
| Failure | Abort on schema drift |
| Consumer | A02+ |

### A02 — Exact account extraction → immutable node map
| Field | Contract |
|---|---|
| Purpose | Build frozen map `user.id → node_idx ∈ {0…16735}` |
| Input | Distinct `user.id` from A (verified N=16736) |
| Output | `node_index_map.parquet` + checksum |
| Deterministic | Sort exact digit-string IDs ascending |
| Identifier | Never float-cast IDs |
| Status | Builder `IMPLEMENTED_REPO`; map **not certified**; V3 may add non-canonical authors → **must follow D2** (exclude expansion) |
| Failure | Count ≠ 16736 → fail closed |
| Env | Local/Cloud CPU |
| Consumer | All stages |

### A03 — Timestamp normalization
| Field | Contract |
|---|---|
| Purpose | Parse `created_at` to UTC |
| Output | Normalized timestamps; invalid → reject/exclude with reason |
| Status | V3 + pilot patterns `IMPLEMENTED_V3` / `IMPLEMENTED_REPO` |
| Consumer | A06 |

### A04 — Duplicate classification
| Field | Contract |
|---|---|
| Purpose | Classify exact/conflicting duplicates |
| Constraint | A tweet `id` float-lossy → do not use as exact cross-dataset key; prefer provenance `(source_file, source_row)` + structural keys |
| Policy | **USER_CONFIRMATION_REQUIRED** (annotate-only vs drop) |
| Status | Detected in forensics; no certified policy |
| Consumer | A05–A07 |

### A05 — Relation event extraction
| Field | Contract |
|---|---|
| Purpose | Emit directed events for retweet/reply/quote/mention |
| Parsing | Safe parse blobs (`ast.literal_eval`/`json`, never `eval`) |
| Direction | author → target |
| Mentions | Pairwise to Core targets; unique per tweet (method + V3) |
| Self-loops | Excluded (method + V3 default) |
| External | `dst` outside map → external summary, not graph node |
| Multi-relation | One tweet may emit multiple relations |
| Status | `IMPLEMENTED_V3` |
| Output | Relation event partitions (+ raw_text on events) |
| Consumer | A07–A08, edge text |

### A06 — Snapshot assignment
| Field | Contract |
|---|---|
| Purpose | Assign each event to calendar snapshot |
| Main freq | Quarterly (method); V3 default `Q` |
| Empty snaps | Kept |
| Tail | **USER_CONFIRMATION_REQUIRED** |
| Status | Logic `IMPLEMENTED_V3` / pilot; calendar not certified |
| Consumer | Aggregation |

### A07 — Aggregate edges & weights
| Field | Contract |
|---|---|
| Purpose | Aggregate `(snapshot_id, relation, src_idx, dst_idx)` |
| Weights | `event_count`, `weight_raw=count`, `weight_log1p=ln(1+count)` (method + V3) |
| Status | `IMPLEMENTED_V3` |
| Output | `edges.parquet` |
| Artifact status | **Not certified** in repo/Drive |
| Consumer | Features, masks, GNN |

### A08 — Activity masks & structural features
| Field | Contract |
|---|---|
| Purpose | Build `active_mask`, relation masks, `X^(t)` |
| Features | Method recommends counts/activity/optional followers; exact `F_struct` **USER_CONFIRMATION_REQUIRED** |
| Status | `METHOD_ONLY` (not certified tensors) |
| Consumer | Model input |

### A09 — Edge-text construction (A-derived)
| Field | Contract |
|---|---|
| Purpose | Attach relation-specific interaction text to aggregated edges; embed offline |
| Source | Dataset A event text (authoring / nested status texts per P-001 §11) |
| Join | No A↔B tweet-level join |
| Status | Events carry `raw_text` in V3; aggregated edge-text + embeddings **METHOD_ONLY** |
| Blocked by | Embedding model decision; text-unit confirm |
| Consumer | Edge gate, L_struct decoder |

### A10 — Model-ready graph package
| Field | Contract |
|---|---|
| Purpose | Export edges, features, masks as Parquet/tensors |
| Status | `METHOD_ONLY` |
| Artifact | **No certified model-ready artifact found** |

---

## Part B — Dataset B pipeline

### B00 — Source inventory
| Field | Contract |
|---|---|
| Purpose | 70 `statuses-*.xlsx` + manifest (~9.58 GB) |
| Status | `IMPLEMENTED_REPO` (12 deep / 58 inventory) |
| Note | Full deep inspect incomplete — data limitation |

### B01 — Schema & ID preservation
| Field | Contract |
|---|---|
| Columns | `id, created_at, user, text, likes, retweets, reply_count, quoted_count, bookmarks, views` |
| Tweet ID | Exact string — preserve |
| User | Serialized structured object — safe parse only |
| Status | `IMPLEMENTED_REPO` + pilot |

### B02 — Author extraction & reconciliation
| Field | Contract |
|---|---|
| Purpose | Extract `user.id`; map to frozen indices |
| Rule | Unmatched authors dropped; **never add nodes** (D2) |
| Pilot evidence | 496/496 matched on 2 files |
| Status | Pilot `IMPLEMENTED_REPO`; full 70-file not run |
| Consumer | All B stages |

### B03 — Timestamp & snapshot assignment
| Field | Contract |
|---|---|
| Purpose | Assign tweets to same canonical calendar as A |
| Out-of-range | Pilot excluded 12,831 post-calendar rows |
| Status | Pilot `IMPLEMENTED_REPO`; full corpus unexecuted |
| Tail | Shared with A — USER_CONFIRMATION |

### B04 — Duplicate classification
| Field | Contract |
|---|---|
| Pilot | Exact groups=747; conflicting=0; report-only (none dropped) |
| Full policy | Align with certified policy when chosen |
| Status | Pilot annotate/report |

### B05 — Conservative text normalization & quality
| Field | Contract |
|---|---|
| Pilot | Text preserved/normalized conservatively; no stemming; no language filter; no embedding |
| Status | Pilot `IMPLEMENTED_REPO` |
| Consumer | Text-unit stage |

### B06 — Atomic text-unit construction
| Field | Contract |
|---|---|
| Method default | Per-tweet embed → mean-pool per `(node_idx, snapshot_id)` |
| Status | **METHOD_ONLY / USER_CONFIRMATION_REQUIRED** if departing from default |
| Blocks | Embedding stage |

### B07 — Text embedding (offline)
| Field | Contract |
|---|---|
| Encoder | Family **USER_CONFIRMED: Qwen3 Embedding only** (Q-EMB `docs/method/16`); exact checkpoint (`Qwen/Qwen3-Embedding-4B` preferred)/config `PENDING_PILOT` |
| Frozen | Yes |
| Status | Not implemented; no embedding libs in `requirements.txt` |
| Env | GPU Colab |
| Artifact | None |

### B08 — Aggregation & text masks
| Field | Contract |
|---|---|
| Output | `X_node_text[T,N,D_text]`, `node_text_available_mask[T,N]` bool, `node_valid_text_count[T,N]` (Q-MISS); missing → exact zero |
| Missing | Zero + mask 0 |
| Status | `METHOD_ONLY` |

### B09 — Model-ready text package
| Field | Contract |
|---|---|
| Status | Full 70-file normalized corpus **not certified**; embeddings absent |
| Statement | No certified model-ready text artifact found |

### Pilot vs full pipeline (mandatory distinction)

| Aspect | 2-file pilot | Full 70-file |
|---|---|---|
| Files | `statuses-2`, `statuses-69` | All 70 |
| Rows | in 2,004,845; retained 1,992,014 | Unknown until run |
| Tests | 31 passed | N/A |
| Normalization corpus | Temporary/local pilot outputs | Not certified |
| Embeddings | None | None |
| Role | Controlled validation | Production Dataset B path |

**Do not describe the pilot as the final Dataset B pipeline.**

---

## Part C — Graph–text alignment

| Alignment key | Status |
|---|---|
| Exact `user.id` | Reliable A↔B account join |
| Immutable `node_idx` | Required after map certification |
| `snapshot_id` | Shared calendar |
| Relation / src / dst | Graph side (A) |
| Provenance file/row | Event identity when tweet_id unsafe |
| Tweet-level A↔B join | **Not proven**; A float `id` unsafe |
| Edge text | **A-derived** per method; B-sourced per-edge text unsupported |

---

## Stage dependency graph (simplified)

```
A00→A01→A02 → A03→A04→A05→A06→A07 → A08 → A09 → A10 ─┐
                                                      ├→ Model inputs
B00→B01→B02 → B03→B04→B05→B06→B07→B08→B09 ───────────┘
```

Critical path risk: data construction (A02–A07, B02–B06) before any embedding or training.
