# 00 — TDMEC Source Inventory (Proposed + active project)

**Scope.** Inventory of **TDMEC-relevant** materials under the external directory
`C:\Users\mrsmo\Documents\UT\Semester 6\Thesis\Proposed` and pointers into this
repo's docs. Phase-1 style: no method invention. Nothing in `Proposed` was modified;
no `Proposed` file is copied into Git.

> **Access note.** Where OS size/timestamps were unavailable, sizes are approximated
> from line counts or marked `nc` (not captured). Timestamps are not used for recency.

---

## 1. Overview (TDMEC materials)

TDMEC-relevant areas under `Proposed/`:

- **Root method documents** — Markdown specs (P-001, P-002, P-003) and TDMEC deck
  builders (P-004, P-005).
- **`temporal_graph_pipeline/`** — tested Python package for TDMEC-aligned **data
  construction** (Dataset A → directed multiplex graph; Dataset B → embedding-ready
  text records). Encoder-agnostic; no model code.
- **`pipeline_output/_checkpoints/`** — resume checkpoints from a
  `temporal_graph_pipeline` run.
- Tooling caches (`.pytest_cache/`) — not method-relevant.

Present TDMEC sources are Markdown / Python / JSON (text-readable). Generated
`.pptx` decks from the builders are **not** in the directory (`00a`).

---

## 2. Inventory table (TDMEC only)

### 2.1 Root — method documents & TDMEC deck builders

| ID | Rel. path | Type | Access | Apparent purpose | Size | Conf |
|---|---|---|---|---|---|---|
| P-001 | `TDMEC_Methodology.md` | md | FULLY_READABLE | **Primary methodology specification** (supervisor-approved end-to-end) | ~1014 ln | H |
| P-002 | `TDMEC_Complete_Technical_Explanation.md` | md | FULLY_READABLE | **Supporting technical companion** — architecture, tensors, losses, training, eval | ~1388 ln | H |
| P-003 | `TDMEC_Presentation_Script.md` | md | FULLY_READABLE | Presentation speaker script (slide-by-slide) | ~1183 ln | H |
| P-004 | `build_tdmec_methodology_presentation.py` | py | FULLY_READABLE | Builds `TDMEC_Methodology_Presentation.pptx` (deck not present) | ~357 ln | H |
| P-005 | `build_academic_pptx.py` | py | FULLY_READABLE | Builds `TDMEC_Academic_Presentation.pptx`; names a multilingual encoder (family user-confirmed **Qwen3 Embedding** — Q-EMB `docs/method/16`) | nc | H |

### 2.2 `temporal_graph_pipeline/` — TDMEC data pipeline (no model)

| ID | Rel. path | Type | Apparent purpose | Conf |
|---|---|---|---|---|
| P-010 | `temporal_graph_pipeline/README.md` | md | Design principles + output schema | H |
| P-011 | `temporal_graph_pipeline/pipeline_config.json` | json | Config: `snapshot_freq=Q`, `include_self_loops=false`, 4 relations, seed 42 | H |
| P-012 | `…/config.py` | py | `PipelineConfig` dataclass | M |
| P-013 | `…/schema.py` | py | Output table/tensor schemas | M |
| P-014 | `…/cli.py` | py | CLI entry | M |
| P-015–P-016 | `__main__.py`, `__init__.py` | py | Package entry | M |
| P-017 | `…/phase1/pass1_metadata.py` | py | Pass A1: node map + snapshot metadata | M |
| P-018 | `…/phase1/pass2_events.py` | py | Pass A2: edge events + tweets | M |
| P-019 | `…/phase1/node_map.py` | py | Node universe / index map | M |
| P-020 | `…/phase1/snapshots.py` | py | Calendar snapshots | M |
| P-021 | `…/phase1/aggregation.py` | py | DuckDB edge aggregation | M |
| P-022 | `…/phase1/validation.py` | py | Graph validation report | M |
| P-023 | `…/phase2/discover.py` | py | Dataset B shard discovery | M |
| P-024 | `…/phase2/filter_extract.py` | py | Filter/extract to Core authors | M |
| P-025 | `…/phase2/embed_records.py` | py | `dataset_b_embedding_records.parquet` (raw text; embedding separate) | M |
| P-026–P-029 | `…/utils/*` | py | Checkpoint, streaming XLSX, safe parse, exact user IDs | M |
| P-030–P-031 | `…/tests/*` | py | Synthetic tests + fixtures | M |

### 2.3 Run checkpoints

| ID | Rel. path | Apparent purpose | Conf |
|---|---|---|---|
| P-090…P-093 | `pipeline_output/_checkpoints/*part_00{1–4}*.json` | Pass-1 checkpoints (Dataset A parts) | M |

### 2.4 Excluded tooling noise

`.pytest_cache/` — **METADATA_ONLY**; ignore for method work.

---

## 3. Topic coverage (TDMEC sources)

| Topic | Specs P-001/002/003 | `temporal_graph_pipeline` |
|---|---|---|
| Research objective | ✔ | – |
| Dataset A / B roles | ✔ | ✔ |
| Graph construction / 4 relations / directed | ✔ | ✔ |
| Quarterly snapshots | ✔ | ✔ (`freq=Q`) |
| Node / edge text roles | ✔ | ✔ (raw text; embed stage separate) |
| Qwen3 Embedding family | user-confirmed (Q-EMB); not named in Proposed files | encoder-agnostic |
| Edge-gated GraphSAGE / masked fusion / GRU | ✔ | – |
| Prototype Student-t / fixed-K sweep | ✔ | – |
| Hierarchical losses | ✔ | – |
| Baselines / ablations / eval | ✔ | validation report only |
| Implementation (data) | contract | ✔ |

---

## 4. Source groups

- **Methodology:** P-001 (primary), P-002 (supporting technical).
- **Explanatory:** P-003, P-004, P-005.
- **Data implementation:** `temporal_graph_pipeline/` (P-010…P-031) + checkpoints
  (P-090…P-093). No model/loss/training code in these packages.

---

## 5. Complete method specifications

- **P-001** — primary end-to-end methodology.
- **P-002** — expanded technical companion (equations, tensors, complexity, FAQ).

Authority hierarchy: **D1** (`02e`); document map: `01_method_version_timeline.md`.

---

## 6. Initial observations

1. TDMEC specs + data pipeline are present; **no TDMEC model code** in `Proposed`.
2. Embedding family is **user-confirmed Qwen3 Embedding only**; exact checkpoint /
   `D_text` **PENDING_PILOT** (Q-EMB `docs/method/16`). Proposed files did not name
   Qwen/Qwen3.
3. P-002 cites some absent siblings (`00a`); graph detail needed for TDMEC is in
   P-001 + `temporal_graph_pipeline`.
4. Active-project contracts live under `docs/method/03`+ and `docs/handoff/`.

See `00a_unreadable_or_partial_sources.md` and `00b_source_inspection_priority.md`.
