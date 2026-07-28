# 00a — Unreadable, Partial, or Absent Sources (TDMEC)

Honest accounting of access gaps for **TDMEC reconstruction**. Content of absent
files is not inferred.

## 1. Binary document formats — none present

Extension search for `pptx, docx, pdf, ppt, doc, txt, ipynb, png, jpg, jpeg, gif,
webp` under `Proposed/` returned **0 files**. Present TDMEC sources are text
(Markdown/Python/JSON) and **FULLY_READABLE**.

## 2. Referenced-but-absent sources (TDMEC-relevant)

`TDMEC_Complete_Technical_Explanation.md` (P-002) cites sibling documents that are
**not present** in `Proposed/`:

| Referenced path (as cited in P-002) | Status | Why it matters for TDMEC | Action |
|---|---|---|---|
| `Proposed/sections/Stage1_TemporalMultiplexGraph.md` | **ABSENT** (no `sections/` dir) | Cited as Stage-1 input specification for the temporal multiplex graph | Provide as Markdown if finer Stage-1-only detail is needed; P-001 §§5–14 + `temporal_graph_pipeline` already cover graph construction for D1 |
| `Proposed/METHODOLOGY.md` | **ABSENT** | Cited as a supervisor-approved / consolidated design sibling | Optional for provenance; P-001 itself contains the supervisor verdict — does not block D1 |

These are **absent references**, not unreadable files. They are
documentation-completeness items (`02e`).

## 3. Generated presentation decks — builders present, outputs absent

| Item | Status | Notes |
|---|---|---|
| `TDMEC_Methodology_Presentation.pptx` (P-004 target) | **ABSENT** | Builder readable; slide text inspectable in `.py` |
| Deck(s) from P-005 `build_academic_pptx.py` | **ABSENT** | Same |

Rendered visual artifacts (layout/diagrams) are unavailable; authored slide text is
readable in the builders.

## 4. Partially inspected (accessible; full read may be deferred)

- P-003 `TDMEC_Presentation_Script.md` (long; sampled early, later read for Phase 2).
- P-004 deck builder (sampled then read).
- Individual `temporal_graph_pipeline` modules — purpose clear from README/config;
  deep reads as needed for data contracts.

## 5. Metadata-only / irrelevant

`.pytest_cache/` (root and under `temporal_graph_pipeline/`) — tooling noise; ignore.

## 6. Environment limitation (historical note)

Where the Shell tool could not capture OS file sizes/timestamps, inventory columns
use approximate line counts or `nc`. That does not affect readability of contents.
