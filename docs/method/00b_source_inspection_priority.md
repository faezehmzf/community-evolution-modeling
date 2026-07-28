# 00b — Source Inspection Priority (TDMEC)

Recommended reading order for reconstructing the TDMEC method. Priorities reflect
completeness and method relevance. This is a reading plan, not a re-litigation of D1.

## Priority 1 — Complete methodology specifications

| ID | File | Why first |
|---|---|---|
| P-001 | `TDMEC_Methodology.md` | Primary supervisor-approved end-to-end TDMEC methodology (D1). |
| P-002 | `TDMEC_Complete_Technical_Explanation.md` | Supporting technical companion (equations, tensors, losses, training, complexity, FAQ). Cross-check against P-001; honor *Source-stated* labels. |

*Note absent citations in P-002 (`00a`); they do not block D1.*

## Priority 2 — Formal architecture, tensors, losses, training, data schemas

| ID | File | Why |
|---|---|---|
| (within P-002) | §6–§14, §19 | Formal notation, tensor table, loss equations, training algorithm |
| (within P-001) | §7–§23 | Graph/text/edge specs, fusion, GRU, head, hierarchical loss, output schemas |
| P-010 | `temporal_graph_pipeline/README.md` | Output schemas and design principles (data bridge) |
| P-011 | `temporal_graph_pipeline/pipeline_config.json` | Concrete data defaults (`freq=Q`, 4 relations, no self-loops, exact IDs) |

## Priority 3 — Presentations (explanatory only)

| ID | File | Why |
|---|---|---|
| P-003 | `TDMEC_Presentation_Script.md` | Slide-by-slide narrative; intent and emphasis |
| P-004 | `build_tdmec_methodology_presentation.py` | Slide text in-source |
| P-005 | `build_academic_pptx.py` | Academic deck; encoder framing corroboration |

## Priority 4 — Data implementation

| ID | Files | Why |
|---|---|---|
| P-012…P-031 | `temporal_graph_pipeline/` package | How graph and embedding records are built; what is *not* implemented (no model/loss) |
| P-090…P-093 | `pipeline_output/_checkpoints/*` | Which Dataset A parts a real run processed |

*(No notebooks in `Proposed`.)*

## Notes

- Embedding family **RESOLVED to Qwen3 Embedding only** (`USER_CONFIRMED`);
  checkpoint / `D_text` / config **PENDING_PILOT** (Q-EMB, `docs/method/16`).
- D1/D2 already confirmed (`02e`, `12`).
- Prefer P-001 for scientific decisions; use pipeline behavior only when compatible
  with P-001 + verified data + D2.
