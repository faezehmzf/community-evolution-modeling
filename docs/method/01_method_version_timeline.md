# 01 — TDMEC Source Chronology & Document Map

**Scope.** Map of TDMEC-relevant sources used to reconstruct the method. No code.
Source IDs (`P-###`) refer to `00_proposed_source_inventory.md`.

---

## 1. TDMEC source groups

| Group | Role | Sources |
|---|---|---|
| **Methodology** | Primary scientific specification | P-001 `TDMEC_Methodology.md` |
| **Technical companion** | Supporting equations, tensors, training detail | P-002 `TDMEC_Complete_Technical_Explanation.md` |
| **Presentations** | Explanatory narrative / deck builders | P-003 script; P-004, P-005 PPTX builders |
| **Data implementation** | Graph / snapshot / embedding-record construction | `temporal_graph_pipeline/` (P-010…P-031) + run checkpoints (P-090…P-093) |

---

## 2. Document map (what each source contributes)

| Source | Contributes |
|---|---|
| **P-001** | End-to-end methodology: objective, Dataset A/B roles, directed 4-relation multiplex, quarterly snapshots, node/edge text, edge-gated GraphSAGE, masked fusion, single GRU, prototype Student-t (fixed-K), hierarchical losses, training stages, baselines, ablations, evaluation |
| **P-002** | Formal expansions of P-001: tensor tables, loss equations (α=1, DEC target P, JS temporal), complexity, inference, FAQ; self-labels *Source-stated* vs *Interpretation* vs *Derived* |
| **P-003** | Slide-by-slide speaker narrative; corroborates fixed-K sweep, single GRU, encoder framing (`D_text` PENDING_PILOT — Q-EMB `docs/method/16`) |
| **P-004 / P-005** | Deck builders (slide text in-source); P-005 names a multilingual encoder (family user-confirmed **Qwen3 Embedding** — Q-EMB) |
| **`temporal_graph_pipeline/`** | Runnable data construction: 4 relations, `freq=Q`, directed, no self-loops by default, exact string IDs, parquet outputs (`node_map`, `snapshots`, `edges`, `dataset_b_embedding_records`, `external_targets`); **no model / loss / embedding** |

---

## 3. Reading order (practical)

1. P-001 (primary scientific authority — D1)
2. P-002 (supporting technical — *Source-stated* binding; *Interpretation/Derived* non-binding)
3. `temporal_graph_pipeline/README.md` + `pipeline_config.json` (data schemas and defaults)
4. P-003 / P-005 for narrative corroboration only (must not override P-001)

---

## 4. Confirmed decisions that anchor this map

- **D1:** P-001 primary scientific authority; P-002 supporting technical; `temporal_graph_pipeline` data-construction authority when compatible with P-001 + verified data + D2.
- **D2:** Canonical node universe **N = 16,736** (indices `0…16735`); Dataset B must not expand nodes; historical **10,040** is not the modeled universe.
- **Q-EMB:** Embedding family **Qwen3 Embedding only**; checkpoint / `D_text` / config **PENDING_PILOT** (`docs/method/16`).

---

## 5. Absent TDMEC-adjacent references

P-002 cites `sections/Stage1_TemporalMultiplexGraph.md` and `METHODOLOGY.md`, which are **not present** under `Proposed/`. Graph-construction detail needed for TDMEC is available in P-001 §§5–14 and in `temporal_graph_pipeline`. Absences are documentation-completeness items; they do not block the D1 authority model (`02e`).

---

## 6. What is *not* in these sources

No in-repo TDMEC **model** code exists yet. Specs define architecture and losses; `temporal_graph_pipeline` covers data artifacts only. Implementation follows `03`–`11` and the handoff contracts.
