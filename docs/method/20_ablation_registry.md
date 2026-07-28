# 20 — Ablation Registry (QEVAL-03)

**Status:** `USER_CONFIRMED_VIA_AUTHORIZED_RESOLUTION` (2026-07-28).  
**Sequencing:** **Do not implement** ablations until Phase 11 — after TDMEC-Full trains, metrics validated, and primary results exist. Prefer **configuration switches**, not forked model code.

## Naming (QVAR-01)

| ID | Meaning | Class |
|---|---|---|
| TDMEC-G | Graph only | primary claim ablation |
| TDMEC-NT | Graph + node text (no edge text) | primary claim ablation |
| TDMEC-ET | Graph + edge text (no node text) — reserved | primary claim ablation (if implemented) |
| TDMEC / TDMEC-Full | Graph + node + edge text | **proposed method** (not an ablation) |

## Registry (complete — none deleted)

| Ablation ID | Intervention | Claim tested | Class |
|---|---|---|---|
| ABL-G | TDMEC-G | Graph sufficient? | primary claim |
| ABL-NT | TDMEC-NT | Node text value | primary claim |
| ABL-ET | TDMEC-ET | Edge text without node text | primary claim / conditional |
| ABL-FULL | TDMEC-Full | Full method reference | reference |
| ABL-REL-AGG | Collapse relations **before** message passing | Multiplex vs aggregated **encoder** | primary claim |
| ABL-FUS-MEAN | Keep relations; replace masked fusion with mean over available relations | Learned fusion vs mean | primary claim |
| ABL-SYM | Symmetrize edges (lose direction) | Direction matter? | primary claim |
| ABL-BIN | Binary weights vs log1p | Weight transform | primary claim |
| ABL-NO-GRU | Remove temporal GRU (snapshot-independent head on z) | Temporal encoder | primary claim |
| ABL-LTEMP0 | Keep GRU; `λ_temp=0` entire joint+temporal | Temporal loss | primary claim |
| ABL-LSEM0 | `λ_sem=0` | Semantic alignment | primary claim |
| ABL-LSTRUCT0 | `λ_struct=0` | Structural reconstruction | supplementary / conditional |
| ABL-LREG0 | `λ_reg=0` | Prototype separation | primary claim |
| ABL-SUM-AGG | Sum aggregation vs mean (QENC-01) | Agg choice | optional / sensitivity-adjacent |
| ABL-NOSCALE | No QHP-02 robust scaling | Feature scaling | optional |
| ABL-HARDNEG | Degree-aware or hard negatives | Negatives | optional |
| ABL-RELTYPE | Add `L_relation_type` | Auxiliary loss | optional |
| ABL-L2 | Depth L=2 | Depth | required sensitivity |
| ABL-DH | `d_h ∈ {32,128}` | Width | required sensitivity |
| ABL-K | `K ∈ {5,10,15,20,30}` | Community count | required sensitivity |
| ABL-M | `m ∈ {0.5,1.0,2.0}` | Margin | required sensitivity |
| ABL-BPTT | BPTT ∈ {2,4} | Truncation | required sensitivity |
| ABL-DREL | `d_rel=32` | Relation emb | optional sensitivity |
| ABL-DROPOUT | Dropout 0.1 in MLPs | Regularization | optional |
| ABL-DETACH | Detach every snapshot | Credit assignment | optional computational |

### Critical separation

**ABL-REL-AGG** ≠ **ABL-FUS-MEAN**.  
- Relation aggregation removes multiplex message passing.  
- Fusion mean keeps per-relation GNN outputs but changes fusion only.

**QEVAL-03 decision:** full ablation registry retained; classes assigned; implementation deferred to Phase 11.
