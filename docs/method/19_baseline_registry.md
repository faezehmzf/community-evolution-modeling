# 19 — Baseline Registry (QEVAL-02)

**Status:** `USER_CONFIRMED_VIA_AUTHORIZED_RESOLUTION` (2026-07-28).  
**Sequencing:** **Do not implement** any baseline until Phase 10 — after TDMEC-Full succeeds, artifacts certified, evaluator validated, and primary TDMEC results exist.  
**TDMEC-Full** is the proposed method, **not** a baseline. TDMEC-G / NT / ET remain **ablations** (see `20`) even if shown beside baselines in tables.

## Registry (complete candidate set — none deleted)

| Official name | Primary citation / source family | Family | Dir | Wgt | Temp | Multiplex | Node attr | Edge text | K behavior | Feasibility | Table status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Aggregated Leiden | Traag et al. / Leiden algorithm; aggregate relations+time | Classical static community | via undirected aggregate | yes | no (aggregate) | collapsed | no | no | inferred | High | **main** |
| Per-Snapshot Leiden | Leiden per snapshot + temporal matching | Classical temporal matching | via undirected | yes | snapshot-wise | per-rel optional | no | no | inferred | High | **main** |
| Per-Snapshot Infomap | Rosvall & Bergstrom MapEquation | Flow / information | yes (flow) | yes | snapshot-wise | limited | no | no | inferred | Medium–High | **main** |
| Multislice modularity | Mucha et al. 2010 | Multilayer modularity | typically undirected slices | yes | interlayer coupling | yes | no | no | inferred | Medium | **main** |
| Peixoto layered / edge-valued / time-varying SBM | Peixoto graph-tool / SBM literature | Statistical SBM | yes | yes | yes | yes | limited | no | inferred (K-free) | Medium (expertise/compute) | **main** if feasible else **conditional** |
| GCN+GRU+community head | Kipf&Welling GCN + GRU + same Student-t/KMeans head protocol | Neural temporal | often symmetrized | yes | yes | weak | features | no | fixed-K matched | High | **main** |
| GraphSAGE+GRU+community head | Hamilton et al. + GRU + head | Neural temporal | yes | yes | yes | weak | features | no | fixed-K | High | **supplementary** |
| R-GCN+GRU+community head | Schlichtkrull et al. R-GCN + GRU + head | Neural multiplex | yes | yes | yes | yes | features | no | fixed-K | High | **main** |
| DySAT+KMeans | Sankar et al. DySAT | Dynamic graph attention | typically undirected | yes | yes | weak | features | no | KMeans K | Medium | **supplementary** |
| HDMI+KMeans (or successor attributed multiplex encoder)+KMeans | Attributed multiplex representation learning literature | Attributed multiplex RL | varies | yes | limited | yes | yes | no | KMeans K | Conditional | **conditional / supplementary** |
| Text-Only KMeans | Clustering on node-text embeddings only | Text baseline | n/a | n/a | snapshot text | n/a | text | no | fixed-K | High | **main** |
| node2vec+KMeans | Grover & Leskovec | Static embedding | via undirected | binary/weight | no | collapsed | no | no | KMeans | High | **supplementary** |
| Static-text (no temporal) KMeans | Same text, ignore time / pool all | Text ablation-as-baseline | n/a | n/a | no | n/a | text | no | fixed-K | High | **supplementary** |

### Capability notes (claims tested)

| Baseline | Covers | Absent vs TDMEC-Full |
|---|---|---|
| Aggregated Leiden | Community structure on collapsed graph | Direction, multiplex, time, text |
| Per-Snapshot Leiden | Time-varying partitions | Soft assignments, text, directed multiplex encoder |
| Infomap | Flow communities | Soft TDMEC head, edge text |
| Multislice | Temporal multilayer coupling | Edge text, directed edge gating |
| Peixoto SBM | Generative layered/temporal/weighted | Edge text semantics, TDMEC losses |
| GCN/SAGE+GRU | Temporal neural communities | Relation-specific gates, edge text |
| R-GCN+GRU | Relational + temporal | Edge-text conditioning, masked fusion |
| DySAT+KMeans | Dynamic attention embeddings | Multiplex relations, edge text |
| HDMI-like+KMeans | Attributed multiplex | Temporal GRU path, edge text |
| Text-Only KMeans | Semantics only | Graph structure |

### Implementation source expectations (later Phase 10)

- Leiden: `leidenalg` / igraph  
- Infomap: official Infomap  
- Multislice: published multilayer modularity optimizers or documented reimplementation  
- Peixoto: `graph-tool`  
- Neural: PyTorch Geometric / DGL under same split, seeds, features fairness rules  

### Fairness (when implemented later)

Same N, calendar, train/val/test, seed policy (QTR-04), eval masks (`18`), Native-K + Matched-K views, no test tuning.

**QEVAL-02 decision:** full registry preserved and extended; **no deletions**; implementation deferred to Phase 10.
