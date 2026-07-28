# 08 — Scientific and Engineering Risk Register

Severity: **CRITICAL** / **HIGH** / **MEDIUM** / **LOW**.

| ID | Severity | Component | Evidence | Scientific consequence | Engineering consequence | Mitigation | Modifies method? | User approval? |
|---|---|---|---|---|---|---|---|---|
| R01 | CRITICAL | Temporal text | P-001 leakage checklist | Future text in past snapshots invalidates temporal claims | Wrong T_i^(t) construction | Strict snapshot-local pooling; eligibility flags | No | No |
| R02 | CRITICAL | Edge masking | P-001 L_struct | Masked edges in encoder → inflated reconstruction | Data leak in training | Remove masked edges from encoder input | No | No |
| R03 | HIGH | A tweet_id | `docs/data/02` float-lossy | Wrong joins / false uniqueness | Broken dedup/join | Provenance IDs; never float reconstruct; no A↔B tweet join | Spec key adaptation | Confirm dedup policy |
| R04 | HIGH | Duplicates | 1.17M A dups flagged | Duplicate amplification of edges/weights | Unstable graphs | Certified annotate/drop policy | Policy choice | **Yes** |
| R05 | HIGH | Hub domination | Social degree skew | Communities centered on hubs | Unstable training | log1p weights; mean agg; edge gates | No | No |
| R06 | HIGH | Relation imbalance | Sparse quote/mention | Degenerate attention to one relation | Poor multiplex use | Masked fusion; report per-layer stats; entropy diagnostics | No | No |
| R07 | HIGH | Edge-text source | Handoff B18 | Unsupported B-sourced edge text | Impossible join | Use A-derived edge text only | No | Confirm if required |
| R08 | HIGH | Prototype collapse | DEC failure mode | Single community | Useless Q | L_reg; KMeans reinit; monitor effective K | No | No |
| R09 | HIGH | Label permutation | Soft clustering | Fake “migration” across snapshots | Misleading evolution plots | Alignment heuristic + report uncertainty | Analysis only | No |
| R10 | MEDIUM | Inactive nodes | P-001 GRU freeze | Stale state after long inactivity | Drift | Document freeze; optional reset ablation later | Ablation only | No |
| R11 | MEDIUM | Missing text bias | Sparse B coverage | Semantic loss on subset only | Biased L_sem | Q-MISS: masks; L_sem mask-True only; required coverage reports; missing-text sensitivity | No | Done (Q-MISS) |
| R12 | MEDIUM | Snapshot aliasing | Quarterly windows | Mix distinct phases | Unstable NMI | Monthly sensitivity | Ablation | Tail confirm |
| R13 | MEDIUM | K sensitivity | Fixed K | Arbitrary partitions | Unstable metrics | Sweep {5…30}; report stability | No | Choose K for main table |
| R14 | MEDIUM | Eval circularity | Modularity-only | Meaningless “good” clusters | False success | Require multi-metric improvement | No | No |
| R15 | MEDIUM | V3 vs D2 universe | V3 may add non-canonical authors | Wrong N | Wrong tensors | Enforce N=16736 map only | Impl constraint | D2 confirmed |
| R16 | MEDIUM | Encoder choice | Family Qwen3-only (USER_CONFIRMED); config PENDING_PILOT (Q-EMB) | Dim/cost/semantic shift | Incomparable runs | Pin one encoder for all ablations | Possible | Family confirmed; config pilot |
| R17 | MEDIUM | Null engagement cols | A 8 cols 100% null | Using null features | NaNs/noise | Exclude null columns from F | Feature choice | **Yes** (F list) |
| R18 | LOW | Reproducibility | Neighbor sampling RNG | Seed variance | Hard reruns | 5 seeds; log SHA/config | No | No |
| R19 | HIGH | Artifact false claims | Code ≠ artifact | Claiming readiness falsely | Cloud wrong start | Status vocabulary; fail-closed | Process | No |

## Unsupported interpretations
- Causal claims about community causes of events — not supported.
- Ground-truth NMI/ARI against labels — unavailable.
