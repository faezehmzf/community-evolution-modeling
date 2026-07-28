# 21 — CONSOLIDATED_PRE_IMPLEMENTATION_CONTRACT_AND_GAP_AUDIT

**Date:** 2026-07-28  
**Verdict:** Core TDMEC **method implementation is now methodologically unambiguous** for coding Phases 1 and 5–9, subject only to evidence-dependent values listed in §12 (calendar/dedup/coverage/Q-EMB/runtime).  
**No further conceptual user-question batch remains.**

## 1. Data contract (summary)
- **D2:** N=16736 frozen; B cannot expand.  
- **QREL-01 / QSELF-01 / Q-WGT / Q-FEAT / Q-TEXT / Q-MISS / QACT / QGRU / QGATE:** as in `12`.  
- **Q-CAL:** quarterly; internal empty kept; bounds POST_DIAGNOSTIC.  
- **Q-DEDUP:** policy+PROC confirmed; signature/L2 POST_DIAGNOSTIC.  
- Provisional diagnostic calendar 2017-Q4…2026-Q2 allowed for pilots only.

## 2. Artifact / certification
- QART-01-FRAME hard gates; coverage reported (numeric thresholds POST_DIAGNOSTIC).  
- Sparse edges; dense node tensors `[T,N,*]`; checksums/manifests/provenance required.

## 3. Tensor shapes (primary)
| Tensor | Shape |
|---|---|
| X_struct | `[T,N,17]` |
| T_node | `[T,N,D_text]` |
| masks | `[T,N]` / edge-aligned |
| h,z,s | `[N,d_h]` per t; `d_h=64` |
| μ | `[K,d_h]`; K=10 |
| Q | `[N,K]` |

## 4. Architecture (exact)
See `05`: concat → MLP_x (Linear-ReLU-Linear); per-relation gated mean SAGE L=1 fanout 15; masked fusion; GRU; Student-t α=1; QFUS fallback z=h0; γ scalar; dropout 0; ReLU.

## 5. Decoder / losses
See `06` + QDEC-01/QCLU-01/QLOSS-*: BCE decoder on s; DEC P/epoch; JS L_temp on model_active pairs; λ and phases 100 / 80 / 120 with 24-epoch temp ramp.

## 6. Training
AdamW 5e-4; wd 1e-4; clip 1.0; patience 20; node mini-batches; epoch = chrono pass; BPTT 3; split ~70/15/15 POST_CAL; seeds QTR-04.

## 7. Selection & metrics
See `18`: early-stop smoothed val loss; select val relation-macro AP + collapse guards + tie-break; formulas with attribution labels.

## 8. Baselines & ablations
Registries `19` and `20` — **no implementation before Phases 10–11**.

## 9. Implementation sequence (binding)

| Phase | Work |
|---|---|
| 1 | Contracts, schemas, config, invariant tests |
| 2 | Real-data diagnostics (calendar, dedup, length, coverage) |
| 3 | Bounded Q-EMB pilot |
| 4 | Finalize evidence-dependent decisions |
| 5 | Certified model-ready artifacts |
| 6 | TDMEC-Full core model |
| 7 | Losses + trainer |
| 8 | Common evaluator + mandatory metrics |
| 9 | Successful primary TDMEC train + primary results |
| 10 | Baselines |
| 11 | Ablations / sensitivities |
| 12 | Multi-seed comparison + final reporting |

## 10. Test plan (Phase 1+)
- Schema/shape/dtype/order/relation-map/self-loop/mask/zero/model_active/checksum tests.  
- Determinism + resume + no double-count.  
- Synthetic tiny-graph forward/loss finite checks before full data.

## 11. Certification gates
As QART-01-FRAME; no CERTIFIED without listed invariants; calendar-certified / dedup-certified only after user approval of diagnostic outcomes.

## 12. Remaining evidence-dependent (not conceptual batches)
- Calendar start/end/T/leading-trailing  
- Dedup signature + L2 thresholds  
- Numeric coverage hard thresholds  
- QEMB-X01…X07  
- Runtime batch size / AMP / OOM  
- Which expensive baselines remain feasible; sensitivity outcomes  

## 13. Gap audit result
| Area | Status |
|---|---|
| Scientific method topology | Closed |
| Trainable module shapes | Closed (Batch 7) |
| Losses / phases / batching | Closed |
| Eval selection / formulas | Closed (`18`) |
| Baseline/ablation *intent* | Closed (registries); *code* deferred |
| Full-data numeric freezes | Open (diagnostics/pilot) |

**Conclusion:** After operational authorization, begin **Phase 1** (schemas/config/invariant tests). Do not start baselines/ablations. Do not run pilots without separate authorization.
