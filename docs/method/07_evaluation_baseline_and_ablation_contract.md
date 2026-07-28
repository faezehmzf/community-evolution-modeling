# 07 — Evaluation, Baseline, and Ablation Contract

**Status:** Batches 6–7 closed by authorized resolution (2026-07-28).  
**Full metric formulas & QEVAL-01:** `docs/method/18_evaluation_metrics_and_selection_contract.md`.  
**Baseline registry (QEVAL-02):** `docs/method/19_baseline_registry.md` — implement only in Phase 10.  
**Ablation registry (QEVAL-03):** `docs/method/20_ablation_registry.md` — implement only in Phase 11.

## 1. Philosophy
Convergent validity across structural, temporal, semantic, predictive, stability, and efficiency. No Dataset A/B ground-truth communities. ARI only on synthetic/annotated data.

## 2. Mandatory metrics (summary)
See `18` for formulas, attribution labels, eligibility, and edge cases.  
Primary predictive: **future-link relation-macro AP**.  
Primary partition similarity over time: **AMI (AMIsum)**.  
Primary structural: **directed weighted modularity** (`count_raw`) + symmetrized weighted conductance.

## 3. Model selection (summary)
Early stop: phase-appropriate smoothed validation loss (`18` §1.2).  
Config select: validation relation-macro AP subject to hard non-collapse rules + deterministic tie-break.  
Never use test for stopping/tuning/K/tie-break.

## 4. Variants
TDMEC-G / TDMEC-NT / TDMEC|TDMEC-Full / reserved TDMEC-ET (**QVAR-01**). Full is the method; G/NT/ET are ablations.

## 5. Fairness
Same N, calendar, split, seeds (QTR-04), shared eval masks (`18`), Native-K and Matched-K (±20% or ±2).

## 6. Status
Evaluator not implemented; begins Phase 8 after TDMEC-Full primary training path exists.
