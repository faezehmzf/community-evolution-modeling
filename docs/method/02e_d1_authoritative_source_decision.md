# 02e — Decision D1: Authoritative Method Source (TDMEC)

**Scope.** Decide **D1**: treat `Proposed/TDMEC_Methodology.md` (**P-001**) as the
**primary authoritative scientific source**, with
`Proposed/TDMEC_Complete_Technical_Explanation.md` (**P-002**) as a **supporting
technical source**, and place `temporal_graph_pipeline` as data-construction
authority when compatible with P-001 + verified data + D2.

**Out of scope here:** embedding config, snapshot policy, text unit, structural
features, dedup diagnostics, model hyperparameters. No code.

---

## 1. Question being decided

> May the canonical method use **P-001** as primary scientific authority and
> **P-002** as supporting technical authority — and do absent referenced siblings
> block canonicalization?

## 2. Sources examined

| ID | Path | Role |
|---|---|---|
| P-001 | `Proposed/TDMEC_Methodology.md` | Candidate primary (read in full) |
| P-002 | `Proposed/TDMEC_Complete_Technical_Explanation.md` | Candidate supporting (read in full) |
| P-003 | `Proposed/TDMEC_Presentation_Script.md` | Presentation |
| P-004 | `Proposed/build_tdmec_methodology_presentation.py` | TDMEC deck builder |
| P-005 | `Proposed/build_academic_pptx.py` | TDMEC deck builder |
| — | `Proposed/temporal_graph_pipeline/` | Data-construction implementation |
| — | Repo `docs/method/00–02d`, `docs/data/*`, `docs/handoff/*` | Prior findings + verified data |
| absent | `sections/Stage1_TemporalMultiplexGraph.md`, `METHODOLOGY.md` | Referenced only; not present |

## 3. Evidence for P-001 authority

| # | Location | Claim (constructive paraphrase) | Conf. |
|---|---|---|---|
| EV1 | P-001 preamble | Specifies the thesis methodology from raw X/Twitter data to temporal community trajectories. | High |
| EV2 | P-001 §1 Supervisor-Level Verdict | Approved main model includes: temporal directed multiplex graph; node text T_i^(t); edge text E_ij^(t,r); edge-gated GraphSAGE; masked fusion; GRU; prototype Student-t; compact hierarchical loss. | High |
| EV3 | P-001 §31 | Method approved as a thesis methodology with the condition that implementation remain disciplined. | High |
| EV4 | P-001 §§4–29 | Self-contained pipeline: data → graph → text → encoder → fusion → GRU → head → losses → training → baselines → ablations → evaluation. | High |
| EV5 | P-002 source-fidelity | P-002 states it is based on the TDMEC methodology specification (`TDMEC_Methodology.md`) — treats P-001 as its source. | High |

**D1:** P-001 presents itself as the current supervisor-approved methodology
specification (EV1–EV4); P-002 acknowledges P-001 as its basis (EV5).

## 4. Limitations (do not undermine scientific authority)

| # | Limitation | Impact |
|---|---|---|
| L1 | P-001 notes final acronym selection after architecture freeze | Naming only |
| L2 | Some parameters left to be tuned / diagnostics-dependent | Handled as open decisions (D2+, Q-*) |
| L3 | P-002 also cites absent `METHODOLOGY.md` | Provenance documentation gap; P-001 itself contains the supervisor verdict |
| L4 | Spec tweet_id-style dedup vs float-lossy A ids | Data-compatibility (`02c`); pipeline uses provenance/aggregation |

## 5. Relationship between P-001 and P-002

**Consistent and hierarchical (P-002 elaborates P-001).**

- No contradiction on scientific decisions (`02` C-01): edge weight `log(1+c)`,
  mean-pool text, edge gate γ, masked fusion, GRU, Student-t α=1, hierarchical
  losses, JS temporal, DEC target P.
- P-002-only material: tensor tables, complexity, inference, failure modes, FAQ,
  example dims (e.g. d_h=64) — either *Interpretation/Derived* (non-binding) or
  expansion of P-001.
- → P-002 is **safe as supporting technical source** when *Source-stated* content
  is binding and *Interpretation/Derived* is not.

## 6. Presentation sources

| Source | Role |
|---|---|
| P-003 | Explanatory narrative; fixed-K, single GRU, encoder framing |
| P-004 / P-005 | Explanatory decks; P-005 corroborates multilingual-encoder framing (Qwen3 family via Q-EMB) |

Presentations simplify but must not override P-001/P-002.

## 7. Role of `temporal_graph_pipeline`

Implements the P-001 **data contract** within its scope:

- 4 relations {retweet, reply, quote, mention}; directed; self-loops excluded by default.
- `log(1+count)` weights; calendar snapshots `freq=Q`; empty snapshots kept.
- Exact string IDs; safe blob parsing; external-target summaries; raw text +
  leakage-eligibility flag.

Scope limits / flags:

- Historical node-universe union with non-canonical A authors — **must obey D2**
  (N = 16,736 only) for the modeled set.
- Structural edges / parquet only — no edge-text embedding stage, no `.pt` tensors,
  no model.
- Dedup via provenance/aggregation rather than float tweet_id (aligned with verified
  data).

**Conclusion:** authoritative for verified **data-construction behavior** when
compatible with P-001 + verified data + D2.

## 8. Missing-source dependency

### `sections/Stage1_TemporalMultiplexGraph.md`
- Cited as Stage-1 input spec. Equivalent graph construction is in P-001 §§5–14 and
  the pipeline. **Does not block canonicalization.**

### `METHODOLOGY.md`
- Cited as a consolidated / supervisor-approved sibling. TDMEC approval text is in
  P-001 §1 and §31. **Does not block canonicalization.** Optional for provenance.

## 9. Category-specific authority

| Category | Primary | Supporting | Explanatory |
|---|---|---|---|
| Research objective | P-001 | P-002 | P-003 |
| Graph (scientific) | P-001 | P-002 | P-003 |
| Text (node/edge) | P-001 | P-002 | P-003, P-005 |
| Architecture / losses / training / eval | P-001 | P-002 (*Source-stated*) | P-003 |
| Data implementation | `temporal_graph_pipeline` | P-001 data contract | — |
| Mathematical detail | P-002 (*Source-stated*) | P-001 | — |

## 10. Recommended D1 decision

- **P-001 = primary scientific authority.**
- **P-002 = supporting technical authority** (*Source-stated* binding;
  *Interpretation/Derived* non-binding).
- **`temporal_graph_pipeline` = data-construction authority** when compatible with
  P-001 + verified data + D2.
- **P-003/P-004/P-005 = explanatory only.**
- **Absent files do not block canonicalization.**

**Status: `D1_RESOLVED_FROM_EVIDENCE`.** Confidence: **High** on hierarchy;
**Medium** only on optional `METHODOLOGY.md` provenance.

## 11. User confirmation (recorded)

**User-confirmed:** D1 authority model as above.  
**Also user-confirmed (D2):** canonical node universe **N = 16,736**, indices
`0…16735`, immutable; Dataset B must not expand the universe; historical **10,040**
is not the modeled universe. Data-construction authority applies only when
compatible with D1/D2.

Canonical reconstruction continues in `docs/method/03`+.
