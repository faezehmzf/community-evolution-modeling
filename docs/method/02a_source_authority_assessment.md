# 02a — Source Authority Assessment (TDMEC Sources)

Authority assessment for **TDMEC-relevant** sources used when reconstructing the
canonical method. Qualitative ratings only (High / Medium / Low). This is not the
final D1 decision (see `02e`).

Dimensions: **Completeness**, **Consistency**, **Math precision**, **TDMEC
alignment**, **Data compatibility**, **Presentation-only?**, **Implementation
evidence**, **Likely authority**, **Confidence**.

---

## 1. P-001 `TDMEC_Methodology.md` — primary methodology specification

- Completeness: **High** — end-to-end (objective → data → graph → text → encoder →
  fusion → GRU → head → losses → training → baselines → ablations → eval → roadmap).
- Consistency: **High**.
- Math precision: **Medium-High** — core equations present; some hyperparameters
  left to be tuned.
- TDMEC alignment: **High** (defines the terminology).
- Data compatibility: **High** on structure (4 relations, directed, quarterly, fixed
  universe, log1p, external-summary policy); **Medium** where a tweet_id-style dedup
  key meets float-lossy Dataset A ids (see `02` C-12 / `02c` R4).
- Presentation-only? **No**.
- Implementation evidence: partial via `temporal_graph_pipeline` (data stages).
- Likely authority: **High** — supervisor-level verdict / approved methodology.
- **Confidence: High.**

## 2. P-002 `TDMEC_Complete_Technical_Explanation.md` — supporting technical

- Completeness: **High** — tensor tables, complexity, decoder form, FAQ, failure modes.
- Consistency: **High**; self-labels *Source-stated* vs *Interpretation* vs *Derived*.
- Math precision: **High** (α=1, Student-t, DEC target P, JS temporal).
- TDMEC alignment: **High**.
- Data compatibility: **Medium-High** (same tweet_id caveat; symbolic dims).
- Presentation-only? **No**.
- Likely authority: **High for Source-stated; Medium for Interpretation/Derived**.
- Caveat: cites some absent sibling docs (`00a`); honor fidelity labels.
- **Confidence: High** (with fidelity caveat).

## 3. P-005 `build_academic_pptx.py` — presentation (most complete deck)

- Completeness: **Medium-High** for a deck; names a multilingual text encoder
  (specific name omitted; family later user-confirmed **Qwen3 Embedding** — Q-EMB
  `docs/method/16`).
- Presentation-only? **Yes** — must not override formal specs.
- Likely authority: **Medium** (corroboration of encoder framing).
- **Confidence: Medium-High.**

## 4. P-004 `build_tdmec_methodology_presentation.py` — concise deck

- Completeness: **Medium**; **no** embedding model named.
- Presentation-only? **Yes.** Authority: **Medium**.
- **Confidence: Medium.**

## 5. P-003 `TDMEC_Presentation_Script.md` — speaker script

- Completeness: **High narrative**; not a formal spec.
- Corroborates fixed-K sweep, single GRU, multilingual-encoder framing
  (`D_text` PENDING_PILOT).
- Presentation-only? **Yes.** Authority: **Medium**.
- **Confidence: Medium-High** as corroboration.

## 6. `temporal_graph_pipeline/` (P-010/P-011+) — data-construction authority

- Completeness: **High for data construction**; **none** for model/loss/embedding.
- Precision: **High** (schemas, DuckDB SQL, exact-ID policy, safe parsing).
- TDMEC alignment: **High** on data decisions (4 relations, directed, `Q`, no
  self-loops default, encoder-agnostic, leakage-eligibility flag).
- Data compatibility: **High** (built around Dataset A/B quirks).
- Implementation evidence: **Strong** (runnable, tested).
- Authority: **High for how data artifacts are built**; not for model/embedding.
- Caveat: historical node-universe = canonical ∪ non-canonical-A-authors must obey
  **D2** (N = 16,736 only) when used for the modeled set.
- **Confidence: High** (within data scope).

---

## Consolidated findings

| Role | Source |
|---|---|
| Primary scientific specification | **P-001** |
| Supporting technical specification | **P-002** (*Source-stated* binding) |
| Data-artifact construction | **`temporal_graph_pipeline`** when compatible with P-001 + verified data + D2 |
| Explanatory only | P-005 > P-003 > P-004 |

**Unsafe for overriding scientific decisions:** presentation decks for exact
math/tensors; any content depending solely on absent referenced files (`00a`).

Final authority model: **`02e` (D1)**.
