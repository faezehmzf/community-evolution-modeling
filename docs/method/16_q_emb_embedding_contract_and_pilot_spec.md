# 16 — Q-EMB: Embedding Contract and Bounded Pilot Specification

**Embedding family:** `USER_CONFIRMED_QWEN3_EMBEDDING_ONLY`. No other embedding-model family may be named, introduced, or compared anywhere in the current TDMEC method.
**Preferred checkpoint:** `Qwen/Qwen3-Embedding-4B` — `PROVISIONAL_PENDING_PILOT`.
**Q-EMB status:** `OPEN_PENDING_EVIDENCE_GROUNDED_PILOT`.
**Scope:** this document specifies the embedding contract boundaries and a bounded, privacy-safe, hardware-agnostic **pilot specification**. It authorizes **no execution**: no model download, no full-data processing, no embedding runs, no commits. A pilot may be executed **only** on explicit user authorization.

Cross-tweet aggregation is already fixed by Q-TEXT (arithmetic mean per node-snapshot and per canonical edge) and is **not** reopened here.

---

## 1. Confirmed vs pending
| Item | Status |
|---|---|
| Qwen3 Embedding family only | `USER_CONFIRMED` |
| `Qwen/Qwen3-Embedding-4B` | `PROVISIONAL_PENDING_PILOT` (any change **within** the Qwen3 family needs a new explicit user decision; no non-Qwen substitution) |
| Exact `D_text` | `PROVISIONAL_PENDING_PILOT` |
| Shared instruction (policy) | `PRIMARY_PILOT_CANDIDATE` (I1); **wording `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28)** |
| No instruction | `PILOT_COMPARATOR` (mandatory) |
| Separate node/edge instructions | `OPTIONAL_PILOT_COMPARATOR` (only after mandatory matrix if compute remains) |
| Exact instruction wording | I1 pilot wording approved; **final production wording = POST_PILOT (QEMB-X03)** |
| Token-level (Stage-A) pooling | `PENDING_OFFICIAL_VERIFICATION` |
| Cross-tweet (Stage-B) mean pooling | `USER_CONFIRMED_CANONICAL_UNDER_Q_TEXT` |
| Per-tweet L2 normalization | `PENDING_OFFICIAL_VERIFICATION` |
| Final pooled-vector L2 normalization | `PRIMARY_PILOT_HYPOTHESIS` (test N2 vs N1) |
| `max_length` | `PROVISIONAL_PENDING_DIAGNOSTICS_AND_PILOT` |
| GPU | `RUNTIME_DETECTED_COLAB_OR_KAGGLE` |
| Pilot gates / matrix / storage eval | QEMB-P01/P02/P03 `USER_CONFIRMED_PILOT_DESIGN` (§14–14c) |
| Q-EMB | `OPEN_PENDING_PILOT_AND_USER_CONFIRMATION` |

**No provisional value above may be propagated into any final tensor contract.**

---

## 2. Execution environment (hardware-agnostic)
Target: hosted GPU notebooks (Google Colab, Kaggle, similar). **Do not assume any specific accelerator.** The pilot/production harness must at runtime:
- detect GPU model, VRAM, system RAM, local + mounted storage;
- record CUDA / PyTorch / Transformers / tokenizer / driver versions and the attention implementation;
- select batch size by **bounded runtime probing**; **safely reduce** batch size after a CUDA OOM (bounded retries, no uncontrolled loop);
- use fp16 and/or bf16 **only** where the detected hardware supports it, and **record the actual inference dtype**;
- be **resumable** after notebook interruption and **portable** between Colab and Kaggle.

Return hardware estimates for several realistic VRAM classes (see §9), not one machine.

---

## 3. Pilot-first model selection
The pilot must determine whether `Qwen/Qwen3-Embedding-4B` is feasible and suitable for the full TDMEC workload. If it fails the acceptance gates (§8), **report the failure and the limiting factor** — do **not** auto-substitute any checkpoint; any change within the Qwen3 family requires a new explicit user decision; no non-Qwen family may be introduced. Evidence required: successful load; GPU compatibility; peak VRAM; stable batch size; throughput (tweets/s); full-corpus runtime estimate; tokenizer throughput; truncation rate; output dimension; storage growth; numerical stability; deterministic repeatability; resume correctness; Persian/multilingual semantic sanity.

---

## 4. Output dimension (`D_text`) — pilot-selected
Do not freeze `D_text` before the pilot. Using **only officially supported Qwen3-Embedding behavior** (to be confirmed against the official model card in the pilot), identify: the **native** output dimension of the 4B checkpoint; officially supported **reduced** dimensions and the documented mechanism (Qwen3-Embedding documents Matryoshka/MRL user-defined output dimensions — **verify** before relying on it); and any official constraints on valid dimensions. **Do not fabricate supported dimensions.**

Pilot a small evidence-grounded set: the **native full dimension**, **one practical reduced dimension**, and **one storage-efficient reduced dimension** (only if officially supported). For each tested dimension report: embedding-quality indicators; Persian semantic-neighborhood consistency; node-snapshot and edge-text vector stability; storage per 1M tweets; estimated full Dataset A and Dataset B embedding storage; final pooled node-tensor size; edge-text artifact size; Fusion-MLP and Edge-Gate input implications; training-memory implications; throughput differences.

The final `D_text` recommendation is made **after** the pilot and requires explicit user confirmation.

---

## 5. Instruction policy (pilot)
It is scientifically reasonable to use **one shared instruction** for both datasets (same encoder; node and edge embeddings should share one comparable semantic space; separate prompts may create artificial node-vs-edge domain shift). The pilot must compare at least:
- **A. No instruction** (`PILOT_COMPARATOR`).
- **B. One shared English instruction** for both A and B (`PRIMARY_PILOT_CANDIDATE`).
- **C. Separate node/edge instructions** (`OPTIONAL_PILOT_COMPARATOR`, analysis only) — test whether it causes prompt-induced (not content-induced) distributional separation between node and edge embeddings.

**Shared instruction candidates** (concise; do not canonicalize before pilot). They must request an accurate semantic representation (topic, stance, sentiment where relevant, agreement/disagreement, hostility/support, interaction context, social/political meaning, Persian/multilingual) and must **not** encode community rules (no "hostile replies ⇒ different communities"):
- I1: `Represent the topic, stance, sentiment, and social meaning of this social-media post for temporal community analysis.`
- I2: `Represent the semantic content, topic, and stance of this social-media post.`
- I3: `Encode the meaning, stance, and sentiment of this post for social analysis.`

Report whether instruction length materially affects token count or embedding quality.

---

## 6. Two distinct pooling levels (kept separate)
- **Stage A — internal encoder token pooling** (token hidden states → one tweet embedding): must follow the **officially documented** Qwen3-Embedding-4B implementation. Verify from official Qwen sources: whether **last-token pooling** is required; padding-side requirements; attention-mask handling; whether the final valid token is selected; whether per-tweet output is officially L2-normalized. **Do not** substitute token mean-pooling unless official evidence explicitly supports it. Status: `PENDING_OFFICIAL_VERIFICATION`.
- **Stage B — Q-TEXT cross-tweet aggregation** (arithmetic mean: per node-snapshot for B; per directed edge/relation/snapshot for A): **already canonical (Q-TEXT); not changed here.**

---

## 7. Pooling & normalization analysis (N1 vs N2)
Proposed pipeline to evaluate: (1) one embedding per valid cleaned tweet via official Stage-A pooling; (2) official per-tweet L2 normalization **if confirmed** by official docs; (3) accumulate valid per-tweet vectors as **float32 sums**; (4) divide by valid count → arithmetic-mean vector; (5) compare two final-vector policies, **separately for B node vectors and A edge vectors**:
- **Policy N1:** keep the raw arithmetic-mean vector, **no** final L2 normalization.
- **Policy N2:** apply L2 normalization to the final pooled vector.

**Mathematical interpretation.** Averaging individually-normalized unit vectors yields a mean whose norm ≤ 1, shrinking as the tweets are more semantically **dispersed** (directional cancellation). Thus under **N1** the pooled-vector norm partly encodes semantic **agreement/concentration/dispersion**; under **N2** magnitude is removed and only **direction** remains. Assess whether norm-based coherence is: scientifically meaningful; stable across tweet count; robust to outlier tweets; confounded by missingness; confounded by tweet count; useful to TDMEC; already represented elsewhere (note: activity volume is already captured by `tweet_count_log1p` and Q-FEAT structural features); likely to destabilize the Fusion MLP or Edge Gate (arbitrary magnitudes).

**Primary hypothesis to test (not canonical):** official Stage-A pooling + per-tweet L2 normalization + float32 mean accumulation + **final pooled-vector L2 normalization (N2)** + store the **pre-normalization pooled norm as a diagnostic value only** (not a model feature; adding it as a feature needs a new explicit user decision). Rationale: node and edge text stay on comparable scales; cosine objectives stable; Fusion MLP / Edge Gate receive no arbitrary magnitude differences; volume already represented structurally; coherence norm still studyable as a diagnostic. The pilot must test this against N1 and recommend N1 or N2 with evidence; the user confirms.

---

## 8. Maximum input length (diagnostics-first)
Do not freeze `max_length` before diagnostics. The atomic input is **one cleaned tweet**, so the model's full context length must **not** automatically become the embedding `max_length`. Measure tokenizer length distributions **separately** for A cleaned event text and B cleaned authored tweets, reporting: median, p90, p95, p99, p99.9, max; % above 128 / 256 / 512 / any proposed limit. Inspect aggregate indicators of extraction-corrupted repetition. **No tweet text or identifiers in outputs.** Use diagnostics to recommend pilot `max_length` candidates; final value confirmed after pilot.

---

## 9. Hardware & storage estimates (indicative; pilot measures actuals)
`Qwen3-Embedding-4B` weights ≈ ~8 GB in fp16/bf16 (verify). Indicative feasibility by VRAM class (batch found by probing):

| VRAM class (example) | 4B load (fp16/bf16) | Expected batch | Note |
|---|---|---|---|
| 16 GB (e.g. T4/V100/P100; Kaggle 2×T4) | fits (~8 GB weights) | small (probe) | headroom-limited; reduce batch on OOM; T4 lacks bf16 → fp16 |
| 24 GB (e.g. L4) | comfortable | moderate | good balance |
| 40 GB+ (e.g. A100) | ample | larger | fastest |

**Storage (indicative), by candidate `D_text` (verify native/reduced against official docs):**

| `D_text` | per-tweet cache (fp16) / 1M tweets | final node tensor `[35, 16736, D]` fp32 | fp16 |
|---|---|---|---|
| 2560 (native, to verify) | ~5.12 GB | ~6.0 GB | ~3.0 GB |
| 1024 (reduced, if MRL-supported) | ~2.05 GB | ~2.4 GB | ~1.2 GB |
| 512 (storage-efficient, if MRL-supported) | ~1.02 GB | ~1.2 GB | ~0.6 GB |

**Implication:** a **permanent full-dimensional per-tweet cache** is likely infeasible on hosted notebooks at scale (e.g. tens of millions of tweets × native dim → hundreds of GB). This favors a **streaming sum/count** or **hybrid resumable** design (§10). Full-corpus tweet/event counts are pilot-measured (2-file pilot ≈ 2.0M rows).

---

## 10. Storage & resume strategy (hypothesis)
Do not assume permanent full-dimensional per-tweet storage exists. Compare: **A.** permanent per-tweet cache; **B.** temporary sharded per-tweet cache; **C.** streaming group sums + counts (no permanent per-tweet storage); **D.** hybrid resumable with shard-level accumulation. **Recommended production hypothesis (evaluate, not final):** deduplicate before embedding; process deterministic input shards; write shard-completion manifests; optionally retain compressed per-tweet embeddings only where reuse is needed; maintain **float32 sum + integer count** accumulators; build node-snapshot and edge-(snapshot,relation) aggregates; prevent double counting after resume; validate model/tokenizer/preprocessing/instruction/dimension/max-length **hashes** before resuming. Return storage estimates per candidate `D_text`.

---

## 11. Pilot sampling (bounded, privacy-safe, stratified)
**Dataset B:** multiple snapshots; low/medium/high-activity users; short/long tweets; Persian and multilingual tweets; tweets with URLs/mentions/hashtags/media markers where retained. **Dataset A:** all four relations; incoming and outgoing directions via event records; single-event and multi-event edges; short/long event text; Persian/multilingual event text; reply/quote interactions where semantic disagreement may occur. **Never expose** raw tweet text, account IDs, tweet IDs, graph endpoints, or private identity-bearing paths in Git-visible reports — aggregate statistics and sanitized semantic checks only.

---

## 12. Semantic pilot tests (privacy-safe)
Evaluate (Persian + multilingual): similar posts close; unrelated posts separated; conflicting **stance** distinguishable from topic similarity; hostile interaction text remains topically similar but stance-aware; node-snapshot pooled vectors stable under resampling; edge-text pooled vectors not dominated by one outlier event; shared instruction improves semantic organization; separate instructions create artificial node-vs-edge separation; reduced `D_text` preserves nearest-neighbor structure; final L2 normalization changes neighborhood rankings. **Do not** claim embeddings determine community membership — they provide semantic evidence to learned TDMEC components.

---

## 13. Computational pilot logging (every configuration)
Record: environment (Colab/Kaggle/other); exact GPU; VRAM; system RAM; storage; model revision; tokenizer revision; Transformers/PyTorch/CUDA versions; attention implementation; inference dtype; output dtype; accumulation dtype; batch size; `max_length`; `D_text`; instruction hash; preprocessing hash; tweets/s; tokens/s (where available); peak allocated + reserved VRAM; model-load time; embedding time; cache growth; NaN/Inf/zero-vector/abnormal-norm counts; deterministic-repeat difference. Batch size reduces after **bounded** OOM failures (no uncontrolled retry loop).

---

## 14. Pilot acceptance gates (`USER_CONFIRMED` pilot design — QEMB-P01 Option C)

### 14.1 Hard gates (must pass)
- Checkpoint loads in the detected environment; no unsupported architecture/tokenizer error.
- **Zero** NaN/Inf embeddings.
- Deterministic repeatability: max absolute cosine deviation ≤ **1e-4** on repeated encode of the same inputs.
- Stable batch size found; no unrecoverable CUDA OOM after bounded backoff.
- Node and edge Stage-B pooling **exactly** match independently recomputed test cases.
- Resume does not duplicate contributions; Q-MISS mask/exact-zero invariants hold after load/dtype/batch/serialize.
- Shared instruction causes no obvious semantic collapse.
- Normalized embeddings have expected norm behavior (per N1/N2 condition).
- Sanitized Persian semantic checks acceptable.
- Reduced dimensions preserve semantic-neighbor structure: mean **Overlap@k ≥ 0.9** vs native (definition below).

### 14.2 Truncation (primary target, reported in strata)
- Primary target: truncation rate ≤ **1%** under the chosen pilot `max_length` candidate.
- Report truncation **overall** and by **Dataset A vs B**, **language strata** (where detectable), and **length strata**.
- Failure to meet 1% does not auto-fail the whole pilot if hard gates pass; it must be reported and drives `max_length` recommendation (POST_PILOT).

### 14.3 Report-only planning metrics (not hard fail)
- Estimated full-corpus runtime (depends on GPU, batch, dtype, token lengths).
- Peak and final storage estimates (depends on hosted storage performance and dim).
- These inform planning and the pilot decision report; they do **not** alone fail the pilot.

### 14.4 Neighbor-overlap metric (explicit before implementation)
| Field | Definition |
|---|---|
| **k** | **10** |
| **Score** | For each query embedding, `Overlap@k = \|NN_k(native) ∩ NN_k(reduced)\| / k` using cosine nearest neighbors in the same pilot sample pool (self excluded). |
| **Aggregation** | **Mean** Overlap@k over the evaluation query set. Hard gate: mean ≥ **0.9**. |
| **Evaluation sample size** | Up to **1,000** stratified query embeddings (or all eligible if fewer). Stratify across Dataset A event texts and Dataset B authored tweets represented in the pilot sample; within each, cover short/long and relation mix where applicable. Same query set used for all reduced dims under comparison. |
| **Privacy** | No raw text/IDs in reports — only aggregate Overlap@k and stratum sizes. |

---

## 14b. Pilot comparator matrix (`USER_CONFIRMED` design — QEMB-P02 Option A)

**Mandatory (first pilot; avoid full Cartesian product — prefer sequential comparisons):**
1. Checkpoint: `Qwen/Qwen3-Embedding-4B` only (no auto-substitution).
2. Instructions: (i) **no instruction**; (ii) **one shared instruction** with **I1 as primary candidate**.
3. I2/I3: **not** in the mandatory first pilot.
4. Separate node/edge instructions: **optional**, only after the mandatory matrix completes and compute remains.
5. Dimensions: native full + one practical reduced + one storage-efficient reduced, **only if officially supported**.
6. Normalization: compare **N1 and N2** on pooled vectors.
7. `max_length`: choose pilot candidates **only after** token-length diagnostics.
8. Sampling: stratified privacy-safe coverage for Dataset A and B (§11).

**I1 wording:** `USER_APPROVED_PILOT_CANDIDATE` (2026-07-28):

> Represent the topic, stance, sentiment, and social meaning of this social-media post for temporal community analysis.

Approved **only** as the shared instruction candidate for the bounded pilot comparison against no-instruction. Exact final production instruction remains POST_PILOT (QEMB-X03) if pilot evidence suggests a change.

---

## 14c. Storage/resume evaluation plan (`USER_CONFIRMED` design — QEMB-P03 Option B)

**Quantitative pilot comparison required:**
- **C.** streaming group sums + counts only  
- **D.** hybrid resumable processing  

Measure for C and D: correctness, peak storage, final storage, write overhead, interruption recovery, resume correctness, double-count protection, manifest integrity, aggregate equivalence (node/edge means match independent recompute).

**Analytical / small-sample estimates only (not full pilot-scale implementations unless a supported reduced dim makes permanent caching plausible):**
- permanent per-tweet cache  
- temporary full sharded per-tweet cache  

Hybrid remains a **pilot hypothesis**. Production storage/resume policy remains **POST_PILOT**.

---

## 15. Required pilot decision report (before finalizing Q-EMB)
Produce a compact report: (1) is 4B feasible; (2) environments/GPUs tested; (3) recommended checkpoint; (4) recommended `D_text`; (5) instruction policy; (6) exact instruction wording; (7) `max_length`; (8) verified token-level pooling; (9) final pooled-vector normalization; (10) inference dtype; (11) storage dtype; (12) accumulation dtype; (13) batch-size policy; (14) cache/resume strategy; (15) full Dataset A runtime; (16) full Dataset B runtime; (17) storage requirements; (18) all failed configurations; (19) unresolved risks; (20) exact decisions still needing user approval. **Do not finalize Q-EMB automatically; wait for explicit user confirmation.**

---

## 16. Constraints for this phase
No non-Qwen embedding model may be named/introduced/compared. Do not propagate provisional values into the final tensor contract. Do not write the full embedding implementation. A bounded pilot **notebook specification** may be designed but **not executed** without explicit instruction. I1 pilot wording is approved; pilot execution still requires separate explicit authorization. Do not download the model, process full datasets, or stage/commit/push.
