# 18 — Evaluation Metrics, Eligibility, and Model Selection (QEVAL-01)

**Status:** `USER_CONFIRMED_VIA_AUTHORIZED_RESOLUTION` (2026-07-28).  
**Authority:** Active TDMEC contracts override papers for project-specific extensions. Primary papers verify base formulas only.  
**Formula labels:** `PUBLISHED_EXACT` · `PUBLISHED_EQUIVALENT` · `PROJECT_SPECIFIC_EXTENSION` · `PROJECT_SPECIFIC_AGGREGATION` · `INVALID`.

---

## 1. QEVAL-01 — Evaluation and selection contract (frozen)

### 1.1 Mandatory reporting dimensions

| Dimension | Mandatory metrics | Role |
|---|---|---|
| Structural | Directed weighted modularity (`weight_raw`); weighted symmetrized conductance; relation-macro versions | Quality of partition on graph |
| Temporal | Consecutive AMI (arithmetic-mean / AMIsum); consecutive NVI; migration rate (descriptor) | Stability / change descriptors — **not** absolute quality alone |
| Semantic | Cosine silhouette on a **shared frozen evaluation encoder** | Embedding coherence |
| Predictive | Future-link **relation-macro AP** (primary); micro-AP; new-links-only AP (complementary) | Primary predictive probe |
| Stability | Seed pairwise-AMI consensus; edge-perturbation AMI (when feasible) | Robustness |
| Efficiency | Wall-clock runtime; peak CPU RAM; peak GPU VRAM | Cost |
| Ground-truth-only | ARI | Synthetic / annotated subsets **only** — not Dataset A/B primary |

### 1.2 Early stopping vs configuration selection

| Phase | Early-stop monitor (validation) | Classification |
|---|---|---|
| Representation pretrain | Smoothed val `L_repr = λ_struct L_struct + λ_sem L_sem` (unweighted components also logged) | `PRIMARY_EXPERIMENTAL_DEFAULT` |
| Joint (λ_temp=0) | Smoothed val `L_total` with current λ (temp=0) | `PRIMARY_EXPERIMENTAL_DEFAULT` |
| Temporal | Smoothed val `L_total` with active λ including ramped λ_temp | `PRIMARY_EXPERIMENTAL_DEFAULT` |

Smoothing: exponential moving average with α=0.1 over epoch-end scalars (`PRIMARY_EXPERIMENTAL_DEFAULT`).

**Final configuration selection (validation only):**

1. Discard any run failing **hard validity / non-collapse** (§1.3).  
2. Among valid runs, maximize **validation future-link relation-macro AP** (§4.5).  
3. **Tie-break (deterministic, in order):** higher val directed weighted modularity (`weight_raw`, relation-macro mean) → higher val cosine silhouette → higher seed-stability AMI (if multi-seed already available for that config; else skip) → lower val NVI (consecutive mean) → lower peak GPU memory → lexicographically smaller config id / seed.  
4. **Never** use test for stopping, tuning, K selection, or tie-break.

**Classification:** selection-by-val-macro-AP + convergent safeguards = `IMMUTABLE_METHOD_CONTRACT` for fairness; numeric EMA α = `PRIMARY_EXPERIMENTAL_DEFAULT`.

### 1.3 Hard rejection / collapse rules

Reject a checkpoint/config if any hold:

- NaN/Inf in losses or assignments  
- Empty partition or all nodes in one community on ≥50% of validation snapshots with ≥1 eligible node  
- Soft-assignment mean entropy > `log(K) - 1e-3` on ≥50% of val snapshots (near-uniform collapse)  
- Any community with size 0 after hard labels on all val snapshots  
- Decoder/future-link eval produces undefined AP on **all** eligible val cells (no usable predictive signal)

Report rejection reason codes.

### 1.4 Test restrictions

Test snapshots: final reporting only after selection frozen. No gradient, no fitting, no selection, no tie-break.

### 1.5 Native-K vs Matched-K

- **Native-K:** method’s natural/selected K.  
- **Matched-K:** train/val-only attempt to achieve community count within **±20%** of TDMEC reference K (or ±2 communities if that is larger); no test; record target, tolerance, achieved counts. Do not discard outputs solely for imperfect K.  
- When K differs: AMI remains primary partition similarity (`PUBLISHED` AMI is label-permutation invariant — **no Hungarian matching**).

---

## 2. Common evaluation universe (method-independent)

### 2.1 Shared masks (from certified data)

| Mask | Definition |
|---|---|
| `eval_node_mask[t,i]` | `True` iff node `i` is in the frozen N=16736 map and is **structurally or text-eligible** for evaluation at `t`: `struct_active_mask[t,i] OR node_text_available_mask[t,i]` (same spirit as model_active; stored separately if needed as copy for eval) |
| `pair_eligible[t-1,t,i]` | `eval_node_mask[t-1,i] AND eval_node_mask[t,i]` **and** both snapshots have a hard community label for `i` from the method under test |
| `graph_weight_W[t,r]` | `Σ_{ij} w_ij^(t,r)` with primary `w = count_raw` |

If a method fails to label an eligible node, that node is **ineligible for that method’s partition metrics** but the shared mask definition does not shrink for other methods. Methods are comparable on the **intersection of pair_eligible under each method’s labels** for pairwise temporal metrics; also report coverage (% labeled). Prefer also reporting metrics on the **shared intersection across all methods in a table** when comparing multiple methods in one table (`PROJECT_SPECIFIC_AGGREGATION`).

### 2.2 Edge cases

| Case | Rule |
|---|---|
| Active in only one of t−1,t | Excluded from consecutive AMI/NVI/migration for that pair |
| Missing label | Excluded from partition metrics; counted in coverage deficit |
| Singleton community | Allowed; silhouette contribution 0 (Rousseeuw convention) |
| Empty partition / all-in-one | Hard reject if triggers §1.3; else report metric as undefined/NaN with flag |
| Baseline fewer/more communities | Native-K reporting; Matched-K optional view |
| Method cannot model a relation | That relation’s structural/predictive cells marked ineligible; excluded from relation-macro denominator |

---

## 3. Metric formulas

### 3.1 Directed weighted modularity

**Indexing (TDMEC):** `w_ij` = weight of directed edge **i → j**.  
`s_i^{out} = Σ_j w_ij`, `s_j^{in} = Σ_i w_ij`, `W = Σ_{ij} w_ij`.

\[
Q_{\mathrm{dir}}
=
\frac{1}{W}
\sum_{i,j}
\left(
w_{ij}
-
\frac{s_i^{\mathrm{out}}\, s_j^{\mathrm{in}}}{W}
\right)
\delta(c_i,c_j)
\quad (W>0).
\]

- **Attribution:** Algebraically equivalent to Leicht & Newman (2008), Phys. Rev. Lett. 100, 118703, Eq. (3), after translating their convention (`A_ij` = edge **j→i**) into source→target indexing. **Label:** `PUBLISHED_EQUIVALENT_FORMULA`.  
- **Weighted substitution `A→w`:** `PROJECT_SPECIFIC_EXTENSION` (LN stated for unweighted adjacency; Newman weighted-network modularity interpretation supports strength-based null models).  
- **Primary weight:** `count_raw`. **Required sensitivity:** `weight_log1p`.  
- **Larger better.** If `W=0`: metric **ineligible** for that (snapshot, relation).  
- **Relation-macro:** mean of eligible per-relation `Q_dir` over relations with `W>0` (`PROJECT_SPECIFIC_AGGREGATION`). Also report micro: pool all relations’ edges into one multiplex digraph (optional complementary).  
- **Snapshot aggregation:** mean over eligible snapshots (`PROJECT_SPECIFIC_AGGREGATION`).

### 3.2 Conductance (symmetrized weighted)

Build undirected weights `\tilde{w}_{ij} = w_{ij}+w_{ji}` (`PROJECT_SPECIFIC` operational symmetrization for standard undirected conductance).  
For community set S (hard labels), cut and volumes as in Leskovec, Lang, Mahoney (often cited conductance literature; classical definition):

\[
\phi(S)
=
\frac{\mathrm{cut}(S)}
{\min(\mathrm{vol}(S),\mathrm{vol}(V\setminus S))},
\quad
\mathrm{cut}(S)=\sum_{i\in S,j\notin S}\tilde{w}_{ij},
\quad
\mathrm{vol}(S)=\sum_{i\in S}\sum_j \tilde{w}_{ij}.
\]

- **Label:** base conductance `PUBLISHED_EXACT` / standard; symmetrization `PROJECT_SPECIFIC_EXTENSION`.  
- **Smaller better** for community quality.  
- Singleton / empty: undefined → exclude from community-level stats.  
- **Partition report:** volume-weighted mean of `φ(S_k)` over communities with defined φ; also median and 90th percentile (`PROJECT_SPECIFIC_AGGREGATION`).  
- Relation-macro: compute per relation then macro-mean over relations with positive undirected volume.

### 3.3 Entropy, MI, AMI, VI, NVI

For hard partitions U,V on the same eligible node set of size n:

\[
H(U)=-\sum_a p_a\log p_a,\quad
I(U;V)=\sum_{a,b}p_{ab}\log\frac{p_{ab}}{p_a p_b},\quad
H(U,V)=H(U)+H(V)-I(U;V).
\]

**AMI (arithmetic-mean normalization / AMIsum)** — Vinh, Epps, Bailey (2010), JMLR:

\[
\mathrm{AMI}_{\mathrm{sum}}(U,V)
=
\frac{I(U;V)-\mathbb{E}[I(U;V)]}
{\tfrac{1}{2}[H(U)+H(V)]-\mathbb{E}[I(U;V)]}.
\]

`E[I]` under the hypergeometric / permutation model as in Vinh et al. **Label:** `PUBLISHED_EXACT_FORMULA`.  
**Larger better.** If denominator ≤ ε: if U=V then AMI=1 else AMI=0.

**NVI** (normalized variation of information; related to Meilă VI and Vinh et al. normalized distances):

\[
\mathrm{NVI}(U,V)=1-\frac{I(U;V)}{H(U,V)}
\quad\text{if }H(U,V)>0;
\quad
\mathrm{NVI}=0\text{ if }H(U,V)=0.
\]

**Label:** `PUBLISHED_EQUIVALENT` / standard information-distance normalization (joint-entropy form). **Smaller better.**  
**Primary temporal similarity:** AMI. **Complementary distance:** NVI. High consecutive AMI ≠ absolute community quality.

**No Hungarian matching** for AMI/NVI/ARI.

### 3.4 Cosine silhouette

Rousseeuw (1987) silhouette with pairwise **cosine dissimilarity** `d_{ij}=1-\cos(x_i,x_j)` on embeddings from a **shared frozen evaluation encoder** E_eval:

- Fixed before final test evaluation; **no fine-tuning**; identical for every method.  
- Prefer an encoder **different from** the TDMEC training text encoder when a second official Qwen3 embedding dimension/checkpoint is available post–Q-EMB pilot; until then, use the frozen training encoder **read-only** and document the limitation.  
- **Never** use TDMEC `s`/`z` as the common evaluation space for cross-method silhouette.  
- Singleton community: silhouette 0.  
- If all-pairs infeasible: stratified sample up to 2000 eligible nodes per snapshot (fixed seed), compute exact silhouette on the sample; record sampling seed (`PROJECT_SPECIFIC` scalability rule).  
- Aggregate: mean over eligible nodes, then mean over snapshots.

**Label:** silhouette `PUBLISHED_EXACT` structure; cosine + shared encoder protocol `PROJECT_SPECIFIC_EXTENSION`.

### 3.5 Future-link Average Precision

**Protocol label:** `PROJECT_SPECIFIC_EXTENSION` (SBM-style / block affinity scoring as an **evaluation decoder**, not a claim that one paper owns the full protocol).

- Horizon: **t → t+1** only.  
- Fit decoder using **only** time-t community hard labels and time-t graph; targets = edges at t+1.  
- Relation-specific directed scores; no self-loops; no future leakage.  
- Shared positive sets and shared sampled negatives (same seeds) across all methods.  
- Primary: **relation-macro AP** = mean AP over **eligible** (snapshot t, relation r) cells.  
- Also: micro-AP (pool all eligible pairs), per-relation AP, all-future-links AP, **new-links-only AP** (positives absent at t).  
- ROC-AUC: complementary diagnostic only.  
- Cell ineligible if zero positive future edges → **exclude from macro denominator**; report eligible vs excluded cell counts. Macro denominator = #eligible cells, **not** automatically R(T−1).

**Scoring (primary evaluation decoder):** for communities c(u),c(v) at t, use empirical block rates on relation r:

\[
\hat{p}_{uv}^{(r)}
=
\frac{N_{c(u),c(v)}^{(t,r)}+\alpha}
{D_{c(u),c(v)}^{(t,r)}+\alpha\cdot |\mathcal{C}|_{\mathrm{norm}}}
\]

with small Laplace α=1 (`PRIMARY_EXPERIMENTAL_DEFAULT`); details of D = possible directed pairs between blocks excluding self-loops. Identical decoder for every method.

AP: standard IR Average Precision over ranked candidates (`PUBLISHED` IR definition).

### 3.6 Seed and perturbation stability

- **Seed stability:** mean pairwise AMI across the 5 seeds’ hard partitions on each snapshot, then mean over snapshots (`PROJECT_SPECIFIC_AGGREGATION` of published AMI). Do **not** claim Vinh et al. defined this multi-seed aggregation.  
- **Perturbation stability:** independently rewire / drop 5% of edges (fixed seeds), recompute partitions (or re-infer assignments with frozen model when applicable), mean AMI vs unperturbed (`PROJECT_SPECIFIC`).

### 3.7 ARI

Hubert & Arabie adjusted Rand index — `PUBLISHED_EXACT`.  
**Not** a primary Dataset A/B accuracy metric (no ground truth). Use only on synthetic / annotated subsets.

### 3.8 Runtime and memory

Wall time for train and for inference-eval; peak process RSS; peak `torch.cuda.max_memory_allocated` when GPU used. Report hardware tag.

---

## 4. Weight policy for structural metrics

- Primary: `count_raw`.  
- Required sensitivity: `weight_log1p`.  
- Model training continues to use `weight_log1p` as the TDMEC edge feature (Q-WGT); evaluation modularity primary raw is intentional for interpretability.

---

## 5. Status

QEVAL-01 closed. Evaluator implementation begins only in Phase 8 of the implementation sequence (after TDMEC-Full trains).
