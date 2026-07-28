# 06 — Loss, Training, and Inference Contract

**Primary:** P-001 §§19–20. **Supporting:** P-002 (*Source-stated*).  
**Batches 5–7:** `USER_CONFIRMED` / authorized resolution 2026-07-28.

## 1. Loss composition (**IMMUTABLE_METHOD_CONTRACT**)

```
L_repr = λ_struct L_struct + λ_sem L_sem
L_comm = λ_cluster L_cluster + λ_reg L_reg
L_time = λ_temp L_temp
L_total = L_repr + L_comm + L_time
```

Targets: `λ_struct=λ_sem=λ_cluster=1.0`, `λ_reg=λ_temp=0.1`.  
Always log **unweighted** component magnitudes.

### 1.1 `L_struct` (**QLOSS-01 / QDEC-01**)
- Mask 15% of observed edges; 3 negatives per masked positive; uniform w/o replacement; same snapshot & relation; no self-loops; exclude all observed edges from negatives.
- Decoder logit: `MLP_dec([s_i, s_j, e_r, E', m_e, w̃])` with BCE-with-logits.
- Positives: true edge text projection / mask / `weight_log1p`.
- Negatives: `E'=0`, `m_e=False`, `w̃=0`.
- Reduction: mean over scored pairs.
- Masked positives excluded from encoder (**IMMUTABLE**).

### 1.2 `L_sem`
- `mean_{mask}(1 − cos(P_z z, P_T_node T))` on pre-GRU `z`; mask-True only; zero-eligible → 0.

### 1.3 `L_cluster` (**QCLU-01**)
- DEC: `p_ik ∝ q_ik²/f_k`; `KL(P‖Q)`.
- Recompute P **once per epoch** (primary). Optional: every N mini-batches.

### 1.4 `L_reg` (**QLOSS-02**)
- `1/(K(K−1)) Σ_{k≠ℓ} max(0, m − ‖μ_k−μ_ℓ‖²)`, `m=1.0`; no equal-size forcing.

### 1.5 `L_temp` (**QLOSS-03**)
- Mean JS for nodes with `model_active` at both endpoints; ε=1e-8 in logs.

### 1.6 Scope
Primary losses only as above. `L_relation_type` optional ablation.

### 1.7 Phases (**QLOSS-04 / QPHASE-01**)
1. Pretrain ≤100 epochs: cluster=reg=temp=0.
2. Joint **80** epochs: cluster=1, reg=0.1, temp=0.
3. Temporal **120** epochs: ramp λ_temp 0→0.1 over first **24** epochs, then hold.

## 2. Batching and epochs (**QBATCH-01**)

- Node mini-batches (default start `B_nodes=512`, **runtime-probed**) + fanout `[15]`.
- Each snapshot epoch: all N nodes covered once (shuffled partition).
- **Epoch** = one chronological pass over all training snapshots with full node coverage each.
- BPTT length 3; no grad windows crossing into val/test; carry state into val/test without grads.
- AdamW all params; lr 5e-4; wd 1e-4; clip 1.0; patience 20; min_delta 1e-4.
- Early-stop monitors and final selection: see `docs/method/18` (QEVAL-01).

## 3. Initialization
- `s^(0)=0`; KMeans++ on training active `s` (QHP-04).

## 4. Seeds (**QTR-04**)
- 5 / ≥3 / 1 rules unchanged; control masking, negatives, sampling, init.

## 5. Inference
- Chronological; no future leakage; outputs Q, hard, conf, entropy, trajectories.

## 6. Status
Trainer not implemented; blocked on certified artifacts + Q-EMB finals as needed + calendar bounds for exact split.
