# 05 — Architecture and Tensor Contract

**Primary:** P-001 §§15–18. **Supporting:** P-002 §§8–12 (*Source-stated*).  
**N = 16,736 (D2).** Symbolic dims for all unfrozen values.  
TDMEC implements only the modules defined in this contract.  
**Batch 4 architecture decisions:** `USER_CONFIRMED` 2026-07-28 (`docs/method/12` QFUS/QENC/QPROJ/QHP/QVAR).

## 1. End-to-end computation graph

```
X^(t), T^(t), masks  →  MLP_x  →  h^(0,t)
G^(t,r), E^(t,r)     →  EdgeContext MLP_e → g
h, g                 →  EdgeGate MLP_g → γ ∈ (0,1)
γ, W_r,in/out        →  Directed messages → M_in, M_out
h^(0), M_in, M_out   →  MLP_r → h^(t,r)          for each r
{h^(t,r)}, a^(t,r)   →  MaskedFusion → z^(t)
z^(t), s^(t-1)       →  GRU → s^(t)   (QACT/QGRU: update iff model_active; else carry s)
s^(t), {μ_k}         →  Student-t → Q^(t), hard, conf, entropy
```

## 2. Component contracts

### 2.1 Initial node input (`MLP_x`) — **QMLP-01**
| Item | Contract |
|---|---|
| Op | Concatenate `[X_struct_scaled_or_raw_per_QHP-02, X_node_text, node_text_available_mask]` → MLP → `d_h` |
| MLP_x | `Linear(in, d_h) → ReLU → Linear(d_h, d_h)` (**PRIMARY_EXPERIMENTAL_DEFAULT**) |
| Activation | ReLU |
| Dropout | **0** primary; **0.1** optional ablation (**QMLP-01**) |
| Norm | **No** LayerNorm/BatchNorm in primary; optional LN after concat as ablation |
| Residual | None in primary |
| Init | Kaiming/He uniform for ReLU layers; biases 0 |
| Missing text | Exact zero + mask False |
| Status | Spec + Batch-7 defaults |

### 2.2 Edge context (`MLP_e`) — **QGATE-01 / QMLP-01 / QPROJ-01**
| Item | Contract |
|---|---|
| Edge-text projection | `E' = Linear(D_text, d_h)` inside module (not `P_T_node`); if mask False, `E'=0` |
| Op | `g = MLP_e([e_r, weight_log1p, E', edge_text_available_mask])` |
| MLP_e | `Linear(d_rel+1+d_h+1, d_h) → ReLU → Linear(d_h, d_h)` |
| Dropout | 0 primary |
| Status | Spec + Batch-7 defaults |

### 2.3 Edge gate — **QGATE-01 / QMLP-01**
| Item | Contract |
|---|---|
| Op | `γ = σ(MLP_g([h_source, h_target, g]))` — **scalar** γ ∈ (0,1) |
| MLP_g | `Linear(3 d_h, d_h) → ReLU → Linear(d_h, 1)` |
| Dropout | 0 primary |
| Status | Spec + Batch-7 defaults |

### 2.4 Directed relation-specific GraphSAGE (**QENC-01, QENC-02, QMLP-01**)
| Item | Contract |
|---|---|
| In-msg | `m_{j→i} = γ · W_{r,in} h_j` |
| Agg | **Mean** separately for in/out; empty neighborhood → **zero vector** |
| Out | `h_i^(t,r) = MLP_r([h_i^(0), M_in, M_out])` |
| MLP_r | `Linear(3 d_h, d_h) → ReLU → Linear(d_h, d_h)`; **separate params per relation** |
| W_{r,in}, W_{r,out} | Separate linear maps per relation and direction |
| Depth | Primary `L=1`; sensitivity `L=2` |
| Training fanout | `[15]` per relation×direction |
| Dropout | 0 primary |
| Status | Spec + Batch-7 defaults |

### 2.5 Masked relation fusion (**QFUS-01**)
| Item | Contract |
|---|---|
| Availability | `a_i^(t,r)=1` if node i has incoming or outgoing edge in relation r |
| Score | `u = q^T tanh(W_f h^(t,r)+b_f)` |
| Masked softmax | β uses only available relations |
| Fuse | `z = Σ_r β_r h^(t,r)` |
| Fallback | If ∑_r a=0: **`z_i^(t) = h_i^(0,t)`** (**IMMUTABLE_METHOD_CONTRACT**) |
| `struct_active_mask` | **Not** redefined by relation availability; remains `(tweet_count>0) OR (any canonical in/out edge)` |
| Ablation | Mean over available relations |
| Status | Spec only |

### 2.6 GRU temporal encoder
| Item | Contract |
|---|---|
| Op | If `model_active`: `s^(t)=GRU(z^(t), s^(t-1))`; else `s^(t)=s^(t-1)` (**QGRU-01**); `s^(0)=0` |
| `model_active_mask` | `struct_active_mask OR node_text_available_mask` (**QACT-01**); masks remain separate artifacts |
| Inactive | Exact hidden-state carry; no GRU step; does **not** carry text embeddings |
| Params | Single GRU (only temporal mechanism); state dim = `d_h` |
| BPTT | Truncated length **3** primary (**QTR-03**); no detach inside window; detach at window boundaries only; carry state across windows; sensitivity lengths 2 and 4 |
| Status | Spec only |

### 2.7 Prototype Student-t head (**QHP-03, QHP-04**)
| Item | Contract |
|---|---|
| Kernel | `q_ik ∝ (1 + ‖s_i−μ_k‖²/α)^(-(α+1)/2)`, **α=1** |
| Normalize | Softmax over k |
| Outputs | Soft Q; hard `argmax`; confidence `max`; entropy |
| Init | **KMeans++** on pretrained temporal states `s`; training snapshots only; `model_active_mask=True` only; snapshot-balanced sampling; `n_init=20`; fixed recorded seed (**QHP-04**) |
| After init | Prototypes remain trainable; empty-cluster → deterministic high-distance reinit + report |
| K | Fixed per run; **primary `K=10`**; sensitivity `{5,10,15,20,30}`; never select K on test (**QHP-03**) |
| Status | Spec only |

### 2.8 Semantic projections (`L_sem`) (**QPROJ-01**)
| Item | Contract |
|---|---|
| `P_z` | Single `Linear(d_h, d_sem)` with `d_sem=d_h` |
| `P_T_node` | Single `Linear(D_text, d_sem)` |
| Edge text | Own linear inside `MLP_e` only |
| `d_rel` | Primary 16; optional 32 |

### 2.9 Structural decoder (`L_struct`) — **QDEC-01**
| Item | Contract |
|---|---|
| Score | `\ell_{ijr} = MLP_dec([s_i, s_j, e_r, E', m_e, \tilde{w}])` → scalar logit |
| States | Use **post-GRU** `s` (P-001) |
| Positive inputs | `E'=` edge-text projection or 0; `m_e=edge_text_available_mask`; `\tilde{w}=weight_log1p` |
| Negative inputs | Same snapshot & relation; `E'=0`, `m_e=False`, `\tilde{w}=0`; no observed edge |
| MLP_dec | `Linear(2 d_h + d_rel + d_h + 1 + 1, d_h) → ReLU → Linear(d_h, 1)` |
| Loss | BCE-with-logits on positives (label 1) and negatives (label 0); mean reduction over scored pairs in batch |
| Encoder | Masked positives **excluded** from encoder graph |
| Status | Spec + Batch-7 |

Primary experimental defaults (Batch 4+7): `d_h=64`, `K=10`, `alpha=1`, `L=1`, `fanout=[15]`, `agg=mean`, `d_rel=16`, ReLU MLPs depth-1-hidden, dropout 0.

## 3. Tensor shape table (symbolic)

| Tensor | Shape | Notes |
|---|---|---|
| `X^(t)` | `[N, F_struct]` or `[T,N,F_struct]` | **F_struct=17** (Q-FEAT); raw artifact; train-time scaling per QHP-02 |
| `T^(t)` | `[N, D_text]` | D_text unresolved (POST_PILOT) |
| `E` | `[num_edges, D_text]` sparse | Not dense `[T,R,N,N,D]` |
| `active_mask` / `struct_active_mask` | `[T, N]` | Q-FEAT / QFUS-01 |
| `node_text_available_mask` | `[T, N]` bool | Q-MISS |
| `node_valid_text_count` | `[T, N]` int | metadata, not feature |
| `edge_text_available_mask` | aligned to edge order | Q-MISS |
| `edge_valid_text_count` | aligned to edge order | metadata, not feature |
| `relation_mask` | `[T, N, R]` | R=4 |
| `e_r` | `[R, d_rel]` | primary `d_rel=16` |
| `h^(0,t)` | `[N, d_h]` | primary `d_h=64` |
| `h^(t,r)` | `[N, d_h]` × R | |
| `z^(t)` | `[N, d_h]` | |
| `s^(t)` | `[N, d_h]` | |
| `μ` | `[K, d_h]` | primary `K=10` |
| `Q^(t)` | `[N, K]` | |

**Confirmed numeric:** `N=16736`, `R=4`, `F_struct=17`, **primary `d_h=64`**, **primary `K=10`**, **primary `d_rel=16`**, **primary `L=1`**, **primary fanout `[15]`**.  
**Not confirmed:** `T` (calendar boundaries), `D_text` (POST_PILOT).

**Certification (QART-01-FRAME, USER_CONFIRMED 2026-07-28):** contract-correctness checks are mandatory hard gates; text/activity coverage always reported and normally warning until numeric thresholds approved; complete absence of a primary-method required artifact/semantic path is a hard failure (explicit graph-only ablation may be certified separately). No artifact may be labeled `CERTIFIED` without schema/shape/dtype/node-order/edge-order/relation-map/self-loop/alignment/exact-zero/`model_active`/NaN-Inf/checksum/manifest/provenance/deterministic ordering/resume/double-count checks. Numeric coverage thresholds remain open.

## 4. Mask and missing-value behavior

| Case | Behavior |
|---|---|
| Missing node text | `T=0`, mask=0 |
| Missing edge text | `E=0`, mask=0; gate uses structure |
| No edges in relation | `a=0`; β=0 for that relation |
| No edges any relation | `z = h^(0)` (**QFUS-01**); do **not** rewrite `struct_active_mask` |
| Inactive node | Per QACT/QGRU: if not `model_active`, exact `s` carry; text tensors unchanged |
| External target | Not a node; summary only |

## 5. Configuration fields (run config)

Primary experimental defaults (Batch 4+5+7): `d_h=64`, `K=10`, `alpha=1`, `L=1`, `fanout=[15]`, `agg=mean`, `d_rel=16`, ReLU MLPs, dropout 0, λ as Batch 5, phases 100/80/120. Remaining open: `D_text` (Q-EMB), exact calendar `T` (diagnostics).

## 6. Implementation dependencies

| Component | Depends on |
|---|---|
| Encoder | Certified edges, features, node/edge text, masks |
| Fusion | Relation masks; QFUS-01 fallback |
| GRU | Sequence of fused states + active/text masks |
| Head | `d_h`, K, pretrained active training states for init |
| Scaling | QHP-02 train-only robust scaler |
| All | Frozen N=16736 map |

## 7. Variant naming (**QVAR-01 — IMMUTABLE_REPORTING_CONTRACT**)

| Variant | Inputs | Notes |
|---|---|---|
| **TDMEC-G** | Graph only | Ablation |
| **TDMEC-NT** | Graph + node text (no edge text) | Ablation |
| **TDMEC** / **TDMEC-Full** | Graph + node text + edge text | **Primary method** |
| **TDMEC-ET** | Graph + edge text only | Reserved ablation **if implemented**; never name the full method |

Scientific definition remains P-001; reporting names are frozen for eval/handoff consistency.
