# 15 — Q-TEXT: Node-Text and Edge-Text Unit Contract

**Status:** `USER_CONFIRMED_CANONICAL` (2026-07-26).
**Scope:** defines the atomic text units, per-snapshot aggregation, temporal/leakage rules, and role separation for the two TDMEC text paths. Encoder/dimension/tokenizer decisions belong to **Q-EMB**; missing-text behavior belongs to **Q-MISS**.
**Confirmed context:** N=16,736 frozen nodes (D2); quarterly snapshots (Q-CAL); Q-DEDUP two-layer policy (record dedup + Layer-2 cleaned text); Q-WGT edges; Q-FEAT F_struct=17.

---

## 1. Canonical role separation
| | Dataset A text | Dataset B text |
|---|---|---|
| Role | **Edge / interaction** semantics | **Node-snapshot** semantics |
| Question answered | "What was expressed in the interaction i→j, relation r, snapshot t?" | "What did node i generally discuss/express in snapshot t?" |
| Artifact | `E_{i→j}^{(t,r)}` | `T_i^(t)` |
| Source unit | one distinct interaction-event tweet | one distinct authored tweet |
| Join to other dataset | **none** (tweet-level A↔B join prohibited) | account-level only (`user.id`) |

The two representations must **not** be collapsed into one artifact; A text must not replace B node text; B text must not be approximately attached to A edges.

---

## 2. Node text (Dataset B) — canonical
Atomic unit = **one distinct authored tweet**. For each retained tweet `q`:
```
z_node_q = TextEncoder(cleaned_text_q)
```
Let `Q_i^(t)` = the set of all retained, distinct Dataset B tweets authored by node `i` whose `created_at` falls **inside** snapshot `t`. Then:
```
T_i^(t) = mean_{ q ∈ Q_i^(t) } z_node_q
```
Tensor: `X_node_text[t, node_idx, :]` with shape `[T, N, D_text]`, `N = 16,736`, `D_text` set under Q-EMB. All tweets in `Q_i^(t)` contribute **equally** (mean pool). **Per-tweet embed → mean-pool**; never concatenate all snapshot tweets into one long string.

**Temporal rule (strict):** only tweets inside snapshot `t`. Never earlier/later snapshots, never the full multi-year history per snapshot, never cumulative-up-to-`t`, never future tweets.

---

## 3. Edge text (Dataset A) — canonical
Atomic unit = **one distinct interaction-event tweet** (the tweet/status that generated the event). For each retained canonical event `q`:
```
z_edge_q = TextEncoder(cleaned_text_q)
```
Edges aggregate by `(snapshot_id, relation_id, source_idx, target_idx)`. Let `Q_{i→j}^{(t,r)}` = the set of distinct Dataset A events from source `i` to target `j` within snapshot `t`, relation `r`. Then:
```
E_{i→j}^{(t,r)} = mean_{ q ∈ Q_{i→j}^{(t,r)} } z_edge_q
```
- One event → embedded once; several events on the same canonical edge → mean-pooled.
- Relations, directions, and snapshots remain **separate**.
- Single-event edge → its edge-text vector is that event's embedding.
- Event text stays associated with `snapshot_id, relation_id, source_idx, target_idx`, and provenance; only text from the event's own snapshot is used.

**Multi-target mention:** one authored tweet may create one event per valid target; the same cleaned event text may be associated with each target-specific event; provenance must record the single original authored tweet; caching must avoid duplicate embedding work.

---

## 4. Cleaned text and deduplication
- Embedding input = **`cleaned_text`** only (Q-DEDUP Layer-2). Original text remains private/read-only for audit/provenance; never the default embedding field.
- Never embed corrupted raw text, extraction-induced repeated blocks, tweet identifiers, or approximately-joined Dataset B text.
- A-event dedup and B-tweet dedup occur **before** semantic aggregation. Duplicates must not inflate embedding calls, node mean-pool, edge mean-pool, tweet counts, or event counts.
- Embedding cache keyed by **stable content + preprocessing provenance**, not by the unreliable float-converted Dataset A tweet id alone. Dataset B exact string tweet ids may be used for identity/caching/dedup, but never as model features.

---

## 5. Why per-tweet embedding + mean-pool
Avoids encoder token-limit overflow for prolific users; prevents uncontrolled truncation; enables deterministic tweet-level caching and per-tweet resume; keeps an explicit atomic contract and explicit snapshot assignment; allows valid reuse of a tweet embedding; avoids order dependence from concatenating thousands of tweets. The pooled vector = **average semantic activity** of the user within the snapshot. Mean pool is the canonical primary; recency-weighted / attention / robust-trimmed / max pooling are **future ablations only** and must not replace mean pooling without a separate explicit user decision.

---

## 6. Downstream use
- **`T_i^(t)`** → node-text TDMEC variant, initial node representation, semantic-alignment objective (`L_sem`), temporal node-state modeling.
- **`E_{i→j}^{(t,r)}`** → conditions the **learned edge gate** / edge-aware message passing. The gate may receive: relation identity, canonical edge weight, source representation, target representation, edge-text representation, edge-text availability mask. The model must **not** hard-code semantic rules (e.g. "hostile text ⇒ different communities"); it must **learn** whether/how content modifies the structural message. Exact encoder/dimensions → Q-EMB + model contract.

---

## 7. Missing-text (Q-MISS — resolved)
**Canonical:** `docs/method/17_q_miss_missing_text_contract.md` (`USER_CONFIRMED_CANONICAL`). Exact all-zero vector + boolean availability mask for node and edge paths; no learned missing embedding; no drop for missing text; no text carry-forward; Fusion MLP / Edge Gate receive masks explicitly; `L_sem` on mask=True only; valid-text counts as metadata. GRU/`model_active_mask` remain separate.

---

## 8. Implementation sequencing
Both paths are **required**. Order: (1) implement and certify the **Dataset B node-text** path; (2) implement and certify the **Dataset A edge-text** path. Edge text is not optional and must not be indefinitely deferred; staging is for engineering manageability only and does not weaken the claim that interaction content can refine/correct structural evidence.

---

## 9. Canonical status
| Item | Status |
|---|---|
| Dataset B node-text source | `USER_CONFIRMED_CANONICAL` |
| Dataset B atomic unit | one distinct authored tweet |
| Dataset B aggregation | per-tweet embedding → mean-pool per node-snapshot |
| Dataset A edge-text source | `USER_CONFIRMED_CANONICAL` |
| Dataset A atomic unit | one distinct interaction-event tweet |
| Dataset A aggregation | per-event embedding → mean-pool per (snapshot, relation, source, target) |
| Embedding input | `cleaned_text` only |
| Temporal rule | strict in-snapshot text only |
| Approximate A↔B tweet join | prohibited |
| Node-text implementation order | first |
| Edge-text implementation order | second, but required |
| `D_text`, encoder, tokenizer, max length, truncation, dtype | deferred to Q-EMB |
| Missing-text behavior | `USER_CONFIRMED_CANONICAL` (Q-MISS M1 — `docs/method/17`) |
