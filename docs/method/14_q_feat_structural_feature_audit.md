# 14 — Q-FEAT Structural Node-Feature Audit

> **RESOLVED — USER_CONFIRMED_CANONICAL (2026-07-26).** The user confirmed **F_struct = 17**, differing from this audit's provisional 19: (1) `active` is **not** a feature — it becomes a separate boolean `struct_active_mask[t,node_idx]`; (2) `n_active_relations` is **excluded** (deterministically derivable from the degree features). The 16 relation-specific degree/strength features + `tweet_count_log1p` are confirmed as recommended. Strength is confirmed as `log1p(Σ_j count_raw)` (log-transformed event total; **not** `Σ log1p`, **not** sum of edge weights). Degree keeps **no** log1p. Normalization is deferred to Q-HPARAM. Canonical schema, mask definition, artifact/validation contracts: see `docs/method/12` (Q-FEAT entry) and `docs/method/03` (S06/S08). The analysis below is retained as the evidence base; where it says "19"/"active as feature 18"/"n_active_relations primary", the **canonical** decision (17, mask, excluded) governs.

**Decision:** Q-FEAT (structural node features `X_struct[t, node_idx, feature_idx]`).
**Status:** `USER_CONFIRMED_CANONICAL` (F_struct = 17). This document is the supporting audit.
**Scope:** analysis only, from the verified Dataset A schema, forensic report (`docs/data/02`), verified contract (`docs/data/03`), and method docs. No code, no data processing, no full-dataset run.
**Confirmed context:** N = 16,736 frozen nodes (D2); R = 4 relations; directed; self-loops excluded (Q-WGT); `count_raw` = distinct events after Q-DEDUP; `weight_log1p = log(1+count_raw)` (Q-WGT); quarterly snapshots (Q-CAL).

All features must be **in-snapshot** (leakage-safe): computed only from events with `created_at` inside snapshot `t`.

---

## 1. Complete Dataset A column → feature assessment

Verified 31 columns (`docs/data/02`). Null profile: **100% null** = `ocr_text, quoted_count, bookmarks, views, engagement, sentiment, topic, copy_count`; `media` ~81% null; `text_emojis` ~99% null; `text` ~fully populated (27 empty total). "Cumulative" = counter read at collection time (~2026) → temporally leaky for historical snapshots (same class of problem that removed engagement from Q-WGT).

| # | Column | Type | Availability | Meaning | Native unit | → node-snapshot feature? | Aggregation | Leakage-safe | Static/Temporal | Already in topology? | Category | Suitability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `timestamp` | num | populated | ingestion/record ts | event | Indirect | — | n/a | — | no | timestamp/provenance | diagnostics/provenance |
| 2 | `is_removed` | bool | populated | tweet later removed | event | No (leaky/label) | — | **No** (post-hoc) | post-hoc | no | provenance | exclude |
| 3 | `id` | float | populated | tweet id (lossy) | identifier | **No** | — | n/a | — | no | identifier | exclude (never a feature; float-lossy) |
| 4 | `created_at` | epoch | populated | authored time | event | **Yes (derived)** | snapshot binning + within-snapshot timing | Yes | temporal | no | timestamp | **snapshot assignment + activity/recency features** |
| 5 | `is_quote_status` | bool | populated | tweet is a quote | event | Maybe (share) | mean over authored tweets | Yes | temporal | partial (quote relation) | authored-activity | ablation (redundant w/ quote out-degree) |
| 6 | `user` | blob | populated | author identity (`user.id`) + account meta | account | identity only | parse `user.id` | id: yes; meta: see §5 | account meta static/cumulative | node identity | identifier/account meta | **node identity**; account meta rejected (leaky/static) |
| 7 | `text` | str | populated | authored text | text | No (structural) | — | — | — | no | textual | **Q-TEXT/Q-EMB**, not Q-FEAT |
| 8 | `ocr_text` | str | **100% null** | image OCR | text | No | — | — | — | no | textual | exclude (empty) |
| 9 | `text_lang` | str | populated | detected text language | text | No (structural) | — | — | static-ish | no | textual | Q-TEXT context; not Q-FEAT |
| 10 | `lang` | str | populated | tweet language | text | No (structural) | — | — | — | no | textual | Q-TEXT context; not Q-FEAT |
| 11 | `text_tags` | list | partial | tags in text | text | No | — | — | — | no | textual | Q-TEXT |
| 12 | `text_hashtags` | list | partial | hashtags | text | Maybe (count) | count/tweet | Yes | temporal | no | textual/authored | ablation/diagnostics (semantic → Q-TEXT) |
| 13 | `text_emojis` | list | ~99% null | emojis | text | No | — | — | — | no | textual | exclude (near-empty) |
| 14 | `user_mentions` | list{id} | populated | mentioned accounts | directed relation event | **Yes** | expand→edges; degree/strength | Yes | temporal | **yes (mention edges)** | structural graph | **primary (mention degree/strength)** |
| 15 | `media` | blob | ~81% null | attached media | event | Maybe (has_media) | mean/tweet | Yes | temporal | no | authored-activity | ablation only (sparse) |
| 16 | `likes` | int | populated | likes **received** | tweet | No | — | **No (cumulative)** | cumulative | no | engagement | reject (leaky) |
| 17 | `retweets` | int | populated | retweets **received** (count) | tweet | No | — | **No (cumulative)** | cumulative | no | engagement | reject (leaky); ≠ retweet relation |
| 18 | `reply_count` | int | populated | replies **received** (count) | tweet | No | — | **No (cumulative)** | cumulative | no | engagement | reject (leaky) |
| 19 | `quoted_count` | int | **100% null** | quotes received | tweet | No | — | — | — | no | engagement | exclude (empty) |
| 20 | `reply_status` | blob | populated (when reply) | reply target (`.user.id`) | directed relation event | **Yes** | →edge; degree/strength | Yes | temporal | **yes (reply edges)** | structural graph | **primary (reply degree/strength)** |
| 21 | `quoted_status` | blob | populated (when quote) | quote target (`.user.id`) | directed relation event | **Yes** | →edge; degree/strength | Yes | temporal | **yes (quote edges)** | structural graph | **primary (quote degree/strength)** |
| 22 | `retweeted_status` | blob | populated (when RT) | retweet target (`.user.id`) | directed relation event | **Yes** | →edge; degree/strength | Yes | temporal | **yes (retweet edges)** | structural graph | **primary (retweet degree/strength)** |
| 23 | `location_tags` | list | partial | location tags | text/meta | No | — | — | — | no | account/text meta | diagnostics only |
| 24 | `bookmarks` | int | **100% null** | bookmarks | tweet | No | — | — | — | no | engagement | exclude (empty) |
| 25 | `views` | int | **100% null** | views | tweet | No | — | — | — | no | engagement | exclude (empty) |
| 26 | `place` | blob | partial | geo place | account/meta | No | — | — | mostly static | no | account meta | diagnostics only |
| 27 | `impression` | int | populated | impressions | tweet | No | — | **No (cumulative)** | cumulative | no | engagement | reject (leaky) |
| 28 | `engagement` | num | **100% null** | engagement score | tweet | No | — | — | — | no | engagement | exclude (empty) |
| 29 | `sentiment` | num/str | **100% null** | sentiment | text | No | — | — | — | no | textual/semantic | exclude (empty; semantic→Q-TEXT) |
| 30 | `topic` | str | **100% null** | topic label | text | No | — | — | — | no | textual/semantic | exclude (empty; semantic→Q-TEXT) |
| 31 | `copy_count` | int | **100% null** | copy count | tweet | No | — | — | — | no | engagement | exclude (empty) |

### Category separation (explicit)
- **Structural graph:** `user_mentions`, `retweeted_status`, `reply_status`, `quoted_status` (+ direction via author `user.id`). → the feature source.
- **Authored-activity:** `created_at` (counts/timing), `is_quote_status`, `text_hashtags` count, `media` presence.
- **Engagement (all rejected):** `likes, retweets, reply_count, impression` (populated but **cumulative/leaky**); `quoted_count, bookmarks, views, engagement, copy_count` (100% null).
- **Account metadata:** `user` blob fields beyond id, `place`, `location_tags` (static/leaky/sparse) → rejected/diagnostics.
- **Textual:** `text, ocr_text, text_lang, lang, text_tags, text_hashtags, text_emojis, sentiment, topic` → Q-TEXT/Q-EMB, not Q-FEAT.
- **Identifiers/provenance:** `id`, `timestamp`, `is_removed` → never numeric model features.
- **Relation target fields:** the four `*_status`/`user_mentions` blobs (structural).
- **Timestamps:** `created_at` (used), `timestamp` (provenance).
- **Duplicate/reconciliation:** handled by Q-DEDUP; **not** behavioral features.

---

## 2. Audit of the currently proposed set

Per relation `r ∈ {mention, retweet, reply, quote}`: `in_degree_r, out_degree_r, in_strength_r, out_strength_r`; node-level `total_in_events_log1p, total_out_events_log1p, tweet_count_log1p, active`.

| Feature | Definition | Source | Grouping key | Counts | Dup rule | Multi-target mention | Self-loops | Inactive node | Redundant w/ feature? | Redundant w/ SAGE? | Hub bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `out_degree_r` | # distinct targets j with an `i→j` edge in (t,r) | edges | (t,r,i) | distinct partners | post-Q-DEDUP | each valid target counts once | excluded | 0 | no | partially (SAGE sees neighbors) but node-local scalar useful | moderate |
| `in_degree_r` | # distinct sources j with `j→i` in (t,r) | edges | (t,r,i) | distinct partners | post-Q-DEDUP | — | excluded | 0 | no | partially | **high** (popular targets) |
| `out_strength_r` | see §3 (ambiguous) | edges | (t,r,i) | events | post-Q-DEDUP | expanded | excluded | 0 | vs degree: distinct | partially | moderate/high |
| `in_strength_r` | see §3 | edges | (t,r,i) | events | post-Q-DEDUP | — | excluded | 0 | vs degree | partially | **high** |
| `total_out_events_log1p` | `log1p(Σ_r Σ_j count_raw(i→j,t,r))` | edges | (t,i) | events | post-Q-DEDUP | expanded | excluded | 0 | **yes — nonlinear sum of relation strengths (§3)** | partially | moderate |
| `total_in_events_log1p` | `log1p(Σ_r Σ_j count_raw(j→i,t,r))` | edges | (t,i) | events | post-Q-DEDUP | — | excluded | 0 | **yes (as above)** | partially | high |
| `tweet_count_log1p` | `log1p(#distinct authored tweets by i in t)` | rows | (t,i) | distinct tweets | post-Q-DEDUP | n/a | n/a | 0 | **no (see §6)** | no (not an edge quantity) | low |
| `active` | `1` if any authored tweet or any incident edge in t | rows/edges | (t,i) | indicator | — | — | — | **0** | no | no | none |

**Findings:** degree (breadth) and strength (intensity) are non-redundant with each other; the two **node-level totals are largely redundant** with the relation-specific strengths (§3); `tweet_count` is **not** redundant with out-events (§6). Directed features attached to nodes are only *partially* redundant with GraphSAGE — SAGE learns neighborhood aggregates, but a compact node-local activity/role scalar still helps seed `h^(0)` and the graph-only baseline. In-degree/in-strength carry the most hub bias (mitigated by `log1p` + robust scaling).

---

## 3. Resolving "strength"

For outgoing (incoming symmetric with `j→i`):
- **A. Raw event strength** `s_raw,out(i,t,r) = Σ_j count_raw(i→j,t,r)` → call this **raw strength** (total outgoing events in relation r). Heavy-tailed.
- **B. Weighted (edge-log) strength** `s_weighted,out(i,t,r) = Σ_j log1p(count_raw(i→j,t,r))` → call this **weighted strength** (sum of per-edge model weights). Conflates partner count with per-edge intensity; strongly correlated with `out_degree_r`.
- **C. Log total event strength** `s_logtotal,out(i,t,r) = log1p(Σ_j count_raw(i→j,t,r)) = log1p(s_raw,out)` → call this the **log-transformed event total**.

Incoming: `s_raw,in(i,t,r)=Σ_j count_raw(j→i,t,r)`, etc.

**These differ.** B ≈ f(degree, intensity) and double-counts breadth already captured by `out_degree_r`; A is unbounded/heavy-tailed. **Recommendation: use C (log-transformed event total) as the single "strength" feature per (node, relation, direction).** It measures interaction *intensity/volume* on a compressed comparable scale (consistent with Q-WGT's `log1p` philosophy), while `degree` separately measures *breadth*. Do **not** also include B (near-duplicate of degree×intensity). So `out_strength_r := s_logtotal,out`, `in_strength_r := s_logtotal,in`.

**Redundancy of node-level totals:** `total_out_events_log1p = log1p(Σ_r s_raw,out(r))`. This is the log of the sum of the per-relation raw strengths — a monotone nonlinear aggregate of information already present in the four relation-specific strengths. It adds little independent signal. **Recommendation: drop `total_in/out_events_log1p`** when relation-specific strengths are included. `tweet_count_log1p` is retained (distinct, §6).

---

## 4. Other A-derived candidate features (leakage-safe)

**Activity:** distinct authored tweets (=tweet_count); # original tweets (no relation), # authored retweets/replies/quotes, # tweets containing mentions; events generated (out) / received (in); active days in snapshot; **fraction of snapshot days active**; recency = (snapshot_end − last_activity) within snapshot; first/last activity position; burstiness (only if populated & defensible).
**Directed connectivity:** unique in/out neighbors (total + per relation); in/out strength (total + per relation); **# relations active in**; in/out activity balance `(out−in)/(out+in)`; source-vs-target role balance; reciprocity count (mutual i↔j); reciprocity rate.
**Multiplex-role:** outgoing/incoming relation **share** proportions (per relation); relation **entropy**; dominant in/out relation.
**Local structural:** clustering coeff (directed); common-neighbor stats; PageRank (+ per-relation); HITS hub/authority; k-core/coreness; ego-density.

Assessment of global graph statistics (PageRank/HITS/k-core/clustering/ego-density): **high cost** on `T≈35 × N=16,736` snapshots; **unstable in sparse/empty snapshots**; **hub-sensitive**; **redundant with GraphSAGE** (the encoder learns these); they **inject heavy handcrafted topology** that would weaken the scientific claim that TDMEC *learns* structure. → **diagnostics/ablation only, never primary.**

**Dominant-relation as categorical:** not appropriate for a numeric tensor (arbitrary ordinal). Prefer **relation-share proportions** and/or **relation entropy** (continuous, order-free).

---

## 5. Should raw columns become features?

- `likes, retweets, reply_count, impression` (populated): **reject** — cumulative-at-collection → temporal leakage; also received-engagement, not structure.
- `quoted_count, bookmarks, views, engagement, copy_count`: **exclude** (100% null).
- `text_hashtags`, `media`, `is_quote_status`: leakage-safe counts possible but low-value / semantic → **ablation/diagnostics** (hashtags & sentiment/topic semantics belong to Q-TEXT).
- `text, *_lang, sentiment, topic`: **Q-TEXT/Q-EMB**, not Q-FEAT.
- `place, location_tags`: sparse/static account-meta → **diagnostics only**.
- `id`, `user.id`, `timestamp`: identifiers/provenance → **never numeric features**.
- Account metadata in `user` blob (followers/following if present): **reject** — cumulative-at-scrape (leaky) and effectively static per snapshot (no in-snapshot value); same reasoning as Q-WGT's E.
- Duplicate/reconciliation flags: **not** behavioral features (reconciliation only).

---

## 6. `tweet_count` vs outgoing events

- `authored_tweet_count(i,t)` = number of **distinct authored tweets** by i in snapshot t (post-Q-DEDUP rows where author = i).
- `out_events(i,t)` = Σ over relations of **directed interaction events** generated by i's tweets in t.

One tweet may generate **0** events (an original tweet with no mention/RT/reply/quote), **1**, or **several** (multi-target mentions expand to one event per target; a quote-with-mention spans relations). Therefore `authored_tweet_count ≠ out_events` in general; a prolific original poster has high tweet_count but low out_events, while a heavy mentioner has out_events ≫ tweet_count. **They encode different signals and both are kept** (tweet_count = authored volume incl. non-interacting originals; out-strength = interaction generation).

---

## 7. Sufficiency for the model

Given: relation-specific GraphSAGE aggregates neighborhoods and multiplex structure; `weight_log1p` already encodes edge intensity; node-text (Q-TEXT/Q-EMB) supplies semantics; the GRU models temporal evolution — the node's **own** feature vector only needs to encode its **in-snapshot activity level, directed role, and multiplex participation**. A compact set does this. Adding many handcrafted centrality statistics would **duplicate what the encoder learns**, **amplify hub bias**, **make ablations harder to interpret**, **raise overfitting/compute cost**, and **weaken the scientific claim** (learned vs engineered structure). **Conclusion: minimize F_struct; the compact activity/role set is sufficient.**

---

## 8. Primary / Ablation / Diagnostic / Rejected

**A. Recommended primary (`X_struct`):**
- Per relation r (4): `out_degree_r`, `in_degree_r`, `out_strength_r` (=C), `in_strength_r` (=C) → 16
- `tweet_count_log1p` (authored volume incl. originals — unique signal)
- `n_active_relations` (compact multiplex-participation breadth)
- `active` (activity indicator; disambiguates real zeros)
→ **19 features.** Each adds unique info: degree=breadth, strength=intensity, per-relation=multiplex, tweet_count=authored volume, n_active_relations=participation breadth, active=presence.

**B. Optional ablation features:** in/out activity balance; reciprocity count & rate; outgoing/incoming relation-share proportions; relation entropy; fraction-of-snapshot-days-active; recency; original-vs-interaction tweet split; `has_media`; hashtag count. (Test individually; not in primary.)

**C. Diagnostics-only:** PageRank / per-relation PageRank; HITS hub/authority; k-core/coreness; clustering coefficient; ego-density; common-neighbor stats; burstiness; `place`/`location_tags` distributions.

**D. Rejected:** `likes, retweets, reply_count, impression` (cumulative→leaky); `quoted_count, bookmarks, views, engagement, copy_count, ocr_text, text_emojis(≈)` (null/near-null); `id`/`user.id`/`timestamp` (identifiers); follower/account metadata (leaky+static); `is_removed` (post-hoc); duplicate flags (reconciliation); text/semantic fields (Q-TEXT); dominant-relation categorical (wrong encoding — use shares/entropy).

---

## 9. Exact tensor contract (recommended primary)

`X_struct[t, node_idx, feature_idx]`, `t=1..T` (quarterly), `node_idx=0..16735` (all frozen nodes present every snapshot), dtype **float32**. Relation order (Q-REL, pending): here mention, retweet, reply, quote (placeholder — to be fixed by Q-REL).

| idx | name | relation | dir | formula | transform | normalization | dtype | inactive value |
|---|---|---|---|---|---|---|---|---|
| 0 | out_degree_mention | mention | out | #distinct targets | none | robust scale (train-fit) | f32 | 0 |
| 1 | in_degree_mention | mention | in | #distinct sources | none | robust scale | f32 | 0 |
| 2 | out_strength_mention | mention | out | `log1p(Σ_j count_raw)` | log1p | robust scale | f32 | 0 |
| 3 | in_strength_mention | mention | in | `log1p(Σ_j count_raw)` | log1p | robust scale | f32 | 0 |
| 4–7 | …retweet (out_deg,in_deg,out_str,in_str) | retweet | | as above | | | f32 | 0 |
| 8–11 | …reply | reply | | as above | | | f32 | 0 |
| 12–15 | …quote | quote | | as above | | | f32 | 0 |
| 16 | tweet_count_log1p | — | — | `log1p(#distinct authored tweets)` | log1p | robust scale | f32 | 0 |
| 17 | n_active_relations | — | — | #relations with any incident event | none | /4 or robust | f32 | 0 |
| 18 | active | — | — | `1` if any authored tweet or incident edge | none | none (binary) | f32 | **0** |

**F_struct = 19.**

**Zero-means-missing check:** for a frozen node with no in-snapshot activity, every value above is a **true zero** (genuinely no activity), not "missing" — because Dataset A defines the full population and absence = inactivity. `active=0` explicitly marks these. **No feature here has a "zero = missing" ambiguity**, so no extra availability mask is needed beyond `active` (the separate `active_mask`/`node_text_mask` already in the model contract cover activity/text presence). Any *rejected* leaky feature (e.g., followers) would have had a zero-vs-missing ambiguity — another reason to exclude them.

---

## 10. Normalization recommendation

| Option | Temporal comparability | Heavy-tail robustness | Leakage | Verdict |
|---|---|---|---|---|
| Raw | poor | poor | none | no |
| log1p only | fair | good | none | applied to counts, but scales differ across features |
| Per-snapshot z-score | **erases network-wide activity trends** | fair | none | no (destroys cross-snapshot signal) |
| **Train-snapshot-fit z-score** | **preserves cross-snapshot trends** | fair | needs temporal split | good |
| Relation-specific train-fit scaling | preserves + comparable relations | fair | needs split | good |
| **Robust scaling (median/IQR), train-fit** | preserves trends | **good** | needs split | **recommended** |
| Winsorization/clipping | — | tames hubs | none | recommended as add-on |

**Recommendation:** `log1p` on count/strength/tweet features, then **train-snapshot-fit robust scaling (median/IQR)** with **winsorization at a high quantile** (e.g. 99th) to control hubs, applied per feature (optionally per relation). This preserves genuine cross-snapshot activity differences (unlike per-snapshot z-score) while controlling heavy tails. **Deferred to Q-HPARAM:** the exact temporal train/val/test split defining "train snapshots", the winsorization quantile, and whether scaling is per-relation. Degree features could stay raw+robust-scaled; strengths are already `log1p`.

---

## 11. Required real-data diagnostics (before freezing schema)

Aggregate/sanitized only (no account IDs, tweet IDs, or text): zero rates per feature; nonzero-node coverage; distributions & quantiles; per-relation and per-snapshot distributions; pairwise correlations; exact linear dependencies (e.g., confirm totals ≈ sums of relation strengths); near-duplicate features (degree vs weighted-strength); hub concentration & Gini; variance across time; constant/near-constant features; relation sparsity; inactive-node frequency per snapshot; feature scale pre/post transform; degree-vs-strength comparison; tweet_count-vs-out_events comparison. These confirm whether to drop `n_active_relations` (if it is a near-deterministic function of the degree features) or any redundant feature before certification.

---

## Direct answers

1. **Is the currently proposed set sufficient?** Yes for the model's needs, **after two fixes**: disambiguate "strength" (§3) and drop the redundant node-level event totals. Extra centrality features are not needed (they duplicate GraphSAGE).
2. **Which proposed features are redundant?** `total_in_events_log1p` and `total_out_events_log1p` (nonlinear sums of the relation-specific strengths). The "weighted strength" variant (B) would be near-duplicate of degree×intensity if added.
3. **Are strength/total-count defined unambiguously today?** **No.** "strength_log1p" conflated three distinct quantities (A/B/C). This audit fixes strength := **C** = `log1p(Σ_j count_raw)` per (node, relation, direction).
4. **Which additional A columns are usable as node features?** Only `created_at` (activity timing/counts) and the four relation blobs (already the source). `is_quote_status`, `text_hashtags` count, `media` presence are marginal (ablation). All engagement/account/text/identifier columns are rejected or belong to Q-TEXT.
5. **Which additional derived features are worthwhile?** Ablation-tier: activity-balance, reciprocity, relation-share proportions, relation entropy, active-days fraction, recency. Not primary.
6. **Primary inputs:** relation-specific `out_degree, in_degree, out_strength(C), in_strength(C)` (16) + `tweet_count_log1p` + `n_active_relations` + `active`.
7. **Ablation-only:** balance, reciprocity, relation shares, entropy, active-days fraction, recency, has_media, hashtag count.
8. **Diagnostics-only:** PageRank/HITS/k-core/clustering/ego-density/common-neighbors/burstiness/place.
9. **Rejected:** cumulative engagement (`likes, retweets, reply_count, impression`), null columns, identifiers, follower/account metadata (leaky+static), `is_removed`, duplicate flags, text/semantic fields, categorical dominant-relation.
10. **Recommended F_struct:** **19** (provisional).
11. **Freeze now or after diagnostics?** Freeze the **design** (which families, strength=C, drop totals, normalization policy) now; keep the **exact field list / F_struct=19** as `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS` so correlation/linear-dependency results can prune redundant features (e.g. `n_active_relations`) before certification.
12. **Precise decision needed from you:** (a) approve strength := C and dropping the two node-level totals; (b) approve the 19-feature primary set (or adjust); (c) confirm the ablation/diagnostic/rejected partition; (d) approve normalization = log1p + train-fit robust scaling + winsorization (details to Q-HPARAM); (e) confirm freeze-after-diagnostics vs freeze-now; (f) confirm engagement, follower/account metadata, and identifiers stay rejected.

**Verdict:** `Q_FEAT_AUDIT_COMPLETE` — compact leakage-safe A-derived set recommended (F_struct = 19, provisional), strength disambiguated to the log-transformed event total, node-level totals dropped as redundant, global centrality relegated to diagnostics/ablation, all engagement/account/identifier/text columns excluded from Q-FEAT.
