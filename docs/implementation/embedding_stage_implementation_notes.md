# Embedding Stage Implementation Notes

**Status labels:** `IMPLEMENTED_NOT_EXECUTED` · `TRANSFER_PENDING_VALIDATION`

This note records what was completed for Embedding Subtasks 5–8 in the authoring
Studio without executing pipelines, tests, model downloads, or real-data runs.

## Subtask coverage

| Subtask | Scope | Code status | Execution in authoring Studio |
|---|---|---|---|
| 1–4 | Contract audit, readers, eligibility, mock encoder/writer | Previously implemented | Previously tested baseline preserved: **165 passed / 4 skipped / 9 warnings** (do not treat new tests as passed) |
| 5 | Node-snapshot + canonical-edge pooling, masks, counts, checksums | Implemented | `NOT_EXECUTED` |
| 6 | End-to-end mock orchestration + CLI | Implemented | `NOT_EXECUTED` |
| 7 | Qwen3 encoder + 64+64 preflight gates | Implemented | `NOT_EXECUTED` |
| 8 | 10k+10k bounded pilot gates + stratified sampling | Implemented | `NOT_EXECUTED` |

## Provisional Q-EMB decisions (not final)

These remain configurable and must not be frozen into the thesis tensor contract
until explicit post-pilot confirmation:

1. Checkpoint: `Qwen/Qwen3-Embedding-4B`
2. Model/tokenizer revision: unresolved in-repo; pin immutable SHAs via
   `QWEN3_MODEL_REVISION` / `QWEN3_TOKENIZER_REVISION` in the target Studio
3. `D_text` / `output_dimension`: provisional `2560` (native hypothesis)
4. `max_length`: provisional `512`
5. Stage-A pooling: last-token with left padding (Transformers example hypothesis)
6. Shared instruction: I1 pilot candidate wording
7. Stage-B final L2 normalization: provisional `none` (N1); N2 remains a pilot comparator
8. Reduced MRL dimensions: disabled unless
   `enable_provisional_mrl_truncation: true`

## Privacy

Aggregate reports, manifests, and CLI JSON output must contain hashes, counts,
shapes, and configuration digests only. Raw cleaned text and private identifiers
must not appear in logs or Git-visible reports.

## Force / overwrite policy

`--force` and `force: true` are rejected. Completed embedding outputs must be
resumed explicitly or written under a new `embedding_run_id`.
