# Phase 2 Evidence Review Template — QCAL-B01 and QDEDUP-B01

Use this template **after** real-data Phase 2 diagnostic reports exist.
Do not fill numeric “final” decisions from synthetic fixtures alone.

Run ID: ________________________________

Config hash: ________________________________

Reviewer: ________________________________

Date (UTC): ________________________________

---

## A. QCAL-B01 — Calendar bounds and T

### Evidence checklist (from `calendar_report.json`)

- [ ] Minimum / maximum valid timestamps reviewed
- [ ] Minimum / maximum valid quarters reviewed
- [ ] Provisional quarter range confirmed as diagnostic-only
- [ ] Internal empty quarters listed and preserved in candidate registry
- [ ] Leading empty quarter candidates reviewed
- [ ] Trailing empty quarter candidates reviewed
- [ ] Invalid / unparsable / missing / epoch-outlier reason counts reviewed
- [ ] Per-quarter Dataset A and B counts reviewed
- [ ] Cross-dataset quarter coverage reviewed
- [ ] Candidate start / end / T recommendation reviewed

### Candidate values (copy from report; do not invent)

| Field | Candidate from report | Reviewer decision |
|---|---|---|
| Certified start quarter | | ACCEPT / REJECT / MODIFY: ______ |
| Certified end quarter | | ACCEPT / REJECT / MODIFY: ______ |
| Certified T | | ACCEPT / REJECT / MODIFY: ______ |
| Leading empty policy | REVIEW_REQUIRED | DROP / KEEP / OTHER: ______ |
| Trailing empty policy | REVIEW_REQUIRED | DROP / KEEP / OTHER: ______ |

### Decision status

- [ ] Still `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`
- [ ] User-approved → may proceed to calendar certification workflow (not automatic)

Notes:

---

## B. QDEDUP-B01 — Exact signature and Layer-2 thresholds

### Evidence checklist (from `dedup_report.json`)

- [ ] Dataset B same-ID concordant / discordant groups reviewed
- [ ] Dataset A candidate composite concordant / discordant groups reviewed
- [ ] Cross-file and within-file duplicate effects reviewed
- [ ] Null / malformed key counts reviewed
- [ ] Multiplicity distributions reviewed
- [ ] Before/after candidate exact-collapse counts reviewed
- [ ] Provenance retention requirements acknowledged
- [ ] Confirmation that raw sources were not modified

### Candidate signature (copy from report)

Dataset A candidate composite fields:

Dataset B key:

### Reviewer decisions

| Item | Decision |
|---|---|
| Accept Dataset A composite signature as certified? | YES / NO / MODIFY |
| Accept Dataset B exact string tweet id key? | YES / NO / MODIFY |
| Discordant handling (retain+flag)? | CONFIRM / MODIFY |
| Layer-2 repeated-span thresholds | DEFER / SET: ______ |

### Decision status

- [ ] Still `REVIEW_REQUIRED_AFTER_REAL_DATA_DIAGNOSTICS`
- [ ] User-approved signature/thresholds recorded for later certification

Notes:

---

## C. Explicitly still unresolved after this review (unless separately decided)

- Numeric coverage certification thresholds
- QEMB-X01 through QEMB-X07
- Production `D_text`
- Hardware batch / AMP / OOM decisions

## D. Certification prohibition

Phase 2 reports must remain non-CERTIFIED. Certification is a later gated step
after this review and any required follow-up diagnostics.
