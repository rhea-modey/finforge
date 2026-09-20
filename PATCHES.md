# Queued patches (apply after phase-2 workflow completes, before the full run)

## P1 — Conditional injection presence (generalization hardening)
Owner: main loop. Status: QUEUED.

Every world currently injects missing_payroll_accrual + missing_prepaid_amort +
bank fees unconditionally; a harness could learn "always book these" without
checking the GL and be rewarded everywhere (incl. held-out). Change:

- engine.py `plan_injections`: add `has_missing_payroll_accrual` (p=0.75) and
  `has_missing_prepaid_amort` (p=0.8).
- engine.py monthly loop: when payroll accrual NOT injected, June run 2 pays
  2026-06-30 and is booked+banked normally (register shows pay_date 06-30).
  When prepaid amort NOT injected, post June amortization as in Apr/May
  (second-prepaid item too, when present).
- tasks.py `_gold_journal_entries`: include only genuinely missing entries
  (derive from truth records only — already true for payroll/dep/misposted;
  make the prepaid-amort block conditional on the truth record).
- tasks.py `_gold_accrual`: unchanged rows; `journal_entry` = the June
  amortization entry regardless (it is the schedule's supporting entry whether
  or not it is already booked) — keep as-is.
- exporters.py checklist: wording is already verify-then-book for depreciation;
  make items 2 and 3 conditional-neutral too ("book ... if not already posted —
  check the GL").
- After edits: python3 -m generator.build && smoke_generator && oracle_check
  must pass; then RE-RUN enrichment (cache invalid — delete
  gold/*/*/enrichment.json first) and qa.py.

## P3 — Semantic coherence (fixes the 12/36 QA failures)
Owner: main loop. Status: QUEUED. Root causes from qa_report.json sweep:

- (a) Capitalization ambiguity: any bill expensed in-period must not be worded
  "annual/yearly license/subscription". engine.py: misposted bill becomes a
  clearly period-expense service (e.g. "trade-show booth services" true acct
  6200, posted 6700); names.MISPOST_DESCS rewritten accordingly. Ordinary 6450
  bills get monthly wording. Only the real prepaid-software item (1310) may use
  annual wording.
- (b) Vendor->account affinity: engine assigns bill/driver accounts ONLY from
  a vendor-affinity map (utilities vendors -> 6500, ad/media -> 6200, print/
  office -> 6700, IT/software -> 6450, legal/accounting -> 6600, freight ->
  5100/flavor, etc.). Driver vendors chosen FROM the affinity pool of the
  driver's account. names.py: add VENDOR_AFFINITY.
- (c) enrich.py constraint: rewritten bill descriptions must stay consistent
  with the posted account's nature (pass the account name into the enrichment
  prompt per bill; validator keyword-checks per affinity class).
- After: rebuild, gates, purge gold/*/*/enrichment.json + qa_report.json,
  re-enrich, re-QA. Target: <= 2/36 QA fails; iterate once more if needed.

## P2 — Writeup honesty note
Held-out shares the generative engine: the transfer claim is unseen
companies/surfaces/industries, not unseen accounting mechanics. State in report.
