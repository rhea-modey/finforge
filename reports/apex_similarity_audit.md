# FinForge vs. APEX-Accounting: Similarity & Contamination Audit

**Date:** 2026-09-19
**Auditor:** automated audit (Claude Code)
**APEX source:** `mercor/apex-accounting` public dev set (CC-BY 4.0), downloaded in full via `huggingface_hub` (120 files) to `/tmp/apex` — all claims below about APEX are verified against the downloaded data, not just the dataset card.
**FinForge source:** `/Users/videetmehta/hack26/finforge` (worlds, gold, generator, harness, DESIGN.md).

**Bottom line:** FinForge is structurally APEX-shaped (agentic file-hunt over a synthetic company's month-end close, four matching task categories, binary rubrics, concise controller-style prompts, step-budgeted runs) but shares **zero content** with APEX: 0 overlapping 8-gram shingles across the full corpora, 0 shared company/vendor/client/person entities, and 0 hits for every APEX-distinctive string. The "APEX-inspired but distinct" claim is legitimate. Two cosmetic name collisions ("A. Whitfield", "Marsh Rd") should be renamed out of caution; the biggest genuine divergences (plain-text-only files, ~140× smaller worlds, schema-pinned deliverables) mean FinForge difficulty is *not* APEX-comparable and should not be presented as such.

---

## 1. Inventory of the two benchmarks

### APEX-accounting public dev set (verified from download)

| Property | Value |
|---|---|
| Tasks | 10 (dev split; the 160-task scored benchmark is closed/held out) |
| Worlds | 1 — World 9: **Sterling, Marsh & Associates LLP**, Philadelphia boutique law firm, December 2024 close, US-GAAP accrual |
| Categories | Reconciliation ×4, Schedules & Accruals ×3, Variance Analysis ×2, Data Entry ×1 (subcategories: AR rec, payroll rec ×2, client cost rec, WIP rollforward, payroll+partner comp, flex, MoM revenue, AR aging exposure, contingency revenue) |
| Prompt length | mean **73.6** words, median 69, range 40–129 |
| Deliverable | **free-form console message** (`output_type: console_text` on all 10 tasks); "share your answer in the console" |
| Gold answers | expert-written narrative text, mean 134.3 words |
| Rubrics | **89 binary criteria** total; mean 8.9/task, median 5.5; `criterion_type` split 73 `Reasoning (numerical)` / 16 `Reasoning (qualitative)`; expert-authored |
| Grading | LM-as-judge (DeepSeek-v4-Flash, temp 0.1, unreleased GEPA-optimized template), one criterion at a time, final output only; headline metric **Mean Criteria@3** |
| World files | **90 shared** (3.82 MB): 53 PDF, 39 XLSX (per card: 34 spreadsheets / 46 PDFs / 10 accounting-system files), incl. QuickBooks exports under `world/apps_data/quickbooks/` + **16 task-specific** files (0.14 MB) under `task_files/` for 8 of 10 tasks |
| Domain flavor | Clio billable hours & WIP, IOLTA trust accounting, retainer deposits/earned-fee transfers, client cost advances, realization write-downs, contingency settlement (Whitfield), partner guaranteed payments, contract-attorney accruals |
| Run limits | **500 steps / 5M tokens** per task; harness-injected tool layer over the QBO data (not shipped) |
| Effort estimates | 0.75–4.0 expert hours/task (mean ≈2.4) |
| Authorship | practicing accountants/bookkeepers wrote tasks, gold, and rubrics; worlds difficulty-filtered against three frontier models |

### FinForge (verified from repo)

| Property | Value |
|---|---|
| Tasks | **144** = 36 worlds × 4 fixed categories (`bank_rec`, `journal_entries`, `variance_analysis`, `accrual_schedule`) |
| Worlds | 36 (30 train w01–w30 + 6 heldout h01–h06), one synthetic company each, June 2026 close, 3 simulated months |
| Industries | 6 — train: saas, ecommerce, wholesale, services; heldout adds **manufacturing, clinic** (industry-level holdout) |
| Prompt length | mean **143.9** words, median 145, range 136–150 (longer than APEX solely because the deliverable JSON schema is inlined) |
| Deliverable | **structured JSON** matching an inline schema, submitted via a `submit_answer` tool |
| Gold answers | structured JSON, exists **by construction** (generator injects discrepancies and records `TruthRecord`s) |
| Rubrics | **749 binary criteria** across 144 `.rubric.json` files; mean 5.2/task, median 4; auto-derived from gold, **reporting-only** — never used by the training judge |
| Grading | deterministic `gold_score.py` (component-weighted 0–100 formulas per category, e.g. bank_rec = 60·item_F1 + 20·balances + 20·je_F1) on heldout; a **gold-blind** LLM training judge (0–10, median of 3 samples) supplies the optimization pseudo-reward; judge-vs-gold divergence is the reward-hacking measurement |
| World files | mean **31.3 files/world** (29–34), mean **27 KB/world**, 1.0 MB total across all 36 worlds; formats **CSV (396) / TXT (586) / MD (144) only** — no XLSX/PDF/DOCX |
| Domain flavor | generic SMB month-end close: bank rec, payroll accrual, prepaid amortization, depreciation, misposted expenses, budget-vs-actual variance with seeded drivers |
| Run limits | orchestrator `max_steps: 40`, each subagent `max_steps: 25` (spec validator caps at 60); $60 total budget cap |
| Authorship | programmatic generator (seeded, byte-deterministic) + LLM narrative enrichment (numbers never LLM-written) + LLM red-team QA pass |

---

## 2. Contamination check — numbers, not vibes

**Method.** Full text was extracted from both corpora: APEX = all 106 world/task files (PDF via pypdf, XLSX via openpyxl, DOCX via python-docx, CSV/TXT raw) + all 10 prompts + all 10 gold outputs (126 text-bearing units); FinForge = all 36 worlds' `files/` trees + all 144 prompts + all 144 gold answers (1,414 text-bearing units). Lowercased word shingles were compared globally and per file pair.

### 2.1 N-gram shingle overlap

| Measure | APEX distinct | FinForge distinct | Intersection | Jaccard |
|---|---|---|---|---|
| **8-gram word shingles** | 426,449 | 140,746 | **0** | **0.000000** |
| Max-overlap file pair (8-gram) | — | — | 0 shared shingles | 0.000000 (no pair shares any) |
| 5-gram word shingles (sensitivity check) | 283,201 | 105,508 | **11** | 0.000028 |

All 11 shared 5-grams are generic accounting boilerplate, e.g.:
- `date account debit credit description` (CSV header convention)
- `1100 accounts receivable 40000 0`, `2000 accounts payable 1200 0` (chart-of-accounts row shapes)
- `please include the invoice number` — APEX Westfield contract-attorney invoices continue "…**on all payments**"; FinForge invoices continue "…**with remittance**". Standard invoice boilerplate, divergent immediately after the shared 5 words.

### 2.2 Entity-name overlap

Extracted from APEX: 112 vendor strings (`vendor_master_list.xlsx`), 81 client/matter strings (`client_matter_list.xlsx`), full staff roster (`attorney_staff_roster.xlsx`). Compared against FinForge's `generator/names.py` pools and all world text.

- **Exact multi-word entity overlap: 0.** No APEX vendor, client, matter, or staff name appears anywhere in FinForge.
- Exact single-word "overlaps" are all generic vendor-category labels (`Catering`, `Cleaning`, `Insurance`, `Marketing`, `Software`, `Supplies`, `Telecom`, `Utilities`, `Events`).
- Six coincidental **name-fragment** collisions exist, each in an unrelated full entity:

| Token | APEX entity | FinForge entity | FF worlds affected |
|---|---|---|---|
| **Whitfield** | Whitfield Family Trust (the dev set's marquee contingency-settlement client; 3 of 10 tasks touch it) | "A. Whitfield" — the controller who signs close checklists/memos | 4 (w05, w17, w19, h04) |
| Hartwell | Hartwell Construction Inc. (client) | "Hartwell Group" (vendor pool) | 32 |
| Eastgate | Eastgate Plaza Development LLC (client) | "Eastgate Collective" (vendor pool) | 21 |
| Vertex | Vertex Technologies Inc. (client) | "Vertex Materials" (vendor pool) | 2 |
| Cascade | Cascade Foods Holdings Inc. (client) | "Cascade Supply Partners" (pool, unused) | 0 |
| Larkspur | Larkspur Properties Inc. (client) | "Larkspur Home Goods" (pool, unused) | 0 |

These are common surname/place fragments attached to different businesses; none carries any APEX data. Still, "Whitfield" is the most distinctive string in the APEX dev storyline — renaming FinForge's controller is cheap insurance (see §5).

### 2.3 Distinctive-string grep (FinForge worlds + gold, and code/docs)

| APEX-distinctive string | Hits in worlds+gold | Hits in code/docs | Context of any hit |
|---|---|---|---|
| Sterling | 0 | 0 | — |
| Marsh | 0 | 1 | `"Marsh Rd"` in the `STREETS` pool of `generator/names.py`; **never selected into any generated world** |
| IOLTA / Clio / QuickBooks / QBO | 0 | 0 | — |
| Philadelphia / Mercor / Ramp | 0 | 0 | — |
| Bayshore / Avila / Kao / Westfield / Gusto | 0 | 0 | — |
| "trust account" / "law firm" / "guaranteed payment" / contingency | 0 | 0 | — |
| attorney | 1 file | 0 | "Attorneys & Counselors" letterhead on a fictional vendor bill (Meridian Legal LLP, w09) — generic |
| retainer | 53 occurrences | 1 | "Retainer Revenue" GL account in services-industry worlds — generic professional-services revenue, no trust/IOLTA mechanics |
| matter | (English word only) | — | "process discipline matters", "events that matter" — not law-firm matters |

`DESIGN.md` line 3 names APEX openly ("APEX-inspired synthetic accounting benchmark") — attribution, not contamination.

### 2.4 Chart-of-accounts overlap

9 of APEX's 61 account names match FinForge's 39-name union — all textbook GAAP names (Accounts Receivable, Accounts Payable, Accumulated Depreciation, Bank Fees, Depreciation Expense, Interest Income, Office Supplies, Rent Expense, Retained Earnings). Three share both name **and** number: `1100 Accounts Receivable`, `2000 Accounts Payable`, `1590 Accumulated Depreciation` — standard QuickBooks-style numbering conventions used by essentially every US small-business COA. All 52 law-firm-specific APEX accounts (Unbilled WIP, Client Cost Advances, IOLTA Trust, Prepaid Malpractice Insurance, partner capital accounts, …) have no FinForge counterpart.

**Contamination verdict: NONE.** Zero copied text at the 8-gram level, zero shared entities, zero distinctive strings, only universal accounting vocabulary and six coincidental surname fragments in different full names.

---

## 3. Task-shape comparison: one reconciliation, side by side

APEX's dev set has no bank rec, so its closest match to FinForge `bank_rec` is **Task 13 (Reconciliation / AR rec)**.

**APEX World 9 Task 13** (full prompt, 129 words — the dev set's longest):

> "Sterling, Marsh & Associates LLP needs the December 2024 accounts receivable reconciliation between Clio (billing system) and QuickBooks Online (accounting system) finalized. A staff accountant prepared a draft reconciliation … Using the provided files, determine the correct total AR balance per QBO and compare it to the Clio outstanding balance. Evaluate each matter-level variance identified in the draft, and state the correct December write-down and write-off totals - identify any misclassified amounts or incorrect variances, calculate the correct corresponding balances, and state the final reconciliation result with any required corrections.
> Provide all numerical values to two decimals, and **share your answer in the console**."

**FinForge w01 `bank_rec`** (prompt excerpt):

> "You are closing the June 2026 books for Ember Signal Inc. (Mesa, AZ). Reconcile the operating cash account (GL account 1000) to the June bank statement. Identify every reconciling item, state the bank statement ending balance, the GL cash balance at 6/30, and the adjusted (true) cash balance both sides tie to, and propose the correcting journal entries the books need …
> Deliverable JSON schema:
> `{ "bank_statement_ending_balance": <number>, … "reconciling_items": [{"kind": "outstanding_check|deposit_in_transit|…", "amount": <positive number>, … "side": "bank|book"}], "proposed_journal_entries": [...] }`
> Submit your final answer with the `submit_answer` tool. `answer_json` must match the schema above exactly."

Both open with company + period, state a reconciliation objective in a controller's register, and require every break identified with corrections. The divergences:

| Axis | APEX Task 13 | FinForge bank_rec |
|---|---|---|
| Rec pairing | subledger system (Clio) vs GL (QBO) | bank statement vs GL cash |
| Deliverable | free-form console prose; format left to the model, rubric-judged by an LM | schema-pinned JSON; format is part of the contract, scored by formula |
| Answer taxonomy | model must invent the framing (write-down vs write-off vs subledger error) | closed `kind` enum names the 6 legal break types up front |
| Evidence hunt | 5 context files among **90+16** world files, incl. a task-specific draft XLSX and reviewer-notes PDF; multi-format parsing (XLSX + PDF) | ~31 plain-text files, no task-specific files; single CSV dialect per world |
| Judgment layer | evaluate a *wrong human draft* against supervisor/controller guidance memos | detect machine-injected discrepancies against clean-by-construction books |
| Step/token budget | 500 steps / 5M tokens | orchestrator 40 steps, subagents 25 (≤60 by validator) |

The APEX task grades whether the model can produce a professional deliverable and adjudicate conflicting human guidance; the FinForge task isolates retrieval + arithmetic + classification into a deterministic target. That is the intended trade (deterministic gold scoring requires the pinned schema) but it removes the "decide what a good answer even looks like" dimension APEX's qualitative criteria grade.

---

## 4. Dimension-by-dimension verdict

| Dimension | APEX-accounting | FinForge | Verdict |
|---|---|---|---|
| Core premise: agent dropped into a synthetic company's files at month-end close | ✓ | ✓ | **Same** |
| World model: one synthetic company backs N tasks; agent must locate its own evidence (no file manifest given) | ✓ | ✓ | **Same** |
| Task taxonomy | Reconciliation / Data Entry / Variance Analysis / Schedules & Accruals | bank_rec / journal_entries / variance_analysis / accrual_schedule | **Similar** — 1:1 category mapping (Data Entry ↔ journal_entries), but FinForge instantiates exactly one fixed task per category per world vs APEX's varied subcategories |
| Prompt register | concise controller's ask, company named, no method hand-holding | same, + inline JSON schema | **Similar** (73.6 vs 143.9 mean words; delta is entirely the schema block) |
| Rubric form | binary, outcome-based, unweighted; 8.9/task; expert-written; drive scoring | binary criteria strings; 5.2/task; auto-derived from gold; reporting-only | **Similar** in form, **different** in provenance and role |
| Deliverable format | free-form console message | schema-pinned JSON via `submit_answer` tool | **Different** (deliberate) |
| Scoring | LM-judge Mean Criteria@k on rubric criteria | deterministic component formulas vs gold; gold-blind LM judge exists only as training pseudo-reward | **Different** (deliberate; FinForge's judge/gold split is its research object) |
| Domain content | law firm: IOLTA, Clio WIP, retainer trust, contingency, partner comp | 6 non-legal industries: saas/ecommerce/wholesale/services/manufacturing/clinic | **Different** (deliberate) |
| Company/people/vendor names, amounts, documents | Sterling Marsh universe | disjoint generated namespaces | **Different — zero overlap (§2)** |
| File formats | 53 PDF / 39 XLSX / 10 CSV-ish exports, 3.8 MB/world | CSV/TXT/MD only, ~27 KB/world | **Different** — and a real difficulty gap (see below) |
| World scale | 90 shared + up to 9 context files/task | ~31 files, ~12–18 substantive | **Different** (~3× fewer files, ~140× fewer bytes) |
| Task-specific files (`task_files/`) | 16 files across 8/10 tasks | none | **Absent** in FinForge |
| Accounting-system tool layer over exports | injected at runtime by harness | plain file tools + `run_python` | **Absent** in FinForge |
| Step/token budget | 500 steps / 5M tokens | 40 (+25/subagent) steps, $60 total cap | **Different** (order of magnitude tighter) |
| Human expert authorship + frontier-model difficulty filtering | ✓ | programmatic + LLM enrichment/QA | **Absent** in FinForge |
| Held-out split discipline | 160 tasks/10 worlds never released | 6 heldout worlds incl. 2 held-out **industries** | **Similar** in spirit; FinForge adds industry-level generalization APEX doesn't test |
| Trap/discrepancy seeding with a truth register | trap register per world spec | `TruthRecord` per injected discrepancy | **Similar** (independent mechanisms, same idea) |

### Contamination verdict

**Clean.** 0 shared 8-gram shingles (426k vs 141k distinct), 0 shared entities among 193+ APEX named entities checked, 0 hits on every APEX-distinctive string in generated content, max file-pair similarity 0.000000. The 11 shared 5-grams and 9 shared COA names are universal accounting vocabulary. Nothing in FinForge was derived from, seeded by, or textually influenced by APEX content.

---

## 5. Overall judgment

FinForge's claim of "APEX-inspired but distinct" is **legitimate and well-executed**: it reproduces exactly the structural skeleton that makes APEX interesting (synthetic company worlds, self-located evidence, the four close-work categories, binary rubrics, controller-voice prompts, capped agentic runs) while sharing literally zero content — a cleaner separation than most "inspired-by" datasets achieve, helped by the fact that FinForge's numbers are machine-generated under disjoint name pools.

**Too similar (should change):** two cosmetic name collisions. Rename `"A. Whitfield"` in `generator/names.py:98` (it appears in 4 shipped worlds and collides with the surname of APEX's marquee contingency client, inviting exactly the suspicion this audit exists to dispel) and `"Marsh Rd"` in `names.py:105` (currently unused in any world, but it is half of "Sterling, Marsh"). Both are one-line pool edits plus regeneration of the affected worlds.

**Too different (weakens the comparison):** three axes materially lower FinForge's difficulty relative to APEX, so FinForge scores must not be framed as APEX-comparable: (1) plain-text-only files remove the XLSX/PDF extraction burden that dominates real accounting evidence work; (2) worlds are ~140× smaller by bytes and ~3× smaller by file count, shrinking the retrieval haystack; (3) the schema-pinned deliverable plus closed `kind` enums convert APEX's open professional-judgment tasks into structured extraction. All three are defensible — they are what make deterministic gold scoring and a $60 self-improvement loop possible — but the writeup should state them as scope reductions, not merely surface changes.

**Unrelated internal finding:** 3 of 6 heldout worlds reuse train-world company names (h01 "Beacon Ledger Software" = w17; h02 "Tidewater Outfitters" = w26/w30; h03 "Clearpath Partners LLC" = w04), contradicting DESIGN §4.1's "unseen companies" for the held-out split. Worth fixing alongside the Whitfield rename.

---

## Appendix: reproduction notes

- APEX download: `snapshot_download('mercor/apex-accounting', repo_type='dataset', local_dir='/tmp/apex')` — 120 files, 4.6 MB.
- Shingling: lowercased `[a-z0-9']+` tokens, 8-gram (and 5-gram) tuples, set intersection/Jaccard, global and per-file-pair; PDF text via pypdf, XLSX via openpyxl, DOCX via python-docx.
- FinForge corpus: `worlds/**` (1,126 files) + 144 prompts from `gold/*/*/tasks.json` + 144 gold-answer JSONs.
- Entity extraction: APEX `vendor_master_list.xlsx`, `client_matter_list.xlsx`, `attorney_staff_roster.xlsx`; FinForge `generator/names.py` string pools + full world text.
- All statistics in this report were computed on 2026-09-19 against the repo state at branch `master`.
