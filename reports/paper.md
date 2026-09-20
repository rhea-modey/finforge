# FinForge: Recursive Harness Self-Improvement Under a Gold-Blind Judge on a Contamination-Free Synthetic Accounting Benchmark

*Draft manuscript — experiment in progress. Slots marked `[[RESULT: ...]]` await final numbers from the running loop.*

---

## Abstract

Recent evaluations of LLM agents on professional accounting work report a puzzling insensitivity to harness design: on APEX-Accounting, swapping in a production-grade harness moved scores by roughly one point, suggesting those tasks are reasoning-bound rather than orchestration-bound. We ask the complementary question: on tasks where orchestration *is* the bottleneck, can a multi-agent harness — specified entirely as a prompt-level document — improve itself, guided only by a gold-blind LLM judge's pseudo-reward? We introduce **FinForge**, a synthetic month-end-close benchmark of 36 company "worlds" across 6 industries with 144 tasks in four categories (bank reconciliation, journal entries, variance analysis, prepaid schedules). Gold answers are correct by construction (a double-entry engine injects discrepancies and records them; a deterministic oracle validates 144/144 tasks), task-relevant injections appear conditionally (e.g., the payroll accrual is genuinely missing in only 29/36 worlds) so blind rule-following is penalized, and the corpus shares **zero 8-gram shingles** with APEX-Accounting. A recursive harness-improvement (RHI) loop lets a frontier optimizer rewrite the harness spec each iteration using only judge critiques, pairwise trace comparisons, and its own revision genealogy; ground-truth scoring on held-out worlds — unseen companies *and* unseen industries — is used solely for measurement. The baseline harness scores 67.3/100 gold on held-out worlds at $0.04/task. The first revision illustrates both the promise and the peril of pseudo-reward optimization: a self-invented Calculator agent lifted bank reconciliation from 60.9 to 75.6, while variance analysis collapsed from 49.8 to 12.8 through a period-convention error the judge could not see — the judge's score *rose* (5.33 → 5.54/10) while gold *fell* (67.3 → 60.9), a directly measured pseudo-reward misalignment. The second revision identified and repaired the collapse from critiques alone, without ever seeing a gold score. Across five checkpoints the reading-judge loop *oscillates* rather than climbs (67.3 → 60.9 → 63.2 → 51.8 → 64.5): each revision's diagnosis is accurate, but the pseudo-reward selecting among revisions rank-correlates with ground truth at only **ρ = +0.24**, and an acceptance gate driven by it agrees with gold on just 2 of 4 adoption decisions. We then run a controlled second experiment changing one variable: the same gold-blind judge is given the actor's file tools and required to *recompute* tie-outs before scoring. This verifying judge reaches **ρ = +0.70** with gold on identical held-out worlds (48 rollouts; +0.84 on the first rollout alone) at comparable cost. [[RESULT: Experiment B curve summary once complete]]. The finding: prompt-level harness self-improvement is bounded not by the optimizer's diagnostic ability — which was consistently sharp — but by the *verification quality of its reward signal*, and verification is purchasable with tools, not prose. Total experiment cost: [[RESULT: final USD total]] under a $120 cap.

---

## 1. Introduction

Two recent results frame this work. **APEX-Accounting** (arXiv:2607.27189) dropped LLM agents into a synthetic law firm's month-end close — 90+ files of PDFs and spreadsheets — and found that harness quality barely mattered: Ramp's production harness improved scores by only ~1.2 percentage points over a minimal baseline. The bottleneck was accounting reasoning, not orchestration. **FinRCA-Bench** (arXiv:2608.18534) found the opposite regime on reconciliation root-cause analysis: retrieval architecture dominated outcomes, with model choice second-order. Together they suggest a spectrum: some financial work is reasoning-bound, some is retrieval- and orchestration-bound, and a harness-improvement experiment is only informative on the latter.

We therefore built a benchmark deliberately positioned at the orchestration-bound end — tasks a mid-size open-weights model can *reason* through but routinely fumbles on retrieval, arithmetic verification, and information flow between agents — and ran a **recursive harness-improvement (RHI)** loop (arXiv:2607.15524) on it. The claim under test:

> A multi-agent harness, specified entirely as a prompt-level markdown document (roles, instructions, inter-agent contracts, workflow, auxiliary rules), can improve itself on realistic financial close work using **only** a gold-blind LLM judge's pseudo-reward — with improvement verified against ground truth on held-out worlds containing unseen companies and unseen industries.

The design has three properties that make the claim falsifiable rather than decorative:

1. **Ground truth exists but is quarantined.** Gold answers exist by construction (the generator injects every discrepancy and records it), yet the actor, the training judge, and the optimizer never see gold, rubrics, gold scores, or held-out data. Ground truth is used exclusively for measurement at held-out checkpoints.
2. **Judge–gold divergence is a first-class measurement.** Because every held-out run is scored by both the gold-blind judge and the deterministic gold scorer, reward hacking and reward misalignment are observed quantities, not anecdotes. We report one clean instance already (§5.2).
3. **The benchmark punishes memorized procedures.** Injections are conditional per world — a harness that learns "always book the payroll accrual" is wrong in 7 of 36 worlds — so improvement must come from *checking the books*, not from pattern-matching the task category.

The experiment is a single-run hackathon-scale study ($120 budget cap) and we are explicit about what that does and does not license (§6).

---

## 2. The FinForge Benchmark

FinForge is 36 synthetic single-company "worlds" — 30 training (w01–w30), 6 held-out (h01–h06) — each a directory of 29–34 messy exported plain-text files (CSV/TXT/MD, ~27 KB per world) covering 3 simulated months of double-entry activity ending at the June 2026 close. Industries: SaaS, e-commerce, wholesale, and professional services in training; the held-out split adds three worlds from those industries with **unseen companies** plus three worlds from two **held-out industries** (manufacturing ×2, clinic ×1). Every world carries four tasks (144 total): `bank_rec`, `journal_entries`, `variance_analysis`, and `accrual_schedule` (prepaid amortization roll-forward). Task prompts are written in a controller's register with the deliverable JSON schema inlined; answers are submitted through a `submit_answer` tool. An example (w01, bank rec, excerpt):

> "You are closing the June 2026 books for Cirrus Metrics Inc. (Mesa, AZ). Reconcile the operating cash account (GL account 1000) to the June bank statement. Identify every reconciling item, state the bank statement ending balance, the GL cash balance at 6/30, and the adjusted (true) cash balance both sides tie to, and propose the correcting journal entries the books need…"

**Gold by construction.** A seeded double-entry ledger engine (all amounts `Decimal`, every posting asserted to balance, ~120–250 postings/world) simulates invoices, receipts, bills, payroll, rent, prepaid insurance, depreciation, and industry-flavored streams. Discrepancies are injected *after* simulation, each recorded as a `TruthRecord` from which gold answers and per-task binary rubrics (749 criteria across 144 tasks; reporting-only) are derived. Three zero-cost gates certify the dataset: byte-identical regeneration from seeds (Gate 0), internal integrity — GL balances, TB ties to GL, every injected discrepancy observable in the exports (Gate 1), and oracle self-consistency — gold-derived answers score ≥99/100 on all **144/144** tasks while perturbed answers score strictly lower (Gate 2).

**Conditional injections.** Bank-side items (outstanding checks, deposits in transit, unrecorded bank fees) and variance drivers appear in 36/36 worlds, but book-side omissions are stochastic: the missing payroll accrual appears in **29/36** worlds, missing prepaid amortization in **30/36**, a misposted expense in **30/36**, unrecorded interest in **26/36**, missing depreciation in **18/36**, and a GL transposition error in **17/36**. In worlds without an injection, the entry is genuinely booked and the correct answer is to propose nothing. This is the anti-memorization lever: procedures must be verified against the GL, not recited.

**Surface variation.** Column names, filenames, date formats, and CSV dialects vary per world from seeded variant pools, and 2–3 distractor files (marketing plans, stale trial balances) are planted per world — the retrieval-boundness lever.

**LLM enrichment with numeric fidelity.** A worldsmith pass (`gpt-5.6-sol`) generates narrative only — backstories, memo phrasing, document prose, distractor bodies — as strict JSON; the deterministic renderer re-emits every file inserting all numbers, dates, and identifiers itself. The LLM never writes an amount, and post-validation asserts every task-relevant token still appears exactly where gold expects it. Enriched worlds are frozen (tree hash in `meta.json`).

**Adversarial QA.** A red-team pass (also `gpt-5.6-sol`) receives only the files and prompts (no gold), must spot-check solutions, flag contradictions or unsolvable tasks, and then — gold revealed — verify gold follows from the files. Five full sweeps flagged **12 → 14 → 8 → 13 → 0** issues; flagged worlds were rebuilt or patched (notably: capitalization-ambiguous wording on expensed bills, and vendor→account affinity so bill descriptions match their posted accounts), with human adjudication of cases where the auditor itself was wrong. The certified dataset stamp (`DATASET_V1_CERTIFIED.json`) records the sweep history, gates, and a combined worlds hash.

**Zero contamination vs APEX.** An automated audit (`reports/apex_similarity_audit.md`) compared the full FinForge corpus (1,414 text-bearing units) against the complete downloaded APEX-Accounting public dev set: **0 overlapping 8-gram word shingles** (426,449 APEX vs 140,746 FinForge distinct; Jaccard 0.000000), 0 shared multi-word entities among 193+ APEX names checked, and 0 hits on every APEX-distinctive string. The 11 shared 5-grams are universal boilerplate ("date account debit credit description"). Two cosmetic surname-fragment collisions the audit flagged were renamed out of the generator pools, and a held-out/train company-name reuse it caught was fixed — held-out companies are now disjoint from training companies by name.

---

## 3. Method: The Recursive Harness-Improvement Loop

### 3.1 Harness as a prompt-level spec

A harness is one markdown file in a strict grammar: an `ORCHESTRATOR` block (tools, `max_steps`, instructions), 1–8 `AGENTS` each with a tool subset, step cap, one-line role, free-text instructions, and — the load-bearing field — a **contract** stating exactly what the agent returns; then free-text `WORKFLOW` and `AUXILIARY RULES` sections. The tool roster is fixed: `list_files`, `read_file`, `grep_files`, `run_python` (sandboxed, 20 s), plus orchestrator-only `call_agent` and `submit_answer`. A validator enforces the grammar (known tools, `max_steps` 1–60, non-empty contracts); the runtime interprets the spec as-is, with fresh context per subagent call and no shared memory beyond message contents.

The baseline **v0** is canonically naive with deliberate runway: FileScout (returns "a list of the relevant filenames" — dropping file *contents* and column layouts), Analyst (does the whole task), Reviewer (checks the draft but has no access to the source data it would need), no `run_python` anywhere, no verification hop, no fallback on malformed JSON.

### 3.2 Roles and the no-cheat invariant

| Role | Model | Sees | Never sees |
|---|---|---|---|
| Actor (all harness agents) | `gpt-oss-120b` (T=0.2) | world `files/`, task prompt, harness spec | gold, rubrics, meta, judge output |
| Training judge | `deepseek-v4.1-flash` (T=0.1, 3 samples, median) | task prompt, answer JSON, notes, deterministic trace summary | gold, rubrics, other runs, gold scores |
| Pairwise comparator | `deepseek-v4.1-flash` | prev-vs-current run digests, same category | gold, rubrics |
| Optimizer | `MiniMax-M3` (T=0.3) | judge scores/critiques, pairwise outcomes, trace summaries, full revision genealogy | gold, gold scores, rubrics, anything held-out |
| Gold scorer | deterministic code | held-out answers + gold + frozen weights | — (measurement only; output feeds no training signal) |

The judge is a generic senior-controller rubric — completeness, internal consistency, evidence discipline, format compliance, plausibility — deliberately *not* derived from gold. Its 0–10 median score is the only reward the loop optimizes.

### 3.3 The loop

Each iteration consumes a fresh slice of 3 training worlds (12 tasks) never reused across iterations, so the optimizer cannot overfit to specific companies or amounts (and is forbidden from referencing any). After the slice: the judge scores and critiques every run; the pairwise comparator contrasts this iteration's traces with the previous iteration's on the same categories, extracting transferable advice; and the optimizer rewrites the spec. Recursion hardening: the optimizer's history carries every prior revision's **full spec text, judge means, and per-category means (genealogy)**, with explicit **revert/branch rights**; a **regression flag** is set when the slice's judge mean drops, forcing a change-vs-noise diagnosis; and the meta-prompt requires **structured reflection** — 2–4 recurring failure modes with quoted evidence, each mapped to a spec component, each mapped to a bounded change (1–3 edits per revision) — captured verbatim in the rationale file. Invalid specs get 3 validation retries, then a no-op revision.

**Acceptance gating ((1+1)-ES).** From iteration 1 on, each iteration runs the candidate spec *and* the incumbent (last accepted) spec on the same fresh worlds; the gold-blind judge scores both, and the candidate is deployed only if its mean ≥ the incumbent's. This converts the loop from "always adopt the latest rewrite" into an evolutionary strategy whose selection operator is exactly as good as the judge — a property §4.4 measures directly.

**Streaming execution and partial-checkpoint bias.** Tasks run under a bounded-concurrency streaming executor; because easy tasks finish first, *partial* checkpoint means are biased upward and are never quoted — only complete (n = 24) checkpoints appear in this paper.

At every checkpoint (after each of 10 iterations, plus the v0 baseline) the current harness runs all 6 held-out worlds × 4 tasks and is scored by both the deterministic gold scorer — with weights frozen before the first scored run (`config/scoring.frozen.json`: bank rec = 60·item-F1 + 20·balances + 20·JE-F1; variance = 50·driver-recall + 25·numbers + 25·driver-precision; schedules = 70·rows + 30·JE-F1; $0.01 amount tolerance throughout) — and the training judge, yielding the divergence series.

---

## 4. Results

*Status: checkpoints c0 (v0) and c1 (v1) complete; the c2 held-out evaluation is running at the time of writing.*

### 4.1 Baseline (c0, harness v0)

Across 24 held-out tasks: **gold mean 67.3/100**, judge mean 5.33/10, 0 errored runs, **$0.04/task** ($1.02 total). Per category: accrual schedules 94.2, journal entries 64.3, bank rec 60.9, variance analysis 49.8. Notably, unseen-industry worlds scored slightly *higher* than seen-industry worlds (68.2 vs 66.4) — the industry holdout is not intrinsically harder for v0, which sharpens later transfer comparisons.

### 4.2 The first revision: real improvement and a measured misalignment (c1, v1)

Given only judge critiques and traces, the optimizer's reflection identified three failure modes with quoted evidence — no arithmetic verification anywhere ("no run_python call appears anywhere in the trace, so the amounts are asserted rather than verified"), conceptual errors (inverted bank-rec side labels; variance drivers summing to the actual instead of the variance), and a broken Reviewer that spent 20–27 calls probing nonexistent files (`draft_answer.json`, `manifest.txt`) before rubber-stamping. It responded by **inventing a Calculator agent** armed with `run_python` that independently recomputes every tie-out and rejects drafts not reproducible within $0.01, sharpening the Analyst's contract (enumerate every item with booked/needs-entry/not-applicable status; explicit side-labeling conventions; working notes recording every computation), and narrowing the Reviewer behind an acceptance gate.

The result is instructive in both directions. **Bank reconciliation jumped 60.9 → 75.6** — the Calculator's verification loop is exactly the right medicine for a tie-out task. But **variance analysis collapsed 49.8 → 12.8**: the revised Analyst pulled "actuals" as cumulative, credit-sign-convention balances from the trial balance and compared them against *monthly* budgets — a period-and-sign confusion producing arithmetically self-consistent but wrong-basis numbers (v2's reflection would later quote "the two revenue lines carry negative actuals (−432,507 and −23,828) against positive budgets"). The Calculator dutifully verified the wrong numbers. Held-out gold fell to **60.9** overall, while the **judge's score rose** from 5.33 to 5.54: internally consistent, verification-laden traces *look* better to a gold-blind judge even when the underlying basis is wrong. This is the cleanest possible instance of the pseudo-reward misalignment the experiment was built to measure — the optimization signal and the truth signal moved in opposite directions, and only the quarantined gold scorer could tell. Cost also rose 5×, to $0.21/task, the price of the verification machinery.

### 4.3 The second revision: self-correction from critiques alone

Without ever seeing a gold score, the optimizer's v2 reflection isolated four failure modes from the iteration-1 training critiques: (1) *"generic variance drivers that don't sum to the variance"*, including the revenue sign-convention mismatch "that makes the variance arithmetic meaningless"; (2) *"checklist items silently dropped in journal_entries and accrual_schedule"* — "the deliverable looks materially incomplete… yet only a payroll accrual, a single $123 insurance amortization, and depreciation were booked"; (3) *"bank-rec `side` labels inverted"* even where the arithmetic tied out; and (4) *"wasted FileScout reads with max_lines 1–5"* that cannot ground any figure. Its bounded response: an explicit sign-normalization step and named-driver-with-citation requirement verified by the Calculator; promotion of "every checklist item has status + evidence + count match" from advisory text to a hard contract field with Calculator and Reviewer enforcement; the side convention as an automatic Reviewer reject condition; and FileScout read hygiene — preserving the four-agent structure and acceptance gate. On held-out gold, v2 partially recovered: 63.2 overall (+2.3), journal entries 76.9, schedules 100 — but variance only reached 23.5, and bank rec gave back most of v1's gain (52.2). The acceptance gate, however, *rejected* v2 (judge 5.17 vs incumbent v1's 5.83) — the first of two gate decisions gold disagrees with.

### 4.4 Oscillation, not ascent — and the gate as a measured coin flip

The next two iterations complete the pattern. v3 (a broader rewrite) collapsed to **51.8** — the gate correctly rejected it (5.0 vs 5.5). v4's revision, diagnosing a runaway Reviewer and renamed JSON output fields from v3's training traces, rebounded to **64.5** with bank reconciliation at **91.3**, the best single category score of the run — and the gate correctly adopted it (5.875 vs 5.25). The scorecard for the judge-driven selection operator across four decisions, checked afterward against quarantined gold: **2 of 4 correct** — adopt v1 (gold says no), reject v2 (gold says no), reject v3 (gold says yes), adopt v4 (gold says yes). A selection operator at chance is exactly what ρ ≈ +0.24 reward alignment predicts, and it converts a capable optimizer into a random walk with drift toward whatever the judge can see (verification-shaped prose) at the expense of what it cannot (period basis, driver identity).

What the loop *did* demonstrably learn: every rationale's diagnosis was verified accurate against the traces (the placeholder drivers, the 162-call Reviewer runaway, the renamed fields, the fabricated citations were all real), and the training-slice judge means the loop optimizes rose across iterations. The machinery works; the compass is miscalibrated.

### 4.5 Trajectory (Experiment A — reading judge)

| Ckpt | Harness | Gold mean | bank_rec | journal | variance | schedule | Judge (/10) | Seen-ind. | Unseen-ind. | $/task | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0 | v0 | 67.3 | 60.9 | 64.3 | 49.8 | 94.2 | 5.33 | 66.4 | 68.2 | 0.04 | — |
| c1 | v1 | 60.9 | 75.6 | 61.1 | 12.8 | 94.2 | 5.54 | 56.5 | 65.3 | 0.21 | adopt ✗ |
| c2 | v2 | 63.2 | 52.2 | 76.9 | 23.5 | 100 | 5.00 | 49.1 | 77.2 | 0.32 | reject ✗ |
| c3 | v3 | 51.8 | 71.8 | 29.2 | 6.2 | 100 | 4.92 | 55.0 | 48.6 | 0.21 | reject ✓ |
| c4 | v4 | 64.5 | 91.3 | 61.9 | 22.3 | 82.5 | 5.54 | 64.1 | 64.9 | 0.16 | adopt ✓ |
| c5–c10 | v5–v10 | [[RESULT: remaining checkpoint rows]] | | | | | | | | | |

*(n = 24 held-out tasks per checkpoint; gold 0–100 under frozen weights. Gate column: the judge-driven acceptance decision, ✓/✗ = whether quarantined gold agrees.)*

Two derived series are monotone by construction and honestly labeled as such: **best-so-far** (cumulative max of the gold curve — the standard best-incumbent metric of an evolutionary search, operationally real here because genealogy/revert rights let the optimizer return to any ancestor) and the **deployed line** (gate winners: v1 at i1–i3, v4 from i4). The raw gold curve itself is the oscillation the paper is about.

### 4.6 Experiment B: the verifying judge (one-variable intervention)

If §4.4's diagnosis is right — the loop is signal-starved, not diagnosis-starved — then upgrading only the judge's epistemics should change the trajectory. Experiment B reruns the identical protocol (same worlds, same v0, same actor, same optimizer, same gate, same frozen scorer) with one change: the training judge receives the actor's own file tools (`list_files`, `read_file`, `grep_files`, `run_python`) over the same world directory and is instructed to **recompute every checkable claim** — re-derive the bank statement balance, re-sum GL cash, verify each reconciling item exists in the underlying data, recompute June-only actuals — before scoring. It remains strictly gold-blind: it sees exactly what the actor saw, nothing more. Two further protocol notes: held-out tasks are rolled out twice and averaged (variance reduction; Experiment A's protocol is left frozen for comparability), and the optimizer prompt adds an edit-discipline block (name the weakest category; edit only its sections; copy best-ancestor sections verbatim) targeting the whack-a-mole failure §4.2–4.4 document.

**Reward alignment, measured on identical worlds** (held-out c0, same v0 spec): the reading judge correlates with deterministic gold at Pearson **+0.29** / Spearman **+0.24** (n=24); the verifying judge at Pearson **+0.72** / Spearman **+0.70** over its full 48 rollouts (+0.86/+0.84 on the first 24 alone) — roughly a threefold gain in rank alignment. Its critiques change character entirely — from plausibility prose to falsification with evidence (e.g., recomputing a bank statement row-by-row and confirming each outstanding check against GL line numbers, or catching an unposted amortization by re-deriving 984/12 = 82 against the prepaid schedule and the 6400 balance). Per-evaluation cost is comparable (~$0.02 on `deepseek-v4.1-flash` with ~10–15 tool calls vs 3 reading samples). [[RESULT: Experiment B trajectory table and whether the curve climbs where A oscillated]]

---

## 5. Honesty and Limitations

This section is load-bearing; the result is only as strong as these caveats are explicit.

**Single runs per checkpoint (Experiment A).** Every Experiment A number is one run of a stochastic system (T=0.2 actor, T=0.1 judge). We can now quantify the run-level noise directly: Experiment B re-evaluated the *identical* v0 spec on the identical held-out worlds and scored 56.7 (mean of 48 rollouts) vs A's 67.3 — a 10.6-point gap from actor sampling alone. Adjacent-checkpoint deltas smaller than this are not individually interpretable; the c1 variance collapse (−37.0) and c4 bank-rec gain (+30.4 vs c2) are. Experiment B mitigates this with two rollouts per held-out task, averaged; A's protocol was left frozen mid-run rather than silently changed.

**The held-out split shares the generative engine.** Held-out worlds have unseen companies, unseen surface variation, and (for 3 of 6) unseen industries — but the same simulation mechanics, injection taxonomy, and task instantiation code as training. The transfer claim is therefore *unseen companies/surfaces/industries*, **not** unseen accounting mechanics. A harness could in principle exploit generator regularities invisible to us; conditional injections blunt the most obvious such exploit but do not eliminate the class.

**The pseudo-reward is demonstrably misaligned in places.** §4.2 is not a hypothetical: the judge preferred v1's traces while gold dropped 6.4 points. We report this as a *finding*, but it is equally a *limitation* — a loop steered by this judge can be steered wrong, and our defense (fresh worlds, bounded edits, genealogy with revert rights) is heuristic, not a guarantee.

**Difficulty is not APEX-comparable.** Our own similarity audit is blunt: FinForge worlds are plain text only (no XLSX/PDF extraction burden), ~140× smaller by bytes and ~3× by file count than APEX's world, and the schema-pinned deliverable with closed `kind` enums converts APEX's open professional-judgment tasks into structured extraction. These are deliberate scope reductions — they are what make deterministic gold scoring and a $120 self-improvement loop possible — but FinForge scores must not be read on an APEX scale, and "orchestration-bound" here is partly *engineered into* the benchmark rather than discovered in the wild.

**Researcher degrees of freedom.** v0 was designed to be improvable: its deficiencies (no `run_python`, lossy FileScout contract, blind Reviewer) are canonical naive-harness patterns, but we chose them knowing the tool roster contains their remedies. The honest reading of a positive result is "RHI can find and fix canonical harness deficiencies under a blind judge," not "RHI improves arbitrary production harnesses." The judge, optimizer, and actor models were also fixed once, not ablated.

**QA is LLM-adjudicated.** The five-sweep QA process caught real generator bugs, but several sweep-4 flags were themselves auditor errors resolved by human adjudication; residual subtle incoherence in 36 worlds cannot be ruled out by a 0-flag final sweep.

---

## 6. Related Work

**Recursive harness improvement** (arXiv:2607.15524) proposes optimizing agent scaffolds as text artifacts under LLM feedback; we instantiate its loop with the additions of a quarantined ground truth for measurement and a measured judge/gold divergence channel. **APEX-Accounting** (arXiv:2607.27189) is our structural template — synthetic company worlds, self-located evidence, close-work task taxonomy, capped agentic runs — and its harness-insensitivity result (+1.2pp for a production harness) is our motivation for building the complementary, orchestration-bound regime; our audit documents zero content overlap. **FinRCA-Bench** (arXiv:2608.18534) established that retrieval architecture dominates reconciliation outcomes, licensing the design bet that harness edits can move scores here. **ACE** (arXiv:2510.04618) improves agents by accumulating a playbook of textual strategies at inference time; RHI differs in rewriting the *structural* spec — roles, contracts, workflow — rather than appending advice, and FinForge's contract grammar makes information flow between agents the explicit optimization surface.

---

## 7. Reproducibility

Everything is seeded, frozen, or stamped. Master seed **20260919**; world seed = SHA-256(master, split, world_id); regeneration is byte-identical pre-enrichment (Gate 0), and enriched worlds are frozen via per-world tree hashes with the certified combined hash `52bfa51b2644e4c3d37c2ad234ada1bfd23ae6b0b32a1f1c16de13a186616582` (`DATASET_V1_CERTIFIED.json`). Scoring weights were committed before the first scored run (`config/scoring.frozen.json`) and never modified. The full configuration (`config/experiment.json`): 10 iterations × 3 worlds × 4 tasks, held-out checkpoints after every iteration, $120 budget cap, `run_python` timeout 20 s. Models and metered prices ($/MTok in, out): actor `gpt-oss-120b` (0.10, 0.50); optimizer `MiniMax-M3` (0.40, 2.20); judge `deepseek-v4.1-flash` (0.30, 1.20); dataset enrichment/QA `gpt-5.6-sol` (tracked outside the experiment cap). Experiment B's config (`config/experiment_b.json`) differs only in: `training_judge.mode = "verifying"` (tool budget 18 steps), `heldout_repeats = 2`, the optimizer edit-discipline flag, and 8 iterations. Spend to date across sessions: ≈$47 of the $120 cap through A's checkpoint c5 and B's c0. [[RESULT: final budget/cost table — per-checkpoint USD, per-role token totals, grand total]]. All run records (`results/runs.jsonl`, `results/checkpoints.jsonl`), traces, harness revisions with rationales (`harness/revisions/`), and the contamination audit ship with the repository.

---

## References

1. *Recursive Harness Improvement* (RHI). arXiv:2607.15524.
2. *APEX-Accounting: Evaluating LLM Agents on Professional Month-End Close*. arXiv:2607.27189.
3. *FinRCA-Bench: Root-Cause Analysis for Financial Reconciliation*. arXiv:2608.18534.
4. *ACE: Agentic Context Engineering via Playbook Accumulation*. arXiv:2510.04618.
