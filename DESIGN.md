# FinForge — Design Contract v1.0

Self-improving harness experiment on an APEX-inspired synthetic accounting benchmark.
This document is the **interface contract**: every module implements exactly the
schemas and signatures here. If you need to deviate, update this file in the same
change.

---

## 1. The experiment in one paragraph

We generate synthetic month-end-close "worlds" (one company each: a directory of
messy exported files) with **four tasks per world** (bank reconciliation, journal
entries, variance analysis, accrual/prepaid schedule). Ground truth (gold) exists
by construction but is **never shown** to the harness, the training judge, or the
optimizer. A **multi-agent harness defined entirely as a prompt-level spec**
(`harness/v0.md`) runs each task. After each iteration (a slice of fresh training
worlds), a **gold-blind LLM judge** scores and critiques the traces
(pseudo-reward), a pairwise comparator contrasts this iteration's traces with the
previous iteration's, and an **optimizer LLM rewrites the harness spec**. After 10
iterations we check whether the evolved harness scores higher than v0 on
**held-out worlds** (including held-out industries), measured against **ground
truth**. Secondary result: judge-score vs gold-score divergence = measured reward
hacking.

## 2. Repository layout

```
finforge/
├── DESIGN.md                    # this file
├── requirements.txt
├── config/
│   ├── experiment.json          # models, iterations, budget, knobs
│   └── scoring.frozen.json      # scoring weights — commit before first scored run
├── generator/
│   ├── __init__.py
│   ├── names.py                 # seeded name/vendor/customer pools per industry
│   ├── profile.py               # CompanyProfile dataclass + industry templates
│   ├── engine.py                # double-entry ledger simulation (events → postings)
│   ├── discrepancies.py         # discrepancy injection + truth records
│   ├── exporters.py             # world files emission (messy CSV/txt/md)
│   ├── tasks.py                 # task instantiation, gold answers, auto-rubrics
│   └── build.py                 # CLI: python -m generator.build
├── worlds/
│   ├── train/w01 .. w30/files/  # ONLY files/ is visible to the agent
│   └── heldout/h01 .. h06/files/
├── gold/
│   └── {split}/{world_id}/{task_id}.json      # gold answers (never mounted)
│   └── {split}/{world_id}/meta.json           # profile, seed, injected truth
├── harness/
│   ├── v0.md                    # the naive multi-agent harness spec
│   └── revisions/v1.md .. vN.md + vN.rationale.md
├── runtime/
│   ├── __init__.py
│   ├── llm.py                   # provider clients + cost meter
│   ├── tools.py                 # file tools + run_python sandbox
│   ├── spec.py                  # harness spec parser/validator
│   ├── orchestrator.py          # spec interpreter (orchestrator + subagent loops)
│   └── trace.py                 # trace recording + summarization
├── judge/
│   ├── __init__.py
│   ├── training_judge.py        # gold-blind score + critique (k samples)
│   └── pairwise.py              # trace-vs-trace comparison
├── scoring/
│   ├── __init__.py
│   ├── gold_score.py            # deterministic per-category scorers vs gold
│   └── report.py                # results aggregation → results/summary.json
├── rhi/
│   ├── __init__.py
│   └── optimizer.py             # meta-prompt → new harness spec (validated)
├── experiment.py                # the loop; also `--heldout-only --harness vX`
├── scripts/
│   ├── smoke_generator.py       # Gate 0/1: determinism + integrity (no API)
│   └── oracle_check.py          # Gate 2: gold self-consistency (no API)
└── results/
    ├── traces/{run_id}.jsonl
    ├── runs.jsonl               # one line per task-run (see §10)
    └── checkpoints.jsonl        # held-out eval records
```

## 3. Config

`config/experiment.json` (defaults; all overridable):

```jsonc
{
  "actor":      {"provider": "anthropic", "model": "claude-opus-5", "effort": "medium",  "max_tokens": 8000},
  "optimizer":  {"provider": "anthropic", "model": "claude-opus-5", "effort": "high",    "max_tokens": 16000},
  "training_judge": {"provider": "judge_env", "model": null, "samples": 3, "max_tokens": 4000},
  "pairwise_judge": {"provider": "judge_env", "model": null, "max_tokens": 4000},
  "iterations": 10,
  "worlds_per_iteration": 3,
  "tasks_per_world": 4,
  "heldout_checkpoints": [0, 2, 4, 6, 8, 10],
  "budget_usd_cap": 60.0,
  "run_python_timeout_s": 20
}
```

Step budgets are controlled ONLY by the spec-level `max_steps` fields of the
harness (validator range 1–60, §6) — there are no config-level step knobs.

**Provider resolution (all roles).** The deployment: actor = a 30–100B
open-weights model, optimizer = a frontier model (MiniMax-class), judge =
DeepSeek v4.1 Pro — all served over OpenAI-compatible APIs. Per role
(`ACTOR`, `OPTIMIZER`, `JUDGE` — both judge configs share the `JUDGE` env
triple), env vars override config:
`{ROLE}_MODEL`, `{ROLE}_API_KEY` (judge accepts `JUDGE_LLM_API_KEY` too),
`{ROLE}_BASE_URL`. If `base_url` present or provider == `openai_compat` → the
`openai` package client (chat.completions with `tools` function-calling); if
the key starts `sk-ant-` or provider == `anthropic` → the `anthropic` SDK
(adaptive thinking semantics, no temperature). Missing key/model at call time →
clear startup error naming the env var. Temperature is used only on the
OpenAI-compatible path.

Cost: `pricing_per_mtok` map in config `{model_substring: [in, out]}` with
`_default` fallback [1.0, 3.0] USD/MTok (warn on fallback). `llm.py` exposes a
global `CostMeter` (tokens + USD by role) that `experiment.py` checks against
`budget_usd_cap` before each task; exceeding it halts gracefully after writing
records.

## 4. Worlds

### 4.1 Industries and splits

Train industries: `saas`, `ecommerce`, `wholesale`, `services` (professional
services). Held-out worlds: 3 from train industries (unseen companies) + 3 from
**held-out industries** `manufacturing`, `clinic`. 30 train worlds (w01..w30,
industries round-robin), 6 held-out (h01..h06). Master seed in
`config/experiment.json`-adjacent constant `MASTER_SEED = 20260919`; world seed =
`sha256(master, split, world_id)`. Regeneration must be byte-identical
(Gate 0). Use `random.Random(seed)` instances only — never the global RNG, never
`datetime.now()`.

### 4.2 The ledger engine (`engine.py`)

Simulates **3 months** of double-entry activity ending at the close month
(month 3 = "the month under close", e.g. 2026-06). All money as
`decimal.Decimal`, 2dp, engine asserts every posting balances. Event kinds
(volume scaled by profile size, ~120–250 postings/world):

- customer invoices + receipts (some memo-mangled at export)
- vendor bills + payments (checks — some uncleared at month end)
- payroll runs (2/mo: gross, withholding, employer taxes, cash)
- rent, utilities, recurring subscriptions
- prepaid insurance policy (12-mo, needs monthly amortization)
- fixed assets + monthly depreciation
- bank fees + interest (bank side; some not booked in GL)
- owner transfers / one industry-flavored event stream (e.g. inventory purchases
  for ecommerce/wholesale, WIP-ish project billing for services)

The engine produces `LedgerState`: chart of accounts, postings (each with
`posting_id, date, account, debit, credit, memo, source_doc`), bank register
(the bank's own view), subledgers (AR/AP), budget (per P&L account, derived from
baseline activity ± seeded planning error), and document stubs.

### 4.3 Discrepancy injection (`discrepancies.py`)

Injected AFTER simulation, each recorded as a `TruthRecord` used by gold:

| id | Injection | Feeds task |
|---|---|---|
| `outstanding_check` (2–4) | vendor checks written, not cleared by bank | bank_rec |
| `deposit_in_transit` (1–2) | receipt booked in GL, lands at bank next month | bank_rec |
| `bank_fee_unrecorded` (1–2) | bank fee on statement, absent from GL | bank_rec |
| `interest_unrecorded` (0–1) | interest earned on statement, absent from GL | bank_rec |
| `gl_amount_error` (0–1) | transposition error in one GL cash posting | bank_rec |
| `missing_payroll_accrual` | last payroll run of month spans period end, not accrued | journal_entries |
| `missing_prepaid_amort` | insurance amortization not booked for close month | journal_entries + schedule |
| `missing_depreciation` (0–1) | depreciation entry for close month not booked | journal_entries |
| `misposted_expense` (0–1) | expense posted to wrong account | journal_entries |
| `variance_drivers` (2–4) | seeded one-off events (big project, price change, headcount add, ad-spend spike) that explain budget-vs-actual gaps | variance_analysis |

### 4.4 Exported files (`exporters.py`) — what the agent sees

Written under `worlds/{split}/{wid}/files/`. **Column names, filenames, date
formats, and CSV dialects vary per world** (seeded from small variant pools) —
this is the retrieval-boundness lever. Baseline set (~12–18 files):

- `bank_statement_<month>.csv` — bank's view (mangled memos; includes items GL missed)
- `general_ledger.csv` — all GL postings (with injected errors/omissions)
- `trial_balance.csv` — end of close month
- `ar_aging.csv`, `ap_aging.csv`
- `budget_fy.csv` — per-account monthly budget
- `payroll_register.csv`
- `prepaid_schedule_prior.csv` — schedule as of *prior* month end
- `fixed_asset_register.csv`
- `chart_of_accounts.csv`
- `invoices/` and `bills/` — individual `.txt` docs for a subset
- `contracts/insurance_policy.txt`, one or two industry contracts
- `notes/close_checklist.md` — the close checklist (mentions what should be booked, not amounts)
- `notes/prior_accountant_memo.md` — partial hints, some stale
- 2–3 distractor files (e.g. `marketing_plan.md`, an old quarter's TB)

**Never** write gold, meta, seeds, or truth records inside `files/`.

### 4.5 Tasks (`tasks.py`) — 4 per world

Each task: `task_id = {wid}-{category}`, a prompt written like a real
controller's ask (concise, references the company by name, states the
**deliverable JSON schema inline**), gold answer JSON, and an auto-rubric
(binary criteria derived from gold — used only for reporting, never for the
training judge).

Prompts end with: *"Submit your final answer with the `submit_answer` tool.
`answer_json` must match the schema above exactly. Put narrative in `notes`."*

## 5. Deliverable schemas + gold scoring

All amounts: positive numbers, 2dp, `0.01` tolerance unless stated. Account
references may be account **number or name** — `gold_score.py` resolves both
(exact number match, else case-insensitive name match, else difflib ratio ≥ 0.85
against the world's chart of accounts).

### 5.1 `bank_rec`

```jsonc
{
  "bank_statement_ending_balance": 12345.67,
  "gl_cash_ending_balance": 12000.00,
  "adjusted_balance": 12200.00,             // both sides after adjustments
  "reconciling_items": [
    {"kind": "outstanding_check|deposit_in_transit|bank_fee_unrecorded|interest_unrecorded|gl_error|other",
     "amount": 250.00, "date": "2026-06-28", "ref": "CHK-1041 or null",
     "description": "...", "side": "bank|book"}
  ],
  "proposed_journal_entries": [ {JE — see 5.2 line shape} ]
}
```

Score (0–100): `60 * item_F1 + 20 * balances_correct + 20 * je_F1`, where
item matching is by `(kind, amount±0.01, side)` greedy bipartite — amounts are
compared signed (gold amounts are positive) and a stated `side` must agree
with gold's (an omitted side is not penalized); `balances_correct` is 1 if all
three balances within tolerance (partial credit 1/3 each); `je_F1` per §5.2
over the gold correcting entries.

### 5.2 `journal_entries`

```jsonc
{"entries": [
  {"memo": "Accrue payroll for Jun 24–30", "date": "2026-06-30",
   "lines": [ {"account": "6000 Salaries Expense", "debit": 8250.00, "credit": 0},
              {"account": "2100 Accrued Liabilities", "debit": 0, "credit": 8250.00} ]}
]}
```

Score: for each gold entry, best-match submitted entry = the one maximizing
line-level F1 (a line matches on resolved account + side + amount±0.01); each
gold entry scores its best F1; entry-level score = mean over gold entries;
penalty: `max(0, submitted_entries - gold_entries) * 5` points off (spam guard).
Scale to 0–100.

### 5.3 `variance_analysis`

```jsonc
{"variances": [
  {"account": "6200 Advertising", "actual": 42000.00, "budget": 30000.00,
   "variance": 12000.00, "direction": "unfavorable",
   "drivers": [{"description": "June brand campaign per marketing invoice INV-...",
                "amount": 11500.00, "evidence": ["bills/AD-2026-118.txt"]}]}
], "summary": "..." }
```

Gold contains the injected `variance_drivers` (account, amount, keyword tags).
The prompt states the variance formula (`variance = actual - budget`) and
directs an as-closed basis (standard checklist entries assumed booked).
Score: `50 * driver_recall + 25 * numbers_accuracy + 25 * driver_precision`.
A submitted driver hits a gold driver if account resolves AND amount (signed)
within ±10% AND any gold keyword appears in its description (case-insensitive).
`numbers_accuracy` = share of gold-flagged accounts where actual and budget are
within tolerance AND the variance is correct: with a stated `direction`, |variance|
must match and the direction must equal gold's; with no stated direction the
signed variance must match gold's `actual - budget`.

### 5.4 `accrual_schedule` (prepaid amortization roll-forward)

```jsonc
{"schedule": [
  {"item": "Hartline GL policy 2026", "opening_balance": 9000.00,
   "additions": 0.00, "amortization": 1000.00, "closing_balance": 8000.00}
], "journal_entry": {JE shape} }
```

Score: `max(0, 70 * row_score + 30 * je_F1 - 5 * extra_rows)`; rows match by
fuzzy item name (max of difflib ratio and ≥4-char-token overlap, threshold
0.6) then all four numbers ±0.01 (each number 25% of the row); `extra_rows =
max(0, submitted_rows - gold_rows)` is a spam guard mirroring §5.2.

### 5.5 Auto-rubrics

`tasks.py` also emits `gold/{split}/{wid}/{task_id}.rubric.json`: a list of
binary criteria strings derived from gold (e.g. "Identifies outstanding check
CHK-1041 for $1,250.00"). Reporting-only.

## 6. Harness spec format (`runtime/spec.py`)

A harness is ONE markdown file, strictly this shape (validator enforces;
optimizer is told the grammar and gets up to 3 retries on validation failure):

```markdown
# HARNESS <version-label>

## ORCHESTRATOR
tools: none            <- comma list from the tool roster, or `none`
max_steps: 40
instructions: |
  <free text: how to run the workflow, what to send agents, when to submit>

## AGENTS

### <AgentName>        <- [A-Za-z][A-Za-z0-9_]{1,30}
tools: read_file, grep_files, list_files, run_python   <- subset of roster
max_steps: 25
role: <one line>
instructions: |
  <free text>
contract: |
  <free text: exactly what this agent must return to the orchestrator>

## WORKFLOW
<free text: the hops — order, branching, retries. Executed by the orchestrator.>

## AUXILIARY RULES
<free text or `none`: acceptance gates, fallbacks, recall triggers>
```

Tool roster (the ONLY tools that exist): `list_files`, `read_file`,
`grep_files`, `run_python`. The orchestrator additionally always has
`call_agent` and `submit_answer` (implicit; not listed). Parser returns
`HarnessSpec{orchestrator: AgentDef, agents: dict[str, AgentDef], workflow: str,
aux: str}` and validates: known tools only, 1–8 agents, max_steps 1–60,
non-empty instructions/contract.

## 7. Runtime (`runtime/`)

### 7.1 Tools (`tools.py`) — all jailed to the world's `files/` dir

- `list_files() -> str` — recursive listing with sizes.
- `read_file(path, start_line=1, max_lines=200) -> str` — line-numbered; refuses
  paths escaping the jail (resolve + `is_relative_to`).
- `grep_files(pattern, glob="**/*", max_matches=50) -> str` — regex over text files.
- `run_python(source, timeout_s=20) -> str` — `subprocess.run([sys.executable,
  "-I", "-c", source], cwd=files_dir, timeout, capture)` returning
  stdout+stderr (truncated to 8000 chars). World files readable from cwd. No
  network guarantee needed beyond `-I` for the hackathon.
- Orchestrator-only: `call_agent(name, message) -> str` (spawns fresh subagent
  loop, returns its final text), `submit_answer(answer_json: str, notes: str) -> str`
  (parses JSON; on parse failure returns the error to the model so it can
  retry; on success ends the task run).

Every tool result is truncated to 8000 chars with a `[truncated]` marker.

### 7.2 Orchestrator loop (`orchestrator.py`)

`run_task(spec: HarnessSpec, world_dir, task: Task, cfg, meter, trace) ->
TaskRunResult{answer: dict|None, notes, steps, usd, tokens, ended: "submitted|max_steps|budget|error"}`

- Orchestrator system prompt = fixed protocol preamble + spec.orchestrator
  .instructions + WORKFLOW + AUXILIARY RULES + agent roster table (name, role,
  contract) + the task prompt.
- Subagent system prompt = fixed preamble + its role/instructions/contract +
  world context line. The `call_agent` message is its user turn. Subagent's
  final assistant text (when it stops calling tools) is returned verbatim to the
  orchestrator as the tool result.
- Manual tool loop per the Anthropic SDK (messages + tool_use blocks); parallel
  tool calls answered in one user message; on `max_steps` the orchestrator gets
  one final forced chance: a user message "Step limit reached — submit your best
  answer now with submit_answer." Subagents at max_steps return their last text.
- Fresh context per subagent call. No shared memory beyond message contents.

### 7.3 Traces (`trace.py`)

JSONL per task-run: `{ts, actor: "orchestrator|<AgentName>", event:
"llm_call|tool_call|tool_result|submit", detail, tokens, usd}`. Plus
`summarize_trace(trace_path, max_chars=6000) -> str`: deterministic (no LLM)
digest — the sequence of agents called, tools used with arg previews, message
sizes, the final answer, token/cost totals. This summary is what judges and the
optimizer read (full traces are too big).

## 8. Judging (`judge/`)

### 8.1 Training judge (gold-blind) — the pseudo-reward

`judge_task(task_prompt, answer_json, notes, trace_summary, cfg) ->
{score: 0..10, critique: str, criteria: [{name, met: bool, comment}]}`

Prompt: senior controller reviewing a staff accountant's close workpaper.
Generic category rubric template (NOT gold-derived): completeness, internal
consistency (do the numbers tie?), evidence discipline (does the trace show the
files were actually consulted?), format compliance, plausibility checks. Judge
sees NO gold, NO other runs. `samples: 3` independent calls → score = median,
critiques concatenated (labeled). JSON output enforced by response-format
instruction + parse-retry (2 attempts), and with the OpenAI-compatible client
use JSON mode if available.

### 8.2 Pairwise comparator

`compare(task_category, prev: RunDigest, cur: RunDigest, cfg) ->
{winner: "prev|cur|tie", reasons: str, transferable_advice: str}`

RunDigest = {task_prompt, answer_json, notes, trace_summary, judge_score}.
Compared across iterations on the **same category** (worlds differ — the prompt
says so and directs attention to process quality, information flow between
agents, and wasted work, not world-specific facts).

## 9. RHI optimizer (`rhi/optimizer.py`)

`propose_revision(history: list[IterationRecord], current_spec_text, cfg) ->
(new_spec_text, rationale)`

The meta-prompt contains, in order: the RHI framing (you improve ROLES /
INSTRUCTIONS / CONTRACTS / WORKFLOW-hops / AUXILIARY RULES; prioritize contracts
and information flow; you may add/remove/merge agents; you may instruct agents
to use `run_python` for arithmetic and searching); the spec grammar of §6; the
full current spec; a compact revision history (per past iteration: version,
mean judge score, per-category means, regression flag, top critique themes,
pairwise outcomes + transferable advice, and the full spec text of that
version); this iteration's evidence (pairwise vs the previous iteration,
per-task judge critiques in full, trace summaries up to 4, truncated); the
required output shape (§9.1 reflection BEFORE the fence, then the spec fence,
then `RATIONALE:`); hard rules: never reference specific
worlds/companies/amounts (fresh worlds every iteration); keep ≤ 8 agents;
exactly one fenced block in the reply (the spec).

Validation: parse with `spec.py`; on failure, feed the error back, retry ≤ 3;
if still failing, keep previous spec (log the event, iteration becomes a no-op
revision).

The optimizer NEVER sees: gold, rubrics, held-out anything, gold scores.

### 9.1 Recursion hardening (required)

- **Genealogy + revert rights.** `history` carries every prior revision's full
  spec text (or a diff vs its parent when large), its train-slice mean judge
  score, and per-category means. The meta-prompt states explicitly: *you may
  base the next revision on ANY prior version — reverting or branching is a
  legitimate move, especially after a regression.* The returned spec's
  `# HARNESS v{i+1}` label stays sequential regardless of parentage; rationale
  must name the parent version.
- **Structured reflection.** The meta-prompt requires, before the spec fence:
  (1) top 2–4 recurring failure modes across this slice with quoted evidence
  from critiques/trace summaries; (2) each mapped to a component (ROLES /
  INSTRUCTIONS / CONTRACTS / WORKFLOW / AUXILIARY); (3) the 1–3 changes chosen
  and the evidence each answers. This text is captured into the rationale file.
- **Bounded edits.** Instruct 1–3 targeted changes per revision; preserve
  working structure; contracts and information flow are the priority axis.
- **Regression flag.** `experiment.py` computes `delta_vs_prev_slice` per
  iteration and sets `regressed: true` on the latest history item when the mean
  judge score dropped. The prompt then requires an explicit change-vs-noise
  diagnosis and consideration of a revert.

## 10. Experiment driver (`experiment.py`)

```
python experiment.py run            # full loop per config
python experiment.py heldout --harness harness/v0.md --tag v0   # eval only
python experiment.py status        # budget + progress from results/
```

Loop: iteration i uses worlds `train[w_{3i+1}..w_{3i+3}]`. For each task:
`run_task` → training judge → append to `results/runs.jsonl`:

```jsonc
{"run_id", "phase": "train|heldout", "iteration", "harness_version",
 "world_id", "task_id", "category", "answer", "ended",
 "judge_score", "judge_critique",
 "gold_score": null,            // filled only for heldout phase
 "usd", "tokens": {"in","out","cache_r","cache_w"}, "steps", "wallclock_s"}
```

After the slice: pairwise comparisons vs iteration i-1 (same categories), then
`propose_revision` → `harness/revisions/v{i+1}.md`. At checkpoints in
`heldout_checkpoints`: run current spec on all 6 held-out worlds × 4 tasks,
score with `gold_score.py` AND the training judge (divergence data), append to
`results/checkpoints.jsonl`. Budget check before every task; on cap: finish
writing records, print status, exit 0. Every LLM call wrapped with retry
(SDK default retries + one outer retry); a task that errors terminally records
`ended: "error"`, judge_score 0, and the loop continues.

Resumability: `results/state.json` {iteration, completed run_ids}; rerunning
`run` skips completed work. Budget across restarts: every record carries a
`session` id and each driver session persists its FULL CostMeter total
(actor + judge + pairwise + optimizer) to `results/spend.json` after every
task; `_prior_spend` sums other sessions' totals (plus per-record usd for
legacy session-less records), and the driver passes `cap - prior` into
run_task so mid-task checks are equally tight.

## 11. Gates (all $0, run before any API spend)

- **Gate 0** `scripts/smoke_generator.py`: build 2 worlds twice → byte-identical
  trees (hash); schemas parse; every task has gold; no gold/meta inside `files/`.
- **Gate 1** (same script): integrity — GL balances; TB ties to GL; bank
  statement ending balance = bank register math; every TruthRecord observable in
  exports (e.g. unrecorded fee IS on the statement and NOT in GL).
- **Gate 2** `scripts/oracle_check.py`: build gold-derived answers (identity
  transform of gold) → `gold_score.py` gives ≥ 99 for every task; then perturbed
  answers (drop one item / wrong amount) score strictly lower.

## 12. v0 harness (`harness/v0.md`)

Canonical-naive, per the grammar: Orchestrator (tools: none) + three agents —
`FileScout` (tools: list_files, read_file; contract returns "a list of the
relevant filenames"), `Analyst` (tools: read_file, grep_files; does the whole
task, contract returns "the draft answer JSON"), `Reviewer` (tools: read_file;
contract returns "APPROVED or a list of concerns"). Workflow: FileScout →
Analyst → Reviewer → submit. Auxiliary rules: none. Deliberate deficiencies
(the improvement runway): no `run_python` anywhere, FileScout's contract drops
file *contents* and column layouts, Analyst never told to verify sums, Reviewer
can't see the source data it would need, no fallback when the Analyst's JSON is
malformed, single-pass with no cross-checks.

## 13. Env vars

- `ACTOR_API_KEY`, `ACTOR_BASE_URL`, `ACTOR_MODEL` — the 30–100B harness model
- `OPTIMIZER_API_KEY`, `OPTIMIZER_BASE_URL`, `OPTIMIZER_MODEL` — frontier optimizer (MiniMax-class)
- `JUDGE_LLM_API_KEY` (or `JUDGE_API_KEY`), `JUDGE_BASE_URL`, `JUDGE_MODEL` — DeepSeek v4.1 Pro
- `ANTHROPIC_API_KEY` — only if a role is configured with provider `anthropic`
- `FINFORGE_BUDGET_CAP` — overrides config cap

## 14. Pinned Python interfaces (all modules must match exactly)

```python
# runtime/llm.py
@dataclass
class LLMResponse:
    text: str                      # concatenated assistant text ("" if none)
    tool_calls: list[dict]         # [{"id": str, "name": str, "arguments": dict}]
    usage: dict                    # {"input": int, "output": int}
    raw: object

class LLMClient:
    @classmethod
    def for_role(cls, role: str, cfg: dict) -> "LLMClient":
        """role in {"actor","optimizer","training_judge","pairwise_judge"};
        cfg is the full experiment.json dict. Resolves env per DESIGN §3."""
    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None, json_mode: bool = False) -> LLMResponse:
        """messages/tools in OpenAI wire format. Tool results are
        {"role":"tool","tool_call_id":...,"content":...}. Assistant turns with
        tool calls are {"role":"assistant","content":...,"tool_calls":[...]}.
        The anthropic adapter converts internally. Retries transient errors."""

METER: CostMeter   # module-level singleton
class CostMeter:
    def add(self, role: str, model: str, usage: dict) -> None: ...
    @property
    def usd_total(self) -> float: ...
    def by_role(self) -> dict: ...
    def snapshot(self) -> dict: ...          # json-safe
    def over_cap(self, cap_usd: float) -> bool: ...

# runtime/spec.py
@dataclass
class AgentDef:
    name: str; tools: list[str]; max_steps: int
    role: str; instructions: str; contract: str
@dataclass
class HarnessSpec:
    version_label: str; orchestrator: AgentDef
    agents: dict[str, AgentDef]; workflow: str; aux: str; source_text: str
def parse_harness(text: str) -> HarnessSpec        # raises SpecError(msg)
def validate_harness(spec: HarnessSpec) -> None    # raises SpecError(msg)

# runtime/orchestrator.py
@dataclass
class TaskRunResult:
    answer: dict | None; notes: str; ended: str    # submitted|max_steps|budget|error
    steps: int; usd: float; tokens: dict; wallclock_s: float; trace_path: str
def run_task(spec: HarnessSpec, world_files_dir: str, task_prompt: str,
             task_id: str, cfg: dict, results_dir: str) -> TaskRunResult

# runtime/trace.py
class TraceWriter:                                  # used inside run_task
    def __init__(self, path: str): ...
    def event(self, actor: str, event: str, detail: dict,
              tokens: dict | None = None, usd: float | None = None) -> None
def summarize_trace(trace_path: str, max_chars: int = 6000) -> str   # no LLM

# generator/build.py  (CLI: python -m generator.build [--worlds-dir worlds] [--gold-dir gold])
def build_all(worlds_dir: str = "worlds", gold_dir: str = "gold",
              config_path: str = "config/experiment.json") -> None
# Emits worlds/{split}/{wid}/files/** and gold/{split}/{wid}/{task_id}.json,
# {task_id}.rubric.json, meta.json, and tasks.json (list of
# {"task_id","category","prompt"}) at gold/{split}/{wid}/tasks.json.

# scoring/gold_score.py
def score_task(category: str, answer: dict | None, gold: dict,
               chart_of_accounts: list[dict], frozen: dict) -> dict
# returns {"score": float 0..100, "components": {...}, "details": str}
# chart_of_accounts: [{"number": "6000", "name": "Salaries Expense"}, ...]
# (read from gold meta.json["chart_of_accounts"]). answer None -> score 0.

# judge/training_judge.py
def judge_task(task_prompt: str, answer: dict | None, notes: str,
               trace_summary: str, cfg: dict) -> dict
# {"score": float 0..10, "critique": str, "criteria": [{"name","met","comment"}]}

# judge/pairwise.py
def compare(category: str, prev_digest: dict, cur_digest: dict, cfg: dict) -> dict
# digests: {"task_prompt","answer","notes","trace_summary","judge_score"}
# returns {"winner": "prev|cur|tie", "reasons": str, "transferable_advice": str}

# rhi/optimizer.py
def propose_revision(history: list[dict], current_spec_text: str,
                     cfg: dict) -> tuple[str, str]   # (new_spec_text, rationale)
# history item: {"iteration","version","mean_judge","per_category",
#   "critiques":[str], "pairwise":[dict], "trace_summaries":[str],
#   "spec_text": str,               # full spec of that version (genealogy, 9.1)
#   "delta_vs_prev": float|None, "regressed": bool}
# Guarantees the returned spec parses+validates (retries internally; falls back
# to current_spec_text on repeated failure and says so in rationale). The
# returned spec's `# HARNESS v{i+1}` label is enforced sequentially.
```

Shared conventions: pure stdlib + `openai` + `anthropic` only. Every module
importable on Python 3.12. No module reads env at import time (only inside
functions). All randomness via `random.Random(seed)` passed explicitly. Amounts
`Decimal` inside the generator, plain floats (2dp) in all JSON.

## 15. Dataset strengthening (LLM enrichment + QA)

Built AFTER the programmatic generator passes Gates 0–2. Two modules:

- `generator/enrich.py` — LLM "worldsmith" pass (default: `DATAGEN_MODEL` via
  `DATAGEN_API_KEY`/`DATAGEN_BASE_URL`, currently `gpt-5.6-sol` on OpenAI —
  the Anthropic key is out of credit; `--model` overridable. gpt-5.x models:
  use `max_completion_tokens`, leave temperature at default). For each world it requests
  **narrative fields only** as strict JSON (company backstory, per-document
  prose descriptions, memo phrasing styles, checklist/memo wording, distractor
  file bodies, industry-flavored file naming). The deterministic renderer then
  re-emits the world files inserting every number/date/id itself — the LLM
  never writes an amount. Post-validation asserts all task-relevant tokens
  (amounts, refs, balances) still appear exactly where gold expects them.
  Results cached at `gold/{split}/{wid}/enrichment.json`; enriched worlds are
  frozen artifacts (tree hash recorded in meta.json). Re-running with the cache
  present is a no-op (determinism preserved via cache, not via re-sampling).
- `generator/qa.py` — red-team QA pass (default `QA_MODEL`, currently
  `gpt-5.6-sol` on OpenAI). Per world:
  given ONLY the files/ tree and the task prompts (no gold), it must (a) solve a
  spot-check subset itself, (b) flag contradictions/ambiguities/unsolvable
  tasks, (c) then, gold revealed, verify gold follows from the files. Output:
  `gold/{split}/{wid}/qa_report.json` with pass|fail + issues. Failing worlds
  are rebuilt with a bumped sub-seed until pass (max 3 attempts, else flagged
  for manual review). QA-pass rate is reported in the writeup.

Gate 0 applies to the pre-enrichment skeleton; post-enrichment the frozen tree
hash in meta.json is the determinism artifact. Enrichment/QA spend is tracked
separately from the experiment's actor budget cap.
