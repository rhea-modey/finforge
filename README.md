# FinForge

**A finance agent that improves itself — with no answer key.**

FinForge is a self-improving multi-agent harness for month-end close work
(bank reconciliation, journal entries, variance analysis, prepaid schedules),
plus the contamination-free synthetic benchmark used to measure it against
ground truth it never sees.

Built as a team project for the Office-of-the-CFO hackathon track at HackMIT 2026.

## My contributions

I co-built the recursive improvement and evaluation workflow used for the HackMIT submission. My work included the synthetic bank-reconciliation benchmark, verifier/evaluation harness, and the rollout/tool-call signals used to iteratively improve the system prompt and toolset across runs.

## Headline results

| | |
|---|---|
| **Bank reconciliation, best evolved harness** | **60.9 → 91.3 / 100** on 6 never-seen companies (+30.4 points) — the loop *self-invented* a Calculator agent that recomputes every tie-out and rejects drafts off by more than $0.01 |
| **The research finding** | Self-improvement is bounded by the *verification quality of its reward*: a judge that only reads the work correlates with truth at ρ = +0.24; the same gold-blind judge armed with tools to redo the math reaches **ρ = +0.70** |
| **The controlled experiment** | Run B reruns the identical protocol with only the verifying judge changed — live at submission time |

The repository reports the exact held-out benchmark score (60.9 → 91.3). Rounded summaries should describe this as a **score improvement**, not classification accuracy.

## How it works

The system improves a **playbook**: one markdown document describing a small AI
team (who reads the files, who does the math, who double-checks, and the
contract each hands the next). Each round:

1. **The Team** (actor: `gpt-oss-120b`) runs the playbook on 3 brand-new
   companies' books — 12 tasks, worlds never reused.
2. **The Grader** (judge: `deepseek-v4.1`) scores the work *gold-blind* — it
   sees only the files the team saw. In Experiment B it must recompute every
   checkable claim with tools (`read_file`, `grep_files`, `run_python`) before
   scoring.
3. **The Coach** (optimizer: `MiniMax-M3`) reads the grader's critiques and
   makes bounded, surgical edits to the playbook — with full revision genealogy
   and the right to revert.
4. **The Tryout** (acceptance gate): new playbook vs incumbent on the same
   fresh books, judged blind; the winner is deployed.
5. **The sealed answer key**: because worlds are generated, true answers exist —
   locked away, used only to measure progress on 6 held-out companies
   (2 industries never seen in training). Neither the team, grader, nor coach
   ever sees gold. That quarantine is what makes "improvement" falsifiable.

## The benchmark

- 36 generated company worlds (30 train / 6 held-out), ~30 messy plain-text
  files each (GL detail, bank statements, bills, payroll registers, budgets),
  3 simulated months ending at the June 2026 close; 144 tasks total.
- Gold answers are correct by construction: a seeded double-entry engine
  (integer cents, every posting balanced) injects each discrepancy and records
  it. Oracle self-consistency: 144/144.
- Injections are **conditional** (e.g. the missing payroll accrual exists in
  only 29/36 worlds), so memorized procedures fail — the books must be checked.
- Zero text overlap with APEX-Accounting by automated audit
  (`reports/apex_similarity_audit.md`); certified after 5 adversarial QA
  sweeps (`DATASET_V1_CERTIFIED.json`).

## Repository layout

```
DESIGN.md            binding design/interface contract for the whole system
experiment.py        the training loop: iterations, acceptance gate, checkpoints
generator/           world generator: ledger engine, injections, exporters, QA
harness/             v0 playbook + every revision with the optimizer's rationale
judge/               training judges: reading (Run A) and verifying (Run B)
rhi/                 the optimizer (playbook rewriter)
runtime/             spec interpreter, agent tool loop, LLM clients, tracing
scoring/             deterministic gold scorer + curve builder
worlds/ gold/        the frozen benchmark (files the agents see / sealed answers)
config/              experiment configs incl. frozen scoring weights
results/ results_b/  full run records for Experiments A and B (JSONL, auditable)
reports/             paper draft, similarity audit, live chart, judge deck
scripts/             env wiring, chart/alignment tooling, Run B launcher
```

## Reproducing

```bash
pip install -r requirements.txt
# provide keys in ../.env or the environment:
#   FIREWORKS_API_KEY (actor + judge), MINIMAX_API_KEY (optimizer)
source scripts/env.sh

python3 -m scripts.smoke_generator     # dataset gates 0-1
python3 scripts/oracle_check.py        # gate 2: oracle 144/144
python3 experiment.py --config config/experiment.json run    # Experiment A
python3 scripts/run_b.py                                     # Experiment B
python3 -m scoring.curve               # held-out curve table
python3 scripts/judge_alignment.py     # judge-vs-gold correlation (A vs B)
```

Everything is seeded and stamped: master seed `20260919`, frozen scorer
weights committed before the first scored run, per-world tree hashes in the
certification stamp. Run records dedupe by `run_id`; partial checkpoints are
never quoted (streaming completes easy tasks first).

## Honesty notes

The full four-category held-out curve for Run A oscillates — that oscillation,
traced to reward misalignment, *is* the research finding, and motivated the
verifying judge. Same-spec re-runs can differ by ~10 points from sampling
alone (measured); Run B doubles rollouts to average this out. See
`reports/paper.md` §5 for the complete limitations discussion.
