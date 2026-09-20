#!/usr/bin/env python3
"""FinForge experiment driver (DESIGN section 10).

    python experiment.py run                                    # full loop
    python experiment.py heldout --harness harness/v0.md --tag v0
    python experiment.py status

The gold-blind boundary: gold answers and rubrics are NEVER passed to the
training judge, the pairwise comparator, or the optimizer. Gold is read only
to fill the gold_score field on heldout checkpoint records.

Imports of runtime/judge/rhi modules are deferred into the subcommands so
`status` works even before those modules exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
STATE_PATH = RESULTS_DIR / "state.json"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"
CHECKPOINTS_PATH = RESULTS_DIR / "checkpoints.jsonl"
DIGESTS_PATH = RESULTS_DIR / "digests.jsonl"       # per-run pairwise digests
ITERATIONS_PATH = RESULTS_DIR / "iterations.jsonl"  # optimizer history
SPEND_PATH = RESULTS_DIR / "spend.json"             # per-session METER totals

# One id per driver process: per-record `usd` covers only the actor's task
# spend, so judge/pairwise/optimizer spend is persisted per session (see
# _record_session_spend) and counted by _prior_spend across restarts.
SESSION_ID = f"{int(time.time() * 1000)}-{os.getpid()}"
HARNESS_V0 = ROOT / "harness" / "v0.md"
REVISIONS_DIR = ROOT / "harness" / "revisions"
GOLD_DIR = ROOT / "gold"
WORLDS_DIR = ROOT / "worlds"
CONFIG_PATH = ROOT / "config" / "experiment.json"
FROZEN_PATH = ROOT / "config" / "scoring.frozen.json"

CATEGORIES = ("bank_rec", "journal_entries", "variance_analysis", "accrual_schedule")


# ------------------------------------------------------------------ helpers

def _log(msg: str) -> None:
    print(f"[finforge] {msg}", flush=True)


def _load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _append_jsonl(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()


def _load_state() -> tuple[dict, set]:
    data = _load_json(STATE_PATH, {}) or {}
    state = {"iteration": int(data.get("iteration") or 0),
             "harness_version": str(data.get("harness_version") or "v0"),
             "accepted_version": data.get("accepted_version")}
    completed = set(data.get("completed") or [])
    # The jsonl records are the source of truth: a stale/lost state.json must
    # never cause a completed run to be re-executed (duplicate records).
    for rec in _read_jsonl(RUNS_PATH) + _read_jsonl(CHECKPOINTS_PATH):
        rid = rec.get("run_id")
        if rid:
            completed.add(rid)
    return state, completed


def _save_state(state: dict, completed: set) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"iteration": int(state.get("iteration", 0)),
            "harness_version": state.get("harness_version", "v0"),
            "accepted_version": state.get("accepted_version"),
            "completed": sorted(completed)}
    tmp = STATE_PATH.with_name("state.json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def _load_config(path=None) -> dict:
    cfg = _load_json(path or CONFIG_PATH)
    if not isinstance(cfg, dict):
        _log(f"ERROR: cannot read config at {path or CONFIG_PATH}")
        sys.exit(1)
    return cfg


def _budget_cap(cfg: dict) -> float:
    env = os.environ.get("FINFORGE_BUDGET_CAP")
    if env:
        try:
            return float(env)
        except ValueError:
            _log(f"WARNING: bad FINFORGE_BUDGET_CAP={env!r}; using config cap")
    return float(cfg.get("budget_usd_cap", 60.0))


def _record_session_spend() -> None:
    """Persist this session's FULL METER total (actor + judge + pairwise +
    optimizer) so a later session's budget check cannot undercount prior
    spend. Called after every task record and at halt/exit."""
    try:
        from runtime.llm import METER
        total = METER.usd_total
    except Exception:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_json(SPEND_PATH, {}) or {}
    if not isinstance(data, dict):
        data = {}
    data[SESSION_ID] = round(float(total), 6)
    tmp = SPEND_PATH.with_name("spend.json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(SPEND_PATH)


def _prior_spend() -> float:
    """Spend by PRIOR sessions: their persisted METER totals (which include
    judge/pairwise/optimizer calls), plus per-record usd for legacy records
    that predate session tracking (no 'session' field)."""
    data = _load_json(SPEND_PATH, {}) or {}
    total = 0.0
    if isinstance(data, dict):
        for sid, usd in data.items():
            if sid == SESSION_ID:
                continue
            try:
                total += float(usd or 0.0)
            except (TypeError, ValueError):
                pass
    for rec in _read_jsonl(RUNS_PATH) + _read_jsonl(CHECKPOINTS_PATH):
        if rec.get("session"):
            continue  # covered by its session's spend.json entry
        try:
            total += float(rec.get("usd") or 0.0)
        except (TypeError, ValueError):
            pass
    return total


def _norm_tokens(t) -> dict:
    t = t if isinstance(t, dict) else {}

    def g(*keys):
        for k in keys:
            v = t.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return 0

    return {"in": g("in", "input", "input_tokens"),
            "out": g("out", "output", "output_tokens"),
            "cache_r": g("cache_r", "cache_read", "cache_read_input_tokens"),
            "cache_w": g("cache_w", "cache_write", "cache_creation_input_tokens")}


def _print_summary() -> None:
    try:
        from scoring.report import aggregate, render_table
        print(render_table(aggregate(str(RESULTS_DIR), str(GOLD_DIR), write=True)))
    except Exception as e:
        _log(f"summary aggregation failed: {e!r}")


def _halt_budget(state: dict, completed: set, cap: float, prior: float) -> None:
    """Budget cap reached: persist state + session spend, print summary, exit 0."""
    _save_state(state, completed)
    _record_session_spend()
    session = 0.0
    try:
        from runtime.llm import METER
        session = METER.usd_total
    except Exception:
        pass
    _log(f"budget cap reached: cap=${cap:.2f} "
         f"(prior records ${prior:.2f} + this session ${session:.2f}); halting")
    _print_summary()
    sys.exit(0)


def _make_budget_check(cap: float, prior: float):
    """True -> halt. Effective cap for this process = cap - spend recorded by
    previous processes (METER only sees the current process)."""
    effective = cap - prior

    def over() -> bool:
        if effective <= 0:
            return True
        from runtime.llm import METER
        return METER.over_cap(effective)

    return over


# ------------------------------------------------------- worlds / tasks / spec

def _iter_worlds(i: int, cfg: dict) -> list[str]:
    wpi = int(cfg.get("worlds_per_iteration", 3))
    return [f"w{i * wpi + k + 1:02d}" for k in range(wpi)]


def _heldout_worlds() -> list[str]:
    d = GOLD_DIR / "heldout"
    if d.is_dir():
        ws = sorted(p.name for p in d.iterdir() if (p / "tasks.json").exists())
        if ws:
            return ws
    return [f"h{n:02d}" for n in range(1, 7)]


def _read_tasks(split: str, wid: str) -> list[dict]:
    data = _load_json(GOLD_DIR / split / wid / "tasks.json")
    out = []
    if isinstance(data, list):
        for t in data:
            if not (isinstance(t, dict) and t.get("task_id") and t.get("prompt")):
                continue
            tid = str(t["task_id"])
            cat = t.get("category") or (tid.split("-", 1)[1] if "-" in tid else "")
            out.append({"task_id": tid, "category": str(cat), "prompt": str(t["prompt"])})
    return out


def _spec_path(version: str) -> Path:
    return HARNESS_V0 if version == "v0" else REVISIONS_DIR / f"{version}.md"


def _load_spec(version: str):
    from runtime.spec import parse_harness, validate_harness
    path = _spec_path(version)
    if not path.exists():
        raise FileNotFoundError(f"harness spec not found: {path}")
    text = path.read_text(encoding="utf-8")
    spec = parse_harness(text)
    validate_harness(spec)
    return spec, text


# ------------------------------------------------------------- task execution

def _execute_task(spec, files_dir: Path, task: dict, cfg: dict) -> tuple[dict, dict]:
    """run_task + training judge, with task-level fault tolerance.

    Returns (core, digest):
      core   = answer/ended/judge fields + usage, ready to merge into a record
      digest = {task_prompt, answer, notes, trace_summary, judge_score}
    Any exception in run_task or judge -> ended "error", judge_score 0.
    Gold-blind: only prompt/answer/notes/trace summary reach the judge.
    """
    from runtime.orchestrator import run_task
    from runtime.trace import summarize_trace
    from judge.training_judge import judge_task

    t0 = time.time()
    answer = None
    notes = ""
    ended = "error"
    steps = 0
    usd = 0.0
    tokens: dict = {}
    wallclock = 0.0
    trace_path = ""
    err = None

    try:
        res = run_task(spec, str(files_dir), task["prompt"], task["task_id"],
                       cfg, str(RESULTS_DIR))
        answer = res.answer if isinstance(res.answer, dict) else None
        notes = str(res.notes or "")
        ended = str(res.ended or "error")
        steps = int(res.steps or 0)
        usd = float(res.usd or 0.0)
        tokens = dict(res.tokens or {})
        wallclock = float(res.wallclock_s or 0.0)
        trace_path = str(res.trace_path or "")
    except Exception as e:
        err = f"run_task failed: {e!r}"
        wallclock = time.time() - t0

    trace_summary = ""
    if trace_path:
        try:
            trace_summary = summarize_trace(trace_path)
        except Exception as e:
            trace_summary = f"[trace summarization failed: {e!r}]"

    judge_score = 0.0
    critique = ""
    if err is None:
        try:
            j = judge_task(task["prompt"], answer, notes, trace_summary,
                           {**cfg, "_files_dir": str(files_dir)}) or {}
            judge_score = max(0.0, min(10.0, float(j.get("score") or 0.0)))
            critique = str(j.get("critique") or "")
        except Exception as e:
            err = f"judge failed: {e!r}"

    if err is not None:
        ended = "error"
        judge_score = 0.0
        critique = (critique + "\n" if critique else "") + err
        _log(f"task {task['task_id']}: {err}")

    core = {"answer": answer, "notes": notes, "ended": ended,
            "judge_score": round(judge_score, 3), "judge_critique": critique,
            "usd": round(usd, 6), "tokens": _norm_tokens(tokens),
            "steps": steps, "wallclock_s": round(wallclock, 2)}
    digest = {"task_prompt": task["prompt"], "answer": answer, "notes": notes,
              "trace_summary": trace_summary,
              "judge_score": core["judge_score"]}
    return core, digest


def _execute_stream(items: list, cfg: dict, over_budget=None):
    """P4b: streaming bounded-concurrency executor. items = [(files_dir, task,
    spec)]. Yields (item_index, core, digest) AS TASKS COMPLETE — no batch
    barrier, so a slow straggler never idles the other workers. Submission
    stops when over_budget() turns true (in-flight tasks drain). All file/
    JSONL writes happen in the caller's thread; METER is lock-protected."""
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

    def _fail(idx, e):
        core = {"answer": None, "notes": "", "ended": "error",
                "judge_score": 0.0,
                "judge_critique": f"executor failure: {e!r}",
                "usd": 0.0, "tokens": _norm_tokens({}), "steps": 0,
                "wallclock_s": 0.0}
        digest = {"task_prompt": items[idx][1].get("prompt", ""),
                  "answer": None, "notes": "", "trace_summary": "",
                  "judge_score": 0.0}
        return core, digest

    workers = max(1, int(cfg.get("parallel_workers", 6)))
    if len(items) == 1 or workers == 1:
        for idx, (fd, task, spec) in enumerate(items):
            if over_budget is not None and over_budget() and idx > 0:
                return
            try:
                core, digest = _execute_task(spec, fd, task, cfg)
            except Exception as e:
                core, digest = _fail(idx, e)
            yield idx, core, digest
        return

    queue = list(enumerate(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = {}
        while queue and len(inflight) < workers:
            idx, (fd, task, spec) = queue.pop(0)
            inflight[ex.submit(_execute_task, spec, fd, task, cfg)] = idx
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                idx = inflight.pop(fut)
                try:
                    core, digest = fut.result()
                except Exception as e:
                    core, digest = _fail(idx, e)
                yield idx, core, digest
            while queue and len(inflight) < workers:
                if over_budget is not None and over_budget():
                    queue.clear()
                    break
                idx, (fd, task, spec) = queue.pop(0)
                inflight[ex.submit(_execute_task, spec, fd, task, cfg)] = idx


# ------------------------------------------------------------ heldout eval

def _gold_score_for(wid: str, task: dict, answer, frozen: dict):
    """Gold scoring for a heldout record. Returns (score|None, components, details)."""
    from scoring.gold_score import score_task
    gold = _load_json(GOLD_DIR / "heldout" / wid / f"{task['task_id']}.json")
    if not isinstance(gold, dict):
        return None, {}, f"gold missing/unreadable for {task['task_id']}"
    meta = _load_json(GOLD_DIR / "heldout" / wid / "meta.json", {}) or {}
    chart = meta.get("chart_of_accounts") or []
    res = score_task(task["category"], answer, gold, chart, frozen)
    return res.get("score", 0.0), res.get("components", {}), res.get("details", "")


def _run_checkpoint(checkpoint, version: str, spec, cfg: dict, frozen: dict,
                    state: dict, completed: set, over_budget, cap: float,
                    prior: float, tag: str | None = None) -> None:
    """Evaluate `spec` on all heldout worlds x tasks with gold + training judge.
    Idempotent: run_ids already in `completed` are skipped."""
    tag = tag or (f"c{checkpoint}" if checkpoint is not None else "adhoc")
    worlds = _heldout_worlds()
    _log(f"heldout eval tag={tag} version={version} worlds={worlds}")
    pending = []   # (wid, task, run_id, files_dir)
    for wid in worlds:
        files_dir = WORLDS_DIR / "heldout" / wid / "files"
        tasks = _read_tasks("heldout", wid)
        if not tasks:
            _log(f"WARNING: no tasks for heldout world {wid} "
                 f"(missing {GOLD_DIR / 'heldout' / wid / 'tasks.json'}?)")
            continue
        for task in tasks:
            # heldout_repeats > 1 tightens the checkpoint estimate: each task
            # is rolled out R times and readers average across the @rN run_ids
            for rep in range(max(1, int(cfg.get("heldout_repeats", 1)))):
                run_id = (f"heldout-{tag}-{task['task_id']}"
                          + (f"@r{rep}" if rep else ""))
                if run_id not in completed:
                    pending.append((wid, task, run_id, files_dir))
    if pending and over_budget():
        _halt_budget(state, completed, cap, prior)
    stream_items = [(fd, t, spec) for (_w, t, _r, fd) in pending]
    for idx, core, _digest in _execute_stream(stream_items, cfg, over_budget):
            wid, task, run_id, _fd = pending[idx]
            gs, gcomp, gdet = _gold_score_for(wid, task, core["answer"], frozen)
            rec = {"run_id": run_id, "phase": "heldout",
                   "iteration": checkpoint, "harness_version": version,
                   "world_id": wid, "task_id": task["task_id"],
                   "category": task["category"], "answer": core["answer"],
                   "ended": core["ended"], "judge_score": core["judge_score"],
                   "judge_critique": core["judge_critique"],
                   "gold_score": gs, "gold_components": gcomp,
                   "gold_details": gdet,
                   "usd": core["usd"], "tokens": core["tokens"],
                   "steps": core["steps"], "wallclock_s": core["wallclock_s"],
                   "checkpoint": checkpoint, "tag": tag,
                   "session": SESSION_ID}
            _append_jsonl(CHECKPOINTS_PATH, rec)
            if core["ended"] != "error":
                # infra-error tasks are re-attempted on resume; readers dedupe
                # by run_id keeping the LAST record
                completed.add(run_id)
            _save_state(state, completed)
            _record_session_spend()
            gs_txt = "-" if gs is None else f"{gs:.1f}"
            _log(f"  {run_id}: ended={core['ended']} "
                 f"gold={gs_txt} judge={core['judge_score']:.1f} "
                 f"usd={core['usd']:.4f}")
    if pending and over_budget():
        _halt_budget(state, completed, cap, prior)


# ------------------------------------------------- iteration post-processing

def _clean_digest(d: dict) -> dict:
    return {"task_prompt": d.get("task_prompt", ""),
            "answer": d.get("answer"),
            "notes": d.get("notes", ""),
            "trace_summary": d.get("trace_summary", ""),
            "judge_score": d.get("judge_score", 0.0)}


def _finalize_iteration(i: int, version: str, cfg: dict,
                        gate_info: dict | None = None) -> dict:
    """Build the iteration record (means, critiques, median digests, pairwise
    vs iteration i-1) and append to iterations.jsonl. Idempotent. Only the
    CANDIDATE spec's records feed means/critiques; incumbent A/B records
    (run_id suffix @inc) inform only the acceptance-gate note."""
    existing = {r.get("iteration"): r for r in _read_jsonl(ITERATIONS_PATH)}
    if i in existing:
        return existing[i]

    recs = [r for r in _read_jsonl(RUNS_PATH)
            if r.get("phase") == "train" and r.get("iteration") == i
            and r.get("harness_version", version) == version]
    digests_all = [d for d in _read_jsonl(DIGESTS_PATH)
                   if d.get("iteration") == i
                   and d.get("harness_version", version) == version]

    scores = [float(r.get("judge_score") or 0.0) for r in recs]
    mean_judge = (sum(scores) / len(scores)) if scores else 0.0
    per_category = {}
    for c in CATEGORIES:
        cs = [float(r.get("judge_score") or 0.0) for r in recs
              if r.get("category") == c]
        if cs:
            per_category[c] = round(sum(cs) / len(cs), 3)

    # median-judge-score digest per category (deterministic tie-break)
    cur_digests = {}
    for c in CATEGORIES:
        ds = sorted((d for d in digests_all if d.get("category") == c),
                    key=lambda d: (float(d.get("judge_score") or 0.0),
                                   str(d.get("task_id") or d.get("run_id") or "")))
        if ds:
            cur_digests[c] = _clean_digest(ds[len(ds) // 2])

    # pairwise vs previous iteration, same categories
    pairwise = []
    prev = existing.get(i - 1)
    if prev:
        try:
            from judge.pairwise import compare
        except Exception as e:
            compare = None
            _log(f"pairwise unavailable: {e!r}")
        if compare is not None:
            for c in CATEGORIES:
                pd = (prev.get("digests") or {}).get(c)
                cd = cur_digests.get(c)
                if not (pd and cd):
                    continue
                try:
                    out = compare(c, pd, cd, cfg) or {}
                    pairwise.append({"category": c,
                                     "winner": out.get("winner", "tie"),
                                     "reasons": str(out.get("reasons") or ""),
                                     "transferable_advice":
                                         str(out.get("transferable_advice") or "")})
                except Exception as e:
                    _log(f"pairwise compare failed for {c}: {e!r}")

    critiques = [str(r.get("judge_critique") or "")[:2000]
                 for r in recs if r.get("judge_critique")]
    trace_summaries = [str(cur_digests[c].get("trace_summary") or "")[:4000]
                       for c in CATEGORIES if c in cur_digests][:4]

    # DESIGN 9.1 regression flag: delta of the mean judge score vs the
    # previous iteration's slice; regressed when it dropped.
    delta_vs_prev = None
    regressed = False
    prev = existing.get(i - 1)
    if isinstance(prev, dict):
        try:
            prev_mean = float(prev.get("mean_judge") or 0.0)
            delta_vs_prev = round(mean_judge - prev_mean, 3)
            regressed = mean_judge < prev_mean
        except (TypeError, ValueError):
            pass

    if gate_info:
        outcome = ("ADOPTED — it will be the base for the next revision"
                   if gate_info["adopted"] == gate_info["candidate"] else
                   "REJECTED — the incumbent stays deployed; diagnose why this "
                   "proposal underperformed head-to-head and either repair it "
                   "or take a different approach from the incumbent base")
        critiques = [(f"[ACCEPTANCE GATE] candidate {gate_info['candidate']} "
                      f"mean judge {gate_info['candidate_mean']} vs incumbent "
                      f"{gate_info['incumbent']} {gate_info['incumbent_mean']} "
                      f"on the SAME worlds -> {outcome}")] + critiques

    rec = {"iteration": i, "version": version,
           "mean_judge": round(mean_judge, 3), "per_category": per_category,
           "delta_vs_prev": delta_vs_prev, "regressed": regressed,
           "adoption": gate_info,
           "critiques": critiques, "pairwise": pairwise,
           "trace_summaries": trace_summaries, "digests": cur_digests}
    _append_jsonl(ITERATIONS_PATH, rec)
    return rec


def _history_for_optimizer(up_to_iteration: int) -> list[dict]:
    """Pinned history shape (DESIGN section 14); strips digests. Gold-blind.
    Carries each version's full spec text (genealogy / revert rights, DESIGN
    9.1) plus the regression flag."""
    items = []
    for r in sorted(_read_jsonl(ITERATIONS_PATH),
                    key=lambda x: int(x.get("iteration") or 0)):
        it = r.get("iteration")
        if not isinstance(it, int) or it > up_to_iteration:
            continue
        spec_text = ""
        try:
            sp = _spec_path(str(r.get("version") or ""))
            if sp.exists():
                spec_text = sp.read_text(encoding="utf-8")
        except Exception:
            pass
        items.append({"iteration": it,
                      "version": r.get("version"),
                      "mean_judge": r.get("mean_judge"),
                      "per_category": r.get("per_category") or {},
                      "delta_vs_prev": r.get("delta_vs_prev"),
                      "regressed": bool(r.get("regressed")),
                      "adoption": r.get("adoption"),
                      "spec_text": spec_text,
                      "critiques": r.get("critiques") or [],
                      "pairwise": r.get("pairwise") or [],
                      "trace_summaries": r.get("trace_summaries") or []})
    return items


def _revise(i: int, current_spec_text: str, cfg: dict) -> None:
    """Write harness/revisions/v{i+1}.md (+ rationale). Idempotent. On
    optimizer failure keep the current spec (no-op revision)."""
    next_md = REVISIONS_DIR / f"v{i + 1}.md"
    if next_md.exists():
        return
    new_spec, rationale = None, ""
    try:
        from rhi.optimizer import propose_revision
        from runtime.spec import parse_harness, validate_harness
        new_spec, rationale = propose_revision(
            _history_for_optimizer(i), current_spec_text, cfg)
        validate_harness(parse_harness(new_spec))  # belt and braces
    except Exception as e:
        _log(f"optimizer failed for v{i + 1}: {e!r}; keeping current spec")
        new_spec = current_spec_text
        rationale = (f"Optimizer failure ({e!r}); previous spec carried "
                     f"forward unchanged (no-op revision).")
    REVISIONS_DIR.mkdir(parents=True, exist_ok=True)
    next_md.write_text(new_spec, encoding="utf-8")
    (REVISIONS_DIR / f"v{i + 1}.rationale.md").write_text(
        rationale or "", encoding="utf-8")
    _log(f"wrote {os.path.relpath(next_md, ROOT)}")


# ------------------------------------------------------------------ commands

def cmd_run(args) -> int:
    cfg = _load_config(args.config)
    frozen = _load_json(FROZEN_PATH, {}) or {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state, completed = _load_state()

    cap = _budget_cap(cfg)
    prior = _prior_spend()
    over_budget = _make_budget_check(cap, prior)
    # Mid-task budget checks inside run_task must respect prior-session spend.
    cfg = {**cfg, "_effective_budget_usd_cap": max(0.0, cap - prior)}
    _log(f"budget cap=${cap:.2f}, already spent per records=${prior:.2f}")

    iterations = int(cfg.get("iterations", 10))
    checkpoints = {int(c) for c in cfg.get("heldout_checkpoints", [])}

    # Checkpoint 0: v0 before any training iteration.
    if 0 in checkpoints:
        spec0, _ = _load_spec("v0")
        _run_checkpoint(0, "v0", spec0, cfg, frozen, state, completed,
                        over_budget, cap, prior)

    gates_map = {}
    for g in _read_jsonl(RESULTS_DIR / "gates.jsonl"):
        if isinstance(g.get("iteration"), int):
            gates_map[g["iteration"]] = g          # last write wins

    for i in range(iterations):
        version = f"v{i}"                      # the candidate (latest proposal)

        # Resume fast-forward: an already-finalized iteration must not re-run
        # its slice or re-gate (a later accepted_version would otherwise leak
        # backwards as this iteration's incumbent). Recompute its adoption
        # from the records and move on.
        finalized = {r.get("iteration"): r for r in _read_jsonl(ITERATIONS_PATH)}
        if i in finalized and (REVISIONS_DIR / f"v{i + 1}.md").exists():
            g = gates_map.get(i) or (finalized[i].get("adoption") or {})
            state["accepted_version"] = g.get("adopted") or version
            state["iteration"] = i + 1
            state["harness_version"] = f"v{i + 1}"
            _save_state(state, completed)
            _log(f"iteration {i}: already finalized "
                 f"(deployed={state['accepted_version']}) — fast-forward")
            if (i + 1) in checkpoints:
                spec_c, _ = _load_spec(f"v{i + 1}")
                _run_checkpoint(i + 1, f"v{i + 1}", spec_c, cfg, frozen, state,
                                completed, over_budget, cap, prior)
            continue

        accepted_version = str(state.get("accepted_version")
                               or (f"v{i - 1}" if i > 0 else "v0"))
        gating = (bool(cfg.get("acceptance_gating"))
                  and version != accepted_version)
        spec, spec_text = _load_spec(version)
        variants = [(version, spec, "")]
        if gating:
            inc_spec, _ = _load_spec(accepted_version)
            variants.append((accepted_version, inc_spec, "@inc"))
        worlds = _iter_worlds(i, cfg)
        _log(f"iteration {i}: candidate={version}"
             + (f" vs incumbent={accepted_version} (acceptance gate)"
                if gating else "") + f" worlds={worlds}")

        ran_or_done = False
        pending = []   # (wid, task, run_id, files_dir, ver, spec_obj)
        for wid in worlds:
            files_dir = WORLDS_DIR / "train" / wid / "files"
            tasks = _read_tasks("train", wid)
            if not tasks:
                _log(f"WARNING: no tasks for train world {wid} "
                     f"(missing {GOLD_DIR / 'train' / wid / 'tasks.json'}?)")
                continue
            for task in tasks:
                for ver, sp, suffix in variants:
                    run_id = f"train-i{i:02d}-{task['task_id']}{suffix}"
                    if run_id in completed:
                        ran_or_done = True
                        continue
                    pending.append((wid, task, run_id, files_dir, ver, sp))
        if pending and over_budget():
            _halt_budget(state, completed, cap, prior)
        stream_items = [(fd, t, sp) for (_w, t, _r, fd, _v, sp) in pending]
        for idx, core, digest in _execute_stream(stream_items, cfg, over_budget):
                wid, task, run_id, _fd, ver, _sp = pending[idx]
                rec = {"run_id": run_id, "phase": "train", "iteration": i,
                       "harness_version": ver, "world_id": wid,
                       "task_id": task["task_id"], "category": task["category"],
                       "answer": core["answer"], "ended": core["ended"],
                       "judge_score": core["judge_score"],
                       "judge_critique": core["judge_critique"],
                       "gold_score": None,
                       "usd": core["usd"], "tokens": core["tokens"],
                       "steps": core["steps"], "wallclock_s": core["wallclock_s"],
                       "session": SESSION_ID}
                _append_jsonl(RUNS_PATH, rec)
                _append_jsonl(DIGESTS_PATH,
                              {"run_id": run_id, "iteration": i,
                               "harness_version": ver,
                               "category": task["category"],
                               "task_id": task["task_id"], **digest})
                _record_session_spend()
                if core["ended"] != "error":
                    completed.add(run_id)
                ran_or_done = True
                state["iteration"] = i
                state["harness_version"] = version
                _save_state(state, completed)
                _log(f"  {run_id}: ended={core['ended']} "
                     f"judge={core['judge_score']:.1f} usd={core['usd']:.4f}")

        if not ran_or_done:
            _log(f"ERROR: no tasks found for iteration {i} worlds {worlds}. "
                 f"Run the generator first (python -m generator.build).")
            _save_state(state, completed)
            return 1

        # -- acceptance gate (judge-only; gold never consulted) --------------
        gate_info = None
        adopted = version
        if gating:
            def _slice_mean(want_inc: bool) -> float:
                latest = {}
                for r in _read_jsonl(RUNS_PATH):
                    if (r.get("phase") == "train"
                            and r.get("iteration") == i
                            and str(r.get("run_id", "")).endswith("@inc") == want_inc):
                        latest[r["run_id"]] = r
                sc = [float(r.get("judge_score") or 0.0) for r in latest.values()]
                return round(sum(sc) / len(sc), 3) if sc else 0.0
            cand_mean, inc_mean = _slice_mean(False), _slice_mean(True)
            adopted = version if cand_mean >= inc_mean else accepted_version
            gate_info = {"candidate": version, "candidate_mean": cand_mean,
                         "incumbent": accepted_version,
                         "incumbent_mean": inc_mean, "adopted": adopted}
            _log(f"  acceptance gate: {version}={cand_mean} vs "
                 f"{accepted_version}={inc_mean} -> deployed {adopted}")
            _append_jsonl(RESULTS_DIR / "gates.jsonl",
                          {"iteration": i, **gate_info})
        state["accepted_version"] = adopted
        _save_state(state, completed)

        _finalize_iteration(i, version, cfg, gate_info)
        adopted_text = (spec_text if adopted == version
                        else _load_spec(adopted)[1])
        _revise(i, adopted_text, cfg)
        _record_session_spend()   # pairwise + optimizer spend just accrued
        state["iteration"] = i + 1
        state["harness_version"] = f"v{i + 1}"
        _save_state(state, completed)

        if (i + 1) in checkpoints:
            spec_c, _ = _load_spec(f"v{i + 1}")
            _run_checkpoint(i + 1, f"v{i + 1}", spec_c, cfg, frozen, state,
                            completed, over_budget, cap, prior)

    _record_session_spend()
    _log("run complete")
    _print_summary()
    return 0


def cmd_heldout(args) -> int:
    cfg = _load_config(args.config)
    frozen = _load_json(FROZEN_PATH, {}) or {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state, completed = _load_state()

    cap = _budget_cap(cfg)
    prior = _prior_spend()
    over_budget = _make_budget_check(cap, prior)
    cfg = {**cfg, "_effective_budget_usd_cap": max(0.0, cap - prior)}

    path = Path(args.harness)
    if not path.is_absolute() and not path.exists():
        path = ROOT / args.harness
    if not path.exists():
        _log(f"ERROR: harness spec not found: {args.harness}")
        return 1

    from runtime.spec import parse_harness, validate_harness
    text = path.read_text(encoding="utf-8")
    spec = parse_harness(text)
    validate_harness(spec)

    tag = args.tag or path.stem
    version = getattr(spec, "version_label", "") or path.stem
    _run_checkpoint(None, version, spec, cfg, frozen, state, completed,
                    over_budget, cap, prior, tag=tag)
    _record_session_spend()
    _log(f"heldout eval '{tag}' complete")
    _print_summary()
    return 0


def cmd_status(args) -> int:
    cfg = _load_json(args.config or CONFIG_PATH, {}) or {}
    state, completed = _load_state()
    runs = _read_jsonl(RUNS_PATH)
    cps = _read_jsonl(CHECKPOINTS_PATH)
    spent = _prior_spend()
    cap = _budget_cap(cfg)

    train = [r for r in runs if r.get("phase") == "train"]
    iters_done = sorted({r.get("iteration") for r in train
                         if isinstance(r.get("iteration"), int)})
    tags = []
    for r in cps:
        t = r.get("tag")
        if t and t not in tags:
            tags.append(t)

    print("FinForge status")
    print(f"  harness version : {state['harness_version']}")
    print(f"  next iteration  : {state['iteration']} "
          f"of {cfg.get('iterations', 10)}")
    print(f"  train runs      : {len(train)} "
          f"({sum(1 for r in train if r.get('ended') == 'error')} errors) "
          f"across iterations {iters_done or '[]'}")
    print(f"  heldout records : {len(cps)} across tags {tags or '[]'}")
    print(f"  completed ids   : {len(completed)}")
    print(f"  budget          : ${spent:.2f} spent of ${cap:.2f} cap "
          f"(${max(0.0, cap - spent):.2f} remaining)")
    if runs or cps:
        print()
        try:
            from scoring.report import aggregate, render_table
            print(render_table(aggregate(str(RESULTS_DIR), str(GOLD_DIR),
                                         write=False)))
        except Exception as e:
            _log(f"summary table unavailable: {e!r}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experiment.py",
                                 description="FinForge experiment driver")
    ap.add_argument("--config", default=None,
                    help="path to experiment.json (default: config/experiment.json)")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("run", help="full training loop per config")

    hp = sub.add_parser("heldout", help="standalone heldout eval of one spec")
    hp.add_argument("--harness", required=True, help="path to a harness spec .md")
    hp.add_argument("--tag", default=None, help="label for checkpoint records")

    sub.add_parser("status", help="budget + progress from results/")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "heldout":
        return cmd_heldout(args)
    if args.cmd == "status":
        return cmd_status(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
