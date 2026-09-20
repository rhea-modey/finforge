#!/usr/bin/env python3
"""LIVE pilot smoke: run harness/v0.md on two w01 tasks end-to-end.

Runs run_task LIVE, then the gold-blind training judge, then (pilot-only,
train worlds, results go nowhere near the optimizer) gold_score. Prints
per-task metrics + METER totals. Writes nothing into results/runs.jsonl.

Usage: source scripts/env.sh && python3 scripts/pilot.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.spec import parse_harness, validate_harness          # noqa: E402
from runtime.orchestrator import run_task                          # noqa: E402
from runtime.trace import summarize_trace                          # noqa: E402
from runtime.llm import METER                                      # noqa: E402
from judge.training_judge import judge_task                        # noqa: E402
from scoring.gold_score import score_task                          # noqa: E402

PILOT_TASKS = ["w01-bank_rec", "w01-journal_entries"]
WID = "w01"


def main() -> int:
    cfg = json.loads((ROOT / "config" / "experiment.json").read_text())
    frozen = json.loads((ROOT / "config" / "scoring.frozen.json").read_text())
    spec_text = (ROOT / "harness" / "v0.md").read_text()
    spec = parse_harness(spec_text)
    validate_harness(spec)

    tasks = {t["task_id"]: t for t in json.loads(
        (ROOT / "gold" / "train" / WID / "tasks.json").read_text())}
    meta = json.loads((ROOT / "gold" / "train" / WID / "meta.json").read_text())
    chart = meta.get("chart_of_accounts") or []
    files_dir = ROOT / "worlds" / "train" / WID / "files"
    results_dir = ROOT / "results" / "pilot"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tid in PILOT_TASKS:
        task = tasks[tid]
        print(f"\n=== {tid} ({task['category']}) ===", flush=True)
        t0 = time.time()
        res = run_task(spec, str(files_dir), task["prompt"], tid, cfg,
                       str(results_dir))
        print(f"run_task done in {time.time() - t0:.0f}s: ended={res.ended} "
              f"steps={res.steps} usd={res.usd:.4f} tokens={res.tokens}",
              flush=True)

        try:
            tsum = summarize_trace(res.trace_path)
        except Exception as e:  # noqa: BLE001
            tsum = f"[trace summarization failed: {e!r}]"

        j = judge_task(task["prompt"], res.answer, res.notes, tsum, cfg)
        gold = json.loads(
            (ROOT / "gold" / "train" / WID / f"{tid}.json").read_text())
        g = score_task(task["category"], res.answer, gold, chart, frozen)

        ans_str = json.dumps(res.answer, default=str) if res.answer else "(none)"
        rows.append({
            "task_id": tid, "ended": res.ended, "steps": res.steps,
            "tokens": res.tokens, "usd": round(res.usd, 4),
            "wallclock_s": res.wallclock_s,
            "judge_score": round(float(j.get("score") or 0.0), 2),
            "gold_score": round(float(g.get("score") or 0.0), 1),
            "gold_components": g.get("components", {}),
            "judge_critique_first300": (j.get("critique") or "")[:300],
            "answer_first400": ans_str[:400],
            "trace_path": res.trace_path,
        })
        r = rows[-1]
        print(f"judge={r['judge_score']}/10 gold={r['gold_score']}/100 "
              f"components={r['gold_components']}", flush=True)
        print(f"answer[:400]: {r['answer_first400']}", flush=True)

    print("\n=== METER ===")
    print(json.dumps(METER.snapshot(), indent=1))
    print("\n=== PILOT SUMMARY (json) ===")
    print(json.dumps({"tasks": rows, "meter": METER.snapshot()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
