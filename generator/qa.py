"""Red-team QA pass (DESIGN 15). Two LLM calls per world:

Call 1 (NO gold): the model sees the full files/ tree + all 4 task prompts and
must (a) flag contradictions/impossibilities, (b) give a per-task solvability
verdict from the files alone, (c) spot-derive the bank statement ending
balance and the GL cash balance at close. Code compares (c) to gold.

Call 2 (gold summary revealed — balances + entry amounts only): "does gold
follow from the files?"

Output: gold/{split}/{wid}/qa_report.json {"pass": bool, "issues": [...],
"detail": {...}}. Failing worlds are FLAGGED only (no auto-rebuild here).

CLI: python3 -m generator.qa [--worlds-dir worlds] [--gold-dir gold]
        [--model M] [--only wid[,wid]] [--workers 6] [--force]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading

_PRICE_IN, _PRICE_OUT = 1.25, 10.0
_LOCK = threading.Lock()
_USAGE = {"prompt": 0, "completion": 0, "calls": 0}

BALANCE_TOL = 0.02


def _client():
    from openai import OpenAI
    key = os.environ.get("DATAGEN_API_KEY")
    if not key:
        raise SystemExit("DATAGEN_API_KEY not set (source scripts/env.sh)")
    return OpenAI(api_key=key, base_url=os.environ.get("DATAGEN_BASE_URL") or None)


def _call(client, model: str, prompt: str, max_tokens: int = 20000) -> dict:
    last_err = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
            )
            with _LOCK:
                _USAGE["prompt"] += r.usage.prompt_tokens
                _USAGE["completion"] += r.usage.completion_tokens
                _USAGE["calls"] += 1
            if r.choices[0].finish_reason == "length":
                max_tokens = min(max_tokens * 2, 48000)
                last_err = "truncated"
                continue
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError as e:
            last_err = f"bad JSON: {e}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            import time
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"QA LLM call failed: {last_err}")


def _files_blob(files_dir: str) -> str:
    parts = []
    for dirpath, dirnames, fns in sorted(os.walk(files_dir)):
        dirnames.sort()
        for fn in sorted(fns):
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, files_dir)
            try:
                text = open(fp, encoding="utf-8").read()
            except UnicodeDecodeError:
                text = "<binary>"
            if len(text) > 20000:
                text = text[:20000] + "\n[truncated]"
            parts.append(f"===== FILE: {rel} =====\n{text}")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# gold summary for call 2: balances + entry amounts only
# --------------------------------------------------------------------------

def _je_lines(je: dict) -> list:
    return [{"account": l["account"], "debit": l["debit"], "credit": l["credit"]}
            for l in je.get("lines", [])]


def gold_summary(gdir: str, wid: str) -> dict:
    g = {}
    br = json.load(open(os.path.join(gdir, f"{wid}-bank_rec.json")))
    g["bank_rec"] = {
        "bank_statement_ending_balance": br["bank_statement_ending_balance"],
        "gl_cash_ending_balance": br["gl_cash_ending_balance"],
        "adjusted_balance": br["adjusted_balance"],
        "reconciling_items": [{"kind": i["kind"], "amount": i["amount"],
                               "side": i["side"]}
                              for i in br["reconciling_items"]],
        "correcting_entries": [{"date": e.get("date"), "lines": _je_lines(e)}
                               for e in br["proposed_journal_entries"]],
    }
    je = json.load(open(os.path.join(gdir, f"{wid}-journal_entries.json")))
    g["journal_entries"] = [{"date": e.get("date"), "lines": _je_lines(e)}
                            for e in je["entries"]]
    va = json.load(open(os.path.join(gdir, f"{wid}-variance_analysis.json")))
    g["variance_analysis"] = [
        {"account": v["account"], "actual": v["actual"], "budget": v["budget"],
         "variance": v["variance"], "direction": v.get("direction"),
         "driver_amounts": [d["amount"] for d in v.get("drivers", [])]}
        for v in va["variances"]]
    ac = json.load(open(os.path.join(gdir, f"{wid}-accrual_schedule.json")))
    g["accrual_schedule"] = {
        "rows": ac["schedule"],
        "journal_entry_lines": _je_lines(ac["journal_entry"]),
        # Convention: the schedule's journal entry is its SUPPORTING entry.
        # "already_booked" = the GL already contains it (it is documentation,
        # NOT a proposal to post again); "to_book" = it must be posted.
        "journal_entry_status": ac.get("journal_entry_status", "to_book"),
    }
    return g


# --------------------------------------------------------------------------
# per-world QA
# --------------------------------------------------------------------------

def qa_world(split: str, wid: str, worlds_dir: str, gold_dir: str,
             model: str, client=None, force: bool = False) -> dict:
    gdir = os.path.join(gold_dir, split, wid)
    report_path = os.path.join(gdir, "qa_report.json")
    if os.path.exists(report_path) and not force:
        rep = json.load(open(report_path))
        return {"wid": wid, "cached": True, "pass": rep["pass"],
                "issues": rep["issues"]}
    if client is None:
        client = _client()
    files_dir = os.path.join(worlds_dir, split, wid, "files")
    blob = _files_blob(files_dir)
    tasks = json.load(open(os.path.join(gdir, "tasks.json")))
    task_ids = [t["task_id"] for t in tasks]
    tasks_txt = "\n\n".join(f"--- TASK {t['task_id']} ({t['category']}) ---\n"
                            f"{t['prompt']}" for t in tasks)

    p1 = (
        "You are a skeptical senior auditor red-teaming a synthetic month-end "
        "close exercise. Below is the COMPLETE file tree a candidate would see, "
        "plus the four task prompts. You have NO other information.\n\n"
        f"{blob}\n\n{tasks_txt}\n\n"
        "Respond with STRICT JSON, keys exactly:\n"
        '{"contradictions": [{"severity": "high|medium|low", "description": '
        '"..."}],\n'
        f' "task_solvability": {{{", ".join(chr(34) + t + chr(34) + ": {\"solvable\": true, \"reason\": \"...\"}" for t in task_ids)}}},\n'
        ' "derived": {"bank_statement_ending_balance": 0.00, '
        '"gl_cash_ending_balance": 0.00}}\n\n'
        "Guidance:\n"
        "- This is a standard bookkeeping exercise; assume ordinary "
        "professional judgment and conventions. GL coding is presumed correct "
        "unless a source document contradicts it. The prepaid population is "
        "defined by the prior prepaid schedule plus the contracts/ folder. "
        "The close checklist's stated conventions govern.\n"
        "- contradictions: only genuine defects — two files disagreeing on the "
        "same fact, arithmetic that does not tie, information a task requires "
        "that is absent, impossible dates. severity=high is reserved for "
        "defects where NO defensible reading of the files yields a unique "
        "answer (a true impossibility or numeric contradiction). Matters of "
        "classification judgment are at most medium. An empty list is a valid "
        "answer.\n"
        "- task_solvability: can a competent accountant produce the requested "
        "deliverable from these files alone? Answer false only when required "
        "information is missing or contradictory — not merely because "
        "judgment is involved.\n"
        "- derived: compute the June bank statement ending balance from the "
        "bank statement file (opening balance plus all transaction rows), and "
        "the GL cash ending balance at 2026-06-30 (the cash line of the trial "
        "balance / GL). Show no work, just the two numbers."
    )
    c1 = _call(client, model, p1)

    p2 = (
        "You are auditing the OFFICIAL answer key of a synthetic month-end "
        "close exercise. Below is the complete file tree, followed by the "
        "answer-key summary (balances and entry amounts only).\n\n"
        f"{blob}\n\n===== ANSWER KEY SUMMARY =====\n"
        f"{json.dumps(gold_summary(gdir, wid), indent=1)}\n\n"
        "Question: does this answer key follow from the files? Check that the "
        "balances match the files, that each correcting/adjusting entry amount "
        "is supported by a document or statement line, and that the variance "
        "actual/budget figures tie to the ledger and budget files.\n"
        "Conventions of this exercise (do NOT flag these as mismatches): "
        "(1) the accrual schedule's journal entry is its supporting entry — "
        "when journal_entry_status is 'already_booked' it documents an entry "
        "already in the GL, not a proposal to post it again; (2) variance "
        "actuals are on an AS-CLOSED basis: the checklist close entries AND "
        "the bank-reconciliation correcting entries (fees, interest, "
        "transposition fixes) are treated as booked when computing June "
        "actuals.\n"
        "Respond with STRICT JSON: {\"gold_consistent\": true|false, "
        "\"mismatches\": [\"...\"]} — list a mismatch ONLY when the files "
        "clearly support a different number than the key."
    )
    c2 = _call(client, model, p2)

    # code-level comparison of the spot-derived balances vs gold
    br = json.load(open(os.path.join(gdir, f"{wid}-bank_rec.json")))
    derived = c1.get("derived") or {}
    checks = {}
    for key, gold_v in (("bank_statement_ending_balance",
                         br["bank_statement_ending_balance"]),
                        ("gl_cash_ending_balance",
                         br["gl_cash_ending_balance"])):
        try:
            got = float(derived.get(key))
            checks[key] = {"gold": gold_v, "derived": got,
                           "match": abs(got - gold_v) <= BALANCE_TOL}
        except (TypeError, ValueError):
            checks[key] = {"gold": gold_v, "derived": derived.get(key),
                           "match": False}

    issues = []
    for c in c1.get("contradictions", []):
        sev = str(c.get("severity", "")).lower()
        desc = c.get("description", str(c))
        if sev == "high":
            issues.append(f"[contradiction/high] {desc}")
    solv = c1.get("task_solvability") or {}
    for tid in task_ids:
        v = solv.get(tid) or next((v for k, v in solv.items() if tid in k
                                   or k in tid), None)
        if not (isinstance(v, dict) and v.get("solvable")):
            reason = v.get("reason") if isinstance(v, dict) else "no verdict"
            issues.append(f"[unsolvable] {tid}: {reason}")
    for key, ck in checks.items():
        if not ck["match"]:
            issues.append(f"[balance-mismatch] {key}: qa derived "
                          f"{ck['derived']} vs gold {ck['gold']}")
    if not c2.get("gold_consistent", False):
        for m in c2.get("mismatches", []) or ["no detail given"]:
            issues.append(f"[gold-mismatch] {m}")

    report = {
        "pass": not issues,
        "issues": issues,
        "detail": {
            "contradictions": c1.get("contradictions", []),
            "task_solvability": solv,
            "balance_checks": checks,
            "gold_consistent": c2.get("gold_consistent"),
            "gold_mismatches": c2.get("mismatches", []),
            "model": model,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")
    return {"wid": wid, "cached": False, "pass": report["pass"],
            "issues": issues}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", default="worlds")
    ap.add_argument("--gold-dir", default="gold")
    ap.add_argument("--model", default=os.environ.get(
        "QA_MODEL", os.environ.get("DATAGEN_MODEL", "gpt-5.6-sol")))
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from .build import TRAIN_IDS, HELDOUT_IDS
    targets = [("train", w) for w in TRAIN_IDS] + \
              [("heldout", w) for w in HELDOUT_IDS]
    if a.only:
        keep = set(a.only.split(","))
        targets = [(s, w) for s, w in targets if w in keep]

    client = _client()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(qa_world, s, w, a.worlds_dir, a.gold_dir, a.model,
                          client, a.force): (s, w) for s, w in targets}
        for fut in concurrent.futures.as_completed(futs):
            s, w = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"wid": w, "error": f"{type(e).__name__}: {e}",
                     "pass": False, "issues": [f"[qa-error] {e}"]}
            results.append(r)
            tag = "cache" if r.get("cached") else ("ERROR" if "error" in r else "fresh")
            status = "PASS" if r.get("pass") else "FAIL"
            print(f"[{tag}] {s}/{w}: {status}", flush=True)
            for i in r.get("issues", []):
                print(f"        {i}", flush=True)

    n_pass = sum(1 for r in results if r.get("pass"))
    usd = (_USAGE["prompt"] * _PRICE_IN + _USAGE["completion"] * _PRICE_OUT) / 1e6
    print(f"\nqa done: {n_pass}/{len(results)} worlds pass; "
          f"{_USAGE['calls']} calls, {_USAGE['prompt']} in / "
          f"{_USAGE['completion']} out tokens, ~${usd:.2f} est.")


if __name__ == "__main__":
    main()
