#!/usr/bin/env python3
"""Gate 2: gold self-consistency. Identity answers built from gold must score
>= 99 under scoring.gold_score; perturbed answers must score strictly lower.
$0 — no API calls. Usage: python3 scripts/oracle_check.py
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.gold_score import score_task            # noqa: E402

FROZEN = json.load(open("config/scoring.frozen.json"))
FAILS = []


def identity_answer(cat: str, gold: dict) -> dict:
    """What a perfect accountant would submit, built from gold fields only."""
    g = copy.deepcopy(gold)
    if cat == "bank_rec":
        return {k: g[k] for k in ("bank_statement_ending_balance",
                                  "gl_cash_ending_balance", "adjusted_balance",
                                  "reconciling_items", "proposed_journal_entries")}
    if cat == "journal_entries":
        return {"entries": g["entries"]}
    if cat == "variance_analysis":
        return {"variances": g["variances"], "summary": g.get("summary", "")}
    out = {"schedule": g["schedule"], "journal_entry": g["journal_entry"]}
    if "journal_entry_status" in g:
        out["journal_entry_status"] = g["journal_entry_status"]
    return out


def perturb(cat: str, ans: dict) -> dict:
    a = copy.deepcopy(ans)
    if cat == "bank_rec" and a["reconciling_items"]:
        a["reconciling_items"] = a["reconciling_items"][1:]
    elif cat == "journal_entries" and a["entries"]:
        a["entries"][0]["lines"][0]["debit"] = (
            (a["entries"][0]["lines"][0]["debit"] or 0) + 100.0)
        a["entries"][0]["lines"][0]["credit"] = 0
    elif cat == "variance_analysis" and a["variances"]:
        a["variances"][0]["drivers"] = []
        a["variances"][0]["actual"] = (a["variances"][0]["actual"] or 0) + 500.0
    elif a.get("schedule"):
        a["schedule"][0]["closing_balance"] = (
            (a["schedule"][0]["closing_balance"] or 0) + 77.0)
    return a


def main():
    n = 0
    for split in ("train", "heldout"):
        base = os.path.join("gold", split)
        for wid in sorted(os.listdir(base)):
            gdir = os.path.join(base, wid)
            meta = json.load(open(os.path.join(gdir, "meta.json")))
            coa = meta["chart_of_accounts"]
            for t in json.load(open(os.path.join(gdir, "tasks.json"))):
                cat, tid = t["category"], t["task_id"]
                gold = json.load(open(os.path.join(gdir, tid + ".json")))
                ident = identity_answer(cat, gold)
                s = score_task(cat, ident, gold, coa, FROZEN)["score"]
                if s < 99.0:
                    FAILS.append(f"{tid}: identity scored {s}")
                    print(f"  [FAIL] {tid}: identity={s}")
                    print("        ", score_task(cat, ident, gold, coa, FROZEN)["details"][:300])
                sp = score_task(cat, perturb(cat, ident), gold, coa, FROZEN)["score"]
                if sp >= s:
                    FAILS.append(f"{tid}: perturbed {sp} >= identity {s}")
                    print(f"  [FAIL] {tid}: perturbed={sp} identity={s}")
                n += 1
    if FAILS:
        print(f"\nGate 2 FAILED: {len(FAILS)}/{n} checks")
        sys.exit(1)
    print(f"Gate 2 PASSED: {n} tasks, identity >= 99 and perturbations strictly lower")


if __name__ == "__main__":
    main()
