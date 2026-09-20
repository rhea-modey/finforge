#!/usr/bin/env python3
"""Gate 0 (determinism) + Gate 1 (integrity). $0 — no API calls.

Usage: python3 scripts/smoke_generator.py   (from repo root)
Exits nonzero on any failure.
"""

import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import build as B                      # noqa: E402
from generator.engine import build_world, bank_statement_june   # noqa: E402
from generator.profile import make_profile            # noqa: E402

FAILS = []


def check(ok: bool, msg: str) -> None:
    tag = "ok " if ok else "FAIL"
    print(f"  [{tag}] {msg}")
    if not ok:
        FAILS.append(msg)


def gate0():
    print("Gate 0: determinism (build w01+h04 twice, compare tree hashes)")
    hashes = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            wdir, gdir = os.path.join(td, "w"), os.path.join(td, "g")
            for split, wid in (("train", "w01"), ("heldout", "h04")):
                B.build_one(20260919, split, wid, wdir, gdir)
            hashes.append((B.tree_hash(wdir), B.tree_hash(gdir)))
    check(hashes[0] == hashes[1], f"byte-identical rebuild {hashes[0][0][:12]}...")


def gate1():
    print("Gate 1: integrity across all worlds")
    for split, ids in (("train", B.TRAIN_IDS), ("heldout", B.HELDOUT_IDS)):
        for wid in ids:
            prof = make_profile(20260919, split, wid)
            w = build_world(prof)
            # SHIPPED tree freshness: the committed worlds/ tree must equal a
            # fresh build AND the hash frozen in gold meta.json (catches stale
            # leftover files from earlier builds). Post-enrichment (DESIGN 15)
            # determinism is cache-based: a fresh skeleton + the cached
            # enrichment.json must reproduce the shipped tree byte-for-byte.
            fdir_shipped = os.path.join("worlds", split, wid, "files")
            enr_path = os.path.join("gold", split, wid, "enrichment.json")
            with tempfile.TemporaryDirectory() as td:
                B.build_one(20260919, split, wid,
                            os.path.join(td, "w"), os.path.join(td, "g"))
                tmp_files = os.path.join(td, "w", split, wid, "files")
                if os.path.exists(enr_path):
                    from generator.enrich import apply_enrichment
                    apply_enrichment(tmp_files, json.load(open(enr_path)))
                fresh_hash = B.tree_hash(tmp_files)
            shipped_hash = B.tree_hash(fdir_shipped)
            meta_hash = json.load(open(
                os.path.join("gold", split, wid, "meta.json")))["files_tree_hash"]
            if not (fresh_hash == shipped_hash == meta_hash):
                check(False, f"{wid}: shipped files/ tree is stale "
                             f"(fresh={fresh_hash[:10]} shipped={shipped_hash[:10]} "
                             f"meta={meta_hash[:10]})")
            dr = sum(p["debit_c"] for p in w.postings)
            cr = sum(p["credit_c"] for p in w.postings)
            if dr != cr:
                check(False, f"{wid}: GL unbalanced ({dr} vs {cr})")
            stmt = bank_statement_june(w)
            calc = stmt["opening_c"] + sum(r["amount_c"] for r in stmt["rows"])
            if calc != stmt["ending_c"]:
                check(False, f"{wid}: bank statement math broken")
            # every truth record observable (payroll/prepaid gaps are now
            # conditional per P1 — presence must match the injection plan)
            kinds = {t.kind for t in w.truth}
            need = {"outstanding_check", "bank_fee_unrecorded"}
            if not need <= kinds:
                check(False, f"{wid}: missing core truth kinds {need - kinds}")
            if w.plan.has_missing_payroll_accrual != ("missing_payroll_accrual" in kinds):
                check(False, f"{wid}: payroll accrual truth/plan mismatch")
            if w.plan.has_missing_prepaid_amort != ("missing_prepaid_amort" in kinds):
                check(False, f"{wid}: prepaid amort truth/plan mismatch")
            # unrecorded fee: on bank, not in GL June
            fees_gl = [p for p in w.postings if p["account"] == "6900"]
            if fees_gl:
                check(False, f"{wid}: bank fee leaked into GL")
            # payroll accrual: July-paid run in register iff injected
            july = [r for r in w.payroll_runs if r["pay_date"].startswith("2026-07")]
            if w.plan.has_missing_payroll_accrual and not july:
                check(False, f"{wid}: no July-paid payroll run in register")
            if not w.plan.has_missing_payroll_accrual and july:
                check(False, f"{wid}: unexpected July-paid run without injection")
            # AR aging must be a populated retrieval surface (DESIGN 4.4)
            open_ar = [i for i in w.invoices if i["paid"] is None]
            if not open_ar:
                check(False, f"{wid}: AR aging is empty (no open invoices at 6/30)")
            # no gold/meta in files/
            fdir = os.path.join("worlds", split, wid, "files")
            for dirpath, _, fns in os.walk(fdir):
                for fn in fns:
                    if "gold" in fn or fn == "meta.json" or fn.endswith(".rubric.json"):
                        check(False, f"{wid}: leak in files/: {fn}")
            # gold exists for all four tasks
            gdir = os.path.join("gold", split, wid)
            tasks = json.load(open(os.path.join(gdir, "tasks.json")))
            if len(tasks) != 4:
                check(False, f"{wid}: expected 4 tasks, got {len(tasks)}")
            for t in tasks:
                for suffix in (".json", ".rubric.json"):
                    p = os.path.join(gdir, t["task_id"] + suffix)
                    if not os.path.exists(p):
                        check(False, f"{wid}: missing {p}")
            # exported TB ties to GL (recompute from exported CSVs)
            _tb_ties(w, fdir, wid)
    check(True, "all worlds passed integrity checks")


def _tb_ties(w, files_dir, wid):
    delim = w.profile.delim
    tb_path = os.path.join(files_dir, "trial_balance_2026-06-30.csv")
    with open(tb_path) as f:
        rows = list(csv.DictReader(f, delimiter=delim))
    tb = {}
    for p in w.postings:
        tb[p["account"]] = tb.get(p["account"], 0) + p["debit_c"] - p["credit_c"]
    for r in rows:
        num = r["account_number"]
        bal = tb.get(num, 0)
        exp_dr = round(bal / 100.0, 2) if bal > 0 else None
        exp_cr = round(-bal / 100.0, 2) if bal < 0 else None
        got_dr = float(r["debit_balance"]) if r["debit_balance"] else None
        got_cr = float(r["credit_balance"]) if r["credit_balance"] else None
        if (exp_dr, exp_cr) != (got_dr, got_cr):
            check(False, f"{wid}: TB mismatch on {num}")
            return


if __name__ == "__main__":
    gate0()
    gate1()
    if FAILS:
        print(f"\n{len(FAILS)} FAILURES")
        sys.exit(1)
    print("\nGates 0-1 PASSED")
