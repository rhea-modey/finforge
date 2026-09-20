"""Emit the world's files/ tree (DESIGN 4.4). Never writes gold/meta/seeds.

Per-world surface variation comes from the profile (column names, date format,
delimiter, filenames). All iteration is sorted for byte-determinism.
"""

from __future__ import annotations

import csv
import io
import os
import random
import shutil
from datetime import datetime

from .engine import World, bank_statement_june, d2, june_actual, _pl_accounts

# Handoff-memo wording variants (seeded per world so no fixed sentinel
# sentence identifies the injections across worlds). Same facts, varied prose.
_MEMO_FEES = [
    "- The bank drops its own service charges on the last statement days —\n"
    "  we historically pick those up at close, not during the month.\n",
    "- Watch for the bank's own charges near the end of the statement; we\n"
    "  book those at close rather than during the month.\n",
    "- Service charges tend to hit the statement in its final days;\n"
    "  historically they get recorded at close.\n",
]
_MEMO_CODING = [
    "- I was rushing in June; coding on one or two vendor bills may be off.\n"
    "  The bill documents in bills/ are authoritative.\n",
    "- June was hectic — double-check expense coding on the vendor bills;\n"
    "  treat the documents in bills/ as authoritative.\n",
    "- A couple of June bills may have landed in the wrong expense account;\n"
    "  the bills/ documents are the source of truth.\n",
]
_MEMO_BUDGET = [
    "- Budget lines were set at the start of Q2; one-off June projects were\n"
    "  never budgeted, so expect them to explain most big gaps.\n",
    "- The budget was locked at the start of the quarter, so unplanned June\n"
    "  projects will not be in it — they usually explain the larger gaps.\n",
    "- Q2 budget was fixed in early April; anything one-off that landed in\n"
    "  June was never planned for, which is where big variances come from.\n",
]
_MEMO_PAYROLL_JULY = [
    "- Payroll: the second June run pays in early July this time - check the\n"
    "  register.\n",
    "- Heads up on payroll: the back half of June pays out in early July;\n"
    "  the register has the dates.\n",
    "- The June 16-30 payroll run does not pay until the first days of July\n"
    "  this month - see the register.\n",
]
_MEMO_PAYROLL_NEUTRAL = [
    "- Payroll timing moves around here: always check the register for when\n"
    "  the June 16-30 run actually pays before deciding on accruals.\n",
    "- On payroll, do not assume - the pay date on the second monthly run\n"
    "  varies, so verify it in the register.\n",
    "- Payroll: the register is the source of truth for pay dates on the\n"
    "  back-half run; some months it slips into the next month.\n",
]


def _fmt_date(iso: str, fmt: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime(fmt)


def _write_csv(path: str, header: list[str], rows: list[list], delim: str) -> None:
    buf = io.StringIO()
    wcsv = csv.writer(buf, delimiter=delim, lineterminator="\n")
    wcsv.writerow(header)
    for r in rows:
        wcsv.writerow(r)
    with open(path, "w", newline="") as f:
        f.write(buf.getvalue())


def export_world(w: World, files_dir: str) -> None:
    # Always start from an empty tree: a stale files_dir from an earlier build
    # must never leave orphaned documents behind (they would contradict gold).
    if os.path.isdir(files_dir):
        shutil.rmtree(files_dir)
    os.makedirs(files_dir)
    for sub in ("invoices", "bills", "contracts", "notes"):
        os.makedirs(os.path.join(files_dir, sub), exist_ok=True)
    p = w.profile
    c = p.cols
    fd = lambda iso: _fmt_date(iso, p.date_fmt)
    rnd_txt = random.Random(p.seed + 4)   # note-wording variation only

    # ---- chart of accounts
    _write_csv(os.path.join(files_dir, "chart_of_accounts.csv"),
               ["account_number", "account_name"],
               [[a["number"], a["name"]] for a in w.coa], p.delim)

    # ---- general ledger (as recorded, injections included)
    rows = [[fd(x["date"]), f"{x['account']} {x['account_name']}",
             (d2(x["debit_c"]) if x["debit_c"] else ""),
             (d2(x["credit_c"]) if x["credit_c"] else ""), x["memo"]]
            for x in sorted(w.postings, key=lambda x: (x["date"], x["entry_id"],
                                                       x["account"]))]
    _write_csv(os.path.join(files_dir, p.gl_file),
               [c["date"], c["account"], c["debit"], c["credit"], c["memo"]],
               rows, p.delim)

    # ---- trial balance at 6/30
    tb = {}
    for x in w.postings:
        tb[x["account"]] = tb.get(x["account"], 0) + x["debit_c"] - x["credit_c"]
    tb_rows = []
    for a in w.coa:
        bal = tb.get(a["number"], 0)
        tb_rows.append([a["number"], a["name"],
                        d2(bal) if bal > 0 else "",
                        d2(-bal) if bal < 0 else ""])
    _write_csv(os.path.join(files_dir, "trial_balance_2026-06-30.csv"),
               ["account_number", "account_name", "debit_balance", "credit_balance"],
               tb_rows, p.delim)

    # ---- bank statement (June)
    stmt = bank_statement_june(w)
    bank_rows = [[fd("2026-05-31"), "OPENING BALANCE", "", d2(stmt["opening_c"])]]
    running = stmt["opening_c"]
    for e in stmt["rows"]:
        running += e["amount_c"]
        bank_rows.append([fd(e["cleared"]), e["desc"], d2(e["amount_c"]),
                          d2(running)])
    bank_rows.append([fd("2026-06-30"), "ENDING BALANCE", "", d2(stmt["ending_c"])])
    _write_csv(os.path.join(files_dir, p.bank_file),
               [c["date"], c["bank_desc"], c["amount"], "balance"],
               bank_rows, p.delim)

    # ---- AR / AP aging at 6/30
    ar_rows = [[i["invoice_id"], i["customer"], fd(i["date"]), d2(i["amount_c"])]
               for i in sorted(w.invoices, key=lambda i: i["invoice_id"])
               if i["paid"] is None]
    _write_csv(os.path.join(files_dir, "ar_aging_2026-06-30.csv"),
               ["invoice", "customer", "invoice_date", "open_amount"], ar_rows, p.delim)
    ap_rows = [[b["bill_id"], b["vendor"], fd(b["date"]), d2(b["amount_c"])]
               for b in sorted(w.bills, key=lambda b: b["bill_id"])
               if b["paid"] is None]
    _write_csv(os.path.join(files_dir, "ap_aging_2026-06-30.csv"),
               ["bill", "vendor", "bill_date", "open_amount"], ap_rows, p.delim)

    # ---- budget (Apr-Jun, per P&L account)
    brows = []
    for acct in sorted(w.budget):
        nm = w.acct_name(acct)
        b = w.budget[acct]
        brows.append([acct, nm, d2(b[4]), d2(b[5]), d2(b[6])])
    _write_csv(os.path.join(files_dir, "budget_fy2026_q2.csv"),
               ["account_number", "account_name", "apr_budget", "may_budget",
                "jun_budget"], brows, p.delim)

    # ---- payroll register
    pr = [[r["period"], r["pay_date"], d2(r["gross_c"]), d2(r["withheld_c"]),
           d2(r["net_c"])]
          for r in sorted(w.payroll_runs, key=lambda r: (r["period"]))]
    _write_csv(os.path.join(files_dir, "payroll_register_q2.csv"),
               ["pay_period", "pay_date", "gross_wages", "withholdings", "net_pay"],
               pr, p.delim)

    # ---- prior prepaid schedule (as of 5/31) + contract
    sched = []
    for item in w.prepaid:
        if item["start"] > "2026-05-31":
            continue                      # June additions are not on the prior schedule
        months_amortized_through_apr = _months_amortized(item, "2026-04-30")
        opening_may = item["premium_c"] - months_amortized_through_apr * item["monthly_c"]
        closing_may = opening_may - item["monthly_c"]
        sched.append([item["item"], d2(opening_may), "0.00",
                      d2(item["monthly_c"]), d2(closing_may)])
    _write_csv(os.path.join(files_dir, "prepaid_schedule_2026-05-31.csv"),
               ["item", "opening_balance_may", "additions", "amortization",
                "closing_balance_may"], sched, p.delim)
    ins = w.prepaid[0]
    with open(os.path.join(files_dir, "contracts", "insurance_policy.txt"), "w") as f:
        f.write(
            f"COMMERCIAL PACKAGE POLICY\nInsured: {p.company}\n"
            f"Broker: {ins.get('vendor', 'the insurance broker')}\n"
            f"Policy term: 2026-01-01 through 2026-12-31 (12 months)\n"
            f"Annual premium: ${d2(ins['premium_c']):,.2f} paid in full at inception.\n"
            f"For accounting purposes the premium is amortized on a straight-line\n"
            f"basis over the policy term (a full month is recognized in each\n"
            f"month the policy is in force).\n")
    # License agreement for any prepaid item beginning in-quarter (the second
    # prepaid) so its term, amount, and first-month amortization convention
    # are all derivable from files/.
    for item in w.prepaid[1:]:
        with open(os.path.join(files_dir, "contracts", "software_license.txt"), "w") as f:
            f.write(
                f"SOFTWARE LICENSE AGREEMENT\nLicensee: {p.company}\n"
                f"Vendor: {item.get('vendor', 'the software vendor')}\n"
                f"License: {item['item']}\n"
                f"Term: {item['term_months']} months beginning {item['start']}.\n"
                f"Total license fee: ${d2(item['premium_c']):,.2f} paid in full "
                f"at signing.\n"
                f"For accounting purposes the fee is amortized on a straight-line\n"
                f"basis over the term, with a full month recognized in the month\n"
                f"of purchase.\n")

    # ---- fixed asset register
    fa = [[a["asset"], d2(a["cost_c"]), a["method"], d2(a["monthly_dep_c"])]
          for a in w.assets]
    _write_csv(os.path.join(files_dir, "fixed_asset_register.csv"),
               ["asset", "cost", "method", "monthly_depreciation"], fa, p.delim)

    # ---- invoice + bill documents (subset; always all driver/misposted bills)
    for i in sorted(w.invoices, key=lambda i: i["invoice_id"])[:5]:
        with open(os.path.join(files_dir, "invoices", f"{i['invoice_id']}.txt"), "w") as f:
            f.write(f"{p.company}\nINVOICE {i['invoice_id']}\nBill to: {i['customer']}\n"
                    f"Date: {i['date']}\nAmount due: ${d2(i['amount_c']):,.2f}\n"
                    f"Terms: Net 30\n")
    # Bills referenced by any truth record always get a document, so files/
    # carries independent evidence (e.g. the true face amount of the
    # transposed check comes from its bill).
    truth_bill_ids = {t.detail.get("bill_id") for t in w.truth
                      if t.detail.get("bill_id")}
    doc_bills = [b for b in w.bills
                 if b.get("desc") or b.get("posted_acct")
                 or b["bill_id"] in truth_bill_ids]
    doc_bills += [b for b in sorted(w.bills, key=lambda b: b["bill_id"])
                  if b not in doc_bills][:4]
    for b in sorted(doc_bills, key=lambda b: b["bill_id"]):
        with open(os.path.join(files_dir, "bills", f"{b['bill_id']}.txt"), "w") as f:
            desc = b.get("desc") or f"Services - {w.acct_name(b['acct'])}"
            f.write(f"{b['vendor']}\nBILL {b['bill_id']}\nTo: {p.company}\n"
                    f"Date: {b['date']}\nDescription: {desc}\n"
                    f"Amount: ${d2(b['amount_c']):,.2f}\n")

    # ---- notes
    checklist = (
        f"# Month-end close checklist — June 2026\n\n"
        f"Prepared by {p.controller}. Work through in order:\n\n"
        f"1. Reconcile operating cash to the June bank statement. Propose one\n"
        f"   correcting journal entry per book-side item the reconciliation\n"
        f"   surfaces.\n"
        f"2. Book payroll accrual for any pay period ending after 6/30 that pays in\n"
        f"   July, if one exists (see payroll register; some months every run pays\n"
        f"   in-month). Accrue the full gross wages to Accrued Liabilities.\n"
        f"3. Verify June prepaid amortization is booked per the prepaid schedule and\n"
        f"   contracts; if missing, book it as one combined journal entry. Check the\n"
        f"   GL first — do not book it twice. Convention: a full month of\n"
        f"   amortization is taken in the month a prepaid item begins.\n"
        f"4. Book June depreciation per the fixed asset register (verify against the GL first).\n"
        f"5. Verify June vendor bills are coded to the correct expense accounts\n"
        f"   (check bill documents against the GL; reclass anything miscoded).\n"
        f"6. Prepare budget-vs-actual variance commentary for June on an as-closed\n"
        f"   basis (after items 1-5 are booked).\n")
    with open(os.path.join(files_dir, "notes", "close_checklist.md"), "w") as f:
        f.write(checklist)
    with open(os.path.join(files_dir, "notes", "prior_accountant_memo.md"), "w") as f:
        f.write(
            f"# Handoff notes\n\nFrom the prior staff accountant, for {p.controller}:\n\n"
            + rnd_txt.choice(_MEMO_FEES)
            + rnd_txt.choice(_MEMO_CODING)
            + rnd_txt.choice(_MEMO_BUDGET)
            + rnd_txt.choice(_MEMO_PAYROLL_JULY
                             if w.plan.has_missing_payroll_accrual
                             else _MEMO_PAYROLL_NEUTRAL))

    # ---- distractors
    with open(os.path.join(files_dir, "notes", "marketing_plan_h2.md"), "w") as f:
        f.write(f"# H2 marketing plan (draft)\n\nGoals for {p.company}: brand refresh,\n"
                f"two webinars per quarter, and a partner co-marketing pilot. Budget\n"
                f"TBD pending Q2 close.\n")
    old_tb = [[a["number"], a["name"], "", ""] for a in w.coa[:8]]
    _write_csv(os.path.join(files_dir, "trial_balance_2025-12-31_ARCHIVE.csv"),
               ["account_number", "account_name", "debit_balance", "credit_balance"],
               old_tb, p.delim)


def _months_amortized(item: dict, through: str) -> int:
    """Whole months of amortization taken from `start` through end of `through`
    month (inclusive)."""
    sy, sm = int(item["start"][:4]), int(item["start"][5:7])
    ty, tm = int(through[:4]), int(through[5:7])
    return max(0, (ty - sy) * 12 + (tm - sm) + 1)
