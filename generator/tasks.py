"""Task instantiation, gold answers, and auto-rubrics (DESIGN 4.5, 5)."""

from __future__ import annotations

from .engine import World, d2, june_actual, reconciliation_gold
from .exporters import _months_amortized

SUBMIT_NOTE = ("\n\nSubmit your final answer with the `submit_answer` tool. "
               "`answer_json` must match the schema above exactly. "
               "Put narrative in `notes`.")

SCHEMAS = {
    "bank_rec": '''{
  "bank_statement_ending_balance": <number>,
  "gl_cash_ending_balance": <number>,
  "adjusted_balance": <number>,
  "reconciling_items": [
    {"kind": "outstanding_check|deposit_in_transit|bank_fee_unrecorded|interest_unrecorded|gl_error|other",
     "amount": <positive number>, "date": "YYYY-MM-DD or null", "ref": "<string or null>",
     "description": "<string>", "side": "bank|book"}
  ],
  "proposed_journal_entries": [
    {"memo": "<string>", "date": "YYYY-MM-DD",
     "lines": [{"account": "<number and/or name>", "debit": <number>, "credit": <number>}]}
  ]
}''',
    "journal_entries": '''{
  "entries": [
    {"memo": "<string>", "date": "YYYY-MM-DD",
     "lines": [{"account": "<number and/or name>", "debit": <number>, "credit": <number>}]}
  ]
}''',
    "variance_analysis": '''{
  "variances": [
    {"account": "<number and/or name>", "actual": <number>, "budget": <number>,
     "variance": <number: actual minus budget>, "direction": "favorable|unfavorable",
     "drivers": [{"description": "<string naming the specific initiative/vendor>",
                  "amount": <positive number>, "evidence": ["<file path>"]}]}
  ],
  "summary": "<string>"
}''',
    "accrual_schedule": '''{
  "schedule": [
    {"item": "<string>", "opening_balance": <number>, "additions": <number>,
     "amortization": <number>, "closing_balance": <number>}
  ],
  "journal_entry": {"memo": "<string>", "date": "YYYY-MM-DD",
    "lines": [{"account": "<number and/or name>", "debit": <number>, "credit": <number>}]},
  "journal_entry_status": "already_booked|to_book"
}''',
}


def _prompt(category: str, w: World) -> str:
    p = w.profile
    head = f"You are closing the June 2026 books for {p.company} ({p.city}). "
    body = {
        "bank_rec": (
            "Reconcile the operating cash account (GL account 1000) to the June "
            "bank statement. Identify every reconciling item, state the bank "
            "statement ending balance, the GL cash balance at 6/30, and the "
            "adjusted (true) cash balance both sides tie to, and propose the "
            "correcting journal entries the books need — one correcting entry "
            "per book-side reconciling item."),
        "journal_entries": (
            "Work the month-end close checklist in notes/close_checklist.md and "
            "book the missing June journal entries (accruals, amortization, "
            "depreciation if unposted, and any reclasses needed). Only propose "
            "entries the books actually need — check the GL before booking. "
            "Book all prepaid amortization as ONE combined entry, and accrue "
            "payroll at full gross wages to accrued liabilities. Cash/bank-side "
            "corrections (bank fees, interest, transposition fixes) are handled "
            "in the separate bank reconciliation task — do not book them here."),
        "variance_analysis": (
            "Prepare June budget-vs-actual variance commentary on an as-closed "
            "basis: assume the standard checklist close entries (payroll "
            "accrual, prepaid amortization, depreciation, reclasses) are booked "
            "before computing June actuals. For each P&L account with a "
            "material remaining gap versus budget, report actual, budget, "
            "variance (actual minus budget), direction (unfavorable when the "
            "gap increases expense or reduces revenue versus budget), and the "
            "specific drivers that explain the gap, citing the underlying "
            "documents."),
        "accrual_schedule": (
            "Prepare the June 2026 prepaid expense roll-forward schedule "
            "(opening balance, additions, June amortization, closing balance for "
            "each prepaid item) and the June amortization journal entry — "
            "report the entry as booked if the GL already records it, or the "
            "entry that must be booked if it is missing (the schedule's "
            "supporting entry either way; do not propose double-posting). "
            "Convention: a full month of amortization is taken in the month a "
            "prepaid item begins."),
    }[category]
    return (head + body +
            "\n\nAll company records are files in your working directory."
            "\n\nDeliverable JSON schema:\n" + SCHEMAS[category] + SUBMIT_NOTE)


# ------------------------------------------------------------------ gold

def _gold_journal_entries(w: World) -> dict:
    entries = []
    for t in w.truth:
        if t.kind == "missing_payroll_accrual":
            entries.append({
                "memo": f"Accrue payroll for period {t.detail['period']} paid "
                        f"{t.detail['pay_date']}",
                "date": "2026-06-30",
                "lines": [
                    {"account": "6000 Salaries Expense", "debit": d2(t.amount_c),
                     "credit": 0},
                    {"account": "2100 Accrued Liabilities", "debit": 0,
                     "credit": d2(t.amount_c)}]})
    amort_lines = []
    amort_missing = any(t.kind == "missing_prepaid_amort" for t in w.truth)
    for item in (w.prepaid if amort_missing else []):
        if item["start"] > "2026-06-30":
            continue
        m = item["monthly_c"]
        amort_lines.append({"account": f"{item['expense_acct']} "
                                       f"{w.acct_name(item['expense_acct'])}",
                            "debit": d2(m), "credit": 0})
        amort_lines.append({"account": f"{item['acct']} {w.acct_name(item['acct'])}",
                            "debit": 0, "credit": d2(m)})
    if amort_lines:
        entries.append({"memo": "June prepaid amortization", "date": "2026-06-30",
                        "lines": amort_lines})
    for t in w.truth:
        if t.kind == "missing_depreciation":
            entries.append({
                "memo": "June depreciation", "date": "2026-06-30",
                "lines": [
                    {"account": "6800 Depreciation Expense",
                     "debit": d2(t.amount_c), "credit": 0},
                    {"account": "1590 Accumulated Depreciation", "debit": 0,
                     "credit": d2(t.amount_c)}]})
        elif t.kind == "misposted_expense":
            entries.append({
                "memo": f"Reclass {t.detail['bill_id']} ({t.detail['vendor']}) "
                        f"to correct account",
                "date": "2026-06-30",
                "lines": [
                    {"account": f"{t.detail['correct_acct']} "
                                f"{w.acct_name(t.detail['correct_acct'])}",
                     "debit": d2(t.amount_c), "credit": 0},
                    {"account": f"{t.detail['posted_acct']} "
                                f"{w.acct_name(t.detail['posted_acct'])}",
                     "debit": 0, "credit": d2(t.amount_c)}]})
    return {"entries": entries}


def _gold_variance(w: World) -> dict:
    by_acct: dict[str, list] = {}
    for t in w.truth:
        if t.kind == "variance_driver":
            by_acct.setdefault(t.detail["account"], []).append(t)
    variances, top_drivers = [], []
    adjust = getattr(w, "june_adjust", {}) or {}
    for acct in sorted(by_acct):
        # As-closed basis (same counterfactual the budget uses): checklist
        # entries — reclasses, missing amort/dep/payroll — treated as booked.
        actual = june_actual(w, acct) + adjust.get(acct, 0)
        budget = w.budget[acct][6]
        var = actual - budget
        drivers = []
        for t in by_acct[acct]:
            base = (t.detail.get("desc")
                    or f"One-time {t.detail['campaign']} initiative")
            desc = (f"{base} billed by "
                    f"{t.detail['vendor']} ({t.detail['bill_id']})")
            drivers.append({"description": desc, "amount": d2(t.amount_c),
                            "evidence": [f"bills/{t.detail['bill_id']}.txt"]})
            top_drivers.append({"account": f"{acct} {w.acct_name(acct)}",
                                "amount": d2(t.amount_c),
                                "keywords": t.detail["keywords"]})
        variances.append({"account": f"{acct} {w.acct_name(acct)}",
                          "actual": d2(actual), "budget": d2(budget),
                          "variance": d2(var),
                          "direction": "unfavorable" if var > 0 else "favorable",
                          "drivers": drivers})
    return {"variances": variances, "drivers": top_drivers,
            "summary": "June overspend is concentrated in unbudgeted one-time "
                       "initiatives; see drivers."}


def _gold_accrual(w: World) -> dict:
    rows, je_lines = [], []
    for item in w.prepaid:
        if item["start"] > "2026-06-30":
            continue
        m = item["monthly_c"]
        if item["start"] >= "2026-06-01":
            opening, additions = 0, item["premium_c"]
        else:
            taken = _months_amortized(item, "2026-05-31")
            opening, additions = item["premium_c"] - taken * m, 0
        closing = opening + additions - m
        rows.append({"item": item["item"], "opening_balance": d2(opening),
                     "additions": d2(additions), "amortization": d2(m),
                     "closing_balance": d2(closing)})
        je_lines.append({"account": f"{item['expense_acct']} "
                                    f"{w.acct_name(item['expense_acct'])}",
                         "debit": d2(m), "credit": 0})
        je_lines.append({"account": f"{item['acct']} {w.acct_name(item['acct'])}",
                         "debit": 0, "credit": d2(m)})
    status = ("to_book"
              if any(t.kind == "missing_prepaid_amort" for t in w.truth)
              else "already_booked")
    return {"schedule": rows,
            "journal_entry": {"memo": "June prepaid amortization",
                              "date": "2026-06-30", "lines": je_lines},
            "journal_entry_status": status}


def build_tasks(w: World) -> list[dict]:
    """Returns [{task_id, category, prompt, gold, rubric}]."""
    wid = w.profile.world_id
    out = []
    golds = {
        "bank_rec": reconciliation_gold(w),
        "journal_entries": _gold_journal_entries(w),
        "variance_analysis": _gold_variance(w),
        "accrual_schedule": _gold_accrual(w),
    }
    for cat in ("bank_rec", "journal_entries", "variance_analysis",
                "accrual_schedule"):
        out.append({"task_id": f"{wid}-{cat}", "category": cat,
                    "prompt": _prompt(cat, w), "gold": golds[cat],
                    "rubric": _rubric(cat, golds[cat])})
    return out


def _rubric(cat: str, gold: dict) -> list[str]:
    r = []
    if cat == "bank_rec":
        for it in gold["reconciling_items"]:
            r.append(f"Identifies {it['kind']} of ${it['amount']:,.2f}"
                     + (f" ({it['ref']})" if it.get("ref") else ""))
        r.append(f"States adjusted cash balance of ${gold['adjusted_balance']:,.2f}")
        r.append("Proposes correcting entries for all book-side items")
    elif cat == "journal_entries":
        for e in gold["entries"]:
            amt = max(l["debit"] or l["credit"] for l in e["lines"])
            r.append(f"Books: {e['memo']} (${amt:,.2f})")
    elif cat == "variance_analysis":
        for v in gold["variances"]:
            r.append(f"Flags {v['account']} variance of ${v['variance']:,.2f} "
                     f"({v['direction']})")
            for dr in v["drivers"]:
                r.append(f"Attributes ${dr['amount']:,.2f} to: {dr['description']}")
    else:
        for row in gold["schedule"]:
            r.append(f"Rolls forward '{row['item']}' to closing balance "
                     f"${row['closing_balance']:,.2f}")
        r.append("Provides the June amortization journal entry")
    return r
