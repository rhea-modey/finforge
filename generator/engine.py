"""Double-entry ledger simulation for one FinForge world (DESIGN 4.2).

All amounts are integer cents. Close month is June 2026; simulation covers
April-June with an opening entry at 2026-04-01. The bank register carries a
cleared date per cash event; June injections (per the InjectionPlan) create the
reconciling differences that gold is built from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import names
from .discrepancies import InjectionPlan, TruthRecord, plan_injections, transpose_cents
from .profile import INDUSTRY, CompanyProfile, chart_of_accounts

CLOSE = "2026-06"


def d2(c: int) -> float:
    return round(c / 100.0, 2)


def day(month: int, dd: int) -> str:
    return f"2026-{month:02d}-{dd:02d}"


@dataclass
class World:
    profile: CompanyProfile
    coa: list = field(default_factory=list)
    postings: list = field(default_factory=list)      # GL as exported (post-injection)
    bank_events: list = field(default_factory=list)   # bank's own view
    invoices: list = field(default_factory=list)
    bills: list = field(default_factory=list)
    payroll_runs: list = field(default_factory=list)
    prepaid: list = field(default_factory=list)
    assets: list = field(default_factory=list)
    budget: dict = field(default_factory=dict)        # {acct_num: {month: cents}}
    truth: list = field(default_factory=list)         # TruthRecords
    plan: InjectionPlan | None = None
    meta: dict = field(default_factory=dict)

    # ---- ledger helpers -------------------------------------------------
    _entry_seq: int = 0

    def acct_name(self, num: str) -> str:
        for a in self.coa:
            if a["number"] == num:
                return a["name"]
        raise KeyError(num)

    def post(self, date: str, memo: str, lines: list[tuple[str, int, int]],
             source: str = "") -> str:
        """lines: [(acct_num, debit_c, credit_c)]. Asserts the entry balances."""
        self._entry_seq += 1
        eid = f"JE-{self._entry_seq:04d}"
        dsum = sum(l[1] for l in lines)
        csum = sum(l[2] for l in lines)
        assert dsum == csum, f"unbalanced entry {eid}: {dsum} != {csum} ({memo})"
        for acct, dr, cr in lines:
            if dr == 0 and cr == 0:
                continue
            self.postings.append({
                "entry_id": eid, "date": date, "account": acct,
                "account_name": self.acct_name(acct),
                "debit_c": dr, "credit_c": cr, "memo": memo, "source": source,
            })
        return eid

    def bank(self, cleared: str, amount_c: int, desc: str, kind: str) -> None:
        """amount_c signed from the bank's perspective (+deposit, -withdrawal)."""
        self.bank_events.append({"cleared": cleared, "amount_c": amount_c,
                                 "desc": desc, "kind": kind})

    def gl_balance(self, acct: str, through: str) -> int:
        s = 0
        for p in self.postings:
            if p["account"] == acct and p["date"] <= through:
                s += p["debit_c"] - p["credit_c"]
        return s


def _mangle(name: str, rnd: random.Random) -> str:
    """Bank-style counterparty mangling: uppercase, punctuation stripped,
    truncated, sometimes vowels dropped from the tail."""
    s = "".join(ch for ch in name.upper() if ch.isalnum() or ch == " ")
    words = s.split()
    s = " ".join(words[:2]) if len(words) > 2 else s
    if rnd.random() < 0.4 and len(s) > 12:
        s = s[:12]
    return s.strip()


def build_world(profile: CompanyProfile) -> World:
    rnd = random.Random(profile.seed + 1)
    w = World(profile=profile, coa=chart_of_accounts(profile))
    w.plan = plan_injections(random.Random(profile.seed + 2))
    ind = INDUSTRY[profile.industry]
    sc = profile.scale
    # Special-role vendors vary per world (no cross-world sentinel names).
    ins_vendor = names.pick(rnd, names.INSURANCE_VENDORS)
    sw_vendor = names.pick(rnd, names.SOFTWARE_VENDORS)

    # ---------------- opening balances (2026-04-01) ----------------------
    m_ins = (rnd.randint(70, 190) * 100)              # monthly insurance amort
    prem = m_ins * 12
    equip = rnd.randint(60_000, 180_000) * 100
    dep_m = equip // 60                               # 5-year straight line
    open_cash = int(rnd.randint(120_000, 320_000) * 100 * sc)
    open_ar = 0
    open_prepaid_ins = prem - 3 * m_ins               # Jan-Mar amortized pre-window
    open_ap = int(rnd.randint(18_000, 45_000) * 100 * sc)
    open_acc_dep = dep_m * rnd.randint(6, 20)
    inv_open = int(rnd.randint(40_000, 90_000) * 100 * sc) if ind["has_inventory"] else 0
    equity = (open_cash + open_ar + open_prepaid_ins + equip + inv_open
              - open_ap - open_acc_dep)
    lines = [("1000", open_cash, 0), ("1300", open_prepaid_ins, 0),
             ("1500", equip, 0), ("1590", 0, open_acc_dep),
             ("2000", 0, open_ap), ("3000", 0, equity)]
    if inv_open:
        lines.append(("1400", inv_open, 0))
    w.post("2026-04-01", "Opening balances", lines, "opening")
    w.bank("2026-03-31", open_cash, "OPENING BALANCE", "opening")
    w.prepaid.append({"item": f"{ins_vendor} annual policy",
                      "vendor": ins_vendor,
                      "acct": "1300", "expense_acct": "6400",
                      "premium_c": prem, "monthly_c": m_ins,
                      "start": "2026-01-01", "term_months": 12})
    w.assets.append({"asset": "Equipment (aggregate)", "cost_c": equip,
                     "monthly_dep_c": dep_m, "method": "SL-60mo"})

    # Pay off part of opening AP in April so AP isn't static.
    ap_pay = open_ap // 2
    w.post(day(4, 6), "Payment on account - opening payables",
           [("2000", ap_pay, 0), ("1000", 0, ap_pay)], "check")
    w.bank(day(4, 9), -ap_pay, "CHECK #1001", "check")

    check_no = 1002
    inv_no = 2200 + rnd.randint(0, 40)
    bill_no = 3100 + rnd.randint(0, 40)
    driver_bills: list[dict] = []
    rent_vendor = names.pick(rnd, names.PROPERTY_VENDORS)

    # Choose variance-driver specs up front (June one-off bills). Accounts are
    # sampled WITHOUT replacement from the distinct driver-account pool so the
    # planned count survives (DESIGN 4.3: 2-4 drivers; the services flavor
    # duplicates 6600, so its pool has 3 distinct accounts).
    # 6300 "Software Subscriptions" excluded: a one-time initiative billed to
    # a subscriptions account reads as a coding error (QA sweep-3 finding).
    # Only accounts where one-off projects are natural (marketing campaigns,
    # professional engagements, industry-flavor projects). 6500 Utilities and
    # 6300 Subscriptions read as coding errors for one-time bills (QA).
    driver_accts = list(dict.fromkeys(
        a for a in ["6200", ind["flavor_expense"][0], "6600"]
        if a not in ("6450", "6700", "6300", "6500")))
    rnd.shuffle(driver_accts)
    driver_specs = []
    for i in range(min(w.plan.n_variance_drivers, len(driver_accts))):
        acct = driver_accts[i]
        camp = names.pick(rnd, names.CAMPAIGN_WORDS)
        tpl = names.pick(rnd, names.DRIVER_DESC_TEMPLATES)
        vend = names.pick(rnd, names.EXPENSE_VENDORS[acct])   # P3b affinity
        driver_specs.append({"acct": acct, "campaign": camp, "vendor": vend,
                             "desc": tpl.format(c=camp),
                             "amount_c": int(rnd.randint(6_000, 18_000) * 100 * sc)})

    # ---------------- monthly activity ----------------------------------
    for month in (4, 5, 6):
        # -- customer invoices -> receipts. Payment lag can roll the receipt
        # into a later month (or past the close), so the 6/30 AR aging is a
        # real, populated retrieval surface (DESIGN 4.4). The first June
        # invoice always pays after close so AR is never empty.
        for inv_i in range(rnd.randint(4, 7)):
            cust = names.pick(rnd, profile.customers)
            amt = int(rnd.randint(4_000, 28_000) * 100 * sc / 100) * 100
            rev = "4000" if rnd.random() < 0.75 else "4100"
            issued = day(month, rnd.randint(1, 24))
            inv_no += 1
            inv = {"invoice_id": f"INV-{inv_no}", "customer": cust, "date": issued,
                   "amount_c": amt, "rev_acct": rev, "paid": None}
            w.invoices.append(inv)
            w.post(issued, f"Invoice {inv['invoice_id']} - {cust}",
                   [("1100", amt, 0), (rev, 0, amt)], inv["invoice_id"])
            if ind["has_inventory"] and rev == "4000":
                cogs = int(amt * rnd.uniform(0.42, 0.58)) // 100 * 100
                w.post(issued, f"COGS - {inv['invoice_id']}",
                       [("5000", cogs, 0), ("1400", 0, cogs)], inv["invoice_id"])
            lag = (rnd.randint(32, 45) if month == 6 and inv_i == 0
                   else rnd.randint(12, 38))
            pd_m, pd_d = month, int(issued[8:]) + lag
            while pd_d > 28:
                pd_m, pd_d = pd_m + 1, pd_d - 28
            if pd_m <= 6:
                paid = day(pd_m, max(1, pd_d))
                inv["paid"] = paid
                w.post(paid, f"Deposit - {cust} {inv['invoice_id']}",
                       [("1000", amt, 0), ("1100", 0, amt)], inv["invoice_id"])
                clear_lag = rnd.randint(0, 2)
                cd = min(28, int(paid[8:]) + clear_lag)
                w.bank(f"{paid[:8]}{cd:02d}", amt,
                       f"DEPOSIT {_mangle(cust, rnd)} {inv['invoice_id'] if rnd.random() < 0.5 else ''}".strip(),
                       "deposit")

        # -- rent
        rent = int(rnd.randint(4_500, 9_000) * 100 * sc / 100) * 100 if month == 4 else rent
        rd = day(month, 3)
        w.post(rd, f"Rent - {rent_vendor}",
               [("6100", rent, 0), ("1000", 0, rent)], "check")
        w.bank(day(month, min(28, 3 + rnd.randint(2, 6))), -rent,
               f"CHECK #{check_no} {_mangle(rent_vendor, rnd)}", "check")
        check_no += 1

        # -- vendor bills (paid by check ~70%, rest open AP)
        n_bills = rnd.randint(4, 6)
        month_bills = []
        for _ in range(n_bills):
            acct = names.pick(rnd, ["6200", "6450", "6500", "6600", "6700",
                                    ind["flavor_expense"][0]])
            vend = names.pick(rnd, names.EXPENSE_VENDORS[acct])  # P3b affinity
            amt = int(rnd.randint(300, 6_500) * 100 * sc / 100) * 100
            bdate = day(month, rnd.randint(2, 22))
            bill_no += 1
            bill = {"bill_id": f"BILL-{bill_no}", "vendor": vend, "date": bdate,
                    "amount_c": amt, "acct": acct, "desc": "", "paid": None}
            month_bills.append(bill)
        # June extras: variance drivers + the misposted bill
        if month == 6:
            for spec in driver_specs:
                bill_no += 1
                b = {"bill_id": f"BILL-{bill_no}", "vendor": spec["vendor"],
                     "date": day(6, rnd.randint(4, 18)),
                     "amount_c": spec["amount_c"], "acct": spec["acct"],
                     "desc": spec["desc"], "paid": None}
                spec["bill"] = b
                month_bills.append(b)
                driver_bills.append(b)
            if w.plan.has_misposted_expense:
                bill_no += 1
                mp_amt = int(rnd.randint(1_800, 4_800) * 100 * sc / 100) * 100
                # P3a: a clearly period-expense marketing service (nothing
                # capitalizable) posted to Office Supplies by mistake.
                mp = {"bill_id": f"BILL-{bill_no}",
                      "vendor": names.pick(rnd, names.EXPENSE_VENDORS["6200"]),
                      "date": day(6, rnd.randint(3, 15)), "amount_c": mp_amt,
                      "acct": "6200", "posted_acct": "6700",
                      "desc": names.pick(rnd, names.MISPOST_DESCS), "paid": None}
                month_bills.append(mp)
                w.truth.append(TruthRecord("misposted_expense", mp_amt, {
                    "bill_id": mp["bill_id"], "vendor": mp["vendor"],
                    "correct_acct": "6200", "posted_acct": "6700"}))
        for bill in sorted(month_bills, key=lambda b: (b["date"], b["bill_id"])):
            post_acct = bill.get("posted_acct", bill["acct"])
            w.post(bill["date"], f"{bill['bill_id']} {bill['vendor']}",
                   [(post_acct, bill["amount_c"], 0), ("2000", 0, bill["amount_c"])],
                   bill["bill_id"])
            w.bills.append(bill)
            if rnd.random() < 0.7:
                pdate = day(month, min(28, int(bill["date"][8:]) + rnd.randint(3, 9)))
                bill["paid"] = pdate
                w.post(pdate, f"Pay {bill['bill_id']} {bill['vendor']} CHK#{check_no}",
                       [("2000", bill["amount_c"], 0), ("1000", 0, bill["amount_c"])],
                       f"CHK-{check_no}")
                bill["check_no"] = check_no
                bill["_pay_date"] = pdate
                if month < 6:
                    # June checks clear (or go outstanding) in the injection pass
                    cd = min(28, int(pdate[8:]) + rnd.randint(2, 7))
                    w.bank(day(month, cd), -bill["amount_c"],
                           f"CHECK #{check_no} {_mangle(bill['vendor'], rnd)}",
                           "check")
                check_no += 1

        # -- payroll (semi-monthly). June run 2 is paid 7/02 -> accrual gap.
        gross = int(rnd.randint(24_000, 52_000) * 100 * sc / 100) * 100
        for half, (pd, period) in enumerate(
                [(15, f"{month:02d}/01-{month:02d}/15"),
                 (28, f"{month:02d}/16-{month:02d}/30")]):
            g = gross + (rnd.randint(-15, 15) * 1000)
            wh = int(g * 0.22) // 100 * 100
            net = g - wh
            run = {"period": f"2026-{period}", "gross_c": g, "withheld_c": wh,
                   "net_c": net}
            if month == 6 and half == 1 and w.plan.has_missing_payroll_accrual:
                run["pay_date"] = "2026-07-02"        # spans period end (unbooked)
                w.payroll_runs.append(run)
                w.truth.append(TruthRecord("missing_payroll_accrual", g, {
                    "period": run["period"], "pay_date": run["pay_date"]}))
                continue
            run["pay_date"] = day(month, pd)
            w.payroll_runs.append(run)
            w.post(run["pay_date"], f"Payroll {run['period']}",
                   [("6000", g, 0), ("2300", 0, wh), ("1000", 0, net)], "payroll")
            w.bank(run["pay_date"], -net, "PAYROLL ACH BATCH", "payroll")

        # -- prepaid amortization (June sometimes missing by injection — P1)
        if month < 6 or not w.plan.has_missing_prepaid_amort:
            w.post(day(month, 28), "Monthly insurance amortization",
                   [("6400", m_ins, 0), ("1300", 0, m_ins)], "amort")
        # -- depreciation (June per plan)
        if month < 6 or not w.plan.has_missing_depreciation:
            w.post(day(month, 28), "Monthly depreciation",
                   [("6800", dep_m, 0), ("1590", 0, dep_m)], "depreciation")

    if w.plan.has_missing_depreciation:
        w.truth.append(TruthRecord("missing_depreciation", dep_m,
                                   {"asset": "Equipment (aggregate)"}))
    if w.plan.has_missing_prepaid_amort:
        w.truth.append(TruthRecord("missing_prepaid_amort", m_ins,
                                   {"item": w.prepaid[0]["item"]}))

    # -- optional second prepaid purchased in June (also unamortized). The
    # gold row name is EXACTLY the GL memo (and the license contract doc
    # carries the same wording), so the item name is derivable from files/.
    if w.plan.has_second_prepaid:
        add = rnd.randint(24, 96) * 100 * 12          # divisible by 12
        pdate = day(6, 5)
        sp_name = f"Prepaid software - 12 month license ({sw_vendor})"
        w.post(pdate, sp_name, [("1310", add, 0), ("1000", 0, add)], "check")
        w.bank(day(6, 8), -add, f"CHECK #{check_no} {_mangle(sw_vendor, rnd)}",
               "check")
        check_no += 1
        w.prepaid.append({"item": sp_name, "vendor": sw_vendor,
                          "acct": "1310", "expense_acct": "6450",
                          "premium_c": add, "monthly_c": add // 12,
                          "start": pdate, "term_months": 12})
        if not w.plan.has_missing_prepaid_amort:
            # amortization was booked this month, so the new item's first
            # month is booked too (full-month convention)
            w.post(day(6, 28), f"Monthly amortization - {sp_name}",
                   [("6450", add // 12, 0), ("1310", 0, add // 12)], "amort")

    # ---------------- June bank-side injections --------------------------
    plan, rnd2 = w.plan, random.Random(profile.seed + 3)

    june_checks = [b for b in w.bills
                   if b.get("_pay_date", "").startswith("2026-06") and b.get("check_no")]
    rnd2.shuffle(june_checks)
    outstanding = june_checks[:plan.n_outstanding_checks]
    # gl_error must not stack on a driver or misposted bill (one injection per
    # bill keeps every task's gold independently derivable).
    gl_err_pool = [b for b in june_checks[plan.n_outstanding_checks:]
                   if b not in driver_bills and not b.get("posted_acct")]

    for b in june_checks:
        if b in outstanding:
            w.truth.append(TruthRecord("outstanding_check", b["amount_c"], {
                "check_no": f"CHK-{b['check_no']}", "vendor": b["vendor"],
                "bill_id": b["bill_id"], "date": b["_pay_date"]}))
            w.bank(f"2026-07-{rnd2.randint(1, 6):02d}", -b["amount_c"],
                   f"CHECK #{b['check_no']} {_mangle(b['vendor'], rnd2)}", "check")
            continue
        bank_amt = b["amount_c"]
        if plan.has_gl_error and gl_err_pool and b is gl_err_pool[0]:
            # The bill document and the bank hold the TRUE amount; the books
            # keyed a transposed figure into BOTH the bill entry and the
            # payment (classic data-entry transposition — consistently wrong
            # in the GL, provable from the bill doc + statement).
            true_amt = b["amount_c"]
            gl_amt = transpose_cents(true_amt, rnd2)
            for p_ in w.postings:
                if p_["source"] in (b["bill_id"], f"CHK-{b['check_no']}"):
                    if p_["debit_c"] == true_amt:
                        p_["debit_c"] = gl_amt
                    if p_["credit_c"] == true_amt:
                        p_["credit_c"] = gl_amt
            w.truth.append(TruthRecord("gl_amount_error", abs(true_amt - gl_amt), {
                "check_no": f"CHK-{b['check_no']}", "bill_id": b["bill_id"],
                "gl_amount": d2(gl_amt), "bank_amount": d2(true_amt),
                "expense_acct": b["acct"],
                "direction": "gl_understated_payment" if true_amt > gl_amt
                             else "gl_overstated_payment"}))
        cd = min(28, int(b["_pay_date"][8:]) + rnd2.randint(2, 7))
        w.bank(f"2026-06-{cd:02d}", -bank_amt,
               f"CHECK #{b['check_no']} {_mangle(b['vendor'], rnd2)}", "check")

    # deposits in transit: convert the latest June deposits to clear in July
    june_deps = [e for e in w.bank_events
                 if e["kind"] == "deposit" and e["cleared"].startswith("2026-06")]
    june_deps.sort(key=lambda e: e["cleared"])
    for e in june_deps[-plan.n_deposits_in_transit:]:
        e["cleared"] = f"2026-07-{rnd2.randint(1, 3):02d}"
        w.truth.append(TruthRecord("deposit_in_transit", e["amount_c"],
                                   {"desc": e["desc"]}))

    # bank-only fees / interest (never posted to GL)
    for i in range(plan.n_bank_fees):
        fee = rnd2.randint(2_500, 6_500)
        w.bank(day(6, 27 + i), -fee, "SERVICE FEE" if i == 0 else "WIRE FEE", "fee")
        w.truth.append(TruthRecord("bank_fee_unrecorded", fee,
                                   {"desc": "SERVICE FEE" if i == 0 else "WIRE FEE"}))
    if plan.has_interest:
        intr = rnd2.randint(1_000, 4_200)
        w.bank(day(6, 30), intr, "INTEREST CREDIT", "interest")
        w.truth.append(TruthRecord("interest_unrecorded", intr, {}))

    # non-AR noise deposit booked in GL as equity contribution (legit, both sides)
    noise = rnd2.randint(5_000, 20_000) * 100
    w.post(day(6, 12), "Owner capital contribution",
           [("1000", noise, 0), ("3000", 0, noise)], "transfer")
    w.bank(day(6, 12), noise, "INCOMING WIRE - MEMBER CONTRIBUTION", "transfer")

    # ---------------- checks against bank + GL constructions -------------
    june_check_events = [e for e in w.bank_events if e["kind"] != "opening"]
    for e in june_check_events:
        assert e["amount_c"] != 0

    # variance drivers -> truth
    for spec in driver_specs:
        b = spec["bill"]
        w.truth.append(TruthRecord("variance_driver", b["amount_c"], {
            "account": spec["acct"], "bill_id": b["bill_id"],
            "vendor": spec["vendor"], "campaign": spec["campaign"],
            "desc": spec["desc"],
            "keywords": [spec["campaign"].lower(),
                         spec["vendor"].split()[0].lower()]}))

    # ---------------- budget ---------------------------------------------
    # June budget is built from the COUNTERFACTUAL complete actuals (the
    # injected omissions/mispostings added back), so the budget file cannot
    # telegraph what is unbooked (a June budget of 0 on an account with
    # Apr/May activity would be a shortcut oracle for the missing entries).
    adjust: dict[str, int] = {}
    if w.plan.has_missing_prepaid_amort:
        adjust["6400"] = m_ins
        for item in w.prepaid[1:]:          # e.g. the June software prepaid
            if item["start"] <= "2026-06-30":
                adjust[item["expense_acct"]] = (
                    adjust.get(item["expense_acct"], 0) + item["monthly_c"])
    if w.plan.has_missing_depreciation:
        adjust["6800"] = adjust.get("6800", 0) + dep_m
    for t in w.truth:
        if t.kind == "missing_payroll_accrual":
            adjust["6000"] = adjust.get("6000", 0) + t.amount_c
        elif t.kind == "misposted_expense":
            ca, pa = t.detail["correct_acct"], t.detail["posted_acct"]
            adjust[ca] = adjust.get(ca, 0) + t.amount_c
            adjust[pa] = adjust.get(pa, 0) - t.amount_c
        elif t.kind == "gl_amount_error":
            # The bank-rec correcting entry hits the bill's expense account;
            # as-closed actuals (and the budget counterfactual) must include
            # it — the expense acct can coincide with a driver account.
            ea = t.detail.get("expense_acct")
            if ea:
                signed = (t.amount_c
                          if t.detail["direction"] == "gl_understated_payment"
                          else -t.amount_c)
                adjust[ea] = adjust.get(ea, 0) + signed
    _build_budget(w, rnd2, adjust)
    # Gold variance actuals must be on the same as-closed basis the task
    # prompt demands (checklist entries booked), so tasks.py needs the same
    # counterfactual adjustments the budget was built from.
    w.june_adjust = adjust
    w.meta = {"m_ins": m_ins, "dep_m": dep_m, "close": CLOSE}
    return w


def _pl_accounts(w: World) -> list[str]:
    return [a["number"] for a in w.coa if a["number"][0] in "456" or a["number"] == "7100"]


def june_actual(w: World, acct: str) -> int:
    s = 0
    for p in w.postings:
        if p["account"] == acct and p["date"].startswith("2026-06"):
            s += p["debit_c"] - p["credit_c"]
    if acct.startswith("4") or acct == "7100":
        s = -s
    return s


def _build_budget(w: World, rnd: random.Random,
                  june_adjust: dict[str, int] | None = None) -> None:
    """Budget = (complete) actuals +/- planning noise. `june_adjust` restores
    the injected June omissions/mispostings so the budget reflects what SHOULD
    have been booked. June noise on non-driver accounts is capped well below
    the smallest driver so drivers stay the only material as-closed gaps."""
    june_adjust = june_adjust or {}
    drivers_by_acct: dict[str, int] = {}
    for t in w.truth:
        if t.kind == "variance_driver":
            drivers_by_acct[t.detail["account"]] = (
                drivers_by_acct.get(t.detail["account"], 0) + t.amount_c)
    noise_cap = (int(min(drivers_by_acct.values()) * 0.35)
                 if drivers_by_acct else None)
    for acct in _pl_accounts(w):
        w.budget[acct] = {}
        for month in (4, 5, 6):
            actual = 0
            for p in w.postings:
                if p["account"] == acct and p["date"].startswith(f"2026-{month:02d}"):
                    actual += p["debit_c"] - p["credit_c"]
            if acct.startswith("4") or acct == "7100":
                actual = -actual
            if month == 6:
                actual += june_adjust.get(acct, 0)
            if month == 6 and acct in drivers_by_acct:
                drv = drivers_by_acct[acct]
                noise = int(drv * rnd.uniform(-0.03, 0.03)) // 100 * 100
                w.budget[acct][month] = max(0, actual - drv + noise)
            else:
                noise = int(actual * rnd.uniform(-0.04, 0.04)) // 100 * 100
                if month == 6 and noise_cap is not None:
                    noise = max(-noise_cap, min(noise_cap, noise))
                w.budget[acct][month] = max(0, actual + noise)


# -------------------- bank statement + reconciliation math ----------------

def bank_statement_june(w: World) -> dict:
    """Rows cleared in June, plus opening/ending balances (bank's view)."""
    opening = sum(e["amount_c"] for e in w.bank_events if e["cleared"] < "2026-06-01")
    rows = sorted((e for e in w.bank_events
                   if "2026-06-01" <= e["cleared"] <= "2026-06-30"),
                  key=lambda e: (e["cleared"], e["desc"], e["amount_c"]))
    ending = opening + sum(r["amount_c"] for r in rows)
    return {"opening_c": opening, "ending_c": ending, "rows": rows}


def reconciliation_gold(w: World) -> dict:
    stmt = bank_statement_june(w)
    gl_cash = w.gl_balance("1000", "2026-06-30")
    items, jes = [], []
    adj_bank = stmt["ending_c"]
    adj_book = gl_cash
    for t in w.truth:
        if t.kind == "outstanding_check":
            items.append({"kind": "outstanding_check", "amount": d2(t.amount_c),
                          "date": t.detail["date"], "ref": t.detail["check_no"],
                          "description": f"Check to {t.detail['vendor']} not yet cleared",
                          "side": "bank"})
            adj_bank -= t.amount_c
        elif t.kind == "deposit_in_transit":
            items.append({"kind": "deposit_in_transit", "amount": d2(t.amount_c),
                          "date": "2026-06-30", "ref": None,
                          "description": f"Deposit in transit: {t.detail['desc']}",
                          "side": "bank"})
            adj_bank += t.amount_c
        elif t.kind == "bank_fee_unrecorded":
            items.append({"kind": "bank_fee_unrecorded", "amount": d2(t.amount_c),
                          "date": "2026-06-30", "ref": None,
                          "description": f"Bank {t.detail.get('desc', 'fee')} not booked",
                          "side": "book"})
            adj_book -= t.amount_c
            jes.append({"memo": "Record June bank fees", "date": "2026-06-30",
                        "lines": [{"account": "6900 Bank Fees",
                                   "debit": d2(t.amount_c), "credit": 0},
                                  {"account": "1000 Cash", "debit": 0,
                                   "credit": d2(t.amount_c)}]})
        elif t.kind == "interest_unrecorded":
            items.append({"kind": "interest_unrecorded", "amount": d2(t.amount_c),
                          "date": "2026-06-30", "ref": None,
                          "description": "Interest earned, not booked", "side": "book"})
            adj_book += t.amount_c
            jes.append({"memo": "Record June interest income", "date": "2026-06-30",
                        "lines": [{"account": "1000 Cash", "debit": d2(t.amount_c),
                                   "credit": 0},
                                  {"account": "7100 Interest Income", "debit": 0,
                                   "credit": d2(t.amount_c)}]})
        elif t.kind == "gl_amount_error":
            d = t.amount_c
            under = t.detail["direction"] == "gl_understated_payment"
            items.append({"kind": "gl_error", "amount": d2(d),
                          "date": "2026-06-30", "ref": t.detail["check_no"],
                          "description": ("GL posted check at transposed amount "
                                          f"({t.detail['gl_amount']} vs bank "
                                          f"{t.detail['bank_amount']})"),
                          "side": "book"})
            adj_book += (-d if under else d)
            ea = t.detail.get("expense_acct", "2000")
            ea_ref = f"{ea} {w.acct_name(ea)}" if ea != "2000" else "2000 Accounts Payable"
            if under:   # books recorded less expense/payment than reality
                jes.append({"memo": f"Correct transposition on {t.detail['check_no']}",
                            "date": "2026-06-30",
                            "lines": [{"account": ea_ref,
                                       "debit": d2(d), "credit": 0},
                                      {"account": "1000 Cash", "debit": 0,
                                       "credit": d2(d)}]})
            else:
                jes.append({"memo": f"Correct transposition on {t.detail['check_no']}",
                            "date": "2026-06-30",
                            "lines": [{"account": "1000 Cash", "debit": d2(d),
                                       "credit": 0},
                                      {"account": ea_ref,
                                       "debit": 0, "credit": d2(d)}]})
    assert adj_bank == adj_book, (
        f"reconciliation does not tie: bank {adj_bank} vs book {adj_book}")
    return {
        "bank_statement_ending_balance": d2(stmt["ending_c"]),
        "gl_cash_ending_balance": d2(gl_cash),
        "adjusted_balance": d2(adj_bank),
        "reconciling_items": items,
        "proposed_journal_entries": jes,
    }
