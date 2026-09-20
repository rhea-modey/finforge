"""Company profiles + industry templates + per-world surface variation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from . import names

INDUSTRY_ORDER_TRAIN = ["saas", "ecommerce", "wholesale", "services"]
HELDOUT_INDUSTRIES = {"h01": "saas", "h02": "ecommerce", "h03": "services",
                      "h04": "manufacturing", "h05": "clinic", "h06": "manufacturing"}

# Industry templates: revenue account names + which optional accounts exist +
# the flavored recurring vendor spend.
INDUSTRY = {
    "saas": {"rev1": "Subscription Revenue", "rev2": "Professional Services Revenue",
             "has_inventory": False, "flavor_expense": ("6300", "Software Subscriptions")},
    "ecommerce": {"rev1": "Product Sales", "rev2": "Shipping Income",
                  "has_inventory": True, "flavor_expense": ("5100", "Freight & Shipping")},
    "wholesale": {"rev1": "Wholesale Sales", "rev2": "Delivery Fees",
                  "has_inventory": True, "flavor_expense": ("5100", "Freight & Shipping")},
    "services": {"rev1": "Consulting Revenue", "rev2": "Retainer Revenue",
                 "has_inventory": False, "flavor_expense": ("6600", "Professional Fees")},
    "manufacturing": {"rev1": "Product Revenue", "rev2": "Tooling & Setup Fees",
                      "has_inventory": True, "flavor_expense": ("5200", "Shop Supplies")},
    "clinic": {"rev1": "Patient Service Revenue", "rev2": "Lab & Imaging Revenue",
               "has_inventory": False, "flavor_expense": ("6350", "Medical Supplies")},
}

# Surface variation pools (seeded per world).
COL = {
    "date": ["date", "txn_date", "posting_date", "Date"],
    "account": ["account", "gl_account", "Account"],
    "debit": ["debit", "dr", "Debit"],
    "credit": ["credit", "cr", "Credit"],
    "memo": ["memo", "description", "Memo"],
    "bank_desc": ["description", "details", "transaction_description"],
    "amount": ["amount", "amt", "Amount"],
}
DATE_FMT = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"]
DELIM = [",", ";"]
BANK_FILE = ["bank_statement_jun2026.csv", "first_national_stmt_062026.csv",
             "statement_june_2026.csv"]
GL_FILE = ["general_ledger.csv", "gl_detail_q2.csv", "ledger_export.csv"]


@dataclass
class CompanyProfile:
    world_id: str
    split: str
    industry: str
    company: str
    seed: int
    scale: float                      # monthly revenue scale multiplier
    delim: str = ","
    date_fmt: str = "%Y-%m-%d"
    cols: dict = field(default_factory=dict)
    bank_file: str = "bank_statement_jun2026.csv"
    gl_file: str = "general_ledger.csv"
    customers: list = field(default_factory=list)
    vendors: list = field(default_factory=list)
    controller: str = ""
    city: str = ""


def world_seed(master_seed: int, split: str, world_id: str) -> int:
    h = hashlib.sha256(f"{master_seed}:{split}:{world_id}".encode()).hexdigest()
    return int(h[:12], 16)


def _industry_of(split: str, world_id: str) -> str:
    if split == "train":
        return INDUSTRY_ORDER_TRAIN[(int(world_id[1:]) - 1)
                                    % len(INDUSTRY_ORDER_TRAIN)]
    return HELDOUT_INDUSTRIES[world_id]


def _company_for(split: str, world_id: str, industry: str) -> str:
    """Collision-free company assignment (DESIGN 4.1 'unseen companies'):
    walk the fixed global world order and give each world the next unused
    name from its industry pool — so no heldout world can reuse a train
    company. Pools are sized to cover every world."""
    seq = ([("train", f"w{i:02d}") for i in range(1, 31)]
           + [("heldout", f"h{i:02d}") for i in range(1, 7)])
    ordinal = 0
    for s, wid in seq:
        if (s, wid) == (split, world_id):
            break
        if _industry_of(s, wid) == industry:
            ordinal += 1
    pool = names.COMPANY_NAMES[industry]
    assert ordinal < len(pool), f"company pool too small for {industry}"
    return pool[ordinal]


def make_profile(master_seed: int, split: str, world_id: str) -> CompanyProfile:
    industry = _industry_of(split, world_id)
    seed = world_seed(master_seed, split, world_id)
    rnd = random.Random(seed)
    company = _company_for(split, world_id, industry)
    prof = CompanyProfile(
        world_id=world_id, split=split, industry=industry, company=company,
        seed=seed, scale=rnd.uniform(0.7, 1.8),
        delim=names.pick(rnd, DELIM),
        date_fmt=names.pick(rnd, DATE_FMT),
        cols={k: names.pick(rnd, v) for k, v in sorted(COL.items())},
        bank_file=names.pick(rnd, BANK_FILE),
        gl_file=names.pick(rnd, GL_FILE),
        customers=names.pick(rnd, names.CUSTOMER_POOL, 8),
        vendors=names.pick(rnd, names.VENDOR_POOL, 9),
        controller=names.pick(rnd, names.PEOPLE),
        city=names.pick(rnd, names.CITIES),
    )
    return prof


def chart_of_accounts(profile: CompanyProfile) -> list[dict]:
    ind = INDUSTRY[profile.industry]
    coa = [
        ("1000", "Cash"),
        ("1100", "Accounts Receivable"),
        ("1300", "Prepaid Insurance"),
        ("1310", "Prepaid Software"),
        ("1500", "Equipment"),
        ("1590", "Accumulated Depreciation"),
        ("2000", "Accounts Payable"),
        ("2100", "Accrued Liabilities"),
        ("2300", "Payroll Liabilities"),
        ("3000", "Retained Earnings"),
        ("4000", ind["rev1"]),
        ("4100", ind["rev2"]),
        ("6000", "Salaries Expense"),
        ("6100", "Rent Expense"),
        ("6200", "Advertising & Marketing"),
        ("6400", "Insurance Expense"),
        ("6450", "Software Expense"),
        ("6500", "Utilities Expense"),
        ("6600", "Professional Fees"),
        ("6700", "Office Supplies"),
        ("6800", "Depreciation Expense"),
        ("6900", "Bank Fees"),
        ("7100", "Interest Income"),
    ]
    if ind["has_inventory"]:
        coa.insert(3, ("1400", "Inventory"))
        coa.append(("5000", "Cost of Goods Sold"))
    fnum, fname = ind["flavor_expense"]
    if fnum not in {n for n, _ in coa}:
        coa.append((fnum, fname))
    coa.sort(key=lambda x: x[0])
    return [{"number": n, "name": nm} for n, nm in coa]
