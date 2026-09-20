"""Discrepancy planning + truth records (DESIGN 4.3).

The engine consults an InjectionPlan while emitting June activity; every
injection produces a TruthRecord that tasks.py turns into gold. All amounts are
integer cents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict


@dataclass
class TruthRecord:
    kind: str                 # e.g. outstanding_check, bank_fee_unrecorded, ...
    amount_c: int             # positive cents
    detail: dict = field(default_factory=dict)

    def to_json(self):
        d = asdict(self)
        d["amount"] = round(self.amount_c / 100.0, 2)
        del d["amount_c"]
        return d


@dataclass
class InjectionPlan:
    n_outstanding_checks: int
    n_deposits_in_transit: int
    n_bank_fees: int
    has_interest: bool
    has_gl_error: bool
    has_missing_depreciation: bool
    has_misposted_expense: bool
    has_second_prepaid: bool
    n_variance_drivers: int
    # P1 generalization: the classic close gaps are only USUALLY present, so a
    # harness must check the GL instead of unconditionally booking them.
    has_missing_payroll_accrual: bool = True
    has_missing_prepaid_amort: bool = True


def plan_injections(rnd: random.Random) -> InjectionPlan:
    return InjectionPlan(
        n_outstanding_checks=rnd.randint(2, 4),
        n_deposits_in_transit=rnd.randint(1, 2),
        n_bank_fees=rnd.randint(1, 2),
        has_interest=rnd.random() < 0.7,
        has_gl_error=rnd.random() < 0.6,
        has_missing_depreciation=rnd.random() < 0.5,
        has_misposted_expense=rnd.random() < 0.8,
        has_second_prepaid=rnd.random() < 0.5,
        n_variance_drivers=rnd.randint(2, 4),
        has_missing_payroll_accrual=rnd.random() < 0.75,
        has_missing_prepaid_amort=rnd.random() < 0.8,
    )


def transpose_cents(amount_c: int, rnd: random.Random) -> int:
    """Return a different amount produced by swapping two adjacent digits of the
    dollar part (classic transposition error). Guaranteed != amount_c."""
    dollars = amount_c // 100
    s = str(dollars)
    idxs = [i for i in range(len(s) - 1) if s[i] != s[i + 1]]
    if not idxs:
        return amount_c + 900  # degenerate (e.g. 1111): fall back to +$9 slip
    i = rnd.choice(idxs)
    swapped = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    out = int(swapped) * 100 + (amount_c % 100)
    return out if out > 0 and out != amount_c else amount_c + 900
