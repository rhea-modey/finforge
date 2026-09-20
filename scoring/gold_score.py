"""Deterministic gold scoring for FinForge (DESIGN section 5).

score_task(category, answer, gold, chart_of_accounts, frozen) -> dict
    {"score": float 0..100, "components": {...}, "details": str}

Pure stdlib. No randomness, no I/O, no LLM calls. Robust to malformed or
missing answer fields: whatever is parseable is scored; answer None -> 0.
"""

from __future__ import annotations

import difflib
import re

_EPS = 1e-9

CATEGORIES = ("bank_rec", "journal_entries", "variance_analysis", "accrual_schedule")

# Top-level keys expected per category (used for wrapper-unwrapping).
_CANONICAL_KEYS = {
    "bank_rec": ("reconciling_items", "bank_statement_ending_balance",
                 "gl_cash_ending_balance", "adjusted_balance",
                 "proposed_journal_entries"),
    "journal_entries": ("entries",),
    "variance_analysis": ("variances", "drivers", "summary"),
    "accrual_schedule": ("schedule", "journal_entry"),
}

_KIND_ALIASES = {
    "gl_amount_error": "gl_error",
    "gl_posting_error": "gl_error",
}


# ---------------------------------------------------------------- primitives

def _num(x):
    """Parse a number robustly. Returns float or None."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(x, str):
        s = x.strip().replace("$", "").replace(",", "").replace(" ", "")
        if not s:
            return None
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]
        try:
            v = float(s)
        except ValueError:
            return None
        return -v if neg else v
    return None


def _close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol + _EPS


def _dict_items(x):
    """Coerce to a list of dicts (drop anything else)."""
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return [i for i in x if isinstance(i, dict)]
    return []


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class AccountResolver:
    """Resolve an account reference (number, name, or 'number name') to a
    canonical key per DESIGN section 5 preamble: exact number match, else
    case-insensitive name match, else difflib ratio >= account_fuzzy_ratio
    against the chart (numbers, names, and 'number name' combos)."""

    def __init__(self, chart_of_accounts, fuzzy_ratio: float = 0.85):
        self.fuzzy_ratio = float(fuzzy_ratio)
        self.by_number: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self.candidates: list[tuple[str, str]] = []  # (lowered candidate, key)
        self._cache: dict[str, str | None] = {}
        for e in chart_of_accounts or []:
            if not isinstance(e, dict):
                continue
            num = str(e.get("number") or e.get("account_number") or "").strip()
            name = str(e.get("name") or e.get("account_name") or "").strip()
            if not num and not name:
                continue
            key = num or name  # canonical identity: the account number
            if num:
                self.by_number.setdefault(num, key)
            if name:
                self.by_name.setdefault(name.lower(), key)
            combo = f"{num} {name}".strip().lower()
            for cand in {num.lower(), name.lower(), combo} - {""}:
                self.candidates.append((cand, key))

    def resolve(self, ref) -> str | None:
        if ref is None:
            return None
        s = str(ref).strip()
        if not s:
            return None
        if s not in self._cache:
            self._cache[s] = self._resolve(s)
        return self._cache[s]

    def _resolve(self, s: str) -> str | None:
        if s in self.by_number:
            return self.by_number[s]
        low = s.lower()
        if low in self.by_name:
            return self.by_name[low]
        best_key, best_ratio = None, 0.0
        for cand, key in self.candidates:
            if cand == low:
                return key
            r = _ratio(low, cand)
            if r > best_ratio:
                best_ratio, best_key = r, key
        if best_ratio >= self.fuzzy_ratio:
            return best_key
        return None


# ---------------------------------------------------------- journal entries

def _je_lines(entry, resolver):
    """Flatten a JE dict into scoring lines: (resolved_acct, raw_acct_lower,
    side, abs_amount). A line with both debit and credit yields two lines."""
    out = []
    if not isinstance(entry, dict):
        return out
    for ln in entry.get("lines") or []:
        if not isinstance(ln, dict):
            continue
        ref = ln.get("account", ln.get("acct"))
        acct = resolver.resolve(ref)
        raw = str(ref or "").strip().lower()
        d = _num(ln.get("debit", ln.get("dr")))
        c = _num(ln.get("credit", ln.get("cr")))
        if d is not None and abs(d) > _EPS:
            out.append((acct, raw, "debit", abs(d)))
        if c is not None and abs(c) > _EPS:
            out.append((acct, raw, "credit", abs(c)))
    return out


def _line_match(g, s, tol):
    ga, graw, gside, gamt = g
    sa, sraw, sside, samt = s
    if gside != sside or not _close(gamt, samt, tol):
        return False
    if ga is not None and sa is not None:
        return ga == sa
    # fall back to raw string equality only if resolution failed on a side
    return graw != "" and graw == sraw


def _lines_f1(gold_lines, sub_lines, tol) -> float:
    if not gold_lines and not sub_lines:
        return 1.0
    used = [False] * len(sub_lines)
    m = 0
    for g in gold_lines:
        for j, s in enumerate(sub_lines):
            if not used[j] and _line_match(g, s, tol):
                used[j] = True
                m += 1
                break
    denom = len(gold_lines) + len(sub_lines)
    return (2.0 * m / denom) if denom else 1.0


def _je_f1(gold_je, sub_je, resolver, tol) -> tuple[float, list[float]]:
    """Per DESIGN 5.2: each gold entry scores its best line-level F1 against
    any submitted entry; overall = mean over gold entries. Gold empty -> 1.0
    (vacuous). Returns (f1, per_gold_entry_scores)."""
    gold_entries = _dict_items(gold_je)
    sub_entries = _dict_items(sub_je)
    if not gold_entries:
        return 1.0, []
    sub_lines = [_je_lines(e, resolver) for e in sub_entries]
    scores = []
    for ge in gold_entries:
        gl = _je_lines(ge, resolver)
        best = 0.0
        for sl in sub_lines:
            f1 = _lines_f1(gl, sl, tol)
            if f1 > best:
                best = f1
        scores.append(best)
    return sum(scores) / len(scores), scores


# ----------------------------------------------------------------- bank_rec

def _norm_kind(k) -> str:
    s = str(k or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _KIND_ALIASES.get(s, s)


def _score_bank_rec(answer, gold, resolver, frozen):
    tol = float(frozen.get("amount_tolerance", 0.01))
    w = frozen.get("bank_rec", {})
    w_item = float(w.get("item_f1", 60))
    w_bal = float(w.get("balances", 20))
    w_je = float(w.get("je_f1", 20))

    gold_items = _dict_items(gold.get("reconciling_items"))
    sub_items = _dict_items(answer.get("reconciling_items"))

    # Greedy bipartite match on (kind, amount +/- tol, side); closest amount
    # first. Amounts are compared SIGNED (gold amounts are positive per the
    # schema) and a stated side must agree with gold's side, so flipped signs
    # or flipped bank/book sides do not score.
    used = [False] * len(sub_items)
    matched = 0
    match_notes = []
    for gi in gold_items:
        gk = _norm_kind(gi.get("kind"))
        ga = _num(gi.get("amount"))
        gside = str(gi.get("side") or "").strip().lower()
        best_j, best_d = -1, None
        for j, si in enumerate(sub_items):
            if used[j] or _norm_kind(si.get("kind")) != gk:
                continue
            sa = _num(si.get("amount"))
            if ga is None or sa is None:
                continue
            sside = str(si.get("side") or "").strip().lower()
            if gside and sside and sside != gside:
                continue
            d = abs(sa - ga)
            if d <= tol + _EPS and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            matched += 1
            match_notes.append(f"hit {gk} {ga:.2f}" if ga is not None else f"hit {gk}")
        else:
            match_notes.append(f"MISS {gk} {ga:.2f}" if ga is not None else f"MISS {gk}")
    if not gold_items and not sub_items:
        item_f1 = 1.0
    else:
        denom = len(gold_items) + len(sub_items)
        item_f1 = (2.0 * matched / denom) if denom else 1.0

    bal_fields = ("bank_statement_ending_balance", "gl_cash_ending_balance",
                  "adjusted_balance")
    bal_hits = 0
    for f in bal_fields:
        gv = _num(gold.get(f))
        if gv is None:  # vacuous if gold lacks it
            bal_hits += 1
        elif _close(_num(answer.get(f)), gv, tol):
            bal_hits += 1
    balances = bal_hits / 3.0

    je_f1, _ = _je_f1(gold.get("proposed_journal_entries"),
                      answer.get("proposed_journal_entries"), resolver, tol)

    score = w_item * item_f1 + w_bal * balances + w_je * je_f1
    components = {
        "item_f1": round(item_f1, 4),
        "balances": round(balances, 4),
        "je_f1": round(je_f1, 4),
        "items_matched": matched,
        "items_gold": len(gold_items),
        "items_submitted": len(sub_items),
        "balances_correct": bal_hits,
    }
    details = (
        f"items: {matched}/{len(gold_items)} gold matched "
        f"({len(sub_items)} submitted), F1={item_f1:.3f} [{'; '.join(match_notes)}]; "
        f"balances: {bal_hits}/3 within {tol}; correcting-JE F1={je_f1:.3f}"
    )
    return score, components, details


# ---------------------------------------------------------- journal_entries

def _score_journal_entries(answer, gold, resolver, frozen):
    tol = float(frozen.get("amount_tolerance", 0.01))
    pen = float(frozen.get("journal_entries", {})
                .get("spam_penalty_per_extra_entry", 5))
    gold_entries = _dict_items(gold.get("entries"))
    sub_entries = _dict_items(answer.get("entries"))

    entry_f1, per_entry = _je_f1(gold_entries, sub_entries, resolver, tol)
    extra = max(0, len(sub_entries) - len(gold_entries))
    penalty = extra * pen
    score = max(0.0, min(100.0, entry_f1 * 100.0 - penalty))

    components = {
        "entry_score": round(entry_f1, 4),
        "per_gold_entry_f1": [round(x, 4) for x in per_entry],
        "spam_penalty": penalty,
        "entries_gold": len(gold_entries),
        "entries_submitted": len(sub_entries),
    }
    details = (
        f"mean best line-F1 over {len(gold_entries)} gold entries = {entry_f1:.3f} "
        f"(per-entry: {[round(x, 2) for x in per_entry]}); "
        f"{len(sub_entries)} submitted, spam penalty -{penalty:g} pts"
    )
    return score, components, details


# -------------------------------------------------------- variance_analysis

_WORD_RE = re.compile(r"[a-z0-9]+")


def _gold_drivers(gold, resolver):
    """Extract gold drivers as {acct, amount, keywords}. Prefers a top-level
    'drivers' list (the injected variance_drivers with keyword tags); falls
    back to drivers nested inside gold['variances']."""
    out = []

    def add(d, parent_acct=None):
        if not isinstance(d, dict):
            return
        acct = resolver.resolve(d.get("account")) or parent_acct
        kws = d.get("keywords") or d.get("keyword_tags") or d.get("tags") or []
        if isinstance(kws, str):
            kws = [kws]
        kws = [str(k).strip().lower() for k in kws if str(k).strip()]
        if not kws:
            desc = str(d.get("description") or "").lower()
            kws = [w for w in _WORD_RE.findall(desc) if len(w) >= 4]
        out.append({"acct": acct, "amount": _num(d.get("amount")), "kws": kws})

    top = gold.get("drivers")
    if isinstance(top, list) and top:
        for d in top:
            add(d)
        return out
    for v in _dict_items(gold.get("variances")):
        pa = resolver.resolve(v.get("account"))
        for d in v.get("drivers") or []:
            add(d, pa)
    return out


def _sub_drivers(answer, resolver):
    out = []
    for v in _dict_items(answer.get("variances")):
        pa = resolver.resolve(v.get("account"))
        for d in v.get("drivers") or []:
            if not isinstance(d, dict):
                continue
            out.append({
                "acct": resolver.resolve(d.get("account")) or pa,
                "amount": _num(d.get("amount")),
                "desc": str(d.get("description") or "").lower(),
            })
    return out


def _score_variance(answer, gold, resolver, frozen):
    tol = float(frozen.get("amount_tolerance", 0.01))
    w = frozen.get("variance_analysis", {})
    w_rec = float(w.get("driver_recall", 50))
    w_num = float(w.get("numbers_accuracy", 25))
    w_prec = float(w.get("driver_precision", 25))
    pct = float(w.get("driver_amount_pct_tol", 0.10))

    gd = _gold_drivers(gold, resolver)
    sd = _sub_drivers(answer, resolver)

    # Greedy bipartite: a submitted driver hits a gold driver if account
    # resolves to the same, amount within +/- pct, and any gold keyword
    # appears in the submitted description (case-insensitive).
    used = [False] * len(sd)
    matched = 0
    for g in gd:
        for j, sub in enumerate(sd):
            if used[j]:
                continue
            if g["acct"] is None or sub["acct"] != g["acct"]:
                continue
            ga, sa = g["amount"], sub["amount"]
            if ga is None or sa is None:
                continue
            amt_tol = pct * abs(ga) if abs(ga) > _EPS else tol
            if abs(sa - ga) > amt_tol + _EPS:
                continue
            if g["kws"] and not any(k in sub["desc"] for k in g["kws"]):
                continue
            used[j] = True
            matched += 1
            break

    recall = (matched / len(gd)) if gd else 1.0
    if sd:
        precision = matched / len(sd)
    else:
        precision = 1.0 if not gd else 0.0

    # numbers_accuracy over gold-flagged accounts
    gold_rows = [v for v in _dict_items(gold.get("variances"))
                 if resolver.resolve(v.get("account")) is not None]
    sub_rows = _dict_items(answer.get("variances"))
    sub_by_acct = {}
    for v in sub_rows:
        a = resolver.resolve(v.get("account"))
        if a is not None and a not in sub_by_acct:
            sub_by_acct[a] = v
    num_hits = 0
    for gv in gold_rows:
        a = resolver.resolve(gv.get("account"))
        sv = sub_by_acct.get(a)
        if sv is None:
            continue
        ok = True
        for f in ("actual", "budget"):
            gvv = _num(gv.get(f))
            if gvv is None:
                continue  # vacuous if gold lacks the field
            if not _close(_num(sv.get(f)), gvv, tol):
                ok = False
                break
        # variance: either signed convention is accepted PROVIDED the stated
        # direction agrees with gold; with no stated direction the signed
        # value must match gold's (variance = actual - budget).
        gvv = _num(gv.get("variance"))
        if ok and gvv is not None:
            svv = _num(sv.get("variance"))
            gdir = str(gv.get("direction") or "").strip().lower()
            sdir = str(sv.get("direction") or "").strip().lower()
            if svv is None:
                ok = False
            elif gdir and sdir:
                ok = _close(abs(svv), abs(gvv), tol) and sdir == gdir
            else:
                ok = _close(svv, gvv, tol)
        if ok:
            num_hits += 1
    numbers_accuracy = (num_hits / len(gold_rows)) if gold_rows else 1.0

    score = w_rec * recall + w_num * numbers_accuracy + w_prec * precision
    components = {
        "driver_recall": round(recall, 4),
        "driver_precision": round(precision, 4),
        "numbers_accuracy": round(numbers_accuracy, 4),
        "drivers_matched": matched,
        "drivers_gold": len(gd),
        "drivers_submitted": len(sd),
        "accounts_numbers_correct": num_hits,
        "accounts_gold_flagged": len(gold_rows),
    }
    details = (
        f"drivers: {matched}/{len(gd)} gold hit ({len(sd)} submitted) -> "
        f"recall={recall:.3f} precision={precision:.3f}; "
        f"numbers: {num_hits}/{len(gold_rows)} gold-flagged accounts with "
        f"actual/budget/variance all within {tol}"
    )
    return score, components, details


# --------------------------------------------------------- accrual_schedule

_ROW_FIELDS = (
    ("opening_balance", "beginning_balance", "opening"),
    ("additions", "addition"),
    ("amortization", "amortization_expense", "amort"),
    ("closing_balance", "ending_balance", "closing"),
)


def _row_name(r) -> str:
    return str(r.get("item") or r.get("name") or r.get("description") or "").strip().lower()


def _name_similarity(a: str, b: str) -> float:
    """max(difflib ratio, overlap of >=4-char word tokens) — so a row named
    with the GL memo's wording still matches a paraphrase that keeps the
    distinctive tokens (vendor name, 'software', 'license', ...)."""
    r = _ratio(a, b)
    ta = {w for w in _WORD_RE.findall(a) if len(w) >= 4}
    tb = {w for w in _WORD_RE.findall(b) if len(w) >= 4}
    if ta and tb:
        common = len(ta & tb)
        if common >= 2:
            r = max(r, 2.0 * common / (len(ta) + len(tb)))
    return r


def _row_field(r, keys):
    for k in keys:
        if k in r:
            return _num(r.get(k))
    return None


def _score_accrual(answer, gold, resolver, frozen):
    tol = float(frozen.get("amount_tolerance", 0.01))
    w = frozen.get("accrual_schedule", {})
    w_rows = float(w.get("rows", 70))
    w_je = float(w.get("je_f1", 30))
    fuzzy = float(w.get("item_fuzzy_ratio", 0.6))

    pen_per = float(w.get("spam_penalty_per_extra_row", 5))
    gold_rows = _dict_items(gold.get("schedule"))
    sub_rows = _dict_items(answer.get("schedule"))

    # Greedy bipartite by descending fuzzy name ratio (>= fuzzy).
    cands = []
    for i, g in enumerate(gold_rows):
        gn = _row_name(g)
        for j, s in enumerate(sub_rows):
            r = _name_similarity(gn, _row_name(s))
            if r >= fuzzy:
                cands.append((-r, i, j))
    cands.sort()
    gused, sused, pairs = set(), set(), {}
    for negr, i, j in cands:
        if i in gused or j in sused:
            continue
        gused.add(i)
        sused.add(j)
        pairs[i] = j

    row_scores = []
    row_notes = []
    for i, g in enumerate(gold_rows):
        if i not in pairs:
            row_scores.append(0.0)
            row_notes.append(f"'{_row_name(g)}': no name match")
            continue
        s = sub_rows[pairs[i]]
        hits = sum(1 for keys in _ROW_FIELDS
                   if _close(_row_field(g, keys), _row_field(s, keys), tol))
        row_scores.append(hits / 4.0)
        row_notes.append(f"'{_row_name(g)}': {hits}/4 numbers")
    row_score = (sum(row_scores) / len(row_scores)) if row_scores else 1.0

    je_f1, _ = _je_f1(gold.get("journal_entry"), answer.get("journal_entry"),
                      resolver, tol)

    extra = max(0, len(sub_rows) - len(gold_rows))
    penalty = extra * pen_per   # spam guard, mirrors journal_entries
    score = max(0.0, w_rows * row_score + w_je * je_f1 - penalty)
    components = {
        "row_score": round(row_score, 4),
        "je_f1": round(je_f1, 4),
        "rows_matched": len(pairs),
        "rows_gold": len(gold_rows),
        "rows_submitted": len(sub_rows),
        "per_row": [round(x, 4) for x in row_scores],
        "spam_penalty": penalty,
    }
    details = (
        f"rows: {len(pairs)}/{len(gold_rows)} name-matched "
        f"(fuzzy>={fuzzy}), row_score={row_score:.3f} "
        f"[{'; '.join(row_notes)}]; amortization-JE F1={je_f1:.3f}; "
        f"{len(sub_rows)} rows submitted, spam penalty -{penalty:g} pts"
    )
    return score, components, details


# ------------------------------------------------------------------- public

def _unwrap(d, category):
    """If the deliverable is nested under a wrapper key, unwrap it."""
    if not isinstance(d, dict):
        return {}
    keys = _CANONICAL_KEYS.get(category, ())
    if any(k in d for k in keys):
        return d
    for wrap in ("answer", "answer_json", "gold", "deliverable"):
        inner = d.get(wrap)
        if isinstance(inner, dict) and any(k in inner for k in keys):
            return inner
    return d


def score_task(category: str, answer: dict | None, gold: dict,
               chart_of_accounts: list[dict], frozen: dict) -> dict:
    """DESIGN section 14 pinned interface. Deterministic 0..100 gold score."""
    frozen = frozen if isinstance(frozen, dict) else {}
    category = str(category or "").strip()
    if not isinstance(answer, dict):
        return {"score": 0.0, "components": {},
                "details": f"{category}: no parseable answer (None or non-dict) -> 0"}

    resolver = AccountResolver(chart_of_accounts,
                               float(frozen.get("account_fuzzy_ratio", 0.85)))
    gold_d = _unwrap(gold if isinstance(gold, dict) else {}, category)
    ans_d = _unwrap(answer, category)

    try:
        if category == "bank_rec":
            score, components, details = _score_bank_rec(ans_d, gold_d, resolver, frozen)
        elif category == "journal_entries":
            score, components, details = _score_journal_entries(ans_d, gold_d, resolver, frozen)
        elif category == "variance_analysis":
            score, components, details = _score_variance(ans_d, gold_d, resolver, frozen)
        elif category == "accrual_schedule":
            score, components, details = _score_accrual(ans_d, gold_d, resolver, frozen)
        else:
            return {"score": 0.0, "components": {},
                    "details": f"unknown category '{category}' -> 0"}
    except Exception as e:  # scoring must never crash the driver
        return {"score": 0.0, "components": {},
                "details": f"{category}: scoring error {e!r} -> 0"}

    score = max(0.0, min(100.0, float(score)))
    return {"score": round(score, 2), "components": components, "details": details}
