"""LLM "worldsmith" enrichment pass (DESIGN 15).

Rewrites ONLY prose files — the LLM never touches a number, date, ID, or CSV:

- notes/close_checklist.md        rewrite wording, preserve every obligation
- notes/prior_accountant_memo.md  rewrite wording, preserve every hint
- contracts/insurance_policy.txt  keep insured, term dates, premium, straight-line
- invoices/*.txt, bills/*.txt     keep doc id, names, date, amount line verbatim
                                  (+ campaign word for variance-driver bills,
                                   + exact description for the misposted bill)
- notes/marketing_plan_h2.md      free rewrite (distractor)
- notes/company_background.md     NEW file: backstory, no financial figures

One LLM call per world returning strict JSON {file_path: new_text}; code-level
validation of every preserved token; one retry for failing files; originals
kept on persistent failure. Accepted JSON cached at
gold/{split}/{wid}/enrichment.json (cache present => no LLM call, ever).
files_tree_hash in gold meta.json is refreshed after applying enrichment.

CLI: python3 -m generator.enrich [--worlds-dir worlds] [--gold-dir gold]
        [--model M] [--only wid[,wid]] [--workers 6]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading

# Estimated list price for the datagen model (USD per MTok in/out) — used only
# for the spend report printed at the end.
_PRICE_IN, _PRICE_OUT = 1.25, 10.0

_LOCK = threading.Lock()
_USAGE = {"prompt": 0, "completion": 0, "calls": 0}

MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+\.\d{2}\b|\b\d+\.\d{2}\b")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DOC_ID_RE = re.compile(r"\b(?:INV|BILL|CHK)-\d+\b")

INDUSTRY_WORDS = {
    "saas": ["software", "saas", "platform", "cloud", "subscription"],
    "ecommerce": ["ecommerce", "e-commerce", "online", "retail", "store", "brand"],
    "wholesale": ["wholesale", "distribut", "supplier", "warehouse"],
    "services": ["consult", "service", "client", "agency", "firm", "project"],
    "manufacturing": ["manufactur", "machin", "fabricat", "production", "plant", "shop"],
    "clinic": ["clinic", "medical", "patient", "health", "care", "practice"],
}

CHECKLIST_GROUPS = [
    ["reconcil"], ["bank statement"], ["journal entr"],
    ["payroll"], ["accru"], ["gross"], ["accrued liabilities"], ["july"],
    ["prepaid"], ["amortiz"], ["full month"], ["begin", "start", "commence"],
    ["combined", "single", "one journal", "one entry", "one combined"],
    ["depreciat"], ["fixed asset"],
    ["bill"], ["coding", "coded", "miscoded", "expense account"], ["reclass"],
    ["variance", "budget-vs-actual", "budget vs actual", "budget versus actual"],
    ["budget"], ["as-closed", "as closed"],
]
CHECKLIST_REGEXES = [r"(?m)^\s*%d[.)]" % n for n in range(1, 7)]

MEMO_GROUPS = [
    ["fee", "charge"], ["close"],
    ["bill"],
    ["coding", "coded", "miscoded", "expense account", "wrong account"],
    ["authoritative", "source of truth", "definitive", "govern", "trust the bill",
     "controlling"],
    ["budget"],
    ["one-off", "one off", "unplanned", "never planned", "not planned",
     "unbudgeted", "not in the budget", "never budgeted"],
    ["payroll"], ["july", "register", "pay date", "pay dates", "pays"],
]


# --------------------------------------------------------------------------
# plan construction: which files, what must survive
# --------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _line_value(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def collect_plan(files_dir: str, meta: dict) -> dict[str, dict]:
    """Return {rel_path: spec}. spec keys: original (str|None), tokens (exact
    substrings), groups (case-insensitive alternative lists), regexes,
    no_money (bool), brief (guidance for the LLM)."""
    company = meta["company"]
    industry = meta["industry"]
    driver_bills = {}      # bill_id -> campaign
    mispost_bill = None    # bill_id
    for t in meta["truth_records"]:
        if t["kind"] == "variance_driver":
            driver_bills[t["detail"]["bill_id"]] = t["detail"]["campaign"]
        elif t["kind"] == "misposted_expense":
            mispost_bill = t["detail"]["bill_id"]

    plan: dict[str, dict] = {}

    p = "notes/close_checklist.md"
    plan[p] = {
        "original": _read(os.path.join(files_dir, p)),
        "tokens": [], "groups": CHECKLIST_GROUPS, "regexes": CHECKLIST_REGEXES,
        "no_money": False,
        "brief": ("Rewrite the wording/voice, but PRESERVE all six numbered "
                  "obligations with the same meaning, same order, numbered 1-6. "
                  "Every procedural detail must survive: reconcile cash to the "
                  "June bank statement with one correcting journal entry per "
                  "book-side item; accrue full gross wages for any pay period "
                  "ending after 6/30 paying in July to Accrued Liabilities; "
                  "book June prepaid amortization as ONE combined journal entry "
                  "with a full month taken in the month an item begins; book "
                  "June depreciation per the fixed asset register; verify "
                  "vendor-bill expense coding and reclass anything miscoded; "
                  "budget-vs-actual variance commentary on an as-closed basis."),
    }

    p = "notes/prior_accountant_memo.md"
    plan[p] = {
        "original": _read(os.path.join(files_dir, p)),
        "tokens": [], "groups": MEMO_GROUPS, "regexes": [], "no_money": False,
        "brief": ("Rewrite the voice but preserve every hint: (1) bank's own "
                  "fees/charges appear near statement end and are booked at "
                  "close; (2) June vendor-bill expense coding may be off and "
                  "the documents in bills/ are authoritative; (3) the budget "
                  "was fixed early so one-off June items were never planned "
                  "and explain the big variances; (4) whatever the CURRENT "
                  "CONTENT says about payroll timing — preserve its meaning "
                  "exactly (do NOT claim the second run pays in July unless "
                  "the current content says so)."),
    }

    p = "contracts/insurance_policy.txt"
    orig = _read(os.path.join(files_dir, p))
    tokens = [company]
    broker = _line_value(orig, "Broker:")
    if broker:
        tokens.append(broker)
    tokens += ISO_DATE_RE.findall(orig)
    m = re.search(r"\$[\d,]+\.\d{2}", orig)
    if m:
        tokens.append(m.group(0))
    plan[p] = {
        "original": orig, "tokens": tokens,
        "groups": [["straight-line", "straight line"], ["full month"]],
        "regexes": [], "no_money": False,
        "brief": ("Rewrite as a realistic policy declaration page. KEEP "
                  "verbatim: insured name, broker name, both term dates, the "
                  "exact premium amount, and state that the premium is "
                  "amortized straight-line over the term with a full month "
                  "recognized in each month in force. Do not add any other "
                  "dollar amounts or dates."),
    }

    for sub, id_prefix in (("invoices", "INV"), ("bills", "BILL")):
        d = os.path.join(files_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".txt"):
                continue
            rel = f"{sub}/{fn}"
            orig = _read(os.path.join(files_dir, rel))
            doc_id = fn[:-4]
            lines = orig.splitlines()
            tokens = [doc_id, lines[0]]            # counterparty or company
            other = _line_value(orig, "Bill to:") or _line_value(orig, "To:")
            if other:
                tokens.append(other)
            tokens += ISO_DATE_RE.findall(orig)
            amt_line = next((l for l in lines if l.startswith("Amount")), None)
            if amt_line:
                tokens.append(amt_line)
            groups, extra = [], ""
            if doc_id in driver_bills:
                groups.append([driver_bills[doc_id]])
                extra = (f" This bill is a one-off project: the word "
                         f"'{driver_bills[doc_id]}' and the vendor name must "
                         f"remain in the text.")
            if doc_id == mispost_bill:
                desc = _line_value(orig, "Description:")
                if desc:
                    tokens.append(desc)
                    extra += (" Keep the description line's text verbatim — "
                              "it identifies what was purchased.")
            plan[rel] = {
                "original": orig, "tokens": tokens, "groups": groups,
                "regexes": [], "no_money": False,
                "brief": ("Rewrite as a realistic vendor/customer document "
                          "(letterhead feel, a remit/terms line, a short "
                          "line-item phrase). KEEP verbatim: the document id, "
                          "both party names, the date, and the entire Amount "
                          "line exactly as-is. If you stylize a name in "
                          "UPPERCASE for a letterhead, ALSO write it once in "
                          "its exact original casing. The description must "
                          "keep the same economic meaning and stay consistent "
                          "with the kind of service the vendor name implies. "
                          "Describe a single-period (June) service: NEVER use "
                          "annual/yearly/12-month/subscription-renewal wording "
                          "in a bill. Do not add any other dollar amounts, "
                          "dates, or reference numbers."
                          + extra),
            }

    p = "notes/marketing_plan_h2.md"
    if os.path.exists(os.path.join(files_dir, p)):
        plan[p] = {
            "original": _read(os.path.join(files_dir, p)),
            "tokens": [], "groups": [], "regexes": [], "no_money": False,
            "brief": ("Distractor file — rewrite freely as a plausible H2 "
                      "marketing plan draft for this company. No dollar "
                      "amounts, no dates beyond quarters, no document ids."),
        }

    p = "notes/company_background.md"
    plan[p] = {
        "original": None, "tokens": [company],
        "groups": [INDUSTRY_WORDS[industry]], "regexes": [], "no_money": True,
        "brief": (f"NEW file. Write a 150-250 word backstory for {company}, a "
                  f"{industry} business: founding story, what it sells, team "
                  "flavor, current situation. Mention the company by name. "
                  "The company is an ordinary going concern — do NOT invent "
                  "dramatic events (shutdowns, acquisitions, lawsuits, fraud). "
                  "STRICTLY no financial figures, no dollar amounts, no "
                  "percentages, no document ids. Years are allowed."),
    }
    return plan


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_file(rel: str, new_text: str, spec: dict) -> list[str]:
    """Return list of problems ([] = accepted)."""
    probs = []
    if not isinstance(new_text, str) or not new_text.strip():
        return ["empty or non-string content"]
    low = new_text.lower()
    for tok in spec["tokens"]:
        if tok not in new_text:
            probs.append(f"missing required verbatim token: {tok!r}")
    for group in spec["groups"]:
        if not any(alt.lower() in low for alt in group):
            probs.append(f"missing all of keyword alternatives: {group}")
    for rx in spec["regexes"]:
        if not re.search(rx, new_text):
            probs.append(f"missing required structure (regex {rx!r})")
    orig = spec["original"] or ""
    allowed_money = set(MONEY_RE.findall(orig))
    allowed_dates = set(ISO_DATE_RE.findall(orig))
    allowed_ids = set(DOC_ID_RE.findall(orig))
    for tok in MONEY_RE.findall(new_text):
        if tok not in allowed_money:
            probs.append(f"introduces new numeric amount {tok!r}")
    for tok in ISO_DATE_RE.findall(new_text):
        if tok not in allowed_dates:
            probs.append(f"introduces new date {tok!r}")
    for tok in DOC_ID_RE.findall(new_text):
        if tok not in allowed_ids:
            probs.append(f"introduces new document id {tok!r}")
    if spec["no_money"] and ("$" in new_text or "%" in new_text):
        probs.append("financial figures forbidden in this file")
    # P3c: bills are period expenses — multi-period wording would create a
    # capitalization (prepaid) ambiguity that contradicts gold.
    if rel.startswith("bills/") and re.search(
            r"annual|yearly|12[- ]month|multi[- ]?year|per year",
            new_text, re.IGNORECASE):
        probs.append("bills must not describe annual/multi-period arrangements "
                     "(capitalization ambiguity)")
    # Period integrity: an invoice/bill must not name a different month than
    # its own date (an April invoice describing "June services" contradicts
    # the GL's recognition period — QA sweep-4 finding).
    if rel.startswith(("bills/", "invoices/")):
        own_months = {int(d[5:7]) for d in ISO_DATE_RE.findall(orig)}
        month_names = ["january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november",
                       "december"]
        for num, name in enumerate(month_names, start=1):
            if num not in own_months and re.search(rf"\b{name}\b", low):
                probs.append(f"names month '{name}' but the document is dated "
                             f"in month(s) {sorted(own_months)} — period "
                             f"confusion")
                break
    if len(new_text) > 4000:
        probs.append("file too long (>4000 chars)")
    return probs


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------

def _client():
    from openai import OpenAI
    key = os.environ.get("DATAGEN_API_KEY")
    if not key:
        raise SystemExit("DATAGEN_API_KEY not set (source scripts/env.sh)")
    return OpenAI(api_key=key, base_url=os.environ.get("DATAGEN_BASE_URL") or None)


def _build_prompt(meta: dict, plan: dict[str, dict],
                  problems: dict[str, list[str]] | None = None) -> str:
    company, industry = meta["company"], meta["industry"]
    parts = [
        f"Company: {company} — a small {industry} business closing June 2026.",
        "You are enriching a synthetic accounting exercise. Rewrite the files",
        "below so they read like authentic, characterful business documents",
        "(varied voices: a terse controller, a chatty predecessor, boilerplate",
        "legalese, real-world letterheads). RULES, non-negotiable:",
        "- NEVER invent, alter, or drop a number, dollar amount, date, or",
        "  document id. Every 'KEEP verbatim' item must appear character-for-",
        "  character in your rewrite.",
        "- Plain text / markdown only. Keep each file under 3500 characters.",
        "- Return STRICT JSON: one object whose keys are EXACTLY the file",
        "  paths listed below and whose values are the complete new file",
        "  contents. No extra keys, no commentary.",
        "",
    ]
    for rel, spec in sorted(plan.items()):
        parts.append(f"### FILE: {rel}")
        parts.append(f"BRIEF: {spec['brief']}")
        if spec["tokens"]:
            parts.append("KEEP verbatim (character-for-character): "
                         + " | ".join(spec["tokens"]))
        if spec["original"] is not None:
            parts.append("CURRENT CONTENT:\n" + spec["original"].rstrip())
        else:
            parts.append("CURRENT CONTENT: (new file — write from scratch)")
        if problems and rel in problems:
            parts.append("YOUR PREVIOUS ATTEMPT WAS REJECTED because: "
                         + "; ".join(problems[rel])
                         + " — fix these exactly.")
        parts.append("")
    return "\n".join(parts)


def _call_llm(client, model: str, prompt: str, max_tokens: int) -> dict:
    last_err = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            with _LOCK:
                _USAGE["prompt"] += r.usage.prompt_tokens
                _USAGE["completion"] += r.usage.completion_tokens
                _USAGE["calls"] += 1
            if r.choices[0].finish_reason == "length":
                max_tokens = min(max_tokens * 2, 32000)
                last_err = "truncated (finish_reason=length)"
                continue
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError as e:
            last_err = f"bad JSON: {e}"
        except Exception as e:            # transient API errors
            last_err = f"{type(e).__name__}: {e}"
            import time
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


# --------------------------------------------------------------------------
# apply + hash bookkeeping
# --------------------------------------------------------------------------

def apply_enrichment(files_dir: str, cache: dict) -> None:
    """Deterministically write cached enriched texts into files_dir."""
    for rel, text in sorted(cache["files"].items()):
        path = os.path.join(files_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else text + "\n")


def _refresh_meta_hash(gdir: str, files_dir: str) -> None:
    from .build import tree_hash
    mpath = os.path.join(gdir, "meta.json")
    meta = json.load(open(mpath))
    meta["files_tree_hash"] = tree_hash(files_dir)
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")


# --------------------------------------------------------------------------
# per-world driver
# --------------------------------------------------------------------------

def enrich_world(split: str, wid: str, worlds_dir: str, gold_dir: str,
                 model: str, client=None) -> dict:
    files_dir = os.path.join(worlds_dir, split, wid, "files")
    gdir = os.path.join(gold_dir, split, wid)
    cache_path = os.path.join(gdir, "enrichment.json")
    meta = json.load(open(os.path.join(gdir, "meta.json")))

    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        apply_enrichment(files_dir, cache)
        _refresh_meta_hash(gdir, files_dir)
        return {"wid": wid, "cached": True,
                "accepted": len(cache["files"]),
                "fallbacks": cache.get("fallbacks", []), "retried": []}

    if client is None:
        client = _client()
    plan = collect_plan(files_dir, meta)

    result = _call_llm(client, model, _build_prompt(meta, plan), 16000)
    accepted: dict[str, str] = {}
    problems: dict[str, list[str]] = {}
    for rel, spec in plan.items():
        probs = validate_file(rel, result.get(rel), spec) \
            if rel in result else ["file missing from LLM response"]
        if probs:
            problems[rel] = probs
        else:
            accepted[rel] = result[rel]

    retried = sorted(problems)
    first_pass_reasons = {k: v[:3] for k, v in problems.items()}
    if problems:                                   # one retry for the failures
        retry_plan = {rel: plan[rel] for rel in problems}
        try:
            result2 = _call_llm(
                client, model, _build_prompt(meta, retry_plan, problems), 16000)
        except RuntimeError:
            result2 = {}
        still_bad: dict[str, list[str]] = {}
        for rel in retry_plan:
            probs = validate_file(rel, result2.get(rel), plan[rel]) \
                if rel in result2 else ["file missing from LLM response"]
            if probs:
                still_bad[rel] = probs
            else:
                accepted[rel] = result2[rel]
        problems = still_bad

    fallbacks = sorted(problems)                   # keep originals for these
    cache = {"model": model, "files": accepted, "fallbacks": fallbacks,
             "fallback_reasons": problems}
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
        f.write("\n")
    apply_enrichment(files_dir, cache)
    _refresh_meta_hash(gdir, files_dir)
    return {"wid": wid, "cached": False, "accepted": len(accepted),
            "fallbacks": fallbacks, "retried": retried,
            "first_pass_reasons": first_pass_reasons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", default="worlds")
    ap.add_argument("--gold-dir", default="gold")
    ap.add_argument("--model", default=os.environ.get("DATAGEN_MODEL", "gpt-5.6-sol"))
    ap.add_argument("--only", default=None, help="comma list of world ids")
    ap.add_argument("--workers", type=int, default=6)
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
        futs = {ex.submit(enrich_world, s, w, a.worlds_dir, a.gold_dir,
                          a.model, client): (s, w) for s, w in targets}
        for fut in concurrent.futures.as_completed(futs):
            s, w = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"wid": w, "error": f"{type(e).__name__}: {e}"}
            results.append(r)
            tag = "cache" if r.get("cached") else ("ERROR" if "error" in r else "fresh")
            print(f"[{tag}] {s}/{w}: " + (r.get("error") or
                  f"accepted={r['accepted']} retried={len(r['retried'])} "
                  f"fallbacks={r['fallbacks']}"), flush=True)
            for rel, reasons in (r.get("first_pass_reasons") or {}).items():
                print(f"        retry {rel}: {'; '.join(reasons)}", flush=True)

    usd = (_USAGE["prompt"] * _PRICE_IN + _USAGE["completion"] * _PRICE_OUT) / 1e6
    n_err = sum(1 for r in results if "error" in r)
    n_fb = sum(len(r.get("fallbacks", [])) for r in results)
    print(f"\nenrich done: {len(results)} worlds, {n_err} errors, "
          f"{n_fb} fallback files; {_USAGE['calls']} calls, "
          f"{_USAGE['prompt']} in / {_USAGE['completion']} out tokens, "
          f"~${usd:.2f} est.")


if __name__ == "__main__":
    main()
