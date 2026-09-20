"""FinForge results aggregation (DESIGN section 2 / section 10).

Aggregates results/runs.jsonl + results/checkpoints.jsonl into
results/summary.json and prints a compact text table to stdout.

Usage: python -m scoring.report [--results-dir results] [--gold-dir gold]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = ("bank_rec", "journal_entries", "variance_analysis", "accrual_schedule")
_CAT_ABBR = {"bank_rec": "bank", "journal_entries": "je",
             "variance_analysis": "var", "accrual_schedule": "accr"}

TRAIN_INDUSTRIES = {"saas", "ecommerce", "wholesale", "services"}
HELDOUT_INDUSTRIES = {"manufacturing", "clinic"}
# Fallback positional mapping per DESIGN 4.1 ("3 from train industries + 3
# from held-out industries", h01..h06 in that order), used only when the
# world's gold meta.json is unavailable.
_FALLBACK_TRAIN_INDUSTRY_WORLDS = {"h01", "h02", "h03"}


# ------------------------------------------------------------------ helpers

def _read_jsonl(path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return (sum(xs) / len(xs)) if xs else None


def _rnd(x, nd=3):
    return None if x is None else round(float(x), nd)


def _judge_scores(recs):
    return [r.get("judge_score") for r in recs]


def _gold_scores(recs):
    return [r.get("gold_score") for r in recs]


def _find_industry(obj, depth=0):
    if depth > 6:
        return None
    if isinstance(obj, dict):
        v = obj.get("industry")
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
        for val in obj.values():
            r = _find_industry(val, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for val in obj:
            r = _find_industry(val, depth + 1)
            if r:
                return r
    return None


def _world_group(world_id: str, gold_dir: Path) -> str:
    """'train_industry' | 'heldout_industry' for a heldout world."""
    meta_path = gold_dir / "heldout" / str(world_id) / "meta.json"
    if meta_path.exists():
        try:
            industry = _find_industry(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            industry = None
        if industry in TRAIN_INDUSTRIES:
            return "train_industry"
        if industry in HELDOUT_INDUSTRIES:
            return "heldout_industry"
    return ("train_industry" if str(world_id) in _FALLBACK_TRAIN_INDUSTRY_WORLDS
            else "heldout_industry")


def _stats(recs) -> dict:
    return {
        "n": len(recs),
        "mean_judge": _rnd(_mean(_judge_scores(recs))),
        "mean_gold": _rnd(_mean(_gold_scores(recs))),
        "errors": sum(1 for r in recs if r.get("ended") == "error"),
        "usd": _rnd(sum(float(r.get("usd") or 0.0) for r in recs), 4),
    }


def _per_category(recs) -> dict:
    out = {}
    for c in CATEGORIES:
        crecs = [r for r in recs if r.get("category") == c]
        if crecs:
            out[c] = {
                "n": len(crecs),
                "mean_judge": _rnd(_mean(_judge_scores(crecs))),
                "mean_gold": _rnd(_mean(_gold_scores(crecs))),
            }
    return out


def _token_totals(recs) -> dict:
    tot = {"in": 0, "out": 0, "cache_r": 0, "cache_w": 0}
    for r in recs:
        t = r.get("tokens")
        if isinstance(t, dict):
            for k in tot:
                v = t.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tot[k] += int(v)
    return tot


# ---------------------------------------------------------------- aggregate

def aggregate(results_dir: str = "results", gold_dir: str = "gold",
              write: bool = True) -> dict:
    results = Path(results_dir)
    gold = Path(gold_dir)
    runs = _read_jsonl(results / "runs.jsonl")
    cps = _read_jsonl(results / "checkpoints.jsonl")

    # ---- training: per-iteration judge means
    train = [r for r in runs if r.get("phase") == "train"]
    per_iteration = {}
    iters = sorted({r.get("iteration") for r in train
                    if isinstance(r.get("iteration"), int)})
    for it in iters:
        recs = [r for r in train if r.get("iteration") == it]
        st = _stats(recs)
        per_iteration[str(it)] = {
            "harness_version": recs[0].get("harness_version"),
            **st,
            "per_category": {c: v["mean_judge"]
                             for c, v in _per_category(recs).items()},
        }

    # ---- heldout checkpoints, grouped by tag (insertion order preserved)
    checkpoints = {}
    for r in cps:
        tag = r.get("tag")
        if tag is None:
            cnum = r.get("checkpoint")
            tag = f"c{cnum}" if cnum is not None else "untagged"
        checkpoints.setdefault(str(tag), []).append(r)

    cp_summary = {}
    for tag, recs in checkpoints.items():
        by_group: dict[str, list] = {"train_industry": [], "heldout_industry": []}
        for r in recs:
            by_group[_world_group(r.get("world_id", ""), gold)].append(r)
        cp_summary[tag] = {
            "tag": tag,
            "checkpoint": recs[0].get("checkpoint"),
            "harness_version": recs[0].get("harness_version"),
            **_stats(recs),
            "per_category": _per_category(recs),
            "by_group": {g: _stats(rs) for g, rs in by_group.items() if rs},
        }

    # ---- cost totals
    all_recs = runs + cps
    cost = {
        "total_usd": _rnd(sum(float(r.get("usd") or 0.0) for r in all_recs), 4),
        "train_usd": _rnd(sum(float(r.get("usd") or 0.0) for r in train), 4),
        "heldout_usd": _rnd(sum(float(r.get("usd") or 0.0) for r in cps), 4),
        "tokens": _token_totals(all_recs),
    }

    summary = {
        "train": {"per_iteration": per_iteration,
                  "n_runs": len(train),
                  "errors": sum(1 for r in train if r.get("ended") == "error")},
        "checkpoints": cp_summary,
        "cost": cost,
        "counts": {"runs_jsonl": len(runs), "checkpoints_jsonl": len(cps)},
    }

    if write:
        results.mkdir(parents=True, exist_ok=True)
        (results / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


# -------------------------------------------------------------- text table

def _fmt(x, nd=2, width=7):
    if x is None:
        return "-".rjust(width)
    return f"{float(x):.{nd}f}".rjust(width)


def render_table(summary: dict) -> str:
    lines = []
    lines.append("== FinForge summary ==")

    per_it = summary.get("train", {}).get("per_iteration", {})
    lines.append("")
    lines.append("Training (judge 0-10, mean per iteration)")
    hdr = (f"{'iter':>4} {'ver':<5} {'n':>3} {'judge':>7} "
           + " ".join(f"{_CAT_ABBR[c]:>6}" for c in CATEGORIES)
           + f" {'err':>3} {'usd':>8}")
    lines.append(hdr)
    if not per_it:
        lines.append("  (no training runs yet)")
    for it in sorted(per_it, key=lambda s: int(s)):
        row = per_it[it]
        pc = row.get("per_category", {})
        lines.append(
            f"{it:>4} {str(row.get('harness_version') or '-'):<5} "
            f"{row.get('n', 0):>3} {_fmt(row.get('mean_judge'))} "
            + " ".join(_fmt(pc.get(c), 2, 6) for c in CATEGORIES)
            + f" {row.get('errors', 0):>3} {_fmt(row.get('usd'), 2, 8)}"
        )

    cps = summary.get("checkpoints", {})
    lines.append("")
    lines.append("Heldout checkpoints (gold 0-100 | judge 0-10; "
                 "ti=train-industry worlds, hi=heldout-industry worlds)")
    lines.append(f"{'tag':<8} {'ver':<5} {'n':>3} {'gold':>7} {'judge':>7} "
                 f"{'gold-ti':>8} {'gold-hi':>8} {'err':>3} {'usd':>8}")
    if not cps:
        lines.append("  (no checkpoints yet)")
    for tag, row in cps.items():
        bg = row.get("by_group", {})
        ti = (bg.get("train_industry") or {}).get("mean_gold")
        hi = (bg.get("heldout_industry") or {}).get("mean_gold")
        lines.append(
            f"{tag:<8} {str(row.get('harness_version') or '-'):<5} "
            f"{row.get('n', 0):>3} {_fmt(row.get('mean_gold'))} "
            f"{_fmt(row.get('mean_judge'))} {_fmt(ti, 2, 8)} {_fmt(hi, 2, 8)} "
            f"{row.get('errors', 0):>3} {_fmt(row.get('usd'), 2, 8)}"
        )
        pc = row.get("per_category", {})
        if pc:
            cats = "  ".join(
                f"{_CAT_ABBR[c]}={_fmt(v.get('mean_gold'), 1, 0).strip()}"
                f"/{_fmt(v.get('mean_judge'), 1, 0).strip()}"
                for c, v in pc.items())
            lines.append(f"{'':<8}   per-cat (gold/judge): {cats}")

    cost = summary.get("cost", {})
    tok = cost.get("tokens", {})
    lines.append("")
    lines.append(
        f"Cost: ${cost.get('total_usd') or 0:.2f} total "
        f"(train ${cost.get('train_usd') or 0:.2f}, "
        f"heldout ${cost.get('heldout_usd') or 0:.2f}); "
        f"tokens in={tok.get('in', 0)} out={tok.get('out', 0)} "
        f"cache_r={tok.get('cache_r', 0)} cache_w={tok.get('cache_w', 0)}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate FinForge results into results/summary.json")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--gold-dir", default="gold")
    ap.add_argument("--no-write", action="store_true",
                    help="print the table without writing summary.json")
    args = ap.parse_args(argv)
    summary = aggregate(args.results_dir, args.gold_dir, write=not args.no_write)
    print(render_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
