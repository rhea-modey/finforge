"""Held-out improvement curve extraction (the experiment's headline data).

Reads results/checkpoints.jsonl -> writes results/curve.json and prints a
terminal table. Pure stdlib; safe to run at any time (partial data ok).

    python3 -m scoring.curve
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

TRAIN_INDUSTRY_WORLDS = {"h01", "h02", "h03"}     # heldout companies, seen industries
HELDOUT_INDUSTRY_WORLDS = {"h04", "h05", "h06"}   # unseen industries


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def build_curve(checkpoints_path: str = "results/checkpoints.jsonl") -> dict:
    if not os.path.exists(checkpoints_path):
        return {"checkpoints": []}
    by_cp: dict = defaultdict(list)
    for line in open(checkpoints_path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") != "heldout":
            continue
        key = (r.get("checkpoint"), r.get("harness_version"), r.get("tag"))
        by_cp[key].append(r)

    # Dedupe within each checkpoint by run_id, keeping the LAST record —
    # retried infra-error tasks supersede their earlier error record.
    for key, recs in by_cp.items():
        latest = {}
        for r in recs:
            latest[r.get("run_id")] = r
        by_cp[key] = list(latest.values())

    out = []
    for (cp, version, tag), recs in sorted(by_cp.items(),
                                           key=lambda kv: (kv[0][0] is None,
                                                           kv[0][0] or 0)):
        golds = [r.get("gold_score") for r in recs]
        entry = {
            "checkpoint": cp, "version": version, "tag": tag,
            "n": len(recs),
            "gold_mean": _mean(golds),
            "judge_mean": _mean([(r.get("judge_score") or 0) * 10 for r in recs]),
            "gold_seen_industries": _mean(
                [r.get("gold_score") for r in recs
                 if r.get("world_id") in TRAIN_INDUSTRY_WORLDS]),
            "gold_unseen_industries": _mean(
                [r.get("gold_score") for r in recs
                 if r.get("world_id") in HELDOUT_INDUSTRY_WORLDS]),
            "by_category": {
                cat: _mean([r.get("gold_score") for r in recs
                            if r.get("category") == cat])
                for cat in ("bank_rec", "journal_entries",
                            "variance_analysis", "accrual_schedule")},
            "errors": sum(1 for r in recs if r.get("ended") == "error"),
            "usd": round(sum(float(r.get("usd") or 0) for r in recs), 4),
            "usd_per_task": round(_mean([float(r.get("usd") or 0)
                                         for r in recs]) or 0, 4),
        }
        out.append(entry)
    # Deployed-harness series: the spec actually accepted at each iteration
    # (acceptance gate winners), with its gold read from that spec's own
    # checkpoint. Monotone-resistant by construction: a proposal must beat the
    # incumbent head-to-head (judge, same worlds) to be deployed.
    gold_by_version = {c["version"]: c["gold_mean"] for c in out}
    deployed = []
    gates_path = os.path.join(os.path.dirname(checkpoints_path), "gates.jsonl")
    gates = {}
    if os.path.exists(gates_path):
        for line in open(gates_path):
            try:
                g = json.loads(line)
                gates[int(g["iteration"])] = g   # last write wins
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
    for it in sorted(gates):
        g = gates[it]
        deployed.append({"iteration": it, "deployed": g.get("adopted"),
                         "candidate": g.get("candidate"),
                         "candidate_mean_judge": g.get("candidate_mean"),
                         "incumbent_mean_judge": g.get("incumbent_mean"),
                         "deployed_gold": gold_by_version.get(g.get("adopted"))})
    return {"checkpoints": out, "deployed": deployed}


def main():
    curve = build_curve()
    os.makedirs("results", exist_ok=True)
    with open("results/curve.json", "w") as f:
        json.dump(curve, f, indent=1, sort_keys=True)
    cps = curve["checkpoints"]
    if not cps:
        print("no heldout checkpoint records yet")
        return
    print(f"{'cp':>3} {'ver':>4} {'n':>3} {'gold':>6} {'judge':>6} "
          f"{'seen-ind':>8} {'unseen':>7} {'$/task':>7}  categories (br/je/va/as)")
    prev = None
    for c in cps:
        cats = c["by_category"]
        cat_s = "/".join("--" if cats[k] is None else f"{cats[k]:.0f}"
                         for k in ("bank_rec", "journal_entries",
                                   "variance_analysis", "accrual_schedule"))
        delta = ""
        if prev is not None and c["gold_mean"] is not None and prev is not None:
            d = c["gold_mean"] - prev
            delta = f"  ({'+' if d >= 0 else ''}{d:.1f})"
        print(f"{str(c['checkpoint']):>3} {c['version'] or '?':>4} {c['n']:>3} "
              f"{c['gold_mean'] if c['gold_mean'] is not None else '--':>6} "
              f"{c['judge_mean'] if c['judge_mean'] is not None else '--':>6} "
              f"{c['gold_seen_industries'] if c['gold_seen_industries'] is not None else '--':>8} "
              f"{c['gold_unseen_industries'] if c['gold_unseen_industries'] is not None else '--':>7} "
              f"{c['usd_per_task']:>7} {cat_s}{delta}")
        prev = c["gold_mean"]
    print("\nwrote results/curve.json")


if __name__ == "__main__":
    main()
