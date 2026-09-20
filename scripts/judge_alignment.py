"""Judge-alignment report: how well does each training judge's gold-blind
score track the deterministic gold score on held-out checkpoint records?

Usage: python3 scripts/judge_alignment.py [tag]
Compares results/checkpoints.jsonl (Run A, reading judge) with
results_b/checkpoints.jsonl (Run B, verifying judge) on the same tag
(default c0 — identical worlds, same v0 spec, so the only variable is
actor sampling noise and the judge).
"""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pairs(path, tag):
    out = {}
    if not path.exists():
        return []
    for line in open(path):
        r = json.loads(line)
        if r.get("tag") != tag:
            continue
        g, j = r.get("gold_score"), r.get("judge_score")
        if g is None or j is None:
            continue
        out[r.get("run_id") or r["task_id"]] = (float(g), float(j))
    return list(out.values())


def _pearson(pairs):
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def _spearman(pairs):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            r = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = r
            i = j + 1
        return ranks
    rx = rank([p[0] for p in pairs])
    ry = rank([p[1] for p in pairs])
    return _pearson(list(zip(rx, ry)))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "c0"
    rows = []
    for label, path in [("A/reading", ROOT / "results/checkpoints.jsonl"),
                        ("B/verifying", ROOT / "results_b/checkpoints.jsonl")]:
        pairs = _pairs(path, tag)
        if not pairs:
            rows.append((label, 0, "-", "-", "-", "-"))
            continue
        rows.append((label, len(pairs),
                     f"{sum(p[0] for p in pairs)/len(pairs):5.1f}",
                     f"{sum(p[1] for p in pairs)/len(pairs):5.2f}",
                     f"{_pearson(pairs):+.3f}",
                     f"{_spearman(pairs):+.3f}"))
    print(f"tag={tag}   judge vs gold on identical held-out worlds")
    print(f"{'run':<12} {'n':>3} {'gold':>6} {'judge':>6} {'pearson':>8} {'spearman':>9}")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:>3} {r[2]:>6} {r[3]:>6} {r[4]:>8} {r[5]:>9}")


if __name__ == "__main__":
    main()
