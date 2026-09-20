"""World builder CLI (DESIGN 14): python3 -m generator.build [--worlds-dir ...]"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from .discrepancies import TruthRecord
from .engine import build_world
from .exporters import export_world
from .profile import make_profile
from .tasks import build_tasks

TRAIN_IDS = [f"w{i:02d}" for i in range(1, 31)]
HELDOUT_IDS = [f"h{i:02d}" for i in range(1, 7)]


def tree_hash(root: str) -> str:
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in sorted(os.walk(root)):
        dirnames.sort()
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            h.update(os.path.relpath(fp, root).encode())
            with open(fp, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _jdump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def build_one(master_seed: int, split: str, wid: str,
              worlds_dir: str, gold_dir: str) -> None:
    prof = make_profile(master_seed, split, wid)
    w = build_world(prof)
    files_dir = os.path.join(worlds_dir, split, wid, "files")
    export_world(w, files_dir)
    gdir = os.path.join(gold_dir, split, wid)
    os.makedirs(gdir, exist_ok=True)
    tasks = build_tasks(w)
    _jdump([{"task_id": t["task_id"], "category": t["category"],
             "prompt": t["prompt"]} for t in tasks],
           os.path.join(gdir, "tasks.json"))
    for t in tasks:
        _jdump(t["gold"], os.path.join(gdir, f"{t['task_id']}.json"))
        _jdump(t["rubric"], os.path.join(gdir, f"{t['task_id']}.rubric.json"))
    meta = {
        "world_id": wid, "split": split, "industry": prof.industry,
        "company": prof.company, "close_month": "2026-06",
        "chart_of_accounts": w.coa,
        "truth_records": [t.to_json() for t in w.truth],
        "files_tree_hash": tree_hash(files_dir),
    }
    _jdump(meta, os.path.join(gdir, "meta.json"))


def build_all(worlds_dir: str = "worlds", gold_dir: str = "gold",
              config_path: str = "config/experiment.json") -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    seed = int(cfg.get("master_seed", 20260919))
    for wid in TRAIN_IDS:
        build_one(seed, "train", wid, worlds_dir, gold_dir)
    for wid in HELDOUT_IDS:
        build_one(seed, "heldout", wid, worlds_dir, gold_dir)
    print(f"built {len(TRAIN_IDS)} train + {len(HELDOUT_IDS)} heldout worlds "
          f"-> {worlds_dir}/ (gold -> {gold_dir}/)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds-dir", default="worlds")
    ap.add_argument("--gold-dir", default="gold")
    ap.add_argument("--config", default="config/experiment.json")
    a = ap.parse_args()
    build_all(a.worlds_dir, a.gold_dir, a.config)
