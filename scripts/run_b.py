"""Experiment B: identical protocol to Run A, ONE variable changed — the
training judge verifies with tools (mode: "verifying") instead of reading.

Own results tree (results_b/) and revision line (harness/revisions_b/), same
frozen worlds, same v0, same actor/optimizer/scorer. Usage:

    source scripts/env.sh && python3 scripts/run_b.py
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiment as X  # noqa: E402

RESULTS_B = ROOT / "results_b"
REVISIONS_B = ROOT / "harness" / "revisions_b"
RESULTS_B.mkdir(exist_ok=True)
(RESULTS_B / "traces").mkdir(exist_ok=True)
REVISIONS_B.mkdir(parents=True, exist_ok=True)

X.RESULTS_DIR = RESULTS_B
X.STATE_PATH = RESULTS_B / "state.json"
X.RUNS_PATH = RESULTS_B / "runs.jsonl"
X.CHECKPOINTS_PATH = RESULTS_B / "checkpoints.jsonl"
X.DIGESTS_PATH = RESULTS_B / "digests.jsonl"
X.ITERATIONS_PATH = RESULTS_B / "iterations.jsonl"
X.SPEND_PATH = RESULTS_B / "spend.json"
X.REVISIONS_DIR = REVISIONS_B

cfg_path = ROOT / "config" / "experiment_b.json"
if not cfg_path.exists():
    base = json.load(open(ROOT / "config" / "experiment.json"))
    base["training_judge"] = {**base["training_judge"], "mode": "verifying",
                              "verify_max_steps": 18}
    base["pairwise_judge"] = dict(base["pairwise_judge"])
    base["parallel_workers"] = 6
    base["iterations"] = 8
    base["heldout_checkpoints"] = list(range(0, 9))
    base["heldout_repeats"] = 2
    base["optimizer"] = {**base.get("optimizer", {}), "surgical_edits": True}
    json.dump(base, open(cfg_path, "w"), indent=2)


class _Args:
    config = str(cfg_path)


sys.exit(X.cmd_run(_Args()))
