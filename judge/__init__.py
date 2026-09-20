"""Gold-blind judging layer (DESIGN section 8).

Exposes:
    judge_task  -- training judge pseudo-reward (section 8.1)
    compare     -- pairwise trace-vs-trace comparator (section 8.2)

Neither function ever receives gold answers, rubrics, or gold scores; both
guard against gold leakage defensively.
"""

from judge.training_judge import judge_task
from judge.pairwise import compare

__all__ = ["judge_task", "compare"]
