"""Pairwise trace-vs-trace comparator (DESIGN section 8.2).

compare(category, prev_digest, cur_digest, cfg) -> dict
    {"winner": "prev|cur|tie", "reasons": str, "transferable_advice": str}

Digests: {"task_prompt", "answer", "notes", "trace_summary", "judge_score"}.
The two runs come from different iterations on DIFFERENT worlds, so the
comparator is explicitly directed at process quality -- information flow
between agents, wasted work, verification discipline -- never at
world-specific facts. Gold-blind; same JSON robustness pattern as the
training judge; returns a "tie" verdict (never raises) on total judge
failure, apart from the gold-leak guard.
"""

from __future__ import annotations

import json

from judge.training_judge import _assert_no_gold, _chat_json, _extract_json, _truncate

_SYSTEM = """You are a senior corporate controller comparing TWO month-end \
close workpapers of the same task category, produced by two successive \
versions of a multi-agent workflow ("harness"): PREV (the earlier version) \
and CUR (the later version).

CRITICAL CONTEXT: the two runs were executed on DIFFERENT synthetic companies \
-- the benchmark generates fresh worlds every iteration. Company names, \
vendors, dollar amounts, dates, and file names WILL differ between the two \
runs, and that difference is meaningless. NEVER base your verdict on \
world-specific facts, and never reward one run for its particular numbers \
"looking better". Compare PROCESS QUALITY only:
- information flow between agents: did each agent receive what it needed, and \
did each agent's return carry the concrete data (file contents, column \
layouts, exact figures, file paths) that downstream steps needed, rather than \
vague summaries?
- wasted work: duplicated file reads, dead-end exploration, agents redoing \
each other's work, steps that produced nothing the final answer used;
- verification discipline: were sums cross-footed and figures checked against \
source files (e.g. via run_python) or merely asserted from memory?
- internal consistency and schema/format compliance of the final deliverable;
- completeness of approach relative to what the task asked.

The judge scores shown are noisy pseudo-scores from another gold-blind \
reviewer, not ground truth -- use them as weak context and form your own view.

OUTPUT CONTRACT -- respond with ONLY one JSON object, no markdown fences, no \
prose outside it:
{"winner": "prev" | "cur" | "tie",
 "reasons": "<3-6 sentences of process-level comparison>",
 "transferable_advice": "<2-4 sentences of advice that would improve the \
workflow on ANY company/world; never mention specific companies, amounts, or \
file names>"}"""


def _digest_block(label: str, digest: dict) -> str:
    digest = digest or {}
    answer = digest.get("answer")
    if answer is None:
        answer_block = "(no answer JSON was submitted)"
    else:
        try:
            answer_block = json.dumps(answer, indent=2, default=str)
        except (TypeError, ValueError):
            answer_block = repr(answer)
    judge_score = digest.get("judge_score")
    score_line = "(unavailable)" if judge_score is None else str(judge_score)
    return (
        f"---- {label} RUN ----\n"
        f"Task prompt:\n{_truncate(digest.get('task_prompt'), 1500)}\n\n"
        f"Submitted answer JSON:\n{_truncate(answer_block, 3000)}\n\n"
        f"Notes:\n{_truncate(digest.get('notes'), 1500) or '(none)'}\n\n"
        f"Process trace summary:\n{_truncate(digest.get('trace_summary'), 4000) or '(none)'}\n\n"
        f"Gold-blind judge score (0-10, noisy): {score_line}\n"
    )


_WINNER_ALIASES = {
    "prev": "prev", "previous": "prev", "prev run": "prev",
    "cur": "cur", "current": "cur", "cur run": "cur",
    "tie": "tie", "draw": "tie", "equal": "tie",
}


def _parse_comparison(text: str) -> dict:
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    winner = _WINNER_ALIASES.get(str(obj.get("winner", "")).strip().lower())
    if winner is None:
        raise ValueError('"winner" must be exactly one of "prev", "cur", or "tie"')
    reasons = str(obj.get("reasons", "")).strip()
    if not reasons:
        raise ValueError('missing or empty "reasons"')
    advice = str(obj.get("transferable_advice", "")).strip()
    return {"winner": winner, "reasons": reasons, "transferable_advice": advice}


def compare(category: str, prev_digest: dict, cur_digest: dict, cfg: dict) -> dict:
    """Compare a previous-iteration run against a current-iteration run of the
    same task category (DESIGN section 8.2). Never raises (except the
    gold-leak guard); on total judge failure returns a tie with an
    explanatory reason.
    """
    _assert_no_gold(prev_digest, "prev_digest")
    _assert_no_gold(cur_digest, "cur_digest")

    pcfg = (cfg or {}).get("pairwise_judge", {}) or {}
    try:
        max_tokens = int(pcfg.get("max_tokens", 4000) or 4000)
    except (TypeError, ValueError):
        max_tokens = 4000

    try:
        from runtime.llm import LLMClient
        client = LLMClient.for_role("pairwise_judge", cfg)
    except Exception as e:  # noqa: BLE001
        return {"winner": "tie",
                "reasons": f"Judge failure: could not initialize the pairwise judge LLM client ({e}).",
                "transferable_advice": ""}

    user_prompt = (
        f"Task category for both runs: {category}\n\n"
        f"{_digest_block('PREV', prev_digest)}\n"
        f"{_digest_block('CUR', cur_digest)}\n"
        "Remember: different worlds, so compare process quality only. "
        "Respond with ONLY the JSON object required by the output contract."
    )
    messages = [{"role": "user", "content": user_prompt}]

    try:
        return _chat_json(client, _SYSTEM, messages, max_tokens,
                          _parse_comparison, retries=2)
    except Exception as e:  # noqa: BLE001 -- parse exhaustion or transport error
        return {"winner": "tie",
                "reasons": f"Judge failure: pairwise comparison failed ({_truncate(str(e), 300)}); defaulting to tie.",
                "transferable_advice": ""}
