"""Gold-blind training judge (DESIGN section 8.1) -- the pseudo-reward.

judge_task(task_prompt, answer, notes, trace_summary, cfg) -> dict
    {"score": float 0..10, "critique": str, "criteria": [{"name","met","comment"}]}

The judge NEVER sees gold answers, auto-rubrics, gold scores, or other runs.
A defensive guard rejects any input dict containing a gold-flavored key.

Robustness contract: apart from the gold-leak guard (which raises ValueError
on a caller bug -- that is the point of the guard), judge_task never raises.
On total judge failure it returns score 0.0 with a critique explaining the
judge failure.
"""

from __future__ import annotations

import json
import re
import statistics

# --------------------------------------------------------------------------
# Shared helpers (also imported by judge.pairwise)
# --------------------------------------------------------------------------


def _assert_no_gold(obj, where: str) -> None:
    """Recursively reject any dict key that looks like gold leakage.

    Raises ValueError -- this is a contract-violation guard, not a runtime
    condition: gold must never reach the judge or optimizer (DESIGN sections
    8-9), so a leak fails loudly instead of silently contaminating a prompt.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key == "gold" or key.startswith("gold_") or key.endswith("_gold") or key == "rubric" or "rubric" in key:
                raise ValueError(
                    f"gold-blind violation: input '{where}' contains forbidden key '{k}' "
                    "(gold answers, gold scores, and rubrics must never reach the judge/optimizer)"
                )
            _assert_no_gold(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _assert_no_gold(item, f"{where}[{i}]")


def _truncate(s, n: int) -> str:
    s = "" if s is None else str(s)
    if len(s) <= n:
        return s
    return s[:n] + " ...[truncated]"


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop <think>...</think> reasoning blocks (some hosts serve reasoning
    models with the think text inline in message.content; a draft JSON object
    inside think must never win over the final one). A stray unmatched
    </think> drops everything up to and including it."""
    out = _THINK_BLOCK_RE.sub("", text or "")
    low = out.lower()
    idx = low.rfind("</think>")
    if idx != -1:
        out = out[idx + len("</think>"):]
    return out


def _extract_json(text: str):
    """Parse a JSON object out of an LLM reply. Think blocks are stripped
    first; then the reply is scanned for ALL parseable JSON objects and the
    LAST one wins (a final answer follows any draft). Raises ValueError with
    a message suitable for feeding back to the model."""
    if not text or not text.strip():
        raise ValueError("empty response (no JSON object found)")
    t = _strip_think(text).strip()
    if not t:
        raise ValueError("response contained only reasoning text -- no JSON object found")
    # Strip a single wrapping markdown code fence if present.
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z]*[ \t]*\r?\n?", "", t)
        if t.endswith("```"):
            t = t[: -3]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    last_obj, found = None, False
    idx = t.find("{")
    while idx != -1:
        try:
            obj, end = dec.raw_decode(t[idx:])
        except json.JSONDecodeError:
            idx = t.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            last_obj, found = obj, True
        idx = t.find("{", idx + max(1, end))
    if found:
        return last_obj
    raise ValueError("no JSON object found in response")


def _chat_json(client, system: str, base_messages: list, max_tokens: int,
               parse_fn, retries: int = 2):
    """One judged LLM exchange with a strict-JSON contract.

    Calls client.chat with json_mode=True; parses with parse_fn (which raises
    ValueError on any contract violation); on parse failure the error text is
    fed back and the call retried, up to `retries` extra attempts (DESIGN 8.1:
    parse-retry, 2 attempts). Raises ValueError after the final failure.
    LLM-transport exceptions propagate to the caller.
    """
    msgs = list(base_messages)
    last_err = None
    for _attempt in range(retries + 1):
        resp = client.chat(system=system, messages=msgs,
                           max_tokens=max_tokens, json_mode=True)
        text = resp.text or ""
        try:
            return parse_fn(text)
        except ValueError as e:
            last_err = e
            msgs.append({"role": "assistant", "content": text})
            msgs.append({
                "role": "user",
                "content": (
                    f"Your reply could not be used: {e}. "
                    "Respond again with ONLY the single JSON object required by the "
                    "output contract -- no prose, no markdown fences, nothing else."
                ),
            })
    raise ValueError(f"output unparseable after {retries + 1} attempts (last error: {last_err})")


# --------------------------------------------------------------------------
# Training-judge prompt
# --------------------------------------------------------------------------

_SYSTEM = """You are a senior corporate controller reviewing a staff accountant's \
month-end close workpaper before it goes into the close binder.

You do NOT have access to the company's underlying books beyond what is shown \
below, and you do NOT know the ground-truth answers -- never pretend that you \
do, and never invent "correct" figures of your own. Review strictly on what is \
in front of you: the task assignment, the submitted answer, the accountant's \
notes, and a summary of the process trace showing how the work was produced.

Score the workpaper on exactly five dimensions:
1. completeness -- does the deliverable address every element the task asked \
for (all required fields populated, all requested analyses attempted)?
2. internal_consistency -- do the numbers tie? Reconciling items should sum to \
the stated differences, journal entries should balance debits and credits, \
schedules should roll forward (opening + additions - amortization = closing), \
subtotals should foot.
3. evidence_discipline -- does the process trace show the relevant source \
files were actually located, read, and used, so the figures are grounded in \
the files rather than invented? Was arithmetic verified (e.g. via a \
run_python computation) instead of asserted from memory?
4. format_compliance -- does the answer JSON match the schema stated in the \
task exactly (field names, nesting, types, enumerated values, date formats)?
5. plausibility -- are signs, magnitudes, dates, and directions plausible for \
a month-end close (no negative balances where impossible, dates inside the \
close period where expected, favorable/unfavorable directions consistent with \
the numbers)?

Scoring scale (0-10): 0-2 unusable or essentially missing; 3-4 major gaps or \
contradictions; 5-6 a real attempt with material issues; 7-8 solid work with \
minor issues; 9-10 excellent -- complete, internally tied-out, well-evidenced.

OUTPUT CONTRACT -- respond with ONLY one JSON object, no markdown fences, no \
prose outside it:
{"score": <number 0-10>,
 "critique": "<3-8 sentences: the strongest aspects, the concrete defects, and what to do differently next time>",
 "criteria": [
   {"name": "completeness", "met": <true|false>, "comment": "<one sentence>"},
   {"name": "internal_consistency", "met": <true|false>, "comment": "<one sentence>"},
   {"name": "evidence_discipline", "met": <true|false>, "comment": "<one sentence>"},
   {"name": "format_compliance", "met": <true|false>, "comment": "<one sentence>"},
   {"name": "plausibility", "met": <true|false>, "comment": "<one sentence>"}
 ]}
Exactly five criteria entries, one per dimension, in that order."""


def _build_user_prompt(task_prompt: str, answer, notes: str, trace_summary: str) -> str:
    if answer is None:
        answer_block = "(no answer JSON was submitted -- the run ended without a valid submission)"
    else:
        try:
            answer_block = json.dumps(answer, indent=2, default=str)
        except (TypeError, ValueError):
            answer_block = repr(answer)
    return (
        "== TASK ASSIGNMENT (as given to the accounting team) ==\n"
        f"{_truncate(task_prompt, 4000)}\n\n"
        "== SUBMITTED ANSWER JSON ==\n"
        f"{_truncate(answer_block, 6000)}\n\n"
        "== ACCOUNTANT'S NOTES ==\n"
        f"{_truncate(notes, 3000) or '(none)'}\n\n"
        "== PROCESS TRACE SUMMARY (how the multi-agent team produced this) ==\n"
        f"{_truncate(trace_summary, 8000) or '(no trace summary available)'}\n\n"
        "== YOUR REVIEW ==\n"
        "Score this workpaper per the five dimensions and the output contract in "
        "your instructions. Respond with ONLY the JSON object."
    )


def _parse_judgment(text: str) -> dict:
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    if "score" not in obj:
        raise ValueError('missing required key "score"')
    if "critique" not in obj:
        raise ValueError('missing required key "critique"')
    try:
        score = float(obj["score"])
    except (TypeError, ValueError):
        raise ValueError('"score" is not a number')
    if score != score:  # NaN
        raise ValueError('"score" is not a finite number')
    score = max(0.0, min(10.0, score))
    critique = str(obj["critique"]).strip()
    if not critique:
        raise ValueError('"critique" is empty')
    raw_criteria = obj.get("criteria", [])
    if not isinstance(raw_criteria, list):
        raise ValueError('"criteria" is not a list')
    criteria = []
    for c in raw_criteria:
        if not isinstance(c, dict):
            continue
        criteria.append({
            "name": str(c.get("name", "")).strip() or "unnamed",
            "met": bool(c.get("met", False)),
            "comment": str(c.get("comment", "")).strip(),
        })
    return {"score": score, "critique": critique, "criteria": criteria}


# --------------------------------------------------------------------------
# Public interface (pinned, DESIGN section 14)
# --------------------------------------------------------------------------


def judge_task(task_prompt: str, answer: dict | None, notes: str,
               trace_summary: str, cfg: dict) -> dict:
    """Gold-blind score + critique for one task run (DESIGN section 8.1).

    k = cfg["training_judge"]["samples"] independent judge calls; score is the
    median of the sample scores; critiques are concatenated with sample
    labels; criteria come from the sample closest to the median. Never raises
    (except the gold-leak guard): on total judge failure returns score 0.0
    with an explanatory critique.
    """
    _assert_no_gold(answer, "answer")

    jcfg = (cfg or {}).get("training_judge", {}) or {}
    if jcfg.get("mode") == "verifying":
        # Experiment B: tool-equipped verifying judge (still gold-blind — it
        # sees only the same files/ the actor saw). Single verification pass;
        # `samples` is ignored in this mode.
        from judge.verifying_judge import judge_task_verifying
        return judge_task_verifying(task_prompt, answer, notes,
                                    trace_summary, cfg)
    try:
        k = max(1, int(jcfg.get("samples", 3) or 3))
    except (TypeError, ValueError):
        k = 3
    try:
        max_tokens = int(jcfg.get("max_tokens", 4000) or 4000)
    except (TypeError, ValueError):
        max_tokens = 4000

    try:
        from runtime.llm import LLMClient
        client = LLMClient.for_role("training_judge", cfg)
    except Exception as e:  # noqa: BLE001 -- judge must never raise
        return {"score": 0.0,
                "critique": f"Judge failure: could not initialize the judge LLM client ({e}).",
                "criteria": []}

    user_prompt = _build_user_prompt(task_prompt, answer, notes, trace_summary)
    base_messages = [{"role": "user", "content": user_prompt}]

    samples: list[dict] = []
    failures: list[str] = []
    for i in range(k):
        try:
            samples.append(_chat_json(client, _SYSTEM, base_messages,
                                      max_tokens, _parse_judgment, retries=2))
        except Exception as e:  # noqa: BLE001 -- parse exhaustion or transport error
            failures.append(f"sample {i + 1}: {_truncate(str(e), 300)}")

    if not samples:
        return {"score": 0.0,
                "critique": ("Judge failure: all judge samples failed, defaulting to score 0. "
                             + "; ".join(failures)),
                "criteria": []}

    med = float(statistics.median(s["score"] for s in samples))
    parts = [f"[sample {i + 1} | score {s['score']:.1f}] {s['critique']}"
             for i, s in enumerate(samples)]
    if failures:
        parts.append(f"[judge notice] {len(failures)} of {k} judge samples failed: "
                     + "; ".join(failures))
    median_sample = min(samples, key=lambda s: abs(s["score"] - med))
    return {"score": med,
            "critique": "\n\n".join(parts),
            "criteria": median_sample["criteria"]}
