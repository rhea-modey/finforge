"""Verifying training judge (Experiment B): a gold-blind judge WITH TOOLS.

Instead of reading the deliverable and scoring plausibility, this judge gets
the same file tools the actor had (read_file / grep_files / list_files /
run_python over the world's files/) and must RE-DERIVE the checkable facts
before scoring. Still strictly gold-blind: it sees only what the actor saw.

Same public contract as training_judge.judge_task, plus the world dir arrives
via cfg["_files_dir"] (plumbed by experiment._execute_task).
"""

from __future__ import annotations

import json
import re

SYSTEM = """You are a senior controller VERIFYING a staff accountant's month-end
close workpaper. You have the same source files they had, and tools to read,
search, and compute over them. Do not trust the submission — check it.

Verification protocol (adapt to the task type):
- Bank reconciliation: recompute the bank statement ending balance and the GL
  cash balance from the files with run_python; check each claimed reconciling
  item exists in the underlying data (a check in the GL missing from the
  statement, a fee on the statement missing from the GL, etc.); verify the
  adjusted balances tie.
- Journal entries: for each proposed entry, verify the underlying obligation
  exists in the files (payroll register period/pay date, prepaid schedule and
  contract, fixed-asset register, miscoded bill) AND that the GL does not
  already contain it. Flag both wrong entries and missing ones.
- Variance analysis: recompute June-only actuals for the flagged accounts from
  the GL with run_python (June rows only — never quarter-to-date totals),
  compare with the budget file, and check claimed drivers correspond to real
  documents whose amounts roughly explain the gaps.
- Prepaid schedule: recompute the roll-forward from the prior schedule,
  contracts, and GL; verify each row's arithmetic and the amortization entry
  against what the GL already shows.

Use run_python for every sum — do not do arithmetic in your head. Cite what
you verified vs what failed verification. Numbers the submission got right
matter more than its prose style.

After verifying, respond with ONLY this JSON (no markdown fence):
{"score": <0-10>, "critique": "<what was verified correct, what failed
verification, what is missing — cite file evidence>",
 "criteria": [{"name": "<check>", "met": true|false, "comment": "<evidence>"}]}

Scoring anchors: 9-10 all checkable facts verified correct and complete;
6-8 mostly verified with minor gaps; 3-5 material errors or omissions found;
0-2 fabricated/unverifiable numbers, wrong-period figures, or unusable output."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_task_verifying(task_prompt: str, answer, notes: str,
                         trace_summary: str, cfg: dict) -> dict:
    """Tool-loop judge. Returns {"score", "critique", "criteria"}. Never raises."""
    import time as _time

    from runtime.llm import LLMClient
    from runtime.tools import TOOL_SCHEMAS, make_toolbox

    files_dir = (cfg or {}).get("_files_dir")
    jcfg = (cfg or {}).get("training_judge", {}) or {}
    max_steps = int(jcfg.get("verify_max_steps", 18))
    # Hard wall-clock: a judge that keeps calling tools without concluding must
    # still return a verdict, or it wedges the worker (observed on held-out
    # h06). On timeout we force a final no-tools verdict from what it has.
    _deadline = _time.monotonic() + float(jcfg.get("verify_wallclock_s", 300))
    try:
        if not files_dir:
            return {"score": 0.0, "critique":
                    "verifying judge: no _files_dir plumbed", "criteria": []}
        client = LLMClient.for_role("training_judge", cfg)
        tools_impl = make_toolbox(files_dir,
                                  int((cfg or {}).get("run_python_timeout_s", 20)))
        tool_names = ["list_files", "read_file", "grep_files", "run_python"]
        tool_defs = [TOOL_SCHEMAS[n] for n in tool_names if n in TOOL_SCHEMAS]

        user = (f"TASK GIVEN TO THE ACCOUNTANT:\n{task_prompt}\n\n"
                f"THEIR SUBMITTED ANSWER JSON:\n"
                f"{json.dumps(answer, indent=1) if isinstance(answer, dict) else 'NO PARSEABLE ANSWER'}\n\n"
                f"THEIR NOTES:\n{(notes or '')[:1500]}\n\n"
                "Verify against the files, then output the JSON verdict.")
        messages = [{"role": "user", "content": user}]
        final_text = ""
        concluded = False
        for _ in range(max_steps):
            if _time.monotonic() > _deadline:
                break  # out of time — forced verdict below
            resp = client.chat(SYSTEM, messages, tools=tool_defs,
                               max_tokens=int(jcfg.get("max_tokens", 4000)))
            if resp.tool_calls:
                messages.append({"role": "assistant",
                                 "content": resp.text or "",
                                 "tool_calls": [
                                     {"id": t["id"], "type": "function",
                                      "function": {"name": t["name"],
                                                   "arguments": json.dumps(t["arguments"])}}
                                     for t in resp.tool_calls]})
                for t in resp.tool_calls:
                    fn = tools_impl.get(t["name"])
                    try:
                        out = fn(**t["arguments"]) if fn else f"unknown tool {t['name']}"
                    except Exception as e:
                        out = f"tool error: {e!r}"
                    messages.append({"role": "tool", "tool_call_id": t["id"],
                                     "content": str(out)[:8000]})
                continue
            final_text = resp.text or ""
            concluded = True
            break
        if not concluded:
            # step budget OR wall-clock reached — force a verdict from what it has
            messages.append({"role": "user",
                             "content": "Stop verifying and output the JSON "
                                        "verdict NOW based on what you checked."})
            final_text = client.chat(SYSTEM, messages, tools=None,
                                     max_tokens=2000).text or ""

        out = _extract_json(final_text)
        # deepseek often deliberates in plain text and truncates at max_tokens
        # before emitting the verdict — nudge it to conclude from what it has
        for _ in range(2):
            if out:
                break
            messages.append({"role": "assistant", "content": final_text})
            messages.append({"role": "user",
                             "content": "Output ONLY the JSON verdict now — "
                                        "no further analysis or prose."})
            final_text = client.chat(SYSTEM, messages, tools=None,
                                     max_tokens=2500).text or ""
            out = _extract_json(final_text)
        if not out:
            return {"score": 0.0, "critique":
                    "verifying judge produced no parseable verdict", "criteria": []}
        score = max(0.0, min(10.0, float(out.get("score") or 0.0)))
        return {"score": score,
                "critique": str(out.get("critique") or "")[:4000],
                "criteria": out.get("criteria") or []}
    except Exception as e:
        return {"score": 0.0,
                "critique": f"verifying judge failed: {e!r}", "criteria": []}
