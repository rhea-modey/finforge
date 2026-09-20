"""JSONL trace recording + deterministic summarization (DESIGN section 7.3).

Trace record shape: {ts, actor, event, detail, tokens, usd} with event in
{llm_call, tool_call, tool_result, submit}. summarize_trace is a pure
function of the trace file (no LLM), used verbatim by judges and the
optimizer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class TraceWriter:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._closed = False

    def event(self, actor: str, event: str, detail: dict,
              tokens: dict | None = None, usd: float | None = None) -> None:
        if self._closed:
            return
        rec = {
            "ts": round(time.time(), 3),
            "actor": actor,
            "event": event,
            "detail": detail,
            "tokens": tokens,
            "usd": round(float(usd), 6) if usd is not None else None,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# summarization
# ---------------------------------------------------------------------------

def _load_events(trace_path: str) -> list[dict]:
    events: list[dict] = []
    try:
        with open(trace_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):
                    events.append(ev)
    except OSError:
        pass
    return events


def _tok(tokens: dict | None, *keys: str) -> int:
    t = tokens or {}
    for k in keys:
        v = t.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def summarize_trace(trace_path: str, max_chars: int = 6000) -> str:
    """Deterministic digest: agents called in order, tool calls with 80-char
    argument previews and result sizes, the submitted answer compacted, and
    per-actor token/usd totals. Always <= max_chars."""
    events = _load_events(trace_path)

    agents_called: list[str] = []
    tool_lines: list[str] = []
    submit_answer = None
    submit_notes = ""
    totals: dict[str, dict] = {}
    actor_order: list[str] = []
    pending: dict[tuple[str, str], list[str]] = {}  # (actor, tool) -> arg previews

    for ev in events:
        actor = str(ev.get("actor", "?"))
        etype = ev.get("event")
        detail = ev.get("detail") or {}
        if not isinstance(detail, dict):
            detail = {"detail": str(detail)}
        if etype == "llm_call":
            if actor not in totals:
                totals[actor] = {"in": 0, "out": 0, "usd": 0.0, "calls": 0}
                actor_order.append(actor)
            t = totals[actor]
            t["in"] += _tok(ev.get("tokens"), "in", "input", "prompt_tokens")
            t["out"] += _tok(ev.get("tokens"), "out", "output", "completion_tokens")
            t["usd"] += float(ev.get("usd") or 0.0)
            t["calls"] += 1
        elif etype == "tool_call":
            tool = str(detail.get("tool", "?"))
            args_prev = str(detail.get("args_preview", ""))[:80]
            pending.setdefault((actor, tool), []).append(args_prev)
            if tool == "call_agent":
                m = args_prev
                # args_preview is compact JSON like {"name":"FileScout",...}
                name = ""
                marker = '"name":'
                if marker in m:
                    rest = m.split(marker, 1)[1].lstrip().lstrip('"')
                    name = rest.split('"', 1)[0]
                agents_called.append(name or "?")
        elif etype == "tool_result":
            tool = str(detail.get("tool", "?"))
            chars = detail.get("chars", "?")
            queue = pending.get((actor, tool)) or [""]
            args_prev = queue.pop(0) if queue else ""
            tool_lines.append(f"[{actor}] {tool}({args_prev}) -> {chars} chars")
        elif etype == "submit":
            submit_answer = detail.get("answer")
            submit_notes = str(detail.get("notes", ""))

    lines: list[str] = [f"TRACE SUMMARY: {Path(trace_path).name}"]
    if submit_answer is not None:
        compact = json.dumps(submit_answer, separators=(",", ":"), default=str)
        if len(compact) > 1800:
            compact = compact[:1800] + "...[truncated]"
        lines.append(f"SUBMITTED ANSWER: {compact}")
        if submit_notes:
            notes = submit_notes if len(submit_notes) <= 300 else submit_notes[:300] + "..."
            lines.append(f"NOTES: {notes}")
    else:
        lines.append("SUBMITTED ANSWER: none (run did not submit)")
    lines.append("AGENTS CALLED (in order): "
                 + (", ".join(agents_called) if agents_called else "(none)"))
    lines.append("PER-ACTOR TOTALS:")
    grand = {"in": 0, "out": 0, "usd": 0.0}
    for actor in actor_order:
        t = totals[actor]
        grand["in"] += t["in"]
        grand["out"] += t["out"]
        grand["usd"] += t["usd"]
        lines.append(f"  {actor}: llm_calls={t['calls']} in={t['in']} "
                     f"out={t['out']} usd=${t['usd']:.4f}")
    lines.append(f"  TOTAL: in={grand['in']} out={grand['out']} "
                 f"usd=${grand['usd']:.4f}")
    lines.append(f"TOOL CALLS ({len(tool_lines)}):")

    # fill tool lines within budget (reserve room for an omission note)
    head = "\n".join(lines)
    budget = max_chars - len(head) - 1
    included: list[str] = []
    omitted = 0
    reserve = 40
    used = 0
    for i, tl in enumerate(tool_lines):
        cost = len(tl) + 1
        remaining_after = budget - used - cost
        if remaining_after < (reserve if i < len(tool_lines) - 1 else 0):
            omitted = len(tool_lines) - i
            break
        included.append(tl)
        used += cost
    if omitted:
        included.append(f"  ... {omitted} more tool calls omitted")
    out = head + ("\n" + "\n".join(included) if included else "")
    if len(out) > max_chars:
        out = out[: max_chars - len("[truncated]")] + "[truncated]"
    return out
