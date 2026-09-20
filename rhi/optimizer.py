"""RHI optimizer (DESIGN section 9): meta-prompt -> new harness spec.

propose_revision(history, current_spec_text, cfg) -> (new_spec_text, rationale)

history item: {"iteration", "version", "mean_judge", "per_category",
               "critiques": [str], "pairwise": [dict], "trace_summaries": [str],
               "spec_text": str, "delta_vs_prev": float|None, "regressed": bool}

Guarantees the returned spec parses and validates via runtime.spec
(parse_harness + validate_harness), retrying up to 3 times with the SpecError
text fed back; on exhaustion it falls back to current_spec_text and says so
in the rationale (a no-op revision -- never raises past the gold guard,
except for genuine environment errors such as runtime.spec being missing).

The optimizer NEVER sees gold, rubrics, held-out anything, or gold scores; a
defensive guard rejects any history dict containing a gold-flavored key.
"""

from __future__ import annotations

import re
from collections import Counter

# --------------------------------------------------------------------------
# Guards + small helpers (self-contained; rhi does not depend on judge/)
# --------------------------------------------------------------------------


def _assert_no_gold(obj, where: str) -> None:
    """Recursively reject any dict key that looks like gold/rubric leakage.

    Raises ValueError -- a contract-violation guard (DESIGN section 9: the
    optimizer never sees gold, rubrics, or gold scores)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key == "gold" or key.startswith("gold_") or key.endswith("_gold") or "rubric" in key:
                raise ValueError(
                    f"gold-blind violation: input '{where}' contains forbidden key '{k}' "
                    "(gold answers, gold scores, and rubrics must never reach the optimizer)"
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


def _one_line(s) -> str:
    return " ".join(("" if s is None else str(s)).split())


def _fmt_score(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "?"


# --------------------------------------------------------------------------
# Meta-prompt sections (DESIGN section 9 order)
# --------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert architect of multi-agent LLM workflows, serving as the "
    "optimizer in a recursive self-improvement loop. Follow the required "
    "output format exactly."
)

_RHI_FRAMING = """== YOUR JOB: RECURSIVE HARNESS IMPROVEMENT ==
A multi-agent "harness" -- defined entirely by the prompt-level spec below --
performs month-end-close accounting tasks (bank reconciliation, journal
entries, variance analysis, accrual/prepaid schedules) on a directory of messy
exported company files. Its agents are LLMs with file tools; the orchestrator
executes the WORKFLOW by calling agents and finally submitting an answer JSON.
Your job is to rewrite the spec so the NEXT iteration performs better.

You improve the harness's ROLES, INSTRUCTIONS, CONTRACTS, WORKFLOW hops, and
AUXILIARY RULES. Prioritize, in roughly this order:
1. CONTRACTS -- exactly what each agent must return. The dominant failure mode
   is agents returning vague summaries instead of the concrete data (file
   contents, column layouts, exact figures, file paths) downstream agents need.
2. INFORMATION FLOW between agents -- what the orchestrator sends each agent
   and what it passes along; agents share no memory beyond message contents.
3. ROLES and INSTRUCTIONS of each agent (and the orchestrator).
4. WORKFLOW hops -- order, branching, retries, verification passes.
5. AUXILIARY RULES -- acceptance gates, fallbacks, recall triggers.

You may add, remove, merge, or rename agents. You may (and often should)
direct agents to use the `run_python` tool for arithmetic, cross-footing,
totaling, and searching/parsing files -- LLM mental arithmetic is unreliable.

Make 1-3 TARGETED changes per revision and preserve structure that is
working; contracts and information flow are the priority axis. The revision
history below includes the full spec text of every prior version: you may
base the next revision on ANY prior version -- reverting to or branching from
an earlier spec is a legitimate move, especially after a regression."""

# The spec grammar of DESIGN section 6, embedded as text so the optimizer can
# comply with it. (No inline arrow-comments: models copy them literally into
# the spec, which then fails validation and burns a retry.)
_SPEC_GRAMMAR = """== HARNESS SPEC GRAMMAR (your output MUST follow this exactly) ==
A harness is ONE markdown file, strictly this shape (a validator enforces it):

# HARNESS <version-label>

## ORCHESTRATOR
tools: none
max_steps: 40
instructions: |
  <free text: how to run the workflow, what to send agents, when to submit>

## AGENTS

### <AgentName>
tools: read_file, grep_files, list_files, run_python
max_steps: 25
role: <one line>
instructions: |
  <free text>
contract: |
  <free text: exactly what this agent must return to the orchestrator>

## WORKFLOW
<free text: the hops -- order, branching, retries. Executed by the orchestrator.>

## AUXILIARY RULES
<free text or `none`: acceptance gates, fallbacks, recall triggers>

Notes on the grammar (do NOT copy these notes into the spec):
- `tools:` is a comma-separated subset of the tool roster, or the word `none`.
- `<AgentName>` must match [A-Za-z][A-Za-z0-9_]{1,30}; one `### <AgentName>`
  block per agent, each with tools, max_steps, role, instructions, contract
  in that order.
- Tool roster (the ONLY tools that exist): `list_files`, `read_file`,
  `grep_files`, `run_python`. The orchestrator additionally always has
  `call_agent` and `submit_answer` (implicit; never listed under `tools:`).
- The validator enforces: known tools only, 1-8 agents, max_steps 1-60,
  non-empty instructions and contract for every agent."""

_OUTPUT_SHAPE = """== REQUIRED OUTPUT SHAPE ==
Produce, in this order:
1. REFLECTION (plain prose BEFORE the fence; no fenced blocks in it):
   (a) the top 2-4 recurring failure modes across this iteration's evidence,
   each with a short quoted fragment from a critique or trace summary;
   (b) the spec component each failure maps to (ROLES / INSTRUCTIONS /
   CONTRACTS / WORKFLOW / AUXILIARY RULES);
   (c) the 1-3 changes you chose and which failure mode each one answers.
2. ONE ```markdown fenced block containing the COMPLETE new spec (nothing
   omitted, no placeholders, no ellipses). Label its first line exactly
   `# HARNESS {label}`.
3. After the closing fence, a line starting `RATIONALE:` followed by a short
   summary of what changed and why, NAMING the parent version this revision
   is based on (usually the current version; any prior version if reverting)."""

_HARD_RULES = """== HARD RULES ==
1. NEVER reference specific worlds, companies, vendors, employees, dollar
   amounts, dates, or concrete file names taken from the evidence above.
   Every iteration runs on FRESH, different synthetic worlds, so
   world-specific instructions are useless or actively harmful. Keep every
   instruction generic to the task categories.
2. Keep at most 8 agents. max_steps must be 1-60. Use only tools from the
   roster. Every agent needs non-empty instructions and contract.
3. Exactly ONE fenced block in your whole reply: the ```markdown fence with
   the complete spec. Reflection prose before it, `RATIONALE:` prose after
   it, no other fences anywhere."""


def _regression_line(h: dict) -> str | None:
    if not h.get("regressed"):
        return None
    delta = h.get("delta_vs_prev")
    try:
        dtxt = f"{float(delta):+.2f}"
    except (TypeError, ValueError):
        dtxt = "negative"
    return (f"REGRESSION: mean judge score dropped vs the previous iteration "
            f"(delta {dtxt}).")


def _render_pairwise(pairwise: list, indent: str) -> list[str]:
    lines = []
    pairwise = [p for p in (pairwise or []) if isinstance(p, dict)]
    if not pairwise:
        return lines
    counts = Counter(str(p.get("winner", "?")) for p in pairwise)
    lines.append(indent + "pairwise vs previous iteration: " + ", ".join(
        f"{w}={n}" for w, n in sorted(counts.items())))
    for p in pairwise[:4]:
        advice = _one_line(p.get("transferable_advice"))
        if advice:
            lines.append(indent + "  advice: " + _truncate(advice, 240))
    return lines


def _compact_history(history: list[dict]) -> str:
    """Digest of all PAST iterations (everything but the last item),
    including each version's full spec text (genealogy: DESIGN 9.1 revert
    rights need the optimizer to actually see prior specs)."""
    past = history[:-1]
    if not past:
        return "(no prior iterations)"
    blocks = []
    for h in past:
        h = h or {}
        lines = [
            f"- iteration {h.get('iteration', '?')} | version {h.get('version', '?')} | "
            f"mean judge score {_fmt_score(h.get('mean_judge'))}"
        ]
        reg = _regression_line(h)
        if reg:
            lines.append("  " + reg)
        per_cat = h.get("per_category") or {}
        if isinstance(per_cat, dict) and per_cat:
            lines.append("  per-category means: " + ", ".join(
                f"{c}={_fmt_score(v)}" for c, v in sorted(per_cat.items())))
        critiques = [c for c in (h.get("critiques") or []) if c]
        if critiques:
            lines.append("  top critique themes:")
            for c in critiques[:3]:
                lines.append("    * " + _truncate(_one_line(c), 240))
        lines.extend(_render_pairwise(h.get("pairwise"), "  "))
        spec_text = str(h.get("spec_text") or "").strip()
        if spec_text:
            lines.append("  full spec of this version (indented):")
            for ln in _truncate(spec_text, 3500).splitlines():
                lines.append("    | " + ln)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _current_evidence(history: list[dict]) -> str:
    """Full critiques + up to 4 truncated trace summaries for THIS iteration
    (the last history item)."""
    if not history:
        return "(no evidence available for this iteration)"
    h = history[-1] or {}
    lines = [
        f"iteration {h.get('iteration', '?')} | version {h.get('version', '?')} | "
        f"mean judge score {_fmt_score(h.get('mean_judge'))}"
    ]
    reg = _regression_line(h)
    if reg:
        lines.append(reg + " Diagnose explicitly whether the last change "
                     "caused the drop or it is noise, and consider reverting "
                     "to (or branching from) the parent or an earlier version.")
    per_cat = h.get("per_category") or {}
    if isinstance(per_cat, dict) and per_cat:
        lines.append("per-category means: " + ", ".join(
            f"{c}={_fmt_score(v)}" for c, v in sorted(per_cat.items())))
    pw_lines = _render_pairwise(h.get("pairwise"), "")
    if pw_lines:
        lines.append("")
        lines.append("-- Pairwise comparison of THIS iteration vs the previous one --")
        lines.extend(pw_lines)
        for p in [p for p in (h.get("pairwise") or []) if isinstance(p, dict)][:4]:
            reasons = _one_line(p.get("reasons"))
            if reasons:
                lines.append(f"  [{p.get('category', '?')}] winner="
                             f"{p.get('winner', '?')}: " + _truncate(reasons, 400))
    critiques = [c for c in (h.get("critiques") or []) if c]
    lines.append("")
    lines.append("-- Gold-blind judge critiques for this iteration's tasks (full) --")
    if critiques:
        for i, c in enumerate(critiques):
            lines.append(f"[critique {i + 1}]\n{c}")
    else:
        lines.append("(no critiques available)")
    summaries = [t for t in (h.get("trace_summaries") or []) if t]
    lines.append("")
    lines.append("-- Trace summaries (up to 4, each truncated) --")
    if summaries:
        for i, t in enumerate(summaries[:4]):
            lines.append(f"[trace {i + 1}]\n{_truncate(t, 4000)}")
    else:
        lines.append("(no trace summaries available)")
    return "\n".join(lines)


def _next_label(history: list[dict]) -> str:
    """Sequential next version label from the latest history item (DESIGN
    9.1: labels stay sequential regardless of parentage)."""
    ver = str(((history[-1] if history else {}) or {}).get("version") or "")
    m = re.match(r"^v(\d+)$", ver)
    if m:
        return f"v{int(m.group(1)) + 1}"
    return ""


_SURGICAL_EDITS = """== EDIT DISCIPLINE (binding for this revision) ==
The revision history shows a recurring failure mode: a rewrite that lifts one
task category while silently regressing another. To prevent it:
1. From the critiques, name the SINGLE weakest category and the concrete
   failure (wrong-period totals, fabricated citations, renamed fields, ...).
2. Edit ONLY the sections that govern that failure. Sections serving
   categories at or above their historical best must be preserved VERBATIM
   from the best-scoring ancestor spec — copy them, do not paraphrase.
3. Every changed line must be justified by a specific critique; if you cannot
   cite one, revert that change.
4. A smaller diff that fixes the named failure beats a broader rewrite."""


def _build_meta_prompt(history: list[dict], current_spec_text: str,
                       ocfg: dict | None = None) -> str:
    label = _next_label(history)
    shape = _OUTPUT_SHAPE.replace(
        "{label}", label or "<the next sequential version label>")
    parts = [
        _RHI_FRAMING,
        _SPEC_GRAMMAR,
        "== CURRENT HARNESS SPEC ==\n" + (current_spec_text or "").strip(),
        "== REVISION HISTORY (compact, oldest first) ==\n" + _compact_history(history),
        "== THIS ITERATION'S EVIDENCE ==\n" + _current_evidence(history),
    ]
    if (ocfg or {}).get("surgical_edits"):
        parts.append(_SURGICAL_EDITS)
    parts.extend([shape, _HARD_RULES])
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Output extraction
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:markdown|md)?[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop <think>...</think> reasoning blocks (reasoning models such as
    MiniMax-M3 emit them inline in message.content, and a draft spec fence
    inside the think block must never be mistaken for the real one). A stray
    unmatched </think> drops everything up to and including it."""
    out = _THINK_BLOCK_RE.sub("", text or "")
    low = out.lower()
    idx = low.rfind("</think>")
    if idx != -1:
        out = out[idx + len("</think>"):]
    return out


def _extract_spec_and_rationale(text: str) -> tuple[str, str]:
    """Pull the fenced spec, the pre-fence reflection, and the trailing
    RATIONALE prose from optimizer output (think blocks stripped first).
    The reflection is captured into the rationale (DESIGN 9.1). Raises
    ValueError with feedback-worthy text on failure."""
    if not text or not text.strip():
        raise ValueError("empty output -- expected reflection prose, one ```markdown "
                         "fence with the full spec, then RATIONALE:")
    cleaned = _strip_think(text)
    matches = list(_FENCE_RE.finditer(cleaned))
    if not matches:
        raise ValueError("no ```markdown fenced block found -- the complete spec must be inside "
                         "one ```markdown fence, followed by RATIONALE: prose")
    # Prefer fences that actually contain a spec; take the LAST such fence
    # (any earlier one is a quoted draft, not the final revision).
    spec_matches = [m for m in matches
                    if m.group(1).lstrip().startswith("# HARNESS")]
    m = (spec_matches or matches)[-1]
    spec_text = m.group(1).strip()
    if not spec_text:
        raise ValueError("the ```markdown fence was empty -- it must contain the complete spec")
    spec_text += "\n"
    reflection = _FENCE_RE.sub("", cleaned[:m.start()]).strip()
    tail = cleaned[m.end():]
    rm = re.search(r"RATIONALE\s*:\s*", tail, re.IGNORECASE)
    rationale = (tail[rm.end():] if rm else tail).strip()
    if reflection:
        rationale = reflection + ("\n\nRATIONALE: " + rationale if rationale else "")
    return spec_text, rationale


# --------------------------------------------------------------------------
# Public interface (pinned, DESIGN section 14)
# --------------------------------------------------------------------------


def propose_revision(history: list[dict], current_spec_text: str,
                     cfg: dict) -> tuple[str, str]:
    """Propose a revised harness spec (DESIGN section 9).

    Returns (new_spec_text, rationale). The returned spec is guaranteed to
    parse and validate; on repeated failure the current spec is returned
    unchanged with an explanatory rationale.
    """
    history = list(history or [])
    for i, h in enumerate(history):
        _assert_no_gold(h, f"history[{i}]")

    # Environment dependencies -- imported lazily so the module is importable
    # while sibling modules are still being built. If runtime.spec itself is
    # missing we cannot honor the validation guarantee, so that ImportError
    # propagates.
    from runtime.spec import SpecError, parse_harness, validate_harness

    ocfg = (cfg or {}).get("optimizer", {}) or {}
    try:
        max_tokens = int(ocfg.get("max_tokens", 16000) or 16000)
    except (TypeError, ValueError):
        max_tokens = 16000

    try:
        from runtime.llm import LLMClient
        client = LLMClient.for_role("optimizer", cfg)
    except Exception as e:  # noqa: BLE001
        return current_spec_text, (
            f"Optimizer unavailable (could not initialize optimizer LLM client: {e}); "
            "keeping the previous spec unchanged (no-op revision).")

    messages = [{"role": "user",
                 "content": _build_meta_prompt(history, current_spec_text,
                                               ocfg)}]
    max_retries = 3
    last_err = None
    for _attempt in range(1 + max_retries):
        try:
            resp = client.chat(system=_SYSTEM, messages=messages, max_tokens=max_tokens)
            out = resp.text or ""
        except Exception as e:  # noqa: BLE001 -- transport failure: burn the attempt
            last_err = f"optimizer LLM call failed: {_truncate(str(e), 300)}"
            continue
        try:
            spec_text, rationale = _extract_spec_and_rationale(out)
            spec = parse_harness(spec_text)
            validate_harness(spec)
            # DESIGN 9.1: the version label stays sequential regardless of
            # parentage -- rewrite the header deterministically if needed.
            expected = _next_label(history)
            if expected and spec.version_label != expected:
                spec_text = re.sub(r"^# HARNESS\s+\S.*$",
                                   f"# HARNESS {expected}", spec_text,
                                   count=1, flags=re.MULTILINE)
                spec = parse_harness(spec_text)
                validate_harness(spec)
        except (ValueError, SpecError) as e:
            last_err = str(e)
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": (
                f"Your previous output was rejected: {last_err}\n"
                "Re-output the corrected, COMPLETE spec now, following the grammar and the "
                "hard rules exactly: one ```markdown fenced block containing the whole spec, "
                "then `RATIONALE:` prose after the closing fence. Nothing else.")})
            continue
        except Exception as e:  # noqa: BLE001 -- unexpected validator failure: same feedback path
            last_err = f"spec validation crashed: {_truncate(str(e), 300)}"
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": (
                f"Your previous output was rejected: {last_err}\n"
                "Re-output the corrected, COMPLETE spec: one ```markdown fence, then RATIONALE: prose.")})
            continue
        if not rationale:
            rationale = "(optimizer provided no rationale)"
        return spec_text, rationale

    return current_spec_text, (
        f"Optimizer failed to produce a valid spec after {1 + max_retries} attempts "
        f"(last error: {last_err}); keeping the previous spec unchanged (no-op revision).")
