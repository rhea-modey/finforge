"""Spec interpreter: orchestrator + subagent loops (DESIGN sections 7.2, 14).

run_task executes one task under a HarnessSpec: a manual tool loop for the
orchestrator (call_agent + submit_answer + any file tools its spec grants),
fresh-context subagent loops with their granted tools, max_steps enforcement
with a forced final-submit nudge, and full token/usd accounting into both the
trace and the TaskRunResult. A bad tool call never crashes the run — errors
come back to the model as tool results.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from runtime import tools as tools_mod
from runtime.llm import METER, LLMClient
from runtime.spec import AgentDef, HarnessSpec
from runtime.trace import TraceWriter

_PREVIEW = 80

_FORCED_SUBMIT_MSG = ("Step limit reached — submit your best answer now with "
                      "submit_answer.")

_ORCH_PREAMBLE = """\
You are the ORCHESTRATOR of a multi-agent workflow solving a month-end-close \
accounting task for one company, using only the tools provided.

Protocol:
- call_agent(name, message) runs the named agent in a FRESH context and \
returns its final reply. Agents share no memory with you or with each other, \
so every message must contain all the context the agent needs: what the task \
is, relevant filenames or numbers you already know, and exactly what the \
agent must return.
- You MUST finish by calling submit_answer(answer_json, notes). answer_json \
must be a JSON string matching the deliverable schema in the task EXACTLY; \
put narrative in notes.
- Follow the workflow and auxiliary rules below. Do not invent file contents \
or figures yourself; get them through your agents and tools.\
"""


@dataclass
class TaskRunResult:
    answer: dict | None
    notes: str
    ended: str          # submitted|max_steps|budget|error
    steps: int
    usd: float
    tokens: dict
    wallclock_s: float
    trace_path: str


class _BudgetExceeded(Exception):
    pass


# ---------------------------------------------------------------------------
# run accounting
# ---------------------------------------------------------------------------

class _RunState:
    def __init__(self, cfg: dict) -> None:
        # The driver injects "_effective_budget_usd_cap" = cap minus spend
        # recorded by PRIOR sessions, so mid-task enforcement on a resumed run
        # is as tight as the driver-level pre-task check. Fallback: env/config.
        cap = cfg.get("_effective_budget_usd_cap")
        if cap is None:
            cap_env = os.environ.get("FINFORGE_BUDGET_CAP")
            cap = cap_env if cap_env is not None else cfg.get("budget_usd_cap")
        try:
            self.cap = float(cap) if cap is not None else None
        except (TypeError, ValueError):
            self.cap = None
        self.steps = 0  # total model calls across orchestrator + subagents
        self.usd = 0.0
        self.tokens = {"in": 0, "out": 0, "cache_r": 0, "cache_w": 0}

    def check_budget(self) -> None:
        if self.cap is not None and METER.over_cap(self.cap):
            raise _BudgetExceeded()

    def record(self, usage: dict, usd: float) -> None:
        self.steps += 1
        self.usd += usd
        self.tokens["in"] += int(usage.get("input") or usage.get("in") or 0)
        self.tokens["out"] += int(usage.get("output") or usage.get("out") or 0)
        self.tokens["cache_r"] += int(usage.get("cache_r") or 0)
        self.tokens["cache_w"] += int(usage.get("cache_w") or 0)


# ---------------------------------------------------------------------------
# prompt assembly
# ---------------------------------------------------------------------------

def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _orchestrator_system(spec: HarnessSpec, task_prompt: str, task_id: str) -> str:
    roster = "\n".join(
        f"- {a.name} | role: {_one_line(a.role)} | contract: {_one_line(a.contract)}"
        for a in spec.agents.values())
    return (f"{_ORCH_PREAMBLE}\n\n"
            f"Task ID: {task_id}\n\n"
            f"ORCHESTRATOR INSTRUCTIONS:\n{spec.orchestrator.instructions}\n\n"
            f"WORKFLOW:\n{spec.workflow}\n\n"
            f"AUXILIARY RULES:\n{spec.aux}\n\n"
            f"AGENT ROSTER:\n{roster}\n\n"
            f"TASK:\n{task_prompt}")


def _subagent_system(agent: AgentDef) -> str:
    tools_line = ", ".join(agent.tools) if agent.tools else "none"
    return (f"You are agent {agent.name} in a multi-agent accounting close "
            f"workflow. The orchestrator's message below is your task. Use "
            f"your tools to inspect the company's exported files; when you "
            f"are done, reply with plain text (no tool call) that fulfils "
            f"your contract — that reply goes back to the orchestrator "
            f"verbatim and you will get no follow-up questions.\n\n"
            f"ROLE: {agent.role}\n\n"
            f"INSTRUCTIONS:\n{agent.instructions}\n\n"
            f"CONTRACT (exactly what you must return):\n{agent.contract}\n\n"
            f"WORLD: the company's exported close files are available through "
            f"your tools ({tools_line}); paths are relative to the task "
            f"working directory.")


def _preview_args(args) -> str:
    try:
        s = json.dumps(args, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        s = str(args)
    return s[:_PREVIEW]


def _assistant_msg(resp) -> dict:
    m: dict = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        m["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"],
                          "arguments": json.dumps(tc["arguments"], default=str)}}
            for tc in resp.tool_calls]
    return m


# ---------------------------------------------------------------------------
# model + tool plumbing
# ---------------------------------------------------------------------------

def _model_call(client, state: _RunState, trace: TraceWriter, actor: str,
                system: str, messages: list[dict], tool_schemas: list[dict]):
    state.check_budget()
    resp = client.chat(system, messages, tools=tool_schemas or None)
    usd = METER.estimate_usd(getattr(client, "model", ""), resp.usage)
    state.record(resp.usage, usd)
    trace.event(actor, "llm_call",
                {"text_preview": (resp.text or "")[:_PREVIEW],
                 "n_tool_calls": len(resp.tool_calls)},
                tokens=resp.usage, usd=usd)
    return resp


def _exec_file_tool(toolbox: dict, granted, tc: dict, trace: TraceWriter,
                    actor: str) -> str:
    """Run one file tool IF granted to this actor; never raises. Grants are
    enforced here (the toolbox binds all roster tools, but an agent may only
    execute the tools its spec lists). Traces call + result."""
    name, args = tc["name"], tc["arguments"]
    granted = set(granted or ())
    trace.event(actor, "tool_call", {"tool": name, "args_preview": _preview_args(args)})
    if not isinstance(args, dict) or "_malformed_arguments" in args:
        result = (f"Error: malformed JSON arguments for {name}; retry with "
                  f"valid JSON arguments.")
    elif name not in toolbox or name not in granted:
        result = (f"Error: unknown or ungranted tool {name!r}. Your tools: "
                  f"{', '.join(sorted(granted)) or 'none'}.")
    else:
        try:
            result = toolbox[name](**args)
        except TypeError as exc:
            result = f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
            result = f"Error: tool {name} failed: {type(exc).__name__}: {exc}"
    result = tools_mod.truncate_result(str(result))
    trace.event(actor, "tool_result",
                {"tool": name, "chars": len(result), "preview": result[:_PREVIEW]})
    return result


def _handle_submit_args(args) -> tuple[dict | None, str, str]:
    """Returns (answer|None, notes, tool_result_text). Parse errors go back
    to the model as the tool result so it can retry."""
    if not isinstance(args, dict):
        return None, "", ("Error: submit_answer needs an 'answer_json' argument "
                          "(a JSON string matching the task schema). Retry.")
    notes = args.get("notes")
    notes = "" if notes is None else str(notes)
    raw = args.get("answer_json")
    if isinstance(raw, dict):  # tolerate models passing the object directly
        return raw, notes, "Answer submitted. The task run is complete."
    if not isinstance(raw, str):
        return None, notes, ("Error: submit_answer requires 'answer_json' as a "
                             "JSON string matching the task schema. Call "
                             "submit_answer again with answer_json set.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, notes, (f"Error: answer_json is not valid JSON: {exc}. "
                             f"Fix the JSON and call submit_answer again.")
    if not isinstance(parsed, dict):
        return None, notes, (f"Error: answer_json must encode a JSON object, "
                             f"got {type(parsed).__name__}. Call submit_answer "
                             f"again with a JSON object.")
    return parsed, notes, "Answer submitted. The task run is complete."


# ---------------------------------------------------------------------------
# subagent loop
# ---------------------------------------------------------------------------

def _run_subagent(agent: AgentDef, client, toolbox: dict, state: _RunState,
                  trace: TraceWriter, message: str) -> str:
    system = _subagent_system(agent)
    schemas = [tools_mod.TOOL_SCHEMAS[t] for t in agent.tools
               if t in tools_mod.TOOL_SCHEMAS]
    messages: list[dict] = [{"role": "user", "content": str(message)}]
    last_text = ""
    nudges = 0
    for _ in range(agent.max_steps):
        resp = _model_call(client, state, trace, agent.name, system, messages, schemas)
        if resp.text:
            last_text = resp.text
        if not resp.tool_calls:
            if not (resp.text or last_text) and nudges < 2:
                # Empty turn (no text, no tool calls) — nudge instead of
                # returning nothing to the orchestrator.
                nudges += 1
                messages.append({"role": "assistant", "content": "(no content)"})
                messages.append({"role": "user", "content":
                                 "You returned no text. Reply NOW with plain "
                                 "text fulfilling your contract."})
                continue
            return resp.text or last_text or "(agent returned no text)"
        messages.append(_assistant_msg(resp))
        for tc in resp.tool_calls:  # all parallel calls answered before next turn
            result = _exec_file_tool(toolbox, agent.tools, tc, trace, agent.name)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": result})
    return last_text or "(agent hit its step limit without a final reply)"


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------

def run_task(spec: HarnessSpec, world_files_dir: str, task_prompt: str,
             task_id: str, cfg: dict, results_dir: str) -> TaskRunResult:
    t0 = time.monotonic()
    run_id = f"{task_id}-{int(time.time() * 1000)}"
    trace_path = str(Path(results_dir) / "traces" / f"{run_id}.jsonl")
    trace = TraceWriter(trace_path)
    state = _RunState(cfg)
    answer: dict | None = None
    notes = ""
    ended = "error"

    try:
        client = LLMClient.for_role("actor", cfg)
        toolbox = tools_mod.make_toolbox(
            world_files_dir, int(cfg.get("run_python_timeout_s", 20) or 20))
        orch = spec.orchestrator
        system = _orchestrator_system(spec, task_prompt, task_id)
        schemas = ([tools_mod.TOOL_SCHEMAS[t] for t in orch.tools
                    if t in tools_mod.TOOL_SCHEMAS]
                   + [tools_mod.TOOL_SCHEMAS["call_agent"],
                      tools_mod.TOOL_SCHEMAS["submit_answer"]])
        messages: list[dict] = [{
            "role": "user",
            "content": ("Begin. Follow the workflow, then submit your final "
                        "answer with submit_answer.")}]

        def run_round(resp, allow_agents: bool) -> bool:
            """Answer every tool call of one assistant turn; True if submitted."""
            nonlocal answer, notes
            if not resp.tool_calls:
                messages.append({"role": "assistant", "content": resp.text or ""})
                messages.append({
                    "role": "user",
                    "content": ("Continue. Use call_agent to run the workflow, "
                                "and finish by calling submit_answer.")})
                return False
            messages.append(_assistant_msg(resp))
            done = False
            for tc in resp.tool_calls:
                name, args = tc["name"], tc["arguments"]
                if done:
                    result = "Run already ended: an answer was submitted."
                    trace.event("orchestrator", "tool_call",
                                {"tool": name, "args_preview": _preview_args(args)})
                    trace.event("orchestrator", "tool_result",
                                {"tool": name, "chars": len(result),
                                 "preview": result[:_PREVIEW]})
                elif name == "submit_answer":
                    trace.event("orchestrator", "tool_call",
                                {"tool": name, "args_preview": _preview_args(args)})
                    ans, nts, result = _handle_submit_args(args)
                    trace.event("orchestrator", "tool_result",
                                {"tool": name, "chars": len(result),
                                 "preview": result[:_PREVIEW]})
                    if ans is not None:
                        answer, notes = ans, nts
                        trace.event("orchestrator", "submit",
                                    {"answer": ans, "notes": nts})
                        done = True
                elif name == "call_agent" and allow_agents:
                    trace.event("orchestrator", "tool_call",
                                {"tool": name, "args_preview": _preview_args(args)})
                    if not isinstance(args, dict) or "_malformed_arguments" in args:
                        result = ("Error: malformed JSON arguments for "
                                  "call_agent; retry with valid JSON.")
                    else:
                        agent_name = str(args.get("name") or "")
                        if agent_name not in spec.agents:
                            result = (f"Error: unknown agent {agent_name!r}. "
                                      f"Available agents: "
                                      f"{', '.join(spec.agents)}.")
                        else:
                            try:
                                result = _run_subagent(
                                    spec.agents[agent_name], client, toolbox,
                                    state, trace,
                                    str(args.get("message") or ""))
                            except _BudgetExceeded:
                                raise
                            except Exception as exc:  # noqa: BLE001
                                result = (f"Error: agent {agent_name} failed: "
                                          f"{type(exc).__name__}: {exc}")
                    result = tools_mod.truncate_result(str(result))
                    trace.event("orchestrator", "tool_result",
                                {"tool": name, "chars": len(result),
                                 "preview": result[:_PREVIEW]})
                elif name == "call_agent":
                    result = ("Error: no more agent calls allowed — call "
                              "submit_answer now.")
                    trace.event("orchestrator", "tool_call",
                                {"tool": name, "args_preview": _preview_args(args)})
                    trace.event("orchestrator", "tool_result",
                                {"tool": name, "chars": len(result),
                                 "preview": result[:_PREVIEW]})
                else:
                    result = _exec_file_tool(toolbox, orch.tools, tc, trace,
                                             "orchestrator")
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": str(result)})
            return done

        submitted = False
        for _ in range(orch.max_steps):
            resp = _model_call(client, state, trace, "orchestrator",
                               system, messages, schemas)
            if run_round(resp, allow_agents=True):
                submitted = True
                break

        if not submitted:
            # one final forced chance: submit_answer only
            messages.append({"role": "user", "content": _FORCED_SUBMIT_MSG})
            resp = _model_call(client, state, trace, "orchestrator",
                               system, messages,
                               [tools_mod.TOOL_SCHEMAS["submit_answer"]])
            submitted = run_round(resp, allow_agents=False)

        ended = "submitted" if submitted else "max_steps"

    except _BudgetExceeded:
        ended = "budget"
        trace.event("orchestrator", "tool_result",
                    {"tool": "_run_end", "chars": 0,
                     "preview": "budget cap reached; run halted"})
    except Exception as exc:  # noqa: BLE001 - a task run never raises
        ended = "error"
        trace.event("orchestrator", "tool_result",
                    {"tool": "_run_error", "chars": 0,
                     "preview": f"{type(exc).__name__}: {exc}"[:200]})
    finally:
        trace.close()

    return TaskRunResult(
        answer=answer, notes=notes, ended=ended, steps=state.steps,
        usd=round(state.usd, 6), tokens=dict(state.tokens),
        wallclock_s=round(time.monotonic() - t0, 3), trace_path=trace_path)
