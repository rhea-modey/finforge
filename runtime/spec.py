"""Harness spec parser/validator (DESIGN section 6, pinned in section 14).

The grammar is strict on structure and tolerant of trailing whitespace.
SpecError messages are consumed verbatim by the optimizer as retry feedback,
so they say exactly what is wrong and what was expected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TOOL_ROSTER = ("list_files", "read_file", "grep_files", "run_python")
IMPLICIT_ORCHESTRATOR_TOOLS = ("call_agent", "submit_answer")

_AGENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,30}$")
_REQUIRED_SECTIONS = ["ORCHESTRATOR", "AGENTS", "WORKFLOW", "AUXILIARY RULES"]


class SpecError(ValueError):
    """Raised on any grammar or validation failure. Message is the feedback."""


@dataclass
class AgentDef:
    name: str
    tools: list[str]
    max_steps: int
    role: str
    instructions: str
    contract: str


@dataclass
class HarnessSpec:
    version_label: str
    orchestrator: AgentDef
    agents: dict[str, AgentDef]
    workflow: str
    aux: str
    source_text: str


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _dedent_block(block_lines: list[str]) -> str:
    non_blank = [ln for ln in block_lines if ln.strip()]
    if not non_blank:
        return ""
    indent = min(len(ln) - len(ln.lstrip()) for ln in non_blank)
    out = [ln[indent:] if ln.strip() else "" for ln in block_lines]
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return "\n".join(out)


def _section_text(lines: list[str]) -> str:
    out = list(lines)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _parse_tools(value: str, where: str) -> list[str]:
    v = value.strip()
    if not v:
        raise SpecError(f"{where}: 'tools:' must be 'none' or a comma-separated "
                        f"list of tool names.")
    if v.lower() == "none":
        return []
    names = [t.strip() for t in v.split(",")]
    if any(not t for t in names):
        raise SpecError(f"{where}: malformed 'tools:' list {value!r} — use "
                        f"comma-separated tool names or 'none'.")
    return names


def _parse_int(value: str, field: str, where: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        raise SpecError(f"{where}: '{field}:' must be an integer, got {value!r}.") from None


def _parse_fields(lines: list[str], wanted: list[tuple[str, str]], where: str) -> dict:
    """Parse `field: value` / `field: |` + indented block, in exactly the
    given order. `wanted` items are (field_name, "inline"|"block")."""
    i = 0
    out: dict[str, str] = {}
    for field, kind in wanted:
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise SpecError(f"{where}: missing field '{field}:'.")
        line = lines[i]
        m = re.match(rf"^{re.escape(field)}:\s*(.*)$", line)
        if not m:
            raise SpecError(f"{where}: expected '{field}:' next but found "
                            f"{line.strip()!r} (fields must appear in order: "
                            f"{', '.join(f for f, _ in wanted)}).")
        value = m.group(1)
        i += 1
        if kind == "inline":
            if not value.strip():
                raise SpecError(f"{where}: '{field}:' must have a non-empty value "
                                f"on the same line.")
            out[field] = value.strip()
        else:  # block scalar
            if value.strip() != "|":
                raise SpecError(f"{where}: '{field}' must be a block scalar — "
                                f"write '{field}: |' then indented lines below it.")
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip()
                                      or lines[i][:1] in (" ", "\t")):
                block.append(lines[i])
                i += 1
            text = _dedent_block(block)
            if not text.strip():
                raise SpecError(f"{where}: the '{field}: |' block is empty — it "
                                f"needs at least one indented line of text.")
            out[field] = text
    while i < len(lines):
        if lines[i].strip():
            raise SpecError(f"{where}: unexpected content after the last field: "
                            f"{lines[i].strip()!r}.")
        i += 1
    return out


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

def parse_harness(text: str) -> HarnessSpec:
    if not isinstance(text, str) or not text.strip():
        raise SpecError("Empty harness spec: expected markdown starting with "
                        "'# HARNESS <version-label>'.")
    lines = [ln.rstrip() for ln in text.splitlines()]

    # header
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    m = re.match(r"^# HARNESS\s+(\S.*)$", lines[idx])
    if not m:
        raise SpecError("First non-blank line must be '# HARNESS <version-label>', "
                        f"got {lines[idx].strip()!r}.")
    version_label = m.group(1).strip()

    # split into ## sections
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current: list[str] = []
    for ln in lines[idx + 1:]:
        if ln.startswith("## "):
            if current_title is not None:
                sections.append((current_title, current))
            current_title = ln[3:].strip()
            current = []
        elif current_title is None:
            if ln.strip():
                raise SpecError("Unexpected content before '## ORCHESTRATOR': "
                                f"{ln.strip()!r}.")
        else:
            current.append(ln)
    if current_title is not None:
        sections.append((current_title, current))

    titles = [t for t, _ in sections]
    if titles != _REQUIRED_SECTIONS:
        raise SpecError("Sections must be exactly '## ORCHESTRATOR', '## AGENTS', "
                        "'## WORKFLOW', '## AUXILIARY RULES' in that order; "
                        f"found: {titles}.")
    body = dict(sections)

    # orchestrator
    ofields = _parse_fields(
        body["ORCHESTRATOR"],
        [("tools", "inline"), ("max_steps", "inline"), ("instructions", "block")],
        "ORCHESTRATOR section")
    orchestrator = AgentDef(
        name="orchestrator",
        tools=_parse_tools(ofields["tools"], "ORCHESTRATOR section"),
        max_steps=_parse_int(ofields["max_steps"], "max_steps", "ORCHESTRATOR section"),
        role="",
        instructions=ofields["instructions"],
        contract="")

    # agents
    agents: dict[str, AgentDef] = {}
    agent_lines = body["AGENTS"]
    chunks: list[tuple[str, list[str]]] = []
    cur_name: str | None = None
    cur_lines: list[str] = []
    for ln in agent_lines:
        if ln.startswith("### "):
            if cur_name is not None:
                chunks.append((cur_name, cur_lines))
            cur_name = ln[4:].strip()
            cur_lines = []
        elif cur_name is None:
            if ln.strip():
                raise SpecError("AGENTS section must contain only "
                                "'### <AgentName>' blocks; unexpected content "
                                f"before the first agent: {ln.strip()!r}.")
        else:
            cur_lines.append(ln)
    if cur_name is not None:
        chunks.append((cur_name, cur_lines))
    if not chunks:
        raise SpecError("AGENTS section defines no agents; add at least one "
                        "'### <AgentName>' block.")
    for name, chunk in chunks:
        if not _AGENT_NAME_RE.match(name):
            raise SpecError(f"Invalid agent name {name!r}: must match "
                            "[A-Za-z][A-Za-z0-9_]{1,30} (letters, digits, "
                            "underscore; 2-31 chars; starts with a letter).")
        if name in agents:
            raise SpecError(f"Duplicate agent name {name!r}.")
        afields = _parse_fields(
            chunk,
            [("tools", "inline"), ("max_steps", "inline"), ("role", "inline"),
             ("instructions", "block"), ("contract", "block")],
            f"Agent '{name}'")
        agents[name] = AgentDef(
            name=name,
            tools=_parse_tools(afields["tools"], f"Agent '{name}'"),
            max_steps=_parse_int(afields["max_steps"], "max_steps", f"Agent '{name}'"),
            role=afields["role"],
            instructions=afields["instructions"],
            contract=afields["contract"])

    workflow = _section_text(body["WORKFLOW"])
    if not workflow.strip():
        raise SpecError("WORKFLOW section is empty; describe the hops the "
                        "orchestrator must execute.")
    aux = _section_text(body["AUXILIARY RULES"])
    if not aux.strip():
        raise SpecError("AUXILIARY RULES section is empty; write rules or the "
                        "literal word 'none'.")

    return HarnessSpec(version_label=version_label, orchestrator=orchestrator,
                       agents=agents, workflow=workflow, aux=aux,
                       source_text=text)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _check_tools(agent: AgentDef, is_orchestrator: bool) -> None:
    where = "ORCHESTRATOR" if is_orchestrator else f"agent '{agent.name}'"
    for t in agent.tools:
        if t in TOOL_ROSTER:
            continue
        if t in IMPLICIT_ORCHESTRATOR_TOOLS and is_orchestrator:
            raise SpecError(f"{where}: '{t}' is an implicit orchestrator tool — "
                            "do not list it under 'tools:'.")
        raise SpecError(f"{where}: unknown tool {t!r}; the only tools are: "
                        f"{', '.join(TOOL_ROSTER)} (or 'none').")


def validate_harness(spec: HarnessSpec) -> None:
    if not 1 <= len(spec.agents) <= 8:
        raise SpecError(f"A harness needs 1-8 agents; this one defines "
                        f"{len(spec.agents)}.")
    for agent, is_orch in [(spec.orchestrator, True)] + [
            (a, False) for a in spec.agents.values()]:
        where = "ORCHESTRATOR" if is_orch else f"agent '{agent.name}'"
        _check_tools(agent, is_orch)
        if not 1 <= agent.max_steps <= 60:
            raise SpecError(f"{where}: max_steps must be between 1 and 60, "
                            f"got {agent.max_steps}.")
        if not agent.instructions.strip():
            raise SpecError(f"{where}: instructions must be non-empty.")
        if not is_orch and not agent.contract.strip():
            raise SpecError(f"{where}: contract must be non-empty.")
    for key, agent in spec.agents.items():
        if key != agent.name:
            raise SpecError(f"Agent dict key {key!r} does not match agent name "
                            f"{agent.name!r}.")


if __name__ == "__main__":
    import sys as _sys

    _path = (_sys.argv[1] if len(_sys.argv) > 1
             else str(Path(__file__).resolve().parent.parent / "harness" / "v0.md"))
    _spec = parse_harness(Path(_path).read_text(encoding="utf-8"))
    validate_harness(_spec)
    print(f"OK: parsed harness {_spec.version_label!r} — orchestrator tools="
          f"{_spec.orchestrator.tools or 'none'}, agents: "
          f"{', '.join(_spec.agents)}")
