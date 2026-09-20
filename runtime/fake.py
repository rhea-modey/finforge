"""Offline FakeLLM + monkeypatch helper for zero-network dry runs.

Scripted behavior: the fake orchestrator calls each rostered agent once (in
roster order), then submits a syntactically valid — but deliberately wrong —
answer for the task category, inferred from the task_id suffix embedded in
the orchestrator system prompt ("Task ID: ..."). Fake subagents make one
list_files call when granted it (exercising the tool path), then reply with
final text. Lets integration exercise run_task end-to-end with zero network.

Usage:
    from runtime.fake import install_fake_llm
    uninstall = install_fake_llm()
    ... run_task(...) ...
    uninstall()
"""
from __future__ import annotations

import json
import re

from runtime.llm import METER, LLMClient, LLMResponse

CATEGORIES = ("bank_rec", "journal_entries", "variance_analysis", "accrual_schedule")

_TASK_ID_RE = re.compile(r"^Task ID:\s*(\S+)", re.MULTILINE)
_ROSTER_LINE_RE = re.compile(r"^- ([A-Za-z][A-Za-z0-9_]{1,30}) \|", re.MULTILINE)
_AGENT_NAME_RE = re.compile(r"You are agent ([A-Za-z][A-Za-z0-9_]{1,30})")

_FAKE_JE = {
    "memo": "Fake accrual entry (offline dry run)",
    "date": "2026-06-30",
    "lines": [
        {"account": "6000 Salaries Expense", "debit": 100.00, "credit": 0},
        {"account": "2100 Accrued Liabilities", "debit": 0, "credit": 100.00},
    ],
}

CANNED_ANSWERS: dict[str, dict] = {
    "bank_rec": {
        "bank_statement_ending_balance": 1000.00,
        "gl_cash_ending_balance": 900.00,
        "adjusted_balance": 950.00,
        "reconciling_items": [
            {"kind": "outstanding_check", "amount": 50.00, "date": "2026-06-28",
             "ref": None, "description": "Fake outstanding check (offline dry run)",
             "side": "bank"},
        ],
        "proposed_journal_entries": [_FAKE_JE],
    },
    "journal_entries": {"entries": [_FAKE_JE]},
    "variance_analysis": {
        "variances": [
            {"account": "6200 Advertising", "actual": 100.00, "budget": 50.00,
             "variance": 50.00, "direction": "unfavorable",
             "drivers": [{"description": "Fake driver (offline dry run)",
                          "amount": 50.00, "evidence": []}]},
        ],
        "summary": "Fake variance summary (offline dry run).",
    },
    "accrual_schedule": {
        "schedule": [
            {"item": "Fake insurance policy", "opening_balance": 1200.00,
             "additions": 0.00, "amortization": 100.00,
             "closing_balance": 1100.00},
        ],
        "journal_entry": {
            "memo": "Fake amortization entry (offline dry run)",
            "date": "2026-06-30",
            "lines": [
                {"account": "6400 Insurance Expense", "debit": 100.00, "credit": 0},
                {"account": "1400 Prepaid Insurance", "debit": 0, "credit": 100.00},
            ],
        },
    },
}


def category_from_task_id(task_id: str) -> str:
    for cat in CATEGORIES:
        if str(task_id).endswith(cat):
            return cat
    return "journal_entries"


class FakeLLM:
    """Drop-in for LLMClient with fully scripted, deterministic behavior."""

    def __init__(self, role: str, cfg: dict | None = None) -> None:
        self.role = role
        self.cfg = cfg or {}
        self.model = "fake-model"
        self._n = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"fake-call-{self._n}"

    @staticmethod
    def _usage() -> dict:
        return {"input": 100, "output": 20}

    def chat(self, system: str, messages: list[dict],
             tools: list[dict] | None = None, max_tokens: int | None = None,
             json_mode: bool = False) -> LLMResponse:
        system = system or ""
        if system.startswith("You are the ORCHESTRATOR"):
            return self._orchestrator_turn(system, messages)
        if "You are agent " in system:
            return self._subagent_turn(system, messages, tools)
        # judge/optimizer/etc. callers: return an empty-but-valid JSON object
        return LLMResponse(text="{}", tool_calls=[], usage=self._usage(), raw=None)

    # -- orchestrator: call each agent once, then submit ---------------------
    def _orchestrator_turn(self, system: str, messages: list[dict]) -> LLMResponse:
        agents = _ROSTER_LINE_RE.findall(system)
        already_called = sum(
            1 for m in messages if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if (tc.get("function") or {}).get("name") == "call_agent")
        last_user = next((m for m in reversed(messages)
                          if m.get("role") == "user"), {})
        forced = "Step limit reached" in str(last_user.get("content", ""))

        if already_called < len(agents) and not forced:
            name = agents[already_called]
            tc = {"id": self._next_id(), "name": "call_agent",
                  "arguments": {"name": name,
                                "message": ("Offline dry run: please complete "
                                            "your part of the task per your "
                                            "contract and reply with your "
                                            "final text.")}}
            return LLMResponse(text="", tool_calls=[tc],
                               usage=self._usage(), raw=None)

        m = _TASK_ID_RE.search(system)
        cat = category_from_task_id(m.group(1) if m else "")
        tc = {"id": self._next_id(), "name": "submit_answer",
              "arguments": {"answer_json": json.dumps(CANNED_ANSWERS[cat]),
                            "notes": f"FAKE offline answer for category {cat}."}}
        return LLMResponse(text="", tool_calls=[tc], usage=self._usage(), raw=None)

    # -- subagent: one list_files call if granted, then final text -----------
    def _subagent_turn(self, system: str, messages: list[dict],
                       tools: list[dict] | None) -> LLMResponse:
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        tool_names = [(t.get("function") or {}).get("name")
                      for t in (tools or []) if isinstance(t, dict)]
        if not has_tool_result and "list_files" in tool_names:
            tc = {"id": self._next_id(), "name": "list_files", "arguments": {}}
            return LLMResponse(text="", tool_calls=[tc],
                               usage=self._usage(), raw=None)
        nm = _AGENT_NAME_RE.search(system)
        name = nm.group(1) if nm else "agent"
        return LLMResponse(
            text=(f"FAKE {name} report: work complete per contract; no "
                  f"findings (offline dry run)."),
            tool_calls=[], usage=self._usage(), raw=None)


def install_fake_llm():
    """Patch LLMClient.for_role to return FakeLLM instances (zero network,
    zero cost). Returns an uninstall() callable that restores the original."""
    original = LLMClient.__dict__["for_role"]

    def fake_for_role(cls, role: str, cfg: dict) -> FakeLLM:  # noqa: ARG001
        return FakeLLM(role, cfg)

    LLMClient.for_role = classmethod(fake_for_role)
    METER.set_pricing({"fake-model": [0.0, 0.0]})

    def uninstall() -> None:
        LLMClient.for_role = original

    return uninstall
