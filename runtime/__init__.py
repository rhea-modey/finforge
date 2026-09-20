"""FinForge runtime package.

Modules:
- llm: provider clients (OpenAI-compatible + anthropic adapter) + CostMeter.
- tools: file tools + run_python sandbox, jailed to a world's files/ dir.
- spec: harness spec parser/validator (DESIGN section 6).
- orchestrator: spec interpreter — orchestrator + subagent loops (section 7.2).
- trace: JSONL trace recording + deterministic summarization (section 7.3).
- fake: offline FakeLLM + monkeypatch helper for zero-network dry runs.

No module reads env at import time; provider SDKs are imported lazily.
"""
