"""LLM provider clients + cost metering (DESIGN sections 3, 13, 14).

Primary path: any OpenAI-compatible server via the `openai` package
(chat.completions with `tools` function-calling). Secondary: the `anthropic`
SDK (adaptive-thinking-era surface: no temperature/top_p, no thinking param,
plain max_tokens). Messages and tools cross the interface in OpenAI wire
format; the anthropic adapter converts internally.

No env reads at import time; SDKs are imported lazily inside methods.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass

ROLE_ENV_PREFIX = {
    "actor": "ACTOR",
    "optimizer": "OPTIMIZER",
    "training_judge": "JUDGE",
    "pairwise_judge": "JUDGE",
}

_DEFAULT_PRICING = [1.0, 3.0]  # USD per MTok [input, output]


class LLMConfigError(RuntimeError):
    """Raised when a role's model/key cannot be resolved (names the env var)."""


@dataclass
class LLMResponse:
    text: str                # concatenated assistant text ("" if none)
    tool_calls: list[dict]   # [{"id": str, "name": str, "arguments": dict}]
    usage: dict              # {"input": int, "output": int} (+ cache_r/cache_w)
    raw: object


# ---------------------------------------------------------------------------
# usage normalization + cost metering
# ---------------------------------------------------------------------------

def _norm_usage(usage: dict | None) -> dict:
    """Normalize any of the usage-key dialects to {input, output, cache_r, cache_w}."""
    u = usage or {}

    def first(*keys: str) -> int:
        for k in keys:
            v = u.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    return {
        "input": first("input", "in", "input_tokens", "prompt_tokens"),
        "output": first("output", "out", "output_tokens", "completion_tokens"),
        "cache_r": first("cache_r", "cache_read", "cache_read_input_tokens", "cached_tokens"),
        "cache_w": first("cache_w", "cache_write", "cache_creation_input_tokens"),
    }


class CostMeter:
    """Global token + USD accounting by role. Pricing: substring match against
    the model name in `pricing_per_mtok` (longest substring wins), `_default`
    fallback [1.0, 3.0] with a one-time warning per model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roles: dict[str, dict] = {}
        self._pricing: dict[str, list[float]] = {"_default": list(_DEFAULT_PRICING)}
        self._warned_models: set[str] = set()

    # -- pricing ------------------------------------------------------------
    def set_pricing(self, pricing: dict | None) -> None:
        with self._lock:
            for key, val in (pricing or {}).items():
                try:
                    self._pricing[key] = [float(val[0]), float(val[1])]
                except (TypeError, ValueError, IndexError):
                    continue

    def _price_for(self, model: str) -> tuple[float, float, bool]:
        model = model or ""
        best_key = None
        for key in self._pricing:
            if key == "_default" or key not in model:
                continue
            if best_key is None or len(key) > len(best_key):
                best_key = key
        if best_key is not None:
            p = self._pricing[best_key]
            return p[0], p[1], True
        d = self._pricing.get("_default", _DEFAULT_PRICING)
        return d[0], d[1], False

    def estimate_usd(self, model: str, usage: dict) -> float:
        """USD estimate for one call's usage. cache_r billed at 0.1x input
        price, cache_w at 1.25x input price (Anthropic convention)."""
        u = _norm_usage(usage)
        pin, pout, matched = self._price_for(model)
        if not matched and model not in self._warned_models:
            self._warned_models.add(model)
            print(
                f"[finforge.llm] warning: no pricing_per_mtok entry matches model "
                f"{model!r}; using _default {self._pricing.get('_default')} USD/MTok",
                file=sys.stderr,
            )
        return (
            u["input"] * pin / 1e6
            + u["output"] * pout / 1e6
            + u["cache_r"] * pin * 0.1 / 1e6
            + u["cache_w"] * pin * 1.25 / 1e6
        )

    # -- accumulation -------------------------------------------------------
    def add(self, role: str, model: str, usage: dict) -> None:
        u = _norm_usage(usage)
        usd = self.estimate_usd(model, u)
        with self._lock:
            r = self._roles.setdefault(
                role,
                {"usd": 0.0, "calls": 0,
                 "tokens": {"in": 0, "out": 0, "cache_r": 0, "cache_w": 0}},
            )
            r["usd"] += usd
            r["calls"] += 1
            t = r["tokens"]
            t["in"] += u["input"]
            t["out"] += u["output"]
            t["cache_r"] += u["cache_r"]
            t["cache_w"] += u["cache_w"]

    @property
    def usd_total(self) -> float:
        with self._lock:
            return sum(r["usd"] for r in self._roles.values())

    def by_role(self) -> dict:
        with self._lock:
            return {
                role: {"usd": r["usd"], "calls": r["calls"], "tokens": dict(r["tokens"])}
                for role, r in self._roles.items()
            }

    def snapshot(self) -> dict:
        by_role = self.by_role()
        return {
            "usd_total": round(sum(r["usd"] for r in by_role.values()), 6),
            "by_role": {
                role: {"usd": round(r["usd"], 6), "calls": r["calls"],
                       "tokens": r["tokens"]}
                for role, r in by_role.items()
            },
        }

    def over_cap(self, cap_usd: float) -> bool:
        return self.usd_total >= float(cap_usd)


METER = CostMeter()  # module-level singleton


# ---------------------------------------------------------------------------
# transient-error detection + one outer retry (on top of SDK default retries)
# ---------------------------------------------------------------------------

_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "OverloadedError", "ServiceUnavailableError",
    "ConnectionError", "TimeoutError",
}


def _is_transient(exc: Exception) -> bool:
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status >= 500 or status == 429)


_OUTER_RETRIES = 4          # attempts = 1 + _OUTER_RETRIES (on top of SDK retries)
_OUTER_BACKOFF_BASE_S = 5.0  # 5, 10, 20, 40s (rate limits need real backoff)


def _with_outer_retry(fn):
    for attempt in range(_OUTER_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - inspected, re-raised if not transient
            if not _is_transient(exc) or attempt == _OUTER_RETRIES:
                raise
            delay = _OUTER_BACKOFF_BASE_S * (2 ** attempt)
            print(f"[finforge.llm] transient {type(exc).__name__}; "
                  f"retry {attempt + 1}/{_OUTER_RETRIES} in {delay:.0f}s",
                  file=sys.stderr)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class LLMClient:
    """One client per role. Construct via `LLMClient.for_role(role, cfg)`."""

    def __init__(self, role: str, kind: str, model: str, api_key: str,
                 base_url: str | None, temperature: float | None,
                 default_max_tokens: int) -> None:
        self.role = role
        self.kind = kind                # "openai" | "anthropic"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature  # only used on the OpenAI-compatible path
        self.default_max_tokens = default_max_tokens
        self._client = None

    # -- resolution (DESIGN sections 3 + 13) --------------------------------
    @classmethod
    def for_role(cls, role: str, cfg: dict) -> "LLMClient":
        if role not in ROLE_ENV_PREFIX:
            raise LLMConfigError(
                f"Unknown role {role!r}; expected one of {sorted(ROLE_ENV_PREFIX)}")
        prefix = ROLE_ENV_PREFIX[role]
        rcfg = dict(cfg.get(role) or {})

        model = os.environ.get(f"{prefix}_MODEL") or rcfg.get("model")
        if isinstance(model, str) and model.startswith("SET_VIA_"):
            model = None  # config placeholder, not a real model id

        key_envs = [f"{prefix}_API_KEY"]
        if prefix == "JUDGE":
            key_envs.append("JUDGE_LLM_API_KEY")
        api_key = next((os.environ[e] for e in key_envs if os.environ.get(e)), None)

        base_url = os.environ.get(f"{prefix}_BASE_URL") or rcfg.get("base_url")
        provider = (rcfg.get("provider") or "").lower()

        if base_url or provider == "openai_compat":
            kind = "openai"
        elif (api_key or "").startswith("sk-ant-") or provider == "anthropic":
            kind = "anthropic"
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                key_envs.append("ANTHROPIC_API_KEY")
        else:
            kind = "openai"

        if not model:
            raise LLMConfigError(
                f"No model configured for role {role!r}: set env {prefix}_MODEL "
                f"(config['{role}']['model'] is unset or a SET_VIA_ placeholder).")
        if not api_key:
            raise LLMConfigError(
                f"No API key for role {role!r}: set env {' or '.join(key_envs)}.")

        METER.set_pricing(cfg.get("pricing_per_mtok"))

        temp = rcfg.get("temperature")
        return cls(role=role, kind=kind, model=model, api_key=api_key,
                   base_url=base_url,
                   temperature=float(temp) if temp is not None else None,
                   default_max_tokens=int(rcfg.get("max_tokens") or 4096))

    # -- public chat ---------------------------------------------------------
    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None, json_mode: bool = False) -> LLMResponse:
        mt = int(max_tokens or self.default_max_tokens)
        if self.kind == "anthropic":
            resp = self._chat_anthropic(system, messages, tools, mt)
        else:
            resp = self._chat_openai(system, messages, tools, mt, json_mode)
        METER.add(self.role, self.model, resp.usage)
        return resp

    # -- OpenAI-compatible path ----------------------------------------------
    def _openai_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            # Hard per-request timeout: a hung socket otherwise blocks a
            # worker thread forever (observed: 2 stuck tasks wedged a whole
            # run). Optimizer emits 16k-token specs on a slow host — give it
            # longer; everything else fails fast into the retry path.
            kwargs["timeout"] = 600.0 if self.role == "optimizer" else 240.0
            self._client = OpenAI(**kwargs)
        return self._client

    def _chat_openai(self, system: str, messages: list[dict],
                     tools: list[dict] | None, max_tokens: int,
                     json_mode: bool) -> LLMResponse:
        client = self._openai_client()
        msgs = [{"role": "system", "content": system}] + list(messages)
        kwargs: dict = {"model": self.model, "messages": msgs, "max_tokens": max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            # Some OpenAI-compatible hosts reject response_format; try it,
            # fall back to a plain call on any failure. If the HTTP call
            # succeeded but parsing failed, the call was still billed — meter
            # it here before falling back (chat() only meters the returned
            # response).
            raw = None
            try:
                raw = _with_outer_retry(lambda: client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs))
            except Exception:  # noqa: BLE001 - fall back to plain call
                raw = None
            if raw is not None:
                try:
                    return self._parse_openai(raw)
                except Exception:  # noqa: BLE001 - meter the billed call, fall back
                    try:
                        u = getattr(raw, "usage", None)
                        METER.add(self.role, self.model, {
                            "input": int(getattr(u, "prompt_tokens", 0) or 0),
                            "output": int(getattr(u, "completion_tokens", 0) or 0)})
                    except Exception:  # noqa: BLE001
                        pass
        raw = _with_outer_retry(lambda: client.chat.completions.create(**kwargs))
        return self._parse_openai(raw)

    @staticmethod
    def _parse_openai(raw) -> LLMResponse:
        msg = raw.choices[0].message
        text = msg.content or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        if not text and not (msg.tool_calls or []):
            # Reasoning-model hosts (e.g. gpt-oss on Baseten) may emit only
            # reasoning_content and stop with empty content. Fall back so the
            # caller gets the model's working text instead of nothing.
            for attr in ("reasoning_content", "reasoning"):
                rc = getattr(msg, attr, None)
                if isinstance(rc, str) and rc.strip():
                    text = rc.strip()
                    break
        tool_calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            args_raw = tc.function.arguments or "{}"
            try:
                args = json.loads(args_raw)
                if not isinstance(args, dict):
                    args = {"value": args}
            except (json.JSONDecodeError, TypeError) as exc:
                args = {"_malformed_arguments": str(args_raw), "_error": str(exc)}
            tool_calls.append(  # hosts may omit ids; synthesize a stable one
                {"id": tc.id or f"call_{i}", "name": tc.function.name,
                 "arguments": args})
        u = getattr(raw, "usage", None)
        usage = {"input": int(getattr(u, "prompt_tokens", 0) or 0),
                 "output": int(getattr(u, "completion_tokens", 0) or 0)}
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached:
            usage["cache_r"] = int(cached)
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage, raw=raw)

    # -- anthropic path --------------------------------------------------------
    def _anthropic_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def _chat_anthropic(self, system: str, messages: list[dict],
                        tools: list[dict] | None, max_tokens: int) -> LLMResponse:
        # Adaptive-thinking-era surface: no temperature/top_p, no thinking param.
        client = self._anthropic_client()
        kwargs: dict = {
            "model": self.model,
            "system": system,
            "messages": _openai_msgs_to_anthropic(messages),
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [_openai_tool_to_anthropic(t) for t in tools]
        raw = _with_outer_retry(lambda: client.messages.create(**kwargs))
        return _parse_anthropic(raw)


# ---------------------------------------------------------------------------
# OpenAI <-> anthropic wire-format conversion
# ---------------------------------------------------------------------------

def _openai_tool_to_anthropic(tool: dict) -> dict:
    fn = tool.get("function") or {}
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _openai_msgs_to_anthropic(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            block = {"type": "tool_result",
                     "tool_use_id": m.get("tool_call_id") or "",
                     "content": str(m.get("content") or "")}
            # All tool results for one assistant turn must sit in ONE user message.
            if (out and out[-1]["role"] == "user"
                    and isinstance(out[-1]["content"], list)
                    and all(b.get("type") == "tool_result" for b in out[-1]["content"])):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            blocks: list[dict] = []
            content = m.get("content")
            if content:
                blocks.append({"type": "text",
                               "text": content if isinstance(content, str) else str(content)})
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {"_malformed_arguments": args}
                if not isinstance(args, dict):
                    args = {"value": args}
                blocks.append({"type": "tool_use", "id": tc.get("id") or "",
                               "name": fn.get("name") or "", "input": args})
            if not blocks:
                blocks = [{"type": "text", "text": "(no content)"}]
            out.append({"role": "assistant", "content": blocks})
        else:  # "user" (and anything unrecognized becomes a user turn)
            out.append({"role": "user", "content": str(m.get("content") or "")})
    return out


def _parse_anthropic(raw) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in getattr(raw, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            args = block.input if isinstance(block.input, dict) else {"value": block.input}
            tool_calls.append({"id": block.id, "name": block.name, "arguments": args})
    u = getattr(raw, "usage", None)
    usage = {"input": int(getattr(u, "input_tokens", 0) or 0),
             "output": int(getattr(u, "output_tokens", 0) or 0)}
    cr = int(getattr(u, "cache_read_input_tokens", 0) or 0)
    cw = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    if cr:
        usage["cache_r"] = cr
    if cw:
        usage["cache_w"] = cw
    return LLMResponse(text="".join(text_parts), tool_calls=tool_calls,
                       usage=usage, raw=raw)
