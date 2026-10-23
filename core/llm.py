"""Single entry point for every LLM call in DataTalk.AI.

All agents call :meth:`LLMClient.structured` with a *purpose* label, a system
prompt, a user payload and a Pydantic response model. This module

* replays a previously validated reply for the same prompt from the decision
  memo (``core.llm_memo``) so identical questions produce identical plans, SQL
  and narratives — a replay makes no provider request,
* talks to Groq's OpenAI-compatible chat-completions API (``groq`` SDK),
* asks the provider for JSON (``json_object`` or best-effort ``json_schema``),
* validates the reply with Pydantic and retries a bounded number of times on
  transient provider errors or malformed / schema-invalid output,
* records one :class:`LLMCallRecord` per HTTP request (purpose, model, duration,
  outcome, token usage) into the active :class:`CallTrace`.

Prompt text and model output are **never** logged.

Retry bound: one logical call issues at most ``LLM_MAX_ATTEMPTS`` requests.
The SDK's own retry loop is disabled (``max_retries=0``) so every request is
counted here.
"""

from __future__ import annotations

import contextvars
import json
import logging
import random
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from core.llm_memo import LLMMemo
from core.settings import LLM_PRICING_PER_1M, Settings

log = logging.getLogger("datatalk.llm")

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when no API key is configured or the provider rejects the request."""


class LLMResponseError(RuntimeError):
    """Raised when every attempt returned malformed or schema-invalid output."""


# ═══════════════════════════════════════════════════════════════════════════════
# Call tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LLMCallRecord:
    """One HTTP request to the provider. Contains no prompt or response content."""

    call_id: str            # groups the attempts that belong to one logical call
    purpose: str            # planner | sql_generation | sql_repair | insight_narrative | synthesis | response_validation
    model: str
    attempt: int
    duration_ms: float
    outcome: str            # success | invalid_output | rate_limited | transient_error | error
    tokens_in: int = 0
    tokens_out: int = 0
    error_type: str = ""


@dataclass
class CallTrace:
    """Thread-safe collector for the LLM requests made while answering one question."""

    records: list[LLMCallRecord] = field(default_factory=list)
    replays: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, record: LLMCallRecord) -> None:
        with self._lock:
            self.records.append(record)

    def add_replay(self) -> None:
        with self._lock:
            self.replays += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
            replays = self.replays
        logical = {r.call_id: r.purpose for r in records}
        tokens_in = sum(r.tokens_in for r in records)
        tokens_out = sum(r.tokens_out for r in records)
        cost = 0.0
        for r in records:
            price = LLM_PRICING_PER_1M.get(r.model)
            if price:
                cost += r.tokens_in / 1e6 * price["input"] + r.tokens_out / 1e6 * price["output"]
        return {
            "llm_calls": len(logical),
            "llm_requests": len(records),
            "llm_replays": replays,
            "calls_by_purpose": dict(Counter(logical.values())),
            "failed_requests": sum(r.outcome != "success" for r in records),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "llm_ms": round(sum(r.duration_ms for r in records), 1),
            "est_cost_usd": round(cost, 6),
        }


_CURRENT_TRACE: contextvars.ContextVar[CallTrace | None] = contextvars.ContextVar(
    "datatalk_call_trace", default=None
)


@contextmanager
def trace_calls() -> Iterator[CallTrace]:
    """Collect every LLM request issued inside the ``with`` block.

    Worker threads must be started with ``contextvars.copy_context().run`` to share
    the trace (the analytics engine does this).
    """
    trace = CallTrace()
    token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    finally:
        _CURRENT_TRACE.reset(token)


def _record(rec: LLMCallRecord) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.add(rec)
    log.info(
        "llm_call purpose=%s model=%s attempt=%d outcome=%s duration_ms=%.0f tokens_in=%d tokens_out=%d%s",
        rec.purpose, rec.model, rec.attempt, rec.outcome, rec.duration_ms, rec.tokens_in,
        rec.tokens_out, f" error={rec.error_type}" if rec.error_type else "",
    )


def _record_replay(purpose: str, model: str) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.add_replay()
    log.info("llm_replay purpose=%s model=%s (validated decision reused, no provider request)", purpose, model)


# ═══════════════════════════════════════════════════════════════════════════════
# JSON extraction
# ═══════════════════════════════════════════════════════════════════════════════

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.S)


def extract_json_object(text: str) -> dict:
    """Return the first JSON object in ``text``; raise ``ValueError`` when none parses."""
    cleaned = _FENCE_RE.sub("", (text or "").strip()).strip()
    if not cleaned:
        raise ValueError("empty response")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in response") from None
        obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response JSON is not an object")
    return obj


def _validation_summary(exc: ValidationError) -> str:
    """Compact error list for the correction message (field paths and messages only)."""
    parts = [f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()[:6]]
    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredLLM(Protocol):
    """What agents depend on. Tests pass a fake implementing this method."""

    def structured(self, purpose: str, system: str, user: str, schema: type[T],
                   *, temperature: float | None = None) -> T: ...


_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMClient:
    """Groq chat-completions client with Pydantic-validated JSON responses."""

    def __init__(self, settings: Settings, sdk_client: Any | None = None,
                 memo: LLMMemo | None = None) -> None:
        self.settings = settings
        self.model = settings.groq_model
        self._client = sdk_client
        if memo is None and getattr(settings, "llm_memo_enabled", False):
            memo = LLMMemo.from_settings(settings)
        self.memo = memo

    def _sdk(self) -> Any:
        if self._client is None:
            if not self.settings.llm_configured:
                raise LLMUnavailableError("GROQ_API_KEY is not configured.")
            from groq import Groq  # imported lazily so the UI can start without the key
            self._client = Groq(
                api_key=self.settings.groq_api_key.get_secret_value(),
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,  # retries are handled (and counted) below
            )
        return self._client

    def _response_format(self, schema: type[BaseModel]) -> dict:
        if self.settings.llm_response_format == "json_schema":
            return {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": False, "schema": schema.model_json_schema()}}
        return {"type": "json_object"}

    def structured(self, purpose: str, system: str, user: str, schema: type[T],
                   *, temperature: float | None = None) -> T:
        """Run one logical LLM call and return a validated ``schema`` instance."""
        temp = self.settings.llm_temperature if temperature is None else temperature

        # Deterministic replay: the same prompt returns the same validated decision.
        memo_key = None
        if self.memo is not None:
            memo_key = self.memo.key(purpose, self.model, schema.__name__, system, user, temp)
            stored = self.memo.get(memo_key)
            if stored is not None:
                try:
                    parsed = schema.model_validate(stored)
                    _record_replay(purpose, self.model)
                    return parsed
                except ValidationError:
                    log.info("llm memo entry no longer matches %s; calling the model", schema.__name__)

        sdk = self._sdk()
        call_id = uuid.uuid4().hex[:12]
        messages: list[dict[str, str]] = [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]
        last_error: Exception | None = None

        for attempt in range(1, self.settings.llm_max_attempts + 1):
            t0 = time.perf_counter()
            tokens_in = tokens_out = 0
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model, "messages": messages, "temperature": temp,
                    "response_format": self._response_format(schema),
                }
                if self.settings.llm_seed is not None:
                    kwargs["seed"] = self.settings.llm_seed
                completion = sdk.chat.completions.create(**kwargs)
                usage = getattr(completion, "usage", None)
                tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
                text = completion.choices[0].message.content or ""
                obj = extract_json_object(text)
                parsed = schema.model_validate(obj)
            except (ValueError, ValidationError) as exc:  # malformed or schema-invalid output
                last_error = exc
                _record(LLMCallRecord(call_id, purpose, self.model, attempt,
                                      (time.perf_counter() - t0) * 1000, "invalid_output",
                                      tokens_in, tokens_out, type(exc).__name__))
                detail = _validation_summary(exc) if isinstance(exc, ValidationError) else str(exc)
                messages = messages[:2] + [{"role": "user", "content": (
                    "Your previous reply could not be used: " + detail[:600]
                    + ". Reply again with exactly one JSON object that follows the response schema.")}]
                continue
            except Exception as exc:  # provider / network errors
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status == 400 and "json_validate_failed" in str(exc):
                    # Groq rejected its own output as invalid JSON: treat like malformed output.
                    _record(LLMCallRecord(call_id, purpose, self.model, attempt,
                                          (time.perf_counter() - t0) * 1000, "invalid_output", 0, 0,
                                          "json_validate_failed"))
                    last_error = ValueError("provider reported invalid JSON output")
                    continue
                transient = status in _TRANSIENT_STATUS or type(exc).__name__ in {
                    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
                outcome = "rate_limited" if status == 429 else ("transient_error" if transient else "error")
                _record(LLMCallRecord(call_id, purpose, self.model, attempt,
                                      (time.perf_counter() - t0) * 1000, outcome, 0, 0, type(exc).__name__))
                if not transient:
                    raise LLMUnavailableError(f"{type(exc).__name__}: provider rejected the request") from exc
                if attempt < self.settings.llm_max_attempts:
                    hint = re.search(r"try again in\s+([0-9.]+)s", str(exc), re.I)
                    delay = float(hint.group(1)) if hint else self.settings.llm_retry_base_seconds * 2 ** (attempt - 1)
                    time.sleep(min(delay + random.uniform(0.1, 0.35), 15.0))
                continue

            _record(LLMCallRecord(call_id, purpose, self.model, attempt,
                                  (time.perf_counter() - t0) * 1000, "success", tokens_in, tokens_out))
            if memo_key is not None:
                self.memo.put(memo_key, parsed.model_dump(mode="json"), purpose=purpose)
            return parsed

        if isinstance(last_error, (ValueError, ValidationError)):
            raise LLMResponseError(f"{purpose}: no valid response after "
                                   f"{self.settings.llm_max_attempts} attempt(s)") from last_error
        raise LLMUnavailableError(f"{purpose}: provider unavailable after "
                                  f"{self.settings.llm_max_attempts} attempt(s)") from last_error
