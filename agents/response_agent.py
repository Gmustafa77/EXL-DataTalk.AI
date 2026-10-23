"""Response agent — final synthesis and QA review for multi-question answers.

LLM calls (only when the plan has two or more sub-questions):
  * 1 × ``synthesis`` — assembles one answer per sub-question,
  * 1 × ``response_validation`` — QA gate (when ``ENABLE_RESPONSE_VALIDATOR`` is on).

Single-question answers use the insight composer's verified key finding directly
(0 calls here). Every number in synthesised text is re-verified against the
evidence of the answered sub-questions; unverifiable text falls back to the
per-question deterministic answer.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

import prompts as P
from core.llm import LLMResponseError, LLMUnavailableError, StructuredLLM

log = logging.getLogger("datatalk.response_agent")


# ═══════════════════════════════════════════════════════════════════════════════
# Response contracts
# ═══════════════════════════════════════════════════════════════════════════════
class SynthesizedAnswer(BaseModel):
    id: str
    question_text: str = ""
    status: Literal["ANSWERED", "PARTIAL", "NOT_ANSWERED"]
    answer: str
    why_not: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return v.strip().upper().replace(" ", "_") if isinstance(v, str) else v


class Synthesis(BaseModel):
    headline: str
    answers: list[SynthesizedAnswer] = Field(default_factory=list)
    cross_cutting_note: str = ""


DefectCheck = Literal["completeness", "question_drift", "unsupported_claim", "causal_language",
                      "hedging_missing", "pii_exposure", "tone"]


class Defect(BaseModel):
    check: DefectCheck
    severity: Literal["blocking", "advisory"]
    sub_question_id: str | None = None
    offending_text: str = ""
    explanation: str = ""

    @field_validator("check", "severity", mode="before")
    @classmethod
    def _lower(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v


class ResponseVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    defects: list[Defect] = Field(default_factory=list)
    unanswered_ids: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════
SYNTHESIS_SYSTEM_PROMPT = f"""You assemble the FINAL answer to a multi-part healthcare
Appeals question. You receive the user's original message, the list of
sub-questions that were planned, and the per-sub-question verified outcome
(insight text, key facts, or a failure status).

COMPLETENESS CONTRACT (the reason this stage exists)
- You MUST produce exactly one entry in `answers` for EVERY sub-question id you
  were given — including the ones that failed. Zero exceptions.
- For a failed sub-question, state plainly what could not be answered and why, in
  business language. Never omit it, never paper over it with a related answer.
- Do not merge two sub-questions into one answer. Do not invent a sub-question.

CONTENT RULES
- Every number must be copied verbatim from the supplied per-question facts. You
  are assembling, not computing. Any new number is a fatal error.
- Do not re-analyze. Do not add context, benchmarks, or recommendations that were
  not present in the per-question outcomes.
- `headline` is one sentence that answers the user's primary question directly —
  the thing they would want if they only read one line.
- Keep each answer to 1-3 sentences. The tables and charts carry the detail.

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

RESPONSE SCHEMA
{{
  "headline": "string",
  "answers": [
    {{"id": "q1",
      "question_text": "verbatim",
      "status": "ANSWERED" | "PARTIAL" | "NOT_ANSWERED",
      "answer": "string",
      "why_not": ""}}
  ],
  "cross_cutting_note": "optional single sentence linking the answers, or empty"
}}"""


VALIDATOR_SYSTEM_PROMPT = f"""You are an automated QA gate. You do NOT rewrite or
improve the response — you only judge it and report defects.

You receive: the user's original message, the planned sub-questions, and the draft
final response.

CHECK EACH OF THESE INDEPENDENTLY
 1. completeness    - is there a distinct, on-point answer for every sub-question id?
 2. question_drift  - does any answer respond to a different question than asked?
 3. unsupported_claim - does any sentence assert a cause, benchmark, target, or
                      comparison that the supplied facts do not contain?
 4. causal_language - does any sentence claim proven causation ("because",
                      "caused by", "driven by", "due to")?
 5. hedging_missing - does a driver hypothesis appear without confirmation_needed?
 6. pii_exposure    - does the response expose an individual member, user name, or
                      case identifier in narrative prose rather than in a table?
 7. tone            - is it free of emoji, hype, and speculation?

Be strict. A false pass costs far more than a false fail in a regulated setting.
Report the defect verbatim — quote the offending sentence.

{P.JSON_DISCIPLINE}

RESPONSE SCHEMA
{{
  "verdict": "PASS" | "FAIL",
  "defects": [
    {{"check": "completeness", "severity": "blocking" | "advisory",
      "sub_question_id": "q2", "offending_text": "", "explanation": ""}}
  ],
  "unanswered_ids": ["q2"]
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
class ResponseAgent:
    def __init__(self, llm: StructuredLLM, *, enable_validator: bool = True) -> None:
        self.llm = llm
        self.enable_validator = enable_validator

    def synthesize(self, user_question: str, outcomes_payload: list[dict]) -> Synthesis | None:
        """One ``synthesis`` call. Returns ``None`` when the model is unavailable or invalid."""
        try:
            return self.llm.structured(
                "synthesis", SYNTHESIS_SYSTEM_PROMPT,
                P.user_payload(user_question=user_question, sub_question_outcomes=outcomes_payload),
                Synthesis)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("synthesis unavailable: %s", type(exc).__name__)
            return None

    def review(self, user_question: str, sub_questions: list[dict], draft: dict) -> ResponseVerdict | None:
        """One ``response_validation`` call (skipped when disabled)."""
        if not self.enable_validator:
            return None
        try:
            return self.llm.structured(
                "response_validation", VALIDATOR_SYSTEM_PROMPT,
                P.user_payload(user_question=user_question, sub_questions=sub_questions, draft_response=draft),
                ResponseVerdict)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("response validator unavailable: %s", type(exc).__name__)
            return None
