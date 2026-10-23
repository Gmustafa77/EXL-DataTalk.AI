"""Planner agent — splits one user message into executable sub-questions.

LLM calls: exactly one logical call (purpose ``planner``) per analytics request.
If that call fails or returns an invalid plan, a deterministic rule-based splitter
creates the sub-questions instead, so a question is never silently dropped.

Input : user question, ``as_of_date``, grounding values, last 3 conversation turns.
Output: :class:`QueryPlan` (validated) → list of sub-question dicts for the engine.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

import prompts as P
from core.llm import LLMResponseError, LLMUnavailableError, StructuredLLM
from core.schemas import Grain

log = logging.getLogger("datatalk.planner")

ANALYSIS_TYPES = ("metric", "breakdown", "trend", "comparison", "ranking", "distribution",
                  "drilldown", "forecast", "anomaly", "root_cause", "overview")
AnalysisType = Literal["metric", "breakdown", "trend", "comparison", "ranking", "distribution",
                       "drilldown", "forecast", "anomaly", "root_cause", "overview"]
ChartHint = Literal["line", "area", "bar", "stacked_bar", "grouped_bar", "pie", "donut", "scatter",
                    "heatmap", "kpi", "none"]
PlanItemStatus = Literal["SUPPORTED", "PARTIAL", "UNRELATED", "AMBIGUOUS", "INVALID_FILTER", "MISSING_FIELD"]


# ═══════════════════════════════════════════════════════════════════════════════
# Response contract
# ═══════════════════════════════════════════════════════════════════════════════
class SubQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=8, description="q1, q2, ... in execution order")
    question_text: str = Field(min_length=1, description="Verbatim slice of the user's message")
    answerable: bool = True
    status: PlanItemStatus = "SUPPORTED"
    reason: str = ""
    analysis_type: AnalysisType = "metric"
    grain: Grain = Grain.NONE
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    depends_on: str | None = None
    dependency_note: str = ""
    expects_chart: bool = True
    chart_hint: ChartHint = "none"

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("analysis_type", "chart_hint", "grain", mode="before")
    @classmethod
    def _lower(cls, v: Any) -> Any:
        return v.strip().lower().replace(" ", "_") if isinstance(v, str) else v

    @field_validator("depends_on", mode="before")
    @classmethod
    def _depends(cls, v: Any) -> Any:
        return None if v in ("", "null", "none", "None") else v

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: Any) -> Any:
        return {} if v is None else v


class QueryPlan(BaseModel):
    overall_status: Literal["SUPPORTED", "PARTIAL", "UNRELATED"]
    question_count: int = Field(ge=0)
    sub_questions: list[SubQuestion] = Field(default_factory=list, max_length=10)

    @field_validator("overall_status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════════════════════
def build_system_prompt(
    fq_table: str,
    raw_columns: Iterable[str],
    derived_columns: Iterable[str],
    sla_days: int,
    escalation_level: int,
) -> str:
    return f"""You are the Query Understanding and Planning stage of a regulated
healthcare Appeals analytics system. You DO NOT answer questions and you DO NOT
write SQL. You only decompose the user's input into an executable task plan.

{P.schema_block(fq_table, raw_columns, derived_columns)}

{P.metric_dictionary(sla_days, escalation_level)}

DECOMPOSITION RULES
- A single user message may contain MULTIPLE questions. You must find every one of
  them. Missing a question is the single worst failure mode of this system.
- Treat each of the following as a separate question:
    * sentences separated by '?', ';', ' and also ', ' also ', ' plus ',
      ' as well as ', numbered or bulleted lists
    * a comparison that spans two populations ("X vs Y") when the user also asks
      for a driver, reason, or trend
    * a request for a number PLUS a request for an explanation, trend, breakdown,
      forecast, or recommendation
- Do NOT split a single coherent analytical request into artificial fragments.
  "overturn rate by provider for the last 3 months" is ONE question.
- Preserve the user's original wording in `question_text` for every sub-question.
- Order sub-questions by execution order. If sub-question B needs B's inputs from
  A's result (for example "then break those providers down by specialty"), set
  `depends_on` to A's id and explain the dependency in `dependency_note`.
- Independent sub-questions MUST have depends_on = null so they can be executed in
  parallel.

PER SUB-QUESTION CLASSIFICATION
analysis_type must be exactly one of:
  metric        - a single scalar or small set of scalars
  breakdown     - one metric grouped by one or more dimensions
  trend         - a metric over a time grain
  comparison    - the same metric across two or more named populations/periods
  ranking       - top/bottom N by a metric
  distribution  - share/mix/composition of a whole
  drilldown     - row-level records matching a filter
  forecast      - projection beyond the observed period
  anomaly       - outlier or deviation detection
  root_cause    - decomposition of a change into contributing factors
  overview      - a compact portfolio summary using the certified overview metric set

grain: the time grain requested, one of day, week, month, quarter, year, none.
dimensions: dimension columns the answer must be grouped by, using EXACT raw
  column spellings from the data contract. Empty list if none.
metrics: metric keys from the certified metric dictionary. Never free-text.
filters: an object of resolved filters. Resolve relative dates against
  `as_of_date` supplied in the user payload and emit ISO dates, never phrases like
  "last month". If a filter value is not present in `available_values`, do not
  guess — mark the sub-question unanswerable with status INVALID_FILTER.
expects_chart: true when a visual materially helps (trend, breakdown, comparison,
  ranking, distribution). false for a single scalar or a row-level drilldown.
chart_hint: one of line, area, bar, stacked_bar, grouped_bar, pie, donut, scatter,
  heatmap, kpi, none. This is ADVISORY ONLY — the rendering layer may override it.

AMBIGUITY POLICY
- Resolve ambiguity in favor of the certified metric dictionary. Do not ask the
  user a clarifying question unless the request is genuinely undecidable.
- If undecidable, emit the sub-question with answerable=false and status AMBIGUOUS,
  and put the single most useful clarifying question in `reason`.
- Partial plans are allowed and expected: some sub-questions may be answerable and
  others not. Never drop an unanswerable sub-question — report it.

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

RESPONSE SCHEMA
{{
  "overall_status": "SUPPORTED" | "PARTIAL" | "UNRELATED",
  "question_count": 2,
  "sub_questions": [
    {{
      "id": "q1",
      "question_text": "verbatim slice of the user's message",
      "answerable": true,
      "status": "SUPPORTED",
      "reason": "",
      "analysis_type": "trend",
      "grain": "month",
      "dimensions": ["Current Unit Name"],
      "metrics": ["overturn_rate"],
      "filters": {{"Received Date": {{"from": "2026-04-01", "to": "2026-09-30"}}}},
      "depends_on": null,
      "dependency_note": "",
      "expects_chart": true,
      "chart_hint": "line"
    }}
  ]
}}

FAILURE OBJECT (use when the whole message is off-domain)
{{"overall_status": "UNRELATED", "question_count": 0, "sub_questions": []}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
_SPLIT_RE = re.compile(
    r"(?:(?<=\?)\s+)"
    r"|\s*;\s*"
    r"|\s+(?:and\s+also|also\s+tell\s+me|also\s+show|as\s+well\s+as|and\s+then|plus\s+also)\s+"
    r"|\s*\n\s*(?=\d+[\.\)]\s)"
    r"|\s*\n\s*(?=[-*•]\s)",
    flags=re.I,
)


def rule_based_split(text: str, max_parts: int) -> list[str]:
    """Deterministic splitter on '?', ';', 'and also', numbered and bulleted lists."""
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip(" -*•\t") for p in _SPLIT_RE.split(raw)]
    parts = [re.sub(r"^\d+[\.\)]\s*", "", p).strip() for p in parts if p and len(p.strip()) > 8]
    out, seen = [], set()
    for part in parts:
        if part.lower() not in seen:
            seen.add(part.lower())
            out.append(part)
    return (out or [raw])[:max_parts]


def _fallback_subs(fragments: list[str]) -> list[dict]:
    return [SubQuestion(id=f"q{i + 1}", question_text=f, chart_hint="none").model_dump(mode="json")
            for i, f in enumerate(fragments)]


class PlannerAgent:
    """Turns a user message into an ordered list of sub-question dicts."""

    purpose = "planner"

    def __init__(self, llm: StructuredLLM, *, fq_table: str, raw_columns: Iterable[str],
                 derived_columns: Iterable[str], sla_days: int, escalation_level: int,
                 max_sub_questions: int) -> None:
        self.llm = llm
        self.max_sub_questions = max_sub_questions
        self.system = build_system_prompt(fq_table, raw_columns, derived_columns, sla_days, escalation_level)

    def plan(self, question: str, *, as_of_date: str, available_values: dict,
             conversation_context: list[str] | None = None) -> tuple[str, list[dict]]:
        """Return ``(overall_status, sub_questions)``.

        ``overall_status`` is ``UNRELATED`` (no sub-questions) when the planner judged the
        whole message off-domain, ``FALLBACK`` when the rule-based splitter was used.
        """
        payload = P.user_payload(user_question=question, as_of_date=as_of_date,
                                 available_values=available_values,
                                 recent_turns=(conversation_context or [])[-3:])
        try:
            plan = self.llm.structured(self.purpose, self.system, payload, QueryPlan)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("planner unavailable, using rule-based split: %s", type(exc).__name__)
            return "FALLBACK", _fallback_subs(rule_based_split(question, self.max_sub_questions))

        if plan.overall_status == "UNRELATED" and not plan.sub_questions:
            return "UNRELATED", []
        subs = [s.model_dump(mode="json") for s in plan.sub_questions][: self.max_sub_questions]
        if not subs:
            return "FALLBACK", _fallback_subs(rule_based_split(question, self.max_sub_questions))
        if len(subs) == 1:
            # Under-splitting silently drops a question; trust the splitter when it finds more.
            fragments = rule_based_split(question, self.max_sub_questions)
            if len(fragments) > 1:
                log.info("planner under-split; recovering %d fragments", len(fragments))
                subs = [{**subs[0], "id": f"q{i + 1}", "question_text": f, "depends_on": None}
                        for i, f in enumerate(fragments)]
        # Keep ids unique and dependencies pointing at known ids.
        ids = [s["id"] for s in subs]
        if len(set(ids)) != len(ids):
            for i, s in enumerate(subs):
                s["id"] = f"q{i + 1}"
            ids = [s["id"] for s in subs]
        for s in subs:
            if s.get("depends_on") not in ids or s.get("depends_on") == s["id"]:
                s["depends_on"] = None
        return plan.overall_status, subs
