"""SQL agent — writes one governed Databricks SELECT per sub-question.

LLM calls per sub-question:
  * 1 × ``sql_generation`` (always, when the sub-question is answerable),
  * 0–1 × ``sql_repair`` (only when the read-only validator rejects the SQL and
    ``ENABLE_SQL_REPAIR`` is on).

Every statement is checked by ``core.sql_guard.validate_read_only`` before it can
run; the agent never executes SQL itself.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

import prompts as P
from core.llm import LLMResponseError, LLMUnavailableError, StructuredLLM
from core.schemas import Grain
from core.sql_guard import SqlCheck

log = logging.getLogger("datatalk.sql_agent")

SqlStatus = Literal["SUPPORTED", "UNRELATED", "MISSING_FIELD", "INVALID_FILTER", "AMBIGUOUS",
                    "UNSUPPORTED_METRIC"]


# ═══════════════════════════════════════════════════════════════════════════════
# Response contract (shared by generation and repair)
# ═══════════════════════════════════════════════════════════════════════════════
class Measure(BaseModel):
    column: str
    type: str = ""
    numerator: str | None = None
    denominator: str | None = None


class SqlGeneration(BaseModel):
    answerable: bool
    status: SqlStatus = "SUPPORTED"
    reason: str = ""
    sql: str = ""
    intent: str = ""
    referenced_columns: list[str] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    grain: Grain = Grain.NONE
    dimensions: list[str] = Field(default_factory=list)
    measures: list[Measure | str] = Field(default_factory=list)
    expected_row_scale: str = ""
    is_aggregate: bool = False
    self_check: dict[str, Any] = Field(default_factory=dict)
    repair_applied: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("grain", mode="before")
    @classmethod
    def _grain(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else (v or "none")

    @field_validator("filters_applied", "self_check", mode="before")
    @classmethod
    def _dicts(cls, v: Any) -> Any:
        return {} if v is None else v

    @model_validator(mode="after")
    def _sql_when_answerable(self) -> SqlGeneration:
        if self.answerable and not self.sql.strip():
            raise ValueError("answerable=true requires a non-empty sql statement")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════
def build_system_prompt(
    fq_table: str,
    raw_columns: Iterable[str],
    derived_columns: Iterable[str],
    sla_days: int,
    escalation_level: int,
    max_row_query_limit: int,
    column_types: dict | None = None,
) -> str:
    return f"""You generate exactly ONE safe, read-only Databricks SQL SELECT
statement for ONE sub-question of a healthcare Appeals analytics request. You are
a compiler, not an analyst. You produce SQL and metadata — never narrative.

{P.schema_block(fq_table, raw_columns, derived_columns, column_types)}

{P.metric_dictionary(sla_days, escalation_level)}

DEPENDENT-QUESTION RULES
- `upstream_results` may contain verified rows from a prerequisite sub-question.
- When the current question says "its", "those", "that sponsor", "that provider",
  or otherwise depends on a winner identified upstream, resolve the reference from
  `upstream_results`. Do not ask the user to repeat it.
- Use only literal values present in upstream_results; never invent a value.
- If the required upstream task failed or has no rows, return AMBIGUOUS with a clear reason.

SQL CONSTRUCTION RULES
- Dialect: Databricks SQL (Spark SQL). Use only Databricks-supported functions.
- Exactly one statement. No semicolon. No SQL comments (`--`, `/* */`).
- Read-only. INSERT, UPDATE, DELETE, MERGE, DROP, CREATE, ALTER, TRUNCATE, COPY,
  GRANT, REVOKE, OPTIMIZE, VACUUM, CALL, EXECUTE, USE, and SET are forbidden.
- `SELECT *` is forbidden. Project named columns only.
- Only {fq_table} may appear in FROM/JOIN. Self-joins on this table are permitted
  only when unavoidable; prefer window functions or conditional aggregation.
- Wrap every column reference in backticks.
- Alias every computed column with a clear, human-readable, snake-free label that
  a chart axis can use directly, e.g. `Overturn Rate`, `Closed Cases`, `Month`.
- NULL SAFETY: guard every division with a CASE or NULLIF so a zero denominator
  returns NULL, never a divide-by-zero and never a silent 0.
- CAST explicitly when mixing integer and decimal arithmetic. Percentages must be
  computed in double precision and ROUND(...) to 2 decimals.
- Emit the denominator of every rate as its own column so downstream validation can
  recompute the rate independently.
- Deterministic ordering: every non-aggregate result must have an ORDER BY that
  fully determines row order. Ties must be broken by a unique-ish column.
- Time grain: when the sub-question has a grain, emit the period as a STRING column
  formatted 'yyyy-MM' (month), 'yyyy-Qq' (quarter), or 'yyyy-MM-dd' (day), and
  ORDER BY it ascending. Never return a raw timestamp for a trend axis.
- Filter values must match the values in `available_values` exactly. Use
  upper(trim(col)) on both sides when matching free-text status/determination
  values.

RESULT SHAPE RULES (these directly control chart correctness)
- Aggregation queries: no LIMIT required, but a ranking must use an explicit
  LIMIT N matching the user's N (default 10 if the user said "top" without N).
- Row-level queries: LIMIT {max_row_query_limit} is mandatory.
- Never return more than 25 grouped rows for a breakdown unless the user explicitly
  asked for the full list — rank and LIMIT instead. Oversized breakdowns produce
  unreadable charts.
- One row per (dimension combination, period). Never emit duplicate keys; the chart
  layer treats duplicate keys as a data-integrity failure and will reject the query.

FORBIDDEN PATTERNS
- Referencing any Python-only derived field.
- Hardcoding a count, rate, date, or threshold that should be computed.
- Using an industry benchmark or target that was not supplied.
- Returning both a raw row set and an aggregate in the same statement.
- Window functions over an unbounded frame without an explicit PARTITION BY when
  the sub-question is scoped to a dimension.

SELF-CHECK BEFORE RETURNING (perform silently, then report in `self_check`)
 1. Does every column in my SQL appear verbatim in the raw column list?
 2. Is every rate's denominator correct per the denominator discipline?
 3. Is every division guarded against zero/NULL?
 4. Does the result shape match the requested analysis_type and grain?
 5. Is row order deterministic?
 6. Have I avoided every forbidden pattern?
If any check fails, fix the SQL and re-run the checks before responding.

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

RESPONSE SCHEMA
{{
  "answerable": true,
  "status": "SUPPORTED",
  "reason": "",
  "sql": "SELECT ...",
  "intent": "one sentence describing precisely what this query returns",
  "referenced_columns": ["Case Status", "Determination"],
  "filters_applied": {{"Received Date": {{"from": "2026-04-01", "to": "2026-09-30"}}}},
  "grain": "month",
  "dimensions": ["Current Unit Name"],
  "measures": [
    {{"column": "Overturn Rate", "type": "percentage", "numerator": "Overturned Cases", "denominator": "Closed Cases"}}
  ],
  "expected_row_scale": "one row per unit per month",
  "is_aggregate": true,
  "self_check": {{"columns_valid": true, "denominators_valid": true, "division_guarded": true, "shape_matches_intent": true, "order_deterministic": true}}
}}

FAILURE OBJECT
{{"answerable": false,
  "status": "UNRELATED" | "MISSING_FIELD" | "INVALID_FILTER" | "AMBIGUOUS" | "UNSUPPORTED_METRIC",
  "reason": "one plain-English sentence a business user can act on",
  "sql": "", "intent": "", "referenced_columns": [], "filters_applied": {{}},
  "grain": "none", "dimensions": [], "measures": [], "expected_row_scale": "",
  "is_aggregate": false, "self_check": {{}}}}"""


REPAIR_SYSTEM_PROMPT = f"""You repair ONE rejected Databricks SQL statement for a
healthcare Appeals analytics system. You will receive the original sub-question,
the rejected SQL, and the exact validator or engine errors.

REPAIR RULES
- Fix ONLY what the errors identify. Do not restructure, re-scope, or "improve"
  a query that was otherwise correct.
- Do not broaden filters, drop a WHERE clause, or remove a GROUP BY to make an
  error disappear. Silently changing the question is worse than failing.
- If the error is an unknown column, map it to the correct raw column from the
  data contract. If no correct column exists, do NOT guess — return the failure
  object with status MISSING_FIELD.
- If the error is a Python-only derived field, recreate the equivalent logic from
  raw columns using the certified metric dictionary.
- If the error indicates the result was too large, add the correct aggregation or
  LIMIT rather than truncating meaning.
- You get exactly one repair attempt. If you are not confident the repair is
  correct, return the failure object.

{P.ANTI_HALLUCINATION}

{P.JSON_DISCIPLINE}

Return the SAME schema as the SQL generator, plus:
  "repair_applied": "one sentence describing exactly what you changed"
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SqlOutcome:
    """Result of generate → validate → (repair → validate)."""

    generation: SqlGeneration | None
    check: SqlCheck | None
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.generation is not None and self.check is not None and self.check.valid


class SqlAgent:
    def __init__(self, llm: StructuredLLM, validate: Callable[[str], SqlCheck], *, fq_table: str,
                 raw_columns: Iterable[str], derived_columns: Iterable[str], sla_days: int,
                 escalation_level: int, max_row_query_limit: int, enable_repair: bool = True) -> None:
        self.llm = llm
        self.validate = validate
        self.enable_repair = enable_repair
        self.system = build_system_prompt(fq_table, raw_columns, derived_columns, sla_days,
                                          escalation_level, max_row_query_limit)

    def generate(self, sub_question: dict, *, as_of_date: str, available_values: dict,
                 upstream_results: list | None = None) -> SqlOutcome:
        diag: dict[str, Any] = {}
        payload = P.user_payload(sub_question=sub_question, as_of_date=as_of_date,
                                 available_values=available_values,
                                 upstream_results=upstream_results or [])
        try:
            gen = self.llm.structured("sql_generation", self.system, payload, SqlGeneration)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("sql generation failed: %s", type(exc).__name__)
            return SqlOutcome(None, None, "The AI service could not generate a query for this question. "
                                          "Please retry.", {"error_type": type(exc).__name__})
        diag["self_check"] = gen.self_check
        if not gen.answerable:
            return SqlOutcome(gen, None, gen.reason or "No supported query could be generated.", diag)

        check = self.validate(gen.sql)
        if not check.valid and self.enable_repair:
            diag["repair_attempted"] = True
            try:
                repaired = self.llm.structured(
                    "sql_repair", REPAIR_SYSTEM_PROMPT,
                    P.user_payload(sub_question=sub_question, rejected_sql=gen.sql, errors=check.errors),
                    SqlGeneration)
            except (LLMResponseError, LLMUnavailableError) as exc:
                log.warning("sql repair failed: %s", type(exc).__name__)
                repaired = None
            if repaired is not None and repaired.answerable:
                check2 = self.validate(repaired.sql)
                if check2.valid:
                    gen, check = repaired, check2
                    diag["repair_applied"] = repaired.repair_applied
        if not check.valid:
            diag["validator_errors"] = list(check.errors)
            log.info("sql blocked by the read-only policy: %s", "; ".join(check.errors))
            return SqlOutcome(gen, check, "The generated query did not pass the read-only safety checks, so it "
                                          "was not run. Try rephrasing with the business terms used in the data "
                                          "(units, sponsors, providers, statuses, dates).", diag)
        return SqlOutcome(gen, check, "", diag)
