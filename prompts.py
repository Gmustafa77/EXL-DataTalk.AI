"""Shared prompt building blocks for every DataTalk.AI agent.

Agent-specific system prompts live inside their agent modules
(``agents/planner_agent.py``, ``agents/sql_agent.py``, ``agents/response_agent.py``,
``agents/insight_generation_agent.py``). This module holds only the text that
several agents must agree on:

* ``JSON_DISCIPLINE`` / ``ANTI_HALLUCINATION`` — output and grounding rules,
* ``schema_block`` — the closed-world table contract,
* ``metric_dictionary`` — the certified SQL definition of every business metric,
* ``user_payload`` — deterministic JSON serialisation of agent inputs.

Bump ``PROMPT_VERSION`` whenever any prompt text changes. It is recorded with every
entry in ``QA.json`` so stored answers can be traced to the prompt set that produced them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

PROMPT_VERSION = "2026.10.20-refactor.1"


def user_payload(**fields: Any) -> str:
    """Serialise agent input with stable key order so identical inputs give identical prompts."""
    return json.dumps(fields, default=str, sort_keys=True, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

JSON_DISCIPLINE = """OUTPUT DISCIPLINE (non-negotiable)
- Return exactly one JSON object and nothing else.
- No markdown, no code fences, no prose before or after the JSON.
- No trailing commas. No comments. No NaN, Infinity, or Python literals.
- Use null for unknown values. Never invent a placeholder string.
- If you cannot comply with the requested schema, return the documented failure
  object for this task rather than an unstructured message."""

ANTI_HALLUCINATION = """GROUNDING RULES (non-negotiable)
- You operate in a closed world. Only the tables, columns, values, and numbers
  supplied in this prompt exist.
- Never invent a table, column, column alias, category value, provider, sponsor,
  member, unit, site, date, threshold, benchmark, metric definition, or industry
  average.
- Never carry knowledge from other healthcare datasets or public benchmarks into
  this answer.
- If a required element is missing, return the failure object with the correct
  status instead of substituting a plausible alternative.
- Uncertainty is reported, never smoothed over."""


def schema_block(
    fq_table: str,
    raw_columns: Iterable[str],
    derived_columns: Iterable[str],
    column_types: dict | None = None,
) -> str:
    """Closed-world schema contract injected into every data-facing prompt."""
    typed = ""
    if column_types:
        typed = "\nColumn data types: " + json.dumps(column_types, sort_keys=True)
    return f"""DATA CONTRACT
Approved table (the ONLY queryable object): {fq_table}
Queryable raw SQL columns: {json.dumps(sorted(raw_columns))}{typed}
Python-only derived fields — these DO NOT EXIST in SQL and must never appear in a
query, not even in an alias, ORDER BY, or HAVING clause:
{json.dumps(sorted(derived_columns))}
Column names contain spaces and are case-sensitive. Every column reference must be
wrapped in backticks exactly as spelled above.
Provider dimension columns use underscores in SQL: `Provider_ID`, `Provider_Name`,
`Provider_Type`, `Provider_Specialty`, `Network_Status`. The space-separated forms
are Python-only and are forbidden in SQL."""


def metric_dictionary(sla_days: int, escalation_level: int) -> str:
    """Single source of truth for every business metric. Keep in sync with load_data()."""
    return f"""CERTIFIED METRIC DICTIONARY
Every metric below has exactly one correct SQL expression. Do not re-derive,
re-interpret, or approximate these.

IMPORTANT LANGUAGE MAPPING
- "appeal volume", "case volume", "number of appeals", "appeal count", "total appeals",
  "how many appeals", and "highest/lowest appeals" ALL mean `total_appeals`.
- "closed appeals/count" means `closed_appeals`; "open/pending appeals" means `open_appeals`.
- These are certified metrics. NEVER return MISSING_FIELD for those phrases.
- "key insights", "summary", or "overview" means use the certified overview set:
  total_appeals, open_appeals, closed_appeals, overdue_appeals, compliance_rate,
  overturn_rate, avg_turnaround_days.
- A request for "risk areas based on determinations" is NOT ambiguous. Default to
  dimension `Determination` with total_appeals and, where meaningful, overturn_rate.
- If the user asks which sponsor/provider/unit is highest by volume and then asks
  about that winner, the second task depends on the first; do not ask the user to
  restate the winning entity. Use `upstream_results` supplied to SQL generation.

total_appeals           : COUNT(`Case Identifier`)
closed_appeals          : SUM(CASE WHEN `Case Status` = 'CLOSED' THEN 1 ELSE 0 END)
open_appeals            : SUM(CASE WHEN `Case Status` = 'OPEN' THEN 1 ELSE 0 END)
overdue_appeals         : SUM(CASE WHEN `Case Status` = 'OPEN'
                          AND `Received Date` <= date_sub(current_date(), {sla_days}) THEN 1 ELSE 0 END)
escalated_appeals       : SUM(CASE WHEN `Level Type Code` = {escalation_level} THEN 1 ELSE 0 END)
avg_turnaround_days     : AVG(CASE WHEN `Case Status` = 'CLOSED'
                          THEN datediff(`Closed Date (CaseHistory)`, `Received Date`) END)

case_status_open        : `Case Status` = 'OPEN'
case_status_closed      : `Case Status` = 'CLOSED'
overdue_open            : `Case Status` = 'OPEN'
                          AND `Received Date` <= date_sub(current_date(), {sla_days})
escalated               : `Level Type Code` = {escalation_level}
transferred             : upper(`Work Item Status Short Name`) = 'TRANSFERRED'
compliant_closure       : `Case Status` = 'CLOSED'
                          AND upper(`Closed In Compliance Indicator`) = 'Y'
non_compliant_closure   : `Case Status` = 'CLOSED'
                          AND upper(`Closed In Compliance Indicator`) <> 'Y'
compliance_rate         : compliant_closure count / closed count * 100
                          (denominator = CLOSED cases only, never total cases)
determination_overturn  : `Determination` IN ('OVERTURNED','PARTIALLY OVERTURNED')
determination_upheld    : `Determination` = 'UPHELD'
overturn_rate           : overturn count / closed count * 100
                          (denominator = CLOSED cases only)
turnaround_days         : datediff(`Closed Date (CaseHistory)`, `Received Date`)
                          (defined for CLOSED cases only; never compute for OPEN)
open_aging_days         : datediff(current_date(), `Received Date`)
                          (defined for OPEN cases only)
out_of_network          : lower(trim(`Network_Status`)) = 'out-of-network'
received_month          : date_format(`Received Date`, 'yyyy-MM')
closed_month            : date_format(`Closed Date (CaseHistory)`, 'yyyy-MM')

DENOMINATOR DISCIPLINE
- Any rate over determinations or compliance uses CLOSED cases as denominator.
- Any rate over workload or intake uses ALL cases as denominator.
- Never mix the two in one expression. Always emit the denominator as its own
  column in the result so it can be independently verified."""
