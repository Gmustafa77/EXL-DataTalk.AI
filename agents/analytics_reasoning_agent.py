"""Analytics reasoning agent — orchestrates the open-question AI workflow.

    question
      → PlannerAgent           (LLM: planner, 1 call)
      → per sub-question, independent ones first, dependent ones after:
           SqlAgent            (LLM: sql_generation 1, sql_repair 0–1)
           read-only validator + SqlExecutor (deterministic)
           ChartBuilder        (deterministic, reconciled to the table)
           compose_insight     (LLM: insight_narrative 0–1)
      → ResponseAgent          (LLM: synthesis 1 + response_validation 0–1, only for ≥2 sub-questions)
      → PipelineResult

Exact call counts are documented in the README ("LLM Usage and Call Accounting").
This module has no Streamlit imports; every dependency is injected so the engine
can be tested with a fake LLM and a fake SQL executor.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

import prompts as P
from agents.insight_generation_agent import ComposerConfig, compose_insight, verify_text
from agents.planner_agent import PlannerAgent
from agents.response_agent import ResponseAgent
from agents.sql_agent import SqlAgent
from agents.visualization_agent import ChartBuilder, ChartConfig
from core.llm import StructuredLLM
from core.sql_guard import SqlCheck

log = logging.getLogger("datatalk.engine")

UNRELATED_MESSAGE = (
    "No matching data was found for your request. Your question does not appear to be related to the "
    "available Appeals dataset. Please ask about appeal status trends, open vs closed appeals, turnaround "
    "time, outcomes, providers, members, date ranges, denials, or overturn analysis."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration and contracts
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class EngineConfig:
    fq_table: str
    raw_columns: list
    derived_columns: set
    sla_days: int = 30
    escalation_level: int = 2
    max_row_query_limit: int = 200
    max_result_rows: int = 5000
    max_sub_questions: int = 5
    max_parallel_workers: int = 1
    enable_sql_repair: bool = True
    enable_response_validator: bool = True
    enable_llm_narrative: bool = True
    polarity_overrides: dict = field(default_factory=dict)   # {"Column": "lower"|"higher"|"neutral"}
    min_denominator_for_rate: int = 5
    charts: ChartConfig = field(default_factory=ChartConfig)


@dataclass
class ResultContract:
    """Immutable evidence package. Everything downstream reads only from here."""

    sub_question_id: str
    question_text: str
    sql: str
    intent: str
    df: pd.DataFrame
    grain: str
    dimensions: list
    measures: list
    row_count: int
    truncated: bool
    result_hash: str
    latency_ms: float
    is_aggregate: bool = True
    from_cache: bool = False

    @staticmethod
    def hash_df(df: pd.DataFrame) -> str:
        try:
            payload = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values
            return hashlib.sha256(payload.tobytes()).hexdigest()[:16]
        except TypeError:  # unhashable cell values
            return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]


@dataclass
class SubQuestionOutcome:
    id: str
    question_text: str
    status: str                      # ANSWERED | NOT_ANSWERED
    failure_reason: str = ""
    contract: ResultContract | None = None
    insight: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    chart_notes: list = field(default_factory=list)
    dropped_claims: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    headline: str
    outcomes: list
    narrative: str
    primary_df: pd.DataFrame | None
    figures: list
    insight_card: dict
    validation: dict
    timings: dict
    question_count: int
    plan_status: str = "SUPPORTED"

    @property
    def text(self) -> str:
        if self.question_count > 1 and self.headline:
            return f"**{self.headline}**\n\n{self.narrative}"
        return self.narrative or self.headline

    @property
    def fully_answered(self) -> bool:
        return (self.question_count > 0 and bool(self.outcomes)
                and all(o.status == "ANSWERED" for o in self.outcomes)
                and not self.validation.get("unanswered_ids")
                and not any(d.get("severity") == "blocking" for d in self.validation.get("defects", [])))


# ═══════════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════════
class AnalyticsEngine:
    """Injected dependencies:

    * ``llm``       — :class:`core.llm.StructuredLLM`
    * ``validate``  — ``sql -> SqlCheck`` (read-only allow-list validator)
    * ``run_sql``   — ``sql -> (DataFrame, from_cache)``
    """

    def __init__(self, cfg: EngineConfig, llm: StructuredLLM, validate: Callable[[str], SqlCheck],
                 run_sql: Callable[[str], tuple[pd.DataFrame, bool]]) -> None:
        self.cfg = cfg
        self.run_sql = run_sql
        common = dict(fq_table=cfg.fq_table, raw_columns=cfg.raw_columns, derived_columns=cfg.derived_columns,
                      sla_days=cfg.sla_days, escalation_level=cfg.escalation_level)
        self.planner = PlannerAgent(llm, **common, max_sub_questions=cfg.max_sub_questions)
        self.sql_agent = SqlAgent(llm, validate, **common, max_row_query_limit=cfg.max_row_query_limit,
                                  enable_repair=cfg.enable_sql_repair)
        self.responder = ResponseAgent(llm, enable_validator=cfg.enable_response_validator)
        self.charts = ChartBuilder(cfg.charts)
        self.llm = llm
        self.composer_cfg = ComposerConfig(sla_days=cfg.sla_days, min_group_volume=cfg.min_denominator_for_rate,
                                           polarity_overrides=dict(cfg.polarity_overrides or {}),
                                           enable_llm_narrative=cfg.enable_llm_narrative)

    # ── public entry point ───────────────────────────────────────────────────────
    def run(self, user_query: str, *, available_values: dict, status_fn: Callable[[str], None] | None = None,
            conversation_context: list[str] | None = None) -> PipelineResult:
        t_start = time.perf_counter()
        timings: dict[str, Any] = {"prompt_version": P.PROMPT_VERSION}
        say = status_fn or (lambda _m: None)
        as_of = date.today().isoformat()

        say("Understanding the question and checking for multiple intents…")
        t0 = time.perf_counter()
        plan_status, subs = self.planner.plan(user_query, as_of_date=as_of, available_values=available_values,
                                              conversation_context=conversation_context)
        timings["plan_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if plan_status == "UNRELATED" or not subs:
            return PipelineResult(UNRELATED_MESSAGE, [], UNRELATED_MESSAGE, None, [], {},
                                  {"verdict": "PASS", "defects": [], "unanswered_ids": []}, timings, 0,
                                  plan_status="UNRELATED")
        if len(subs) > 1:
            say(f"Detected {len(subs)} distinct questions — planning execution…")

        t0 = time.perf_counter()
        outcomes = self._execute_plan(subs, available_values, as_of, say)
        timings["execute_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        say("Assembling the verified answer…")
        t0 = time.perf_counter()
        headline, narrative = self._synthesize(user_query, outcomes)
        validation = self._validate_response(user_query, subs, outcomes, headline, narrative)
        if validation.get("unanswered_ids"):
            narrative += self._gap_disclosure(validation["unanswered_ids"], outcomes)
        timings["synthesis_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        answered = [o for o in outcomes if o.status == "ANSWERED" and o.contract is not None]
        return PipelineResult(
            headline=headline, outcomes=outcomes, narrative=narrative,
            primary_df=answered[0].contract.df if answered else None,
            figures=[f for o in outcomes for f in o.figures],
            insight_card=self._build_insight_card(outcomes, headline), validation=validation,
            timings=timings, question_count=len(subs), plan_status=plan_status,
        )

    # ── DAG execution ────────────────────────────────────────────────────────────
    def _execute_plan(self, subs: list[dict], values: dict, as_of: str, say: Callable) -> list[SubQuestionOutcome]:
        independent = [s for s in subs if not s.get("depends_on")]
        dependent = [s for s in subs if s.get("depends_on")]
        results: dict[str, SubQuestionOutcome] = {}
        workers = max(1, min(self.cfg.max_parallel_workers, len(independent)))

        if independent:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # copy_context() lets worker threads record LLM calls in the caller's CallTrace.
                futures = {pool.submit(contextvars.copy_context().run, self._run_sub_question, s, values, as_of): s
                           for s in independent}
                for done, fut in enumerate(as_completed(futures), 1):
                    s = futures[fut]
                    try:
                        results[s["id"]] = fut.result()
                    except Exception as exc:  # one branch must never kill the request
                        log.exception("sub-question %s crashed", s.get("id"))
                        results[s["id"]] = self._crashed(s, exc)
                    say(f"Completed {done}/{len(subs)} question(s)…")

        for s in dependent:  # sequential by design; small N
            try:
                results[s["id"]] = self._run_sub_question(s, values, as_of, upstream=results)
            except Exception as exc:
                log.exception("dependent sub-question %s crashed", s.get("id"))
                results[s["id"]] = self._crashed(s, exc)
            say(f"Completed dependent question {s.get('id')}…")
        return [results[s["id"]] for s in subs if s["id"] in results]

    @staticmethod
    def _crashed(s: dict, exc: Exception) -> SubQuestionOutcome:
        return SubQuestionOutcome(id=s.get("id", "q?"), question_text=s.get("question_text", ""),
                                  status="NOT_ANSWERED",
                                  failure_reason="The analytics service could not complete this question. "
                                                 "Please retry the request.",
                                  diagnostics={"error_type": type(exc).__name__})

    def _run_sub_question(self, sub: dict, values: dict, as_of: str,
                          upstream: dict | None = None) -> SubQuestionOutcome:
        qid, qtext = sub.get("id", "q1"), sub.get("question_text", "")
        if not sub.get("answerable", True):
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED",
                                      sub.get("reason") or "This question is not supported by the Appeals dataset.")

        upstream_rows: list[dict] = []
        if upstream and sub.get("depends_on") in upstream:
            up = upstream[sub["depends_on"]]
            if up.contract is not None:
                head = up.contract.df.head(10).copy()
                for column in head.columns:
                    if pd.api.types.is_datetime64_any_dtype(head[column]):
                        head[column] = head[column].astype(str)
                upstream_rows = head.astype(object).where(pd.notna(head), None).to_dict("records")

        sql_out = self.sql_agent.generate(sub, as_of_date=as_of, available_values=values,
                                          upstream_results=upstream_rows)
        if not sql_out.ok:
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED", sql_out.failure_reason,
                                      diagnostics=sql_out.diagnostics)
        gen, check = sql_out.generation, sql_out.check
        sql = check.sql

        t0 = time.perf_counter()
        try:
            df, from_cache = self.run_sql(sql)
        except Exception as exc:  # Databricks errors are reported, not raised
            log.warning("sql execution failed for %s: %s", qid, type(exc).__name__)
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED",
                                      "The validated query could not be completed against the live database. "
                                      "Please retry, or refine the names, statuses or date ranges.",
                                      diagnostics={**sql_out.diagnostics, "sql": sql})
        latency = (time.perf_counter() - t0) * 1000
        if df is None or df.empty:
            return SubQuestionOutcome(qid, qtext, "NOT_ANSWERED",
                                      "No records matched the criteria for this question. "
                                      "Verify names, statuses, and date ranges.",
                                      diagnostics={**sql_out.diagnostics, "sql": sql})

        df, norm_notes = self._normalize_result(df)
        contract = ResultContract(
            sub_question_id=qid, question_text=qtext, sql=sql, intent=gen.intent, df=df,
            grain=str(gen.grain), dimensions=list(gen.dimensions),
            measures=[m.model_dump() if hasattr(m, "model_dump") else m for m in gen.measures],
            row_count=len(df), truncated=len(df) >= self.cfg.max_result_rows,
            result_hash=ResultContract.hash_df(df), latency_ms=latency,
            is_aggregate=check.is_aggregate, from_cache=from_cache,
        )
        figures, notes = self.charts.build(contract, sub.get("analysis_type", ""))
        insight = self._compose(contract, notes + norm_notes)
        return SubQuestionOutcome(
            qid, qtext, "ANSWERED", contract=contract, insight=insight, figures=figures,
            chart_notes=notes + norm_notes, dropped_claims=insight.get("_dropped_claims", []),
            diagnostics={**sql_out.diagnostics, "sql": sql, "latency_ms": round(latency, 1),
                         "from_cache": from_cache, "result_hash": contract.result_hash,
                         "narrative_source": insight.get("narrative_source")},
        )

    def _compose(self, c: ResultContract, notes: list) -> dict:
        try:
            insight = compose_insight(c.df, question=c.question_text, sql=c.sql, config=self.composer_cfg,
                                      llm=self.llm if self.cfg.enable_llm_narrative else None, chart_notes=notes)
        except Exception:  # the answer must not fail because the narrative did
            log.exception("insight composition failed for %s", c.sub_question_id)
            msg = f"The validated query returned {c.row_count:,} verified row(s). Review the result table below."
            insight = {"insight": msg, "key_finding": {"text": msg, "refs": []}, "detailed_analysis": [],
                       "business_impact": [], "reasoning_steps": [], "evidence": [], "supporting_metrics": [],
                       "recommended_actions": [], "data_caveats": notes, "grounded": False,
                       "_dropped_claims": [], "_allowed_numbers": [], "composer_version": "fallback"}
        insight["question"] = c.question_text
        insight["_sql"] = c.sql
        insight["_result_hash"] = c.result_hash
        return insight

    @staticmethod
    def _normalize_result(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """Trim column names and drop exact duplicates. Never sums averages or rates."""
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        notes = []
        before = len(out)
        out = out.drop_duplicates()
        if len(out) < before:
            notes.append(f"{before - len(out)} exact duplicate row(s) were removed from the result.")
        dim_like = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
        if dim_like and out.duplicated(subset=dim_like).any():
            notes.append("Some groups appear more than once in the result; values are shown exactly as returned.")
        return out.reset_index(drop=True), notes

    # ── synthesis ────────────────────────────────────────────────────────────────
    def _synthesize(self, user_query: str, outcomes: list[SubQuestionOutcome]) -> tuple[str, str]:
        if len(outcomes) == 1:
            o = outcomes[0]
            if o.status == "ANSWERED":
                kf = o.insight.get("insight", "")
                return kf, kf
            return "Unable to answer.", o.failure_reason

        payload = [{
            "id": o.id, "question_text": o.question_text, "status": o.status,
            "failure_reason": o.failure_reason, "insight": o.insight.get("insight", ""),
            "key_facts": [{"label": e.get("label"), "value": e.get("display")}
                          for e in (o.insight.get("evidence") or [])[:12]],
        } for o in outcomes]
        res = self.responder.synthesize(user_query, payload)

        # Every synthesised number must exist in the evidence of some answered question.
        allowed = sorted({tuple(a) for o in outcomes for a in (o.insight.get("_allowed_numbers") or [])})
        by_id = {a.id: a for a in (res.answers if res else [])}
        lines = []
        for i, o in enumerate(outcomes, 1):
            fallback = o.insight.get("insight", "") if o.status == "ANSWERED" else o.failure_reason
            body = by_id[o.id].answer if o.id in by_id and by_id[o.id].answer else fallback
            if o.status == "ANSWERED" and body != fallback and not verify_text(body, allowed):
                body = fallback
            lines.append(f"**{i}. {o.question_text.strip()}**\n\n{body}")
        headline = res.headline if res else ""
        if headline and not verify_text(headline, allowed):
            headline = ""
        if not headline:
            first = next((o for o in outcomes if o.status == "ANSWERED"), None)
            headline = first.insight.get("insight", "") if first else "Unable to answer."
        note = res.cross_cutting_note if res else ""
        if note and not verify_text(note, allowed):
            note = ""
        return headline, "\n\n".join(lines) + (f"\n\n_{note}_" if note else "")

    def _validate_response(self, user_query: str, subs: list[dict], outcomes: list[SubQuestionOutcome],
                           headline: str, narrative: str) -> dict:
        unanswered = [o.id for o in outcomes if o.status != "ANSWERED"]
        missing = [s["id"] for s in subs if s["id"] not in {o.id for o in outcomes}]
        verdict: dict[str, Any] = {
            "verdict": "PASS" if not unanswered and not missing else "FAIL",
            "defects": [], "unanswered_ids": unanswered + missing,
            "charts_reconciled": all(
                (f.layout.meta or {}).get("result_hash") == o.contract.result_hash
                for o in outcomes if o.contract is not None for f in o.figures),
            "dropped_numeric_claims": sum(len(o.dropped_claims) for o in outcomes),
        }
        if len(subs) < 2:
            return verdict
        judged = self.responder.review(user_query, subs, {"headline": headline, "narrative": narrative})
        if judged is not None:
            verdict["defects"] = [d.model_dump() for d in judged.defects[:5]]
            for uid in judged.unanswered_ids:
                if uid not in verdict["unanswered_ids"] and uid in {s["id"] for s in subs}:
                    verdict["unanswered_ids"].append(uid)
        return verdict

    @staticmethod
    def _gap_disclosure(unanswered_ids: list, outcomes: list[SubQuestionOutcome]) -> str:
        by_id = {o.id: o for o in outcomes}
        lines = [f"- **{by_id[u].question_text.strip()}** — {by_id[u].failure_reason or 'Flagged by the QA review.'}"
                 for u in unanswered_ids if u in by_id]
        return ("\n\n**Not answered from the available data**\n\n" + "\n".join(lines)) if lines else ""

    # ── output ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_insight_card(outcomes: list[SubQuestionOutcome], headline: str) -> dict:
        """First answered question is the top card; all answered ones go under ``questions``."""
        answered = [o for o in outcomes if o.status == "ANSWERED" and o.insight]
        if not answered:
            return {}
        cards = [{**o.insight, "question": o.question_text,
                  "_sql": o.contract.sql if o.contract is not None else ""} for o in answered]
        lead = dict(cards[0])
        lead["questions"] = cards
        lead["_sql"] = "\n\n".join(f"-- {o.id}: {o.question_text}\n{o.contract.sql}"
                                   for o in answered if o.contract is not None)
        if len(cards) > 1:
            lead["insight"] = headline
            lead["key_finding"] = {"text": headline, "refs": []}
        return lead


__all__ = ["EngineConfig", "AnalyticsEngine", "ResultContract", "SubQuestionOutcome", "PipelineResult",
           "UNRELATED_MESSAGE"]
