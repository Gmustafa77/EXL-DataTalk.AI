"""Supervisor agent — decides how every chat question is answered (no LLM calls itself).

Routing order for one question:

1. **Input check**      — empty / too short / too long → guidance message (0 LLM calls).
2. **Quick action**     — deterministic analysis on the in-memory dataset (0 LLM calls).
3. **Q&A store**        — normalised question found in ``QA.json`` → stored answer
   (0 LLM calls). Sits in front of every AI invocation and is provider-independent.
4. **Configuration**    — no ``GROQ_API_KEY`` → explanatory message (0 LLM calls).
5. **Analytics engine** — planner → SQL → insight → synthesis/validation (see README).
6. **Direct-SQL fallback** — only if the engine raised unexpectedly: one SQL generation
   (+0–1 repair) and the composer narrative (0–1).

A fully answered, aggregate AI result is written to ``QA.json`` for reuse. The store
is a backend concern only: a reused answer has the same route, content and layout as
a newly generated one, and nothing user-facing mentions it.

Every AI run is wrapped in :func:`core.llm.trace_calls`; the per-question call
accounting is returned in ``AssistantAnswer.meta`` for server-side telemetry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

import prompts as P
from agents.analytics_reasoning_agent import AnalyticsEngine, PipelineResult
from agents.insight_generation_agent import compose_insight
from agents.visualization_agent import ChartSource, fallback_charts
from core.fast_paths import match_fast_path, run_fast_path
from core.llm import trace_calls
from core.plot_style import style_figure
from core.qa_store import (
    QAStore,
    answer_from_payload,
    answer_to_payload,
    depends_on_conversation,
    question_may_contain_pii,
)
from core.schemas import AssistantAnswer
from core.settings import SENSITIVE_COLUMNS, Settings
from core.sql_guard import projects_sensitive_columns

log = logging.getLogger("datatalk.supervisor")

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000
EXECUTION_ERROR_MESSAGE = ("The query could not be completed against the live database. Please retry, or refine "
                           "the names, statuses or date ranges in your question.")


@dataclass
class QuestionContext:
    """Everything about the current session the supervisor needs to answer a question."""

    base_df: pd.DataFrame            # full loaded dataset (fast paths run on this)
    available_values: dict           # grounding values for the planner / SQL agent
    conversation: list[str]          # recent chat messages (planner uses the last three)


class Supervisor:
    def __init__(self, settings: Settings, engine: AnalyticsEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.qa_store = QAStore.from_settings(settings) if settings.qa_cache_enabled else None

    # ── entry point ──────────────────────────────────────────────────────────────
    def answer(self, question: str, ctx: QuestionContext, status_fn: Callable[[str], None]) -> AssistantAnswer:
        t0 = time.perf_counter()
        q = " ".join((question or "").split())
        if len(q) < MIN_QUESTION_CHARS:
            return self._done(AssistantAnswer(text="Please provide a more specific Appeals question.",
                                              route="rejected"), t0)
        if len(q) > MAX_QUESTION_CHARS:
            return self._done(AssistantAnswer(text=f"Please shorten the question to under {MAX_QUESTION_CHARS} "
                                                   "characters.", route="rejected"), t0)

        match = match_fast_path(q, ctx.base_df)
        if match is not None:
            status_fn("⚡ Running a deterministic quick action…")
            answer = run_fast_path(match, ctx.base_df, status_fn)
            self._ensure_chart(answer, q)
            answer.meta.update(fast_path=match.name)
            return self._done(answer, t0)

        # Q&A store: checked before any AI/LLM invocation.
        as_of = date.today().isoformat()
        reusable = self.qa_store is not None and not depends_on_conversation(q, ctx.conversation)
        if reusable:
            stored = self._from_store(q, as_of, ctx)
            if stored is not None:
                status_fn("Assembling the verified answer…")
                return self._done(stored, t0)

        if not self.settings.llm_configured:
            return self._done(AssistantAnswer(
                text="⚠️ The AI assistant is not configured: set `GROQ_API_KEY` in `.env` and restart the app. "
                     "Quick actions and dashboards keep working.", route="rejected"), t0)

        log.info(
            "Q&A cache miss or bypass; starting AI workflow"
        )

        answer = self._run_ai(
            q,
            ctx,
            ctx.conversation[-3:],
            status_fn,
        )

        # Backend diagnostics only. Existing UI remains unchanged.
        if self.qa_store is None:
            answer.meta["qa_cache"] = "disabled"

        elif not reusable:
            answer.meta["qa_cache"] = (
                "bypass_conversation_context"
            )

        elif not answer.cacheable:
            answer.meta["qa_cache"] = (
                "bypass_result_not_cacheable"
            )

        elif question_may_contain_pii(q):
            answer.meta["qa_cache"] = (
                "bypass_sensitive_question"
            )

        else:
            try:
                payload = answer_to_payload(answer)

                if payload is None:
                    answer.meta["qa_cache"] = (
                        "serialization_failed"
                    )
                else:
                    saved = self.qa_store.store(
                        q,
                        payload,
                        as_of_date=as_of,
                        metadata={
                            "model": self.settings.groq_model,
                            "prompt_version": P.PROMPT_VERSION,
                        },
                    )

                    answer.meta["qa_cache"] = (
                        "stored" if saved else "not_written"
                    )

            except Exception:
                # A persistence problem must not discard
                # an otherwise successful chatbot answer.
                log.exception(
                    "Q&A persistence failed; returning "
                    "the generated answer"
                )
                answer.meta["qa_cache"] = "write_error"

        log.info(
            "Q&A cache outcome=%s, route=%s, llm_calls=%s",
            answer.meta.get("qa_cache"),
            answer.route,
            answer.meta.get("llm_calls", 0),
        )

        return self._done(answer, t0)

    def _from_store(
        self,
        q: str,
        as_of: str,
        ctx: QuestionContext,
    ) -> AssistantAnswer | None:
        payload = self.qa_store.lookup(q, as_of_date=as_of)

        if payload is None:
            return None

        try:
            answer = answer_from_payload(payload, ctx.base_df)
        except Exception:
            log.exception(
                "stored answer could not be rebuilt; "
                "running the AI workflow"
            )
            return None

        answer.meta.update(
            llm_calls=0,
            llm_requests=0,
            calls_by_purpose={},
            tokens_in=0,
            tokens_out=0,
            est_cost_usd=0.0,
        )

        # NEW: record that this answer came from the cache.
        answer.meta["qa_cache"] = "hit"

        log.info(
            "question answered from the Q&A store (0 LLM calls)"
        )

        return answer

    # ── AI workflow ──────────────────────────────────────────────────────────────
    def _run_ai(self, q: str, ctx: QuestionContext, recent: list[str],
                status_fn: Callable[[str], None]) -> AssistantAnswer:
        with trace_calls() as trace:
            try:
                result = self.engine.run(q, available_values=ctx.available_values, status_fn=status_fn,
                                         conversation_context=recent)
                answer = self._from_pipeline(result, ctx)
            except Exception as exc:  # unexpected engine failure → degrade, never crash the chat
                log.exception("analytics engine failed; using direct-SQL fallback")
                status_fn("Primary pipeline unavailable; using the direct query path…")
                answer = self._direct_sql_fallback(q, ctx, type(exc).__name__)
        answer.meta.update(trace.summary())
        self._ensure_chart(answer, q)
        return answer

    def _from_pipeline(self, result: PipelineResult, ctx: QuestionContext) -> AssistantAnswer:
        sensitive = any(
            o.contract is not None and (not o.contract.is_aggregate
                                        or projects_sensitive_columns(o.contract.sql, SENSITIVE_COLUMNS))
            for o in result.outcomes)
        meta: dict = {"plan_status": result.plan_status, "question_count": result.question_count,
                "timings": result.timings, "validation": {k: v for k, v in result.validation.items()
                                                          if k in ("verdict", "unanswered_ids",
                                                                   "dropped_numeric_claims", "charts_reconciled")},
                "defects": len(result.validation.get("defects", []))}
        return AssistantAnswer(text=result.text, table=result.primary_df, figures=result.figures,
                               insight=result.insight_card or None, filtered_df=ctx.base_df,
                               route="analytics_engine", cacheable=result.fully_answered and not sensitive,
                               meta=meta)

    def _direct_sql_fallback(self, q: str, ctx: QuestionContext, error_type: str) -> AssistantAnswer:
        sub = {"id": "q1", "question_text": q, "answerable": True, "analysis_type": "metric", "depends_on": None}
        out = self.engine.sql_agent.generate(sub, as_of_date=date.today().isoformat(),
                                             available_values=ctx.available_values)
        meta = {"engine_error": error_type}
        if not out.ok:
            return AssistantAnswer(text=out.failure_reason or "The query could not be completed.",
                                   route="direct_sql_fallback", meta=meta)
        try:
            df, _ = self.engine.run_sql(out.check.sql)
        except Exception as exc:
            log.warning("fallback sql execution failed: %s", type(exc).__name__)
            return AssistantAnswer(text=EXECUTION_ERROR_MESSAGE, route="direct_sql_fallback", meta=meta)
        if df.empty:
            return AssistantAnswer(text="No records matched the criteria. Verify names, statuses and date ranges.",
                                   table=df, route="direct_sql_fallback", meta=meta)
        insight = compose_insight(df, question=q, sql=out.check.sql, config=self.engine.composer_cfg,
                                  llm=self.engine.llm if self.settings.insight_llm_narrative else None)
        insight["_sql"] = out.check.sql
        figures, _ = self.engine.charts.build(ChartSource("q1", df, ""))
        return AssistantAnswer(text=insight.get("insight", ""), table=df, figures=figures, insight=insight,
                               filtered_df=ctx.base_df, route="direct_sql_fallback", meta=meta)

    # ── helpers ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_chart(answer: AssistantAnswer, question: str) -> None:
        """Add a fallback chart when a table arrived without figures."""
        if answer.figures or answer.table is None or answer.table.empty:
            return
        answer.figures = fallback_charts(answer.table, question, style_fn=lambda f: style_figure(f, 440))
        if not answer.figures:
            answer.text += ("\n\n_No reliable chart type was available for this result shape; "
                            "the verified table is shown instead._")

    @staticmethod
    def _done(answer: AssistantAnswer, t0: float) -> AssistantAnswer:
        answer.meta.setdefault("llm_calls", 0)
        answer.meta["route"] = answer.route
        answer.meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return answer
