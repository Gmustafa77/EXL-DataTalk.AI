"""Streamlit wiring: cached resources, data loading and session state.

This is the only ``core`` module that imports Streamlit. It keeps the caches in
one place:

* ``st.cache_data``     — the loaded appeals frame (``load_dataset``) and the
  grounding values sent to the AI (``grounding_values``); cleared by *Refresh data*.
* ``st.cache_resource`` — process-wide singletons: SQL executor, LLM client,
  analytics engine and supervisor.

The backend Q&A store (``core.qa_store``, ``data/QA.json``) is owned by the
supervisor and is independent of these in-memory caches: clearing Streamlit
caches does not delete stored answers.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import pandas as pd
import streamlit as st

from agents.analytics_reasoning_agent import AnalyticsEngine, EngineConfig
from agents.supervisor_agent import QuestionContext, Supervisor
from agents.visualization_agent import ChartConfig
from core import data as D
from core.llm import LLMClient
from core.plot_style import style_figure
from core.settings import (
    DERIVED_COLUMNS,
    ESCALATION_LEVEL,
    RAW_COLUMNS,
    SLA_DAYS,
    Settings,
    get_settings,
)
from core.sql_guard import SqlExecutor, validate_read_only

log = logging.getLogger("datatalk.runtime")


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger("datatalk")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    root.setLevel(settings.log_level.upper())


# ── resources ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_executor() -> SqlExecutor:
    s = get_settings()
    return SqlExecutor(partial(D.databricks_connect, s), max_result_rows=s.max_result_rows,
                       cache_ttl_seconds=s.sql_cache_ttl_seconds, cache_max_entries=s.sql_cache_max_entries)


@st.cache_resource(show_spinner=False)
def get_supervisor() -> Supervisor:
    s = get_settings()
    table = s.databricks_table
    validate = partial(validate_read_only, table=table, raw_columns=RAW_COLUMNS,
                       max_row_limit=s.max_row_query_limit)
    cfg = EngineConfig(
        fq_table=s.fq_table, raw_columns=list(RAW_COLUMNS), derived_columns=set(DERIVED_COLUMNS),
        sla_days=SLA_DAYS, escalation_level=ESCALATION_LEVEL, max_row_query_limit=s.max_row_query_limit,
        max_result_rows=s.max_result_rows, max_sub_questions=s.max_sub_questions,
        max_parallel_workers=s.max_parallel_workers, enable_sql_repair=s.enable_sql_repair,
        enable_response_validator=s.enable_response_validator, enable_llm_narrative=s.insight_llm_narrative,
        charts=ChartConfig(style_fn=lambda f: style_figure(f, 460)),
    )
    engine = AnalyticsEngine(cfg, LLMClient(s), validate, get_executor().run_validated)
    return Supervisor(s, engine)


# ── data ───────────────────────────────────────────────────────────────────────
def _ttl() -> float | None:
    ttl = get_settings().data_cache_ttl_seconds
    return float(ttl) if ttl else None


@st.cache_data(show_spinner=False, ttl=_ttl())
def load_dataset() -> pd.DataFrame:
    """Full appeals table with derived columns. Raises on connection/configuration errors."""
    s = get_settings()
    if not s.databricks_configured:
        raise RuntimeError("Databricks credentials missing. Set DATABRICKS_SERVER_HOSTNAME, "
                           "DATABRICKS_HTTP_PATH and DATABRICKS_ACCESS_TOKEN in .env.")
    raw = get_executor().run_trusted(D.full_load_sql(s))
    return D.enrich(raw)


@st.cache_data(show_spinner=False, ttl=_ttl())
def grounding_values() -> dict:
    """Distinct categorical values and date range sent to the planner and SQL agent."""
    return D.value_context(load_dataset(), get_settings().prompt_max_values_per_field)


def refresh_data() -> None:
    load_dataset.clear()
    grounding_values.clear()
    get_executor().clear()


@st.cache_data(ttl=60, show_spinner=False)
def ping_databricks() -> tuple[str, float]:
    """Connectivity check (``SELECT 1``), cached for 60 s."""
    s = get_settings()
    if not s.databricks_configured:
        return "not configured", 0.0
    try:
        t0 = time.perf_counter()
        with D.databricks_connect(s) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchall()
        return "connected", round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:  # report, do not crash the header
        log.warning("databricks ping failed: %s", type(exc).__name__)
        return "error", 0.0


# ── session state ──────────────────────────────────────────────────────────────
def init_session(base_df: pd.DataFrame) -> None:
    defaults: dict[str, Any] = {
        "chat_history": [], "active_query": None, "dashboard_df": base_df, "voice_transcript": "",
        "telemetry": {"questions": 0, "llm_calls": 0, "llm_requests": 0, "tokens_in": 0, "tokens_out": 0,
                      "cost_usd": 0.0, "dropped_claims": 0, "latencies_ms": [],
                      "session_start": pd.Timestamp.now()},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def record_answer_telemetry(meta: dict) -> None:
    t = st.session_state["telemetry"]
    t["questions"] += 1
    t["llm_calls"] += int(meta.get("llm_calls", 0))
    t["llm_requests"] += int(meta.get("llm_requests", 0))
    t["tokens_in"] += int(meta.get("tokens_in", 0))
    t["tokens_out"] += int(meta.get("tokens_out", 0))
    t["cost_usd"] += float(meta.get("est_cost_usd", 0.0))
    t["dropped_claims"] += int((meta.get("validation") or {}).get("dropped_numeric_claims", 0))
    t["latencies_ms"] = (t["latencies_ms"] + [meta.get("elapsed_ms", 0.0)])[-50:]


def question_context(base_df: pd.DataFrame) -> QuestionContext:
    """Context for the question being processed (the last chat entry is that question itself)."""
    values = grounding_values()
    previous = st.session_state.get("chat_history", [])[:-1]
    history = [m.get("content", "") for m in previous[-6:] if isinstance(m, dict)]
    return QuestionContext(base_df=base_df, available_values=values, conversation=history)
