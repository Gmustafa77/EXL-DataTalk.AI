"""DataTalk.AI — Streamlit entry point.

Responsibilities: page configuration, theme CSS, data loading, the shared header
and KPI strip, sidebar status, page navigation and the footer. All features live
in ``dashboards/``, ``components/``, ``agents/`` and ``core/``.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from anomaly_detection import detect_anomalies
from components import ui
from components.startup_loading import StartupLoader
from components.risk_findings import RiskConfig, render_risk_findings_tab
from core import runtime
from core.data import KPI_META, calculate_kpis, fmt_kpi, kpi_delta
from core.fast_paths import ANOMALY_CONFIG
from core.settings import AGING_THRESHOLD, ESCALATION_LEVEL, SLA_DAYS, get_settings
from dashboards.ai_assistant import render_ai_assistant
from dashboards.data_explorer import render_data_explorer
from dashboards.deep_dive_dashboard import render_deep_dive_tab
from dashboards.documentation import render_documentation_page
from dashboards.intelligence_dashboard import render_intelligence_page

st.set_page_config(page_title="DataTalk.AI • EXL", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")

settings = get_settings()
runtime.configure_logging(settings)
ui.inject_css()


# Full-width page layout without overriding individual widget containers.
st.markdown(
    """
    <style>
    /* Main page only: retain your full-width layout and spacing. */
    html body [data-testid="stAppViewContainer"]
        [data-testid="stMain"]
        [data-testid="stMainBlockContainer"],
    html body [data-testid="stAppViewContainer"] .main .block-container,
    html body .stMainBlockContainer,
    html body .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-top: 3.5rem !important;
        padding-left: 0.95rem !important;
        padding-right: 0.95rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* Outer wrappers: remove extra horizontal spacing. */
    html body [data-testid="stAppViewContainer"],
    html body [data-testid="stMain"],
    html body section.main {
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /*
    Leave blocks, element containers, and markdown wrappers untouched.
    This preserves the native sizing of microphone and other widgets.
    */

    /* Charts and tables only. */
    html body [data-testid="stPlotlyChart"],
    html body [data-testid="stDataFrame"],
    html body [data-testid="stTable"],
    html body [data-testid="stVegaLiteChart"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Mobile spacing. */
    @media (max-width: 900px) {
        html body [data-testid="stMainBlockContainer"],
        html body .block-container {
            padding-top: 7rem !important;
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── data ───────────────────────────────────────────────────────────────────────
startup_loader = StartupLoader()
startup_loader.show("loading")
try:
    base_df = runtime.load_dataset()
except Exception:  # configuration or connectivity problem: explain and stop (details in the server log)
    startup_loader.clear()
    runtime.log.exception("data load failed")
    st.error("❌ Could not load the appeals data from the data source.", icon="🔴")
    st.info("Ask the application owner to check the data-source settings in `.env` (see README → Configuration).")
    if st.button("Retry connection", key="startup_retry"):
        st.rerun()
    ui.footer()
    st.stop()
if base_df.empty:
    startup_loader.clear()
    st.warning("The appeals table returned no rows.")
    ui.footer()
    st.stop()

startup_loader.show("loaded", rows=len(base_df))
try:
    runtime.init_session(base_df)
    current_df: pd.DataFrame = st.session_state["dashboard_df"]
except Exception:
    startup_loader.clear()
    runtime.log.exception("session initialization failed")
    st.error("The dataset loaded, but the analytics session could not be initialized.")
    ui.footer()
    st.stop()
startup_loader.show("preparing", rows=len(base_df))


# ── header ─────────────────────────────────────────────────────────────────────
def render_header() -> None:
    t = st.session_state["telemetry"]
    db_status, db_ms = runtime.ping_databricks()
    lat = t["latencies_ms"]
    pills = "".join([
        ui.pill("🟢" if db_status == "connected" else "🔴" if db_status == "error" else "⚪", "DB",
                f"{db_ms:.0f} ms" if db_status == "connected" else db_status, "Databricks round trip (SELECT 1), refreshed every 60 s"),
        ui.pill("🧠", "LLM", settings.groq_model if settings.llm_configured else "not configured", "Groq model"),
        ui.pill("🤖", "AI calls", f"{t['llm_calls']}", f"{t['llm_requests']} provider request(s) this session"),
        ui.pill("🪙", "Tokens", f"{t['tokens_in'] + t['tokens_out']:,}", f"{t['tokens_in']:,} in · {t['tokens_out']:,} out"),
        ui.pill("💰", "Est. cost", f"${t['cost_usd']:.4f}", "Estimate from the price table in core/settings.py"),
        ui.pill("⏱️", "Avg answer", f"{(sum(lat) / len(lat) / 1000) if lat else 0:.1f}s", f"Across the last {len(lat)} answer(s)"),
        ui.pill("🛡️", "Blocked claims", f"{t['dropped_claims']}", "AI sentences removed because a number was not in the evidence"),
    ])
    ui.header(pills)


def render_kpis() -> None:
    baseline, active = calculate_kpis(base_df), calculate_kpis(current_df)
    cards = []
    for label, (key, icon) in KPI_META.items():
        delta, good = kpi_delta(label, active[key], baseline)
        active_value = fmt_kpi(label, active[key])
        baseline_value = fmt_kpi(label, baseline[key])
        sla_pct_key = {
            "Received SLA Met": "received_sla_met_pct",
            "Assigned SLA Met": "assigned_sla_met_pct",
        }.get(label)
        if sla_pct_key:
            active_value += f" ({active[sla_pct_key]:.1f}%)"
            baseline_value += f" ({baseline[sla_pct_key]:.1f}%)"
        cards.append(ui.kpi_card(icon, label, active_value, baseline_value, delta, good))
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)
    if len(current_df) < len(base_df):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.info(f"🎯 Dashboard filtered by the last quick action — showing **{len(current_df):,}** of "
                f"**{len(base_df):,}** appeals.")
        if c2.button("Show all appeals", width="stretch"):
            st.session_state["dashboard_df"] = base_df
            st.rerun()


def render_sidebar(documentation: st.Page) -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Session")
        st.caption("Light / dark mode: ⋮ menu → **Settings** → Theme.")
        if st.button("🔄 Refresh data", width="stretch", help="Reload the appeals table from Databricks"):
            runtime.refresh_data()
            st.session_state.pop("dashboard_df", None)
            st.rerun()
        st.markdown("### 📘 Documentation")
        st.page_link(documentation, label="Open DataTalk.AI documentation", icon="📘")
        st.markdown("### ℹ️ Business rules")
        st.caption(f"SLA {SLA_DAYS} days · aging threshold {AGING_THRESHOLD} days · escalation level {ESCALATION_LEVEL}")


# ── pages ──────────────────────────────────────────────────────────────────────
def assistant_page() -> None:
    render_ai_assistant(base_df)


def deep_dive_page() -> None:
    render_deep_dive_tab(current_df)


def intelligence_page() -> None:
    render_intelligence_page(current_df)


def risk_page() -> None:
    render_risk_findings_tab(
        current_df, RiskConfig(sla_days=SLA_DAYS, escalation_level=ESCALATION_LEVEL, aging_threshold=AGING_THRESHOLD),
        detect_anomalies_fn=lambda d: detect_anomalies(d, ANOMALY_CONFIG),
    )


def explorer_page() -> None:
    render_data_explorer(current_df)


documentation = st.Page(render_documentation_page, title="Documentation", icon="📘", url_path="documentation")

pages = [
    st.Page(assistant_page, title="AI Assistant", icon="🧠", url_path="assistant", default=True),
    st.Page(deep_dive_page, title="Deep Dive", icon="🔬", url_path="deep-dive"),
    st.Page(intelligence_page, title="Intelligence", icon="📋", url_path="intelligence"),
    st.Page(risk_page, title="Risk & Findings", icon="🏦", url_path="risk"),
    st.Page(explorer_page, title="Data Explorer", icon="📑", url_path="explorer"),
    documentation,
]
navigation = st.navigation(pages, position="top")

try:
    render_header()
    if navigation.url_path != documentation.url_path:  # the KPI strip is data, not documentation
        render_kpis()
    render_sidebar(documentation)
finally:
    startup_loader.clear()
navigation.run()
ui.footer()
