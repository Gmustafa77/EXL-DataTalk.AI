"""AI Assistant page: chat input (text or voice), quick actions, example questions
and the answer thread.

Questions are answered by ``agents.supervisor_agent.Supervisor``; this module only
handles interaction state and rendering. Typed, spoken, quick-action and example
questions all enter the same workflow through :func:`_submit`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.insight_panel import render_insight_panel
from components.ui import section
from core import runtime
from core.example_questions import EXAMPLE_CATEGORIES
from core.plot_style import PLOTLY_CONFIG
from speech_utils import render_mic_recorder, render_speaker_button

PRIMARY_ACTIONS = [
    ("🔮 Forecast", "forecast next 6 months"),
    ("🚨 Anomalies", "detect anomalies"),
    ("⚠️ Overdue", "overdue appeals"),
    ("🏢 Sponsors", "top plan sponsors"),
    ("🔀 Compare", "Compare East site vs West site side by side"),
]
MORE_ACTIONS = [
    ("🚨 Risk & anomalies", [("⚠️ Escalated appeals", "Escalated appeals"), ("⏳ Pending appeals", "Pending appeals"),
                            ("🎯 Risk overview", "Risk overview"), ("🏭 Reviewer segments", "Reviewer segment analysis")]),
    ("📊 Quick analysis", [("📈 Monthly trend", "Monthly trend"), ("🌍 By site", "Appeals by site"),
                          ("🏭 By unit", "Unit analysis"), ("💱 Source & filer", "Source breakdown"),
                          ("📦 By task type", "Task breakdown")]),
]


def _submit(question: str) -> None:
    text = " ".join((question or "").split())
    if text:
        st.session_state["chat_history"].append({"role": "user", "content": text})
        st.session_state["active_query"] = text


def _process_active_query(base_df: pd.DataFrame) -> None:
    question = st.session_state["active_query"]
    with st.chat_message("assistant", avatar="🧠"):
        with st.status("🔍 DataTalk.AI is analysing…", expanded=True) as status:
            try:
                answer = runtime.get_supervisor().answer(question, runtime.question_context(base_df), status.write)
            except Exception:  # keep the chat usable; details go to the server log
                runtime.log.exception("question processing failed")
                status.update(label="Something went wrong", state="error")
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": "Unable to complete this request. Please retry; the error "
                                                     "has been logged.", "meta": {"route": "error"}})
                st.session_state["active_query"] = None
                return
            status.update(label="✅ Analysis ready", state="complete", expanded=False)
    st.session_state["chat_history"].append({
        "role": "assistant", "content": answer.text, "df": answer.table, "figs": list(answer.figures),
        "insight": answer.insight, "meta": answer.meta,
    })
    st.session_state["dashboard_df"] = answer.filtered_df if answer.filtered_df is not None else base_df
    runtime.record_answer_telemetry(answer.meta)
    st.session_state["active_query"] = None
    st.rerun()


def _render_figures(figs: list, key: str) -> None:
    if not figs:
        return
    if len(figs) == 1:
        st.plotly_chart(figs[0], width="stretch", key=f"{key}_f0", config=PLOTLY_CONFIG)
        return
    if len(figs) > 2:
        st.plotly_chart(figs[0], width="stretch", key=f"{key}_f0", config=PLOTLY_CONFIG)
        rest = list(enumerate(figs))[1:]
    else:
        rest = list(enumerate(figs))
    for i in range(0, len(rest), 2):
        cols = st.columns(2)
        for col, (idx, fig) in zip(cols, rest[i:i + 2]):
            with col:
                st.plotly_chart(fig, width="stretch", key=f"{key}_f{idx}", config=PLOTLY_CONFIG)


def render_answer(message: dict, key: str) -> None:
    """Render one assistant turn (text, charts, read-aloud, insight panel, table)."""
    st.markdown("**AI Insight Findings**")
    st.markdown(message.get("content", ""))
    _render_figures(message.get("figs") or [], key)
    if message.get("content"):
        render_speaker_button(message["content"], key=key)
    if message.get("insight"):
        st.divider()
        render_insight_panel(message["insight"], result_df=message.get("df"), card_key=key,
                             answer_text=message.get("content", ""))
    table = message.get("df")
    if isinstance(table, pd.DataFrame) and not table.empty:
        with st.expander("📊 View result data", expanded=False):
            st.dataframe(table, width="stretch", hide_index=True)


def _render_example_questions(processing: bool, expanded: bool) -> None:
    """Browsable example questions; a click submits the question like a typed one."""
    with st.expander("💡 Example Questions", expanded=expanded):
        st.caption("Each example is answered from the live appeals data through the same workflow as a typed "
                   "question.")
        tabs = st.tabs([f"{c.icon} {c.name}" for c in EXAMPLE_CATEGORIES])
        for ci, (tab, category) in enumerate(zip(tabs, EXAMPLE_CATEGORIES)):
            with tab:
                st.caption(category.description)
                cols = st.columns(2)
                for qi, question in enumerate(category.questions):
                    if cols[qi % 2].button(question, key=f"example_{ci}_{qi}", width="stretch",
                                           disabled=processing):
                        _submit(question)
                        st.rerun()


def render_ai_assistant(base_df: pd.DataFrame) -> None:
    history = st.session_state["chat_history"]
    processing = st.session_state["active_query"] is not None

    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        section("🧠 Ask DataTalk.AI · live Databricks data")
    with right:
        if history and st.button("🗑️ New chat", width="stretch", key="new_chat"):
            st.session_state["chat_history"] = []
            st.session_state["dashboard_df"] = base_df
            st.rerun()

    if not history and not processing:
        st.markdown("<div class='dt-hero-chat'><div class='dt-orb'>🧠</div>"
                    "<div class='dt-welcome'>Welcome to DataTalk.AI</div>"
                    "<div style='font-weight:700;font-size:1.05rem;margin-top:2px'>Chat with your appeals data</div>"
                    "<div class='dt-muted' style='font-size:.85rem;margin-top:4px'>Access trusted business intelligence through conversational queries against governed data sources.</div></div>",
                    unsafe_allow_html=True)

    with st.container(border=True):
        with st.container(key="chat_input_wrap"):
            prompt = st.chat_input(
                "Ask a question about your appeals…",
                disabled=processing,
            )

        with st.container(key="mic_below_chat"):
            render_mic_recorder("voice_transcript")
        if prompt:
            _submit(prompt)
            st.rerun()

        if st.session_state.get("voice_transcript"):
            edited = st.text_area("🎙️ Speech converted to text — review and send",
                                  value=st.session_state["voice_transcript"], key="voice_transcript_edit", height=68)
            s1, s2, _ = st.columns([1, 1, 4])
            if s1.button("➤ Send", key="voice_send_btn", type="primary", width="stretch"):
                st.session_state["voice_transcript"] = ""
                _submit(edited)
                st.rerun()
            if s2.button("✖ Discard", key="voice_discard_btn", width="stretch"):
                st.session_state["voice_transcript"] = ""
                st.rerun()

        cols = st.columns(len(PRIMARY_ACTIONS))
        for col, (label, question) in zip(cols, PRIMARY_ACTIONS):
            if col.button(label, width="stretch", key=f"pill_{label}", disabled=processing):
                _submit(question)
                st.rerun()

    _render_example_questions(processing, expanded=not history and not processing)

    if processing:
        with st.chat_message("user", avatar="🙋"):
            st.markdown(st.session_state["active_query"])
        _process_active_query(base_df)

    done = history[:-1] if processing else history
    pairs = [(done[i], done[i + 1]) for i in range(0, len(done) - 1, 2)
             if done[i].get("role") == "user" and done[i + 1].get("role") == "assistant"]
    for idx in range(len(pairs) - 1, -1, -1):
        user_msg, ai_msg = pairs[idx]
        with st.chat_message("user", avatar="🙋"):
            st.markdown(user_msg.get("content", ""))
        with st.chat_message("assistant", avatar="🧠"):
            render_answer(ai_msg, key=f"hist_{idx}")

    with st.expander("🗂️ More quick actions", expanded=False):
        for title, buttons in MORE_ACTIONS:
            st.caption(title)
            cols = st.columns(len(buttons))
            for col, (label, question) in zip(cols, buttons):
                if col.button(label, width="stretch", key=f"more_{label}", disabled=processing):
                    _submit(question)
                    st.rerun()
