"""Determination & Sponsor Intelligence page (deterministic — no LLM calls).

Twelve filters drive summary tiles, the open-case detail, monthly closed-volume and
overturn trends, sponsor / provider-type / task distributions, determination mix,
routing reasons, month-over-month sponsor growth, reviewer segments and exports.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.ui import page_title, section
from core.plot_style import DETERMINATION_COLORS
from core.plot_style import PALETTE as CAT_SEQ
from core.plot_style import style_figure as S
from core.settings import AGING_THRESHOLD, ESCALATION_LEVEL, SLA_DAYS

PD_FILTER_KEYS = ["pd_myclosed", "pd_casetype", "pd_behalf", "pd_sponsor", "pd_aso",
                  "pd_source", "pd_determination", "pd_unit", "pd_openclosed",
                  "pd_level", "pd_site", "pd_caseid"]


def _pd_pct(numer, denom):
    return round(numer / denom * 100, 2) if denom else 0.0


def _pd_group_determination(d: pd.DataFrame, group_col: str, top_n: int = None) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(group_col).agg(
        TotalCases=("Case Identifier", "count"),
        Closed=("IsClosed", "sum"),
        Upheld=("IsUpheld", "sum"),
        Overturned=("IsOverturned", "sum"),
        PartialOverturn=("IsPartiallyOverturned", "sum"),
        Open=("IsClosed", lambda s: int((~s).sum())),
    ).reset_index()
    g["OverturnVolume"] = g["Overturned"] + g["PartialOverturn"]
    g["UpheldPct"] = g.apply(lambda r: _pd_pct(r["Upheld"], r["Closed"]), axis=1)
    g["OverturnPct"] = g.apply(lambda r: _pd_pct(r["OverturnVolume"], r["Closed"]), axis=1)
    g = g.sort_values("TotalCases", ascending=False)
    if top_n:
        g = g.head(top_n)
    return g


def render_intelligence_page(base_df: pd.DataFrame) -> None:
    for k in PD_FILTER_KEYS:
        if k not in st.session_state:
            st.session_state[k] = [] if k != "pd_openclosed" and k != "pd_caseid" else ("(All)" if k == "pd_openclosed" else "")

    page_title("📋", "Determination & Sponsor Intelligence",
               "Determination outcomes, sponsor and provider distributions, open-case detail and exports.")
    st.caption(
        "Quick Hits Dash — CPT Code / Provider TIN / Action Code fields are "
        "represented here by the new Provider ID / Provider Type / Provider Specialty "
        "dimensions, alongside Plan Sponsor, ASO Grouping, Work Item Task and Determination."
    )

    f1, f2, f3, f4, f5, f6 = st.columns(6)
    with f1:
        month_opts = sorted(
            base_df.loc[base_df["Closed Date (CaseHistory)"].notna(), "Closed Date (CaseHistory)"]
            .dt.to_period("M").astype(str).unique()
        )
        st.multiselect("📅 Month/Year Closed", month_opts, key="pd_myclosed")
    with f2:
        st.multiselect("📁 Case Type", sorted(base_df["CaseTypeLabel"].dropna().unique()), key="pd_casetype")
    with f3:
        st.multiselect("👤 Behalf Of", sorted(base_df["BehalfOfLabel"].dropna().unique()), key="pd_behalf")
    with f4:
        st.multiselect("🏢 Plan Sponsor", sorted(base_df["Plan Sponsor Name"].dropna().unique()), key="pd_sponsor")
    with f5:
        st.multiselect("🏷️ ASO Grouping", sorted(base_df["ASO Grouping"].dropna().unique()), key="pd_aso")
    with f6:
        st.multiselect("💱 Source", sorted(base_df["SourceLabel"].dropna().unique()), key="pd_source")

    f7, f8, f9, f10, f11, f12 = st.columns(6)
    with f7:
        st.multiselect("⚖️ Determination", sorted(base_df["DeterminationLabel"].dropna().unique()), key="pd_determination")
    with f8:
        st.multiselect("🏭 Current Unit", sorted(base_df["Current Unit Name"].dropna().unique()), key="pd_unit")
    with f9:
        st.selectbox("🔓 Open vs Closed", ["(All)", "Open", "Closed"], key="pd_openclosed")
    with f10:
        st.multiselect("🎚️ Level Type", sorted(base_df["Level Type Code"].dropna().unique()), key="pd_level")
    with f11:
        st.multiselect("🌍 Site", sorted(base_df["Routed To Site Name"].dropna().unique()), key="pd_site")
    with f12:
        st.text_input("🔎 Case ID contains", key="pd_caseid")

    pdf = base_df.copy()
    if st.session_state["pd_myclosed"]:
        pdf = pdf[pdf["Closed Date (CaseHistory)"].dt.to_period("M").astype(str).isin(st.session_state["pd_myclosed"])]
    if st.session_state["pd_casetype"]:
        pdf = pdf[pdf["CaseTypeLabel"].isin(st.session_state["pd_casetype"])]
    if st.session_state["pd_behalf"]:
        pdf = pdf[pdf["BehalfOfLabel"].isin(st.session_state["pd_behalf"])]
    if st.session_state["pd_sponsor"]:
        pdf = pdf[pdf["Plan Sponsor Name"].isin(st.session_state["pd_sponsor"])]
    if st.session_state["pd_aso"]:
        pdf = pdf[pdf["ASO Grouping"].isin(st.session_state["pd_aso"])]
    if st.session_state["pd_source"]:
        pdf = pdf[pdf["SourceLabel"].isin(st.session_state["pd_source"])]
    if st.session_state["pd_determination"]:
        pdf = pdf[pdf["DeterminationLabel"].isin(st.session_state["pd_determination"])]
    if st.session_state["pd_unit"]:
        pdf = pdf[pdf["Current Unit Name"].isin(st.session_state["pd_unit"])]
    if st.session_state["pd_openclosed"] == "Open":
        pdf = pdf[~pdf["IsClosed"]]
    if st.session_state["pd_openclosed"] == "Closed":
        pdf = pdf[pdf["IsClosed"]]
    if st.session_state["pd_level"]:
        pdf = pdf[pdf["Level Type Code"].isin(st.session_state["pd_level"])]
    if st.session_state["pd_site"]:
        pdf = pdf[pdf["Routed To Site Name"].isin(st.session_state["pd_site"])]
    if st.session_state["pd_caseid"]:
        pdf = pdf[pdf["Case Identifier"].astype(str).str.contains(st.session_state["pd_caseid"], case=False, na=False)]

    if pdf.empty:
        st.warning("⚠️ No data for the selected filters.")
        return

    closed_n = int(pdf["IsClosed"].sum())
    overturn_n = int(pdf["IsAnyOverturn"].sum())
    overturn_pct = _pd_pct(overturn_n, closed_n)
    s1, s2, s3, s4 = st.columns(4)
    for col, (lbl, val, icon) in zip([s1, s2, s3, s4], [
        ("Total Closed Cases", f"{closed_n:,}", "✅"),
        ("Overturn Volume", f"{overturn_n:,}", "🔀"),
        ("Overturn %", f"{overturn_pct}%", "🎯"),
        ("Total Cases (all)", f"{len(pdf):,}", "📁"),
    ]):
        col.markdown(f"""<div style="background:var(--dt-surface);
            border:1px solid var(--dt-border);border-top:3px solid var(--dt-accent);border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:.65rem;color:inherit;opacity:.72;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">{icon} {lbl}</div>
            <div style="font-size:1.5rem;font-weight:800;color:inherit;margin-top:4px;">{val}</div></div>""",
                    unsafe_allow_html=True)

    render_open_cases_section(pdf)

    st.markdown("<br>", unsafe_allow_html=True)

    section('📈 Monthly Trend — Closed Volume & Overturn %')
    trend_src = pdf[pdf["IsClosed"] & pdf["Closed Date (CaseHistory)"].notna()].copy()
    if not trend_src.empty:
        trend_src["ClosedMonth"] = trend_src["Closed Date (CaseHistory)"].dt.to_period("M").astype(str)
        tm = trend_src.groupby("ClosedMonth").agg(
            ClosedCases=("Case Identifier", "count"),
            Overturned=("IsAnyOverturn", "sum"),
        ).reset_index().sort_values("ClosedMonth")
        tm["OverturnPct"] = tm.apply(lambda r: _pd_pct(r["Overturned"], r["ClosedCases"]), axis=1)

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Bar(
            x=tm["ClosedMonth"], y=tm["ClosedCases"], name="Closed Cases",
            marker_color="#3b82f6", opacity=0.55,
        ), secondary_y=False)
        fig_trend.add_trace(go.Scatter(
            x=tm["ClosedMonth"], y=tm["OverturnPct"], name="Overturn %",
            mode="lines+markers", line=dict(color="#ef4444", width=3),
            marker=dict(size=8, symbol="diamond"),
        ), secondary_y=True)
        S(fig_trend, height=420, margin=dict(l=16, r=110, t=24, b=24))
        fig_trend.update_layout(hovermode="x unified",
                                legend=dict(orientation="v", x=1.08, y=1, xanchor="left", font=dict(size=11)))
        fig_trend.update_yaxes(title_text="Closed Cases", secondary_y=False)
        fig_trend.update_yaxes(title_text="Overturn %", secondary_y=True, showgrid=False, ticksuffix="%")
        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info("No closed-case history in the current filter to plot a trend.")

    st.markdown("<br>", unsafe_allow_html=True)

    section('🏢 Plan Sponsor Volume Distribution')
    sp = _pd_group_determination(pdf, "Plan Sponsor Name", top_n=12)
    if not sp.empty:
        disp = sp.rename(columns={
            "Plan Sponsor Name": "Plan Sponsor", "TotalCases": "Total Cases",
            "Closed": "Closed Cases", "Upheld": "Upheld", "UpheldPct": "Upheld %",
            "OverturnVolume": "Overturn Volume", "OverturnPct": "Overturn %", "Open": "Open Volume",
        })
        st.dataframe(
            disp[["Plan Sponsor", "Total Cases", "Closed Cases", "Upheld", "Upheld %",
                  "Overturn Volume", "Overturn %", "Open Volume"]],
            width="stretch", hide_index=True,
            column_config={
                "Upheld %": st.column_config.NumberColumn(format="%.2f%%"),
                "Overturn %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section('🩺 Provider Type Volume Distribution')
    pv = _pd_group_determination(pdf, "Provider Type", top_n=12)
    if not pv.empty:
        dispp = pv.rename(columns={
            "Provider Type": "Provider Type", "TotalCases": "Total Cases",
            "Closed": "Closed Cases", "Upheld": "Upheld", "UpheldPct": "Upheld %",
            "OverturnVolume": "Overturn Volume", "OverturnPct": "Overturn %", "Open": "Open Volume",
        })
        st.dataframe(
            dispp[["Provider Type", "Total Cases", "Closed Cases", "Upheld", "Upheld %",
                   "Overturn Volume", "Overturn %", "Open Volume"]],
            width="stretch", hide_index=True,
            column_config={
                "Upheld %": st.column_config.NumberColumn(format="%.2f%%"),
                "Overturn %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        section('📦 Task Type Volume Distribution')
        tk = _pd_group_determination(pdf, "Work Item Task Short Name")
        if not tk.empty:
            fig = go.Figure(go.Bar(
                x=tk["Work Item Task Short Name"], y=tk["OverturnPct"],
                marker_color=[DETERMINATION_COLORS.get("OVERTURNED", "#ef4444")] * len(tk),
                text=[f"{v:.1f}%" for v in tk["OverturnPct"]], textposition="outside",
            ))
            fig = S(fig, height=380)
            fig.update_layout(title="Overturn % by Task Type", showlegend=False, xaxis=dict(tickangle=-15))
            st.plotly_chart(fig, width="stretch")

    with c2:
        section('⚖️ Determination Mix')
        dmix = pdf[pdf["IsClosed"]].groupby("DeterminationLabel")["Case Identifier"].count().reset_index()
        dmix.columns = ["Determination", "Count"]
        if not dmix.empty:
            fig = go.Figure(go.Pie(
                labels=dmix["Determination"], values=dmix["Count"], hole=0.5,
                marker=dict(colors=[DETERMINATION_COLORS.get(d, "#3b82f6") for d in dmix["Determination"]]),
            ))
            fig = S(fig, height=380)
            fig.update_layout(title="Closed-Case Determination Mix", legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    section('📋 Appeal & Complaint Reasons — by ASO Grouping')
    r1, r2 = st.columns(2)
    with r1:
        rz = pdf.groupby(["ASO Grouping", "Work Item Task Short Name"])["Case Identifier"].count().reset_index()
        rz.columns = ["ASO Grouping", "Task", "Count"]
        if not rz.empty:
            fig = px.bar(rz, x="Count", y="Task", color="ASO Grouping", orientation="h",
                         color_discrete_sequence=CAT_SEQ)
            fig = S(fig, height=420, margin=dict(l=16, r=120, t=24, b=24))
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                               legend=dict(orientation="v", x=1.02, y=1, xanchor="left", font=dict(size=11)))
            st.plotly_chart(fig, width="stretch")
    with r2:
        rr = pdf.groupby("Routing Comments Text")["Case Identifier"].count().reset_index()
        rr.columns = ["Routing Reason", "Count"]
        if not rr.empty:
            fig = px.pie(rr, names="Routing Reason", values="Count", hole=0.45,
                         color_discrete_sequence=CAT_SEQ, title="Routing Reason Distribution")
            fig = S(fig, height=420)
            fig.update_layout(legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    section('🧭 Routing Reason by Current Unit')
    ra = pdf.groupby(["Routing Comments Text", "Current Unit Name"])["Case Identifier"].count().reset_index()
    ra.columns = ["Routing Reason", "Current Unit", "Case Count"]
    ra = ra.sort_values("Case Count", ascending=False)
    st.dataframe(ra, width="stretch", hide_index=True, height=320)

    st.markdown("<br>", unsafe_allow_html=True)

    section("📈 Top Plan Sponsors — Largest % Increase in Closed Volume (Month over Month)")
    cd = pdf[pdf["IsClosed"] & pdf["Closed Date (CaseHistory)"].notna()].copy()
    if not cd.empty:
        cd["ClosedMonth"] = cd["Closed Date (CaseHistory)"].dt.to_period("M").astype(str)
        months = sorted(cd["ClosedMonth"].unique())
        if len(months) >= 2:
            piv = cd.groupby(["Plan Sponsor Name", "ClosedMonth"])["Case Identifier"].count().unstack(fill_value=0)
            prev_m, last_m = months[-2], months[-1]
            if prev_m in piv.columns and last_m in piv.columns:
                inc = piv[[prev_m, last_m]].copy()
                inc.columns = ["Prev Month", "Last Month"]
                inc = inc[inc["Prev Month"] > 0]
                inc["% Increase"] = ((inc["Last Month"] - inc["Prev Month"]) / inc["Prev Month"] * 100).round(2)
                inc = inc.sort_values("% Increase", ascending=False).head(10).reset_index()
                inc = inc.rename(columns={"Plan Sponsor Name": "Plan Sponsor"})
                st.dataframe(inc, width="stretch", hide_index=True,
                             column_config={"% Increase": st.column_config.NumberColumn(format="%.2f%%")})
                st.caption(f"Comparing **{prev_m}** → **{last_m}** — most recent two months with closed volume in the filtered data.")
            else:
                st.info("Not enough overlapping closed-volume months to compute a period-over-period increase.")
        else:
            st.info("Need at least 2 months of closed-case data for a period-over-period comparison.")

    st.markdown("<br>", unsafe_allow_html=True)

    section('🏭 Reviewer Segment Distribution')
    seg = pdf.groupby("Routed To User Segment Description")["Case Identifier"].count().reset_index()
    seg.columns = ["Segment", "Case Count"]
    seg["% of Total"] = (seg["Case Count"] / seg["Case Count"].sum() * 100).round(2)
    seg = seg.sort_values("Case Count", ascending=False)
    st.dataframe(seg, width="stretch", hide_index=True,
                 column_config={"% of Total": st.column_config.NumberColumn(format="%.2f%%")})

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        section('📅 Counts By Received Date')
        rcv = pdf.copy()
        rcv["Year"] = rcv["Received Date"].dt.year
        rcv["Month"] = rcv["Received Date"].dt.strftime("%B")
        rc = rcv.groupby(["Year", "Month"]).agg(
            TotalClosedCases=("IsClosed", "sum"), OverturnVolume=("IsAnyOverturn", "sum")
        ).reset_index()
        rc["Overturn %"] = rc.apply(lambda r: _pd_pct(r["OverturnVolume"], r["TotalClosedCases"]), axis=1)
        rc = rc.rename(columns={"TotalClosedCases": "Total Closed Cases", "OverturnVolume": "Overturn Volume"})
        st.dataframe(rc.sort_values(["Year", "Month"], ascending=False), width="stretch",
                     hide_index=True, height=300,
                     column_config={"Overturn %": st.column_config.NumberColumn(format="%.2f%%")})
    with d2:
        section('📅 Counts By Closed Date')
        cls = pdf[pdf["Closed Date (CaseHistory)"].notna()].copy()
        cls["Year"] = cls["Closed Date (CaseHistory)"].dt.year
        cls["Month"] = cls["Closed Date (CaseHistory)"].dt.strftime("%B")
        cc = cls.groupby(["Year", "Month"]).agg(
            TotalClosedCases=("IsClosed", "sum"), OverturnVolume=("IsAnyOverturn", "sum")
        ).reset_index()
        cc["Overturn %"] = cc.apply(lambda r: _pd_pct(r["OverturnVolume"], r["TotalClosedCases"]), axis=1)
        cc = cc.rename(columns={"TotalClosedCases": "Total Closed Cases", "OverturnVolume": "Overturn Volume"})
        st.dataframe(cc.sort_values(["Year", "Month"], ascending=False), width="stretch",
                     hide_index=True, height=300,
                     column_config={"Overturn %": st.column_config.NumberColumn(format="%.2f%%")})

    st.markdown("<br>", unsafe_allow_html=True)



    section('📥 Export Current View')
    csv_bytes = pdf.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Data (CSV)", data=csv_bytes,
                        file_name="EXL_Determination_Intelligence_Export.csv", mime="text/csv",
                        width="content", type="primary")


def render_open_cases_section(pdf: pd.DataFrame):
    st.markdown("<br>", unsafe_allow_html=True)
    open_df = pdf[~pdf["IsClosed"]].copy()
    open_count = len(open_df)

    st.markdown(f"""<div style="background:var(--dt-surface);
        border:1px solid rgba(245,158,11,.35);border-radius:12px;padding:16px;
        text-align:center;max-width:260px;">
        <div style="font-size:.65rem;color:inherit;opacity:.72;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">
            🔓 Open Cases</div>
        <div style="font-size:1.8rem;font-weight:800;color:#d97706;margin-top:4px;">{open_count:,}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section('📂 Open Cases — Detail')

    if open_df.empty:
        st.success("✅ No open cases in the current filter.")
        return

    def _reason(r):
        reasons = []
        if r["IsEscalated"]:   reasons.append(f"Escalated (Level {ESCALATION_LEVEL})")
        if r["IsOverdue"]:     reasons.append(f"Past {SLA_DAYS}-day SLA")
        if r["IsTransferred"]: reasons.append("Recently transferred")
        return "; ".join(reasons) if reasons else "In normal processing"

    def _priority(days):
        if days > AGING_THRESHOLD:      return "🔴 Critical"
        if days > SLA_DAYS:              return "🟠 High"
        if days > SLA_DAYS / 2:          return "🟡 Medium"
        return "🟢 Low"

    open_df["Reason Open"]     = open_df.apply(_reason, axis=1)
    open_df["Priority (Age-Based)"] = open_df["CaseAgeDays"].apply(_priority)
    open_df["Pending Action"]  = open_df["Work Item Task Short Name"].fillna("Unassigned")
    open_df["Assigned Team"]   = open_df["Current Unit Name"]
    open_df["Assigned Owner"]  = open_df.get("Current User Name", "Unassigned")
    open_df["Created Date"]    = open_df["Received Date"].dt.date
    open_df["Last Updated"]    = (open_df["Routed Timestamp"].dt.date
                                   if "Routed Timestamp" in open_df.columns else pd.NaT)
    open_df["SLA Target Date"] = (open_df["Received Date"] + pd.Timedelta(days=SLA_DAYS)).dt.date

    disp_cols = ["Case Identifier", "Work Item Status Short Name", "Reason Open", "Pending Action",
                 "Assigned Team", "Assigned Owner", "Priority (Age-Based)", "Created Date",
                 "Last Updated", "SLA Target Date", "CaseAgeDays", "ComplianceRiskScore",
                 "Plan Sponsor Name"]
    disp = open_df[disp_cols].rename(columns={
        "Case Identifier": "Case ID",
        "Work Item Status Short Name": "Current Status",
        "CaseAgeDays": "Age (Days)",
        "ComplianceRiskScore": "Risk Score",
        "Plan Sponsor Name": "Plan Sponsor",
    }).sort_values("Age (Days)", ascending=False)

    st.dataframe(
        disp, width="stretch", hide_index=True, height=420,
        column_config={"Risk Score": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)},
    )

    csv_bytes = disp.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Open Cases (CSV)", data=csv_bytes,
                        file_name="EXL_Open_Cases_Export.csv", mime="text/csv")


