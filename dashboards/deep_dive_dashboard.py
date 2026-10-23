"""Deep Dive Analytics page (deterministic — no LLM calls).

All KPIs, charts, tables and exports derive from one filtered dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.plot_style import PALETTE, PLOTLY_CONFIG, style_figure
from core.settings import SLA_DAYS

TOP_N = 12
H = 420
COLORS = PALETTE
STATUS_COLORS = {"CLOSED":"#10b981", "OPEN":"#f59e0b", "PENDING":"#f59e0b", "TRANSFERRED":"#ef4444"}
RISK_COLORS = {"Very Low":"#10b981", "Low":"#3b82f6", "Medium":"#f59e0b", "High":"#ef4444", "Critical":"#991b1b"}

DIMENSION_CONFIG = {
    "Current Unit Name": "Operational Unit",
    "Routed To Site Name": "Site",
    "Work Item Task Short Name": "Task Type",
    "Plan Sponsor Name": "Plan Sponsor",
    "ASO Grouping": "ASO Segment",
    "BehalfOfLabel": "Filer Type",
    "Routed To User Segment Description": "Reviewer Segment",
    "Work Item Status Short Name": "Workflow Status",
    "Provider Type": "Provider Type",
    "Provider Specialty": "Provider Specialty",
    "Network Status": "Network Status",
    "DeterminationLabel": "Determination",
}
FILTERS = {
    "dd_site": ("Routed To Site Name", "Site"),
    "dd_unit": ("Current Unit Name", "Unit"),
    "dd_task": ("Work Item Task Short Name", "Task Type"),
    "dd_segment": ("ASO Grouping", "ASO Segment"),
    "dd_sponsor": ("Plan Sponsor Name", "Plan Sponsor"),
    "dd_ptype": ("Provider Type", "Provider Type"),
    "dd_network": ("Network Status", "Network Status"),
    "dd_filer": ("BehalfOfLabel", "Filer Type"),
    "dd_status": ("CaseStatusLabel", "Case Status"),
    "dd_det": ("DeterminationLabel", "Determination"),
}
REQUIRED = {"Case Identifier", "Received Date", "CaseAgeDays", "IsClosed", "IsOverdue", "IsEscalated", "IsCompliant", "ComplianceRiskScore"}
PLOTLY_CFG = PLOTLY_CONFIG

@dataclass(frozen=True)
class KPIs:
    total: int
    closed: int
    open: int
    overdue: int
    escalated: int
    compliance: float
    tat: float
    overturn: float
    risk: float


def _label(c): return DIMENSION_CONFIG.get(c, str(c).replace("_", " "))
def _rate(n, d): return round(n / d * 100, 1) if d else 0.0

def _fig(fig, height=H):
    return style_figure(fig, height, dict(l=20, r=20, t=65, b=55))


def _prepare(base_df):
    if base_df is None or base_df.empty: return pd.DataFrame()
    d = base_df.copy()
    missing = sorted(REQUIRED - set(d.columns))
    if missing: raise ValueError("Deep Dive missing required columns: " + ", ".join(missing))
    d["Received Date"] = pd.to_datetime(d["Received Date"], errors="coerce")
    if "Closed Date (CaseHistory)" in d: d["Closed Date (CaseHistory)"] = pd.to_datetime(d["Closed Date (CaseHistory)"], errors="coerce")
    if "ReceivedMonth" not in d: d["ReceivedMonth"] = d["Received Date"].dt.to_period("M").astype(str)
    if "ReceivedYear" not in d: d["ReceivedYear"] = d["Received Date"].dt.year
    if "Quarter" not in d: d["Quarter"] = d["Received Date"].dt.to_period("Q").astype(str)
    if "CaseStatusLabel" not in d:
        d["CaseStatusLabel"] = d["Case Status"].astype(str).str.strip().str.upper() if "Case Status" in d else np.where(d["IsClosed"], "CLOSED", "OPEN")
    if "DeterminationLabel" not in d:
        d["DeterminationLabel"] = d["Determination"].fillna("Pending Decision").astype(str).str.strip().str.upper() if "Determination" in d else "Pending Decision"
    if "IsAnyOverturn" not in d:
        det = d["Determination"].astype(str).str.strip().str.upper() if "Determination" in d else pd.Series("", index=d.index)
        d["IsAnyOverturn"] = det.isin(["OVERTURNED", "PARTIALLY OVERTURNED"])
    for c in ["IsClosed","IsOverdue","IsEscalated","IsCompliant","IsAnyOverturn"]: d[c] = d[c].fillna(False).astype(bool)
    d["CaseAgeDays"] = pd.to_numeric(d["CaseAgeDays"], errors="coerce").fillna(0).clip(lower=0)
    d["ComplianceRiskScore"] = pd.to_numeric(d["ComplianceRiskScore"], errors="coerce")
    return d


def _init_state():
    defaults = {"dd_dimension":"Provider Type", "dd_dimension2":"Work Item Task Short Name", "dd_compare":False, "dd_drill_dim":None, "dd_drill_value":None}
    defaults.update({k:[] for k in FILTERS})
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


def _selected(df, exclude=None):
    out = df
    for key,(col,_) in FILTERS.items():
        if key == exclude or col not in out: continue
        vals = st.session_state.get(key, [])
        if vals: out = out[out[col].isin(vals)]
    return out


def _options(df, key):
    col,_ = FILTERS[key]
    if col not in df: return []
    vals = _selected(df, exclude=key)[col].dropna().unique().tolist()
    cur = st.session_state.get(key, [])
    vals.extend(v for v in cur if v not in vals)
    return sorted(vals, key=str)


def _reset():
    for k in FILTERS: st.session_state[k] = []
    st.session_state["dd_drill_dim"] = None
    st.session_state["dd_drill_value"] = None


def _filters(df):
    valid = df["Received Date"].dropna()
    lo, hi = valid.min().date(), valid.max().date()
    with st.container(border=True):
        h1,h2 = st.columns([6,1])
        h1.markdown("**Filters & Scope**")
        if h2.button("Reset", key="dd_reset", width="stretch"):
            _reset(); st.session_state.pop("dd_dates", None); st.rerun()
        for cols, keys in [(st.columns(5), list(FILTERS)[:5]), (st.columns(5), list(FILTERS)[5:])]:
            for ui,key in zip(cols,keys):
                col,label = FILTERS[key]
                with ui:
                    st.multiselect(label, _options(df,key) if col in df else [], key=key, disabled=col not in df)
        d1,d2 = st.columns([3,1])
        with d1: dates = st.date_input("Received Date", value=st.session_state.get("dd_dates", (lo,hi)), min_value=lo, max_value=hi, key="dd_dates")
        with d2: st.toggle("Compare periods", key="dd_compare")
    out = _selected(df)
    # Handle Streamlit date_input return types safely
    if isinstance(dates, (tuple, list)):
        if len(dates) == 2:
            a, b = dates[0], dates[1]
        elif len(dates) == 1:
            a = b = dates[0]
        else:
            a, b = lo, hi
    else:
        a = b = dates

    # Apply date filter
    out = out[
        out["Received Date"].notna()
        & (out["Received Date"].dt.date >= a)
        & (out["Received Date"].dt.date <= b)
    ]
    return out.copy(), a, b


def _kpis(df):
    if df.empty: return KPIs(0,0,0,0,0,0,0,0,0)
    total=len(df); closed=int(df["IsClosed"].sum()); opened=total-closed
    cc=int((df["IsClosed"] & df["IsCompliant"]).sum())
    ov=int((df["IsClosed"] & df["IsAnyOverturn"]).sum())
    tat=float(df.loc[df["IsClosed"],"CaseAgeDays"].mean()) if closed else 0
    risk=float(df["ComplianceRiskScore"].mean()) if df["ComplianceRiskScore"].notna().any() else 0
    return KPIs(total,closed,opened,int(df["IsOverdue"].sum()),int(df["IsEscalated"].sum()),_rate(cc,closed),round(tat,1),_rate(ov,closed),round(risk,3))


def _periods(df):
    if not st.session_state.get("dd_compare") or df.empty: return df,None,""
    lo,hi=df["Received Date"].min(),df["Received Date"].max()
    if pd.isna(lo) or pd.isna(hi) or lo>=hi: return df,None,"Comparison unavailable."
    mid=lo+(hi-lo)/2
    return df[df["Received Date"]>=mid].copy(), df[df["Received Date"]<mid].copy(), f"Current: {mid.date()} to {hi.date()} | Prior: {lo.date()} to {mid.date()}"


def _delta(cur, prev, lower=False, pp=False):
    if prev is None: return "Current selection", "neutral"
    d=cur-prev; good=d<=0 if lower else d>=0
    return (f"{d:+.1f} pts" if pp else f"{d:+,.1f}"), ("good" if good else "bad")


def _render_kpis(cur_df, prior_df):
    c=_kpis(cur_df); p=_kpis(prior_df) if prior_df is not None and not prior_df.empty else None
    cards=[
        ("Total Appeals",f"{c.total:,}",c.total,p.total if p else None,False,False,"#3b82f6"),
        ("Closed",f"{c.closed:,}",c.closed,p.closed if p else None,False,False,"#10b981"),
        ("Open",f"{c.open:,}",c.open,p.open if p else None,True,False,"#f59e0b"),
        ("Compliance",f"{c.compliance:.1f}%",c.compliance,p.compliance if p else None,False,True,"#10b981"),
        ("Overdue",f"{c.overdue:,}",c.overdue,p.overdue if p else None,True,False,"#ef4444"),
        ("Avg TAT",f"{c.tat:.1f}d",c.tat,p.tat if p else None,True,False,"#06b6d4"),
        ("Overturn",f"{c.overturn:.1f}%",c.overturn,p.overturn if p else None,True,True,"#8b5cf6"),
        ("Avg Risk",f"{c.risk:.3f}",c.risk,p.risk if p else None,True,False,"#f59e0b")]
    for ui,(label,val,raw,prev,lower,pp,color) in zip(st.columns(8),cards):
        dt,cls=_delta(raw,prev,lower,pp)
        ui.markdown(f'<div class="dd-kpi" style="--a:{color}"><div class="dd-kl">{label}</div><div class="dd-kv">{val}</div><div class="dd-kd {cls}">{dt}</div></div>',unsafe_allow_html=True)


def _score(df, dim):
    g=df.groupby(dim,dropna=False).agg(Cases=("Case Identifier","count"),Closed=("IsClosed","sum"),Overdue=("IsOverdue","sum"),Escalated=("IsEscalated","sum"),AvgRisk=("ComplianceRiskScore","mean"),AvgAge=("CaseAgeDays","mean")).reset_index()
    compliant=(df.assign(_cc=df["IsClosed"] & df["IsCompliant"]).groupby(dim,dropna=False)["_cc"].sum().rename("Compliant"))
    overturn=(df.assign(_ov=df["IsClosed"] & df["IsAnyOverturn"]).groupby(dim,dropna=False)["_ov"].sum().rename("Overturns"))
    g=g.merge(compliant,on=dim,how="left").merge(overturn,on=dim,how="left")
    g["Open"]=g["Cases"]-g["Closed"]
    g["Compliance %"]=np.where(g["Closed"]>0,g["Compliant"]/g["Closed"]*100,np.nan)
    g["Overturn %"]=np.where(g["Closed"]>0,g["Overturns"]/g["Closed"]*100,np.nan)
    g["Overdue %"]=np.where(g["Cases"]>0,g["Overdue"]/g["Cases"]*100,0)
    g["Escalated %"]=np.where(g["Cases"]>0,g["Escalated"]/g["Cases"]*100,0)
    for c in ["Compliance %","Overturn %","Overdue %","Escalated %"]: g[c]=g[c].round(1)
    g["AvgRisk"]=g["AvgRisk"].round(3); g["AvgAge"]=g["AvgAge"].round(1)
    return g.sort_values("Cases",ascending=False)


def _performance(df,dim):
    st.markdown(f'<div class="dd-sec">{_label(dim)} Performance</div>',unsafe_allow_html=True)
    s=_score(df,dim)
    l,r=st.columns([1.35,1])
    with l:
        v=s.head(TOP_N).sort_values("Cases")
        fig=px.bar(v,x="Cases",y=dim,orientation="h",color="Cases",text="Cases",color_continuous_scale=["#93c5fd","#3b82f6","#1e3a8a"],title=f"Appeal Volume by {_label(dim)}",custom_data=["Closed","Open","Overdue","Compliance %"])
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Appeals: %{x:,}<br>Closed: %{customdata[0]:,}<br>Open: %{customdata[1]:,}<br>Overdue: %{customdata[2]:,}<br>Compliance: %{customdata[3]:.1f}%<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        ev=st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG,key="dd_vol",on_select="rerun",selection_mode="points")
        if ev and ev.get("selection",{}).get("points"):
            clicked=ev["selection"]["points"][0].get("y")
            if clicked: st.session_state["dd_drill_dim"]=dim; st.session_state["dd_drill_value"]=clicked; st.rerun()
    with r:
        v=s.head(TOP_N).sort_values("AvgRisk")
        fig=px.bar(v,x="AvgRisk",y=dim,orientation="h",color="AvgRisk",text="AvgRisk",color_continuous_scale=["#10b981","#f59e0b","#ef4444"],title="Average Compliance Risk")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)
    display=s.rename(columns={dim:_label(dim),"AvgRisk":"Avg Risk","AvgAge":"Avg Age"})
    st.dataframe(display[[_label(dim),"Cases","Closed","Open","Overdue","Compliance %","Overturn %","Escalated %","Avg Risk","Avg Age"]],width="stretch",hide_index=True,height=330,column_config={"Compliance %":st.column_config.NumberColumn(format="%.1f%%"),"Overturn %":st.column_config.NumberColumn(format="%.1f%%"),"Avg Risk":st.column_config.ProgressColumn(format="%.3f",min_value=0,max_value=1)})


def _trends(df,dim):
    st.markdown('<div class="dd-sec">Temporal Trends</div>',unsafe_allow_html=True)
    l,r=st.columns(2)
    with l:
        top=df[dim].value_counts().head(6).index; src=df[df[dim].isin(top)]
        m=src.groupby(["ReceivedMonth",dim])["Case Identifier"].count().reset_index(name="Appeals").sort_values("ReceivedMonth")
        fig=px.line(m,x="ReceivedMonth",y="Appeals",color=dim,markers=True,color_discrete_sequence=COLORS,title=f"Monthly Volume | Top {_label(dim)}")
        fig.update_layout(hovermode="x unified",legend=dict(orientation="h",y=1.13)); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)
    with r:
        m=df.groupby("ReceivedMonth").agg(Cases=("Case Identifier","count"),Overdue=("IsOverdue","sum"),Closed=("IsClosed","sum")).reset_index().sort_values("ReceivedMonth")
        cc=df.assign(_cc=df["IsClosed"] & df["IsCompliant"]).groupby("ReceivedMonth")["_cc"].sum(); m=m.merge(cc.rename("Compliant"),on="ReceivedMonth",how="left")
        m["Compliance %"]=np.where(m["Closed"]>0,m["Compliant"]/m["Closed"]*100,np.nan); m["Overdue %"]=m["Overdue"]/m["Cases"]*100
        fig=make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=m["ReceivedMonth"],y=m["Cases"],name="Appeals",marker_color="#3b82f6",opacity=.5),secondary_y=False)
        fig.add_trace(go.Scatter(x=m["ReceivedMonth"],y=m["Compliance %"],name="Compliance %",mode="lines+markers",line=dict(color="#10b981",width=3)),secondary_y=True)
        fig.add_trace(go.Scatter(x=m["ReceivedMonth"],y=m["Overdue %"],name="Overdue %",mode="lines+markers",line=dict(color="#ef4444",dash="dot")),secondary_y=True)
        fig.update_yaxes(title_text="Appeals",secondary_y=False); fig.update_yaxes(title_text="Rate",ticksuffix="%",secondary_y=True); fig.update_layout(title="Volume, Compliance & Overdue",hovermode="x unified",legend=dict(orientation="h",y=1.13)); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)


def _cross(df, p, s):
    st.markdown(
        f'<div class="dd-sec">'
        f'Cross-Dimensional Intelligence | '
        f'{_label(p)} × {_label(s)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    l, r = st.columns([1.2, 1])

    # ============================================================
    # LEFT: PRIMARY × SECONDARY HEATMAP
    # ============================================================
    with l:

        # Safety check:
        # primary and secondary should normally be different,
        # but protect the dashboard anyway.
        if p == s:
            st.info(
                "Select a different 'Break Down By' dimension "
                "to display the cross-dimensional heatmap."
            )

        elif p not in df.columns or s not in df.columns:
            st.info(
                "The selected dimensions are not available "
                "for the current dataset."
            )

        else:
            top_p = (
                df[p]
                .dropna()
                .value_counts()
                .head(TOP_N)
                .index
            )

            top_s = (
                df[s]
                .dropna()
                .value_counts()
                .head(TOP_N)
                .index
            )

            w = df[
                df[p].isin(top_p)
                & df[s].isin(top_s)
            ].copy()

            if w.empty:

                st.info(
                    "No cross-dimensional records are available "
                    "for the current filters."
                )

            else:

                h = w.pivot_table(
                    index=p,
                    columns=s,
                    values="Case Identifier",
                    aggfunc="count",
                    fill_value=0,
                )

                if h.empty:

                    st.info(
                        "No cross-dimensional values are available."
                    )

                else:

                    fig = go.Figure(
                        go.Heatmap(
                            z=h.values,
                            x=h.columns.astype(str),
                            y=h.index.astype(str),
                            text=h.values,
                            texttemplate="%{text:.0f}",
                            colorscale=[
                                [0, "#dbeafe"],
                                [0.5, "#2563eb"],
                                [1, "#ef4444"],
                            ],
                            colorbar=dict(
                                title="Cases",
                                thickness=12,
                            ),
                            hovertemplate=(
                                f"{_label(p)}: %{{y}}<br>"
                                f"{_label(s)}: %{{x}}<br>"
                                "Cases: %{z:,}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig.update_layout(
                        title="Case Concentration Heatmap"
                    )

                    st.plotly_chart(
                        _fig(fig),
                        width="stretch",
                        config=PLOTLY_CFG,
                    )

    # ============================================================
    # RIGHT: STATUS MIX
    # ============================================================
    with r:

        preferred_status = "Work Item Status Short Name"

        if preferred_status in df.columns:
            status = preferred_status
        elif "CaseStatusLabel" in df.columns:
            status = "CaseStatusLabel"
        else:
            status = None

        # --------------------------------------------------------
        # CRITICAL FIX:
        # Do NOT group by the same column twice.
        # --------------------------------------------------------
        if status is None:

            st.info(
                "No supported status field is available "
                "for Status Mix."
            )

        elif p == status:

            # The primary dimension itself is already status.
            # Instead of doing:
            #
            # groupby([p, status])
            #
            # aggregate status directly.
            status_mix = (
                df.groupby(
                    p,
                    dropna=False,
                )["Case Identifier"]
                .count()
                .reset_index(name="Cases")
                .sort_values(
                    "Cases",
                    ascending=False,
                )
            )

            if status_mix.empty:

                st.info(
                    "No status records are available "
                    "for the current selection."
                )

            else:

                status_mix = status_mix.head(TOP_N)

                fig = px.bar(
                    status_mix,
                    x=p,
                    y="Cases",
                    color=p,
                    text="Cases",
                    color_discrete_map=STATUS_COLORS,
                    color_discrete_sequence=COLORS,
                    title="Appeal Volume by Workflow Status",
                )

                fig.update_traces(
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Cases: %{y:,}"
                        "<extra></extra>"
                    ),
                )

                fig.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-20,
                )

                st.plotly_chart(
                    _fig(fig),
                    width="stretch",
                    config=PLOTLY_CFG,
                )

        # --------------------------------------------------------
        # Normal case:
        #
        # Primary dimension != status dimension
        # --------------------------------------------------------
        else:

            if p not in df.columns:

                st.info(
                    "The selected primary dimension is "
                    "not available."
                )

            else:

                top = (
                    df[p]
                    .dropna()
                    .value_counts()
                    .head(10)
                    .index
                )

                status_source = df[
                    df[p].isin(top)
                ].copy()

                if status_source.empty:

                    st.info(
                        "No status records are available "
                        "for the current filters."
                    )

                else:

                    m = (
                        status_source
                        .groupby(
                            [p, status],
                            dropna=False,
                        )["Case Identifier"]
                        .count()
                        .rename("Cases")
                        .reset_index()
                    )

                    if m.empty:

                        st.info(
                            "No status breakdown is available "
                            "for the current selection."
                        )

                    else:

                        fig = px.bar(
                            m,
                            x=p,
                            y="Cases",
                            color=status,
                            barmode="stack",
                            color_discrete_map=STATUS_COLORS,
                            color_discrete_sequence=COLORS,
                            title=(
                                f"Status Mix by "
                                f"{_label(p)}"
                            ),
                            custom_data=[
                                status,
                            ],
                        )

                        fig.update_traces(
                            hovertemplate=(
                                f"<b>{_label(p)}: "
                                "%{x}</b><br>"
                                "Status: %{customdata[0]}<br>"
                                "Cases: %{y:,}"
                                "<extra></extra>"
                            )
                        )

                        fig.update_layout(
                            legend=dict(
                                orientation="h",
                                y=1.13,
                            ),
                            xaxis_tickangle=-20,
                        )

                        st.plotly_chart(
                            _fig(fig),
                            width="stretch",
                            config=PLOTLY_CFG,
                        )

def _risk(df,dim):
    st.markdown('<div class="dd-sec">Risk & Outcome Intelligence</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        x=df.copy(); x["Risk Band"]=pd.cut(x["ComplianceRiskScore"],[-np.inf,.2,.4,.6,.8,np.inf],labels=["Very Low","Low","Medium","High","Critical"]); g=x.groupby("Risk Band",observed=False)["Case Identifier"].count().reset_index(name="Cases")
        fig=px.bar(g,x="Risk Band",y="Cases",color="Risk Band",text="Cases",color_discrete_map=RISK_COLORS,title="Risk Band Distribution"); fig.update_layout(showlegend=False); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)
    with c2:
        x=df[df["IsClosed"]].groupby("DeterminationLabel")["Case Identifier"].count().reset_index(name="Cases")
        if not x.empty:
            fig=px.pie(x,names="DeterminationLabel",values="Cases",hole=.55,color_discrete_sequence=COLORS,title="Determination Mix"); fig.update_traces(textinfo="percent+label"); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)
    with c3:
        x=df[~df["IsClosed"]].copy()
        if not x.empty:
            x["Aging Band"]=pd.cut(x["CaseAgeDays"],[-np.inf,15,SLA_DAYS,60,np.inf],labels=["0-15d",f"16-{SLA_DAYS}d",f"{SLA_DAYS+1}-60d","60d+"]); g=x.groupby("Aging Band",observed=False)["Case Identifier"].count().reset_index(name="Cases")
            fig=px.bar(g,x="Aging Band",y="Cases",color="Aging Band",text="Cases",color_discrete_sequence=["#10b981","#3b82f6","#f59e0b","#ef4444"],title="Open Backlog Aging"); fig.update_layout(showlegend=False); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)
    s=_score(df,dim); l,r=st.columns(2)
    with l:
        v=s[s["Cases"]>=5].nlargest(10,"Overdue %").sort_values("Overdue %"); fig=px.bar(v,x="Overdue %",y=dim,orientation="h",color="Overdue %",text="Overdue %",color_continuous_scale=["#f59e0b","#ef4444"],title=f"Highest Overdue Rate | {_label(dim)}"); fig.update_traces(texttemplate="%{x:.1f}%"); fig.update_layout(coloraxis_showscale=False); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)
    with r:
        v=s[s["Closed"]>=5].nlargest(10,"Overturn %").sort_values("Overturn %"); fig=px.bar(v,x="Overturn %",y=dim,orientation="h",color="Overturn %",text="Overturn %",color_continuous_scale=["#8b5cf6","#ef4444"],title=f"Highest Overturn Rate | {_label(dim)}"); fig.update_traces(texttemplate="%{x:.1f}%"); fig.update_layout(coloraxis_showscale=False); st.plotly_chart(_fig(fig),width="stretch",config=PLOTLY_CFG)


def _leaderboard(df,dim):
    st.markdown(f'<div class="dd-sec">Performance Leaderboard | {_label(dim)}</div>',unsafe_allow_html=True)
    s=_score(df,dim); s=s[s["Cases"]>=5]
    if s.empty: st.info("No categories meet the 5-case minimum for rate leaderboards."); return
    for ui,title,col,fields in zip(st.columns(3),["Highest Compliance","Highest Overdue","Highest Overturn"],["Compliance %","Overdue %","Overturn %"],[[dim,"Cases","Compliance %"],[dim,"Cases","Overdue","Overdue %"],[dim,"Closed","Overturns","Overturn %"]]):
        with ui:
            st.markdown(f"##### {title}"); st.dataframe(s.dropna(subset=[col]).nlargest(5,col)[fields],hide_index=True,width="stretch",column_config={col:st.column_config.NumberColumn(format="%.1f%%")})


def _detail(df):
    st.markdown('<div class="dd-sec">Case Detail & Export</div>',unsafe_allow_html=True)
    q=st.text_input("Search current view",placeholder="Case ID, sponsor, provider or unit",key="dd_search")
    d=df.copy()
    if q.strip():
        search=[c for c in ["Case Identifier","Plan Sponsor Name","Provider Name","Current Unit Name"] if c in d]
        mask=pd.Series(False,index=d.index)
        for c in search: mask |= d[c].astype(str).str.contains(q.strip(),case=False,na=False,regex=False)
        d=d[mask]
    cols=[c for c in ["Case Identifier","Received Date","CaseStatusLabel","Plan Sponsor Name","Provider Name","Provider Type","Provider Specialty","Current Unit Name","Routed To Site Name","Work Item Task Short Name","DeterminationLabel","CaseAgeDays","IsOverdue","IsEscalated","ComplianceRiskScore"] if c in d]
    st.caption(f"{len(d):,} rows in current detail view")
    st.dataframe(d[cols].head(500),width="stretch",hide_index=True,height=420,column_config={"ComplianceRiskScore":st.column_config.ProgressColumn("Risk",format="%.3f",min_value=0,max_value=1),"CaseAgeDays":st.column_config.NumberColumn("Age (Days)",format="%d")})
    if len(d)>500: st.caption(f"Preview is limited to 500 rows; export contains all {len(d):,} matching rows.")
    st.download_button("Download Current View (CSV)",data=d.to_csv(index=False).encode("utf-8-sig"),file_name="Data_Deep_Dive_Current_View.csv",mime="text/csv",type="primary")


def _css():
    st.markdown('''<style>
.dd-hero{background:linear-gradient(135deg,rgba(37,99,235,.09),rgba(124,58,237,.05));border:1px solid var(--dt-border);
  border-radius:14px;padding:16px 20px;margin:6px 0 14px}
.dd-title{font-size:1.3rem;font-weight:800}.dd-sub{font-size:.82rem;opacity:.72;margin-top:4px}
.dd-sec{font-size:.7rem;letter-spacing:.11em;text-transform:uppercase;color:var(--dt-accent);font-weight:800;
  border-left:3px solid var(--dt-accent);padding-left:9px;margin:20px 0 10px}
.dd-kpi{background:var(--dt-surface);border:1px solid var(--dt-border);border-top:3px solid var(--a);border-radius:12px;
  padding:12px 10px;min-height:105px}
.dd-kl{font-size:.64rem;opacity:.72;text-transform:uppercase;font-weight:700}
.dd-kv{font-size:1.3rem;font-weight:800;margin-top:5px}
.dd-kd{font-size:.66rem;margin-top:6px;font-weight:700}.dd-kd.good{color:#16a34a}.dd-kd.bad{color:#dc2626}
.dd-kd.neutral{opacity:.6}
</style>''', unsafe_allow_html=True)


def render_deep_dive_tab(base_df: pd.DataFrame):
    _css(); _init_state()
    try: source=_prepare(base_df)
    except ValueError as e: st.error(str(e)); return
    if source.empty or source["Received Date"].dropna().empty: st.warning("No data is available for Deep Dive Analytics."); return
    dims=[d for d in DIMENSION_CONFIG if d in source and source[d].notna().any()]
    if not dims: st.warning("No supported analytical dimensions are available."); return
    if st.session_state["dd_dimension"] not in dims: st.session_state["dd_dimension"]=dims[0]
    sec=[d for d in dims if d!=st.session_state["dd_dimension"]] or [dims[0]]
    if st.session_state["dd_dimension2"] not in sec: st.session_state["dd_dimension2"]=sec[0]
    a,b=st.columns(2)
    with a: st.selectbox("Analyze By",dims,format_func=_label,key="dd_dimension")
    sec=[d for d in dims if d!=st.session_state["dd_dimension"]] or [dims[0]]
    with b: st.selectbox("Break Down By",sec,format_func=_label,key="dd_dimension2")
    filtered,d0,d1=_filters(source)
    drill_dim=st.session_state.get("dd_drill_dim"); drill=st.session_state.get("dd_drill_value")
    if drill is not None and drill_dim in filtered:
        x,y=st.columns([5,1]); x.info(f"Cross-filter: {_label(drill_dim)} = {drill}")
        if y.button("Clear selection",key="dd_clear_drill",width="stretch"): st.session_state["dd_drill_dim"]=None; st.session_state["dd_drill_value"]=None; st.rerun()
        filtered=filtered[filtered[drill_dim]==drill]
    share=_rate(len(filtered),len(source))
    st.markdown(f'<div class="dd-hero"><div class="dd-title">Deep Dive Analytics</div><div class="dd-sub">Interactive operational intelligence | {len(filtered):,} of {len(source):,} cases ({share:.1f}%) | {d0} to {d1}</div></div>',unsafe_allow_html=True)
    if filtered.empty: st.warning("No records match the current filters."); return
    current,prior,caption=_periods(filtered)
    if current.empty: st.warning("Current comparison period contains no records."); return
    if caption: st.caption(caption)
    p=st.session_state["dd_dimension"]; s=st.session_state["dd_dimension2"]
    _render_kpis(current,prior); _performance(current,p); _trends(current,p); _cross(current,p,s); _risk(current,p); _leaderboard(current,p); _detail(current)
