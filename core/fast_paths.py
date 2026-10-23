"""Deterministic quick actions ("fast paths") — pandas only, zero LLM calls.

``match_fast_path`` decides whether a question is a quick action; ``run_fast_path``
computes the answer from the in-memory dataset. Matching rules (unchanged
behaviour):

1. two values of the same dimension + "compare / vs / versus / side by side",
2. provider overturn month-over-month questions,
3. the first ``FAST_QUERIES`` trigger contained in the question (substring, case-insensitive).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from anomaly_detection import AnomalyConfig, detect_anomalies
from core.data import calculate_kpis
from core.plot_style import (
    PALETTE,
    RISK_BAND_COLORS,
    SEQUENTIAL_BLUE,
    SEVERITY_COLORS,
    STATUS_COLORS,
    UNIT_COLORS,
    style_figure,
)
from core.schemas import AssistantAnswer
from core.settings import AGING_THRESHOLD, ESCALATION_LEVEL, SLA_DAYS

ANOMALY_CONFIG = AnomalyConfig(sla_days=SLA_DAYS, aging_threshold_days=AGING_THRESHOLD,
                               escalation_level=ESCALATION_LEVEL)

# Trigger phrase → handler. Evaluated in order; first substring match wins.
FAST_QUERIES: dict[str, str] = {
    "give me a summary": "summary",
    "top plan sponsors": "top_sponsors",
    "top sponsors": "top_sponsors",
    "task breakdown": "task_breakdown",
    "overdue appeals": "overdue",
    "escalated appeals": "escalated",
    "appeals by site": "by_region",
    "appeals by region": "by_region",
    "unit analysis": "unit",
    "monthly trend": "monthly_trend",
    "source breakdown": "source",
    "risk overview": "risk_overview",
    "pending appeals": "pending",
    "reviewer segment analysis": "reviewer_segment",
    "forecast": "forecast",
    "predict": "forecast",
    "future trend": "forecast",
    "anomaly": "anomaly_scan",
    "anomalies": "anomaly_scan",
    "detect anomalies": "anomaly_scan",
}
_COMPARE_RE = re.compile(r"\b(compare|vs|versus)\b|side by side", re.I)
COMPARE_DIMENSIONS = ("Routed To Site Name", "Current Unit Name", "Work Item Task Short Name",
                      "ASO Grouping", "Provider Type", "Network Status")


@dataclass
class FastPathMatch:
    name: str
    compare_column: str | None = None
    compare_values: tuple[str, str] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Matching
# ═══════════════════════════════════════════════════════════════════════════════
def is_provider_overturn_mom_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    provider = any(t in q for t in ("provider", "providers", "provider name"))
    overturn = any(t in q for t in ("overturn", "overturned", "overturn rate"))
    comparison = any(t in q for t in ("previous month", "prior month", "last month", "month over month",
                                      "month-over-month", "mom", "increase", "increased", "increasing"))
    multi_month = any(t in q for t in ("past 3 month", "past three month", "last 3 month", "last three month",
                                       "continuous", "continuously", "consecutive", "every month"))
    return provider and overturn and comparison and not multi_month


def match_fast_path(query: str, df: pd.DataFrame) -> FastPathMatch | None:
    q = (query or "").strip().lower()
    if _COMPARE_RE.search(q):
        for col in COMPARE_DIMENSIONS:
            if col not in df.columns:
                continue
            found = [v for v in df[col].dropna().unique().tolist() if str(v).lower() in q]
            if len(found) >= 2:
                return FastPathMatch("comparison", col, (found[0], found[1]))
    if is_provider_overturn_mom_query(query):
        return FastPathMatch("provider_mom")
    for trigger, name in FAST_QUERIES.items():
        if trigger in q:
            return FastPathMatch(name)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Insight helper for deterministic answers
# ═══════════════════════════════════════════════════════════════════════════════
def static_insight(risk: str, summary: str, impact: str, success: str, actions: list[str],
                   rows: int, reasoning: str = "", metrics: dict | None = None) -> dict:
    """Insight contract (rendered by the insight panel) for a deterministic answer."""
    urgent = risk in {"Critical", "High"}
    return {
        "insight": summary, "key_finding": {"text": summary, "refs": []},
        "detailed_analysis": [summary],
        "business_impact": impact,
        "reasoning": reasoning or "Calculated from the loaded appeals dataset using deterministic application rules.",
        "supporting_metrics": metrics or {"Rows Evaluated": f"{rows:,}"},
        "recommended_actions": [
            {"priority": "Immediate" if i == 0 and urgent else "Monitor", "action": a, "owner": "Business owner",
             "timeframe": "Immediate review" if i == 0 and urgent else "Next review cycle",
             "success_measure": success, "evidence_ids": []}
            for i, a in enumerate(actions[:4])
        ],
        "strategic_improvements": [], "data_caveats": [], "_sql": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Forecast
# ═══════════════════════════════════════════════════════════════════════════════
def build_forecast(df: pd.DataFrame, periods: int = 6) -> dict:
    """Linear (np.polyfit) monthly forecast for volume, overdue and non-compliant cases."""
    if df.empty or "ReceivedMonth" not in df.columns:
        return {}
    monthly = (df.groupby("ReceivedMonth")
               .agg(Volume=("Case Identifier", "count"), Overdue=("IsOverdue", "sum"),
                    NonCompliant=("IsCompliant", lambda x: int((~x).sum())))
               .reset_index().sort_values("ReceivedMonth"))
    if len(monthly) < 3:
        return {}
    t = np.arange(len(monthly))
    results: dict = {}
    for metric in ("Volume", "Overdue", "NonCompliant"):
        y = monthly[metric].to_numpy(dtype=float)
        coeffs = np.polyfit(t, y, 1)
        fitted = np.polyval(coeffs, t)
        residuals = y - fitted
        future = np.clip(np.polyval(coeffs, np.arange(len(monthly), len(monthly) + periods)), 0, None)
        last = pd.Period(monthly["ReceivedMonth"].iloc[-1], "M")
        results[metric] = {
            "historical": monthly[["ReceivedMonth", metric]].rename(columns={"ReceivedMonth": "Month", metric: "Value"}),
            "forecast_months": [(last + i + 1).strftime("%Y-%m") for i in range(periods)],
            "forecast_values": future.tolist(),
            "slope": float(coeffs[0]),
            "trend": "upward" if coeffs[0] > 0 else "downward",
            "volatility": round(float(np.std(residuals)), 2),
            "r2": round(1 - float(np.sum(residuals ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-9), 3),
        }
    return results


def _forecast_figure(fd: dict) -> go.Figure:
    hist = fd["historical"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Month"].tolist(), y=hist["Value"].tolist(), mode="lines+markers",
                             name="Historical", line=dict(color=PALETTE[0], width=2.5)))
    fig.add_trace(go.Scatter(x=fd["forecast_months"], y=fd["forecast_values"], mode="lines+markers",
                             name="Forecast", line=dict(color=PALETTE[1], width=2.5, dash="dot"),
                             marker=dict(size=8, symbol="diamond")))
    upper = [v + fd["volatility"] for v in fd["forecast_values"]]
    lower = [max(0, v - fd["volatility"]) for v in fd["forecast_values"]]
    fig.add_trace(go.Scatter(x=fd["forecast_months"] + fd["forecast_months"][::-1], y=upper + lower[::-1],
                             fill="toself", fillcolor="rgba(16,185,129,0.12)", line=dict(color="rgba(0,0,0,0)"),
                             name="±1σ band"))
    fig.update_layout(title="Appeal Volume Forecast — Next 6 Months", legend=dict(orientation="h", y=1.1))
    return style_figure(fig, 460)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider overturn month-over-month
# ═══════════════════════════════════════════════════════════════════════════════
def build_provider_overturn_mom(df: pd.DataFrame, minimum_closed_cases: int = 1):
    """Providers whose overturn rate rose between the latest two closed months."""
    required = ["Provider Name", "Closed Date (CaseHistory)", "IsClosed", "IsAnyOverturn", "Case Identifier"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    w = df.copy()
    w["Closed Date (CaseHistory)"] = pd.to_datetime(w["Closed Date (CaseHistory)"], errors="coerce")
    w = w[w["IsClosed"] & w["Closed Date (CaseHistory)"].notna() & w["Provider Name"].notna()]
    w = w[w["Provider Name"].astype(str).str.strip().ne("")].copy()
    if w.empty:
        return pd.DataFrame(), None, None
    w["ClosedMonth"] = w["Closed Date (CaseHistory)"].dt.to_period("M").astype(str)
    months = sorted(w["ClosedMonth"].dropna().unique())
    if len(months) < 2:
        return pd.DataFrame(), None, None
    prev_m, cur_m = months[-2], months[-1]
    g = (w[w["ClosedMonth"].isin([prev_m, cur_m])]
         .groupby(["Provider Name", "ClosedMonth"])
         .agg(ClosedCases=("Case Identifier", "count"), OverturnedCases=("IsAnyOverturn", "sum"))
         .reset_index())
    g["OverturnRate"] = np.where(g["ClosedCases"] > 0, g["OverturnedCases"] / g["ClosedCases"] * 100, 0.0)
    piv = g.pivot_table(index="Provider Name", columns="ClosedMonth",
                        values=["OverturnRate", "ClosedCases", "OverturnedCases"], aggfunc="sum", fill_value=0)
    out = pd.DataFrame({"Provider Name": piv.index})
    for label, month in (("Previous", prev_m), ("Current", cur_m)):
        out[f"{label} Month"] = month
        for metric, name in (("ClosedCases", "Closed Cases"), ("OverturnedCases", "Overturned Cases"),
                             ("OverturnRate", "Overturn Rate")):
            out[f"{label} {name}"] = piv[(metric, month)].to_numpy() if (metric, month) in piv.columns else 0
    out["Rate Increase"] = out["Current Overturn Rate"] - out["Previous Overturn Rate"]
    out = out[(out["Previous Closed Cases"] >= minimum_closed_cases) & (out["Current Closed Cases"] >= minimum_closed_cases)
              & (out["Rate Increase"] > 0)].copy()
    rate_cols = ["Previous Overturn Rate", "Current Overturn Rate", "Rate Increase"]
    out[rate_cols] = out[rate_cols].round(2)
    out = out.sort_values(["Rate Increase", "Current Closed Cases"], ascending=[False, False]).reset_index(drop=True)
    return out, prev_m, cur_m


def _provider_mom(df: pd.DataFrame, status_fn: Callable[[str], None]) -> AssistantAnswer:
    status_fn("📈 Comparing provider overturn rates for the latest two closed months…")
    try:
        mom, prev_m, cur_m = build_provider_overturn_mom(df)
    except ValueError as exc:
        return AssistantAnswer(text=f"Unable to calculate the provider month-over-month comparison: {exc}",
                               route="fast_path")
    if prev_m is None:
        return AssistantAnswer(text="At least two months of closed-case history are required to compare "
                                    "provider overturn rates.", route="fast_path")
    if mom.empty:
        return AssistantAnswer(text=f"No providers had a higher overturn rate in **{cur_m}** than in **{prev_m}**.",
                               table=mom, route="fast_path")
    chart = mom.head(15)
    fig = go.Figure()
    for name, col, color in ((f"Previous: {prev_m}", "Previous Overturn Rate", "#94a3b8"),
                             (f"Current: {cur_m}", "Current Overturn Rate", "#ef4444")):
        fig.add_trace(go.Bar(name=name, x=chart["Provider Name"], y=chart[col], marker_color=color,
                             text=[f"{v:.1f}%" for v in chart[col]], textposition="outside"))
    fig.update_layout(barmode="group", title=f"Providers With Increased Overturn Rate: {prev_m} vs {cur_m}",
                      yaxis_title="Overturn Rate (%)", legend=dict(orientation="h", y=1.12),
                      xaxis=dict(tickangle=-30), hovermode="x unified")
    top = mom.iloc[0]
    text = (f"**{len(mom):,} provider(s)** had a higher overturn rate in **{cur_m}** than in **{prev_m}**.\n\n"
            f"The largest increase was for **{top['Provider Name']}**, rising from "
            f"**{top['Previous Overturn Rate']:.2f}%** to **{top['Current Overturn Rate']:.2f}%**, an increase of "
            f"**{top['Rate Increase']:.2f} percentage points**.")
    insight = static_insight(
        "Medium", text,
        "A rising provider overturn rate may indicate a need to review initial determination consistency. The "
        "underlying cases should be reviewed before inferring a root cause.",
        "Overturn rate for the flagged providers reviewed in the next monthly cycle.",
        ["Review the providers with the largest increases and verify that each comparison has sufficient "
         "closed-case volume.", "Examine determination reasons and provider specialties for the affected providers.",
         "Monitor the same providers during the next monthly reporting cycle."],
        len(df),
        reasoning=(f"Compared {prev_m} with {cur_m} using Closed Date (CaseHistory). Overturn rate = overturned or "
                   "partially overturned closed cases ÷ closed cases per provider. Only providers present in both "
                   "periods with a positive increase are included."),
        metrics={"Previous Month": prev_m, "Current Month": cur_m, "Providers Increased": f"{len(mom):,}",
                 "Largest Increase": f"{top['Rate Increase']:.2f} percentage points"},
    )
    return AssistantAnswer(text=text, table=mom.head(100), figures=[style_figure(fig, 520, dict(l=30, r=30, t=90, b=150))],
                           insight=insight, route="fast_path")


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def comparison_charts(df: pd.DataFrame, col: str, val_a, val_b) -> list[go.Figure]:
    da, db = df[df[col] == val_a], df[df[col] == val_b]
    ka, kb = calculate_kpis(da), calculate_kpis(db)
    metrics = ["total_appeals", "overdue_count", "compliance_rate", "avg_turnaround"]
    labels = ["Total Appeals", "Overdue", "Compliance %", "Avg Turnaround"]
    fig_kpi = go.Figure()
    for vals, name, color in ((ka, val_a, PALETTE[0]), (kb, val_b, PALETTE[2])):
        fig_kpi.add_trace(go.Bar(name=str(name), x=labels, y=[vals.get(m, 0) for m in metrics], marker_color=color,
                                 text=[round(vals.get(m, 0), 2) for m in metrics], textposition="outside"))
    fig_kpi.update_layout(barmode="group", title=f"KPI: {val_a} vs {val_b}")
    tasks = make_subplots(rows=1, cols=2, subplot_titles=[str(val_a), str(val_b)])
    for frame, ci, color in ((da, 1, PALETTE[0]), (db, 2, PALETTE[2])):
        g = frame.groupby("Work Item Task Short Name")["Case Identifier"].count().reset_index()
        if not g.empty:
            tasks.add_trace(go.Bar(x=g["Work Item Task Short Name"], y=g["Case Identifier"], marker_color=color,
                                   showlegend=False), row=1, col=ci)
    tasks.update_layout(title="Appeal Volume by Task Type")
    age = go.Figure()
    for frame, name, color in ((da, val_a, PALETTE[0]), (db, val_b, PALETTE[2])):
        age.add_trace(go.Histogram(x=frame["CaseAgeDays"], name=str(name), opacity=0.7, marker_color=color, nbinsx=20))
    age.update_layout(barmode="overlay", title="Case Age Distribution")
    return [style_figure(fig_kpi, 420), style_figure(tasks, 400), style_figure(age, 380)]


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════
def run_fast_path(match: FastPathMatch, df: pd.DataFrame, status_fn: Callable[[str], None]) -> AssistantAnswer:
    """Compute a quick-action answer from the in-memory dataset (no SQL, no LLM)."""
    name = match.name
    n = len(df)

    if name == "comparison" and match.compare_values:
        a, b = match.compare_values
        status_fn(f"🔀 Building comparison: {a} vs {b}…")
        insight = static_insight(
            "Medium", f"Comparison of {a} vs {b} across {match.compare_column}.",
            "Segment-level performance differences drive staffing allocation and SLA-clearance priorities.",
            "Align SLA targets and routing workflows across both segments.",
            ["Clear overdue items in the underperforming segment within 30 days.",
             "Review escalation thresholds in both groups.",
             "Standardize routing comments/reason codes to reduce hand-off delay."], n)
        return AssistantAnswer(text=f"🔀 **{a} vs {b}** — side-by-side analysis across KPIs, task mix, and case age.",
                               figures=comparison_charts(df, match.compare_column, a, b), insight=insight,
                               route="fast_path")

    if name == "provider_mom":
        return _provider_mom(df, status_fn)

    if name == "forecast":
        status_fn("🔮 Building predictive forecast model…")
        fc = build_forecast(df)
        if "Volume" not in fc:
            return AssistantAnswer(text="Insufficient history for a forecast — at least 3 months of received "
                                        "appeals are needed.", route="fast_path")
        v = fc["Volume"]
        insight = static_insight(
            "Medium",
            f"Linear trend analysis projects a {v['trend']} appeal-volume trajectory over the next 6 months. "
            f"R² = {v['r2']:.3f} indicates model fit quality.",
            "Forecasting overdue backlog growth is critical for staffing and SLA-compliance planning.",
            "Align forecast with unit staffing and SLA targets.",
            ["Share the 6-month forecast with compliance leadership for staffing review.",
             "Set automated alerts at 110% of the forecast overdue threshold.",
             "Reconcile forecast assumptions with seasonal intake patterns."], n,
            reasoning=f"Trend slope: {v['slope']:+,.1f} cases/month. Volatility: {v['volatility']:,.1f}. "
                      "Linear least-squares fit on monthly received volume; R² shows fit quality.",
            metrics={"Trend Direction": v["trend"].title(), "Monthly Slope": f"{v['slope']:+,.1f}",
                     "Model R²": str(v["r2"]), "Volatility (1σ)": f"{v['volatility']:,.1f}"})
        return AssistantAnswer(text="🔮 **Predictive Forecast — 6-Month Outlook**", figures=[_forecast_figure(v)],
                               insight=insight, route="fast_path")

    if name == "anomaly_scan":
        status_fn("🔍 Running statistical anomaly detection…")
        anomalies = detect_anomalies(df, ANOMALY_CONFIG)
        figs, table = [], None
        if anomalies:
            counts: dict[str, int] = {}
            for item in anomalies:
                counts[item.get("severity", "Medium")] = counts.get(item.get("severity", "Medium"), 0) + 1
            fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), text=list(counts.values()),
                                   textposition="outside",
                                   marker_color=[SEVERITY_COLORS.get(k, "#94a3b8") for k in counts]))
            fig.update_layout(title="Anomaly Count by Severity Level", showlegend=False)
            figs.append(style_figure(fig, 380))
            table = pd.DataFrame([{"Severity": a.get("severity"), "Anomaly": a.get("detected"),
                                   "Category": a.get("category"), "Confidence": a.get("confidence"),
                                   "Recommended Action": a.get("recommended_action")} for a in anomalies])
        critical = any(a.get("severity") == "Critical" for a in anomalies)
        summary = (f"Statistical analysis detected {len(anomalies)} anomalies across the appeals portfolio. "
                   + ("Critical severity items require immediate escalation." if critical
                      else "High-priority items are flagged for review." if anomalies else ""))
        insight = static_insight(
            "Critical" if critical else ("High" if anomalies else "Low"), summary.strip(),
            "Unaddressed anomalies increase regulatory-complaint risk, SLA breach exposure, and potential "
            "member-experience escalation.",
            "Every Critical and High anomaly has an owner and a documented remediation.",
            ["Triage Critical anomalies within 24 hours — assign unit leads.",
             "Schedule High-severity reviews within 5 business days.",
             "Document remediation actions in the compliance audit trail."], n,
            reasoning="Seven statistical detectors (two-proportion z-tests with leave-one-out baselines, 2σ "
                      "risk-score outliers and aging thresholds) were run on the loaded dataset.",
            metrics={"Total Anomalies": str(len(anomalies)),
                     "Critical": str(sum(a.get("severity") == "Critical" for a in anomalies)),
                     "High": str(sum(a.get("severity") == "High" for a in anomalies)),
                     "Categories": str(len({a.get("category", "") for a in anomalies}))})
        return AssistantAnswer(text=f"🚨 **AI Anomaly Scan Complete — {len(anomalies)} Anomalies Detected**",
                               table=table, figures=figs, insight=insight, route="fast_path")

    if name == "summary":
        status_fn("⚡ Computing summary KPIs…")
        return AssistantAnswer(text="📊 **Dashboard Summary** — KPI cards above show the full picture.",
                               route="fast_path")

    if name == "top_sponsors":
        status_fn("⚡ Ranking plan sponsors by case volume…")
        top = (df.groupby("Plan Sponsor Name")["Case Identifier"].count().reset_index()
               .nlargest(10, "Case Identifier"))
        top.columns = ["Plan Sponsor", "Total Appeals"]
        fig = px.bar(top, x="Total Appeals", y="Plan Sponsor", orientation="h", text_auto=True,
                     color="Total Appeals", color_continuous_scale=SEQUENTIAL_BLUE,
                     title="Top 10 Plan Sponsors by Appeal Volume")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        return AssistantAnswer(text="🏆 **Top 10 Plan Sponsors by Appeal Volume**", table=top,
                               figures=[style_figure(fig, 500)], route="fast_path")

    if name == "task_breakdown":
        status_fn("⚡ Grouping by task type…")
        grp = (df.groupby("Work Item Task Short Name")
               .agg(Count=("Case Identifier", "count"), AvgRisk=("ComplianceRiskScore", "mean")).reset_index())
        pie = px.pie(grp, names="Work Item Task Short Name", values="Count", hole=0.45,
                     title="Appeal Volume by Task Type", color_discrete_sequence=PALETTE)
        bar = px.bar(grp, x="Work Item Task Short Name", y="AvgRisk", text_auto=".2f",
                     color="Work Item Task Short Name", color_discrete_sequence=PALETTE,
                     title="Avg Compliance Risk by Task Type")
        bar.update_layout(showlegend=False)
        return AssistantAnswer(text="📈 **Task Type Breakdown**", table=grp,
                               figures=[style_figure(pie), style_figure(bar)], route="fast_path")

    if name == "overdue":
        status_fn("⚡ Isolating overdue appeals…")
        sub = df[df["IsOverdue"]].copy()
        by_unit = (sub.groupby(["Current Unit Name", "Work Item Status Short Name"])["Case Identifier"]
                   .count().reset_index())
        by_unit.columns = ["Unit", "Status", "Count"]
        fig = px.bar(by_unit, x="Unit", y="Count", color="Status", barmode="stack",
                     title=f"Overdue Appeals by Unit ({len(sub):,} total)", color_discrete_map=STATUS_COLORS)
        avg_age = sub["CaseAgeDays"].mean() if len(sub) else 0.0
        rate = len(sub) / n * 100 if n else 0.0
        top_unit = by_unit.groupby("Unit")["Count"].sum().idxmax() if not by_unit.empty else "N/A"
        insight = static_insight(
            "High", f"{len(sub)} appeals are overdue (>{SLA_DAYS} days), averaging {avg_age:.0f} days open.",
            "Unresolved SLA breaches inflate regulatory-complaint risk and jeopardize CMS/state turnaround compliance.",
            "Clear SLA exceptions before month-end compliance reporting.",
            [f"Run an expedited-clearance report for all appeals beyond {AGING_THRESHOLD} days.",
             "Escalate overdue highest-level items to unit leads.",
             "Automate SLA-breach alerts to prevent recurrence."], n,
            reasoning=f"Overdue rate is {rate:.1f}% of {n:,} appeals. Largest contributor: {top_unit}.",
            metrics={"Overdue Count": f"{len(sub):,}", "Avg Age (Days)": f"{avg_age:.0f}",
                     "Overdue Rate": f"{rate:.1f}%",
                     "Avg Risk (Overdue)": f"{sub['ComplianceRiskScore'].mean():.3f}" if len(sub) else "—"})
        cols = ["Case Identifier", "Plan Sponsor Name", "Current Unit Name", "CaseAgeDays",
                "Work Item Status Short Name", "ComplianceRiskScore"]
        return AssistantAnswer(
            text=(f"🚨 **Overdue Appeals Isolated** — **{len(sub):,}** appeals beyond the {SLA_DAYS}-day SLA. "
                  f"Avg age: **{avg_age:.0f} days**"),
            table=sub[cols].head(50), figures=[style_figure(fig)], insight=insight, filtered_df=sub,
            route="fast_path")

    if name == "escalated":
        status_fn(f"⚡ Filtering Level-{ESCALATION_LEVEL} escalated appeals…")
        sub = df[df["IsEscalated"]].copy()
        grp = sub.groupby("Work Item Task Short Name")["Case Identifier"].count().reset_index()
        grp.columns = ["Task", "Count"]
        fig = px.bar(grp, x="Task", y="Count", text_auto=True, color="Count",
                     color_continuous_scale=["#fca5a5", "#ef4444", "#7f1d1d"],
                     title=f"Level-{ESCALATION_LEVEL} Escalated Appeals by Task ({len(sub):,} flagged)")
        fig.update_layout(coloraxis_showscale=False)
        still_open = int((~sub["IsClosed"]).sum())
        insight = static_insight(
            "High", f"{len(sub)} appeals are at the highest escalation level (Level {ESCALATION_LEVEL}).",
            "Escalation concentration increases quality risk and requires enhanced senior-reviewer oversight.",
            f"Dual sign-off and a documented decision trail for all Level-{ESCALATION_LEVEL} items.",
            [f"Verify decision-maker sign-off for all open Level-{ESCALATION_LEVEL} escalations.",
             "Flag clusters by plan sponsor for quality review.",
             f"Set automated alerts for Level-{ESCALATION_LEVEL} cases nearing SLA breach."], n,
            metrics={"Escalated Count": f"{len(sub):,}", "Still Open": f"{still_open:,}",
                     "Avg Age (Days)": f"{sub['CaseAgeDays'].mean():.0f}" if len(sub) else "—",
                     "Avg Risk (Escalated)": f"{sub['ComplianceRiskScore'].mean():.3f}" if len(sub) else "—"})
        cols = ["Case Identifier", "Plan Sponsor Name", "Current Unit Name", "CaseTypeLabel", "Routed To Site Name",
                "ComplianceRiskScore", "Work Item Status Short Name"]
        return AssistantAnswer(
            text=f"⚠️ **Level-{ESCALATION_LEVEL} Escalated Appeals** — **{len(sub):,}** flagged. "
                 f"Still open: **{still_open:,}**",
            table=sub[[c for c in cols if c in sub.columns]].head(50), figures=[style_figure(fig)],
            insight=insight, filtered_df=sub, route="fast_path")

    if name == "by_region":
        status_fn("⚡ Grouping by site…")
        grp = (df.groupby("Routed To Site Name")
               .agg(Count=("Case Identifier", "count"), Overdue=("IsOverdue", "sum"),
                    Avg_Risk=("ComplianceRiskScore", "mean")).reset_index())
        bar = px.bar(grp, x="Routed To Site Name", y="Count", text_auto=True, color="Routed To Site Name",
                     color_discrete_sequence=PALETTE, title="Appeal Volume by Site")
        bar.update_layout(showlegend=False, xaxis=dict(tickangle=-20))
        pie = px.pie(grp, names="Routed To Site Name", values="Count", hole=0.45, title="Volume Share by Site",
                     color_discrete_sequence=PALETTE)
        return AssistantAnswer(text="🗺️ **Site-Level Appeal Analysis**", table=grp,
                               figures=[style_figure(bar), style_figure(pie)], route="fast_path")

    if name == "unit":
        status_fn("⚡ Analysing by unit…")
        grp = (df.groupby("Current Unit Name")
               .agg(Count=("Case Identifier", "count"), Closed=("IsClosed", "sum"), Overdue=("IsOverdue", "sum"),
                    Escalated=("IsEscalated", "sum"), Avg_Risk=("ComplianceRiskScore", "mean")).reset_index())
        fig = px.bar(grp, x="Current Unit Name", y="Count", color="Current Unit Name", text_auto=True,
                     color_discrete_map=UNIT_COLORS, title="Appeal Volume by Unit")
        fig.update_layout(showlegend=False, xaxis=dict(tickangle=-25))
        return AssistantAnswer(text="🏢 **Unit-Level Analysis**", table=grp, figures=[style_figure(fig)],
                               route="fast_path")

    if name == "monthly_trend":
        status_fn("⚡ Building monthly trend…")
        trend = (df.groupby(["ReceivedMonth", "Current Unit Name"]).agg(Count=("Case Identifier", "count"))
                 .reset_index().sort_values("ReceivedMonth"))
        fig = px.area(trend, x="ReceivedMonth", y="Count", color="Current Unit Name", color_discrete_map=UNIT_COLORS,
                      title="Monthly Appeal Volume by Unit")
        return AssistantAnswer(text="📈 **Monthly Appeal Volume Trend**", figures=[style_figure(fig)],
                               route="fast_path")

    if name == "source":
        status_fn("⚡ Source & filer breakdown…")
        src = df.groupby("SourceLabel").agg(Count=("Case Identifier", "count")).reset_index()
        pie = px.pie(src, names="SourceLabel", values="Count", hole=0.45,
                     title="Appeal Count by Source (Pre/Post-Service)",
                     color_discrete_map={"Pre-Service": "#10b981", "Post-Service": "#3b82f6"})
        site = df.groupby(["Routed To Site Name", "BehalfOfLabel"])["Case Identifier"].count().reset_index()
        bar = px.bar(site, x="Routed To Site Name", y="Case Identifier", color="BehalfOfLabel", barmode="group",
                     color_discrete_map={"Provider": "#10b981", "Member": "#3b82f6"},
                     title="Appeal Count by Site & Filer (Provider/Member)")
        bar.update_layout(xaxis=dict(tickangle=-20))
        return AssistantAnswer(text="💱 **Source & Filer Analysis**", table=src,
                               figures=[style_figure(pie), style_figure(bar)], route="fast_path")

    if name == "risk_overview":
        status_fn("⚡ Computing risk distribution…")
        bands = pd.cut(df["ComplianceRiskScore"], bins=[0, .2, .4, .6, .8, 1.01],
                       labels=["Very Low", "Low", "Medium", "High", "Critical"])
        grp = df.groupby(bands, observed=False)["Case Identifier"].count().reset_index()
        grp.columns = ["Risk Band", "Count"]
        fig = px.bar(grp, x="Risk Band", y="Count", text_auto=True, color="Risk Band",
                     color_discrete_map=RISK_BAND_COLORS, title="Appeal Distribution by Compliance Risk Band")
        fig.update_layout(showlegend=False)
        return AssistantAnswer(text="🎯 **Compliance Risk Distribution Overview**", table=grp,
                               figures=[style_figure(fig)], route="fast_path")

    if name == "pending":
        status_fn("⚡ Filtering pending/open appeals…")
        sub = df[~df["IsClosed"]].copy()
        by_unit = sub.groupby("Current Unit Name")["Case Identifier"].count().reset_index()
        fig = px.bar(by_unit, x="Current Unit Name", y="Case Identifier", color="Current Unit Name",
                     color_discrete_map=UNIT_COLORS, title=f"Open/Pending Appeals by Unit ({len(sub)} total)")
        fig.update_layout(showlegend=False, xaxis=dict(tickangle=-25))
        cols = ["Case Identifier", "Plan Sponsor Name", "Current Unit Name", "Work Item Status Short Name",
                "CaseAgeDays", "ComplianceRiskScore"]
        return AssistantAnswer(text=f"⏳ **Open/Pending Appeals** — **{len(sub):,}** appeals still in progress.",
                               table=sub[cols].head(50), figures=[style_figure(fig)], filtered_df=sub,
                               route="fast_path")

    if name == "reviewer_segment":
        status_fn("⚡ Analysing reviewer segments…")
        grp = (df.groupby("Routed To User Segment Description")
               .agg(Count=("Case Identifier", "count"), Avg_Risk=("ComplianceRiskScore", "mean"),
                    Overdue=("IsOverdue", "sum")).reset_index())
        fig = px.bar(grp, x="Routed To User Segment Description", y="Count",
                     color="Routed To User Segment Description", text_auto=True,
                     title="Reviewer Segment — Appeal Volume", color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False)
        return AssistantAnswer(text="🏭 **Reviewer Segment Analysis**", table=grp, figures=[style_figure(fig)],
                               route="fast_path")

    raise ValueError(f"Unknown fast path: {name}")


__all__ = ["FAST_QUERIES", "FastPathMatch", "match_fast_path", "run_fast_path", "build_forecast",
           "build_provider_overturn_mom", "comparison_charts", "static_insight", "ANOMALY_CONFIG"]
