"""Risk & Findings page for DataTalk.AI (deterministic — no LLM calls).

Usage:

    from components.risk_findings import RiskConfig, render_risk_findings_tab
    with t_risk:
        render_risk_findings_tab(
            current_df,
            RiskConfig(sla_days=SLA_DAYS, escalation_level=ESCALATION_LEVEL,
                       aging_threshold=AGING_THRESHOLD),
            detect_anomalies_fn=lambda d: detect_anomalies(d, ANOMALY_CONFIG),
            risk_emoji=RISK_EMOJI,
        )

Sections
  0. Scope bar            filters (date, sponsor, unit, site, task, filer) + active-filter summary
  1. Risk posture         overall risk index, 8 KPI cards with status, period-over-period deltas
  2. Findings             evidence-based findings, severity-ranked, each with owner, timeframe,
                          success measure, contributors and a Draft in Outlook button
  3. Risk register        composite risk score per Sponsor / Unit / Site / Task, tiers, mail per entity
  4. Risk analytics       heatmap · Pareto · risk matrix · aging buckets · trend · site radar
  5. Case watchlist       oldest / highest-risk open cases, overdue cases, mail + CSV
  6. Anomaly intelligence your detect_anomalies() scan, with mail per anomaly and CSV
  7. Exports              findings, register, watchlist (CSV) and full risk summary mail

Design rules
  * Every number shown is computed from the filtered dataframe; no hardcoded values.
  * Thresholds live in RiskConfig, not in the UI code.
  * Optional columns are detected — sections hide themselves when a column is missing.
  * Rates are never summed; portfolio rates are recomputed from case counts.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.outlook_mail import MAIL_CSS, esc, mail_button, mail_url, short
from core.plot_style import style_figure


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RiskConfig:
    sla_days: int = 30
    escalation_level: int = 2
    aging_threshold: int = 60
    high_risk_score: float = 0.5
    min_entity_volume: int = 30               # entities below this are flagged "low volume"
    watchlist_size: int = 50
    trend_window_days: int = 30               # current vs prior window for deltas
    # (watch, critical) thresholds per KPI
    thresholds: dict = field(default_factory=lambda: {
        "overdue_pct": (10.0, 25.0),
        "escalated_pct": (10.0, 20.0),
        "transferred_pct": (8.0, 15.0),
        "open_pct": (50.0, 70.0),
        "backlog_pct": (25.0, 45.0),
        "avg_age": (45.0, 90.0),
        "avg_risk": (0.2, 0.5),
        "member_pct": (55.0, 70.0),
        "aged_open_pct": (15.0, 30.0),
    })
    # composite score weights (sum to 1)
    weights: dict = field(default_factory=lambda: {
        "overdue_rate": 0.35, "avg_risk": 0.20, "aged_open_rate": 0.20,
        "escalation_rate": 0.15, "transfer_rate": 0.10,
    })
    # dimension label -> column (missing columns are skipped automatically)
    dimensions: dict = field(default_factory=lambda: {
        "Plan Sponsor": "Plan Sponsor Name",
        "Unit": "Current Unit Name",
        "Site": "Routed To Site Name",
        "Task": "Work Item Task Short Name",
    })
    date_column_candidates: tuple = ("Received Date", "ReceivedDate", "Created Date")
    case_id: str = "Case Identifier"
    key_prefix: str = "rf"


SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
SEV_COLOR = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#eab308", "Low": "#16a34a", "Info": "#3b82f6"}
SEV_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Info": "🔵"}
TIER_COLOR = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#eab308", "Low": "#16a34a"}

MARGIN = dict(l=10, r=20, t=50, b=20)
SEQ = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4", "#f43f5e", "#6366f1"]

CSS = """
<style>
.rf-head{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin:4px 0 14px;}
.rf-title{font-size:1.25rem;font-weight:800;}
.rf-sub{font-size:.8rem;opacity:.72;margin-top:2px;}
.rf-sec{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--dt-accent);font-weight:800;margin:22px 0 10px;
  display:flex;align-items:center;gap:8px;}
.rf-sec:after{content:"";flex:1;height:1px;background:var(--dt-border);}
.rf-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:8px;}
@media (max-width:1100px){.rf-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
.rf-card{background:var(--dt-surface);border:1px solid var(--dt-border);border-radius:12px;
  padding:14px 16px;position:relative;overflow:hidden;}
.rf-card:before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:var(--acc);}
.rf-l{font-size:.64rem;opacity:.72;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:6px;}
.rf-v{font-size:1.6rem;font-weight:800;line-height:1.05;}
.rf-s{font-size:.74rem;opacity:.72;margin:4px 0 6px;min-height:1em;}
.rf-row{display:flex;justify-content:space-between;align-items:center;gap:6px;}
.rf-badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.64rem;font-weight:800;letter-spacing:.03em;}
.rf-delta{font-size:.7rem;font-weight:700;}
.rf-index{display:flex;gap:18px;align-items:center;background:linear-gradient(135deg,rgba(37,99,235,.08),rgba(124,58,237,.05));
  border:1px solid var(--dt-border);border-radius:14px;padding:16px 20px;margin-bottom:14px;flex-wrap:wrap;}
.rf-index-score{font-size:2.6rem;font-weight:900;line-height:1;}
.rf-index-lbl{font-size:.66rem;opacity:.75;text-transform:uppercase;letter-spacing:.12em;font-weight:800;}
.rf-index-txt{font-size:.86rem;line-height:1.6;flex:1;min-width:260px;}
.rf-find{background:var(--dt-surface);border:1px solid var(--bc);border-left:4px solid var(--sc);
  border-radius:0 12px 12px 0;padding:14px 16px;margin-bottom:12px;}
.rf-find-t{font-size:.96rem;font-weight:800;margin:4px 0 6px;}
.rf-find-d{font-size:.86rem;line-height:1.7;}
.rf-chip{display:inline-block;background:var(--dt-surface-strong);border-radius:20px;padding:2px 9px;
  font-size:.68rem;margin:2px 4px 2px 0;}
.rf-kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:10px 0;}
.rf-kv div{background:var(--dt-surface-strong);border:1px solid var(--dt-border);border-radius:8px;padding:7px 10px;}
.rf-kv b{display:block;font-size:.62rem;opacity:.72;text-transform:uppercase;letter-spacing:.08em;}
.rf-kv span{font-size:.88rem;font-weight:700;}
.rf-act{background:rgba(37,99,235,.07);border:1px solid rgba(37,99,235,.22);border-radius:8px;padding:9px 12px;
  font-size:.84rem;margin-top:8px;}
.rf-foot{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px;
  padding-top:10px;border-top:1px dashed var(--dt-border);}
.rf-hint{font-size:.7rem;opacity:.72;}
.rf-bar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 14px;
  margin:6px 0 10px;border-radius:10px;background:var(--dt-surface);border:1px solid var(--dt-border);}
.rf-bar-t{font-size:.84rem;font-weight:700;}
.rf-bar-s{font-size:.72rem;opacity:.72;margin-top:2px;}
.rf-empty{background:rgba(37,99,235,.06);border:1px solid rgba(37,99,235,.18);border-radius:12px;padding:18px 22px;
  text-align:center;font-size:.9rem;}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) > 0
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _prepare(df: pd.DataFrame, cfg: RiskConfig) -> tuple[pd.DataFrame, str | None, list]:
    d = df.copy()
    missing = []
    for col in ("IsOverdue", "IsEscalated", "IsTransferred", "IsClosed"):
        if col in d:
            d[col] = _bool(d[col])
        else:
            d[col] = False
            missing.append(col)
    for col in ("CaseAgeDays", "OpenAgingDays", "ComplianceRiskScore"):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = np.nan
            missing.append(col)
    if d["OpenAgingDays"].isna().all() and not d["CaseAgeDays"].isna().all():
        d["OpenAgingDays"] = np.where(~d["IsClosed"], d["CaseAgeDays"], 0)
    if cfg.case_id not in d:
        d[cfg.case_id] = np.arange(len(d)).astype(str)
    date_col = next((c for c in cfg.date_column_candidates if c in d), None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        if d[date_col].isna().all():
            date_col = None
    d["IsOpen"] = ~d["IsClosed"]
    d["IsAgedOpen"] = d["IsOpen"] & (d["CaseAgeDays"] > cfg.aging_threshold)
    d["IsHighRisk"] = d["ComplianceRiskScore"] > cfg.high_risk_score
    return d, date_col, missing


def _dims(d: pd.DataFrame, cfg: RiskConfig) -> dict:
    return {lbl: col for lbl, col in cfg.dimensions.items() if col in d and d[col].nunique(dropna=True) > 1}


def _pct(n: float, base: float) -> float:
    return round(n / base * 100, 1) if base else 0.0


def _kpis(d: pd.DataFrame) -> dict:
    n = len(d)
    open_n = int(d["IsOpen"].sum())
    total_age = float(d["CaseAgeDays"].sum()) or 1.0
    backlog_days = float(d.loc[d["IsOpen"], "OpenAgingDays"].sum())
    member_n = int((d["BehalfOfLabel"] == "Member").sum()) if "BehalfOfLabel" in d else None
    return {
        "n": n,
        "overdue_n": int(d["IsOverdue"].sum()), "overdue_pct": _pct(d["IsOverdue"].sum(), n),
        "escalated_n": int(d["IsEscalated"].sum()), "escalated_pct": _pct(d["IsEscalated"].sum(), n),
        "transferred_n": int(d["IsTransferred"].sum()), "transferred_pct": _pct(d["IsTransferred"].sum(), n),
        "open_n": open_n, "open_pct": _pct(open_n, n),
        "aged_open_n": int(d["IsAgedOpen"].sum()), "aged_open_pct": _pct(d["IsAgedOpen"].sum(), open_n),
        "backlog_days": int(backlog_days), "backlog_pct": round(backlog_days / total_age * 100, 1),
        "avg_age": round(float(d["CaseAgeDays"].mean()), 1) if d["CaseAgeDays"].notna().any() else 0.0,
        "max_age": int(d["CaseAgeDays"].max()) if d["CaseAgeDays"].notna().any() else 0,
        "oldest_open": int(d.loc[d["IsOpen"], "CaseAgeDays"].max()) if (d["IsOpen"].any() and d["CaseAgeDays"].notna().any()) else 0,
        "avg_risk": round(float(d["ComplianceRiskScore"].mean()), 3) if d["ComplianceRiskScore"].notna().any() else 0.0,
        "high_risk_n": int(d["IsHighRisk"].sum()),
        "member_n": member_n, "member_pct": _pct(member_n, n) if member_n is not None else None,
    }


def _status(value: float | None, key: str, cfg: RiskConfig) -> str:
    if value is None or key not in cfg.thresholds:
        return "Info"
    watch, crit = cfg.thresholds[key]
    if value > crit:
        return "Critical"
    if value > watch:
        return "Medium"
    return "Low"


# ═══════════════════════════════════════════════════════════════════════════════
# RISK REGISTER (composite score per entity)
# ═══════════════════════════════════════════════════════════════════════════════
def build_register(d: pd.DataFrame, col: str, cfg: RiskConfig) -> pd.DataFrame:
    g = d.groupby(col, dropna=True).agg(
        Cases=(cfg.case_id, "count"), Open=("IsOpen", "sum"), Overdue=("IsOverdue", "sum"),
        Escalated=("IsEscalated", "sum"), Transferred=("IsTransferred", "sum"),
        AgedOpen=("IsAgedOpen", "sum"), HighRisk=("IsHighRisk", "sum"),
        AvgRisk=("ComplianceRiskScore", "mean"), AvgAge=("CaseAgeDays", "mean"),
        OldestOpen=("OpenAgingDays", "max"),
    ).reset_index().rename(columns={col: "Entity"})
    if g.empty:
        return g
    for c in ("Open", "Overdue", "Escalated", "Transferred", "AgedOpen", "HighRisk"):
        g[c] = g[c].astype(int)
    g["Overdue %"] = (g["Overdue"] / g["Cases"] * 100).round(1)
    g["Escalated %"] = (g["Escalated"] / g["Cases"] * 100).round(1)
    g["Transferred %"] = (g["Transferred"] / g["Cases"] * 100).round(1)
    g["Aged Open %"] = np.where(g["Open"] > 0, g["AgedOpen"] / g["Open"].replace(0, np.nan) * 100, 0).round(1)
    g["Share of Overdue %"] = (g["Overdue"] / max(int(g["Overdue"].sum()), 1) * 100).round(1)
    g["AvgRisk"] = g["AvgRisk"].fillna(0).round(3)
    g["AvgAge"] = g["AvgAge"].fillna(0).round(1)
    g["OldestOpen"] = g["OldestOpen"].fillna(0).astype(int)

    # composite score: percentile rank of each driver (eligible entities only), weighted, 0–100
    eligible = g["Cases"] >= cfg.min_entity_volume
    base = g[eligible] if eligible.sum() >= 3 else g
    drivers = {"overdue_rate": "Overdue %", "avg_risk": "AvgRisk", "aged_open_rate": "Aged Open %",
               "escalation_rate": "Escalated %", "transfer_rate": "Transferred %"}
    score = np.zeros(len(g))
    for key, colname in drivers.items():
        w = cfg.weights.get(key, 0)
        ref = base[colname].values
        if w == 0 or np.nanstd(ref) == 0:
            continue
        score += w * np.array([float((ref <= v).mean()) for v in g[colname].values])
    g["Risk Score"] = (score * 100).round(0).astype(int)
    g["Tier"] = pd.cut(g["Risk Score"], bins=[-1, 29, 49, 69, 101], labels=["Low", "Medium", "High", "Critical"]).astype(str)
    g["Low Volume"] = ~eligible
    g.loc[g["Low Volume"], "Tier"] = g.loc[g["Low Volume"], "Tier"] + " (low vol.)"
    g["Top Driver"] = [
        max(((k, (base[c] <= row[c]).mean() * cfg.weights.get(k, 0)) for k, c in drivers.items()),
            key=lambda t: t[1])[0].replace("_", " ")
        for _, row in g.iterrows()
    ]
    return g.sort_values(["Risk Score", "Overdue"], ascending=[False, False]).reset_index(drop=True)


def portfolio_index(reg: pd.DataFrame) -> int:
    """Volume-weighted composite score across entities (0–100)."""
    if reg is None or reg.empty or reg["Cases"].sum() == 0:
        return 0
    return int(round(float((reg["Risk Score"] * reg["Cases"]).sum() / reg["Cases"].sum())))


# ═══════════════════════════════════════════════════════════════════════════════
# FINDINGS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def _finding(sev, title, detail, metrics, action, owner, timeframe, success, contributors=None, category=""):
    return {"severity": sev, "title": title, "detail": detail, "metrics": metrics,
            "action": action, "owner": owner, "timeframe": timeframe, "success": success,
            "contributors": contributors or [], "category": category}


def _top_contrib(reg: pd.DataFrame | None, count_col: str, k: int = 3) -> list:
    if reg is None or reg.empty or count_col not in reg:
        return []
    top = reg.nlargest(k, count_col)
    total = max(int(reg[count_col].sum()), 1)
    return [f"{r.Entity} — {int(getattr(r, count_col)):,} ({getattr(r, count_col) / total * 100:.1f}%)"
            for r in top.itertuples() if getattr(r, count_col) > 0]


def build_findings(d: pd.DataFrame, k: dict, cfg: RiskConfig, reg: pd.DataFrame | None,
                   dim_label: str, prev: dict | None = None, trend: tuple = (None, None)) -> list:
    out = []
    t = cfg.thresholds
    ent = dim_label.lower()

    # 1 Overdue
    if k["overdue_pct"] > t["overdue_pct"][0]:
        sev = "Critical" if k["overdue_pct"] > t["overdue_pct"][1] else "High"
        top = reg.nlargest(1, "Overdue").iloc[0] if reg is not None and not reg.empty else None
        out.append(_finding(
            sev, f"Overdue rate {k['overdue_pct']}% exceeds the {t['overdue_pct'][0]:.0f}% watch level",
            f"{k['overdue_n']:,} of {k['n']:,} appeals are past the {cfg.sla_days}-day SLA."
            + (f" {top.Entity} carries the most overdue cases ({int(top.Overdue):,}, {top['Share of Overdue %']}% of all overdue)." if top is not None else ""),
            {"Overdue": f"{k['overdue_n']:,}", "Overdue rate": f"{k['overdue_pct']}%",
             "Watch / critical": f"{t['overdue_pct'][0]:.0f}% / {t['overdue_pct'][1]:.0f}%"},
            f"Run a daily SLA-clearance huddle for the top overdue {ent}s, oldest cases first.",
            "Operations Lead", "Within 1 week" if sev == "Critical" else "Next 2 weeks",
            f"Overdue rate below {t['overdue_pct'][0]:.0f}%.", _top_contrib(reg, "Overdue"), "SLA"))

    # 2 Open backlog
    if k["open_pct"] > t["open_pct"][0]:
        sev = "High" if k["open_pct"] > t["open_pct"][1] else "Medium"
        out.append(_finding(
            sev, f"Open backlog at {k['open_pct']}% of appeals",
            f"{k['open_n']:,} appeals are not closed; {k['aged_open_n']:,} of them are older than {cfg.aging_threshold} days "
            f"and the oldest open case is {k['oldest_open']:,} days old.",
            {"Open": f"{k['open_n']:,}", "Open rate": f"{k['open_pct']}%", "Aged open": f"{k['aged_open_n']:,}"},
            "Review capacity against open volume by unit and rebalance work queues.",
            "Workforce / Capacity Planner", "Next planning cycle",
            f"Open rate below {t['open_pct'][0]:.0f}%.", _top_contrib(reg, "Open"), "Backlog"))

    # 3 Aged open
    if k["aged_open_pct"] > t["aged_open_pct"][0]:
        sev = "High" if k["aged_open_pct"] > t["aged_open_pct"][1] else "Medium"
        out.append(_finding(
            sev, f"{k['aged_open_pct']}% of open appeals are older than {cfg.aging_threshold} days",
            f"{k['aged_open_n']:,} open appeals have passed the {cfg.aging_threshold}-day aging threshold; "
            f"open-case aging totals {k['backlog_days']:,} days ({k['backlog_pct']}% of portfolio age).",
            {"Aged open": f"{k['aged_open_n']:,}", "Oldest open": f"{k['oldest_open']:,} days",
             "Backlog days": f"{k['backlog_days']:,}"},
            f"Assign a named owner to every open appeal older than {cfg.aging_threshold} days and review weekly.",
            "Unit Leads", "This week", f"No open appeal older than {cfg.aging_threshold} days without an owner.",
            _top_contrib(reg, "AgedOpen"), "Aging"))

    # 4 Escalation
    if k["escalated_pct"] > t["escalated_pct"][0]:
        sev = "High" if k["escalated_pct"] > t["escalated_pct"][1] else "Medium"
        out.append(_finding(
            sev, f"Escalation rate {k['escalated_pct']}% (Level {cfg.escalation_level})",
            f"{k['escalated_n']:,} appeals are escalated to Level {cfg.escalation_level}.",
            {"Escalated": f"{k['escalated_n']:,}", "Escalation rate": f"{k['escalated_pct']}%"},
            "Require senior-reviewer sign-off and a documented decision trail for escalated appeals.",
            "Appeals Quality Lead", "Next 2 weeks", f"Escalation rate below {t['escalated_pct'][0]:.0f}%.",
            _top_contrib(reg, "Escalated"), "Escalation"))

    # 5 Transfers
    if k["transferred_pct"] > t["transferred_pct"][0]:
        sev = "High" if k["transferred_pct"] > t["transferred_pct"][1] else "Medium"
        out.append(_finding(
            sev, f"Reassignment rate {k['transferred_pct']}%",
            f"{k['transferred_n']:,} appeals were transferred, which may indicate unclear routing rules or capacity imbalance.",
            {"Transferred": f"{k['transferred_n']:,}", "Transfer rate": f"{k['transferred_pct']}%"},
            "Audit routing rules for the most-transferred work types and fix first-time-right routing.",
            "Routing / Workflow Owner", "Next 30 days", f"Transfer rate below {t['transferred_pct'][0]:.0f}%.",
            _top_contrib(reg, "Transferred"), "Routing"))

    # 6 High risk score
    if k["high_risk_n"] > 0 and k["avg_risk"] > t["avg_risk"][0]:
        sev = "High" if k["avg_risk"] > t["avg_risk"][1] else "Medium"
        out.append(_finding(
            sev, f"{k['high_risk_n']:,} appeals score above {cfg.high_risk_score} compliance risk",
            f"Average compliance risk score is {k['avg_risk']} against a watch level of {t['avg_risk'][0]}.",
            {"High-risk cases": f"{k['high_risk_n']:,}", "Avg risk score": f"{k['avg_risk']}"},
            "Prioritise high-risk appeals in the watchlist for compliance review before closure.",
            "Compliance Lead", "Next 2 weeks", f"Average risk score below {t['avg_risk'][0]}.",
            _top_contrib(reg, "HighRisk"), "Compliance"))

    # 7 Concentration
    if reg is not None and len(reg) >= 3 and k["overdue_n"] > 0:
        top = reg.nlargest(1, "Overdue").iloc[0]
        if top["Share of Overdue %"] >= 40:
            out.append(_finding(
                "High", f"{top.Entity} holds {top['Share of Overdue %']}% of all overdue appeals",
                f"One {ent} accounts for {int(top.Overdue):,} of {k['overdue_n']:,} overdue appeals, so SLA performance "
                f"depends heavily on it.",
                {"Overdue": f"{int(top.Overdue):,}", "Share": f"{top['Share of Overdue %']}%", "Its overdue rate": f"{top['Overdue %']}%"},
                f"Set a dedicated recovery plan for {top.Entity} with weekly progress tracking.",
                f"{dim_label} Lead — {top.Entity}", "Next 2 weeks",
                f"{top.Entity} share of overdue below 40%.", [], "Concentration"))

    # 8 Statistical outliers among entities
    if reg is not None and not reg.empty:
        el = reg[~reg["Low Volume"]]
        if len(el) >= 5:
            mu, sd = el["Overdue %"].mean(), el["Overdue %"].std(ddof=0)
            outl = el[el["Overdue %"] > mu + 2 * sd] if sd > 0 else el.iloc[0:0]
            for _, r in outl.head(3).iterrows():
                out.append(_finding(
                    "High", f"{r['Entity']} is an overdue outlier ({r['Overdue %']}%)",
                    f"Its overdue rate is more than two standard deviations above the {ent} average of {mu:.1f}% "
                    f"({int(r['Overdue']):,} overdue of {int(r['Cases']):,}).",
                    {"Overdue rate": f"{r['Overdue %']}%", f"{dim_label} average": f"{mu:.1f}%",
                     "Cases": f"{int(r['Cases']):,}"},
                    f"Root-cause review of {r['Entity']}: intake mix, staffing, pend reasons.",
                    f"{dim_label} Lead — {r['Entity']}", "Next 2 weeks",
                    f"{r['Entity']} overdue rate at or below {mu:.1f}%.", [], "Outlier"))

    # 9 Trend vs prior window
    tc, tp = trend
    if tc and tp and tp.get("n", 0) >= 20 and tc["n"] >= 20:
        delta = round(tc["overdue_pct"] - tp["overdue_pct"], 1)
        prev = tp
        if delta >= 5:
            out.append(_finding(
                "High" if delta >= 10 else "Medium",
                f"Overdue rate up {delta} percentage points vs the prior {cfg.trend_window_days} days",
                f"Appeals received in the last {cfg.trend_window_days} days are overdue at {tc['overdue_pct']}% "
                f"compared with {prev['overdue_pct']}% in the prior window.",
                {"Current": f"{tc['overdue_pct']}%", "Prior": f"{prev['overdue_pct']}%", "Change": f"+{delta} percentage points"},
                "Check intake volume, staffing and routing changes in the latest window.",
                "Operations Lead", "Within 1 week", "Overdue rate back to the prior-window level.", [], "Trend"))

    # 10 Member share
    if k.get("member_pct") is not None and k["member_pct"] > t["member_pct"][0]:
        out.append(_finding(
            "Medium" if k["member_pct"] > t["member_pct"][1] else "Low",
            f"Member-filed appeals at {k['member_pct']}%",
            f"{k['member_n']:,} appeals were filed by members, which raises member-experience exposure.",
            {"Member-filed": f"{k['member_n']:,}", "Share": f"{k['member_pct']}%"},
            "Track member-experience measures for member-filed appeals and review self-service intake options.",
            "Member Experience Lead", "Next quarter", f"Member-filed share below {t['member_pct'][0]:.0f}%.",
            [], "Member"))

    if not out:
        out.append(_finding(
            "Low", "Risk profile within thresholds",
            f"Overdue ({k['overdue_pct']}%), open ({k['open_pct']}%) and escalated ({k['escalated_pct']}%) are all within "
            f"their watch levels.",
            {"Overdue rate": f"{k['overdue_pct']}%", "Open rate": f"{k['open_pct']}%", "Escalation rate": f"{k['escalated_pct']}%"},
            "Continue weekly monitoring of the KPIs on this tab.", "Reporting / Analytics Owner", "Weekly",
            "All KPIs stay within watch levels.", [], "Health"))
    out.sort(key=lambda f: SEV_ORDER.get(f["severity"], 9))
    for i, f in enumerate(out, 1):
        f["id"] = f"RF-{i:02d}"
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIL BODIES
# ═══════════════════════════════════════════════════════════════════════════════
def _scope_text(scope: list) -> str:
    return "; ".join(scope) if scope else "All appeals"


def _finding_mail(f: dict, scope: list) -> tuple:
    subject = f"[DataTalk.AI] {f['severity']} risk finding: {short(f['title'], 90)}"
    lines = ["Hi,", "", "Please take ownership of the following risk finding from the DataTalk.AI Risk & Findings review.", "",
             f"FINDING ({f['id']} · {f['severity']})", f["title"], "", f["detail"], "", "KEY METRICS"]
    lines += [f"- {k}: {v}" for k, v in f["metrics"].items()]
    if f["contributors"]:
        lines += ["", "TOP CONTRIBUTORS"] + [f"- {c}" for c in f["contributors"]]
    lines += ["", "REQUESTED ACTION", f["action"], "", f"Owner role: {f['owner']}", f"Timeframe: {f['timeframe']}",
              f"Success measure: {f['success']}", "", f"Scope: {_scope_text(scope)}",
              f"Generated: {date.today():%d %b %Y}", "", "Please confirm ownership and the target date by reply.", "", "Thanks,"]
    return subject, "\n".join(lines)


def _entity_mail(row: pd.Series, dim_label: str, cfg: RiskConfig, scope: list) -> tuple:
    subject = f"[DataTalk.AI] {dim_label} risk review: {short(row['Entity'], 60)} — {row['Tier']} (score {row['Risk Score']})"
    lines = ["Hi,", "", f"The DataTalk.AI risk register flags {row['Entity']} for review.", "",
             f"{dim_label.upper()} RISK PROFILE",
             f"- Composite risk score: {row['Risk Score']} / 100 ({row['Tier']})",
             f"- Top driver: {row['Top Driver']}",
             f"- Cases: {int(row['Cases']):,} · Open: {int(row['Open']):,}",
             f"- Overdue: {int(row['Overdue']):,} ({row['Overdue %']}%) — {row['Share of Overdue %']}% of all overdue",
             f"- Open older than {cfg.aging_threshold} days: {int(row['AgedOpen']):,} ({row['Aged Open %']}% of open)",
             f"- Escalated: {int(row['Escalated']):,} ({row['Escalated %']}%) · Transferred: {int(row['Transferred']):,} ({row['Transferred %']}%)",
             f"- Avg risk score: {row['AvgRisk']} · Avg case age: {row['AvgAge']} days · Oldest open: {int(row['OldestOpen']):,} days",
             "", "REQUESTED ACTION",
             f"Review the open and overdue appeals for {row['Entity']}, confirm the main cause behind the {row['Top Driver']} "
             f"and share a recovery plan with target dates.",
             "", f"Scope: {_scope_text(scope)}", f"Generated: {date.today():%d %b %Y}", "", "Thanks,"]
    return subject, "\n".join(lines)


def _summary_mail(k: dict, findings: list, reg: pd.DataFrame | None, dim_label: str, idx: int, scope: list, cfg) -> tuple:
    subject = f"[DataTalk.AI] Risk & Findings summary — {date.today():%d %b %Y} — risk index {idx}/100"
    crit = sum(f["severity"] == "Critical" for f in findings)
    high = sum(f["severity"] == "High" for f in findings)
    lines = ["Hi,", "", "Summary of the current appeals risk position from DataTalk.AI.", "",
             f"Scope: {_scope_text(scope)}", f"Portfolio risk index: {idx} / 100", "", "HEADLINE KPIs",
             f"- Appeals: {k['n']:,}",
             f"- Overdue (> {cfg.sla_days} days): {k['overdue_n']:,} ({k['overdue_pct']}%)",
             f"- Open: {k['open_n']:,} ({k['open_pct']}%) · older than {cfg.aging_threshold} days: {k['aged_open_n']:,}",
             f"- Escalated: {k['escalated_n']:,} ({k['escalated_pct']}%) · Transferred: {k['transferred_n']:,} ({k['transferred_pct']}%)",
             f"- Avg case age: {k['avg_age']} days · Avg risk score: {k['avg_risk']}", "",
             f"FINDINGS ({len(findings)} total · {crit} critical · {high} high)"]
    lines += [f"{i}. [{f['severity']}] {f['title']} — {f['owner']}, {f['timeframe']}" for i, f in enumerate(findings[:6], 1)]
    if reg is not None and not reg.empty:
        lines += ["", f"TOP {dim_label.upper()} RISKS"]
        lines += [f"- {r['Entity']}: score {int(r['Risk Score'])} ({r['Tier']}), overdue {int(r['Overdue']):,}"
                  for _, r in reg[~reg["Low Volume"]].head(5).iterrows()]
    lines += ["", "Please review and confirm owners by reply.", "", "Thanks,"]
    return subject, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _sec(title: str) -> None:
    st.markdown(f"<div class='rf-sec'>{esc(title)}</div>", unsafe_allow_html=True)


def _style(fig: go.Figure, h: int) -> go.Figure:
    return style_figure(fig, h, MARGIN)


def _delta_html(cur: float | None, prev: float | None, lower_is_better: bool = True, unit: str = "") -> str:
    if cur is None or prev is None:
        return ""
    dv = round(cur - prev, 1)
    if abs(dv) < 0.05:
        return "<span class='rf-delta' style='color:inherit;opacity:.72'>→ flat</span>"
    good = (dv < 0) if lower_is_better else (dv > 0)
    col = "#10b981" if good else "#ef4444"
    arrow = "▲" if dv > 0 else "▼"
    return f"<span class='rf-delta' style='color:{col}' title='vs prior window'>{arrow} {abs(dv)}{unit}</span>"


def _card(label, value, sub, status, delta="", accent="#3b82f6") -> str:
    sc = SEV_COLOR.get(status, "#3b82f6")
    word = {"Critical": "Critical", "Medium": "Watch", "Low": "Healthy", "Info": "Info", "High": "High"}.get(status, status)
    return (f"<div class='rf-card' style='--acc:{accent}'><div class='rf-l'>{esc(label)}</div>"
            f"<div class='rf-v'>{esc(value)}</div><div class='rf-s'>{esc(sub)}</div>"
            f"<div class='rf-row'><span class='rf-badge' style='background:{sc}22;color:{sc};border:1px solid {sc}44'>"
            f"{esc(word)}</span>{delta}</div></div>")


def _window_kpis(d: pd.DataFrame, date_col: str | None, cfg: RiskConfig) -> tuple:
    if not date_col:
        return None, None
    end = d[date_col].max()
    if pd.isna(end):
        return None, None
    w = timedelta(days=cfg.trend_window_days)
    cur = d[(d[date_col] > end - w) & (d[date_col] <= end)]
    prv = d[(d[date_col] > end - 2 * w) & (d[date_col] <= end - w)]
    if len(cur) < 20 or len(prv) < 20:
        return None, None
    return _kpis(cur), _kpis(prv)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _scope_bar(d: pd.DataFrame, date_col: str | None, dims: dict, cfg: RiskConfig) -> tuple:
    kp = cfg.key_prefix
    scope = []
    with st.expander("🎛️ Scope & filters", expanded=False):
        cols = st.columns(3)
        if date_col:
            lo, hi = d[date_col].min().date(), d[date_col].max().date()
            with cols[0]:
                rng = st.date_input("Received date range", (lo, hi), min_value=lo, max_value=hi, key=f"{kp}_dates")
            if isinstance(rng, (list, tuple)) and len(rng) == 2 and (rng[0] != lo or rng[1] != hi):
                d = d[(d[date_col].dt.date >= rng[0]) & (d[date_col].dt.date <= rng[1])]
                scope.append(f"Received {rng[0]:%d %b %Y} – {rng[1]:%d %b %Y}")
        slots = list(dims.items())
        if "BehalfOfLabel" in d and d["BehalfOfLabel"].nunique() > 1:
            slots.append(("Filed on behalf of", "BehalfOfLabel"))
        for i, (lbl, col) in enumerate(slots):
            with cols[(i + 1) % 3]:
                opts = sorted(d[col].dropna().astype(str).unique().tolist())
                sel = st.multiselect(lbl, opts, key=f"{kp}_f_{col}", placeholder=f"All {lbl.lower()}s")
            if sel:
                d = d[d[col].astype(str).isin(sel)]
                scope.append(f"{lbl}: {', '.join(sel[:3])}{' +' + str(len(sel) - 3) if len(sel) > 3 else ''}")
        c1, c2 = st.columns(2)
        with c1:
            only_open = st.toggle("Open appeals only", key=f"{kp}_open_only")
        with c2:
            only_high = st.toggle(f"High risk only (score > {cfg.high_risk_score})", key=f"{kp}_high_only")
        if only_open:
            d = d[d["IsOpen"]]
            scope.append("Open only")
        if only_high:
            d = d[d["IsHighRisk"]]
            scope.append("High risk only")
    return d, scope


def _posture(k: dict, cur: dict | None, prv: dict | None, idx: int, findings: list, cfg: RiskConfig) -> None:
    tier = "Critical" if idx >= 70 else "High" if idx >= 50 else "Medium" if idx >= 30 else "Low"
    col = TIER_COLOR[tier]
    crit = sum(f["severity"] == "Critical" for f in findings)
    high = sum(f["severity"] == "High" for f in findings)
    lead = findings[0]["title"] if findings else "No findings"
    st.markdown(
        f"<div class='rf-index'><div style='text-align:center;min-width:120px'>"
        f"<div class='rf-index-lbl'>Risk index</div><div class='rf-index-score' style='color:{col}'>{idx}</div>"
        f"<div class='rf-index-lbl' style='color:{col}'>{tier}</div></div>"
        f"<div class='rf-index-txt'><b style='color:inherit'>{len(findings)} finding(s)</b> · "
        f"<span style='color:#dc2626'>{crit} critical</span> · <span style='color:#ea580c'>{high} high</span><br>"
        f"Lead issue: {esc(lead)}<br><span style='color:inherit;opacity:.72;font-size:.74rem'>Risk index = volume-weighted composite "
        f"score of the register below (0 = lowest relative risk, 100 = highest).</span></div></div>",
        unsafe_allow_html=True,
    )
    p = prv or {}
    c = cur or {}
    has = bool(cur and prv)
    unit_pp = " pts"
    cards = [
        _card("🔗 Overdue appeals", f"{k['overdue_pct']}%", f"{k['overdue_n']:,} of {k['n']:,} (> {cfg.sla_days}d)",
              _status(k["overdue_pct"], "overdue_pct", cfg),
              _delta_html(c.get("overdue_pct"), p.get("overdue_pct"), True, unit_pp) if has else "", "#ef4444"),
        _card("⏳ Open / pending", f"{k['open_pct']}%", f"{k['open_n']:,} not yet closed",
              _status(k["open_pct"], "open_pct", cfg),
              _delta_html(c.get("open_pct"), p.get("open_pct"), True, unit_pp) if has else "", "#f43f5e"),
        _card(f"🕰️ Open > {cfg.aging_threshold} days", f"{k['aged_open_pct']}%",
              f"{k['aged_open_n']:,} of open · oldest {k['oldest_open']:,}d",
              _status(k["aged_open_pct"], "aged_open_pct", cfg), "", "#06b6d4"),
        _card(f"⚠️ Escalated (L{cfg.escalation_level})", f"{k['escalated_pct']}%", f"{k['escalated_n']:,} appeals",
              _status(k["escalated_pct"], "escalated_pct", cfg),
              _delta_html(c.get("escalated_pct"), p.get("escalated_pct"), True, unit_pp) if has else "", "#f59e0b"),
        _card("📉 Transferred", f"{k['transferred_pct']}%", f"{k['transferred_n']:,} reassignments",
              _status(k["transferred_pct"], "transferred_pct", cfg),
              _delta_html(c.get("transferred_pct"), p.get("transferred_pct"), True, unit_pp) if has else "", "#8b5cf6"),
        _card("⏱️ Avg case age", f"{k['avg_age']}d", f"Max {k['max_age']:,}d",
              _status(k["avg_age"], "avg_age", cfg),
              _delta_html(c.get("avg_age"), p.get("avg_age"), True, "d") if has else "", "#3b82f6"),
        _card("🎯 Avg compliance risk", f"{k['avg_risk']}", f"{k['high_risk_n']:,} above {cfg.high_risk_score}",
              _status(k["avg_risk"], "avg_risk", cfg),
              _delta_html(c.get("avg_risk"), p.get("avg_risk"), True) if has else "", "#6366f1"),
        _card("💱 Member-filed share", f"{k['member_pct']}%" if k["member_pct"] is not None else "n/a",
              f"{k['member_n']:,} member appeals" if k["member_n"] is not None else "Column not available",
              _status(k["member_pct"], "member_pct", cfg), "", "#10b981"),
    ]
    st.markdown("<div class='rf-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)
    if has:
        st.caption(f"Arrows compare appeals received in the last {cfg.trend_window_days} days with the "
                   f"{cfg.trend_window_days} days before (pts = percentage points). Green = improving.")


def _findings_section(findings: list, scope: list, cfg: RiskConfig) -> None:
    kp = cfg.key_prefix
    sevs = [s for s in SEV_ORDER if any(f["severity"] == s for f in findings)]
    cats = sorted({f["category"] for f in findings})
    c1, c2 = st.columns([1, 1])
    with c1:
        sel_s = st.multiselect("Severity", sevs, default=sevs, key=f"{kp}_fs")
    with c2:
        sel_c = st.multiselect("Category", cats, default=cats, key=f"{kp}_fc")
    shown = [f for f in findings if f["severity"] in sel_s and f["category"] in sel_c]
    if not shown:
        st.info("No findings match the selected filters.")
        return
    left, right = st.columns(2, gap="medium")
    for i, f in enumerate(shown):
        sc = SEV_COLOR[f["severity"]]
        kv = "".join(f"<div><b>{esc(k)}</b><span>{esc(v)}</span></div>" for k, v in f["metrics"].items())
        contrib = ("<div style='margin-top:6px'><span class='rf-hint'>Top contributors: </span>"
                   + "".join(f"<span class='rf-chip'>{esc(c)}</span>" for c in f["contributors"]) + "</div>"
                   if f["contributors"] else "")
        subj, body = _finding_mail(f, scope)
        btn = mail_button(mail_url(subj, body), "Draft in Outlook", f"Outlook draft for {f['id']}")
        html_card = (
            f"<div class='rf-find' style='--sc:{sc};--bc:{sc}33'>"
            f"<div class='rf-row'><span class='rf-badge' style='background:{sc}22;color:{sc};border:1px solid {sc}44'>"
            f"{SEV_ICON[f['severity']]} {esc(f['severity'])}</span>"
            f"<span class='rf-hint'>{esc(f['id'])} · {esc(f['category'])}</span></div>"
            f"<div class='rf-find-t'>{esc(f['title'])}</div><div class='rf-find-d'>{esc(f['detail'])}</div>"
            f"<div class='rf-kv'>{kv}</div>{contrib}"
            f"<div class='rf-act'><b style='color:{sc}'>⚡ Action:</b> {esc(f['action'])}<br>"
            f"<span class='rf-chip'>👤 {esc(f['owner'])}</span><span class='rf-chip'>🗓️ {esc(f['timeframe'])}</span>"
            f"<span class='rf-chip'>🎯 {esc(f['success'])}</span></div>"
            f"<div class='rf-foot'><span class='rf-hint'>Add the assignee in To, then send.</span>{btn}</div></div>"
        )
        with (left if i % 2 == 0 else right):
            st.markdown(html_card, unsafe_allow_html=True)


def _register_section(reg: pd.DataFrame, dim_label: str, cfg: RiskConfig, scope: list) -> None:
    kp = cfg.key_prefix
    tiers = ["Critical", "High", "Medium", "Low"]
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        q = st.text_input(f"Search {dim_label.lower()}", key=f"{kp}_reg_q", placeholder="Type to filter…")
    with c2:
        sel_t = st.multiselect("Tier", tiers, default=tiers, key=f"{kp}_reg_t")
    with c3:
        hide_low = st.toggle(f"Hide low volume (< {cfg.min_entity_volume} cases)", value=False, key=f"{kp}_reg_lv")
    view = reg.copy()
    if q:
        view = view[view["Entity"].astype(str).str.contains(q, case=False, na=False)]
    view = view[view["Tier"].str.split(" ").str[0].isin(sel_t)]
    if hide_low:
        view = view[~view["Low Volume"]]
    st.dataframe(
        view[["Entity", "Tier", "Risk Score", "Top Driver", "Cases", "Open", "Overdue", "Overdue %",
              "Share of Overdue %", "Aged Open %", "Escalated %", "Transferred %", "AvgRisk", "AvgAge", "OldestOpen"]],
        hide_index=True, width="stretch", height=min(38 * (len(view) + 1), 460),
        column_config={
            "Entity": st.column_config.TextColumn(dim_label, width="large"),
            "Risk Score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
            "Overdue %": st.column_config.NumberColumn(format="%.1f%%"),
            "Share of Overdue %": st.column_config.NumberColumn("Share of overdue", format="%.1f%%"),
            "Aged Open %": st.column_config.NumberColumn(f"Open > {cfg.aging_threshold}d", format="%.1f%%"),
            "Escalated %": st.column_config.NumberColumn(format="%.1f%%"),
            "Transferred %": st.column_config.NumberColumn(format="%.1f%%"),
            "AvgRisk": st.column_config.ProgressColumn("Avg risk", min_value=0, max_value=1, format="%.3f"),
            "AvgAge": st.column_config.NumberColumn("Avg age", format="%.1f d"),
            "OldestOpen": st.column_config.NumberColumn("Oldest open", format="%d d"),
        },
    )
    st.caption("Risk score = weighted percentile rank on overdue rate (35%), avg risk (20%), open-aging rate (20%), "
               "escalation rate (15%) and transfer rate (10%), compared across "
               f"{dim_label.lower()}s with at least {cfg.min_entity_volume} cases.")
    if view.empty:
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        pick = st.selectbox(f"Email a {dim_label.lower()} risk review", view["Entity"].astype(str).tolist(), key=f"{kp}_reg_pick")
    row = view[view["Entity"].astype(str) == pick].iloc[0]
    subj, body = _entity_mail(row, dim_label, cfg, scope)
    with c2:
        st.markdown("<div style='height:28px'></div>" + mail_button(mail_url(subj, body), "Draft in Outlook",
                    f"Outlook draft for {pick}"), unsafe_allow_html=True)


def _analytics_section(d: pd.DataFrame, reg: pd.DataFrame, dim_label: str, dim_col: str,
                       date_col: str | None, dims: dict, cfg: RiskConfig) -> None:
    kp = cfg.key_prefix
    tabs = st.tabs(["🔥 Heatmap", "📊 Pareto", "🎯 Risk matrix", "🕰️ Aging", "📈 Trend", "🗺️ Site radar", "🧩 Task mix"])

    top = reg[~reg["Low Volume"]].head(15) if (~reg["Low Volume"]).sum() >= 3 else reg.head(15)

    with tabs[0]:
        metrics = ["Overdue %", "Aged Open %", "Escalated %", "Transferred %", "AvgRisk"]
        z = top[metrics].copy()
        zs = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
        fig = go.Figure(go.Heatmap(
            z=zs.values, x=["Overdue %", f"Open > {cfg.aging_threshold}d %", "Escalated %", "Transferred %", "Avg risk"],
            y=top["Entity"].astype(str), colorscale=[[0, "#0f2744"], [0.5, "#1e3a8a"], [1, "#ef4444"]],
            text=z.round(2).values, texttemplate="%{text}", zmid=0,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>", colorbar=dict(title="vs avg (σ)"),
        ))
        fig.update_layout(title=f"Risk drivers by {dim_label.lower()} — colour = distance from average", yaxis=dict(autorange="reversed"))
        st.plotly_chart(_style(fig, max(360, 40 + 30 * len(top))), width="stretch", key=f"{kp}_heat")

    with tabs[1]:
        p = reg[reg["Overdue"] > 0].sort_values("Overdue", ascending=False).head(20)
        if p.empty:
            st.info("No overdue appeals in scope.")
        else:
            cum = p["Overdue"].cumsum() / reg["Overdue"].sum() * 100
            fig = go.Figure()
            fig.add_bar(x=p["Entity"].astype(str), y=p["Overdue"], name="Overdue", marker_color="#ef4444",
                        text=p["Overdue"], textposition="outside")
            fig.add_scatter(x=p["Entity"].astype(str), y=cum, name="Cumulative %", yaxis="y2", mode="lines+markers",
                            line=dict(color="#f59e0b", width=3))
            fig.add_hline(y=80, line_dash="dot", line_color="#94a3b8", yref="y2",
                          annotation_text="80%", annotation_font_color="#94a3b8")
            n80 = int((cum <= 80).sum()) + 1
            fig.update_layout(title=f"Overdue Pareto — {min(n80, len(p))} {dim_label.lower()}(s) drive ~80% of overdue",
                              yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%", showgrid=False),
                              legend=dict(orientation="h", y=1.1), xaxis=dict(tickangle=-30))
            st.plotly_chart(_style(fig, 440), width="stretch", key=f"{kp}_pareto")

    with tabs[2]:
        m = reg.copy()
        mx, my = m["Cases"].median(), m["Overdue %"].median()
        fig = px.scatter(m, x="Cases", y="Overdue %", size=np.clip(m["AvgRisk"].fillna(0) * 60 + 6, 6, 60),
                         color="Tier", hover_name="Entity", log_x=bool(m["Cases"].max() / max(m["Cases"].min(), 1) > 50),
                         color_discrete_map={**TIER_COLOR, **{f"{t} (low vol.)": "#64748b" for t in TIER_COLOR}},
                         hover_data={"Risk Score": True, "Overdue": True, "AvgRisk": ":.3f"})
        fig.add_vline(x=mx, line_dash="dot", line_color="#475569")
        fig.add_hline(y=my, line_dash="dot", line_color="#475569")
        fig.add_annotation(x=1, y=1, xref="paper", yref="paper", text="High volume · high overdue → act first",
                           showarrow=False, font=dict(color="#ef4444", size=11), xanchor="right")
        fig.update_layout(title="Risk matrix — volume vs overdue rate (bubble = avg risk score)",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(_style(fig, 460), width="stretch", key=f"{kp}_matrix")

    with tabs[3]:
        op = d[d["IsOpen"] & d["CaseAgeDays"].notna()].copy()
        if op.empty:
            st.info("No open appeals in scope.")
        else:
            a = cfg.aging_threshold
            bins = sorted({-1, 15, cfg.sla_days, a, 90, 10 ** 6})
            labels = [f"0–{bins[1]}d"] + [f"{bins[i] + 1}–{bins[i + 1]}d" for i in range(1, len(bins) - 2)] + [f">{bins[-2]}d"]
            op["Bucket"] = pd.cut(op["CaseAgeDays"], bins=bins, labels=labels)
            top_e = op[dim_col].value_counts().head(10).index
            op["Group"] = np.where(op[dim_col].isin(top_e), op[dim_col].astype(str), "Other")
            ct = op.groupby(["Group", "Bucket"], observed=False).size().reset_index(name="Open")
            fig = px.bar(ct, x="Open", y="Group", color="Bucket", orientation="h",
                         color_discrete_sequence=["#10b981", "#84cc16", "#f59e0b", "#f97316", "#ef4444", "#991b1b"],
                         category_orders={"Bucket": labels})
            fig.update_layout(title=f"Open appeals by age bucket — top {dim_label.lower()}s", barmode="stack",
                              legend=dict(orientation="h", y=1.1), yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(_style(fig, 440), width="stretch", key=f"{kp}_aging")

    with tabs[4]:
        if not date_col:
            st.info("A received-date column is needed for trends.")
        else:
            gran = st.radio("Granularity", ["Week", "Month"], horizontal=True, key=f"{kp}_gran")
            freq = "W-MON" if gran == "Week" else "MS"
            t = d.dropna(subset=[date_col]).set_index(date_col).groupby(pd.Grouper(freq=freq)).agg(
                Received=(cfg.case_id, "count"), Overdue=("IsOverdue", "sum"),
                Escalated=("IsEscalated", "sum"), Open=("IsOpen", "sum")).reset_index()
            t = t[t["Received"] > 0]
            if len(t) < 2:
                st.info("Not enough periods for a trend.")
            else:
                t["Overdue %"] = (t["Overdue"] / t["Received"] * 100).round(1)
                t["Escalated %"] = (t["Escalated"] / t["Received"] * 100).round(1)
                fig = go.Figure()
                fig.add_bar(x=t[date_col], y=t["Received"], name="Received", marker_color="rgba(37,99,235,.45)")
                fig.add_scatter(x=t[date_col], y=t["Overdue %"], name="Overdue %", yaxis="y2", mode="lines+markers",
                                line=dict(color="#ef4444", width=3))
                fig.add_scatter(x=t[date_col], y=t["Escalated %"], name="Escalated %", yaxis="y2", mode="lines+markers",
                                line=dict(color="#f59e0b", width=2, dash="dot"))
                fig.add_hline(y=cfg.thresholds["overdue_pct"][1], yref="y2", line_dash="dash", line_color="#991b1b",
                              annotation_text="Overdue critical", annotation_font_color="#ef4444")
                fig.update_layout(title=f"Received volume and risk rates by {gran.lower()} (by received date)",
                                  yaxis=dict(title="Received"), hovermode="x unified",
                                  yaxis2=dict(overlaying="y", side="right", ticksuffix="%", showgrid=False),
                                  legend=dict(orientation="h", y=1.12))
                st.plotly_chart(_style(fig, 440), width="stretch", key=f"{kp}_trend")
                st.caption("The most recent period may be incomplete and recent cases have had less time to become overdue.")

    with tabs[5]:
        site_col = dims.get("Site")
        if not site_col:
            st.info("Site column not available.")
        else:
            sd = d.groupby(site_col).agg(o=("IsOverdue", "mean"), e=("IsEscalated", "mean"), t=("IsTransferred", "mean"),
                                         a=("IsAgedOpen", "mean"), r=("ComplianceRiskScore", "mean"),
                                         n=(cfg.case_id, "count")).reset_index()
            sd = sd.nlargest(8, "n")
            cats = ["Overdue %", "Escalated %", "Transferred %", "Aged open %", "Risk ×100"]
            fig = go.Figure()
            for i, r in sd.iterrows():
                vals = [r.o * 100, r.e * 100, r.t * 100, r.a * 100, (r.r if pd.notna(r.r) else 0) * 100]
                clr = SEQ[i % len(SEQ)]
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                                              name=str(r[site_col]), line=dict(color=clr, width=2), opacity=.35))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                         radialaxis=dict(visible=True, gridcolor="rgba(99,102,241,.15)", ticksuffix="%"),
                                         angularaxis=dict(gridcolor="rgba(99,102,241,.15)")),
                              title="Site risk profile (all axes are percentages)",
                              legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(_style(fig, 460), width="stretch", key=f"{kp}_radar")

    with tabs[6]:
        task_col = dims.get("Task")
        if not task_col:
            st.info("Task column not available.")
        else:
            ca = d.groupby(task_col).agg(Count=(cfg.case_id, "count"), Overdue=("IsOverdue", "mean"),
                                         Risk=("ComplianceRiskScore", "mean")).reset_index().sort_values("Count", ascending=False)
            ca["Overdue %"] = (ca["Overdue"] * 100).round(1)
            fig = px.bar(ca, x=task_col, y="Count", color="Overdue %", text="Count",
                         color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                         hover_data={"Risk": ":.3f", "Overdue %": True})
            fig.update_layout(title="Volume by work-item task (colour = overdue rate)", xaxis=dict(tickangle=-20))
            fig.update_traces(textposition="outside")
            st.plotly_chart(_style(fig, 420), width="stretch", key=f"{kp}_task")


def _watchlist_section(d: pd.DataFrame, dims: dict, cfg: RiskConfig, scope: list) -> pd.DataFrame:
    kp = cfg.key_prefix
    c1, c2 = st.columns([1.3, 1])
    with c1:
        mode = st.radio("List", ["Oldest open", "Highest risk (open)", "Overdue"], horizontal=True, key=f"{kp}_wl_mode")
    with c2:
        size = st.slider("Rows", 10, 200, cfg.watchlist_size, 10, key=f"{kp}_wl_n")
    if mode == "Oldest open":
        w = d[d["IsOpen"]].sort_values("CaseAgeDays", ascending=False)
    elif mode == "Highest risk (open)":
        w = d[d["IsOpen"]].sort_values(["ComplianceRiskScore", "CaseAgeDays"], ascending=False)
    else:
        w = d[d["IsOverdue"]].sort_values("CaseAgeDays", ascending=False)
    cols = [cfg.case_id] + list(dims.values())
    date_col = next((c for c in cfg.date_column_candidates if c in d), None)
    if date_col:
        cols.append(date_col)
    cols += ["CaseAgeDays", "ComplianceRiskScore", "IsOverdue", "IsEscalated", "IsTransferred"]
    cols = [c for c in dict.fromkeys(cols) if c in w]
    w = w[cols].head(size)
    if w.empty:
        st.info("No cases match this list.")
        return w
    st.dataframe(w, hide_index=True, width="stretch", height=min(36 * (len(w) + 1), 420),
                 column_config={"CaseAgeDays": st.column_config.NumberColumn("Age", format="%d d"),
                                "ComplianceRiskScore": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.3f"),
                                "IsOverdue": st.column_config.CheckboxColumn("Overdue"),
                                "IsEscalated": st.column_config.CheckboxColumn("Escalated"),
                                "IsTransferred": st.column_config.CheckboxColumn("Transferred")})
    ids = w[cfg.case_id].astype(str).head(20).tolist()
    subj = f"[DataTalk.AI] Case watchlist — {mode} ({len(w)} cases)"
    body = "\n".join(["Hi,", "", f"Please review the following {mode.lower()} appeals flagged in DataTalk.AI.", "",
                      f"Scope: {_scope_text(scope)}", f"Cases listed: {len(w)} (first {len(ids)} shown below; full list attached as CSV)", ""]
                     + [f"- {cid} · {int(a) if pd.notna(a) else '?'} days · risk {r:.3f}" if pd.notna(r) else f"- {cid} · {int(a) if pd.notna(a) else '?'} days"
                        for cid, a, r in zip(ids, w["CaseAgeDays"].head(20), w["ComplianceRiskScore"].head(20))]
                     + ["", "Please confirm owners and expected closure dates by reply.", "", "Thanks,"])
    st.markdown("<div class='rf-bar'><div><div class='rf-bar-t'>Share this watchlist</div>"
                "<div class='rf-bar-s'>Download the CSV, then attach it to the Outlook draft</div></div>"
                + mail_button(mail_url(subj, body), "Draft in Outlook", "Outlook draft with the case list", "secondary")
                + "</div>", unsafe_allow_html=True)
    st.download_button("📥 Download watchlist (CSV)", w.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"Risk_Watchlist_{mode.replace(' ', '_')}.csv", mime="text/csv", key=f"{kp}_wl_dl")
    return w


def _anomaly_section(d: pd.DataFrame, detect_fn: Callable | None, cfg: RiskConfig, scope: list,
                     risk_emoji: dict | None) -> None:
    kp = cfg.key_prefix
    if detect_fn is None:
        st.markdown("<div class='rf-empty'>Anomaly detection is not connected. Pass <code>detect_anomalies_fn</code> "
                    "to <code>render_risk_findings_tab</code>.</div>", unsafe_allow_html=True)
        return
    sig = (len(d), int(d["IsOverdue"].sum()), int(d["IsOpen"].sum()))
    if st.button("🔍 Run anomaly detection scan", key=f"{kp}_anom_btn"):
        with st.spinner("Running statistical anomaly detection…"):
            try:
                st.session_state[f"{kp}_anoms"] = (sig, detect_fn(d) or [])
            except Exception as e:  # never break the tab
                st.error(f"Anomaly scan failed: {e}")
                return
    saved = st.session_state.get(f"{kp}_anoms")
    if saved is None:
        st.markdown("<div class='rf-empty'>Click <b style='color:var(--dt-accent)'>Run anomaly detection scan</b> to check overdue "
                    "rates, escalations, transfer spikes, risk-score outliers and aging concentration in the current scope."
                    "</div>", unsafe_allow_html=True)
        return
    old_sig, anomalies = saved
    if old_sig != sig:
        st.warning("Filters changed since the last scan — run the scan again to refresh results.")
    if not anomalies:
        st.success("✅ No statistical anomalies detected in the current scope.")
        return
    counts = {}
    for a in anomalies:
        counts[a.get("severity", "Medium")] = counts.get(a.get("severity", "Medium"), 0) + 1
    chips = "".join(
        f"<div style='flex:1;min-width:110px;text-align:center;background:var(--dt-surface);border:1px solid "
        f"{SEV_COLOR.get(s, '#3b82f6')}44;border-radius:10px;padding:10px'><div class='rf-l'>{esc(s)}</div>"
        f"<div style='font-size:1.6rem;font-weight:800;color:{SEV_COLOR.get(s, '#94a3b8')}'>{c}</div></div>"
        for s, c in sorted(counts.items(), key=lambda kv: SEV_ORDER.get(kv[0], 9)))
    st.markdown(f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px'>{chips}</div>", unsafe_allow_html=True)
    cats = sorted({a.get("category", "") for a in anomalies})
    sel = st.multiselect("Category", cats, default=cats, key=f"{kp}_anom_cat")
    shown = sorted([a for a in anomalies if a.get("category", "") in sel],
                   key=lambda a: SEV_ORDER.get(a.get("severity", "Medium"), 9))
    emo = risk_emoji or SEV_ICON
    for _i, a in enumerate(shown):
        sev = a.get("severity", "Medium")
        sc = SEV_COLOR.get(sev, "#94a3b8")
        conf = round(float(a.get("confidence", 0) or 0) * 100)
        with st.expander(f"{emo.get(sev, '⚪')} {a.get('detected', 'Anomaly')} — {sev} ({conf}% confidence)",
                         expanded=(sev == "Critical")):
            c1, c2 = st.columns([1.6, 1])
            with c1:
                parts = [("Category", a.get("category", "")), ("Root cause", a.get("root_cause", "")),
                         ("Financial risk", a.get("financial_risk", "")), ("Historical context", a.get("historical_comparison", ""))]
                st.markdown("".join(f"<div class='rf-find-d' style='margin-bottom:8px'><b style='color:inherit'>{esc(k)}:</b> "
                                    f"{esc(v)}</div>" for k, v in parts if v), unsafe_allow_html=True)
                if a.get("recommended_action"):
                    st.markdown(f"<div class='rf-act'><b style='color:{sc}'>⚡ Recommended action:</b> "
                                f"{esc(a['recommended_action'])}</div>", unsafe_allow_html=True)
            with c2:
                sup = a.get("supporting_metrics") or {}
                if sup:
                    st.markdown("<div class='rf-kv'>" + "".join(f"<div><b>{esc(k)}</b><span>{esc(v)}</span></div>"
                                                               for k, v in sup.items()) + "</div>", unsafe_allow_html=True)
            subj = f"[DataTalk.AI] {sev} anomaly: {short(a.get('detected', 'Anomaly'), 90)}"
            body = "\n".join(["Hi,", "", "DataTalk.AI detected the following anomaly in the appeals data.", "",
                              f"ANOMALY ({sev}, {conf}% confidence)", str(a.get("detected", "")), "",
                              f"Category: {a.get('category', '')}", f"Root cause hypothesis: {a.get('root_cause', '')}"]
                             + [f"- {k}: {v}" for k, v in sup.items()]
                             + ["", "RECOMMENDED ACTION", str(a.get("recommended_action", "")), "",
                                f"Scope: {_scope_text(scope)}", "", "Please investigate and confirm by reply.", "", "Thanks,"])
            st.markdown("<div class='rf-foot'><span class='rf-hint'>Add the assignee in To, then send.</span>"
                        + mail_button(mail_url(subj, body), "Draft in Outlook") + "</div>", unsafe_allow_html=True)
    exp = pd.DataFrame([{"ID": a.get("id"), "Detected": a.get("detected"), "Category": a.get("category"),
                         "Severity": a.get("severity"), "Confidence": a.get("confidence"),
                         "Root Cause": a.get("root_cause"), "Recommended Action": a.get("recommended_action")} for a in shown])
    st.download_button("📥 Download anomaly report (CSV)", exp.to_csv(index=False).encode("utf-8-sig"),
                       file_name="Anomaly_Report.csv", mime="text/csv", key=f"{kp}_anom_dl")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def render_risk_findings_tab(df: pd.DataFrame, cfg: RiskConfig | None = None,
                             detect_anomalies_fn: Callable[[pd.DataFrame], list] | None = None,
                             risk_emoji: dict | None = None) -> None:
    cfg = cfg or RiskConfig()
    kp = cfg.key_prefix
    st.markdown(CSS + MAIL_CSS, unsafe_allow_html=True)

    if df is None or df.empty:
        st.info("No appeals data is loaded.")
        return
    base, date_col, missing = _prepare(df, cfg)
    dims = _dims(base, cfg)

    # header
    period = (f"{base[date_col].min():%d %b %Y} – {base[date_col].max():%d %b %Y}" if date_col else "All dates")
    st.markdown(f"<div class='rf-head'><div><div class='rf-title'>🛡️ Risk & Findings</div>"
                f"<div class='rf-sub'>{len(base):,} appeals · {esc(period)} · SLA {cfg.sla_days} days · "
                f"aging threshold {cfg.aging_threshold} days</div></div></div>", unsafe_allow_html=True)
    if missing:
        st.caption(f"Columns not found and treated as empty: {', '.join(missing)}.")

    d, scope = _scope_bar(base, date_col, dims, cfg)
    if d.empty:
        st.warning("No appeals match the selected filters.")
        return
    if scope:
        st.markdown("".join(f"<span class='rf-chip'>🔎 {esc(s)}</span>" for s in scope), unsafe_allow_html=True)

    # analyse-by dimension
    if not dims:
        st.warning("No grouping columns (sponsor, unit, site, task) were found.")
        return
    labels = list(dims)
    c1, _ = st.columns([1, 3])
    with c1:
        dim_label = st.selectbox("Analyse risk by", labels, key=f"{kp}_dim")
    dim_col = dims[dim_label]

    k = _kpis(d)
    cur, prv = _window_kpis(d, date_col, cfg)
    reg = build_register(d, dim_col, cfg)
    idx = portfolio_index(reg)
    # trend rule compares the last window with the one before (current-window KPIs vs prior-window KPIs)
    findings = build_findings(d, k, cfg, reg, dim_label, prev=None, trend=(cur, prv))

    # 1 posture
    _sec("Risk posture")
    _posture(k, cur, prv, idx, findings, cfg)

    # summary mail bar
    subj, body = _summary_mail(k, findings, reg, dim_label, idx, scope, cfg)
    st.markdown("<div class='rf-bar'><div><div class='rf-bar-t'>Share the risk summary</div>"
                f"<div class='rf-bar-s'>Index, headline KPIs, {len(findings)} finding(s) and top {dim_label.lower()} risks "
                f"in one Outlook draft</div></div>"
                + mail_button(mail_url(subj, body), "Email risk summary", "Opens one Outlook draft", "secondary")
                + "</div>", unsafe_allow_html=True)

    # sections as tabs
    t1, t2, t3, t4, t5 = st.tabs([f"🚨 Findings ({len(findings)})", f"📋 Risk register ({len(reg)})",
                                  "📊 Risk analytics", "🗂️ Case watchlist", "🤖 Anomaly intelligence"])
    with t1:
        _findings_section(findings, scope, cfg)
        fx = pd.DataFrame([{"ID": f["id"], "Severity": f["severity"], "Category": f["category"], "Finding": f["title"],
                            "Detail": f["detail"], "Action": f["action"], "Owner": f["owner"], "Timeframe": f["timeframe"],
                            "Success measure": f["success"], "Top contributors": "; ".join(f["contributors"])}
                           for f in findings])
        st.download_button("📥 Download findings (CSV)", fx.to_csv(index=False).encode("utf-8-sig"),
                           file_name="Risk_Findings.csv", mime="text/csv", key=f"{kp}_find_dl")
    with t2:
        _register_section(reg, dim_label, cfg, scope)
        st.download_button("📥 Download risk register (CSV)", reg.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"Risk_Register_{dim_label.replace(' ', '_')}.csv", mime="text/csv",
                           key=f"{kp}_reg_dl")
    with t3:
        _analytics_section(d, reg, dim_label, dim_col, date_col, dims, cfg)
    with t4:
        _watchlist_section(d, dims, cfg, scope)
    with t5:
        _anomaly_section(d, detect_anomalies_fn, cfg, scope, risk_emoji)


__all__ = ["RiskConfig", "render_risk_findings_tab", "build_register", "build_findings", "portfolio_index"]
