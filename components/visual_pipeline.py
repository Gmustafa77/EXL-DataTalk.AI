"""Deterministic, defensive visualization pipeline (fallback charts, no LLM calls).

    Raw query result
      → 1. Data validation          (empty, all-null, duplicate columns, ±inf)
      → 2. Missing-value handling   (NaN / "None" / "" never become categories)
      → 3. Type detection           (time · dimension · measure · id · text;
                                     percent · currency · duration · count · score · number)
      → 4. Eligibility check        (rows, distinct values, negatives, zero/constant data)
      → 5. Chart recommendation     (KPI · line · bar · grouped bar · donut · scatter · table)
      → 6. Chart preparation        (sorting, top-N, "Other" only for additive measures,
                                     label truncation, unit scaling)
      → 7. Rendering                (professional number formats, hover text, axis formats)
      → 8. Output validation        (no NaN / inf in any plotted array)
      → 9. Fallback                 (formatted table figure + explanatory note)

Visualization only. It does not touch SQL, the engine pipeline, insights, prompts
or the database. No Streamlit imports.

Public API
    build_visuals(df, question="", cfg=None) -> VizResult(figures, notes, chart_type, used_fallback)
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.plot_style import FONT_FAMILY, GRID, PALETTE

log = logging.getLogger("datatalk.visual_pipeline")

PIPELINE_VERSION = "visual-pipeline-v1.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class VizConfig:
    max_charts: int = 2
    max_bar_categories: int = 15          # beyond this → top-N (+ "Other" for additive measures)
    max_pie_slices: int = 6
    min_pie_slices: int = 2
    max_line_series: int = 6
    max_grouped_series: int = 6
    min_periods_line: int = 3
    min_rows_scatter: int = 10
    max_kpis: int = 4
    max_label_len: int = 28
    max_table_rows: int = 200
    height: int = 440
    table_fallback_figure: bool = True    # render a formatted table figure when no chart is reliable
    color_sequence: list = field(
        default_factory=lambda: list(PALETTE)
    )
    style_fn: Callable | None = None   # applied last to every figure (core.plot_style.style_figure)


@dataclass
class ColumnProfile:
    name: str
    role: str            # time | dimension | measure | id | text
    kind: str = ""       # measures: percent | currency | duration | count | score | number
    scale: float = 1.0   # multiply raw value before display (0–1 rates → 100)
    decimals: int = 0
    additive: bool = False
    n_unique: int = 0
    null_pct: float = 0.0


@dataclass
class VizResult:
    figures: list
    notes: list
    chart_type: str
    used_fallback: bool
    profiles: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════
_TIME_WORDS = {"date", "month", "year", "quarter", "week", "period", "day", "yearmonth", "time", "dt"}
_ID_WORDS = {"id", "identifier", "key", "uuid", "guid", "number", "no", "code"}
_PCT_WORDS = {"rate", "percent", "percentage", "pct", "share", "ratio"}
_CUR_WORDS = {"amount", "cost", "paid", "price", "revenue", "charge", "charges", "billed",
              "allowed", "usd", "dollar", "dollars", "spend", "payment", "payments", "savings"}
_DUR_WORDS = {"days", "day", "hours", "hrs", "minutes", "mins", "tat", "turnaround", "age", "aging", "duration"}
_CNT_WORDS = {"count", "cases", "total", "volume", "num", "appeals", "records", "claims", "items", "n"}
_AVG_WORDS = {"avg", "average", "mean", "median"}
_SCORE_WORDS = {"score", "index"}
_SHARE_Q = re.compile(r"\b(share|distribution|breakdown|mix|proportion|split|composition|percentage of)\b", re.I)
_NULL_TOKENS = {"", "nan", "none", "null", "nat", "n/a", "na", "<na>", "-"}
_PERIOD_RE = re.compile(r"^\s*\d{4}(-\d{1,2}(-\d{1,2})?|[-\s]?Q[1-4])\s*$", re.I)


def _words(name: Any) -> set:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return set(re.findall(r"[a-z]+", s.lower()))


def pretty(name: Any) -> str:
    s = str(name).strip()
    if "_" in s or s.islower():
        s = s.replace("_", " ").title()
    return s


def truncate(label: Any, n: int) -> str:
    s = re.sub(r"\s+", " ", str(label)).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ═══════════════════════════════════════════════════════════════════════════════
# 1–2. VALIDATION + MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════
def _is_null_token(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(v, str) and v.strip().lower() in _NULL_TOKENS


def _coerce_numeric_text(s: pd.Series) -> tuple[pd.Series, str]:
    """Convert text like '1,234', '45%', '$1,200.50', '(300)' to numbers when ≥90% parse.
    Returns (series, hint) where hint is 'percent' | 'currency' | ''."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s, ""
    txt = s.astype("string").str.strip()
    nonnull = txt[~txt.map(_is_null_token)]
    if nonnull.empty:
        return s, ""
    has_pct = nonnull.str.endswith("%").mean() >= 0.9
    has_cur = nonnull.str.match(r"^\(?-?\s*[$€£₹]").mean() >= 0.9
    cleaned = (nonnull.str.replace(r"[,$€£₹%\s]", "", regex=True)
               .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
    num = pd.to_numeric(cleaned, errors="coerce")
    if num.notna().mean() < 0.9:
        return s, ""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[num.index] = num.astype("float64")
    return out, ("percent" if has_pct else "currency" if has_cur else "")


def validate_frame(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list, dict]:
    notes: list = []
    hints: dict = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, ["The query returned no rows, so there is nothing to chart."], hints
    d = df.copy()
    d.columns = [str(c).strip() or f"Column {i + 1}" for i, c in enumerate(d.columns)]
    if d.columns.duplicated().any():
        d = d.loc[:, ~d.columns.duplicated()]
        notes.append("Duplicate column names were removed before charting.")
    for c in d.columns:
        d[c], hint = _coerce_numeric_text(d[c])
        if hint:
            hints[c] = hint
        if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        elif not pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].map(lambda v: np.nan if _is_null_token(v) else v)
    empty = [c for c in d.columns if d[c].isna().all()]
    if empty:
        d = d.drop(columns=empty)
        notes.append(f"Column(s) with no values were ignored: {', '.join(empty[:4])}.")
    if d.empty or d.shape[1] == 0:
        return None, notes + ["The result has no usable values to chart."], hints
    return d.reset_index(drop=True), notes, hints


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def _to_time(s: pd.Series, name: str) -> pd.Series | None:
    w = _words(name)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.dropna()
        if not v.empty and "year" in w and not (w & (_AVG_WORDS | _PCT_WORDS | _CNT_WORDS)) \
                and v.between(1900, 2100).all() and np.allclose(v, v.round()):
            return pd.to_datetime(s.round().astype("Int64").astype("string"), format="%Y", errors="coerce")
        return None
    sample = s.dropna().astype(str).str.strip()
    if sample.empty:
        return None
    if sample.head(50).map(lambda x: bool(_PERIOD_RE.match(x))).mean() >= 0.8:
        q = sample.str.extract(r"^(\d{4})[-\s]?Q([1-4])$", flags=re.I)
        if q.notna().all(axis=1).mean() >= 0.8:
            months = ((q[1].astype(float) - 1) * 3 + 1).astype("Int64").astype("string")
            parsed = pd.to_datetime(q[0] + "-" + months + "-01", errors="coerce")
        else:
            parsed = pd.to_datetime(sample, errors="coerce")
        out = pd.Series(pd.NaT, index=s.index)
        out.loc[parsed.index] = parsed
        return pd.to_datetime(out, errors="coerce")
    if w & _TIME_WORDS or sample.head(50).str.match(r"^\d{4}-\d{2}-\d{2}").mean() >= 0.8:
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            out = pd.Series(pd.NaT, index=s.index)
            out.loc[parsed.index] = parsed
            return pd.to_datetime(out, errors="coerce")
    return None


def _measure_profile(name: str, s: pd.Series, hint: str) -> ColumnProfile:
    w = _words(name)
    v = s.dropna().astype(float)
    integer = (not v.empty) and bool(np.allclose(v, v.round()))
    nonneg = (not v.empty) and bool(v.min() >= 0)
    p = ColumnProfile(name, "measure", n_unique=int(s.nunique()), null_pct=float(s.isna().mean()))
    if hint == "percent" or "%" in name or (w & _PCT_WORDS):
        p.kind = "percent"
        p.scale = 100.0 if (hint != "percent" and not v.empty and v.between(0, 1).all() and v.max() <= 1) else 1.0
        p.decimals = 1
        if not v.empty and (v * p.scale).abs().max() > 1000:   # not really a percentage
            p.kind, p.scale, p.decimals = "number", 1.0, (0 if integer else 2)
    elif hint == "currency" or (w & _CUR_WORDS) or "$" in name:
        p.kind, p.decimals = "currency", (0 if integer or (not v.empty and v.abs().median() >= 100) else 2)
        p.additive = not (w & _AVG_WORDS)
    elif w & _DUR_WORDS:
        p.kind, p.decimals = "duration", (0 if integer else 1)
        p.additive = bool(w & {"total", "sum"}) and not (w & _AVG_WORDS)
    elif w & _SCORE_WORDS:
        p.kind, p.decimals = "score", (0 if integer else 2)
    elif (w & _CNT_WORDS and not (w & _AVG_WORDS)) or (integer and nonneg and not (w & _AVG_WORDS)):
        p.kind, p.decimals, p.additive = "count", 0, True
    else:
        p.kind, p.decimals = "number", (0 if integer else 2)
    return p


def detect_types(d: pd.DataFrame, hints: dict) -> tuple[pd.DataFrame, dict]:
    profiles: dict = {}
    n = len(d)
    out = d.copy()
    time_found = False
    for c in d.columns:
        s = d[c]
        w = _words(c)
        if not time_found:
            t = _to_time(s, c)
            if t is not None and t.notna().sum() >= max(1, int(0.8 * s.notna().sum())):
                out[c] = t
                profiles[c] = ColumnProfile(c, "time", n_unique=int(t.nunique()), null_pct=float(t.isna().mean()))
                time_found = True
                continue
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.map({True: "Yes", False: "No"})
            profiles[c] = ColumnProfile(c, "dimension", n_unique=int(s.nunique()))
            continue
        if pd.api.types.is_numeric_dtype(s):
            uniq = int(s.nunique())
            looks_id = bool(w & _ID_WORDS) and not (w & (_CNT_WORDS | _PCT_WORDS | _CUR_WORDS | _AVG_WORDS)) \
                and n > 1 and uniq >= 0.9 * s.notna().sum()
            if looks_id:
                out[c] = s.astype("Int64").astype("string")
                profiles[c] = ColumnProfile(c, "id", n_unique=uniq)
            else:
                profiles[c] = _measure_profile(c, s, hints.get(c, ""))
            continue
        txt = s.dropna().astype(str)
        uniq = int(txt.nunique())
        avg_len = float(txt.str.len().mean()) if not txt.empty else 0
        if (w & _ID_WORDS) and n > 1 and uniq >= 0.9 * len(txt):
            role = "id"
        elif avg_len > 60 and uniq >= 0.8 * max(len(txt), 1):
            role = "text"
        else:
            role = "dimension"
        out[c] = s.where(s.isna(), s.astype(str).str.strip())
        profiles[c] = ColumnProfile(c, role, n_unique=uniq, null_pct=float(s.isna().mean()))
    return out, profiles


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_value(v: Any, p: ColumnProfile | None) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—" if _is_null_token(v) else str(v)
    if math.isnan(x) or math.isinf(x):
        return "—"
    if p is None:
        return f"{x:,.2f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else f"{x:,.0f}"
    x *= p.scale
    d = p.decimals
    if p.kind == "percent":
        return f"{x:,.{d}f}%"
    if p.kind == "currency":
        if abs(x) >= 1e6:
            return f"${x / 1e6:,.2f}M"
        return f"${x:,.{d}f}"
    if p.kind == "duration":
        return f"{x:,.{d}f} days" if "hour" not in p.name.lower() else f"{x:,.{d}f} h"
    return f"{x:,.{d}f}"


def _axis_fmt(p: ColumnProfile) -> dict:
    if p.kind == "percent":
        return {"ticksuffix": "%", "tickformat": ",.0f" if p.decimals == 0 else ",.1f"}
    if p.kind == "currency":
        return {"tickprefix": "$", "tickformat": "~s"}
    if p.kind == "count":
        return {"tickformat": "~s"}
    return {"tickformat": ",.2~f" if p.decimals else ",.0f"}


def _hover_num(p: ColumnProfile, ref: str) -> str:
    d = p.decimals
    if p.kind == "percent":
        return f"%{{{ref}:,.{d}f}}%"
    if p.kind == "currency":
        return f"$%{{{ref}:,.{d}f}}"
    if p.kind == "duration":
        return f"%{{{ref}:,.{d}f}} days"
    return f"%{{{ref}:,.{d}f}}"


def _unique_labels(labels: list, n: int) -> list:
    out, seen = [], {}
    for lab in labels:
        t = truncate(lab, n)
        if t in seen:
            seen[t] += 1
            t = truncate(lab, n - 3) + f" ({seen[t]})"
        else:
            seen[t] = 1
        out.append(t)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4–5. ELIGIBILITY + RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
def _pick_measure(measures: list, profiles: dict, question: str) -> str | None:
    if not measures:
        return None
    qw = _words(question)
    generic = {"avg", "average", "total", "count", "number", "rate", "of", "the", "by"}
    scored = sorted(((len((_words(m) - generic) & qw), -i, m) for i, m in enumerate(measures)), reverse=True)
    if scored[0][0] > 0:
        return scored[0][2]
    non_count = [m for m in measures if profiles[m].kind != "count"]
    return (non_count or measures)[0]


def _ordered_categories(values: pd.Series) -> bool:
    """True when category labels carry their own order (e.g. '0-15d', '16-30d', 'Q1'…)."""
    v = values.astype(str)
    return bool(v.str.match(r"^\s*[<>]?\s*\d").mean() >= 0.8)


def recommend(d: pd.DataFrame, profiles: dict, question: str, cfg: VizConfig) -> dict:
    time_cols = [c for c, p in profiles.items() if p.role == "time"]
    dims = [c for c, p in profiles.items() if p.role == "dimension"]
    measures = [c for c, p in profiles.items() if p.role == "measure"]
    ids = [c for c, p in profiles.items() if p.role == "id"]
    plan = {"type": "table", "reason": "", "time": time_cols[0] if time_cols else None,
            "dims": dims, "measures": measures, "ids": ids}

    usable = [m for m in measures if d[m].notna().any()]
    if not usable:
        plan["reason"] = "The result has no numeric measure to plot."
        return plan
    m = _pick_measure(usable, profiles, question)
    plan["measure"] = m
    if len(d) == 1:
        plan["type"] = "kpi"
        return plan
    valid_n = int(d[m].notna().sum())
    if valid_n < 2 and len(d) > 1:
        plan["reason"] = f"Only {valid_n} row(s) have a valid value for {pretty(m)}, which is not enough for a chart."
        return plan
    if d[m].dropna().nunique() <= 1 and len(d) > 1 and not plan["time"]:
        plan["type"] = "table"
        plan["reason"] = f"Every row has the same {pretty(m)}, so a chart would not show any difference."
        return plan

    if plan["time"]:
        periods = d[plan["time"]].dropna().nunique()
        series = next((c for c in dims if 1 < d[c].nunique() <= 50), None)
        plan["series"] = series
        if periods >= cfg.min_periods_line:
            plan["type"] = "line"
        elif periods >= 1:
            plan["type"] = "bar_time"
        return plan

    if dims:
        dim = next((c for c in dims if d[c].nunique() > 1), dims[0])
        plan["dim"] = dim
        n_cat = d[dim].nunique()
        second = next((c for c in dims if c != dim and 1 < d[c].nunique() <= cfg.max_grouped_series), None)
        if second and d.duplicated(subset=[dim, second]).sum() == 0 and n_cat <= cfg.max_bar_categories:
            plan["type"], plan["series"] = "grouped_bar", second
            return plan
        if d[dim].duplicated().any() and not profiles[m].additive:
            plan["type"] = "table"
            plan["reason"] = (f"{pretty(dim)} repeats across rows and {pretty(m)} cannot be added up, "
                              f"so a single bar per {pretty(dim)} would be misleading.")
            return plan
        if n_cat < 2:
            plan["type"] = "kpi"
            return plan
        plan["type"] = "bar"
        p = profiles[m]
        v = d[m].dropna()
        plan["donut"] = (p.additive and (v >= 0).all() and v.sum() > 0
                         and cfg.min_pie_slices <= n_cat <= cfg.max_pie_slices
                         and (bool(_SHARE_Q.search(question or "")) or n_cat <= 4))
        return plan

    others = [x for x in usable if x != m]
    if others and len(d) >= cfg.min_rows_scatter:
        plan["type"], plan["y"] = "scatter", others[0]
        return plan
    if len(usable) >= 2 and len(d) <= cfg.max_bar_categories and ids:
        plan["type"], plan["dim"] = "bar", ids[0]
        return plan
    plan["reason"] = "The result has no category or time column to plot against."
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# 6–7. PREPARATION + RENDERING
# ═══════════════════════════════════════════════════════════════════════════════
def _base_layout(fig: go.Figure, title: str, cfg: VizConfig, height: int | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=12),
        height=height or cfg.height, margin=dict(l=10, r=24, t=60, b=20),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def _render_kpi(d: pd.DataFrame, profiles: dict, plan: dict, cfg: VizConfig) -> tuple[list, list]:
    row = d.iloc[0]
    ms = [plan["measure"]] + [m for m in plan["measures"] if m != plan["measure"] and pd.notna(row[m])]
    ms = [m for m in ms if pd.notna(row[m])][: cfg.max_kpis]
    if not ms:
        return [], []
    fig = go.Figure()
    k = len(ms)
    for i, m in enumerate(ms):
        p = profiles[m]
        val = float(row[m]) * p.scale
        num = {"valueformat": f",.{p.decimals}f", "font": {"size": 40 if k <= 2 else 32}}
        if p.kind == "percent":
            num["suffix"] = "%"
        elif p.kind == "currency":
            num["prefix"] = "$"
        elif p.kind == "duration":
            num["suffix"] = " days"
        fig.add_trace(go.Indicator(mode="number", value=val, number=num,
                                   title={"text": truncate(pretty(m), 34), "font": {"size": 13, "color": "#94a3b8"}},
                                   domain={"x": [i / k + 0.02, (i + 1) / k - 0.02], "y": [0, 1]}))
    ctx = [f"{pretty(c)}: {row[c]}" for c in plan["dims"] if pd.notna(row.get(c))][:2]
    title = "Key figures" + (f" — {truncate(' · '.join(ctx), 60)}" if ctx else "")
    return [_base_layout(fig, title, cfg, height=240)], []


def _render_line(d: pd.DataFrame, profiles: dict, plan: dict, cfg: VizConfig) -> tuple[list, list]:
    t, m, series = plan["time"], plan["measure"], plan.get("series")
    p = profiles[m]
    notes = []
    x = d[[t, m] + ([series] if series else [])].dropna(subset=[t, m]).sort_values(t)
    dropped = len(d) - len(x)
    if dropped:
        notes.append(f"{dropped} row(s) with a missing date or {pretty(m)} were left out of the chart.")
    if series:
        x = x.dropna(subset=[series])
        agg = "sum" if p.additive else "mean"
        rank = x.groupby(series)[m].agg(agg).sort_values(ascending=False)
        keep = rank.index[: cfg.max_line_series]
        if len(rank) > cfg.max_line_series:
            notes.append(f"Showing the top {cfg.max_line_series} of {len(rank)} {pretty(series)} values by {pretty(m)}.")
        x = x[x[series].isin(keep)]
        if x.duplicated(subset=[t, series]).any():
            x = x.groupby([t, series], as_index=False)[m].agg(agg)
            notes.append(f"Repeated {pretty(t)} × {pretty(series)} rows were combined ({agg}).")
    elif x.duplicated(subset=[t]).any():
        agg = "sum" if p.additive else "mean"
        x = x.groupby(t, as_index=False)[m].agg(agg)
        notes.append(f"Repeated {pretty(t)} values were combined ({agg}).")
    monthly = bool((x[t].dt.day == 1).all()) and x[t].nunique() > 1
    fig = go.Figure()
    groups = [(None, x)] if not series else [(g, x[x[series] == g]) for g in keep if (x[series] == g).any()]
    for i, (g, gx) in enumerate(groups):
        fig.add_trace(go.Scatter(
            x=gx[t], y=gx[m] * p.scale, mode="lines+markers", name=truncate(g, 30) if g is not None else pretty(m),
            line=dict(width=3 if not series else 2, color=cfg.color_sequence[i % len(cfg.color_sequence)]),
            marker=dict(size=7), connectgaps=False,
            hovertemplate=(f"{truncate(g, 40)}<br>" if g is not None else "") + "%{x|%b %d, %Y}<br>"
                          + f"{pretty(m)}: {_hover_num(p, 'y')}<extra></extra>",
        ))
    title = f"{pretty(m)} over {pretty(t)}" + (f" by {pretty(series)}" if series else "")
    _base_layout(fig, title, cfg)
    fig.update_xaxes(tickformat="%b %Y" if monthly else None)
    fig.update_yaxes(title=pretty(m), **_axis_fmt(p), rangemode="tozero" if p.kind in {"count", "currency"} else "normal")
    fig.update_layout(showlegend=bool(series), hovermode="x unified" if not series else "closest")
    return [fig], notes


def _bar_frame(d: pd.DataFrame, dim: str, m: str, p: ColumnProfile, question: str, cfg: VizConfig) -> tuple[pd.DataFrame, list]:
    notes = []
    x = d[[dim, m]].copy()
    miss_dim = int(x[dim].isna().sum())
    miss_m = int(x[m].isna().sum())
    x = x.dropna(subset=[dim, m])
    if miss_dim:
        notes.append(f"{miss_dim} row(s) with a blank {pretty(dim)} were left out of the chart.")
    if miss_m:
        notes.append(f"{miss_m} row(s) with no {pretty(m)} value were left out of the chart.")
    x[dim] = x[dim].astype(str)
    if x[dim].duplicated().any():
        x = x.groupby(dim, as_index=False)[m].sum()
        notes.append(f"Repeated {pretty(dim)} values were added together.")
    ordered = _ordered_categories(x[dim])
    asc = bool(re.search(r"\b(lowest|least|fewest|bottom|fastest|shortest|minimum)\b", question or "", re.I))
    if not ordered:
        x = x.sort_values(m, ascending=asc, kind="mergesort")
    n = len(x)
    if n > cfg.max_bar_categories:
        top = x.head(cfg.max_bar_categories) if not ordered else x.nlargest(cfg.max_bar_categories, m)
        rest = x.drop(top.index)
        if p.additive and (rest[m] >= 0).all():
            other = pd.DataFrame({dim: [f"Other ({len(rest)})"], m: [rest[m].sum()]})
            x = pd.concat([top, other], ignore_index=True)
            notes.append(f"Showing the top {cfg.max_bar_categories} of {n} {pretty(dim)} values; "
                         f"the remaining {len(rest)} are grouped as Other.")
        else:
            x = top
            notes.append(f"Showing the {'lowest' if asc else 'highest'} {cfg.max_bar_categories} of {n} "
                         f"{pretty(dim)} values by {pretty(m)}; the table has the full list.")
    return x.reset_index(drop=True), notes


def _render_bar(d: pd.DataFrame, profiles: dict, plan: dict, question: str, cfg: VizConfig) -> tuple[list, list]:
    dim, m = plan["dim"], plan["measure"]
    p = profiles[m]
    x, notes = _bar_frame(d, dim, m, p, question, cfg)
    if x.empty:
        return [], notes
    full = x[dim].tolist()
    labels = _unique_labels(full, cfg.max_label_len)
    vals = (x[m] * p.scale).tolist()
    long_labels = max(len(str(v)) for v in full) > 14
    horizontal = len(x) > 8 or long_labels
    text = [fmt_value(v, p) for v in x[m]]
    colors = [("#64748b" if str(f).startswith("Other (") else cfg.color_sequence[0]) for f in full]
    hover = f"%{{customdata}}<br>{pretty(m)}: {_hover_num(p, 'x' if horizontal else 'y')}<extra></extra>"
    if horizontal:
        bar = go.Bar(y=labels, x=vals, orientation="h", text=text, textposition="outside", cliponaxis=False,
                     marker=dict(color=colors), customdata=full, hovertemplate=hover)
    else:
        bar = go.Bar(x=labels, y=vals, text=text, textposition="outside", cliponaxis=False,
                     marker=dict(color=colors), customdata=full, hovertemplate=hover)
    fig = go.Figure(bar)
    title = f"{pretty(m)} by {pretty(dim)}"
    h = max(cfg.height, 90 + 30 * len(x)) if horizontal else cfg.height
    _base_layout(fig, title, cfg, height=min(h, 900))
    fig.update_layout(showlegend=False, bargap=0.25)
    if horizontal:
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title=pretty(m), **_axis_fmt(p), showgrid=True, gridcolor=GRID)
    else:
        fig.update_xaxes(title=None, tickangle=-25 if len(x) > 5 else 0)
        fig.update_yaxes(title=pretty(m), **_axis_fmt(p))
    if vals and min(vals) >= 0:
        rng_axis = "xaxis" if horizontal else "yaxis"
        fig.update_layout({rng_axis: {"range": [0, max(vals) * 1.15 if max(vals) > 0 else 1]}})
    figs = [fig]

    if plan.get("donut") and len(figs) < cfg.max_charts and not any(str(f).startswith("Other (") for f in full):
        pie = go.Figure(go.Pie(
            labels=labels, values=x[m].tolist(), hole=0.55, sort=False, customdata=full,
            marker=dict(colors=cfg.color_sequence[: len(x)], line=dict(color="rgba(128,128,128,.35)", width=1)),
            textinfo="percent", textposition="inside", insidetextorientation="horizontal",
            hovertemplate=f"%{{customdata}}<br>{pretty(m)}: %{{value:,.{p.decimals}f}}<br>Share: %{{percent}}<extra></extra>",
        ))
        _base_layout(pie, f"Share of {pretty(m)} by {pretty(dim)}", cfg, height=380)
        pie.update_layout(legend=dict(orientation="v", x=1.02, y=0.5))
        figs.append(pie)
    return figs, notes


def _render_grouped(d: pd.DataFrame, profiles: dict, plan: dict, cfg: VizConfig) -> tuple[list, list]:
    dim, ser, m = plan["dim"], plan["series"], plan["measure"]
    p = profiles[m]
    x = d[[dim, ser, m]].dropna()
    notes = []
    if len(x) < len(d):
        notes.append(f"{len(d) - len(x)} row(s) with missing values were left out of the chart.")
    x[dim], x[ser] = x[dim].astype(str), x[ser].astype(str)
    order = x.groupby(dim)[m].sum().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, (g, gx) in enumerate(x.groupby(ser, sort=True)):
        gx = gx.set_index(dim).reindex(order).dropna(subset=[m]).reset_index()
        fig.add_trace(go.Bar(
            x=_unique_labels(gx[dim].tolist(), cfg.max_label_len), y=gx[m] * p.scale, name=truncate(g, 30),
            marker_color=cfg.color_sequence[i % len(cfg.color_sequence)], customdata=gx[dim],
            hovertemplate=f"%{{customdata}}<br>{truncate(g, 40)}<br>{pretty(m)}: {_hover_num(p, 'y')}<extra></extra>",
        ))
    _base_layout(fig, f"{pretty(m)} by {pretty(dim)} and {pretty(ser)}", cfg)
    fig.update_layout(barmode="group" if not p.additive else "stack")
    fig.update_xaxes(tickangle=-25 if len(order) > 5 else 0)
    fig.update_yaxes(title=pretty(m), **_axis_fmt(p))
    return [fig], notes


def _render_scatter(d: pd.DataFrame, profiles: dict, plan: dict, cfg: VizConfig) -> tuple[list, list]:
    xm, ym = plan["measure"], plan["y"]
    px_, py_ = profiles[xm], profiles[ym]
    label = (plan["ids"] or plan["dims"] or [None])[0]
    cols = [xm, ym] + ([label] if label else [])
    x = d[cols].dropna(subset=[xm, ym])
    notes = [f"{len(d) - len(x)} row(s) with missing values were left out of the chart."] if len(x) < len(d) else []
    trace = go.Scattergl if len(x) > 2000 else go.Scatter
    fig = go.Figure(trace(
        x=x[xm] * px_.scale, y=x[ym] * py_.scale, mode="markers",
        marker=dict(size=8, color=cfg.color_sequence[0], opacity=0.75, line=dict(width=0.5, color="rgba(128,128,128,.5)")),
        customdata=x[label].astype(str) if label else None,
        hovertemplate=("%{customdata}<br>" if label else "") + f"{pretty(xm)}: {_hover_num(px_, 'x')}<br>"
                      + f"{pretty(ym)}: {_hover_num(py_, 'y')}<extra></extra>",
    ))
    _base_layout(fig, f"{pretty(ym)} vs {pretty(xm)}", cfg)
    fig.update_xaxes(title=pretty(xm), **_axis_fmt(px_), showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=pretty(ym), **_axis_fmt(py_))
    return [fig], notes


def _render_bar_time(d: pd.DataFrame, profiles: dict, plan: dict, cfg: VizConfig) -> tuple[list, list]:
    t, m = plan["time"], plan["measure"]
    p = profiles[m]
    x = d[[t, m]].dropna().sort_values(t)
    agg = "sum" if p.additive else "mean"
    x = x.groupby(t, as_index=False)[m].agg(agg)
    labels = [ts.strftime("%b %Y") if ts.day == 1 else ts.strftime("%d %b %Y") for ts in x[t]]
    fig = go.Figure(go.Bar(x=labels, y=x[m] * p.scale, text=[fmt_value(v, p) for v in x[m]], textposition="outside",
                           marker_color=cfg.color_sequence[0], cliponaxis=False,
                           hovertemplate=f"%{{x}}<br>{pretty(m)}: {_hover_num(p, 'y')}<extra></extra>"))
    _base_layout(fig, f"{pretty(m)} by {pretty(t)}", cfg)
    fig.update_yaxes(title=pretty(m), **_axis_fmt(p))
    return [fig], [f"Only {len(x)} period(s) available — shown as bars rather than a trend line."]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def _bad_number(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return math.isnan(v) or math.isinf(v)
    return False


def figure_is_valid(fig: go.Figure) -> bool:
    if fig is None or not fig.data:
        return False
    for tr in fig.data:
        if isinstance(tr, go.Indicator):
            if tr.value is None or _bad_number(float(tr.value)):
                return False
            continue
        if isinstance(tr, go.Table):
            continue
        arrays = [getattr(tr, a, None) for a in ("x", "y", "values")]
        arrays = [a for a in arrays if a is not None]
        if not arrays or all(len(a) == 0 for a in arrays):
            return False
        for a in arrays:
            for v in list(a):
                if _bad_number(v) or (isinstance(v, str) and v.strip().lower() in {"nan", "none", "nat"}):
                    return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FALLBACK — formatted table
# ═══════════════════════════════════════════════════════════════════════════════
def table_figure(d: pd.DataFrame, profiles: dict, cfg: VizConfig, title: str = "Result table") -> go.Figure | None:
    if d is None or d.empty:
        return None
    x = d.head(cfg.max_table_rows)
    cols = list(x.columns)[:12]
    cells = []
    for c in cols:
        p = profiles.get(c)
        if p is not None and p.role == "measure":
            cells.append([fmt_value(v, p) for v in x[c]])
        elif p is not None and p.role == "time":
            cells.append(["—" if pd.isna(v) else (v.strftime("%b %Y") if v.day == 1 else v.strftime("%d %b %Y"))
                          for v in x[c]])
        else:
            cells.append(["—" if _is_null_token(v) else truncate(v, 60) for v in x[c]])
    align = ["right" if (profiles.get(c) and profiles[c].role == "measure") else "left" for c in cols]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{truncate(pretty(c), 30)}</b>" for c in cols], fill_color="rgba(37,99,235,.18)",
                    font=dict(size=12), align=align, height=32),
        cells=dict(values=cells, fill_color=[["rgba(128,128,128,.04)", "rgba(128,128,128,.10)"] * (len(x) // 2 + 1)],
                   font=dict(size=12), align=align, height=28),
        columnwidth=[max(80, min(260, 9 * max(len(str(c)), 8))) for c in cols],
    ))
    extra = f" (first {cfg.max_table_rows} of {len(d):,} rows)" if len(d) > cfg.max_table_rows else ""
    _base_layout(fig, title + extra, cfg, height=min(120 + 28 * len(x), 560))
    fig.update_layout(margin=dict(l=4, r=4, t=50, b=4))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def _finalise(figs: list, cfg: VizConfig, chart_type: str) -> list:
    out = []
    for f in figs[: cfg.max_charts]:
        if not figure_is_valid(f):
            log.warning("visual_pipeline: dropped a %s figure that failed output validation", chart_type)
            continue
        f.update_layout(meta={"vp_validated": True, "chart_type": chart_type, "pipeline": PIPELINE_VERSION})
        if cfg.style_fn:
            try:
                styled = cfg.style_fn(f)
                f = styled if isinstance(styled, go.Figure) else f
                f.update_layout(meta={"vp_validated": True, "chart_type": chart_type, "pipeline": PIPELINE_VERSION})
            except Exception as e:  # styling must never remove a valid chart
                log.warning("visual_pipeline: style_fn failed: %s", e)
        out.append(f)
    return out


def build_visuals(df: pd.DataFrame | None, question: str = "", cfg: VizConfig | None = None) -> VizResult:
    """Run the full pipeline. Never raises; always returns figures and/or notes."""
    cfg = cfg or VizConfig()
    try:
        d, notes, hints = validate_frame(df)
        if d is None:
            return VizResult([], notes, "empty", True)
        d, profiles = detect_types(d, hints)
        plan = recommend(d, profiles, question, cfg)
        ctype = plan["type"]
        figs: list = []
        if ctype == "kpi":
            figs, n = _render_kpi(d, profiles, plan, cfg)
        elif ctype == "line":
            figs, n = _render_line(d, profiles, plan, cfg)
        elif ctype == "bar_time":
            figs, n = _render_bar_time(d, profiles, plan, cfg)
        elif ctype == "bar":
            figs, n = _render_bar(d, profiles, plan, question, cfg)
        elif ctype == "grouped_bar":
            figs, n = _render_grouped(d, profiles, plan, cfg)
        elif ctype == "scatter":
            figs, n = _render_scatter(d, profiles, plan, cfg)
        else:
            n = []
        notes += n
        figs = _finalise(figs, cfg, ctype)
        if figs:
            return VizResult(figs, list(dict.fromkeys(notes)), ctype, False, profiles)
        reason = plan.get("reason") or "No reliable chart could be produced for this result."
        notes.append(reason + " The formatted table is shown instead.")
        tf = table_figure(d, profiles, cfg) if cfg.table_fallback_figure else None
        return VizResult(_finalise([tf], cfg, "table") if tf is not None else [],
                         list(dict.fromkeys(notes)), "table", True, profiles)
    except Exception as e:  # visualization must never break the answer
        log.exception("visual_pipeline failed: %s", e)
        return VizResult([], ["A chart could not be produced for this result; the data table is shown instead."],
                         "error", True)


__all__ = ["VizConfig", "VizResult", "ColumnProfile", "build_visuals",
           "validate_frame", "detect_types", "recommend", "table_figure", "figure_is_valid",
           "fmt_value", "PIPELINE_VERSION"]
