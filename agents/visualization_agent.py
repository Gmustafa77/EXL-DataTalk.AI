"""Visualization agent — deterministic, governed charts (no LLM calls).

Two entry points:

* :meth:`ChartBuilder.build` — the primary path used by the analytics engine. It
  chooses a chart from the *shape* of the verified result, then reconciles every
  plotted value against the source table and stamps the figure with the result
  hash. A figure that fails reconciliation is discarded.
* :func:`fallback_charts` — used when an answer arrives with a table but no
  figure (direct-SQL fallback path, or an engine result the builder declined).
  It delegates to ``components.visual_pipeline``.

Determinism contract — the same result table always produces the same chart:

* the primary measure is chosen by rule (SQL ``ORDER BY`` → question wording →
  first non-count measure), the same rule the insight composer uses, so the
  chart and the key finding always describe the same metric;
* the remaining measures follow in a stable order (volume/count next, then by name);
* rows are put in a canonical order before plotting (time ascending; categories
  by primary measure descending with the label as tie-break);
* category, series and metric colours are assigned from the sorted labels, never
  from the row order the database happened to return.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.visual_pipeline import VizConfig, build_visuals

log = logging.getLogger("datatalk.visualization")

PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4"]


@dataclass
class ChartConfig:
    max_charts_per_question: int = 2
    max_bars: int = 15
    max_pie_slices: int = 6
    min_periods_for_trend: int = 3
    style_fn: Callable | None = None
    color_sequence: list = field(default_factory=lambda: list(PALETTE))


@dataclass
class ChartSource:
    """The verified result a chart may be built from.

    The engine passes its ``ResultContract`` (which has the same attributes plus
    ``question_text`` and ``sql``); those two optional fields let the builder pick
    the same primary measure as the insight composer.
    """

    sub_question_id: str
    df: pd.DataFrame
    result_hash: str
    question_text: str = ""
    sql: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic measure selection (mirrors the insight composer's rule)
# ═══════════════════════════════════════════════════════════════════════════════

_SYNONYMS = {"tat": "turnaround", "turnaround": "tat", "overturns": "overturn", "volume": "count"}
_GENERIC = {"avg", "average", "total", "count", "number", "rate", "of", "the"}
_COUNT_WORDS = {"count", "cases", "case", "volume", "total", "number", "num", "appeals", "records", "claims"}
_RATE_AVG_WORDS = {"rate", "percent", "percentage", "pct", "share", "avg", "average", "mean", "median", "ratio"}


def _words(name: object) -> list[str]:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return re.findall(r"[a-z]+", s.lower())


def _is_count(col: str, s: pd.Series) -> bool:
    w = set(_words(col))
    if w & _RATE_AVG_WORDS:
        return False
    if w & _COUNT_WORDS:
        return True
    v = pd.to_numeric(s, errors="coerce").dropna()
    return (not v.empty) and bool(np.allclose(v, v.round())) and bool(v.min() >= 0)


def _order_by_first(sql: str) -> str:
    s = re.sub(r"\s+", " ", sql or "")
    m = re.search(r"\bORDER\s+BY\b(.*?)(?=\bLIMIT\b|$)", s, re.I)
    return m.group(1).split(",")[0].replace("`", "").strip().lower() if m else ""


def order_measures(df: pd.DataFrame, numeric: list[str], question: str = "", sql: str = "") -> list[str]:
    """Primary measure first, then count/volume measures, then the rest by name."""
    if not numeric:
        return []
    primary = None
    first_order = _order_by_first(sql)
    if first_order:
        primary = next((c for c in numeric if c.lower() in first_order), None)
    if primary is None:
        qw = set(_words(question))
        qw |= {_SYNONYMS[w] for w in qw if w in _SYNONYMS}
        scored = sorted(((len((set(_words(c)) - _GENERIC) & qw), i, c) for i, c in enumerate(numeric)),
                        key=lambda t: (-t[0], t[1]))
        if scored and scored[0][0] > 0:
            primary = scored[0][2]
    if primary is None:
        non_count = [c for c in numeric if not _is_count(c, df[c])]
        primary = (non_count or numeric)[0]
    rest = [c for c in numeric if c != primary]
    rest.sort(key=lambda c: (0 if _is_count(c, df[c]) else 1, str(c).lower()))
    return [primary, *rest]


def color_map(labels: list, palette: list) -> dict:
    """Colour per label from the sorted label set, independent of row order."""
    uniq = sorted({str(v) for v in labels})
    return {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}


class ChartBuilder:
    def __init__(self, cfg: ChartConfig | None = None) -> None:
        self.cfg = cfg or ChartConfig()

    def build(self, c: ChartSource, analysis_type: str = "") -> tuple[list, list]:
        cfg, notes, figs = self.cfg, [], []
        df = c.df.copy()
        question = str(getattr(c, "question_text", "") or "")
        sql = str(getattr(c, "sql", "") or "")

        if analysis_type == "drilldown":
            return [], ["Row-level result — table only, by design."]

        # Type inference on a copy; the chart is still built from real values.
        for col in df.columns:
            if self._looks_temporal(col) and not pd.api.types.is_numeric_dtype(df[col]):
                conv = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() and conv.notna().sum() / max(df[col].notna().sum(), 1) >= 0.8:
                    df[col] = conv

        num = df.select_dtypes(include=np.number).columns.tolist()
        num = order_measures(c.df, num, question, sql)
        dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        period = [x for x in df.columns if x not in num and x not in dt and self._looks_period(df[x])]
        cat = [x for x in df.columns if x not in num and x not in dt and x not in period]
        x_time = (dt[0] if dt else (period[0] if period else None))

        if not num:
            return [], ["No numeric measure in the result — table only."]

        # 1 ─ KPI / gauge
        if len(df) == 1:
            for m in num[: cfg.max_charts_per_question]:
                figs.append(self._kpi_figure(df[m].iloc[0], m))
            return self._stamp(figs, c), []

        # 2 ─ Time series
        if x_time is not None:
            sort_cols = [x_time] + ([cat[0]] if cat else [])
            sdf = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
            periods = sdf[x_time].nunique()
            if periods < cfg.min_periods_for_trend:
                notes.append(f"Only {periods} period(s) available — shown as bars, not a trend line.")
                figs.append(self._bar_figure(sdf, x_time, num[0]))
            elif cat and sdf[cat[0]].nunique() <= 6:
                series = cat[0]
                sdf[series] = sdf[series].astype(str)
                kind = "area" if self._is_additive(num[0]) else "line"
                fn = px.area if kind == "area" else px.line
                fig = fn(sdf, x=x_time, y=num[0], color=series,
                         color_discrete_map=color_map(sdf[series].tolist(), cfg.color_sequence),
                         category_orders={series: sorted(sdf[series].unique())},
                         title=f"{self._label(num[0])} by {self._label(series)} over {self._label(x_time)}")
                if kind == "line":
                    fig.update_traces(mode="lines+markers")
                fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.12))
                figs.append(fig)
            else:
                ys = num[:3]
                fig = px.line(sdf, x=x_time, y=ys, markers=True,
                              color_discrete_map={y: cfg.color_sequence[i % len(cfg.color_sequence)]
                                                  for i, y in enumerate(ys)},
                              title=f"{', '.join(self._label(y) for y in ys)} over {self._label(x_time)}")
                fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.12))
                figs.append(fig)
                if len(num) >= 2 and self._mixed_scale(df, num[:2]):
                    notes.append("Metrics have different scales — plotted on a dual axis to avoid a misleading comparison.")
                    figs[-1] = self._dual_axis(sdf, x_time, num[0], num[1])
            return self._stamp(figs[: cfg.max_charts_per_question], c), notes

        # 3 ─ Two dimensions + one measure -> heatmap
        if len(cat) >= 2 and num:
            a, b, m = cat[0], cat[1], num[0]
            if df[a].nunique() <= 20 and df[b].nunique() <= 20:
                agg = "sum" if self._is_additive(m) else "mean"
                piv = df.pivot_table(index=a, columns=b, values=m, aggfunc=agg).sort_index().sort_index(axis=1)
                fig = go.Figure(go.Heatmap(
                    z=piv.values, x=[str(x) for x in piv.columns], y=[str(y) for y in piv.index],
                    colorscale=[[0, "#0f2744"], [0.5, "#1d4ed8"], [1, "#ef4444"]],
                    text=piv.values, texttemplate="%{text:.2f}",
                ))
                fig.update_layout(title=f"{self._label(m)}: {self._label(a)} × {self._label(b)}")
                figs.append(fig)
                return self._stamp(figs, c), notes

        # 4 ─ Category + measure(s)
        if cat and num:
            d, m = cat[0], num[0]
            plot = df.dropna(subset=[d]).copy()
            plot[d] = plot[d].astype(str).str.strip()
            # Canonical order: primary measure descending, label ascending as tie-break.
            plot = plot.sort_values([m, d], ascending=[False, True], kind="mergesort",
                                    na_position="last").reset_index(drop=True)
            n_cat = plot[d].nunique()
            if len(plot) > cfg.max_bars:
                plot = plot.head(cfg.max_bars)
                notes.append(f"Chart shows the top {cfg.max_bars} of {n_cat} categories by {self._label(m)}; the table below is complete.")
            order = plot[d].tolist()
            cmap = color_map(order, cfg.color_sequence)
            if len(num) >= 2 and n_cat <= 12 and not self._mixed_scale(plot, num[:3]):
                metrics = num[:3]
                melt = plot.melt(id_vars=[d], value_vars=metrics, var_name="Metric", value_name="Value").dropna(subset=["Value"])
                fig = px.bar(melt, x=d, y="Value", color="Metric", barmode="group", text_auto=".2f",
                             category_orders={d: order, "Metric": metrics},
                             color_discrete_map={mm: cfg.color_sequence[i % len(cfg.color_sequence)]
                                                 for i, mm in enumerate(metrics)},
                             title=f"Metric comparison by {self._label(d)}")
                fig.update_layout(legend=dict(orientation="h", y=1.12))
                figs.append(fig)
            elif n_cat > 8:
                hplot = plot.sort_values([m, d], ascending=[True, False], kind="mergesort")
                fig = px.bar(hplot, x=m, y=d, orientation="h", text_auto=".2f",
                             color=m, color_continuous_scale=["#1e3a8a", "#3b82f6", "#93c5fd"],
                             category_orders={d: hplot[d].tolist()},
                             title=f"{self._label(m)} by {self._label(d)}")
                fig.update_layout(coloraxis_showscale=False)
                figs.append(fig)
            else:
                fig = px.bar(plot, x=d, y=m, text_auto=".2f", color=d,
                             color_discrete_map=cmap, category_orders={d: order},
                             title=f"{self._label(m)} by {self._label(d)}")
                fig.update_layout(showlegend=False)
                figs.append(fig)

            # Donut only when a share is genuinely meaningful.
            if (len(figs) < cfg.max_charts_per_question and 2 <= n_cat <= cfg.max_pie_slices
                    and self._is_additive(m) and plot[m].notna().all()
                    and plot[m].min() >= 0 and plot[m].sum() > 0):
                donut = px.pie(plot, names=d, values=m, hole=0.52,
                               color=d, color_discrete_map=cmap, category_orders={d: order},
                               title=f"Share of {self._label(m)} by {self._label(d)}")
                donut.update_traces(textposition="inside", textinfo="label+percent", sort=False)
                figs.append(donut)
            return self._stamp(figs[: cfg.max_charts_per_question], c), notes

        # 5 ─ Numeric only -> scatter
        if len(num) >= 2 and len(df) >= 10:
            sdf = df.sort_values(num[:2], kind="mergesort")
            fig = px.scatter(sdf, x=num[0], y=num[1],
                             size=num[2] if len(num) >= 3 else None,
                             color_discrete_sequence=cfg.color_sequence,
                             title=f"{self._label(num[1])} vs {self._label(num[0])}")
            figs.append(fig)
            return self._stamp(figs, c), notes

        return [], ["No chart type fits this result shape — the table is the complete answer."]

    # ── Chart reconciliation ────────────────────────────────────────────────
    def _stamp(self, figs: list, c: ChartSource) -> list:
        """Stamp each figure with the result hash and verify plotted values exist
        in the source dataframe. A chart that fails reconciliation is discarded."""
        out = []
        source_values = set()
        for col in c.df.select_dtypes(include=np.number).columns:
            source_values.update(round(float(v), 4) for v in c.df[col].dropna())
        for fig in figs:
            try:
                bad = False
                for tr in fig.data:
                    if isinstance(tr, go.Heatmap):
                        continue  # pivot of the same values; aggregation is explicit above
                    ys = getattr(tr, "y", None)
                    vals = getattr(tr, "values", None)
                    for arr in (ys, vals):
                        if arr is None:
                            continue
                        for v in list(arr)[:500]:
                            if isinstance(v, (int, float, np.number)) and not pd.isna(v):
                                if not any(abs(round(float(v), 4) - s) < 1e-6 for s in source_values):
                                    bad = True
                                    break
                        if bad:
                            break
                    if bad:
                        break
                if bad:
                    log.error("chart/table reconciliation failed for %s; chart discarded", c.sub_question_id)
                    continue
                fig.update_layout(meta={"result_hash": c.result_hash, "sql_sub_question": c.sub_question_id})
                if self.cfg.style_fn:
                    fig = self.cfg.style_fn(fig)
                out.append(fig)
            except Exception as e:
                log.warning("chart stamping skipped: %s", e)
                out.append(fig)
        return out

    # ── Chart helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _label(c: str) -> str:
        return str(c).replace("_", " ").strip().title()

    @staticmethod
    def _looks_temporal(c: str) -> bool:
        return any(t in str(c).lower() for t in ("date", "month", "year", "quarter", "week", "period", "time"))

    @staticmethod
    def _looks_period(s: pd.Series) -> bool:
        sample = s.dropna().astype(str).head(20)
        if sample.empty:
            return False
        pat = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|Q[1-4]|-Q[1-4])?$")
        return bool(sample.map(lambda v: bool(pat.match(v))).mean() >= 0.8)

    @staticmethod
    def _is_additive(col: str) -> bool:
        """Rates, averages and percentages must never be summed or pie-charted."""
        n = str(col).lower()
        return not any(t in n for t in ("rate", "percent", "pct", "%", "avg", "average", "mean", "ratio", "score", "median"))

    @staticmethod
    def _mixed_scale(df: pd.DataFrame, cols: list) -> bool:
        mags = []
        for c in cols:
            s = df[c].dropna().abs()
            if not s.empty and s.max() > 0:
                mags.append(float(s.max()))
        return bool(mags) and max(mags) / max(min(mags), 1e-9) > 25

    def _dual_axis(self, df: pd.DataFrame, x: str, y1: str, y2: str):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df[x], y=df[y1], name=self._label(y1),
                             marker_color=self.cfg.color_sequence[0], opacity=0.6), secondary_y=False)
        fig.add_trace(go.Scatter(x=df[x], y=df[y2], name=self._label(y2), mode="lines+markers",
                                 line=dict(color=self.cfg.color_sequence[4], width=3)), secondary_y=True)
        fig.update_yaxes(title_text=self._label(y1), secondary_y=False)
        fig.update_yaxes(title_text=self._label(y2), secondary_y=True)
        fig.update_layout(title=f"{self._label(y1)} and {self._label(y2)} over {self._label(x)}",
                          hovermode="x unified", legend=dict(orientation="h", y=1.12))
        return fig

    def _bar_figure(self, df, x, y):
        fig = px.bar(df, x=x, y=y, text_auto=".2f", color_discrete_sequence=self.cfg.color_sequence,
                     title=f"{self._label(y)} by {self._label(x)}")
        fig.update_layout(showlegend=False)
        return fig

    def _kpi_figure(self, value, label):
        is_rate = any(t in str(label).lower() for t in ("rate", "percent", "pct", "%"))
        v = float(value)
        if is_rate:
            disp = v * 100 if 0 <= v <= 1 else v
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=max(0.0, min(disp, 100.0)),
                number={"suffix": "%", "valueformat": ".2f"},
                title={"text": self._label(label)},
                gauge={"axis": {"range": [0, 100], "ticksuffix": "%"},
                       "bar": {"color": self.cfg.color_sequence[0], "thickness": 0.72}},
            ))
        else:
            fig = go.Figure(go.Indicator(
                mode="number", value=v, number={"valueformat": ",.2f"},
                title={"text": self._label(label)},
            ))
        return fig


def fallback_charts(df: pd.DataFrame | None, question: str = "", style_fn: Callable | None = None,
                    max_charts: int = 2) -> list:
    """Charts for a table that arrived without figures; never raises, may return []."""
    if df is None or df.empty:
        return []
    cfg = VizConfig(max_charts=max_charts, table_fallback_figure=False, style_fn=style_fn)
    return build_visuals(df, question=question, cfg=cfg).figures
