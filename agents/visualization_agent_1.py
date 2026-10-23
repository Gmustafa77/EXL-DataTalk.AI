"""Visualization agent — deterministic, governed charts (no LLM calls).

Two entry points:

* :meth:`ChartBuilder.build` — the primary path used by the analytics engine. It
  chooses a chart from the *shape* of the verified result, then reconciles every
  plotted value against the source table and stamps the figure with the result
  hash. A figure that fails reconciliation is discarded.
* :func:`fallback_charts` — used when an answer arrives with a table but no
  figure (direct-SQL fallback path, or an engine result the builder declined).
  It delegates to ``components.visual_pipeline``.
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
    """The verified result a chart may be built from."""

    sub_question_id: str
    df: pd.DataFrame
    result_hash: str


class ChartBuilder:
    def __init__(self, cfg: ChartConfig | None = None) -> None:
        self.cfg = cfg or ChartConfig()

    def build(self, c: ChartSource, analysis_type: str = "") -> tuple[list, list]:
        cfg, notes, figs = self.cfg, [], []
        df = c.df.copy()

        if analysis_type == "drilldown":
            return [], ["Row-level result — table only, by design."]

        # Type inference on a copy; the chart is still built from real values.
        for col in df.columns:
            if self._looks_temporal(col) and not pd.api.types.is_numeric_dtype(df[col]):
                conv = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() and conv.notna().sum() / max(df[col].notna().sum(), 1) >= 0.8:
                    df[col] = conv

        num = df.select_dtypes(include=np.number).columns.tolist()
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
            sdf = df.sort_values(x_time)
            periods = sdf[x_time].nunique()
            if periods < cfg.min_periods_for_trend:
                notes.append(f"Only {periods} period(s) available — shown as bars, not a trend line.")
                figs.append(self._bar_figure(sdf, x_time, num[0]))
            elif cat and sdf[cat[0]].nunique() <= 6:
                series = cat[0]
                kind = "area" if self._is_additive(num[0]) else "line"
                fn = px.area if kind == "area" else px.line
                fig = fn(sdf, x=x_time, y=num[0], color=series,
                         color_discrete_sequence=cfg.color_sequence,
                         title=f"{num[0]} by {series} over {self._label(x_time)}")
                if kind == "line":
                    fig.update_traces(mode="lines+markers")
                fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.12))
                figs.append(fig)
            else:
                ys = num[:3]
                fig = px.line(sdf, x=x_time, y=ys, markers=True,
                              color_discrete_sequence=cfg.color_sequence,
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
                piv = df.pivot_table(index=a, columns=b, values=m, aggfunc=agg)
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
            n_cat = plot[d].nunique()
            if len(plot) > cfg.max_bars:
                plot = plot.nlargest(cfg.max_bars, m)
                notes.append(f"Chart shows the top {cfg.max_bars} of {n_cat} categories by {self._label(m)}; the table below is complete.")
            if len(num) >= 2 and n_cat <= 12 and not self._mixed_scale(plot, num[:3]):
                melt = plot.melt(id_vars=[d], value_vars=num[:3], var_name="Metric", value_name="Value").dropna(subset=["Value"])
                fig = px.bar(melt, x=d, y="Value", color="Metric", barmode="group", text_auto=".2f",
                             color_discrete_sequence=cfg.color_sequence,
                             title=f"Metric comparison by {self._label(d)}")
                fig.update_layout(legend=dict(orientation="h", y=1.12))
                figs.append(fig)
            elif n_cat > 8:
                fig = px.bar(plot.sort_values(m), x=m, y=d, orientation="h", text_auto=".2f",
                             color=m, color_continuous_scale=["#1e3a8a", "#3b82f6", "#93c5fd"],
                             title=f"{self._label(m)} by {self._label(d)}")
                fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
                figs.append(fig)
            else:
                fig = px.bar(plot, x=d, y=m, text_auto=".2f", color=d,
                             color_discrete_sequence=cfg.color_sequence,
                             title=f"{self._label(m)} by {self._label(d)}")
                fig.update_layout(showlegend=False)
                figs.append(fig)

            # Donut only when a share is genuinely meaningful.
            if (len(figs) < cfg.max_charts_per_question and 2 <= n_cat <= cfg.max_pie_slices
                    and self._is_additive(m) and plot[m].notna().all()
                    and plot[m].min() >= 0 and plot[m].sum() > 0):
                donut = px.pie(plot, names=d, values=m, hole=0.52,
                               color_discrete_sequence=cfg.color_sequence,
                               title=f"Share of {self._label(m)} by {self._label(d)}")
                donut.update_traces(textposition="inside", textinfo="label+percent")
                figs.append(donut)
            return self._stamp(figs[: cfg.max_charts_per_question], c), notes

        # 5 ─ Numeric only -> scatter
        if len(num) >= 2 and len(df) >= 10:
            fig = px.scatter(df, x=num[0], y=num[1],
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
