"""Shared Plotly styling that works in both light and dark themes.

Figures are rendered with ``st.plotly_chart(..., theme="streamlit")`` (the default),
so Streamlit supplies text, grid and hover colours for the active theme. Styling
here therefore sets only layout, transparency and the categorical palette — never
fixed text colours.
"""

from __future__ import annotations

import plotly.graph_objects as go

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4", "#ec4899", "#84cc16", "#f97316"]
GRID = "rgba(128,128,128,0.18)"
DEFAULT_MARGIN = dict(l=16, r=16, t=56, b=24)

UNIT_COLORS = {
    "MCM APPEALS UNIT": "#3b82f6", "MEDICAL MEMBER - CLINICAL RESOLVES": "#10b981",
    "SHARED SERVICES MPO - NCAU - ICC TEAM": "#f59e0b", "MEDICAL MEMBER - SAG - FULLY INSURED": "#8b5cf6",
    "MEDICAL MEMBER - SAG - MEMBER": "#ef4444", "PHARMACY COMMERCIAL SERVICES": "#06b6d4",
    "TRAIGE TEAM MEMBER": "#ec4899", "NATIONAL ACCOUNTS CUSTOMER RESOLUTION TEAM": "#84cc16",
    "SHARED SERVICES UM - NCAU -AUTHORIZATION FALL": "#f97316",
}
STATUS_COLORS = {"CLOSED": "#10b981", "PENDING": "#f59e0b", "In Progress": "#3b82f6", "CHECKED IN": "#8b5cf6",
                 "CHECKED OUT": "#06b6d4", "TRANSFERRED": "#ef4444", "OPEN": "#f59e0b"}
DETERMINATION_COLORS = {"UPHELD": "#10b981", "OVERTURNED": "#ef4444", "PARTIALLY OVERTURNED": "#f59e0b",
                        "Pending Decision": "#94a3b8"}
RISK_BAND_COLORS = {"Very Low": "#10b981", "Low": "#3b82f6", "Medium": "#f59e0b", "High": "#ef4444",
                    "Critical": "#991b1b"}
SEVERITY_COLORS = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#eab308", "Low": "#16a34a"}
SEQUENTIAL_BLUE = ["#93c5fd", "#3b82f6", "#1e3a8a"]


def style_figure(fig: go.Figure, height: int = 440, margin: dict | None = None) -> go.Figure:
    """Apply the shared, theme-neutral layout to a figure and return it."""
    fig.update_layout(
        height=height, margin=margin or DEFAULT_MARGIN,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=13), colorway=PALETTE,
        title=dict(x=0.01, xanchor="left", font=dict(size=15)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "toImageButtonOptions": {"format": "png", "scale": 2}}
