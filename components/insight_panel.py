"""Insight panel — renders an insight contract under the "AI Insight Findings" text.

Layout
  📊 Evidence at a Glance  (tiles, each linked to an evidence id)
  Tabs: 🔍 Detailed Analysis · 💼 Business Impact · 🧠 AI Reasoning ·
        📑 Evidence · ⚡ Recommended Actions · ⚠️ Data Considerations · 🧾 SQL
  Multi-question answers get one tab per question.

Every recommended action has a "Draft in Outlook" button (subject and body
pre-filled, To left empty); "Email action plan" drafts all actions at once.
Mail links are built by ``components.outlook_mail``.

Accepts both the composer contract (``agents.insight_generation_agent``) and the
simpler dicts produced by fast paths (string / dict metrics, string actions).
Sections that repeat the answer text are suppressed automatically.
"""
from __future__ import annotations

import difflib
import html
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.outlook_mail import MAIL_CSS, mail_button, mail_url, short
from core.plot_style import style_figure

PRIORITY_STYLE = {
    "Immediate": ("🔴", "#dc2626"),
    "Short-term": ("🟠", "#ea580c"),
    "Strategic": ("🟣", "#8b5cf6"),
    "Monitor": ("🔵", "#3b82f6"),
}
PRIORITY_ORDER = {"Immediate": 0, "Short-term": 1, "Strategic": 2, "Monitor": 3}
_DUP_THRESHOLD = 0.80



# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _md(v: Any) -> str:
    t = _esc(v)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


def _norm(t: str) -> str:
    t = re.sub(r"[*_`#>]", " ", str(t or "").lower())
    t = re.sub(r"[\d.,%×]+", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t)).strip()


def _is_dup(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    return a == b or a in b or difflib.SequenceMatcher(None, a, b).ratio() >= _DUP_THRESHOLD


def _label(text: str, color: str = "var(--dt-accent)") -> None:
    st.markdown(
        f"<div style='font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:{color};"
        f"font-weight:800;margin:4px 0 8px;'>{_esc(text)}</div>",
        unsafe_allow_html=True,
    )


def _chips(refs: list) -> str:
    if not refs:
        return ""
    return "".join(
        "<span style='background:rgba(100,116,139,.15);border:1px solid rgba(100,116,139,.35);"
        "color:inherit;opacity:.72;border-radius:10px;padding:0 6px;font-size:.62rem;margin-left:4px;"
        f"white-space:nowrap;'>{_esc(r)}</span>"
        for r in refs[:5]
    )


def _as_bullets(value: Any) -> list:
    """Accept list[{"text","refs"}], list[str] or str and return list[{"text","refs"}]."""
    if not value:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", value) if p.strip()]
        return [{"text": p, "refs": []} for p in parts]
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            out.append({"text": str(item["text"]).strip(), "refs": list(item.get("refs") or [])})
        elif isinstance(item, str) and item.strip():
            out.append({"text": item.strip(), "refs": []})
    return out


def _bullet_html(items: list, accent: str) -> str:
    rows = "".join(
        "<div style='display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--dt-border);'>"
        f"<span style='color:{accent};font-weight:800;'>▸</span>"
        f"<span style='color:inherit;font-size:.87rem;line-height:1.7;'>{_md(b['text'])}{_chips(b['refs'])}</span>"
        "</div>"
        for b in items
    )
    return f"<div>{rows}</div>"


def _style(fig: go.Figure, height: int) -> go.Figure:
    return style_figure(fig, height, dict(l=10, r=30, t=50, b=20))


def _fmt(v: float, spec: dict, delta: bool = False) -> str:
    x = float(v) * float(spec.get("scale") or 1)
    d = int(spec.get("decimals") or 0)
    if d == 0 and abs(x - round(x)) > 1e-9:
        d = 1
    unit = spec.get("unit") or ""
    if delta and unit == "%":
        unit = " percentage points"
    sign = "+" if delta and x > 0 else ""
    return f"{sign}{x:,.{d}f}{unit}"


# ═══════════════════════════════════════════════════════════════════════════════
# Outlook mail drafts
# ═══════════════════════════════════════════════════════════════════════════════
def _evidence_lookup(insight: dict) -> dict:
    return {str(e.get("id")): f"{e.get('label', '')}: {e.get('display', '')}"
            for e in (insight.get("evidence") or []) if isinstance(e, dict) and e.get("id")}


def _action_mail(a: dict, ctx: dict) -> tuple:
    subject = f"[DataTalk.AI] {a['priority']} action: {short(a['action'], 80)}"
    lines = ["Hi,", "", "Please take ownership of the following action identified in the DataTalk.AI analysis.", "",
             "ACTION", a["action"], ""]
    if a["owner"]:
        lines.append(f"Owner role: {a['owner']}")
    lines.append(f"Priority: {a['priority']}")
    if a["timeframe"]:
        lines.append(f"Timeframe: {a['timeframe']}")
    if a["success_measure"]:
        lines.append(f"Success measure: {a['success_measure']}")
    ev = [ctx["evidence"][i] for i in a["evidence_ids"] if i in ctx["evidence"]]
    if ev:
        lines += ["", "SUPPORTING EVIDENCE"] + [f"- {e}" for e in ev[:4]]
    if ctx.get("question"):
        lines += ["", "CONTEXT", f"Question: {short(ctx['question'], 200)}"]
    if ctx.get("finding"):
        lines.append(f"Finding: {short(ctx['finding'], 300)}")
    lines += ["", "Please confirm ownership and the target date by reply.", "", "Thanks,"]
    return subject, "\n".join(lines)


def _plan_mail(actions: list, ctx: dict) -> tuple:
    subject = f"[DataTalk.AI] Action plan: {short(ctx.get('question') or 'analysis follow-up', 80)}"
    lines = ["Hi,", "", "Below is the action plan from the DataTalk.AI analysis.", ""]
    if ctx.get("finding"):
        lines += [f"Finding: {short(ctx['finding'], 300)}", ""]
    for i, a in enumerate(actions, 1):
        meta = " | ".join(x for x in (a["priority"], a["owner"], a["timeframe"]) if x)
        lines.append(f"{i}. {a['action']}")
        lines.append(f"   {meta}")
        if a["success_measure"]:
            lines.append(f"   Success measure: {a['success_measure']}")
        lines.append("")
    lines += ["Please confirm owners and target dates by reply.", "", "Thanks,"]
    return subject, "\n".join(lines)



# ═══════════════════════════════════════════════════════════════════════════════
# normalisation (new + legacy contracts)
# ═══════════════════════════════════════════════════════════════════════════════
def _tiles(insight: dict) -> list:
    raw = insight.get("supporting_metrics") or []
    if isinstance(raw, dict):
        return [{"label": str(k), "value": str(v), "evidence": ""} for k, v in raw.items()][:6]
    out = []
    for m in raw if isinstance(raw, list) else []:
        if isinstance(m, dict) and m.get("label") and m.get("value") not in (None, ""):
            out.append({"label": str(m["label"]), "value": f"{m['value']}{(' ' + m['unit']) if m.get('unit') else ''}",
                        "evidence": str(m.get("evidence", ""))})
    return out[:6]


def _actions(insight: dict) -> list:
    raw = insight.get("recommended_actions") or insight.get("actions") or []
    out = []
    for _i, a in enumerate(raw if isinstance(raw, list) else []):
        if isinstance(a, str):
            if a.strip():
                out.append({"priority": "Monitor", "action": a.strip(), "owner": "", "timeframe": "",
                            "success_measure": "", "evidence_ids": []})
            continue
        if not isinstance(a, dict):
            continue
        text = str(a.get("action") or a.get("recommendation") or "").strip()
        if not text:
            continue
        pr = str(a.get("priority") or "Monitor").strip().title().replace("Short-Term", "Short-term")
        sm = str(a.get("success_measure") or a.get("success_metric") or "").strip()
        if sm == "Confirm completion and review the affected metric.":
            sm = ""
        out.append({
            "priority": pr if pr in PRIORITY_STYLE else "Monitor",
            "action": text,
            "owner": "" if str(a.get("owner", "")).strip() in {"", "Unassigned"} else str(a.get("owner")).strip(),
            "timeframe": "" if str(a.get("timeframe", "")).strip() in {"", "Not specified"} else str(a.get("timeframe")).strip(),
            "success_measure": sm,
            "evidence_ids": list(a.get("evidence_ids") or []),
        })
    out.sort(key=lambda x: PRIORITY_ORDER.get(x["priority"], 3))
    return out


def _reasoning(insight: dict) -> list:
    steps = insight.get("reasoning_steps")
    if isinstance(steps, list) and steps:
        return [str(s) for s in steps if str(s).strip()]
    text = str(insight.get("reasoning") or "").strip()
    return [text] if text else []


# ═══════════════════════════════════════════════════════════════════════════════
# evidence chart (different from the answer chart — shows WHY the claim holds)
# ═══════════════════════════════════════════════════════════════════════════════
def _evidence_figure(spec: dict | None) -> go.Figure | None:
    if not spec or not spec.get("data"):
        return None
    data = pd.DataFrame(spec["data"])
    if data.empty or "value" not in data:
        return None
    metric = spec.get("metric", "Value")
    if spec.get("type") == "ranking":
        med = float(spec.get("median", data["value"].median()))
        data["dev"] = data["value"] - med
        data = data.sort_values("dev")
        pol = spec.get("polarity", "neutral")

        def color(row):
            if pol == "neutral":
                return "#3b82f6"
            better = row["dev"] < 0 if pol == "lower" else row["dev"] > 0
            return "#10b981" if better else ("#94a3b8" if abs(row["dev"]) < 1e-12 else "#ef4444")

        colors = [color(r) for _, r in data.iterrows()]
        widths = [3 if str(lbl) == str(spec.get("focus")) else 0 for lbl in data["label"]]
        scale = float(spec.get("scale") or 1)
        fig = go.Figure(go.Bar(
            x=data["dev"] * scale, y=data["label"].astype(str), orientation="h",
            marker=dict(color=colors, line=dict(color="#f59e0b", width=widths)),
            text=[f"{_fmt(v, spec)}  ({_fmt(d, spec, delta=True)})" for v, d in zip(data["value"], data["dev"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>Deviation from median: %{x:,.2f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Median {_fmt(med, spec)}", annotation_font_color="#f59e0b")
        legend = ("green = better than median, red = worse" if pol != "neutral" else "bars show distance from the median")
        fig.update_layout(title=f"{metric}: deviation from the median ({legend})", showlegend=False,
                          xaxis_title=f"Difference from median{' (percentage points)' if spec.get('unit') == '%' else ''}")
        return _style(fig, max(320, 70 + 34 * len(data)))
    if spec.get("type") == "trend":
        scale = float(spec.get("scale") or 1)
        fig = go.Figure(go.Scatter(x=data["label"].astype(str), y=data["value"] * scale, mode="lines+markers",
                                   line=dict(color="#3b82f6", width=3), name=str(metric)))
        mean = spec.get("mean")
        if mean is not None:
            fig.add_hline(y=float(mean) * scale, line_dash="dot", line_color="#f59e0b",
                          annotation_text=f"Average {_fmt(mean, spec)}", annotation_font_color="#f59e0b")
        for key, col, sym in (("peak", "#ef4444", "triangle-up"), ("trough", "#10b981", "triangle-down")):
            lbl = spec.get(key)
            row = data[data["label"].astype(str) == str(lbl)]
            if not row.empty:
                fig.add_trace(go.Scatter(x=[str(lbl)], y=[float(row["value"].iloc[0]) * scale], mode="markers+text",
                                         marker=dict(size=13, color=col, symbol=sym), text=[key.title()],
                                         textposition="top center", showlegend=False))
        fig.update_layout(title=f"{metric} with average, peak and trough", showlegend=False)
        return _style(fig, 340)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# section renderers
# ═══════════════════════════════════════════════════════════════════════════════
def _render_tiles(tiles: list) -> None:
    if not tiles:
        return
    cards = "".join(
        f"<div class='metric-pill' title='{_esc(t['evidence'])}' style='background:var(--dt-surface);"
        "border:1px solid rgba(59,130,246,.18);border-radius:10px;padding:10px 12px;flex:1;min-width:150px;'>"
        "<div style='font-size:.6rem;color:inherit;opacity:.72;text-transform:uppercase;letter-spacing:.06em;font-weight:700;"
        f"margin-bottom:4px;line-height:1.3;'>{_esc(t['label'])}</div>"
        f"<div style='font-size:1.02rem;font-weight:800;color:inherit;'>{_esc(t['value'])}</div></div>"
        for t in tiles
    )
    _label("📊 Evidence at a Glance")
    st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;'>{cards}</div>",
                unsafe_allow_html=True)


def _render_actions(actions: list, strategic: list, card_key: str, ctx: dict | None = None) -> None:
    ctx = ctx or {"evidence": {}}
    if actions:
        st.markdown(MAIL_CSS, unsafe_allow_html=True)
    for i, a in enumerate(actions, 1):
        icon, color = PRIORITY_STYLE.get(a["priority"], PRIORITY_STYLE["Monitor"])
        meta = [f"<span style='background:{color}22;color:{color};border-radius:20px;padding:2px 9px;"
                f"font-size:.67rem;font-weight:700;'>{icon} {_esc(a['priority'])}</span>"]
        if a["owner"]:
            meta.append("<span style='background:var(--dt-border);color:inherit;opacity:.72;border-radius:20px;"
                        f"padding:2px 9px;font-size:.67rem;'>👤 {_esc(a['owner'])}</span>")
        if a["timeframe"]:
            meta.append("<span style='background:var(--dt-border);color:inherit;opacity:.72;border-radius:20px;"
                        f"padding:2px 9px;font-size:.67rem;'>🗓️ {_esc(a['timeframe'])}</span>")
        success = (f"<div style='font-size:.76rem;color:inherit;opacity:.72;margin-top:6px;'><strong>🎯 Success measure:"
                   f"</strong> {_md(a['success_measure'])}</div>") if a["success_measure"] else ""
        ev = (f"<div style='font-size:.7rem;color:inherit;opacity:.72;margin-top:4px;'>Evidence:{_chips(a['evidence_ids'])}"
              f"</div>") if a["evidence_ids"] else ""
        subject, body = _action_mail(a, ctx)
        button = mail_button(mail_url(subject, body), "Draft in Outlook",
                              f"Opens an Outlook draft for action {i} with subject and body filled in")
        footer = (f"<div class='dt-card-foot'><span class='dt-card-hint'>Add the assignee in To, then send.</span>"
                  f"{button}</div>")
        st.markdown(
            f"<div style='padding:12px 14px;margin-bottom:12px;background:var(--dt-surface);"
            f"border:1px solid {color}33;border-left:4px solid {color};border-radius:0 10px 10px 0;'>"
            f"<div style='color:inherit;font-size:.88rem;line-height:1.6;font-weight:600;margin-bottom:8px;'>"
            f"{i}. {_md(a['action'])}</div><div style='display:flex;gap:6px;flex-wrap:wrap;'>{''.join(meta)}</div>"
            f"{success}{ev}{footer}</div>",
            unsafe_allow_html=True,
        )
    if actions:
        plan = pd.DataFrame([{"#": i, "Priority": a["priority"], "Action": a["action"], "Owner": a["owner"],
                              "Timeframe": a["timeframe"], "Success measure": a["success_measure"],
                              "Evidence": ", ".join(a["evidence_ids"])} for i, a in enumerate(actions, 1)])
        subject, body = _plan_mail(actions, ctx)
        st.markdown(
            "<div class='dt-plan-bar'><div><div class='dt-plan-title'>Share the full action plan</div>"
            f"<div class='dt-plan-sub'>{len(actions)} action(s) in one Outlook draft</div></div>"
            + mail_button(mail_url(subject, body), "Email action plan",
                           "Opens one Outlook draft containing every action", variant="secondary")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.download_button("📥 Export action plan (CSV)", plan.to_csv(index=False).encode("utf-8-sig"),
                           file_name="DataTalk_Action_Plan.csv", mime="text/csv", key=f"{card_key}_plan_dl")
    if strategic:
        _label("🚀 Strategic Improvements", "#8b5cf6")
        st.markdown(_bullet_html([{"text": s, "refs": []} for s in strategic], "#8b5cf6"), unsafe_allow_html=True)


def _render_evidence(insight: dict, card_key: str) -> None:
    fig = _evidence_figure(insight.get("evidence_chart"))
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key=f"{card_key}_evidence_chart",
                        config={"displaylogo": False})
    ev = insight.get("evidence") or []
    if ev:
        table = pd.DataFrame([{"ID": e.get("id", ""), "Evidence": e.get("label", ""),
                               "Value": e.get("display", ""), "Source": e.get("source", "")} for e in ev])
        st.dataframe(table, width="stretch", hide_index=True,
                     height=min(38 * (len(table) + 1), 420),
                     column_config={"ID": st.column_config.TextColumn(width="small"),
                                    "Value": st.column_config.TextColumn(width="small")})
        st.caption("IDs (F1, F2 …) are referenced next to every statement on this card.")
    else:
        legacy = insight.get("evidence_references") or []
        if legacy:
            st.dataframe(pd.DataFrame(legacy), width="stretch", hide_index=True)
        elif fig is None:
            st.info("This answer was produced by deterministic application logic; the result table is the evidence.")


def _render_body(insight: dict, card_key: str, answer_text: str) -> None:
    kf = insight.get("key_finding")
    kf_text = kf.get("text") if isinstance(kf, dict) else str(insight.get("insight") or "")
    anchors = [t for t in (answer_text, kf_text) if t]

    def not_dup(items: list) -> list:
        return [b for b in items if not any(_is_dup(b["text"], a) for a in anchors)]

    detailed = not_dup(_as_bullets(insight.get("detailed_analysis")))
    impact = [b for b in not_dup(_as_bullets(insight.get("business_impact")))
              if not any(_is_dup(b["text"], d["text"]) for d in detailed)]
    impact = [b for b in impact if "derived from validated, read-only queries" not in b["text"].lower()]
    reasoning = [s for s in _reasoning(insight) if not any(_is_dup(s, a) for a in anchors)]
    actions = _actions(insight)
    strategic = [str(s) for s in (insight.get("strategic_improvements") or []) if str(s).strip()]
    caveats = [str(c) for c in (insight.get("data_caveats") or []) if str(c).strip()]
    sql = str(insight.get("_sql") or "").strip()

    _render_tiles(_tiles(insight))

    sections = []
    if detailed:
        sections.append(("🔍 Detailed Analysis", "detailed"))
    if impact:
        sections.append(("💼 Business Impact", "impact"))
    if reasoning:
        sections.append(("🧠 AI Reasoning", "reasoning"))
    if insight.get("evidence") or insight.get("evidence_chart") or insight.get("evidence_references"):
        sections.append(("📑 Evidence", "evidence"))
    if actions or strategic:
        sections.append((f"⚡ Recommended Actions ({len(actions)})", "actions"))
    if caveats:
        sections.append((f"⚠️ Data Considerations ({len(caveats)})", "caveats"))
    if sql:
        sections.append(("🧾 SQL", "sql"))
    if not sections:
        return

    tabs = st.tabs([s[0] for s in sections])
    for tab, (_, kind) in zip(tabs, sections):
        with tab:
            if kind == "detailed":
                st.caption("How the result is distributed — range, position, spread and outliers.")
                st.markdown(_bullet_html(detailed, "#059669"), unsafe_allow_html=True)
            elif kind == "impact":
                st.caption("What the result means operationally — sized from the result where possible.")
                st.markdown(
                    "<div style='background:rgba(234,88,12,.07);border-left:3px solid #ea580c;"
                    "border-radius:0 8px 8px 0;padding:4px 14px;'>" + _bullet_html(impact, "#ea580c") + "</div>",
                    unsafe_allow_html=True,
                )
            elif kind == "reasoning":
                st.caption("How the answer was derived from the data — scope, definitions and method.")
                st.markdown(
                    "".join(
                        "<div style='display:flex;gap:10px;padding:7px 0;border-bottom:1px solid var(--dt-border);'>"
                        f"<span style='min-width:22px;height:22px;border-radius:50%;background:rgba(139,92,246,.2);"
                        f"color:inherit;font-size:.7rem;font-weight:800;display:flex;align-items:center;"
                        f"justify-content:center;'>{i}</span>"
                        f"<span style='color:inherit;font-size:.85rem;line-height:1.7;'>{_md(s)}</span></div>"
                        for i, s in enumerate(reasoning, 1)
                    ),
                    unsafe_allow_html=True,
                )
            elif kind == "evidence":
                _render_evidence(insight, card_key)
            elif kind == "actions":
                _render_actions(actions, strategic, card_key,
                                ctx={"evidence": _evidence_lookup(insight), "question": insight.get("question", ""),
                                     "finding": kf_text or answer_text})
            elif kind == "caveats":
                for c in caveats:
                    st.warning(c)
            elif kind == "sql":
                st.code(sql, language="sql")


# ═══════════════════════════════════════════════════════════════════════════════
# public API
# ═══════════════════════════════════════════════════════════════════════════════
def render_insight_panel(insight: dict | None, result_df: pd.DataFrame | None = None,
                         card_key: str = "default", answer_text: str = "") -> None:
    """Render the insight sections under the AI Insight Findings text.

    `answer_text` is the text already shown as "AI Insight Findings"; any section
    that merely repeats it is hidden.
    """
    if not insight:
        return
    questions = [q for q in (insight.get("questions") or []) if isinstance(q, dict)]
    if len(questions) > 1:
        tabs = st.tabs([f"Q{i}: {str(q.get('question', ''))[:48]}" for i, q in enumerate(questions, 1)])
        for i, (tab, q) in enumerate(zip(tabs, questions), 1):
            with tab:
                kf = q.get("key_finding")
                if isinstance(kf, dict) and kf.get("text"):
                    st.markdown(
                        f"<div style='font-size:.9rem;color:inherit;font-weight:600;margin:4px 0 10px;'>"
                        f"{_md(kf['text'])}{_chips(kf.get('refs') or [])}</div>",
                        unsafe_allow_html=True,
                    )
                _render_body(q, f"{card_key}_q{i}", answer_text="")
        return
    _render_body(insight, card_key, answer_text)


__all__ = ["render_insight_panel"]
