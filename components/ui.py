"""Shared UI building blocks: theme-aware CSS, header, KPI cards, section labels and footer.

Light and dark mode come from Streamlit's native theming (``.streamlit/config.toml``
defines ``[theme.light]`` and ``[theme.dark]``; users switch in ⋮ → Settings).
Custom HTML in this app is written to be theme-neutral:

* text colour is inherited from the active Streamlit theme,
* secondary text uses opacity instead of a fixed grey,
* surfaces and borders are translucent overlays,
* only the accent colour is tuned per theme (via ``st.context.theme.type``).
"""

from __future__ import annotations

from html import escape

import streamlit as st

FOOTER_TEXT = "Made and Designed by EXL India | 2026"

_ACCENTS = {"light": ("#1d4ed8", "#6d28d9"), "dark": ("#60a5fa", "#a78bfa")}

_CSS = """
<style>
:root{
  --dt-accent:%(accent)s; --dt-accent-2:%(accent2)s;
  --dt-surface:rgba(127,127,127,.06); --dt-surface-strong:rgba(127,127,127,.11);
  --dt-border:rgba(127,127,127,.24); --dt-radius:14px;
  --dt-good:#16a34a; --dt-bad:#dc2626; --dt-warn:#d97706;
}
.block-container{padding-top:1.4rem;padding-bottom:1.2rem;max-width:1500px;}
.dt-muted{opacity:.72;}
.dt-hero{display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:18px 24px;margin-bottom:16px;
  border:1px solid var(--dt-border);border-left:5px solid var(--dt-accent);border-radius:var(--dt-radius);
  background:linear-gradient(135deg,rgba(37,99,235,.10),rgba(124,58,237,.06));}
.dt-badge{background:#ea580c;color:#fff;font-weight:900;padding:8px 16px;border-radius:10px;letter-spacing:2px;
  font-size:1.1rem;}
.dt-title{font-size:1.8rem;font-weight:800;line-height:1.1;letter-spacing:-.4px;
  background:linear-gradient(90deg,var(--dt-accent),var(--dt-accent-2));-webkit-background-clip:text;
  background-clip:text;-webkit-text-fill-color:transparent;}
.dt-sub{font-size:.9rem;margin-top:4px;opacity:.72;display:flex;align-items:center;gap:8px;}
.dt-live{width:8px;height:8px;border-radius:50%%;background:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.2);}
.dt-pills{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;max-width:720px;}
.dt-pill{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:999px;font-size:.74rem;
  font-weight:600;border:1px solid var(--dt-border);background:var(--dt-surface);white-space:nowrap;}
.dt-pill b{color:var(--dt-accent);font-weight:800;}
.dt-sec{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;font-weight:800;color:var(--dt-accent);
  border-left:3px solid var(--dt-accent);padding-left:9px;margin:18px 0 10px;}
.dt-card{background:var(--dt-surface);border:1px solid var(--dt-border);border-radius:var(--dt-radius);
  padding:14px 16px;}
.dt-kpi{position:relative;height:128px;perspective:1000px;margin-bottom:10px;}
.dt-kpi-inner{position:relative;width:100%%;height:100%%;transition:transform .5s cubic-bezier(.4,0,.2,1);
  transform-style:preserve-3d;}
.dt-kpi:hover .dt-kpi-inner{transform:rotateY(180deg);}
.dt-kpi-face{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;
  border-radius:var(--dt-radius);padding:14px 16px;display:flex;flex-direction:column;justify-content:center;
  background:var(--dt-surface);border:1px solid var(--dt-border);border-top:3px solid var(--dt-accent);}
.dt-kpi-back{transform:rotateY(180deg);background:var(--dt-surface-strong);}
.dt-kpi-lbl{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;opacity:.72;}
.dt-kpi-val{font-size:1.6rem;font-weight:800;line-height:1.15;margin:4px 0;}
.dt-delta{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 8px;border-radius:999px;width:fit-content;}
.dt-delta.good{color:var(--dt-good);background:rgba(22,163,74,.12);}
.dt-delta.bad{color:var(--dt-bad);background:rgba(220,38,38,.10);}
.dt-row{display:flex;justify-content:space-between;font-size:.8rem;margin:2px 0;}
.dt-row span:first-child{opacity:.72;}
.dt-row span:last-child{font-weight:700;}
.dt-chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.7rem;font-weight:700;margin:2px 4px 2px 0;
  border:1px solid var(--dt-border);background:var(--dt-surface);}
.dt-hero-chat{text-align:center;padding:26px 10px 8px;}
.dt-orb{width:58px;height:58px;border-radius:50%%;margin:0 auto 14px;display:flex;align-items:center;
  justify-content:center;font-size:26px;background:radial-gradient(circle at 35%% 30%%,#60a5fa,#2563eb 60%%,#1e40af);
  box-shadow:0 0 0 6px rgba(37,99,235,.12);}
.dt-welcome{font-size:1.7rem;font-weight:800;background:linear-gradient(90deg,var(--dt-accent),var(--dt-accent-2));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.dt-footer{margin-top:36px;padding:14px 0 6px;border-top:1px solid var(--dt-border);text-align:center;
  font-size:.78rem;letter-spacing:.04em;opacity:.65;}
.dt-footer strong{font-weight:700;}
div[data-testid="stChatInput"] textarea{padding-right:56px !important;}
.st-key-chat_input_wrap{position:relative;}
/* Remove the gap only in the container holding chat + microphone. */
[data-testid="stVerticalBlock"]:has(> .st-key-chat_input_wrap):has(> .st-key-mic_below_chat) {
    gap: 0 !important;
}

.st-key-chat_input_wrap {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Microphone directly below the chatbox. */
.st-key-mic_below_chat {
    position: static !important;
    inset: auto !important;
    transform: none !important;
    width: 100%% !important;
    max-width: 100%% !important;
    height: auto !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

.st-key-mic_below_chat iframe {
    display: block !important;
    width: 100%% !important;
    min-width: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}
</style>
"""


def theme_type() -> str:
    """Active Streamlit theme type ("light" or "dark"); defaults to light when unknown."""
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def inject_css() -> None:
    accent, accent2 = _ACCENTS.get(theme_type(), _ACCENTS["light"])
    st.html(_CSS % {"accent": accent, "accent2": accent2})


def section(title: str) -> None:
    st.markdown(f"<div class='dt-sec'>{escape(title)}</div>", unsafe_allow_html=True)


def page_title(icon: str, title: str, subtitle: str = "") -> None:
    sub = f"<div class='dt-sub' style='margin:0'>{escape(subtitle)}</div>" if subtitle else ""
    st.markdown(f"<div style='margin:4px 0 12px'><div style='font-size:1.3rem;font-weight:800'>{icon} "
                f"{escape(title)}</div>{sub}</div>", unsafe_allow_html=True)


def pill(icon: str, label: str, value: str, tooltip: str = "") -> str:
    return (f"<span class='dt-pill' title='{escape(tooltip, quote=True)}'>{icon} "
            f"<span class='dt-muted'>{escape(label)}</span> <b>{escape(value)}</b></span>")


def header(pills_html: str) -> None:
    st.markdown(
        "<div class='dt-hero'><div class='dt-badge'>EXL</div><div><div class='dt-title'>DataTalk.AI</div>"
        "<div class='dt-sub'><span class='dt-live'></span>Healthcare Appeals Intelligence Platform · POC</div>"
        f"</div><div class='dt-pills'>{pills_html}</div></div>",
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, label: str, value: str, baseline: str, delta: str, good: bool) -> str:
    cls = "good" if good else "bad"
    return (
        "<div class='dt-kpi'><div class='dt-kpi-inner'>"
        f"<div class='dt-kpi-face'><div class='dt-kpi-lbl'>{icon} {escape(label)}</div>"
        f"<div class='dt-kpi-val'>{escape(value)}</div><span class='dt-delta {cls}'>{escape(delta)} vs base</span></div>"
        f"<div class='dt-kpi-face dt-kpi-back'><div class='dt-kpi-lbl'>{escape(label)}</div>"
        f"<div class='dt-row'><span>Current</span><span>{escape(value)}</span></div>"
        f"<div class='dt-row'><span>Baseline</span><span>{escape(baseline)}</span></div>"
        f"<div class='dt-row'><span>Δ vs base</span><span>{escape(delta)}</span></div></div>"
        "</div></div>"
    )


def footer() -> None:
    """Application footer, rendered once per page by the entry point."""
    st.markdown(f"<div class='dt-footer' role='contentinfo'><strong>{escape(FOOTER_TEXT)}</strong></div>",
                unsafe_allow_html=True)
