"""Shared Outlook draft helpers for DataTalk.AI (no mail is ever sent by the app).

Builds Outlook compose links with the subject and body already filled in and
the To field left empty, plus a styled "Draft in Outlook" button.

  INSIGHT_MAIL_CLIENT=outlook_web   (default) Outlook on the web compose window
  INSIGHT_MAIL_CLIENT=desktop       mailto: link -> default desktop mail app
  INSIGHT_MAIL_CC=team@company.com  (optional) pre-filled CC

HTML tags are assembled with tag() so the markup survives copy/paste through
chat tools that strip raw anchors.
"""
from __future__ import annotations

import html
import re
from urllib.parse import quote

from core.settings import get_settings

OUTLOOK_WEB_COMPOSE = "https://outlook.office.com/mail/deeplink/compose"
MAX_BODY_CHARS = 1500  # keeps the encoded link inside browser / Outlook URL limits

_PP_RE = re.compile(r"(?<=\d)\s*(?:pp|p\.p\.|ppts?)(?![A-Za-z])", re.I)


def esc(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def spell_pp(text) -> str:
    return _PP_RE.sub(" percentage points", "" if text is None else str(text))


def short(text, n: int) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def mail_url(subject: str, body: str) -> str:
    """Outlook compose link with subject and body pre-filled and an empty To field."""
    subject, body = spell_pp(subject), spell_pp(body)
    if len(body) > MAX_BODY_CHARS:
        body = body[: MAX_BODY_CHARS - 40].rstrip() + "\n\n[…shortened — see DataTalk.AI]"
    settings = get_settings()
    cc = settings.insight_mail_cc.strip()
    if settings.insight_mail_client == "desktop":
        params = f"subject={quote(subject)}&body={quote(body.replace(chr(10), chr(13) + chr(10)))}"
        if cc:
            params += f"&cc={quote(cc)}"
        return f"mailto:?{params}"
    url = f"{OUTLOOK_WEB_COMPOSE}?subject={quote(subject)}&body={quote(body)}"
    if cc:
        url += f"&cc={quote(cc)}"
    return url


def tag(name: str, attrs: dict, inner: str) -> str:
    lt, gt = chr(60), chr(62)
    attr_txt = "".join(f" {k}='{esc(v)}'" for k, v in attrs.items())
    return f"{lt}{name}{attr_txt}{gt}{inner}{lt}/{name}{gt}"


_MAIL_ICON = (
    "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
    "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
    "<rect x='2.5' y='4.5' width='19' height='15' rx='2.5'/><path d='m3 6.5 9 6.5 9-6.5'/></svg>"
)
_SEND_ICON = (
    "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
    "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
    "<path d='M22 2 11 13'/><path d='M22 2 15 22l-4-9-9-4 20-7z'/></svg>"
)
_ARROW_ICON = (
    "<svg width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.4' "
    "stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'><path d='M7 17 17 7'/><path d='M8 7h9v9'/></svg>"
)

MAIL_CSS = """
<style>
.dt-mail-btn{display:inline-flex;align-items:center;gap:8px;padding:7px 14px 7px 12px;border-radius:8px;
  font-size:.78rem;font-weight:600;letter-spacing:.01em;text-decoration:none!important;white-space:nowrap;cursor:pointer;
  transition:transform .15s ease,box-shadow .15s ease,background .15s ease,border-color .15s ease;}
.dt-mail-btn:focus-visible{outline:2px solid #2563eb;outline-offset:2px;}
.dt-mail-primary{color:#ffffff!important;background:linear-gradient(135deg,#0f6cbd 0%,#2563eb 100%);
  border:1px solid rgba(37,99,235,.45);box-shadow:0 1px 2px rgba(0,0,0,.18),0 4px 12px rgba(15,108,189,.22);}
.dt-mail-primary:hover{transform:translateY(-1px);background:linear-gradient(135deg,#115ea3 0%,#1d4ed8 100%);
  box-shadow:0 2px 4px rgba(0,0,0,.2),0 8px 20px rgba(37,99,235,.32);}
.dt-mail-secondary{color:var(--dt-accent,#2563eb)!important;background:rgba(37,99,235,.08);
  border:1px solid rgba(37,99,235,.40);}
.dt-mail-secondary:hover{transform:translateY(-1px);background:rgba(37,99,235,.16);border-color:#2563eb;
  box-shadow:0 6px 16px rgba(37,99,235,.20);}
.dt-mail-btn .dt-arrow{opacity:.75;margin-left:2px;transition:transform .15s ease;}
.dt-mail-btn:hover .dt-arrow{transform:translate(2px,-2px);opacity:1;}
.dt-mail-sep{width:1px;height:14px;background:currentColor;opacity:.3;}
.dt-card-foot{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;
  margin-top:12px;padding-top:10px;border-top:1px dashed var(--dt-border,rgba(127,127,127,.24));}
.dt-card-hint{font-size:.7rem;opacity:.72;}
.dt-plan-bar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
  padding:12px 14px;margin:6px 0 10px;border-radius:10px;background:var(--dt-surface,rgba(127,127,127,.06));
  border:1px solid var(--dt-border,rgba(127,127,127,.24));}
.dt-plan-title{font-size:.82rem;font-weight:700;}
.dt-plan-sub{font-size:.72rem;opacity:.72;margin-top:2px;}
</style>
"""


def mail_button(url: str, label: str, tooltip: str = "", variant: str = "primary") -> str:
    """HTML for a styled Outlook button. Render with st.markdown(..., unsafe_allow_html=True)
    after MAIL_CSS has been injected once on the page."""
    icon = _SEND_ICON if variant == "secondary" else _MAIL_ICON
    attrs = {"class": f"dt-mail-btn dt-mail-{variant}", "href": url, "title": tooltip or label}
    if not url.startswith("mailto:"):
        attrs["target"] = "_blank"
        attrs["rel"] = "noopener noreferrer"
    inner = (icon + tag("span", {}, esc(label)) + tag("span", {"class": "dt-mail-sep"}, "")
             + tag("span", {"class": "dt-arrow"}, _ARROW_ICON))
    return tag("a", attrs, inner)


__all__ = ["MAIL_CSS", "mail_url", "mail_button", "tag", "esc", "short", "spell_pp"]
