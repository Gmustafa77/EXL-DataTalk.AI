"""Lightweight startup indicator."""
from html import escape

import streamlit as st


_STYLE = """
<style>
.dtl-mini {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 0;
    color: inherit;
    font-family: inherit;
}

.dtl-mini-spinner {
    width: 26px;
    height: 26px;
    flex: 0 0 26px;
    border: 3px solid rgba(127, 127, 127, 0.25);
    border-top-color: var(--dt-accent, #60a5fa);
    border-radius: 50%;
    animation: dtl-mini-spin 0.85s linear infinite;
}

.dtl-mini-title {
    font-size: 0.95rem;
    font-weight: 650;
    line-height: 1.5;
}

.dtl-mini-message {
    font-size: 0.8rem;
    opacity: 0.78;
    line-height: 1.5;
}

@keyframes dtl-mini-spin {
    to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
    .dtl-mini-spinner {
        animation: none;
    }
}
</style>
"""


class StartupLoader:
    def __init__(self):
        self.slot = st.empty()

    def show(self, phase="loading", rows=None):
        messages = {
            "loading": "Loading appeals data…",
            "loaded": "Initializing analytics…",
            "preparing": "Preparing your dashboard…",
        }

        message = messages.get(phase, "Loading…")

        self.slot.markdown(
            _STYLE
            + '<div class="dtl-mini" role="status" aria-live="polite">'
            + '<span class="dtl-mini-spinner" aria-hidden="true"></span>'
            + '<div>'
            + '<div class="dtl-mini-title">DataTalk.AI</div>'
            + f'<div class="dtl-mini-message">{escape(message)}</div>'
            + '</div></div>',
            unsafe_allow_html=True,
        )

    def clear(self):
        self.slot.empty()