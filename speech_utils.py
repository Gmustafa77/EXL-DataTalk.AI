"""Voice helpers: speech-to-text input and browser read-aloud (no audio leaves the browser
for read-aloud; speech-to-text uses the browser's Web Speech API via streamlit-mic-recorder)."""

from __future__ import annotations

import re

import streamlit as st
from streamlit_mic_recorder import speech_to_text


def render_mic_recorder(state_key: str, lang: str = "en", key_suffix: str = "") -> None:
    """Microphone button (streamlit-mic-recorder). A new transcript is stored in
    ``st.session_state[state_key]`` and the app reruns so the user can review it."""
    text = speech_to_text(
        language=lang,
        start_prompt="🎙️",
        stop_prompt="⏹️",
        use_container_width=False,
        just_once=True,
        key=f"stt_{state_key}{key_suffix}",
    )
    if text:
        last_key = f"_{state_key}_last_capture"
        if st.session_state.get(last_key) != text:
            st.session_state[last_key] = text
            st.session_state[state_key] = text
            st.rerun()


def _escape_for_js_template(text: str) -> str:
    """Escape text so it can be safely embedded inside a JS template literal."""
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
            .replace("`", "\\`")
            .replace("${", "\\${")
    )


def render_speaker_button(text: str, key: str, height: int = 44) -> None:
    """Play / Pause-Resume / Stop controls that read ``text`` aloud with the browser's
    speechSynthesis API. Purely client-side — no audio is generated, stored or sent."""
    if not text or not str(text).strip():
        return
    clean_text = re.sub(r"<[^>]+>", " ", str(text))       # strip stray HTML
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    safe_text = _escape_for_js_template(clean_text)
    cid = f"tts_{key}"
    js = f"""
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
      <button id="{cid}_play" onclick="ttsPlay_{cid}()" title="Read this response aloud"
        style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.35);
        color:#059669;border-radius:8px;padding:4px 10px;font-size:.75rem;cursor:pointer;
        font-family:inherit;">🔊 Listen</button>
      <button id="{cid}_pause" onclick="ttsPause_{cid}()" title="Pause / Resume"
        style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);
        color:#d97706;border-radius:8px;padding:4px 10px;font-size:.75rem;cursor:pointer;
        font-family:inherit;">⏸️</button>
      <button id="{cid}_stop" onclick="ttsStop_{cid}()" title="Stop"
        style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.35);
        color:#dc2626;border-radius:8px;padding:4px 10px;font-size:.75rem;cursor:pointer;
        font-family:inherit;">⏹️</button>
      <span id="{cid}_status" style="font-size:.7rem;color:#64748b;"></span>
    </div>
    <script>
    (function() {{
      let utter = null;
      window.ttsPlay_{cid} = function() {{
        if (!window.speechSynthesis) {{
          document.getElementById("{cid}_status").innerText = "⚠️ Text-to-speech isn't supported in this browser.";
          return;
        }}
        window.speechSynthesis.cancel();
        utter = new SpeechSynthesisUtterance(`{safe_text}`);
        utter.rate = 1.0;
        utter.onstart = () => document.getElementById("{cid}_status").innerText = "🔊 Playing";
        utter.onend   = () => document.getElementById("{cid}_status").innerText = "";
        utter.onerror = () => document.getElementById("{cid}_status").innerText = "⚠️ Playback error.";
        window.speechSynthesis.speak(utter);
      }};
      window.ttsPause_{cid} = function() {{
        if (!window.speechSynthesis) return;
        if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {{
          window.speechSynthesis.pause();
          document.getElementById("{cid}_status").innerText = "⏸️ Paused";
        }} else if (window.speechSynthesis.paused) {{
          window.speechSynthesis.resume();
          document.getElementById("{cid}_status").innerText = "🔊 Playing";
        }}
      }};
      window.ttsStop_{cid} = function() {{
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        document.getElementById("{cid}_status").innerText = "";
      }};
    }})();
    </script>
    """
    st.iframe(js, height=height)
