"""Persistent question-answer store backed by ``QA.json`` (backend only).

The supervisor consults this store *before* any AI/LLM call:

    normalise question → lookup
        hit  → rebuild the stored answer; no AI call is made
        miss → existing AI workflow → answer returned → answer stored

Design rules
* Provider-independent: the store keeps the final, verified ``AssistantAnswer``
  (text, table, charts, insight panel), never raw model output.
* Invisible to users: a stored answer is rebuilt with the same route, content and
  layout as the original. Nothing in the UI refers to this module.
* Robust: missing, empty, malformed or corrupted files never break answering.
  Corrupted files are quarantined (``QA.corrupt-<timestamp>.json``) and a fresh
  store is started; invalid or duplicate entries are dropped on load.
* Safe concurrent writes: in-process lock + cross-process file lock, atomic
  temp-file + ``os.replace`` writes, existing entries are always preserved.
* Unicode is preserved (UTF-8, ``ensure_ascii=False``).

No Streamlit imports.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import unicodedata
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # declared dependency; degrade to in-process locking if unavailable
    from filelock import FileLock
except ImportError:  # pragma: no cover
    FileLock = None  # type: ignore[assignment]

log = logging.getLogger("datatalk.qa_store")

STORE_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = "data/QA.json"

# Meta fields that describe the answer itself (call accounting is recomputed per request).
_STORED_META_KEYS = ("plan_status", "question_count", "validation", "defects")


# ═══════════════════════════════════════════════════════════════════════════════
# Question analysis
# ═══════════════════════════════════════════════════════════════════════════════

_QUOTE_CHARS = "\"'`\u2018\u2019\u201c\u201d\u00ab\u00bb"
_TRAILING_PUNCT = "?.!;:,\u3002\uff1f\uff01 "


def normalize_question(text: str | None) -> str:
    """Canonical form used for matching: Unicode NFKC, case-folded, surrounding
    quotes removed, whitespace collapsed, no space before punctuation, trailing
    punctuation removed."""
    t = unicodedata.normalize("NFKC", str(text or "")).casefold()
    t = re.sub(r"\s+", " ", t).strip().strip(_QUOTE_CHARS).strip()
    t = re.sub(r"\s+([?.!,;:])", r"\1", t)
    return t.rstrip(_TRAILING_PUNCT).strip()


_PII_RE = re.compile(r"\d{6,}|[\w.+-]+@[\w-]+\.[\w.-]+")


def question_may_contain_pii(text: str) -> bool:
    """Long digit runs (case/member ids) or e-mail addresses are never stored."""
    return bool(_PII_RE.search(text or ""))


_FOLLOW_UP_RE = re.compile(
    r"^(and|also|then|what about|how about)\b"
    r"|\b(those|these|them|above|aforementioned|that one|this one|the same|drill (?:into|down)"
    r"|break (?:it|that|this|them) down)\b",
    re.I,
)


def depends_on_conversation(question: str, conversation: list[str] | None) -> bool:
    """True for follow-ups whose meaning depends on earlier chat turns. Their answer
    is not reusable for the same wording in a different conversation."""
    prior_turns = [m for m in (conversation or []) if m and m.strip() != (question or "").strip()]
    return bool(prior_turns) and bool(_FOLLOW_UP_RE.search(question or ""))


_RELATIVE_TIME_RE = re.compile(
    r"\b(today|yesterday|tomorrow|ytd|mtd|qtd|year[- ]to[- ]date|month[- ]to[- ]date"
    r"|(?:last|past|previous|prior|this|current|next|coming)\s+(?:\d+\s+)?"
    r"(?:days?|weeks?|months?|quarters?|years?|fy))\b",
    re.I,
)


def is_time_relative(question: str) -> bool:
    """Questions such as "last 3 months" are only reusable on the same calendar day."""
    return bool(_RELATIVE_TIME_RE.search(question or ""))


# ═══════════════════════════════════════════════════════════════════════════════
# Serialisation
# ═══════════════════════════════════════════════════════════════════════════════

def _jsonable(obj: Any) -> Any:
    """Recursively convert pandas/numpy/datetime/Decimal values to plain JSON types."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, np.bool_):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating, Decimal)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return table_to_payload(obj)
    if isinstance(obj, pd.Series):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return None if pd.isna(obj) else obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def table_to_payload(df: pd.DataFrame | None) -> dict | None:
    if not isinstance(df, pd.DataFrame):
        return None
    return {
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "data": [[_jsonable(v) for v in row] for row in df.itertuples(index=False, name=None)],
    }


def table_from_payload(payload: dict | None) -> pd.DataFrame | None:
    if not isinstance(payload, dict) or "columns" not in payload:
        return None
    df = pd.DataFrame(payload.get("data") or [], columns=payload["columns"])
    for col, dtype in (payload.get("dtypes") or {}).items():
        if col not in df.columns:
            continue
        try:
            if dtype.startswith("datetime64"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "bool" and df[col].notna().all():
                df[col] = df[col].astype(bool)
            elif dtype.lower().startswith(("int", "uint", "float")):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "object":
                vals = df[col].dropna()
                if not vals.empty and vals.map(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)).all():
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        except (TypeError, ValueError):
            log.debug("dtype restore skipped for column %s", col)
    return df


def answer_to_payload(answer: Any) -> dict | None:
    """Serialise an ``AssistantAnswer``. Returns ``None`` when any part cannot be
    stored faithfully, so a partial answer is never reused."""
    try:
        figures = [json.loads(fig.to_json()) for fig in (answer.figures or [])]
        meta = answer.meta or {}
        return {
            "text": str(answer.text or ""),
            "route": answer.route,
            "table": table_to_payload(answer.table),
            "figures": figures,
            "insight": _jsonable(answer.insight) if answer.insight else None,
            "meta": _jsonable({k: meta[k] for k in _STORED_META_KEYS if k in meta}),
        }
    except Exception:  # serialisation must never affect the live answer
        log.exception("answer could not be serialised; it will not be stored")
        return None


def answer_from_payload(payload: dict, filtered_df: pd.DataFrame | None) -> Any:
    """Rebuild an ``AssistantAnswer`` identical in content and route to the original."""
    import plotly.io as pio

    from core.schemas import AssistantAnswer

    figures = [pio.from_json(json.dumps(f), skip_invalid=True) for f in payload.get("figures") or []]
    return AssistantAnswer(
        text=payload.get("text", ""),
        table=table_from_payload(payload.get("table")),
        figures=figures,
        insight=payload.get("insight") or None,
        filtered_df=filtered_df,
        route=payload.get("route") or "analytics_engine",
        cacheable=False,
        meta=dict(payload.get("meta") or {}),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Store
# ═══════════════════════════════════════════════════════════════════════════════

class QAStore:
    """Normalised-question → stored-answer map persisted in ``QA.json``.

    File layout::

        {"version": 1, "updated_at": "...",
         "entries": [{"question": "...", "normalized_question": "...",
                      "answer": {...}, "created_at": "...", "as_of_date": "...",
                      "time_relative": false, "model": "...", "prompt_version": "..."}]}
    """

    def __init__(self, path: str | Path = DEFAULT_PATH, *, max_entries: int = 500,
                 max_entry_kb: int = 768, max_age_hours: float = 0.0) -> None:
        p = Path(path)
        self.path = p if p.is_absolute() else PROJECT_ROOT / p
        self.max_entries = max(1, int(max_entries))
        self.max_entry_bytes = max(1, int(max_entry_kb)) * 1024
        self.max_age_seconds = max(0.0, float(max_age_hours)) * 3600
        self._lock = threading.RLock()
        self._entries: dict[str, dict] = {}
        self._mtime: int | None = None
        self._loaded = False

    @classmethod
    def from_settings(cls, settings: Any) -> QAStore:
        """Build the answer cache without sharing the LLM memo file."""

        def resolved_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve()

        qa_path = resolved_path(
            getattr(settings, "qa_cache_path", DEFAULT_PATH)
            or DEFAULT_PATH
        )

        memo_path = resolved_path(
            getattr(
                settings,
                "llm_memo_path",
                "data/llm_memo.json",
            )
            or "data/llm_memo.json"
        )

        if os.path.normcase(str(qa_path)) == os.path.normcase(
            str(memo_path)
        ):
            fallback = resolved_path(DEFAULT_PATH)

            # Also handle an LLM memo explicitly configured as data/QA.json.
            if os.path.normcase(str(fallback)) == os.path.normcase(
                str(memo_path)
            ):
                fallback = resolved_path(
                    "data/qa_responses.json"
                )

            log.warning(
                "Q&A cache and LLM memo paths overlap. "
                "Using a separate Q&A file: %s",
                fallback,
            )
            qa_path = fallback

        store = cls(
            qa_path,
            max_entries=int(
                getattr(settings, "qa_cache_max_entries", 500)
                or 500
            ),
            max_entry_kb=int(
                getattr(settings, "qa_cache_max_entry_kb", 768)
                or 768
            ),
            max_age_hours=float(
                getattr(settings, "qa_cache_max_age_hours", 0.0)
                or 0.0
            ),
        )

        log.info(
            "Q&A cache enabled: path=%s, max_entries=%d",
            store.path,
            store.max_entries,
        )
        return store

    # ── public API ──────────────────────────────────────────────────────────────
    def lookup(self, question: str, *, as_of_date: str) -> dict | None:
        """Stored answer payload for ``question`` or ``None``. Never raises."""
        key = normalize_question(question)
        if not key:
            return None
        try:
            with self._lock:
                self._refresh()
                entry = self._entries.get(key)
            if entry is None or not self._is_fresh(entry, as_of_date):
                return None
            return entry["answer"]
        except Exception:
            log.exception("QA store lookup failed; continuing without it")
            return None

    def store(self, question: str, answer_payload: dict, *, as_of_date: str,
              metadata: dict | None = None) -> bool:
        """Add one entry. Returns ``False`` for duplicates, oversize entries or I/O errors."""
        key = normalize_question(question)
        if not key or not isinstance(answer_payload, dict):
            return False
        entry = {
            "question": " ".join(str(question).split()),
            "normalized_question": key,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "as_of_date": as_of_date,
            "time_relative": is_time_relative(question),
            **{k: _jsonable(v) for k, v in (metadata or {}).items()},
            "answer": answer_payload,
        }
        try:
            encoded = json.dumps(entry, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            log.warning("QA store entry is not valid JSON; skipped")
            return False
        if len(encoded.encode("utf-8")) > self.max_entry_bytes:
            log.info("QA store entry larger than the configured limit; skipped")
            return False
        try:
            with self._lock, self._file_lock():
                self._refresh(force=True)  # merge with entries written by other processes
                existing = self._entries.get(key)
                if existing is not None and self._is_fresh(existing, as_of_date):
                    return False
                self._entries.pop(key, None)  # replace an expired entry
                self._entries[key] = entry
                while len(self._entries) > self.max_entries:
                    self._entries.pop(next(iter(self._entries)))  # oldest first
                self._write()
            log.info("QA store: answer stored (%d entries)", len(self._entries))
            return True
        except Exception:
            log.exception("QA store write failed; answer not stored")
            return False

    # ── internals ───────────────────────────────────────────────────────────────
    def _is_fresh(self, entry: dict, as_of_date: str) -> bool:
        if entry.get("time_relative") and entry.get("as_of_date") != as_of_date:
            return False
        if self.max_age_seconds:
            try:
                created = datetime.fromisoformat(str(entry.get("created_at")))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except ValueError:
                return False
            return (datetime.now(timezone.utc) - created).total_seconds() <= self.max_age_seconds
        return True

    def _file_lock(self):
        if FileLock is None:
            return contextlib.nullcontext()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.path) + ".lock", timeout=10)

    def _current_mtime(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh(self, force: bool = False) -> None:
        mtime = self._current_mtime()
        if self._loaded and not force and mtime == self._mtime:
            return
        self._entries = self._read()
        self._mtime = self._current_mtime()
        self._loaded = True

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            if isinstance(exc, UnicodeDecodeError):
                self._quarantine("not valid UTF-8")
                return {}
            log.warning("QA store unreadable (%s); keeping in-memory entries", type(exc).__name__)
            return dict(self._entries)
        if not raw.strip():
            return {}
        try:
            doc = json.loads(raw)
        except ValueError:
            self._quarantine("invalid JSON")
            return {}
        items = doc if isinstance(doc, list) else doc.get("entries") if isinstance(doc, dict) else None
        if isinstance(items, dict):  # tolerate a mapping of entries
            items = list(items.values())
        if not isinstance(items, list):
            self._quarantine("unexpected structure")
            return {}
        entries: dict[str, dict] = {}
        dropped = duplicates = 0
        for item in items:
            entry = self._valid_entry(item)
            if entry is None:
                dropped += 1
            elif entry["normalized_question"] in entries:
                duplicates += 1
            else:
                entries[entry["normalized_question"]] = entry
        if dropped or duplicates:
            log.warning("QA store: ignored %d invalid and %d duplicate entr(ies)", dropped, duplicates)
        return entries

    @staticmethod
    def _valid_entry(item: Any) -> dict | None:
        if not isinstance(item, dict):
            return None
        question, answer = item.get("question"), item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
            return None
        key = normalize_question(question)
        if not key:
            return None
        return {**item, "normalized_question": key}

    def _quarantine(self, reason: str) -> None:
        target = self.path.with_name(f"{self.path.stem}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}{self.path.suffix}")
        try:
            os.replace(self.path, target)
            log.warning("QA store %s; moved to %s and starting a new store", reason, target.name)
        except OSError:
            log.warning("QA store %s and could not be quarantined; starting in memory", reason)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"version": STORE_VERSION,
               "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "entries": list(self._entries.values())}
        fd, tmp = tempfile.mkstemp(prefix=f"{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=2, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            for attempt in range(5):  # Windows: the target may be briefly held by a reader
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            with contextlib.suppress(OSError):
                if os.path.exists(tmp):
                    os.remove(tmp)
        self._mtime = self._current_mtime()


__all__ = ["QAStore", "normalize_question", "question_may_contain_pii", "depends_on_conversation",
           "is_time_relative", "answer_to_payload", "answer_from_payload"]
