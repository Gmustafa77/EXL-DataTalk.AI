"""Deterministic replay store for LLM decisions (backend only).

Why this exists
---------------
Temperature 0 and a fixed seed do **not** make hosted LLMs fully deterministic:
the same prompt can return a slightly different plan, a differently shaped SQL
statement (column order, aliases, extra measures) or a reworded narrative. A
differently shaped result table then legitimately produces a different chart.

This store makes every AI decision repeatable: the first validated reply to a
prompt is recorded, and the same prompt later replays that reply instead of
calling the model again. Because every agent builds its prompt deterministically
(``prompts.user_payload`` sorts keys), the chain becomes reproducible end to end:

    same question → same plan → same SQL → same result → same chart → same narrative

Keying rules
------------
* Key = SHA-256 of (purpose, model, response schema, temperature, system prompt,
  canonical user payload). A prompt edit, a model change or different data in
  the payload (for example new evidence values) produces a new key automatically.
* ``as_of_date`` is removed from the key unless the question is time-relative
  ("last 3 months", "YTD" …). A question such as "How many appeals were received
  each month?" therefore keeps the same SQL from one day to the next, while
  "last 3 months" is re-planned each day so its dates stay correct.
* Only replies that passed Pydantic validation are stored. Prompt text is never
  stored — only its hash and the validated model output.

Robustness
----------
Missing, empty or corrupted files never break answering (a corrupted file is
renamed ``*.corrupt-<timestamp>.json``). Writes are atomic (temp file +
``os.replace``) under an in-process lock and a cross-process file lock. Entries
are capped (oldest dropped first) and can optionally expire (``ttl_hours``).

No Streamlit imports.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.qa_store import is_time_relative

try:  # declared dependency; degrade to in-process locking if unavailable
    from filelock import FileLock
except ImportError:  # pragma: no cover
    FileLock = None  # type: ignore[assignment]

log = logging.getLogger("datatalk.llm_memo")

MEMO_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = "data/llm_memo.json"


def _question_of(payload: dict) -> str:
    """The user-facing question a payload is about (planner or per-sub-question agents)."""
    q = payload.get("user_question")
    if isinstance(q, str) and q.strip():
        return q
    sub = payload.get("sub_question")
    if isinstance(sub, dict):
        return str(sub.get("question_text") or "")
    return ""


def canonical_user(user: str) -> str:
    """Canonical form of a user payload for keying (see module docstring)."""
    try:
        payload = json.loads(user)
    except (TypeError, ValueError):
        return user or ""
    if not isinstance(payload, dict):
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    question = _question_of(payload)
    if "as_of_date" in payload and question and not is_time_relative(question):
        payload = {k: v for k, v in payload.items() if k != "as_of_date"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


class LLMMemo:
    """Prompt-hash → validated LLM reply, persisted as JSON."""

    def __init__(self, path: str | Path = DEFAULT_PATH, *, max_entries: int = 2000,
                 ttl_hours: float = 0.0) -> None:
        p = Path(path)
        self.path = p if p.is_absolute() else PROJECT_ROOT / p
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_hours)) * 3600
        self._lock = threading.RLock()
        self._entries: dict[str, dict] = {}
        self._mtime: int | None = None
        self._loaded = False

    @classmethod
    def from_settings(cls, settings: Any) -> LLMMemo:
        return cls(getattr(settings, "llm_memo_path", DEFAULT_PATH) or DEFAULT_PATH,
                   max_entries=int(getattr(settings, "llm_memo_max_entries", 2000) or 2000),
                   ttl_hours=float(getattr(settings, "llm_memo_ttl_hours", 0.0) or 0.0))

    # ── keys ────────────────────────────────────────────────────────────────────
    @staticmethod
    def key(purpose: str, model: str, schema_name: str, system: str, user: str,
            temperature: float | None) -> str:
        parts = [f"v{MEMO_VERSION}", purpose, model, schema_name, f"{temperature}",
                 hashlib.sha256((system or "").encode("utf-8")).hexdigest(), canonical_user(user)]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    # ── public API (never raises) ───────────────────────────────────────────────
    def get(self, key: str) -> dict | None:
        try:
            with self._lock:
                self._refresh()
                entry = self._entries.get(key)
            if entry is None or not self._fresh(entry):
                return None
            value = entry.get("value")
            return value if isinstance(value, dict) else None
        except Exception:
            log.exception("llm memo lookup failed; calling the model")
            return None

    def put(self, key: str, value: dict, *, purpose: str) -> None:
        if not isinstance(value, dict):
            return
        entry = {"purpose": purpose, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "value": value}
        try:
            json.dumps(entry, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return
        try:
            with self._lock, self._file_lock():
                self._refresh(force=True)
                existing = self._entries.get(key)
                if existing is not None and self._fresh(existing):
                    return  # first validated answer wins: keeps replays stable
                self._entries.pop(key, None)
                self._entries[key] = entry
                while len(self._entries) > self.max_entries:
                    self._entries.pop(next(iter(self._entries)))
                self._write()
        except Exception:
            log.exception("llm memo write failed; reply not recorded")

    # ── internals ───────────────────────────────────────────────────────────────
    def _fresh(self, entry: dict) -> bool:
        if not self.ttl_seconds:
            return True
        try:
            created = datetime.fromisoformat(str(entry.get("created_at")))
        except ValueError:
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds() <= self.ttl_seconds

    def _file_lock(self):
        if FileLock is None:
            return contextlib.nullcontext()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.path) + ".lock", timeout=10)

    def _mtime_now(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh(self, force: bool = False) -> None:
        mtime = self._mtime_now()
        if self._loaded and not force and mtime == self._mtime:
            return
        self._entries = self._read()
        self._mtime = self._mtime_now()
        self._loaded = True

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            self._quarantine("not valid UTF-8")
            return {}
        except OSError:
            return dict(self._entries)
        if not raw.strip():
            return {}
        try:
            doc = json.loads(raw)
        except ValueError:
            self._quarantine("invalid JSON")
            return {}
        items = doc.get("entries") if isinstance(doc, dict) else None
        if not isinstance(items, dict):
            self._quarantine("unexpected structure")
            return {}
        return {k: v for k, v in items.items()
                if isinstance(k, str) and isinstance(v, dict) and isinstance(v.get("value"), dict)}

    def _quarantine(self, reason: str) -> None:
        target = self.path.with_name(f"{self.path.stem}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}{self.path.suffix}")
        try:
            os.replace(self.path, target)
            log.warning("llm memo %s; moved to %s and starting a new memo", reason, target.name)
        except OSError:
            log.warning("llm memo %s and could not be quarantined", reason)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"version": MEMO_VERSION, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "entries": self._entries}
        fd, tmp = tempfile.mkstemp(prefix=f"{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(doc, fh, ensure_ascii=False, indent=1, allow_nan=False)
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
        self._mtime = self._mtime_now()


__all__ = ["LLMMemo", "canonical_user"]
