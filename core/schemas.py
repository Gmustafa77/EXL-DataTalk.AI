"""Shared Pydantic contracts.

Agent-specific LLM response models live next to their prompts in ``agents/``.
This module holds the vocabulary several agents share and the application-level
answer contract used by the supervisor, the UI and the Q&A store.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


def normalise_token(value: Any) -> Any:
    """Before-validator helper: trim and upper-case enum-like strings from the LLM."""
    return value.strip().upper().replace(" ", "_").replace("-", "_") if isinstance(value, str) else value


class Grain(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    NONE = "none"


class Priority(StrEnum):
    IMMEDIATE = "Immediate"
    SHORT_TERM = "Short-term"
    STRATEGIC = "Strategic"
    MONITOR = "Monitor"

    @classmethod
    def coerce(cls, value: Any) -> Priority:
        text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
        mapping = {"immediate": cls.IMMEDIATE, "short-term": cls.SHORT_TERM, "shortterm": cls.SHORT_TERM,
                   "strategic": cls.STRATEGIC, "monitor": cls.MONITOR}
        if text not in mapping:
            raise ValueError(f"priority must be one of {[p.value for p in cls]}")
        return mapping[text]


Route = Literal["fast_path", "analytics_engine", "direct_sql_fallback", "rejected", "error"]


class AssistantAnswer(BaseModel):
    """Everything the chat UI renders for one question (one assistant turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(description="Markdown answer shown under 'AI Insight Findings'.")
    table: pd.DataFrame | None = Field(default=None, description="Verified result table, if any.")
    figures: list[Any] = Field(default_factory=list, description="Plotly figures.")
    insight: dict[str, Any] | None = Field(default=None, description="Insight contract for the insight panel.")
    filtered_df: pd.DataFrame | None = Field(
        default=None, description="Population a fast path selected; becomes the dashboard frame.")
    route: Route = "analytics_engine"
    cacheable: bool = Field(default=False, description="Complete and safe enough to store in QA.json.")
    meta: dict[str, Any] = Field(default_factory=dict, description="Call accounting and timings.")
