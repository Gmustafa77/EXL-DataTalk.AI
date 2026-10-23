"""Insight generation agent — evidence-first insight composer.

LLM calls: 0 or 1 per answered sub-question (purpose ``insight_narrative``).
The call happens only when ``INSIGHT_LLM_NARRATIVE`` is on, an LLM is supplied and
the result has a numeric metric. Everything else is deterministic.

What it does
  1. Profiles the exact SQL result (dimension / time / measures, metric kind,
     unit, better-direction) — schema-agnostic, no hardcoded table or columns.
  2. Builds an EVIDENCE REGISTRY (F1..Fn): observed values + derived statistics
     (median, gap, spread, weighted average, dispersion, IQR outliers,
     correlation, SLA headroom, excess workload sizing, trend slope ...).
  3. Writes every card section deterministically, each with ONE job:
        key_finding        -> the direct answer + one piece of context
        detailed_analysis  -> the shape of the result (range, spread, outliers…)
        business_impact    -> operational consequence, sized where possible
        reasoning_steps    -> how the answer was derived (SQL scope, metric
                              definition, ranking rule, benchmark, completeness)
        evidence           -> every number, its value and its source
        recommended_actions-> specific, owned, measurable, evidence-linked
        data_caveats       -> only caveats that are actually true for this result
  4. Optionally lets the LLM rewrite key_finding / detailed_analysis /
     business_impact / actions for fluency. The reply is validated against
     :class:`InsightNarrative`, then every sentence is checked:
       * every number must match an evidence value at its displayed precision
         (13.93 may be written 13.9, but not 14 — blocks invented thresholds)
       * causal wording ("because", "driven by" …) is rejected
       * sections must not restate each other (similarity gate)
     Any section that fails falls back to the deterministic version.

No Streamlit imports. Safe to unit-test and to reuse behind an API.
"""
from __future__ import annotations

import difflib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from core.llm import LLMResponseError, LLMUnavailableError, StructuredLLM
from core.schemas import Priority

log = logging.getLogger("datatalk.insight_composer")

COMPOSER_VERSION = "insight-composer-v2.1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ComposerConfig:
    sla_days: float | None = None            # configured SLA (days); None disables SLA logic
    min_group_volume: int = 5                   # groups below this volume are flagged as unstable
    polarity_overrides: dict = field(default_factory=dict)  # {"Column": "lower"|"higher"|"neutral"}
    enable_llm_narrative: bool = True
    llm_temperature: float = 0.0
    similarity_threshold: float = 0.72          # bullets above this similarity are treated as repeats
    max_bullets: int = 7
    max_actions: int = 4


LOWER_BETTER = {
    "turnaround", "tat", "age", "aging", "aged", "overdue", "backlog", "escalated",
    "escalation", "escalations", "overturn", "overturned", "overturns", "denial",
    "denials", "denied", "pending", "delay", "delayed", "late", "breach", "breaches",
    "error", "errors", "defect", "defects", "transfer", "transferred", "transfers",
    "reopen", "reopened", "risk", "wait", "duration", "days", "hours", "open",
    "noncompliant", "complaint", "complaints", "rework", "cost",
}
HIGHER_BETTER = {
    "compliance", "compliant", "ontime", "resolved", "resolution", "accuracy",
    "quality", "satisfaction", "throughput", "productivity", "success", "sla_met",
}
TIME_WORDS = {"date", "month", "year", "quarter", "week", "period", "yearmonth"}
COUNT_WORDS = {"count", "cases", "case", "volume", "total", "number", "num", "appeals",
               "records", "claims", "items", "closed", "opened"}
RATE_WORDS = {"rate", "percent", "percentage", "pct", "share"}
AVG_WORDS = {"avg", "average", "mean", "median"}
SPEED_WORDS = {"turnaround", "tat", "resolution", "handling", "processing", "cycle", "response"}
ID_WORDS = {"id", "identifier", "code", "key", "uuid"}
_DIM_STOP = {"name", "names", "id", "identifier", "code", "desc", "description", "short",
             "current", "routed", "to", "label", "the", "of", "by"}
_VOL_STOP = {"count", "total", "number", "num", "of", "no", "n", "sum"}


# ═══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _words(name: Any) -> list:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return re.findall(r"[a-z]+", s.lower())


def _fmt_num(x: float, decimals: int) -> str:
    if decimals == 0 and abs(x - round(x)) > 1e-9:
        decimals = 1
    return f"{x:,.{decimals}f}"


def _pct(x: float, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}%"


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _plural(noun: str) -> str:
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _dim_nouns(col: str) -> tuple:
    w = [x for x in _words(col) if x not in _DIM_STOP]
    noun = w[-1] if w else "group"
    return noun, _plural(noun)


def _vol_noun(col: str) -> str:
    w = [x for x in _words(col) if x not in _VOL_STOP]
    if not w:
        return "records"
    if not w[-1].endswith("s"):
        w[-1] = _plural(w[-1])
    return " ".join(w)


def _norm_text(t: str) -> str:
    t = re.sub(r"\[F\d+(?:\s*,\s*F\d+)*\]", " ", str(t or ""))
    t = re.sub(r"[\d.,%×]+", " ", t.lower())
    return re.sub(r"[^a-z ]+", " ", re.sub(r"\s+", " ", t)).strip()


def similarity(a: str, b: str) -> float:
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return None
    rx, ry = x[ok].rank().values, y[ok].rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC SPEC
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class MetricSpec:
    column: str
    kind: str          # rate | duration | average | score | ratio | count | measure
    polarity: str      # lower | higher | neutral
    unit: str          # " days", "%", ""
    decimals: int
    scale: float
    additive: bool
    speed_word: bool

    def fmt(self, v: Any, delta: bool = False, signed: bool = False) -> str:
        f = _safe_float(v)
        if f is None:
            return "N/A"
        x = f * self.scale
        body = _fmt_num(abs(x) if (delta and not signed) else x, self.decimals)
        if signed and x > 0:
            body = "+" + body
        unit = " pp" if (delta and self.unit == "%") else self.unit
        return f"{body}{unit}"

    @property
    def better_label(self) -> str:
        return {"lower": "lower is better", "higher": "higher is better"}.get(self.polarity, "no better/worse direction")


def build_metric_spec(col: str, series: pd.Series, overrides: dict | None = None) -> MetricSpec:
    w = _words(col)
    ws = set(w)
    joined = " ".join(w)
    vals = pd.to_numeric(series, errors="coerce").dropna()

    is_ratio = "ratio" in ws
    is_rate = (bool(ws & RATE_WORDS) or "%" in str(col)) and not is_ratio
    is_avg = bool(ws & AVG_WORDS)
    unit = ""
    if ws & {"day", "days"}:
        unit = " days"
    elif ws & {"hour", "hours", "hrs"}:
        unit = " hours"
    elif ws & {"minute", "minutes", "mins"}:
        unit = " minutes"
    is_duration = bool(unit) or bool(ws & {"turnaround", "tat", "age", "aging", "duration", "elapsed"})
    is_score = bool(ws & {"score", "index"})
    integer_valued = (not vals.empty) and bool(np.all(np.isclose(vals, np.round(vals)))) and vals.min() >= 0

    if is_rate:
        kind = "rate"
    elif is_ratio:
        kind = "ratio"
    elif is_duration:
        kind = "duration"
    elif is_avg:
        kind = "average"
    elif is_score:
        kind = "score"
    elif (ws & COUNT_WORDS) or integer_valued:
        kind = "count"
    else:
        kind = "measure"

    additive = kind == "count" or (kind == "measure" and bool(ws & {"amount", "sum", "total", "cost"}))
    if kind == "duration" and not is_avg and ws & {"total", "sum"}:
        additive = True

    scale = 100.0 if (kind == "rate" and not vals.empty and vals.min() >= 0 and vals.max() <= 1.0) else 1.0
    if kind == "rate":
        unit, decimals = "%", 1
    elif kind == "count":
        decimals = 0
    else:
        decimals = 0 if integer_valued else 2

    pol = (overrides or {}).get(col)
    if pol not in {"lower", "higher", "neutral"}:
        if re.search(r"\bnon ?complian", joined) or "noncompliant" in ws:
            pol = "lower"
        elif ws & LOWER_BETTER:
            pol = "lower"
        elif ws & HIGHER_BETTER:
            pol = "higher"
        else:
            pol = "neutral"

    return MetricSpec(col, kind, pol, unit, decimals, scale, additive, bool(ws & SPEED_WORDS))


# ═══════════════════════════════════════════════════════════════════════════════
# SQL HELPERS (regex based — no sqlglot dependency)
# ═══════════════════════════════════════════════════════════════════════════════
def _alias_expression(sql: str, alias: str) -> str:
    """Return the SELECT expression that produced `alias` (balanced-paren walk-back)."""
    m = re.search(r"\bAS\s+`?" + re.escape(alias) + r"`?(?=\s*,|\s+FROM\b|\s*$)", sql, re.I)
    if not m:
        return ""
    i, depth = m.start() - 1, 0
    while i >= 0:
        ch = sql[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
        elif ch == "," and depth == 0:
            break
        if depth == 0 and re.match(r"(?i)SELECT\b", sql[i:i + 6]) and (i == 0 or not sql[i - 1].isalnum()):
            i += 5
            break
        i -= 1
    return sql[i + 1:m.start()].strip()


def parse_sql(sql: str) -> dict:
    out = {"where": "", "limit": None, "order": "", "group_by": ""}
    s = re.sub(r"\s+", " ", (sql or "").strip())
    if not s:
        return out
    m = re.search(r"\bWHERE\b(.*?)(?=\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|$)", s, re.I)
    if m:
        out["where"] = m.group(1).strip()
    m = re.search(r"\bLIMIT\s+(\d+)\s*$", s, re.I)
    if m:
        out["limit"] = int(m.group(1))
    m = re.search(r"\bORDER\s+BY\b(.*?)(?=\bLIMIT\b|$)", s, re.I)
    if m:
        out["order"] = m.group(1).strip()
    m = re.search(r"\bGROUP\s+BY\b(.*?)(?=\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|$)", s, re.I)
    if m:
        out["group_by"] = m.group(1).strip()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERIC VERIFICATION (strict, precision-aware)
# ═══════════════════════════════════════════════════════════════════════════════
_NUM_RE = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")


def _numbers_with_decimals(text: str) -> list:
    out = []
    for tok in _NUM_RE.findall(str(text or "")):
        clean = tok.replace(",", "")
        f = _safe_float(clean)
        if f is None:
            continue
        dec = len(clean.split(".")[1]) if "." in clean else 0
        out.append((abs(f), dec))
    return out


def build_allowed_numbers(*texts: Any) -> list:
    allowed = set()
    for t in texts:
        for v, d in _numbers_with_decimals(t):
            allowed.add((round(v, 6), d))
    return sorted(allowed)


def verify_text(text: str, allowed: list) -> bool:
    """True when every number in `text` matches an evidence number at its displayed precision.
    A value displayed with D decimals may be written with D or D-1 decimals (13.93 -> 13.9),
    never coarser (13.93 -> 14 is rejected, which blocks invented thresholds)."""
    for v, d in _numbers_with_decimals(text):
        if d == 0 and (v in (0, 1) or 1900 <= v <= 2100):
            continue
        ok = False
        for a, ad in allowed:
            if d < max(ad - 1, 0):
                continue
            if abs(round(a, d) - v) < 1e-9:
                ok = True
                break
        if not ok:
            return False
    return True


def verify_with_reason(text: str, allowed: list) -> str | None:
    """Return the first number in `text` that cannot be matched, or None."""
    for v, d in _numbers_with_decimals(text):
        if d == 0 and (v in (0, 1) or 1900 <= v <= 2100):
            continue
        if not any(d >= max(ad - 1, 0) and abs(round(a, d) - v) < 1e-9 for a, ad in allowed):
            return f"{v:.{d}f}"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
class Evidence:
    def __init__(self):
        self.items: list = []
        self._by_key: dict = {}

    def add(self, key: str, label: str, display: str, source: str, value: Any = None) -> str:
        if key in self._by_key:
            return self._by_key[key]["id"]
        fid = f"F{len(self.items) + 1}"
        item = {"id": fid, "key": key, "label": label, "display": str(display),
                "source": source, "value": value}
        self.items.append(item)
        self._by_key[key] = item
        return fid

    def has(self, key: str) -> bool:
        return key in self._by_key

    def d(self, key: str) -> str:
        return self._by_key[key]["display"] if key in self._by_key else ""

    def v(self, key: str) -> Any:
        return self._by_key[key]["value"] if key in self._by_key else None

    def ids(self, *keys: str) -> list:
        return [self._by_key[k]["id"] for k in keys if k in self._by_key]


def _b(ev: Evidence, text: str, *keys: str) -> dict:
    return {"text": text, "refs": ev.ids(*keys)}


def _action(ev: Evidence, priority: str, action: str, owner: str, timeframe: str,
            success: str, *keys: str) -> dict:
    return {"priority": priority, "action": action, "owner": owner, "timeframe": timeframe,
            "success_measure": success, "evidence_ids": ev.ids(*keys), "evidence": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Profile:
    df: pd.DataFrame
    question: str
    sql: str
    sqlp: dict
    cfg: ComposerConfig
    time_col: str | None
    dims: list
    metric: MetricSpec | None
    volume: MetricSpec | None
    other_metrics: list
    analysis_type: str


def _is_time_column(col: str, s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    w = set(_words(col))
    if not (w & TIME_WORDS):
        return False
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty or (w & (AVG_WORDS | RATE_WORDS | {"count", "total", "days"})):
            return False
        if "year" in w:
            return bool(vals.between(1900, 2100).all())
        return bool(vals.between(1, 53).all())
    sample = s.dropna().astype(str).head(30)
    if sample.empty:
        return False
    pat = re.compile(r"^\d{4}([-/]\d{1,2}([-/]\d{1,2})?|[-\s]?Q[1-4])?$|^\d{4}-\d{2}-\d{2}")
    if sample.map(lambda v: bool(pat.match(v.strip()))).mean() >= 0.7:
        return True
    return pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.7


def _to_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if vals.dropna().between(1900, 2100).all():
            return pd.to_datetime(vals.astype("Int64").astype(str), format="%Y", errors="coerce")
        return pd.Series(pd.NaT, index=s.index)
    txt = s.astype(str).str.strip()
    q = txt.str.extract(r"^(\d{4})[-\s]?Q([1-4])$")
    if q.notna().all(axis=1).mean() >= 0.7:
        return pd.to_datetime(q[0] + "-" + ((q[1].astype(float) - 1) * 3 + 1).astype(int).astype(str) + "-01",
                              errors="coerce")
    return pd.to_datetime(txt, errors="coerce")


def _time_label(v: Any, original: Any) -> str:
    if isinstance(v, pd.Timestamp) and not pd.isna(v):
        if isinstance(original, (pd.Timestamp, np.datetime64)) or re.match(r"^\d{4}-\d{2}-\d{2}", str(original)):
            return v.strftime("%Y-%m-%d") if v.day != 1 else v.strftime("%Y-%m")
    return str(original)


def _pick_metric(numeric: list, specs: dict, question: str, sqlp: dict) -> str | None:
    if not numeric:
        return None
    order = sqlp.get("order", "")
    if order:
        first = order.split(",")[0]
        for c in numeric:
            if c.lower() in first.lower().replace("`", ""):
                return c
    synonyms = {"tat": "turnaround", "turnaround": "tat", "overturns": "overturn", "volume": "count"}
    qw = set(_words(question))
    qw |= {synonyms[w] for w in qw if w in synonyms}
    generic = {"avg", "average", "total", "count", "number", "rate", "of", "the"}
    scored = sorted(
        ((len((set(_words(c)) - generic) & qw), i, c) for i, c in enumerate(numeric)),
        key=lambda t: (-t[0], t[1]),
    )
    if scored and scored[0][0] > 0:
        return scored[0][2]
    non_count = [c for c in numeric if specs[c].kind != "count"]
    return (non_count or numeric)[0]


def profile_result(df: pd.DataFrame, question: str, sql: str, cfg: ComposerConfig) -> Profile:
    sqlp = parse_sql(sql)
    time_col, dims, numeric = None, [], []
    for c in df.columns:
        s = df[c]
        if time_col is None and _is_time_column(c, s):
            time_col = c
            continue
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if is_num and not (set(_words(c)) & ID_WORDS):
            numeric.append(c)
        else:
            dims.append(c)

    specs = {c: build_metric_spec(c, df[c], cfg.polarity_overrides) for c in numeric}
    metric_col = _pick_metric(numeric, specs, question, sqlp)
    metric = specs.get(metric_col) if metric_col else None

    volume = None
    if metric is not None:
        counts = [specs[c] for c in numeric if c != metric_col and specs[c].kind == "count"]
        if counts:
            pref = [s for s in counts if set(_words(s.column)) & {"cases", "case", "count", "volume", "appeals"}]
            volume = (pref or counts)[0]
    others = [specs[c] for c in numeric if c != metric_col and (volume is None or c != volume.column)]

    if df.empty:
        atype = "empty"
    elif metric is None:
        atype = "records"
    elif len(df) == 1:
        atype = "scalar"
    elif time_col is not None:
        atype = "trend_by_group" if dims and df[dims[0]].nunique() > 1 else "trend"
    elif dims:
        atype = "ranking"
    else:
        atype = "scalar" if len(df) == 1 else "records"
    return Profile(df, question, sql, sqlp, cfg, time_col, dims, metric, volume, others, atype)


def _direction(question: str, sqlp: dict, metric: MetricSpec) -> str:
    q = (question or "").lower()
    pol = metric.polarity
    if pol != "neutral" and re.search(r"\b(best|top[- ]performing|strongest|most efficient|fastest|quickest)\b", q):
        return "min" if pol == "lower" else "max"
    if pol != "neutral" and re.search(r"\b(worst|poorest|weakest|under-?performing|bottom[- ]performing|least efficient|slowest|lagging)\b", q):
        return "max" if pol == "lower" else "min"
    if re.search(r"\b(lowest|least|fewest|minimum|smallest|shortest)\b", q):
        return "min"
    if re.search(r"\b(highest|most|maximum|largest|biggest|longest|greatest|top|peak)\b", q):
        return "max"
    order = sqlp.get("order", "")
    if order and metric.column.lower() in order.split(",")[0].lower().replace("`", ""):
        return "max" if re.search(r"\bDESC\b", order.split(",")[0], re.I) else "min"
    return "min" if pol == "lower" else "max"


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED REASONING / CAVEATS
# ═══════════════════════════════════════════════════════════════════════════════
def _reasoning_common(p: Profile, ev: Evidence, intent_line: str, extra: list) -> list:
    steps = [intent_line]
    m = p.metric
    if m is not None:
        expr = _alias_expression(p.sql, m.column)
        if expr:
            steps.append(f"Metric definition: {m.column} = {expr.replace('`', '')}.")
        steps.append(
            f"Direction: {m.column} is treated as {m.better_label}"
            + (" because it measures elapsed time, backlog or adverse outcomes." if m.polarity == "lower"
               else " because it measures a favourable outcome." if m.polarity == "higher"
               else "; results are described as higher/lower, not better/worse.")
        )
    if p.sqlp.get("where"):
        steps.append(f"Population: only rows matching {p.sqlp['where'].replace('`', '')} are included.")
    else:
        steps.append("Population: no row filter was applied — the full table was aggregated.")
    steps.extend(extra)
    n, lim = len(p.df), p.sqlp.get("limit")
    if lim:
        if n < lim:
            steps.append(f"Completeness: the query returned {n} row(s) under a LIMIT of {lim}, so every group "
                         "matching the filters is included.")
        else:
            steps.append(f"Completeness: the query returned {n} row(s), equal to its LIMIT of {lim}; groups "
                         "beyond the limit are not included.")
    steps.append("Verification: every number on this card is traced to the Evidence tab; AI-written sentences "
                 "containing numbers that cannot be traced are removed before display.")
    return steps


def _caveats_common(p: Profile) -> list:
    out = []
    n, lim = len(p.df), p.sqlp.get("limit")
    if lim and n >= lim:
        out.append(f"The query returned {n} rows, equal to its LIMIT of {lim}; lower-ranked groups may be "
                   "excluded, so medians and shares describe only the returned rows.")
    if re.search(r"'CLOSED'", p.sqlp.get("where", ""), re.I):
        out.append("Only closed cases are included; open cases are not reflected in this metric.")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# WRITER — RANKING (category + measure)
# ═══════════════════════════════════════════════════════════════════════════════
def _compose_ranking(p: Profile) -> dict:
    cfg, m, vspec = p.cfg, p.metric, p.volume
    df = p.df.copy()
    if len(p.dims) >= 2:
        label_col = "__label"
        df[label_col] = df[p.dims[0]].astype(str) + " · " + df[p.dims[1]].astype(str)
        sing, plur = "combination", "combinations"
        dim_name = f"{p.dims[0]} × {p.dims[1]}"
    else:
        label_col = p.dims[0]
        sing, plur = _dim_nouns(label_col)
        dim_name = label_col
    df["__v"] = pd.to_numeric(df[m.column], errors="coerce")
    df = df.dropna(subset=["__v"])
    df[label_col] = df[label_col].astype(str).str.strip()
    if vspec is not None:
        df["__w"] = pd.to_numeric(df[vspec.column], errors="coerce")
    n = len(df)
    ev = Evidence()
    col = m.column
    if n < 2:
        return _compose_scalar(p)

    direction = _direction(p.question, p.sqlp, m)
    ordered = df.sort_values("__v", ascending=(direction == "min"), kind="mergesort").reset_index(drop=True)
    focus, contrast = ordered.iloc[0], ordered.iloc[-1]
    ties = ordered.loc[np.isclose(ordered["__v"], focus["__v"]), label_col].tolist()

    asc = df.sort_values("__v", kind="mergesort")
    low_row, high_row = asc.iloc[0], asc.iloc[-1]
    if m.polarity == "lower":
        best, worst = low_row, high_row
    elif m.polarity == "higher":
        best, worst = high_row, low_row
    else:
        best, worst = None, None
    focus_is_best = best is not None and focus[label_col] == best[label_col]

    values = df["__v"]
    median, mean = float(values.median()), float(values.mean())
    std = float(values.std(ddof=0))
    fl, fv = str(focus[label_col]), float(focus["__v"])
    cl, cv = str(contrast[label_col]), float(contrast["__v"])

    # ── evidence ──
    ev.add("n", f"{plur.title()} compared", f"{n}", f"Row count of the query result ({dim_name})", n)
    ev.add("focus", f"{fl} — {col}", m.fmt(fv), f"Result row '{fl}', column '{col}'", fv)
    ev.add("contrast", f"{cl} — {col}", m.fmt(cv), f"Result row '{cl}', column '{col}'", cv)
    ev.add("median", f"Median {col} across {n} {plur}", m.fmt(median), f"Derived: median of '{col}' over {n} rows", median)
    ev.add("mean", f"Simple average {col}", m.fmt(mean), f"Derived: unweighted mean of '{col}' over {n} rows", mean)
    diff = fv - median
    ev.add("gap_med", f"{fl} vs median ({col})", m.fmt(diff, delta=True), "Derived: focus value − median", diff)
    if abs(median) > 1e-12 and m.kind != "rate":
        ev.add("gap_med_pct", f"{fl} vs median (relative)", _pct(abs(diff) / abs(median) * 100),
               "Derived: |focus − median| ÷ median", abs(diff) / abs(median) * 100)
    spread = float(high_row["__v"] - low_row["__v"])
    ev.add("spread", f"Spread of {col} (highest − lowest)", m.fmt(spread, delta=True),
           "Derived: max − min of the result", spread)
    if float(low_row["__v"]) > 0 and m.kind != "rate":
        ev.add("ratio", f"{high_row[label_col]} ÷ {low_row[label_col]}",
               f"{float(high_row['__v']) / float(low_row['__v']):.2f}×", "Derived: max ÷ min",
               float(high_row["__v"]) / float(low_row["__v"]))
    ev.add("low", f"Lowest {col}: {low_row[label_col]}", m.fmt(low_row["__v"]),
           f"Result row '{low_row[label_col]}'", float(low_row["__v"]))
    ev.add("high", f"Highest {col}: {high_row[label_col]}", m.fmt(high_row["__v"]),
           f"Result row '{high_row[label_col]}'", float(high_row["__v"]))
    runner = ordered.iloc[1]
    margin = abs(float(runner["__v"]) - fv)
    ev.add("runner", f"Next {sing}: {runner[label_col]}", m.fmt(runner["__v"]),
           f"Result row '{runner[label_col]}'", float(runner["__v"]))
    ev.add("margin", f"Margin of {fl} over {runner[label_col]}", m.fmt(margin, delta=True),
           "Derived: |runner-up − focus|", margin)
    if m.polarity != "neutral":
        better_n = int((values < median).sum() if m.polarity == "lower" else (values > median).sum())
        ev.add("better_n", f"{plur.title()} better than the median", f"{better_n}",
               "Derived: count of rows on the better side of the median", better_n)
    if abs(mean) > 1e-12:
        cvp = std / abs(mean) * 100
        ev.add("cv", "Coefficient of variation", _pct(cvp), "Derived: population std ÷ mean", cvp)
    outliers, lf, uf = [], None, None
    if n >= 4:
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if float(values.min()) >= 0:
            lf = max(lf, 0.0)
        if m.kind == "rate":
            uf = min(uf, 1.0 if m.scale == 100 else 100.0)
        ev.add("lf", "Lower IQR fence", m.fmt(lf), "Derived: Q1 − 1.5 × IQR", lf)
        ev.add("uf", "Upper IQR fence", m.fmt(uf), "Derived: Q3 + 1.5 × IQR", uf)
        outliers = df.loc[(values < lf) | (values > uf), label_col].astype(str).tolist()

    # volume-related evidence
    small = []
    rho = None
    vnoun = _vol_noun(vspec.column) if vspec is not None else "records"
    if vspec is not None and df["__w"].notna().all() and df["__w"].sum() > 0:
        total = float(df["__w"].sum())
        ev.add("vol_total", f"Total {vnoun}", vspec.fmt(total), f"Derived: sum of '{vspec.column}'", total)
        fw = float(focus["__w"])
        ev.add("vol_focus", f"{fl} — {vspec.column}", vspec.fmt(fw), f"Result row '{fl}', column '{vspec.column}'", fw)
        ev.add("vol_focus_share", f"{fl} share of {vnoun}", _pct(fw / total * 100),
               f"Derived: {fl} ÷ total", fw / total * 100)
        cw = float(contrast["__w"])
        ev.add("vol_contrast", f"{cl} — {vspec.column}", vspec.fmt(cw), f"Result row '{cl}', column '{vspec.column}'", cw)
        if not m.additive:
            wavg = float((df["__v"] * df["__w"]).sum() / total)
            ev.add("wavg", f"Volume-weighted average {col}", m.fmt(wavg),
                   f"Derived: Σ('{col}' × '{vspec.column}') ÷ Σ('{vspec.column}')", wavg)
        vol_med = float(df["__w"].median())
        small = df.loc[(df["__w"] < cfg.min_group_volume) | (df["__w"] < 0.25 * vol_med), label_col].astype(str).tolist()
        rho = _spearman(df["__v"], df["__w"])
        if rho is not None:
            ev.add("rho", f"Spearman correlation: {vspec.column} vs {col}", f"{rho:.2f}",
                   "Derived: rank correlation across rows", rho)

        # impact sizing
        worse_mask = (df["__v"] > median) if m.polarity == "lower" else (df["__v"] < median) if m.polarity == "higher" else None
        if worse_mask is not None and worse_mask.any():
            affected = float(df.loc[worse_mask, "__w"].sum())
            ev.add("affected", f"{vnoun.capitalize()} in {plur} worse than the median", vspec.fmt(affected),
                   f"Derived: Σ '{vspec.column}' where '{col}' is worse than the median", affected)
            if m.kind in {"duration", "average"} and m.unit in {" days", " hours"}:
                unit_word = "case-days" if m.unit == " days" else "case-hours"
                excess = float((abs(df.loc[worse_mask, "__v"] - median) * df.loc[worse_mask, "__w"]).sum())
                ev.add("excess", f"Excess {unit_word} vs median pace", f"{excess:,.0f} {unit_word}",
                       f"Derived: Σ |'{col}' − median| × '{vspec.column}' for rows worse than the median", excess)
                wr = worst if worst is not None else contrast
                wexcess = abs(float(wr["__v"]) - median) * float(wr["__w"])
                ev.add("excess_worst", f"Excess {unit_word}: {wr[label_col]}", f"{wexcess:,.0f} {unit_word}",
                       f"Derived: |{wr[label_col]} − median| × its '{vspec.column}'", wexcess)
            elif m.kind == "rate":
                frac = (1.0 if m.scale == 100 else 0.01)
                excess = float((abs(df.loc[worse_mask, "__v"] - median) * frac * df.loc[worse_mask, "__w"]).sum())
                ev.add("excess", f"Excess cases implied by {col} above/below median",
                       f"{excess:,.0f}", f"Derived: Σ |rate − median| × '{vspec.column}' for rows worse than the median", excess)

    if m.additive and values.min() >= 0 and values.sum() > 0:
        tot = float(values.sum())
        shares = values / tot
        ev.add("m_total", f"Total {col}", m.fmt(tot), f"Derived: sum of '{col}'", tot)
        ev.add("top_share", f"{high_row[label_col]} share of total {col}", _pct(float(high_row['__v']) / tot * 100),
               "Derived: max ÷ total", float(high_row["__v"]) / tot * 100)
        ev.add("top3_share", f"Top 3 share of total {col}", _pct(float(shares.nlargest(3).sum() * 100)),
               "Derived: sum of three largest ÷ total", float(shares.nlargest(3).sum() * 100))
        hhi = float((shares ** 2).sum() * 10000)
        ev.add("hhi", "Concentration (HHI)", f"{hhi:,.0f}", "Derived: Σ share² × 10,000", hhi)

    sla = cfg.sla_days if (m.unit == " days" and m.kind in {"duration", "average"} and cfg.sla_days) else None
    if sla:
        ev.add("sla", "Configured SLA", f"{_fmt_num(float(sla), 0)} days", "Application configuration (SLA_DAYS)", sla)
        above = df.loc[df["__v"] > sla, label_col].astype(str).tolist()
        ev.add("sla_above", f"{plur.title()} averaging above the SLA", f"{len(above)}",
               "Derived: count of rows with value > SLA", len(above))
        headroom = sla - float(high_row["__v"])
        ev.add("sla_headroom", f"Headroom of {high_row[label_col]} to SLA", m.fmt(headroom, delta=True),
               "Derived: SLA − highest value", headroom)

    # ── words ──
    extreme = "lowest" if direction == "min" else "highest"
    if m.speed_word and m.kind == "duration":
        cmp_word = "faster" if diff < 0 else "slower"
    elif m.polarity == "neutral":
        cmp_word = "below" if diff < 0 else "above"
    else:
        cmp_word = "lower" if diff < 0 else "higher"
    than = "" if cmp_word in {"above", "below"} else " than"
    rank_word = ""
    if m.polarity != "neutral":
        rank_word = f" — the best of {n} {plur}" if focus_is_best else f" — the weakest of {n} {plur}"
    rel = f" ({ev.d('gap_med_pct')})" if ev.has("gap_med_pct") else ""
    tie_txt = f" (tied with {', '.join(t for t in ties if t != fl)})" if len(ties) > 1 else ""

    # KEY FINDING ─ the direct answer + one piece of context
    if abs(diff) < 1e-12:
        kf_text = (f"{fl} has the {extreme} {col} at {ev.d('focus')}{tie_txt}{rank_word}, "
                   f"in line with the median {sing} ({ev.d('median')}).")
        kf = _b(ev, kf_text, "focus", "median", "n")
    else:
        kf_text = (f"{fl} has the {extreme} {col} at {ev.d('focus')}{tie_txt}{rank_word}, "
                   f"{ev.d('gap_med')}{rel} {cmp_word}{than} the median {sing} ({ev.d('median')}).")
        kf = _b(ev, kf_text, "focus", "gap_med", "gap_med_pct", "median", "n")

    # DETAILED ANALYSIS ─ the shape of the result
    da = []
    rng = (f"Across {n} {plur}, {col} ranges from {ev.d('low')} ({low_row[label_col]}) to "
           f"{ev.d('high')} ({high_row[label_col]}), a spread of {ev.d('spread')}")
    rng += f"; the highest is {ev.d('ratio')} the lowest." if ev.has("ratio") else "."
    da.append(_b(ev, rng, "n", "low", "high", "spread", "ratio"))
    spread_frac = margin / spread if spread > 0 else 0
    strength = "narrow" if spread_frac < 0.10 else "clear" if spread_frac > 0.30 else "moderate"
    if m.polarity != "neutral" and not focus_is_best:
        pos_txt = f"shows how {'clearly' if strength == 'clear' else 'narrowly'} {fl} trails the rest"
    else:
        pos_txt = f"shows how {'secure' if strength == 'clear' else 'sensitive'} this position is"
    da.append(_b(ev, f"The next-closest {sing}, {runner[label_col]}, is at {ev.d('runner')}; the {strength} gap of "
                     f"{ev.d('margin')} to {fl} {pos_txt}.", "runner", "margin"))
    if ev.has("better_n"):
        da.append(_b(ev, f"{ev.d('better_n')} of {n} {plur} perform better than the median of {ev.d('median')}; "
                         f"the simple average is {ev.d('mean')}.", "better_n", "n", "median", "mean"))
    if ev.has("vol_focus"):
        t = (f"{fl} handled {ev.d('vol_focus')} {vnoun} ({ev.d('vol_focus_share')} of the "
             f"{ev.d('vol_total')} total)")
        t += (" — a small base, so its result is sensitive to a handful of cases." if fl in small else ".")
        da.append(_b(ev, t, "vol_focus", "vol_focus_share", "vol_total"))
    if ev.has("wavg"):
        w, s_ = ev.v("wavg"), mean
        if abs(w - s_) / max(abs(s_), 1e-9) > 0.02:
            direction_w = "higher" if w > s_ else "lower"
            da.append(_b(ev, f"Weighted by {vnoun}, the average is {ev.d('wavg')} versus a simple average of "
                             f"{ev.d('mean')}: higher-volume {plur} sit {direction_w} than the typical {sing}, "
                             f"so the experience of the typical case differs from the typical {sing}.",
                         "wavg", "mean"))
        else:
            da.append(_b(ev, f"Weighting by {vnoun} barely changes the picture ({ev.d('wavg')} weighted vs "
                             f"{ev.d('mean')} simple), so volume mix is not distorting the comparison.",
                         "wavg", "mean"))
    if ev.has("cv"):
        cvv = ev.v("cv")
        level = "low" if cvv < 10 else "high" if cvv > 25 else "moderate"
        da.append(_b(ev, f"Variation across {plur} is {level} (coefficient of variation {ev.d('cv')}).", "cv"))
    if lf is not None:
        if outliers:
            da.append(_b(ev, f"{', '.join(outliers[:4])} fall outside the IQR outlier fences "
                             f"({ev.d('lf')} to {ev.d('uf')}) and should be read as exceptions rather than "
                             f"typical performance.", "lf", "uf"))
        else:
            da.append(_b(ev, f"No {sing} falls outside the IQR outlier fences ({ev.d('lf')} to {ev.d('uf')}), "
                             f"so the ranking reflects a gradual spread rather than one extreme {sing}.", "lf", "uf"))
    if rho is not None:
        if abs(rho) >= 0.5:
            da.append(_b(ev, f"Across {plur}, higher {vnoun} tends to coincide with "
                             f"{'higher' if rho > 0 else 'lower'} {col} (Spearman ρ = {ev.d('rho')}); this is an "
                             f"association, not proof of cause.", "rho"))
        else:
            da.append(_b(ev, f"There is no consistent relationship between {vnoun} and {col} across {plur} "
                             f"(Spearman ρ = {ev.d('rho')}).", "rho"))
    if ev.has("top_share"):
        hh = ev.v("hhi")
        lvl = "high" if hh >= 2500 else "moderate" if hh >= 1500 else "low"
        da.append(_b(ev, f"{high_row[label_col]} accounts for {ev.d('top_share')} of the total and the top three "
                         f"for {ev.d('top3_share')} — {lvl} concentration (HHI {ev.d('hhi')}).",
                     "top_share", "top3_share", "hhi"))

    # BUSINESS IMPACT ─ operational consequence
    bi = []
    if ev.has("excess") and m.kind in {"duration", "average"}:
        bi.append(_b(ev, f"If every {sing} slower than the median worked at the median pace ({ev.d('median')}), "
                         f"about {ev.d('excess')} of elapsed handling time would be removed across "
                         f"{ev.d('affected')} {vnoun}.", "excess", "median", "affected"))
        if worst is not None and ev.has("excess_worst"):
            bi.append(_b(ev, f"{worst[label_col]} is the single largest opportunity: at {m.fmt(worst['__v'])} it "
                             f"contributes {ev.d('excess_worst')} of that excess.", "excess_worst"))
    elif ev.has("excess") and m.kind == "rate":
        bi.append(_b(ev, f"Bringing every {sing} on the wrong side of the median {col} to the median would change "
                         f"the outcome of about {ev.d('excess')} {vnoun}.", "excess", "median"))
    if best is not None and worst is not None and str(best[label_col]) != str(worst[label_col]):
        bt = (f"{best[label_col]} is the internal benchmark: if {worst[label_col]} matched it, its {col} would "
              f"improve by {ev.d('spread')}")
        bt += f" — the gap is {ev.d('ratio')} between the two ends." if ev.has("ratio") else "."
        bi.append(_b(ev, bt, "spread", "ratio"))
    if sla:
        above_n = ev.v("sla_above")
        if above_n == 0:
            bi.append(_b(ev, f"All {n} {plur} average inside the SLA of {ev.d('sla')}, with {ev.d('sla_headroom')} of "
                             f"headroom for the slowest; averages inside SLA do not guarantee every individual case "
                             f"met it.", "sla", "sla_headroom", "n"))
        else:
            names = df.loc[df["__v"] > sla, label_col].astype(str).tolist()
            bi.append(_b(ev, f"{ev.d('sla_above')} of {n} {plur} average above the SLA of {ev.d('sla')} "
                             f"({', '.join(names[:3])}), which indicates systematic SLA exposure rather than "
                             f"isolated late cases.", "sla_above", "sla", "n"))
    if ev.has("top_share") and not bi:
        bi.append(_b(ev, f"With {ev.d('top3_share')} of all {col} in three {plur}, a capacity or quality issue in "
                         f"{high_row[label_col]} would move the overall result; workload resilience depends on "
                         f"those {plur}.", "top3_share"))
    if not bi:
        bi.append(_b(ev, f"The result shows a {ev.d('spread')} difference between {low_row[label_col]} and "
                         f"{high_row[label_col]}, but it does not include case volumes, so the workload effect "
                         f"cannot be sized; adding a case count by {sing} to the query would allow sizing.",
                     "spread"))

    # ACTIONS
    acts = []
    owner_dim = sing.title()
    if best is not None and worst is not None and str(best[label_col]) != str(worst[label_col]):
        acts.append(_action(ev, "Short-term",
            f"Run a side-by-side process review of {best[label_col]} ({m.fmt(best['__v'])}) and "
            f"{worst[label_col]} ({m.fmt(worst['__v'])}) — routing, hand-offs, pend reasons and reviewer mix — to "
            f"find the practices behind the gap of {ev.d('spread')}.",
            "Operations Excellence Lead", "Next 2 weeks",
            f"Documented practice differences agreed with both {sing} leads, each with an owner.",
            "spread"))
        urgent = (sla and float(worst["__v"]) > sla) or (str(worst[label_col]) in outliers)
        acts.append(_action(ev, "Immediate" if urgent else "Short-term",
            f"Set an improvement target for {worst[label_col]} to reach the median {col} of {ev.d('median')}.",
            f"{owner_dim} Lead — {worst[label_col]}", "Next reporting cycle",
            f"{worst[label_col]} {col} at or better than {ev.d('median')}.",
            "median"))
    if fl in small and ev.has("vol_focus"):
        acts.append(_action(ev, "Monitor",
            f"Validate {fl}'s result before using it as a benchmark — it rests on only {ev.d('vol_focus')} {vnoun}.",
            "Analytics Owner", "Before the next review",
            "Ranking re-confirmed on a wider date window.", "vol_focus"))
    if ev.has("top_share") and m.polarity == "neutral":
        acts.append(_action(ev, "Short-term",
            f"Review capacity allocation for {high_row[label_col]}, which carries {ev.d('top_share')} of all {col}.",
            "Workforce / Capacity Planner", "Next planning cycle",
            f"Staffing plan for {high_row[label_col]} sized to its share of {col}.", "top_share"))
    if lf is not None:
        acts.append(_action(ev, "Monitor",
            f"Track {col} by {sing} monthly and flag any {sing} that moves outside the IQR fences "
            f"({ev.d('lf')} to {ev.d('uf')}).",
            "Reporting / Analytics Owner", "Monthly",
            "Automatic exception flag included in the monthly operations review.", "lf", "uf"))
    else:
        acts.append(_action(ev, "Monitor",
            f"Track {col} by {sing} monthly to confirm whether the current ranking holds.",
            "Reporting / Analytics Owner", "Monthly",
            "Ranking trend reviewed in the monthly operations review."))

    # STRATEGIC
    strat = []
    if ev.has("wavg") and abs(ev.v("wavg") - mean) / max(abs(mean), 1e-9) > 0.02:
        strat.append(f"Report a volume-weighted {col} alongside the simple average so leadership sees the "
                     f"experience of the typical case, not only the typical {sing}.")
    strat.append(f"Add a monthly trend of {col} by {sing} to confirm whether {fl}'s position is stable rather "
                 f"than a single-period result.")
    if sla:
        strat.append(f"Add the share of cases closed within SLA by {sing}; an average can sit inside SLA while "
                     f"individual cases breach it.")

    # CAVEATS
    cav = _caveats_common(p)
    if small:
        verb = "has a small volume" if len(small) == 1 else "have small volumes"
        cav.append(f"{', '.join(small[:4])} {verb} ({vnoun}); {'its value is' if len(small) == 1 else 'their values are'} "
                   f"sensitive to a handful of cases.")
    if rho is not None and abs(rho) >= 0.5:
        cav.append(f"The association between {vnoun} and {col} does not establish cause.")

    intent = (f"Question interpreted as: rank {plur} by `{col}` and identify the {extreme}"
              + (" (best)" if focus_is_best else " (weakest)" if m.polarity != "neutral" else "") + " value.")
    extra = [f"Benchmark: each {sing} is compared with the median of the {n} {plur} ({ev.d('median')}), which is "
             f"not pulled by a single extreme value."]
    if ev.has("wavg"):
        extra.append(f"Weighting: the volume-weighted average ({ev.d('wavg')}) uses `{vspec.column}` as weights, "
                     f"because a simple average treats a small {sing} the same as a large one.")
    if ev.has("excess"):
        extra.append("Impact sizing: excess = Σ |value − median| × volume for every group on the worse side of "
                     "the median; it is a sizing estimate, not a forecast.")

    tiles = _tiles(ev, ["focus", "median", "gap_med", "spread", "vol_focus_share", "excess", "sla_headroom",
                        "top_share", "better_n"])
    chart = {"type": "ranking", "metric": col, "dimension": dim_name, "median": median, "focus": fl,
             "polarity": m.polarity, "scale": m.scale, "unit": m.unit, "decimals": m.decimals,
             "data": [{"label": str(r[label_col]), "value": float(r["__v"])} for _, r in df.head(30).iterrows()]}
    return dict(ev=ev, key_finding=kf, detailed=da, impact=bi, actions=acts, strategic=strat, caveats=cav,
                reasoning=_reasoning_common(p, ev, intent, extra), tiles=tiles, chart=chart,
                title=f"{col} by {dim_name}", focus=fl, direction=direction)


# ═══════════════════════════════════════════════════════════════════════════════
# WRITER — TREND (time + measure)
# ═══════════════════════════════════════════════════════════════════════════════
def _ordered_time_frame(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    out = df.copy()
    out["__t"] = _to_time(out[tcol])
    if out["__t"].notna().mean() >= 0.7:
        out = out.sort_values("__t", kind="mergesort")
        out["__lbl"] = [_time_label(t, o) for t, o in zip(out["__t"], out[tcol])]
    else:
        out = out.sort_values(tcol, kind="mergesort")
        out["__lbl"] = out[tcol].astype(str)
    return out.reset_index(drop=True)


def _compose_trend(p: Profile) -> dict:
    m, cfg = p.metric, p.cfg
    col = m.column
    s = p.df.copy()
    s["__v"] = pd.to_numeric(s[col], errors="coerce")
    s = s.dropna(subset=["__v"])
    if s[p.time_col].duplicated().any():
        agg = "sum" if m.additive else "mean"
        s = s.groupby(p.time_col, as_index=False)["__v"].agg(agg)
    s = _ordered_time_frame(s, p.time_col)
    ev = Evidence()
    partial = None
    today = date.today()
    tl = s["__t"].iloc[-1] if "__t" in s and len(s) else pd.NaT
    if (len(s) >= 4 and isinstance(tl, pd.Timestamp) and not pd.isna(tl)
            and tl.year == today.year and tl.month == today.month):
        partial = s.iloc[-1]
        ev.add("partial", f"{col} in {partial['__lbl']} (month in progress)", m.fmt(partial["__v"]),
               f"Result row {partial['__lbl']} — excluded from trend statistics", float(partial["__v"]))
        s = s.iloc[:-1].reset_index(drop=True)
    n = len(s)
    if n < 2:
        return _compose_scalar(p)
    first, last = s.iloc[0], s.iloc[-1]
    fv, lv = float(first["__v"]), float(last["__v"])
    ev.add("periods", "Periods in result", f"{n}", f"Distinct values of '{p.time_col}'", n)
    ev.add("first", f"{col} in {first['__lbl']}", m.fmt(fv), f"Result row {first['__lbl']}", fv)
    ev.add("last", f"{col} in {last['__lbl']}", m.fmt(lv), f"Result row {last['__lbl']}", lv)
    ch = lv - fv
    ev.add("change", f"Change {first['__lbl']} → {last['__lbl']}", m.fmt(ch, delta=True, signed=True),
           "Derived: last − first", ch)
    if abs(fv) > 1e-12 and m.kind != "rate":
        ev.add("change_pct", "Relative change first → last", f"{ch / abs(fv) * 100:+.1f}%",
               "Derived: (last − first) ÷ first", ch / abs(fv) * 100)
    prev = s.iloc[-2]
    pc = lv - float(prev["__v"])
    ev.add("prev", f"{col} in {prev['__lbl']}", m.fmt(prev["__v"]), f"Result row {prev['__lbl']}", float(prev["__v"]))
    ev.add("last_change", f"Latest period change vs {prev['__lbl']}", m.fmt(pc, delta=True, signed=True),
           "Derived: last − previous", pc)
    pk = s.loc[s["__v"].idxmax()]
    tr = s.loc[s["__v"].idxmin()]
    ev.add("peak", f"Peak: {pk['__lbl']}", m.fmt(pk["__v"]), f"Result row {pk['__lbl']}", float(pk["__v"]))
    ev.add("trough", f"Trough: {tr['__lbl']}", m.fmt(tr["__v"]), f"Result row {tr['__lbl']}", float(tr["__v"]))
    mean, std = float(s["__v"].mean()), float(s["__v"].std(ddof=0))
    ev.add("mean", f"{n}-period average", m.fmt(mean), "Derived: mean over periods", mean)
    ev.add("std", "Period-to-period standard deviation", m.fmt(std, delta=True), "Derived: population std", std)
    slope = None
    if n >= 3:
        slope = float(np.polyfit(np.arange(n), s["__v"].values.astype(float), 1)[0])
        ev.add("slope", "Fitted linear slope per period", m.fmt(slope, delta=True, signed=True),
               "Derived: least-squares slope over period index", slope)
        ev.add("slope3", "Linear extrapolation: 3 more periods", m.fmt(lv + 3 * slope),
               "Derived: last + 3 × slope (indicative, not a forecast)", lv + 3 * slope)
    diffs = s["__v"].diff().iloc[1:]
    if not diffs.empty:
        i_up, i_dn = diffs.idxmax(), diffs.idxmin()
        if diffs.max() > 0:
            ev.add("rise", f"Largest rise into {s.loc[i_up, '__lbl']}", m.fmt(diffs.max(), delta=True, signed=True),
                   "Derived: max period-over-period change", float(diffs.max()))
        if diffs.min() < 0:
            ev.add("fall", f"Largest fall into {s.loc[i_dn, '__lbl']}", m.fmt(diffs.min(), delta=True, signed=True),
                   "Derived: min period-over-period change", float(diffs.min()))
    above = int((s["__v"] > mean).sum())
    ev.add("above_mean", "Periods above the average", f"{above}", "Derived: count of periods > mean", above)
    if abs(mean) > 1e-12 and m.kind != "rate":
        ev.add("last_vs_mean", "Latest period vs average", f"{(lv - mean) / abs(mean) * 100:+.1f}%",
               "Derived: (last − mean) ÷ mean", (lv - mean) / abs(mean) * 100)

    trend_val = slope if slope is not None else ch
    up = trend_val > 0
    if m.polarity == "neutral":
        dir_word = "an upward" if up else "a downward"
    else:
        improving = (not up) if m.polarity == "lower" else up
        dir_word = "an improving" if improving else "a deteriorating"
    rel = f", {ev.d('change_pct')}" if ev.has("change_pct") else ""
    kf = _b(ev, f"{col} moved from {ev.d('first')} in {first['__lbl']} to {ev.d('last')} in {last['__lbl']} "
                f"({ev.d('change')}{rel}) — {dir_word} trend across {n} periods.",
            "first", "last", "change", "change_pct", "periods")

    da = [_b(ev, f"The peak was {ev.d('peak')} in {pk['__lbl']} and the trough {ev.d('trough')} in "
                 f"{tr['__lbl']}, against a {n}-period average of {ev.d('mean')}.", "peak", "trough", "mean")]
    da.append(_b(ev, f"The most recent move was {ev.d('last_change')} versus {prev['__lbl']} "
                     f"({ev.d('prev')}).", "last_change", "prev"))
    if slope is not None:
        da.append(_b(ev, f"The fitted slope is {ev.d('slope')} per period, which describes the underlying "
                         f"direction better than the first-to-last comparison alone.", "slope"))
    if ev.has("rise") or ev.has("fall"):
        parts = []
        if ev.has("rise"):
            parts.append(f"the largest single rise was {ev.d('rise')}")
        if ev.has("fall"):
            parts.append(f"the largest single fall was {ev.d('fall')}")
        da.append(_b(ev, "Period-to-period movement: " + " and ".join(parts) + ".", "rise", "fall"))
    if partial is not None:
        da.append(_b(ev, f"{partial['__lbl']} is still in progress ({ev.d('partial')} so far) and is excluded from "
                         f"the trend statistics so an incomplete month does not read as a decline.", "partial"))
    cvp = std / abs(mean) * 100 if abs(mean) > 1e-12 else 0
    vol_word = "stable" if cvp < 10 else "volatile" if cvp > 25 else "moderately variable"
    da.append(_b(ev, f"{ev.d('above_mean')} of {n} periods sit above the average and the series is {vol_word} "
                     f"(standard deviation {ev.d('std')}).", "above_mean", "std"))

    bi = []
    if m.kind == "count" and ev.has("last_vs_mean"):
        over = ev.v("last_vs_mean") > 0
        bi.append(_b(ev, f"The latest period is {ev.d('last_vs_mean')} versus the {n}-period average of "
                         f"{ev.d('mean')}; capacity planned on the average would "
                         f"{'under' if over else 'over'}-provision for current demand.", "last_vs_mean", "mean"))
    if slope is not None and m.polarity != "neutral":
        bi.append(_b(ev, f"If the current slope continued, {col} would be about {ev.d('slope3')} three periods "
                         f"after {last['__lbl']} (linear extrapolation for planning, not a forecast).", "slope3"))
    sla = cfg.sla_days if (m.unit == " days" and m.kind in {"duration", "average"} and cfg.sla_days) else None
    if sla:
        ev.add("sla", "Configured SLA", f"{_fmt_num(float(sla), 0)} days", "Application configuration (SLA_DAYS)", sla)
        ev.add("sla_gap", f"Latest {col} vs SLA", m.fmt(lv - sla, delta=True, signed=True), "Derived: last − SLA", lv - sla)
        ev.add("sla_gap_abs", f"Distance of latest {col} from SLA", m.fmt(abs(lv - sla), delta=True), "Derived: |last − SLA|", abs(lv - sla))
        bi.append(_b(ev, f"The latest value is {ev.d('sla_gap_abs')} {'outside' if lv > sla else 'inside'} the SLA of {ev.d('sla')}"
                         + (", so the average case is already outside SLA." if lv > sla else
                            "; the average remains inside SLA, but the trend direction determines how long that lasts."),
                     "sla_gap_abs", "sla"))
    if not bi:
        bi.append(_b(ev, f"The {dir_word.split()[-1]} direction over {n} periods matters more than any single period; "
                         f"decisions should be based on the slope and the {n}-period average rather than the latest "
                         f"value alone.", "periods"))

    acts = []
    deteriorating = m.polarity != "neutral" and dir_word.startswith("a det")
    if deteriorating and ev.has("rise" if m.polarity == "lower" else "fall"):
        k = "rise" if m.polarity == "lower" else "fall"
        acts.append(_action(ev, "Immediate",
            f"Investigate the period with the {'largest rise' if k == 'rise' else 'largest fall'} ({ev.d(k)}) — "
            f"intake mix, staffing and routing changes in that period.",
            "Operations Lead", "Within 1 week",
            f"Root-cause note for the {ev.d(k)} movement reviewed by operations leadership.", k))
    acts.append(_action(ev, "Monitor",
        f"Set a control band of the {n}-period average ({ev.d('mean')}) ± one standard deviation "
        f"({ev.d('std')}) and alert when a period falls outside it.",
        "Reporting / Analytics Owner", "Monthly",
        "Alert configured and reviewed in each monthly operations review.", "mean", "std"))
    if m.polarity != "neutral" and not deteriorating:
        acts.append(_action(ev, "Short-term",
            f"Document what changed between {first['__lbl']} and {last['__lbl']} so the improvement "
            f"({ev.d('change')}) can be sustained.",
            "Process Improvement Lead", "Next 30 days", "Change log of contributing actions agreed with unit leads.",
            "change"))

    cav = _caveats_common(p)
    tl = s["__t"].iloc[-1] if "__t" in s else pd.NaT
    if partial is not None:
        cav.append(f"{partial['__lbl']} is the current month and is incomplete; it is shown in the evidence but "
                   f"excluded from trend statistics.")
    elif isinstance(tl, pd.Timestamp) and not pd.isna(tl) and tl.year == today.year and tl.month == today.month:
        cav.append(f"{last['__lbl']} is the current month and may be incomplete; its value can change as the "
                   f"month closes.")
    if n < 4:
        cav.append(f"Only {n} periods are available; the trend direction is indicative.")
    if slope is not None:
        cav.append("The 3-period extrapolation assumes a straight-line continuation and ignores seasonality.")

    intent = f"Question interpreted as: how `{col}` changes over `{p.time_col}`."
    extra = ["Trend method: first-vs-last change, latest period-over-period change, and a least-squares slope "
             "across all periods (the slope is less sensitive to a single unusual period)."]
    tiles = _tiles(ev, ["last", "change", "change_pct", "last_change", "slope", "peak", "mean"])
    chart = {"type": "trend", "metric": col, "dimension": p.time_col, "mean": mean, "polarity": m.polarity,
             "scale": m.scale, "unit": m.unit, "decimals": m.decimals,
             "peak": str(pk["__lbl"]), "trough": str(tr["__lbl"]),
             "data": [{"label": str(r["__lbl"]), "value": float(r["__v"])} for _, r in s.iterrows()]}
    return dict(ev=ev, key_finding=kf, detailed=da, impact=bi, actions=acts, strategic=[
        f"Pair {col} with its volume driver in the same view so changes in mix are not mistaken for performance."],
        caveats=cav, reasoning=_reasoning_common(p, ev, intent, extra), tiles=tiles, chart=chart,
        title=f"{col} over {p.time_col}", focus=str(last["__lbl"]), direction="trend")


def _compose_trend_by_group(p: Profile) -> dict:
    m = p.metric
    col, g = m.column, p.dims[0]
    sing, plur = _dim_nouns(g)
    d = p.df.copy()
    d["__v"] = pd.to_numeric(d[col], errors="coerce")
    d = _ordered_time_frame(d.dropna(subset=["__v"]), p.time_col)
    periods = list(dict.fromkeys(d["__lbl"].tolist()))
    if len(periods) < 2:
        return _compose_ranking(p)
    p0, p1 = periods[0], periods[-1]
    rows = []
    for name, grp in d.groupby(g):
        grp = grp.sort_values("__t" if "__t" in grp else "__lbl")
        if len(grp) >= 2:
            rows.append((str(name), str(grp["__lbl"].iloc[0]), float(grp["__v"].iloc[0]),
                         str(grp["__lbl"].iloc[-1]), float(grp["__v"].iloc[-1])))
    if not rows:
        return _compose_ranking(p)
    mv = pd.DataFrame(rows, columns=["name", "f_lbl", "f", "l_lbl", "l"])
    mv["chg"] = mv["l"] - mv["f"]
    ev = Evidence()
    ev.add("groups", f"{plur.title()} with ≥2 periods", f"{len(mv)}", "Derived from result", len(mv))
    ev.add("periods", "Periods in result", f"{len(periods)}", f"Distinct values of '{p.time_col}'", len(periods))
    inc = mv.loc[mv["chg"].idxmax()]
    dec = mv.loc[mv["chg"].idxmin()]
    ev.add("inc", f"{inc['name']}: {inc['f_lbl']} → {inc['l_lbl']}", f"{m.fmt(inc['f'])} → {m.fmt(inc['l'])}",
           f"Result rows for '{inc['name']}'", float(inc["chg"]))
    ev.add("inc_chg", f"{inc['name']} change", m.fmt(inc["chg"], delta=True, signed=True), "Derived: last − first", float(inc["chg"]))
    ev.add("dec", f"{dec['name']}: {dec['f_lbl']} → {dec['l_lbl']}", f"{m.fmt(dec['f'])} → {m.fmt(dec['l'])}",
           f"Result rows for '{dec['name']}'", float(dec["chg"]))
    ev.add("dec_chg", f"{dec['name']} change", m.fmt(dec["chg"], delta=True, signed=True), "Derived: last − first", float(dec["chg"]))
    n_up, n_dn = int((mv["chg"] > 0).sum()), int((mv["chg"] < 0).sum())
    ev.add("n_up", f"{plur.title()} increasing", f"{n_up}", "Derived", n_up)
    ev.add("n_dn", f"{plur.title()} decreasing", f"{n_dn}", "Derived", n_dn)

    worse_is_up = m.polarity == "lower"
    lead = inc if (m.polarity != "higher") else dec
    kf = _b(ev, f"Between {p0} and {p1}, {lead['name']} moved the most in {col} ({ev.d('inc' if lead is inc else 'dec')}, "
                f"{ev.d('inc_chg' if lead is inc else 'dec_chg')})"
                + (" — the largest deterioration across " if m.polarity != "neutral" else " — the largest change across ")
                + f"{len(mv)} {plur}.", "inc", "inc_chg", "dec", "dec_chg", "groups")
    da = [_b(ev, f"{ev.d('n_up')} {plur} increased and {ev.d('n_dn')} decreased over the period.", "n_up", "n_dn"),
          _b(ev, f"The largest increase was {inc['name']} ({ev.d('inc_chg')}); the largest decrease was "
                 f"{dec['name']} ({ev.d('dec_chg')}).", "inc_chg", "dec_chg")]
    if m.additive:
        tot = d.groupby("__lbl", sort=False)["__v"].sum()
        ev.add("tot_first", f"Total {col} in {p0}", m.fmt(tot.iloc[0]), "Derived: sum across groups", float(tot.iloc[0]))
        ev.add("tot_last", f"Total {col} in {p1}", m.fmt(tot.iloc[-1]), "Derived: sum across groups", float(tot.iloc[-1]))
        da.append(_b(ev, f"In total, {col} went from {ev.d('tot_first')} to {ev.d('tot_last')}.", "tot_first", "tot_last"))
    worst_mover = inc if worse_is_up else dec
    bi = [_b(ev, f"{worst_mover['name']} is where the change is concentrated, so it is the first place to look "
                 f"before drawing conclusions about the whole portfolio.", "inc_chg" if worst_mover is inc else "dec_chg")]
    acts = [_action(ev, "Short-term" if m.polarity != "neutral" else "Monitor",
                    f"Review {worst_mover['name']}'s change in {col} between {p0} and {p1} with its {sing} lead.",
                    f"{sing.title()} Lead — {worst_mover['name']}", "Next 2 weeks",
                    f"Explanation of the {m.fmt(worst_mover['chg'], delta=True, signed=True)} change recorded.",
                    "inc_chg" if worst_mover is inc else "dec_chg")]
    intent = f"Question interpreted as: how `{col}` changes over `{p.time_col}` for each {sing}."
    tiles = _tiles(ev, ["inc_chg", "dec_chg", "n_up", "n_dn", "tot_last"])
    return dict(ev=ev, key_finding=kf, detailed=da, impact=bi, actions=acts, strategic=[],
                caveats=_caveats_common(p), reasoning=_reasoning_common(p, ev, intent, [
                    f"Movers: change = value in the last available period − value in the first, per {sing}."]),
                tiles=tiles, chart=None, title=f"{col} by {g} over {p.time_col}", focus=str(lead["name"]),
                direction="trend")


# ═══════════════════════════════════════════════════════════════════════════════
# WRITER — SCALAR (single row)
# ═══════════════════════════════════════════════════════════════════════════════
def _compose_scalar(p: Profile) -> dict:
    m, cfg = p.metric, p.cfg
    row = p.df.iloc[0]
    ev = Evidence()
    col = m.column
    val = _safe_float(row[col])
    ev.add("value", col, m.fmt(val), f"Result column '{col}'", val)
    specs = [m] + ([p.volume] if p.volume else []) + list(p.other_metrics)
    for s in specs[1:6]:
        ev.add(f"o_{s.column}", s.column, s.fmt(row[s.column]), f"Result column '{s.column}'", _safe_float(row[s.column]))
    vol_txt = ""
    if p.volume is not None:
        vol_txt = f" across {ev.d('o_' + p.volume.column)} {_vol_noun(p.volume.column)}"
    kf = _b(ev, f"{col} is {ev.d('value')}{vol_txt}.", "value", f"o_{p.volume.column}" if p.volume else "value")

    da = []
    for s in specs[1:6]:
        if p.volume is not None and s.column == p.volume.column:
            continue
        da.append(_b(ev, f"{s.column} for the same population is {ev.d('o_' + s.column)}.", f"o_{s.column}"))
    counts = [s for s in specs if s.kind == "count"]
    total = next((s for s in counts if set(_words(s.column)) & {"total", "all"}), None)
    for s in counts:
        if total is not None and s.column != total.column and _safe_float(row[total.column]):
            share = _safe_float(row[s.column]) / _safe_float(row[total.column]) * 100
            ev.add(f"share_{s.column}", f"{s.column} as share of {total.column}", _pct(share), "Derived: ratio", share)
            da.append(_b(ev, f"{s.column} represents {ev.d('share_' + s.column)} of {total.column}.", f"share_{s.column}"))
    da.append(_b(ev, "This is a single aggregate for the filtered population; it shows the overall level but not "
                     "where the result comes from or how it varies."))

    bi = []
    sla = cfg.sla_days if (m.unit == " days" and m.kind in {"duration", "average"} and cfg.sla_days) else None
    if sla and val is not None:
        ev.add("sla", "Configured SLA", f"{_fmt_num(float(sla), 0)} days", "Application configuration (SLA_DAYS)", sla)
        ev.add("sla_gap", f"{col} vs SLA", m.fmt(val - sla, delta=True, signed=True), "Derived: value − SLA", val - sla)
        ev.add("sla_gap_abs", f"Distance of {col} from SLA", m.fmt(abs(val - sla), delta=True), "Derived: |value − SLA|", abs(val - sla))
        bi.append(_b(ev, f"The average sits {ev.d('sla_gap_abs')} {'outside' if val > sla else 'inside'} the SLA of {ev.d('sla')}"
                         + ("; at portfolio level the average case is outside SLA." if val > sla
                            else "; an average inside SLA can still hide individual cases that breach it."),
                     "sla_gap_abs", "sla"))
    if p.volume is not None and m.kind in {"duration", "average"} and m.unit in {" days", " hours"} and val is not None:
        vol = _safe_float(row[p.volume.column]) or 0
        unit_word = "case-days" if m.unit == " days" else "case-hours"
        ev.add("exposure", f"Total {unit_word}", f"{val * vol:,.0f} {unit_word}", f"Derived: {col} × {p.volume.column}", val * vol)
        bi.append(_b(ev, f"Across the population this equals about {ev.d('exposure')} of elapsed time — the "
                         f"workload that any per-case improvement would act on.", "exposure"))
    if not bi:
        bi.append(_b(ev, f"A single portfolio figure cannot show which teams or periods drive {col}; the operational "
                         f"lever only becomes visible once the result is broken down."))
    acts = [_action(ev, "Short-term", f"Break {col} down by organisational unit and by month to locate where the "
                                      f"{ev.d('value')} comes from.",
                    "Reporting / Analytics Owner", "Next 2 weeks",
                    "Breakdown reviewed and the top contributing unit identified.", "value"),
            _action(ev, "Monitor", f"Add {col} ({ev.d('value')}) to the recurring KPI pack as the baseline for "
                                   f"future comparison.", "Reporting / Analytics Owner", "Monthly",
                    "Baseline recorded and compared each reporting cycle.", "value")]
    intent = f"Question interpreted as: a single aggregate value of `{col}`."
    tiles = _tiles(ev, ["value"] + [f"o_{s.column}" for s in specs[1:4]] + ["sla_gap", "exposure"])
    return dict(ev=ev, key_finding=kf, detailed=da, impact=bi, actions=acts,
                strategic=[f"Track {col} as a monthly series so changes can be judged against a baseline."],
                caveats=_caveats_common(p), reasoning=_reasoning_common(p, ev, intent, []), tiles=tiles,
                chart=None, title=col, focus=col, direction="scalar")


# ═══════════════════════════════════════════════════════════════════════════════
# WRITER — RECORDS (no numeric measure)
# ═══════════════════════════════════════════════════════════════════════════════
def _compose_records(p: Profile) -> dict:
    df = p.df
    ev = Evidence()
    n = len(df)
    ev.add("n", "Records returned", f"{n:,}", "Row count of the query result", n)
    da = []
    for c in [c for c in df.columns][:4]:
        vc = df[c].astype(str).value_counts()
        if vc.empty:
            continue
        k = f"top_{c}"
        ev.add(k, f"Most frequent {c}: {vc.index[0]}", f"{int(vc.iloc[0]):,}", f"Value counts of '{c}'", int(vc.iloc[0]))
        ev.add(f"dist_{c}", f"Distinct {c}", f"{vc.size:,}", f"Distinct values of '{c}'", int(vc.size))
        da.append(_b(ev, f"{c}: {ev.d('dist_' + c)} distinct values; the most frequent is {vc.index[0]} "
                         f"({ev.d(k)} of {ev.d('n')} records).", k, f"dist_{c}", "n"))
    kf = _b(ev, f"The query returned {ev.d('n')} matching records.", "n")
    bi = [_b(ev, f"These {ev.d('n')} records are a work list rather than a summary; prioritise them by age or "
                 f"status before assigning owners.", "n")]
    acts = [_action(ev, "Short-term", "Assign owners to the returned records and agree a review date.",
                    "Operations Lead", "This week", "Every record has an owner and a review date.", "n")]
    return dict(ev=ev, key_finding=kf, detailed=da, impact=bi, actions=acts, strategic=[],
                caveats=_caveats_common(p),
                reasoning=_reasoning_common(p, ev, "Question interpreted as: a record-level listing.", []),
                tiles=_tiles(ev, ["n"]), chart=None, title="Matching records", focus="", direction="records")


# ═══════════════════════════════════════════════════════════════════════════════
# TILES
# ═══════════════════════════════════════════════════════════════════════════════
def _tiles(ev: Evidence, keys: list, limit: int = 6) -> list:
    out = []
    for k in keys:
        if ev.has(k) and len(out) < limit:
            it = ev._by_key[k]
            out.append({"label": it["label"], "value": it["display"], "raw_value": it["value"], "unit": "",
                        "source_columns": [], "evidence": f"{it['id']} · {it['source']}", "evidence_id": it["id"]})
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL LLM NARRATIVE (fluency only — strictly verified)
# ═══════════════════════════════════════════════════════════════════════════════
class NarrativeAction(BaseModel):
    priority: Priority
    action: str = Field(min_length=1)
    owner: str = ""
    timeframe: str = ""
    success_measure: str = ""
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, v: Any) -> Priority:
        return Priority.coerce(v)


class InsightNarrative(BaseModel):
    """LLM response contract for the optional narrative rewrite (purpose ``insight_narrative``)."""

    key_finding: str = Field(min_length=1)
    detailed_analysis: list[str] = Field(default_factory=list)
    business_impact: list[str] = Field(default_factory=list)
    recommended_actions: list[NarrativeAction] = Field(default_factory=list)

    @field_validator("detailed_analysis", "business_impact", mode="before")
    @classmethod
    def _as_list(cls, v: Any) -> Any:
        if v is None:
            return []
        return [v] if isinstance(v, str) else v


DEFAULT_NARRATIVE_SYSTEM = """You write the narrative sections of an analytics insight card for operations and compliance leaders.

You receive the user's question, the analysis type, the metric (with its direction: lower_is_better, higher_is_better or neutral), an EVIDENCE list and a reference draft. Each evidence item has an id (F1, F2, ...), a label and a display value computed deterministically from the exact query result. Evidence is the only source of truth.

Write four sections. Each has ONE job; never repeat a sentence, phrase or framing across sections:
1. key_finding: the direct answer in 1-2 sentences, plus ONE piece of context that makes it meaningful (gap to median, share of volume, or change versus the prior period).
2. detailed_analysis: 3-5 sentences on the SHAPE of the result: range and spread, runner-up margin, how many are better/worse than the median, weighting or volume effects, dispersion and outliers, or trend dynamics. Do not restate the key finding.
3. business_impact: 2-3 sentences on the operational consequence: workload, capacity, SLA exposure, member experience, using sizing evidence (for example excess case-days) when present. If no sizing evidence exists, explain the practical implication without inventing a number.
4. recommended_actions: 2-4 actions that each follow from specific evidence: what to do, for which entity, an owner role (never a person's name), a timeframe, and a measurable success_measure that uses evidence values as the baseline or target.

RULES
- End every sentence with the evidence ids it uses, e.g. "... 4.17 days faster than the median [F5, F4]".
- Copy every number exactly from an evidence display value. Do not compute, re-round, or invent thresholds, targets, benchmarks or percentages.
- Respect metric direction. For lower_is_better, a lower value is better. For neutral metrics say higher/lower, not better/worse.
- No causal claims: use "may indicate", "is consistent with", "warrants review".
- Plain text only. No markdown, no emoji, no HTML.

Return one JSON object only:
{"key_finding": "", "detailed_analysis": [""], "business_impact": [""], "recommended_actions": [{"priority": "Immediate|Short-term|Strategic|Monitor", "action": "", "owner": "", "timeframe": "", "success_measure": "", "evidence_ids": ["F1"]}]}"""

_CITE_RE = re.compile(r"\[\s*(F\d+(?:\s*,\s*F\d+)*)\s*\]")
_PRIORITIES = {"Immediate", "Short-term", "Strategic", "Monitor"}


_CAUSAL_RE = re.compile(r"\b(because|caused by|driven by|due to|results? from|as a result of)\b", re.I)


def _clean_llm_sentence(t: Any, valid_ids: set) -> tuple:
    text = str(t or "").strip()
    refs = []
    for grp in _CITE_RE.findall(text):
        refs.extend(x.strip() for x in grp.split(","))
    text = _CITE_RE.sub("", text)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s+", " ", text)).strip()
    return text, [r for r in dict.fromkeys(refs) if r in valid_ids]


def _llm_narrative(llm: StructuredLLM, system: str, p: Profile, draft: dict, allowed: list,
                   cfg: ComposerConfig) -> tuple:
    ev: Evidence = draft["ev"]
    valid = {i["id"] for i in ev.items}
    payload = {
        "question": p.question,
        "analysis_type": p.analysis_type,
        "metric": {"column": p.metric.column, "direction": {"lower": "lower_is_better", "higher": "higher_is_better"}
                   .get(p.metric.polarity, "neutral"), "kind": p.metric.kind} if p.metric else None,
        "dimension": p.dims[0] if p.dims else p.time_col,
        "evidence": [{"id": i["id"], "label": i["label"], "value": i["display"]} for i in ev.items],
        "reference_draft": {
            "key_finding": draft["key_finding"]["text"],
            "detailed_analysis": [b["text"] for b in draft["detailed"]],
            "business_impact": [b["text"] for b in draft["impact"]],
        },
    }
    reply = llm.structured("insight_narrative", system,
                           json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True),
                           InsightNarrative, temperature=cfg.llm_temperature)
    raw = reply.model_dump(mode="json")
    dropped: list = []
    used: dict = {}

    def bullets(items: Any, limit: int) -> list:
        out = []
        for it in (items if isinstance(items, list) else [items])[:limit + 2]:
            text, refs = _clean_llm_sentence(it if not isinstance(it, dict) else it.get("text"), valid)
            if len(text) < 25:
                continue
            if _CAUSAL_RE.search(text):
                dropped.append(f"{text}  [unsupported causal claim]")
                continue
            bad = verify_with_reason(text, allowed)
            if bad:
                dropped.append(f"{text}  [untraceable number: {bad}]")
                continue
            out.append({"text": text, "refs": refs})
        return out[:limit]

    kf_list = bullets(raw.get("key_finding"), 1)
    kf = kf_list[0] if kf_list else None
    used["key_finding"] = bool(kf)
    kf = kf or draft["key_finding"]

    da = [b for b in bullets(raw.get("detailed_analysis"), cfg.max_bullets)
          if similarity(b["text"], kf["text"]) < cfg.similarity_threshold]
    used["detailed_analysis"] = len(da) >= 2
    da = da if used["detailed_analysis"] else draft["detailed"]

    bi = [b for b in bullets(raw.get("business_impact"), 4)
          if similarity(b["text"], kf["text"]) < cfg.similarity_threshold
          and all(similarity(b["text"], x["text"]) < cfg.similarity_threshold for x in da)]
    used["business_impact"] = len(bi) >= 1
    bi = bi if used["business_impact"] else draft["impact"]

    acts = []
    for a in (raw.get("recommended_actions") or [])[: cfg.max_actions + 2]:
        if not isinstance(a, dict):
            continue
        action, r1 = _clean_llm_sentence(a.get("action"), valid)
        success, r2 = _clean_llm_sentence(a.get("success_measure"), valid)
        if len(action) < 20:
            continue
        bad = verify_with_reason(action, allowed) or verify_with_reason(success, allowed)
        if bad:
            dropped.append(f"{action}  [untraceable number: {bad}]")
            continue
        ids = [i for i in (a.get("evidence_ids") or []) if i in valid] or (r1 + r2)
        prio = str(a.get("priority", "Short-term")).strip().title().replace("Short-Term", "Short-term")
        acts.append({
            "priority": prio if prio in _PRIORITIES else "Short-term",
            "action": action,
            "owner": re.sub(r"\s+", " ", str(a.get("owner") or "Operations Lead")).strip()[:60],
            "timeframe": re.sub(r"\s+", " ", str(a.get("timeframe") or "Next reporting cycle")).strip()[:40],
            "success_measure": success or "Agreed completion criteria met and metric re-measured.",
            "evidence_ids": list(dict.fromkeys(ids)), "evidence": "",
        })
    used["recommended_actions"] = len(acts) >= 2
    acts = acts[: cfg.max_actions] if used["recommended_actions"] else draft["actions"]
    return {"key_finding": kf, "detailed": da, "impact": bi, "actions": acts}, dropped, used


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════
def _dedupe_sections(kf: dict, da: list, bi: list, thr: float) -> tuple:
    da2 = [b for b in da if similarity(b["text"], kf["text"]) < thr]
    bi2 = [b for b in bi if similarity(b["text"], kf["text"]) < thr
           and all(similarity(b["text"], x["text"]) < thr for x in da2)]
    return da2 or da, bi2 or bi


def compose_insight(
    result_df: pd.DataFrame | None,
    *,
    question: str,
    sql: str = "",
    config: ComposerConfig | None = None,
    llm: StructuredLLM | None = None,
    chart_notes: list | None = None,
    narrative_system: str | None = None,
) -> dict:
    """Return a complete, evidence-linked insight contract for one query result."""
    cfg = config or ComposerConfig()
    if result_df is None or result_df.empty:
        msg = "No records matched the criteria for this question."
        return {"composer_version": COMPOSER_VERSION, "analysis_type": "empty", "title": "No data",
                "key_finding": {"text": msg, "refs": []}, "insight": msg, "executive_summary": msg,
                "detailed_analysis": [], "business_impact": [], "reasoning_steps": [], "reasoning": "",
                "evidence": [], "supporting_metrics": [], "recommended_actions": [], "actions": [],
                "strategic_improvements": [], "data_caveats": [], "risk_level": "Low", "impact": "Informational",
                "grounded": False, "_dropped_claims": [], "_allowed_numbers": []}

    df = result_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    p = profile_result(df, question, sql, cfg)
    writer = {"ranking": _compose_ranking, "trend": _compose_trend, "trend_by_group": _compose_trend_by_group,
              "scalar": _compose_scalar, "records": _compose_records}.get(p.analysis_type, _compose_records)
    try:
        draft = writer(p)
    except Exception as exc:  # never let insight composition break the answer
        log.exception("deterministic writer failed: %s", exc)
        p.analysis_type = "records"
        draft = _compose_records(p)

    ev: Evidence = draft["ev"]
    label_values = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            label_values.extend(df[c].astype(str).head(60).tolist())
    allowed = build_allowed_numbers(*[i["display"] for i in ev.items], *[i["label"] for i in ev.items],
                                    *label_values, sql, question)

    sections = {"key_finding": draft["key_finding"], "detailed": draft["detailed"],
                "impact": draft["impact"], "actions": draft["actions"]}
    dropped, used = [], {}
    if llm is not None and cfg.enable_llm_narrative and p.metric is not None:
        try:
            sections, dropped, used = _llm_narrative(llm, narrative_system or DEFAULT_NARRATIVE_SYSTEM,
                                                     p, draft, allowed, cfg)
        except (LLMResponseError, LLMUnavailableError) as exc:
            log.warning("LLM narrative unavailable, deterministic narrative used: %s", type(exc).__name__)

    da, bi = _dedupe_sections(sections["key_finding"], sections["detailed"], sections["impact"],
                              cfg.similarity_threshold)
    caveats = list(dict.fromkeys(draft["caveats"] + [n for n in (chart_notes or []) if n]))[:6]
    reasoning = draft["reasoning"]
    if dropped:
        reasoning = reasoning + [f"{len(dropped)} AI-written sentence(s) were removed because they contained "
                                 f"numbers not present in the evidence."]
    source = "deterministic" if not any(used.values()) else (
        "ai_verified" if all(used.values()) else "mixed")
    kf_text = sections["key_finding"]["text"]
    return {
        "composer_version": COMPOSER_VERSION,
        "analysis_type": p.analysis_type,
        "title": draft["title"],
        "focus": draft.get("focus", ""),
        "key_finding": sections["key_finding"],
        "insight": kf_text,
        "executive_summary": kf_text,
        "detailed_analysis": da[: cfg.max_bullets],
        "business_impact": bi[:4],
        "reasoning_steps": reasoning,
        "reasoning": " ".join(reasoning),
        "evidence": [{k: v for k, v in i.items() if k != "key"} for i in ev.items],
        "supporting_metrics": draft["tiles"],
        "recommended_actions": sections["actions"][: cfg.max_actions],
        "actions": sections["actions"][: cfg.max_actions],
        "strategic_improvements": draft["strategic"][:3],
        "data_caveats": caveats,
        "evidence_chart": draft.get("chart"),
        "metric": {"column": p.metric.column, "kind": p.metric.kind, "polarity": p.metric.polarity,
                   "unit": p.metric.unit} if p.metric else None,
        "narrative_source": source,
        "narrative_sections_from_ai": used,
        "risk_level": "Low",
        "impact": "Operational",
        "grounded": True,
        "_dropped_claims": dropped,
        "_allowed_numbers": allowed,
    }


__all__ = ["ComposerConfig", "compose_insight", "verify_text", "build_allowed_numbers", "similarity",
           "DEFAULT_NARRATIVE_SYSTEM", "COMPOSER_VERSION", "InsightNarrative"]
