"""Read-only SQL governance and execution.

Every statement produced by an LLM passes :func:`validate_read_only` before it can
reach Databricks. :class:`SqlExecutor` runs validated SQL with a row cap and a small
thread-safe TTL result cache (deterministic; no AI involved).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp

log = logging.getLogger("datatalk.sql")

_MUTATION_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|MERGE|TRUNCATE|COPY|GRANT|REVOKE|OPTIMIZE|VACUUM|CALL|EXECUTE|USE|SET)\b",
    re.I,
)
_AGGREGATES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Group)


@dataclass
class SqlCheck:
    """Outcome of :func:`validate_read_only`. ``sql`` may have a LIMIT added."""

    valid: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    referenced_columns: list[str] = field(default_factory=list)
    is_aggregate: bool = False


def validate_read_only(query: str, *, table: str, raw_columns: Iterable[str],
                       max_row_limit: int) -> SqlCheck:
    """Allow exactly one read-only SELECT against the approved table and raw columns.

    ``table`` is the bare table name (no catalog/schema). Non-aggregate queries get
    ``LIMIT max_row_limit`` added or reduced.
    """
    q = (query or "").strip().rstrip(";").strip()
    check = SqlCheck(valid=False, sql=q)
    if not q:
        check.errors.append("No SQL was generated.")
        return check
    if re.search(r"(--|/\*|\*/)", q):
        check.errors.append("SQL comments are not allowed.")
    if _MUTATION_RE.search(q):
        check.errors.append("Only read-only SELECT queries are allowed.")
    if ";" in q:
        check.errors.append("Only one SQL statement is allowed.")

    allowed = set(raw_columns)
    try:
        parsed = sqlglot.parse(q, read="databricks")
    except Exception as exc:  # sqlglot raises several error types
        check.errors.append(f"SQL parsing failed: {exc}")
        return check
    if len(parsed) != 1 or parsed[0] is None:
        check.errors.append("Exactly one SQL statement is required.")
        return check

    tree = parsed[0]
    if not isinstance(tree, (exp.Select, exp.Union)) and tree.find(exp.Select) is None:
        check.errors.append("Only SELECT queries are allowed.")
    if any(tree.find(node) is not None for node in (exp.Insert, exp.Update, exp.Delete, exp.Drop,
                                                     exp.Create, exp.Alter)):
        check.errors.append("Mutating SQL operations are prohibited.")
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    if tables != {table.lower()}:
        check.errors.append(f"Only table {table} may be queried.")

    cols = sorted({c.name for c in tree.find_all(exp.Column) if c.name})
    aliases = {a.alias for a in tree.find_all(exp.Alias) if a.alias}
    invalid = [c for c in cols if c not in allowed and c not in aliases and c != "*"]
    if invalid:
        check.errors.append("Unknown or derived SQL column(s): " + ", ".join(invalid))
    check.referenced_columns = cols
    check.is_aggregate = any(tree.find(x) is not None for x in _AGGREGATES)

    if not check.is_aggregate:
        limit = tree.args.get("limit")
        if limit is None:
            q = f"{q} LIMIT {max_row_limit}"
            check.warnings.append(f"LIMIT {max_row_limit} added to row-level query.")
        else:
            try:
                if int(limit.expression.name) > max_row_limit:
                    q = re.sub(r"\bLIMIT\s+\d+\s*$", f"LIMIT {max_row_limit}", q, flags=re.I)
                    check.warnings.append(f"LIMIT reduced to {max_row_limit}.")
            except (TypeError, ValueError, AttributeError):
                check.errors.append("LIMIT must be a fixed integer.")

    check.sql = q
    check.valid = not check.errors
    return check


def projects_sensitive_columns(sql: str, sensitive: Iterable[str]) -> bool:
    """True when a SELECT projection exposes a sensitive column outside an aggregate.

    ``COUNT(`Case Identifier`)`` is fine; ``SELECT `Current User Name` ...`` is not.
    Unparseable SQL is treated as sensitive.
    """
    names = set(sensitive)
    try:
        tree = sqlglot.parse_one(sql, read="databricks")
    except Exception:
        return True
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            if isinstance(projection, exp.Star):
                return True
            for column in projection.find_all(exp.Column):
                if column.name in names and column.find_ancestor(exp.AggFunc) is None:
                    return True
    return False


class SqlExecutor:
    """Runs SQL through a connection factory with a row cap and a TTL result cache."""

    def __init__(self, connect: Callable[[], Any], *, max_result_rows: int,
                 cache_ttl_seconds: int = 300, cache_max_entries: int = 64) -> None:
        self._connect = connect
        self.max_result_rows = max_result_rows
        self.cache_ttl = cache_ttl_seconds
        self.cache_max = cache_max_entries
        self._cache: OrderedDict[str, tuple[float, pd.DataFrame]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(sql: str) -> str:
        return hashlib.sha256(re.sub(r"\s+", " ", sql.strip().lower()).encode()).hexdigest()

    def _fetch(self, sql: str, *, cap: int | None) -> pd.DataFrame:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql)
            if cap is None:
                rows = cur.fetchall()
            else:
                rows = cur.fetchmany(cap + 1)
                if len(rows) > cap:
                    raise ValueError(f"Result exceeded the safe limit of {cap:,} rows. "
                                     "Please use an aggregated query or add more filters.")
            cols = [d[0] for d in cur.description]
        return pd.DataFrame([tuple(r) for r in rows], columns=cols)

    def run_trusted(self, sql: str) -> pd.DataFrame:
        """Application-authored SQL (full table load): no validation, no cap, no cache."""
        return self._fetch(sql, cap=None)

    def run_validated(self, sql: str) -> tuple[pd.DataFrame, bool]:
        """Run already-validated SQL. Returns ``(frame, from_cache)``."""
        key = self._key(sql)
        now = time.time()
        if self.cache_ttl and self.cache_max:
            with self._lock:
                hit = self._cache.get(key)
                if hit and now - hit[0] < self.cache_ttl:
                    self._cache.move_to_end(key)
                    return hit[1].copy(), True
        t0 = time.perf_counter()
        frame = self._fetch(sql, cap=self.max_result_rows)
        log.info("sql_exec rows=%d duration_ms=%.0f", len(frame), (time.perf_counter() - t0) * 1000)
        if self.cache_ttl and self.cache_max:
            with self._lock:
                self._cache[key] = (now, frame.copy())
                while len(self._cache) > self.cache_max:
                    self._cache.popitem(last=False)
        return frame, False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
