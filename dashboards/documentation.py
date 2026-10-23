"""Documentation page: business-friendly and technically accurate guide to DataTalk.AI.

Diagrams are read from ``docs/`` (and ``assets/`` as a fallback) at runtime, so
updating an SVG there updates this page. Figures and limits shown here come from the
running configuration where available. Deterministic — no data or LLM access.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

import streamlit as st

from agents.supervisor_agent import MAX_QUESTION_CHARS, MIN_QUESTION_CHARS
from components.ui import page_title, section
from core.settings import AGING_THRESHOLD, ESCALATION_LEVEL, SLA_DAYS, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_DIRS = (PROJECT_ROOT / "docs", PROJECT_ROOT / "assets")

PRIMARY_FLOW = ("User Question", "Request Routing", "Context Grounding", "Intent Understanding", "Query Planning",
                "SQL Generation", "Safety Validation", "Live Data Execution", "Insight & Chart Synthesis",
                "Response QA", "Data-Grounded Answer")

EXECUTIVE_STAGES = (
    ("1", "ASK", "#3b82f6", ("User Question", "Business Intent", "Question Context")),
    ("2", "UNDERSTAND", "#8b5cf6", ("AI Reasoning", "Business Context", "SQL Generation")),
    ("3", "VERIFY", "#f59e0b", ("Safety Guardrails", "Read-Only Policy", "Query Validation")),
    ("4", "EXECUTE", "#06b6d4", ("Live Data Query", "Live Data Retrieval", "Repair & Retry")),
    ("5", "DELIVER", "#10b981", ("Answer & Visuals", "Plain-English Summary", "Chart, Table & SQL")),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration-backed values (fall back to the documented defaults)
# ═══════════════════════════════════════════════════════════════════════════════

def _limits() -> dict:
    s = get_settings()
    return {
        "max_sub": int(getattr(s, "max_sub_questions", 5) or 5),
        "max_rows": int(getattr(s, "max_result_rows", 5000) or 5000),
        "row_limit": int(getattr(s, "max_row_query_limit", 200) or 200),
        "attempts": int(getattr(s, "llm_max_attempts", 3) or 3),
        "values": int(getattr(s, "prompt_max_values_per_field", 25) or 25),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Diagrams
# ═══════════════════════════════════════════════════════════════════════════════

def _diagram_signature() -> tuple:
    files = []
    for folder in DIAGRAM_DIRS:
        if folder.is_dir():
            files.extend((str(p), p.stat().st_mtime_ns) for p in sorted(folder.glob("*.svg")))
    return tuple(files)


@st.cache_data(show_spinner=False)
def _load_diagrams(signature: tuple) -> list[dict]:
    """Read every SVG once per file change. ``signature`` keys the cache."""
    seen, out = set(), []
    for path_str, _ in signature:
        path = Path(path_str)
        if path.name.lower() in seen:
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
        except OSError:
            continue
        if "<svg" not in text:
            continue
        seen.add(path.name.lower())
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
        title = html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else \
            path.stem.replace("_", " ").replace("-", " ").title()
        label = f"{path.stem} {title}".lower()
        kind = "overview" if "overview" in label else "flow" if "flow" in label else "other"
        out.append({"name": path.name, "title": title, "kind": kind,
                    "b64": base64.b64encode(raw).decode("ascii"), "bytes": raw})
    return out


def _diagrams() -> list[dict]:
    return _load_diagrams(_diagram_signature())


def _render_diagram(kind: str, caption: str) -> None:
    found = [d for d in _diagrams() if d["kind"] == kind]
    if not found:
        st.info(f"The {caption.lower()} is not available. Add the SVG to the `docs` folder "
                f"(a file name containing “{kind}”) and reload this page.")
        return
    for d in found:
        _figure(d, caption)


def _figure(d: dict, caption: str) -> None:
    st.markdown(
        f"<figure class='doc-figure'><img src='data:image/svg+xml;base64,{d['b64']}' "
        f"alt='{html.escape(d['title'])}'/><figcaption>{html.escape(caption)} · "
        f"{html.escape(d['title'])}</figcaption></figure>",
        unsafe_allow_html=True,
    )
    st.download_button("⬇️ Download diagram (SVG)", data=d["bytes"], file_name=d["name"], mime="image/svg+xml",
                       key=f"doc_dl_{d['name']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Small presentational helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _css() -> None:
    st.markdown("""<style>
.doc-hero{background:linear-gradient(135deg,rgba(37,99,235,.10),rgba(16,185,129,.06));
  border:1px solid var(--dt-border,rgba(128,128,128,.25));border-radius:14px;padding:18px 22px;margin:4px 0 14px}
.doc-hero h3{margin:0 0 6px;font-size:1.25rem}
.doc-hero p{margin:0;opacity:.85;font-size:.92rem;line-height:1.55}
.doc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:8px 0 14px}
.doc-card{background:var(--dt-surface,rgba(128,128,128,.06));border:1px solid var(--dt-border,rgba(128,128,128,.25));
  border-radius:12px;padding:13px 15px;border-top:3px solid var(--c,var(--dt-accent,#3b82f6))}
.doc-card b{display:block;font-size:.92rem;margin-bottom:4px}
.doc-card span{font-size:.82rem;opacity:.8;line-height:1.45}
.doc-stage-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:8px 0 14px}
.doc-stage{border:1px solid var(--dt-border,rgba(128,128,128,.25));border-radius:12px;overflow:hidden;
  background:var(--dt-surface,rgba(128,128,128,.06))}
.doc-stage .h{background:var(--c);color:#fff;font-weight:800;font-size:.8rem;letter-spacing:.08em;padding:8px 12px}
.doc-stage ul{margin:8px 0 10px 0;padding-left:28px;font-size:.84rem;line-height:1.6}
.doc-flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0 14px}
.doc-flow .s{border:1px solid var(--dt-border,rgba(128,128,128,.3));background:var(--dt-surface,rgba(128,128,128,.06));
  border-radius:999px;padding:5px 12px;font-size:.8rem;font-weight:600}
.doc-flow .s.end{background:#10b981;color:#fff;border-color:#10b981}
.doc-flow .a{opacity:.55;font-weight:700}
.doc-figure{margin:6px 0 4px;padding:10px;border:1px solid var(--dt-border,rgba(128,128,128,.25));border-radius:12px;
  background:#ffffff;overflow-x:auto}
.doc-figure img{width:100%;min-width:640px;height:auto;display:block}
.doc-figure figcaption{color:#475569;font-size:.78rem;margin-top:6px;text-align:center}
</style>""", unsafe_allow_html=True)


def _cards(items: list[tuple[str, str]], color: str = "") -> None:
    style = f" style='--c:{color}'" if color else ""
    body = "".join(f"<div class='doc-card'{style}><b>{html.escape(t)}</b><span>{html.escape(d)}</span></div>"
                   for t, d in items)
    st.markdown(f"<div class='doc-grid'>{body}</div>", unsafe_allow_html=True)


def _flow() -> None:
    parts = []
    for i, step in enumerate(PRIMARY_FLOW):
        cls = "s end" if i == len(PRIMARY_FLOW) - 1 else "s"
        parts.append(f"<span class='{cls}'>{html.escape(step)}</span>")
        if i < len(PRIMARY_FLOW) - 1:
            parts.append("<span class='a'>→</span>")
    st.markdown(f"<div class='doc-flow'>{''.join(parts)}</div>", unsafe_allow_html=True)


def _stages() -> None:
    body = "".join(
        f"<div class='doc-stage' style='--c:{c}'><div class='h'>{n} · {name}</div><ul>"
        + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul></div>"
        for n, name, c, items in EXECUTIVE_STAGES)
    st.markdown(f"<div class='doc-stage-grid'>{body}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_overview(lim: dict) -> None:
    st.markdown(
        "<div class='doc-hero'><h3>DataTalk.AI — conversational analytics for Appeals &amp; Grievances</h3>"
        "<p>Ask a question in plain English and receive a data-grounded answer from live Databricks data: a direct "
        "finding, reconciled charts, an evidence-linked insight panel, recommended actions with Outlook drafts, the "
        "result table and the exact SQL that produced it. The AI understands the question and writes the query; "
        "every number shown is computed from the live result and verified before it is displayed.</p></div>",
        unsafe_allow_html=True)

    section("🎯 Business problem")
    st.markdown(
        "Appeals operations and compliance teams are measured against turnaround (SLA) and compliance targets and "
        "need fast answers to ad-hoc questions about units, sites, plan sponsors, providers and trends. Today each "
        "question needs an analyst, a SQL query, a chart and a written explanation, so answers are slow and "
        "inconsistent, and dashboards only cover the questions someone anticipated. A language model alone is not "
        "trustworthy with numbers, so DataTalk.AI uses AI for understanding and wording and keeps every calculation "
        "deterministic and verifiable.")

    section("🧭 Project objectives")
    _cards([
        ("Answer in seconds", "Turn free-text questions — including multi-part and dependent ones — into governed, "
                              "read-only queries on live data."),
        ("Zero untraceable numbers", "Every figure is traced to a query result or a documented calculation; "
                                     "unverifiable AI sentences are removed."),
        ("Decision-ready output", "Plain-English findings, business impact, owned and measurable recommended "
                                  "actions, ready-to-edit Outlook drafts."),
        ("Transparent and safe", "Exact SQL shown with every answer; strict read-only access to one approved "
                                 "table; consistent, validated AI behaviour."),
    ])

    section("✨ Key capabilities")
    _cards([
        ("Natural-language analytics", f"Up to {lim['max_sub']} questions in one message, with dependencies "
                                       "(“…then break those down by specialty”)."),
        ("Governed SQL", "One read-only Databricks SELECT per question, validated before it can run."),
        ("Verified insights", "Evidence registry (F1…Fn) behind every key finding, analysis and impact statement."),
        ("Reconciled charts", "Chart type chosen from the result shape; every plotted value checked against the table."),
        ("Recommended actions", "Priority, owner role, timeframe and success measure, each linked to evidence."),
        ("Quick actions", "Instant deterministic analyses: forecast, anomalies, overdue, sponsors, comparisons and more."),
        ("Example questions", "Curated, categorised business questions that run through the same workflow."),
        ("Dashboards", "Deep Dive, Intelligence, Risk & Findings and Data Explorer on the same data."),
        ("Voice", "Speak a question, review the transcript, then send; listen to answers read aloud."),
    ])

    section("🪜 Executive view — five stages")
    _stages()
    _render_diagram("overview", "DataTalk.AI overview")


def _tab_architecture(lim: dict) -> None:
    section("➡️ Primary flow")
    _flow()
    st.markdown(f"""
| Step | What happens | Performed by |
|---|---|---|
| User Question | Typed, spoken (reviewed before sending), a quick action or an example question | User |
| Request Routing | Input validated ({MIN_QUESTION_CHARS}–{MAX_QUESTION_CHARS:,} characters); quick actions go to deterministic analytics, open questions to the AI pipeline | Supervisor Agent |
| Context Grounding | Approved schema, certified metric definitions and real data values supplied to every AI step | Analytics Reasoning Agent |
| Intent Understanding | Message split into up to {lim['max_sub']} sub-questions with metrics, filters, groupings and dependencies | Planner Agent (AI) |
| Query Planning | Independent sub-questions run first; dependent ones run after and receive upstream results | Analytics Reasoning Agent |
| SQL Generation | One read-only query per sub-question using the certified metric definitions | SQL Agent (AI) |
| Safety Validation | Read-only, approved table and columns, AST syntax, single statement, safe row limits; one self-healing repair | SQL safety guard + SQL Agent |
| Live Data Execution | Validated SQL runs on Databricks (max {lim['max_rows']:,} rows) | Databricks SQL warehouse |
| Insight & Chart Synthesis | Facts computed by code, worded by AI and verified; rule-based charts reconciled to the data | Insight Generation + Visualization Agents |
| Response QA | Multi-part answers combined and reviewed for gaps or unsupported claims | Response Agent (AI) |
| Data-Grounded Answer | Answer first, then charts, insight panel, result table and the exact SQL | AI Assistant page |
""")

    section("🏗️ End-to-end architecture")
    st.markdown("""
| Layer | Components | Responsibility |
|---|---|---|
| Presentation | Streamlit pages: AI Assistant, Deep Dive, Intelligence, Risk & Findings, Data Explorer, Documentation | Interaction, rendering, exports, Outlook links |
| Orchestration | Supervisor Agent, Analytics Reasoning Agent | Routing, sub-question execution order, fallbacks |
| AI agents | Planner, SQL (generation + repair), Insight narrative, Response (synthesis + QA) | Understanding, query drafting, wording, review — all through one validated LLM client |
| Deterministic services | SQL safety guard, chart builder, evidence registry and verification, quick actions, anomaly detection, forecasting | Every calculation, check and visual |
| Data | One approved appeals table in Databricks, read-only | Single source of truth |
""")
    section("🔄 End-to-end data flow")
    _render_diagram("flow", "DataTalk.AI data flow")
    others = [d for d in _diagrams() if d["kind"] == "other"]
    if others:
        section("🖼️ Additional diagrams")
        for d in others:
            _figure(d, "Diagram")


def _tab_pipeline(lim: dict) -> None:
    st.caption("Each component below states what it does, whether it uses AI, and what happens when it cannot "
               "complete its job.")
    section("🧭 Routing and input")
    with st.expander("🛂 Supervisor Agent & request routing — decides how every question is answered", expanded=True):
        st.markdown(f"""
The Supervisor makes no AI calls itself. For each question it applies, in order:

1. **Input validation** — the question must be {MIN_QUESTION_CHARS}–{MAX_QUESTION_CHARS:,} characters; otherwise a guidance message is shown and nothing runs.
2. **Quick actions** — recognised quick-action requests are answered instantly by deterministic analytics on the loaded dataset.
3. **AI configuration check** — if the AI service is not configured, a clear message is shown; quick actions and dashboards keep working.
4. **Analytics Reasoning Agent** — open questions go through the AI reasoning pipeline.
5. **Direct-SQL fallback** — only if the pipeline fails unexpectedly (see Reliability).
""")
    with st.expander("✅ Input validation — protects the pipeline from empty or oversized requests"):
        st.markdown(f"Whitespace is collapsed before checks. Inputs shorter than {MIN_QUESTION_CHARS} characters ask "
                    f"for a more specific question; inputs longer than {MAX_QUESTION_CHARS:,} characters ask the "
                    "user to shorten them. Spoken questions are always shown as editable text before sending.")
    with st.expander("⚡ Quick Actions — instant, deterministic analyses with no AI"):
        st.markdown("""
Buttons and recognised phrases run pandas analytics on the loaded dataset — no AI and no SQL generation:
forecast (next 6 months), anomaly scan, overdue appeals, top plan sponsors, site comparison, escalated and pending
appeals, risk overview, reviewer segments, monthly trend, appeals by site and by unit, source and filer breakdown,
task breakdown. A quick-action result can also narrow the dashboard population (for example to overdue cases);
**Show all appeals** restores the full view.
""")

    section("🧠 Understanding the question")
    with st.expander("🎛️ Analytics Reasoning Agent — orchestrates the AI pipeline"):
        st.markdown("""
Runs plan → SQL → validation → execution → charts → insight → synthesis for one message.
Independent sub-questions run first; dependent sub-questions run afterwards and receive up to 10 verified rows of
the result they depend on, so “which sponsor has the most appeals and how are *those* appeals determined?” is
answered without asking the user to repeat the sponsor. A failure in one sub-question never stops the others.
""")
    with st.expander("📚 Context grounding — schema and business-rule grounding"):
        st.markdown(f"""
Every AI step receives a closed-world contract:

* **Approved schema** — the single approved table and its raw columns. Python-only derived fields are listed as forbidden in SQL.
* **Certified metric dictionary** — one agreed SQL definition per business metric, for example:

| Metric | Definition |
|---|---|
| Compliance rate | Compliant closures ÷ closed cases × 100 (closed cases only) |
| Overturn rate | Overturned + partially overturned ÷ closed cases × 100 |
| Overdue appeals | Open appeals received more than {SLA_DAYS} days ago |
| Escalated appeals | Level type code = {ESCALATION_LEVEL} |
| Average turnaround (TAT) | Days from received to closed, closed cases only |

* **Real data values** — up to {lim['values']} actual values per categorical field (units, sites, sponsors…), so filters use names that exist.
* **As-of date** — relative periods such as “last 3 months” are resolved to explicit dates.
* **Recent conversation** — the planner also sees the last three chat messages for follow-up questions.

Business rules in force: SLA {SLA_DAYS} days · aging threshold {AGING_THRESHOLD} days · escalation level {ESCALATION_LEVEL}.
""")
    with st.expander("🗺️ Planner Agent — intent understanding and multi-part question decomposition (AI)"):
        st.markdown(f"""
Splits the message into up to **{lim['max_sub']} sub-questions**, keeping the user's wording, and classifies each:
analysis type (metric, breakdown, trend, comparison, ranking, distribution, drill-down, forecast, anomaly, root
cause, overview), time grain, dimensions, certified metrics, resolved filters and dependencies. Filter values not
present in the data are reported as not answerable rather than guessed. The reply must pass schema validation.
""")
    with st.expander("✂️ Rule-based question splitting fallback — no question is silently dropped"):
        st.markdown("""
If the planner is unavailable or returns an invalid plan, a deterministic splitter creates the sub-questions from
`?`, `;`, “and also”, “also tell me”, “also show”, “as well as”, “and then”, “plus also”, numbered and bulleted
lists. If the planner returns a single question but the splitter finds several, the splitter's result is used.
""")
    with st.expander("🚧 Off-topic handling — no SQL for unrelated questions"):
        st.markdown("When the planner judges the whole message unrelated to the appeals data, the user receives "
                    "guidance on the kinds of questions supported (status, turnaround, outcomes, providers, sponsors, "
                    "date ranges, overturns). No SQL is generated or run.")

    section("🧮 Building and answering")
    with st.expander("🧾 SQL Agent — SQL generation (AI)"):
        st.markdown(f"""
Writes exactly one Databricks SELECT per sub-question using only the approved table, backticked raw columns and the
certified metric definitions. Rates guard against division by zero and expose their denominator; trends return
formatted periods in order; rankings use an explicit LIMIT; row-level queries are limited to {lim['row_limit']} rows.
The query then goes through the safety guardrails (see **Safety & Guardrails**).
""")
    with st.expander("💡 Insight generation — facts computed by code, worded clearly"):
        st.markdown("""
The result is profiled (dimension, period, measure, unit, better-direction) and an **evidence registry** (F1…Fn) is
built: observed values plus derived statistics such as median, gaps, spread, weighted averages, outliers, trend
slope and SLA headroom. Each insight section is written deterministically from that evidence:
Key finding · Detailed analysis · Business impact · AI reasoning · Evidence · Recommended actions · Data considerations.
The AI may then reword sections for fluency, subject to evidence verification.
""")
    with st.expander("🔍 Evidence verification — no unverified numbers or causal claims"):
        st.markdown("""
Every AI-written sentence is checked before display:

* every number must match an evidence value at its displayed precision (13.93 may appear as 13.9, never as 14);
* causal wording (“because”, “driven by”, “due to”…) is removed;
* sections must not repeat each other.

Failing sentences are removed and the data-grounded deterministic version is used for that section. In multi-part
answers, the combined text is verified again against the evidence of every answered sub-question.
""")
    with st.expander("📊 Chart generation — rule-based and reconciled to the data"):
        st.markdown("""
Charts are chosen from the **shape of the verified result**, not by the AI: single value → KPI or gauge; periods →
trend line or area (bars when fewer than 3 periods); two dimensions → heatmap; category + measure → bar chart, with a
donut only for genuine shares; numeric pairs → scatter. Rates and averages are never summed or shown as shares.
Every plotted value is reconciled against the result table; a chart that does not reconcile is discarded and the
table is shown instead.
""")
    with st.expander("✅ Recommended Actions — owned, time-bound and measurable"):
        st.markdown("""
Each action has a priority (Immediate, Short-term, Strategic, Monitor), an owner role (never a person's name), a
timeframe and a success measure that uses evidence values as baseline or target, with links to its evidence. Each
action can be turned into a pre-filled Outlook draft.
""")
    with st.expander("🧩 Response Agent — multi-part response QA (AI)"):
        st.markdown("""
For messages with two or more sub-questions, the Response Agent assembles exactly one answer per sub-question,
including the ones that could not be answered, plus a one-line headline. A QA review then checks completeness,
question drift, unsupported claims, causal language and exposure of individual identifiers. Sub-questions that were
not answered are listed explicitly with the reason. Single questions skip this stage and use the verified key finding.
""")


def _tab_safety(lim: dict) -> None:
    section("🛡️ SQL safety and policy guardrails")
    _cards([
        ("Strict read-only access", "Only a single SELECT can run. Writes, deletes and DDL are blocked."),
        ("Approved table & columns only", "One approved table; raw columns only; derived fields rejected."),
        ("Blocks writes, drops & comments", "INSERT, UPDATE, DELETE, MERGE, DROP, CREATE, ALTER, TRUNCATE, GRANT… and SQL comments."),
        ("Validates SQL syntax (AST)", "The statement is parsed and checked structurally, not by text matching."),
        ("Single-statement only", "Stacked statements are rejected."),
        ("Enforces safe row limits", f"Row-level queries limited to {lim['row_limit']} rows; at most {lim['max_rows']:,} rows per query."),
    ], color="#f59e0b")
    with st.expander("🔒 Read-only enforcement"):
        st.markdown("Generated SQL is validated before execution and anything other than one read-only SELECT is "
                    "rejected. The Databricks credential should also hold SELECT permission only, so the database "
                    "itself enforces read-only access as a second line of defence.")
    with st.expander("📋 Approved tables and columns"):
        st.markdown("Only the configured appeals table may appear in FROM/JOIN, and only its raw columns may be "
                    "referenced. Python-only derived fields (for example risk scores and flags computed in the app) "
                    "are rejected in SQL.")
    with st.expander("🧱 SQL syntax / AST validation and single-statement enforcement"):
        st.markdown("Each statement is parsed into a syntax tree. Validation checks the statement type, the tables "
                    "and columns referenced, comments and the number of statements. Text tricks such as a second "
                    "statement after a semicolon or commented-out clauses are rejected.")
    with st.expander("📏 Safe row limits and the 5,000-row execution behaviour"):
        st.markdown(f"Row-level (non-aggregate) queries must carry a LIMIT of {lim['row_limit']}. Every generated "
                    f"query is executed with a hard cap of {lim['max_rows']:,} rows; results are never presented as "
                    "complete when they are not. For larger questions, ask for an aggregate or add filters.")
    with st.expander("🩹 Self-healing SQL repair"):
        st.markdown("If validation rejects a query, the SQL Agent makes **one** repair attempt that fixes only the "
                    "reported problems — it may not broaden filters, drop conditions or change the question. The "
                    "repaired query is validated again before it can run.")
    with st.expander("⛔ Unsafe-query blocking"):
        st.markdown("If the query is still invalid after repair, it is not run. The sub-question is reported as not "
                    "answered with plain-English guidance, and the rest of the message is still answered.")
    with st.expander("🗄️ Databricks live database execution"):
        st.markdown("Validated SQL runs on the Databricks SQL warehouse against the live appeals table, so answers "
                    "reflect the current data. The connection status and round-trip time are shown in the header.")
    with st.expander("🧯 Graceful query-error and no-data recovery"):
        st.markdown("If a query returns no rows, the answer explains that nothing matched and suggests checking "
                    "names, statuses and date ranges. If the database reports an error, the user sees a clear "
                    "message without technical details; details are written to the server log only.")
    with st.expander("🔐 Data protection"):
        st.markdown("Credentials come from the environment and are never shown or logged. Logs contain AI call "
                    "metadata only — no prompts, AI replies, query results or secrets. The AI receives the "
                    "question, schema, metric definitions, a capped list of real category values, recent chat "
                    "messages and the evidence values needed for wording.")


def _tab_outputs() -> None:
    section("📦 Output deliverables")
    _cards([
        ("AI Answer First", "The key finding in plain English, at the top of every answer."),
        ("Interactive Chart", "Reconciled visual breakdown; hover, zoom and export."),
        ("Data Table", "The verified result records behind the answer, inspectable in the app."),
        ("SQL Audit Trail", "The exact validated query, per sub-question, in the SQL tab."),
        ("Outlook Draft", "Pre-filled e-mail for a recommended action — never sent automatically."),
    ], color="#10b981")
    section("📰 Response-first delivery")
    st.markdown("""
Every answer is laid out in the same order:

1. **AI Insight Findings** — the direct answer (multi-part answers: a headline plus one answer per question, with any unanswered question listed explicitly).
2. **Charts** — reconciled to the result table.
3. **🔊 Listen** — read the answer aloud in the browser.
4. **Insight panel** — evidence tiles plus Detailed Analysis, Business Impact, AI Reasoning, Evidence, Recommended Actions, Data Considerations and SQL (one tab per question for multi-part answers).
5. **View result data** — the verified table.
""")
    section("✉️ Outlook draft capability")
    st.markdown("Recommended actions offer **Draft in Outlook**, which opens a compose window (Outlook on the web or "
                "the desktop mail client, as configured) with the subject and body pre-filled from the action and "
                "its evidence. The user reviews, edits and sends it; DataTalk.AI never sends e-mail.")
    section("💡 Example questions")
    st.markdown("The AI Assistant includes **Example questions** grouped by Performance, Drill-down and Trends. "
                "Selecting one asks it exactly as if it had been typed — same validation, routing, analytics, "
                "safety checks, visuals and response.")


def _tab_reliability(lim: dict) -> None:
    section("🔁 Error handling and fallback strategy")
    st.markdown(f"""
| Situation | What the user sees | Fallback |
|---|---|---|
| Empty, too short or too long input | Guidance message | Nothing runs |
| Question unrelated to appeals data | Guidance on supported questions | No SQL generated |
| AI service not configured | Configuration message | Quick actions and dashboards keep working |
| AI provider error or malformed reply | Normal answer (slightly slower) | Bounded retries (up to {lim['attempts']} attempts per call), then the step's backup |
| Planner unavailable or invalid | Normal answer | Rule-based question splitting |
| Generated SQL rejected | Normal answer | One self-healing repair, validated again |
| SQL still unsafe after repair | Sub-question marked not answered, with guidance | Other sub-questions still answered |
| No rows / database error | Clear explanation and what to check | Technical details logged server-side only |
| AI wording fails verification | Normal answer | Deterministic, data-grounded wording |
| Synthesis or QA review unavailable | Per-question verified answers | Deterministic assembly |
| Whole pipeline fails unexpectedly | Normal answer when possible | Direct-SQL fallback: SQL Agent → guardrails → execute → insight |
| Unexpected error in the chat | “Unable to complete this request” | Chat stays usable; error logged |
""")
    section("⚖️ Deterministic vs AI-driven processing")
    st.markdown("""
| Deterministic (code) | AI-driven (validated) |
|---|---|
| Input validation and routing | Understanding intent and splitting questions |
| Quick actions, forecasting, anomaly detection | Drafting SQL from the certified definitions |
| SQL safety validation and row limits | Repairing a rejected query (once) |
| Query execution on Databricks | Rewording insight sections for fluency |
| Every number, statistic and evidence item | Combining multi-part answers |
| Chart selection and reconciliation | QA review of multi-part answers |
| Evidence verification of AI text | |
| Dashboards and exports | |

**Consistent AI:** the same settings every time (temperature 0, fixed seed, deterministic prompt payloads); every AI
reply is schema-checked and retried if invalid; if a step still fails, its backup takes over.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════════════════════════════

def render_documentation_page() -> None:
    _css()
    page_title("📘", "DataTalk.AI Documentation",
               "How a plain-English question becomes a verified, data-backed answer — for business and technical "
               "audiences.")
    lim = _limits()
    tabs = st.tabs(["🏠 Overview", "🏗️ Architecture & Data Flow", "🧠 AI Reasoning Pipeline",
                    "🛡️ Safety & Guardrails", "📦 Outputs & Delivery", "🔁 Reliability"])
    with tabs[0]:
        _tab_overview(lim)
    with tabs[1]:
        _tab_architecture(lim)
    with tabs[2]:
        _tab_pipeline(lim)
    with tabs[3]:
        _tab_safety(lim)
    with tabs[4]:
        _tab_outputs()
    with tabs[5]:
        _tab_reliability(lim)
