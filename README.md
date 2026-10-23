# DataTalk.AI — Healthcare Appeals Intelligence (POC)

DataTalk.AI is a Streamlit application that lets appeals operations and compliance teams ask plain-English questions about an Appeals & Grievances portfolio and get **data-grounded answers from live Databricks data**: a direct finding, reconciled charts, an evidence-linked insight panel, recommended actions with Outlook drafts, the result table and the exact SQL.

The AI understands the question and drafts SQL; **code computes, validates and verifies every number**. Detailed architecture and data-flow explanations live in the app's **📘 Documentation** page.

## Objectives

- Answer free-text questions — including multi-part and dependent ones — from governed, read-only SQL in seconds.
- Never display a number that cannot be traced to a query result or a documented calculation.
- Deliver decision-ready output: findings, business impact, owned and measurable actions.
- Keep AI behaviour consistent, validated and transparent (exact SQL shown with every answer).

## Key capabilities

| Capability | Summary |
|---|---|
| AI Assistant | Plain-English, voice (reviewed before sending), quick actions and **Example questions**; up to 5 sub-questions per message |
| Governed SQL | One read-only Databricks SELECT per sub-question, validated before it runs, one self-healing repair |
| Verified insights | Evidence registry (F1…Fn); AI sentences with untraceable numbers or causal claims are removed |
| Reconciled charts | Rule-based chart choice; every plotted value checked against the result |
| Recommended actions | Priority, owner role, timeframe, success measure; **Draft in Outlook** (never auto-sent) |
| Quick actions | Deterministic forecast, anomaly scan, overdue, sponsors, comparisons and more (no AI) |
| Dashboards | Deep Dive, Intelligence, Risk & Findings, Data Explorer |
| Documentation | Business and technical guide with the SVG diagrams from `docs/` |
| Q&A store | Backend-only reuse of verified answers via `QA.json` (invisible in the UI) |

## Architecture

```
Streamlit pages (dashboards/, components/)
        │
Supervisor Agent ── input check → quick action → QA.json lookup → AI pipeline → direct-SQL fallback
        │
Analytics Reasoning Agent ── Planner (AI) → SQL Agent (AI) → SQL guard → Databricks
        │                     → Visualization (code) → Insight composer (code + verified AI wording)
        │                     → Response Agent (AI, multi-part only)
Core services ── settings · LLM client · SQL guard · data load · quick actions · QA store
```

**Simplified data flow:** User Question → Request Routing → Context Grounding → Intent Understanding → Query Planning → SQL Generation → Safety Validation → Live Data Execution → Insight & Chart Synthesis → Response QA → Data-Grounded Answer.

## Project structure

```
datatalk-ai/
├── app.py                      # Entry point: config, data load, header/KPIs, navigation, sidebar, footer
├── prompts.py                  # Shared prompt blocks (schema contract, metric dictionary) + PROMPT_VERSION
├── anomaly_detection.py        # Seven statistical anomaly detectors (pandas)
├── speech_utils.py             # Speech-to-text input and read-aloud
├── agents/
│   ├── supervisor_agent.py     # Routing, Q&A store lookup/save, direct-SQL fallback (no LLM)
│   ├── analytics_reasoning_agent.py  # Orchestrator: plan → SQL → execute → charts → insight → synthesis
│   ├── planner_agent.py        # LLM: sub-question plan + rule-based split fallback
│   ├── sql_agent.py            # LLM: SQL generation + one repair
│   ├── insight_generation_agent.py   # Evidence registry, deterministic sections, verified AI wording
│   ├── response_agent.py       # LLM: multi-part synthesis + QA review
│   └── visualization_agent.py  # Deterministic, reconciled charts
├── core/
│   ├── settings.py             # Settings (env/.env), business rules, schema
│   ├── llm.py                  # Single LLM client: retries, Pydantic parsing, call tracking
│   ├── schemas.py              # Shared enums + AssistantAnswer
│   ├── sql_guard.py            # Read-only SQL validator + executor
│   ├── data.py                 # Databricks load, derived columns, KPIs, grounding values
│   ├── fast_paths.py           # Deterministic quick actions and forecast
│   ├── qa_store.py             # QA.json question-answer store (backend only)
│   ├── example_questions.py    # Example-question catalogue
│   ├── plot_style.py           # Theme-neutral Plotly styling
│   └── runtime.py              # Streamlit caching wrappers + session state
├── components/                 # ui, insight_panel, risk_findings, outlook_mail, visual_pipeline
├── dashboards/                 # ai_assistant, deep_dive_dashboard, intelligence_dashboard,
│                               # data_explorer, documentation
├── docs/                       # datatalk_overview.svg, datatalk_dataflow_diagram.svg (Documentation page)
├── assets/                     # Fallback location for diagrams (e.g. ai_data_flow.svg)
├── data/QA.json                # Q&A store (created/updated at runtime)
├── .streamlit/config.toml      # Light/dark themes
├── pyproject.toml · uv.lock · .python-version · .env.example
└── QA.txt                      # Development question bank
```

## Setup and run

Prerequisites: **uv** ≥ 0.9, Python 3.12+ (uv installs the pinned version), a Databricks SQL warehouse with SELECT on the appeals table, and a Groq API key (AI answers only — dashboards and quick actions work without it).

```bash
uv sync                                   # install locked dependencies
cp .env.example .env                      # Windows: Copy-Item .env.example .env, then edit values
uv run streamlit run app.py               # http://localhost:8501
uv run ruff check .                       # lint (rules in pyproject.toml)
```

Behind a corporate TLS proxy, set `UV_SYSTEM_CERTS=1` so uv trusts the OS certificate store.

## Configuration

Real environment variables override `.env`, which overrides code defaults. Secrets (🔒) are never logged or displayed.

| Variable | Default | Purpose |
|---|---|---|
| `DATABRICKS_SERVER_HOSTNAME` / `DATABRICKS_HTTP_PATH` | — | SQL warehouse connection (required) |
| `DATABRICKS_ACCESS_TOKEN` 🔒 | — | Access token, SELECT-only recommended (required) |
| `DATABRICKS_CATALOG` / `DATABRICKS_SCHEMA` / `DATABRICKS_TABLE` | — / — / `appeals_final` | The single approved table |
| `GROQ_API_KEY` 🔒 / `GROQ_MODEL` | — / `openai/gpt-oss-20b` | LLM provider and model |
| `LLM_TEMPERATURE` / `LLM_SEED` / `LLM_MAX_ATTEMPTS` | `0.0` / `7` / `3` | Consistent, bounded AI behaviour |
| `MAX_SUB_QUESTIONS` / `MAX_ROW_QUERY_LIMIT` / `MAX_RESULT_ROWS` | `5` / `200` / `5000` | Plan size and SQL row limits |
| `ENABLE_SQL_REPAIR` / `ENABLE_RESPONSE_VALIDATOR` / `INSIGHT_LLM_NARRATIVE` | `true` | Optional AI steps |
| `QA_CACHE_ENABLED` | `true` | Turn the Q&A store on/off |
| `QA_CACHE_PATH` | `data/QA.json` | Q&A store location (relative paths resolve from the project root) |
| `QA_CACHE_MAX_ENTRIES` / `QA_CACHE_MAX_ENTRY_KB` | `500` / `768` | Store size limits (oldest entries dropped first) |
| `QA_CACHE_MAX_AGE_HOURS` | `0` | Expire stored answers after N hours (`0` = never) |
| `INSIGHT_MAIL_CLIENT` / `INSIGHT_MAIL_CC` | `outlook_web` / — | Outlook draft target and CC |
| `LOG_LEVEL` | `INFO` | `datatalk.*` loggers |

Business rules (SLA 30 days, aging threshold 60 days, escalation level 2) and the approved column list are in `core/settings.py`.

## Databricks integration

Data is read through `databricks-sql-connector` from one approved table. The full table is loaded once for KPIs, dashboards and quick actions (**Refresh data** in the sidebar reloads it). AI answers always run a fresh, validated query against the live table, capped at `MAX_RESULT_ROWS`.

## AI / LLM integration

All model calls go through one client (`core/llm.py`): Groq chat completions with JSON output, temperature 0, fixed seed, Pydantic-validated replies, bounded retries with back-off and per-call tracking (purpose, tokens, latency — never prompt or response content in logs). Each agent has a deterministic fallback, so a failed AI step degrades the answer rather than breaking it. Prompts live with their agents; shared blocks live in `prompts.py`.

## Q&A caching (`QA.json`)

A backend-only store that sits **in front of every AI/LLM call** (`core/qa_store.py`, used by `agents/supervisor_agent.py`):

1. Input checks and quick actions run first (they never call the AI).
2. The question is normalised (Unicode NFKC, case-folded, quotes/extra whitespace/trailing punctuation removed) and looked up in `QA.json`.
3. **Hit** → the stored answer (text, table, charts, insight panel) is rebuilt and returned with **zero AI calls** — even if the AI provider is unavailable.
4. **Miss** → the normal AI workflow runs; a fully answered result is saved with the original question, normalised question and answer.

Not stored: partial or fallback answers, row-level/person-level results, questions containing long digit runs or e-mail addresses, and follow-ups that depend on earlier chat turns ("break *those* down…"). Answers to relative-date questions ("last 3 months") are reused only on the same day.

Resilience: persisted across restarts; duplicates prevented; writes use a file lock plus atomic replace and always preserve existing entries; Unicode kept as UTF-8; missing/empty files start a new store; malformed or duplicate entries are skipped; a corrupted file is renamed `QA.corrupt-<timestamp>.json` and a fresh store is started. The UI never indicates whether an answer was reused. To reset, stop the app and delete `data/QA.json`.

## Safety and SQL guardrails

Generated SQL must be a **single read-only SELECT** on the approved table using raw columns only, with no comments or mutation keywords; it is validated structurally (sqlglot AST), row-level queries are limited to `MAX_ROW_QUERY_LIMIT`, and results are capped at `MAX_RESULT_ROWS`. A rejected query gets one repair attempt and is validated again; if still unsafe it is not run and the sub-question is reported as not answered. Database and validator details are logged, not shown to users.

## Documentation page

**📘 Documentation** (top navigation and sidebar link) explains objectives, architecture, the primary flow, the executive five-stage flow (Ask · Understand · Verify · Execute · Deliver), every agent, guardrails, outputs, error handling and deterministic-vs-AI processing. It renders every `*.svg` in `docs/` (falling back to `assets/`): files whose name or `<title>` contains "overview" appear on the Overview tab, "flow" on the Architecture tab, others under Additional diagrams. Replace an SVG and reload to update.

## Example questions

The AI Assistant includes an **💡 Example questions** panel with four curated questions grouped into Performance, Drill-down and Trends. Selecting a question submits it through the same path as typed input (validation, routing, Q&A store, SQL, safety, charts, response). Edit the catalogue in `core/example_questions.py`.

## Error handling and fallbacks

Invalid input and off-topic questions get guidance messages; AI errors are retried, then the step's fallback is used (rule-based question split, deterministic insight wording, per-question assembly); unexpected pipeline failures use the direct-SQL fallback; query errors and empty results are explained in business language. Unexpected chat errors show a generic message and are logged with stack traces server-side only.

## Security considerations

- Credentials come only from the environment/`.env` (git-ignored) and are never shown or logged.
- Use a SELECT-only Databricks token as a second line of defence behind the SQL guard.
- Data sent to the LLM: question, schema, metric definitions, capped category values, last three chat messages and evidence values. Review against your PHI/PII policy before use with real data.
- `QA.json` holds aggregate answers on local disk — treat it as internal data. There is no built-in authentication; anyone who can reach the app sees all data and shares the Q&A store.

---
Made and Designed by EXL India | 2026
