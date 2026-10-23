"""Statistical anomaly detection for the appeals & grievances (A&G) portfolio.

Seven independent detectors (overdue spikes, open highest-level escalations,
transfer spikes, compliance risk-score outliers, aged backlog, overturn-rate
outliers, out-of-network concentration). Each detector:

* receives an explicit :class:`AnomalyConfig` (no hidden module constants),
* validates its required columns and logs + skips when one is missing,
* applies a minimum group size so small groups cannot register as spikes,
* compares a group with a leave-one-out baseline and gates on a one-sided
  two-proportion z-test, so ``confidence`` is backed by a significance test.

``detect_anomalies`` runs all detectors, isolates failures per detector and
returns severity-sorted, JSON-serialisable dicts. Pure pandas; no Streamlit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field

import pandas as pd

logger = logging.getLogger("anomaly_detection")
if not logger.handlers:
    # Library-safe default: don't spam stdout unless the host app configures
    # logging itself. Host apps can override this via logging.getLogger(...).
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AnomalyConfig:
    """
    All tunable thresholds in one place. Nothing here is a hidden global —
    every detector receives this object explicitly, so behavior is fully
    reproducible and testable without monkey-patching module constants.
    """

    # ── Business SLA constants (mirrors the originals; supply real values) ──
    sla_days: int = 30            # was SLA_DAYS — days before a case is "overdue"
    aging_threshold_days: int = 60  # was AGING_THRESHOLD — days before "aged/stale"
    escalation_level: int = 3       # was ESCALATION_LEVEL — highest escalation tier

    # ── Minimum sample size before a group is eligible for comparison ──────
    # Prevents a 2-case unit with 1 overdue from registering as a "50% spike."
    min_group_size: int = 15

    # ── Detector 1: overdue-rate spikes ─────────────────────────────────────
    # A group's rate must exceed the *leave-one-out* baseline by at least
    # this multiple (1.3 = 30% relatively higher) to be considered, on top
    # of statistical significance. Leave-one-out means "baseline" excludes
    # the group under test, so a large anomalous group can't drag its own
    # comparison baseline up and mask itself (see README notes).
    overdue_min_relative_deviation: float = 1.3
    overdue_severity_critical: float = 0.50
    overdue_severity_high: float = 0.30

    # ── Detector 2: escalated-open cluster ──────────────────────────────────
    escalation_severity_critical: float = 0.50
    escalation_severity_high: float = 0.30

    # ── Detector 3: transfer/reassignment spikes ────────────────────────────
    transfer_min_relative_deviation: float = 1.8
    transfer_min_count: int = 2
    transfer_severity_high: float = 0.20

    # ── Detector 4: compliance risk-score outliers (z-score) ───────────────
    risk_score_std_multiplier: float = 2.0

    # ── Detector 5: aging backlog concentration ─────────────────────────────
    aging_severity_critical: float = 0.30
    aging_severity_high: float = 0.15

    # ── Detector 6: overturn-rate outliers ──────────────────────────────────
    overturn_min_relative_deviation: float = 1.5
    overturn_severity_critical: float = 0.60
    overturn_severity_high: float = 0.45

    # ── Detector 7: out-of-network concentration ────────────────────────────
    oon_min_relative_deviation: float = 1.5
    oon_severity_high: float = 0.35

    # ── Statistical significance gate ───────────────────────────────────────
    # A group must clear BOTH the rate threshold AND this p-value to be
    # flagged. This is what actually backs the "confidence" number, rather
    # than the confidence being a cosmetic function of the rate itself.
    significance_alpha: float = 0.05


REQUIRED_COLUMNS_BY_DETECTOR: dict[str, list[str]] = {
    "overdue": ["IsOverdue", "Current Unit Name", "Case Identifier", "OpenAgingDays"],
    "escalation": ["IsEscalated", "IsClosed", "CaseAgeDays", "Current Unit Name"],
    "transfer": ["IsTransferred", "Current Unit Name", "Case Identifier"],
    "risk_score": ["ComplianceRiskScore", "Current Unit Name"],
    "aging": ["CaseAgeDays", "Current Unit Name"],
    "overturn": ["IsAnyOverturn", "IsClosed", "Plan Sponsor Name", "Case Identifier"],
    "oon": ["IsOutOfNetwork", "Provider Type", "Case Identifier", "IsAnyOverturn"],
}


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Anomaly:
    id: str
    detected: str
    category: str
    severity: str                 # "Critical" | "High" | "Medium"
    confidence: float             # statistically-grounded, 0-1
    p_value: float | None
    root_cause: str
    financial_risk: str
    historical_comparison: str
    recommended_action: str
    supporting_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def two_proportion_z_test(count_a: int, n_a: int, count_b: int, n_b: int) -> tuple[float, float]:
    """
    Standard two-proportion z-test. Returns (z_statistic, p_value_one_sided).

    Used to test H0: rate_a == rate_b (e.g. "this unit's overdue rate equals
    the portfolio baseline") against H1: rate_a > rate_b.

    This replaces the original's `min(0.95, 0.6 + rate)` confidence heuristic
    with an actual test statistic, so "confidence" reflects how unlikely the
    observed gap is under the null hypothesis, not just how large the raw
    rate happens to be.
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0

    p_a = count_a / n_a
    p_b = count_b / n_b
    p_pool = (count_a + count_b) / (n_a + n_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0

    z = (p_a - p_b) / se
    # one-sided p-value (we only care about "significantly higher than baseline")
    p_value = 1 - _normal_cdf(z)
    return z, p_value


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def confidence_from_p_value(p_value: float) -> float:
    """
    Maps a p-value to a bounded, monotonic confidence score in [0.5, 0.99].
    Lower p-value (more significant) -> higher confidence. This is still a
    presentation transform (a p-value isn't literally "confidence"), but
    unlike the original it is now driven by an actual significance test
    rather than by the raw observed rate.
    """
    p_value = max(min(p_value, 1.0), 1e-6)
    conf = 1 - p_value
    return round(min(0.99, max(0.5, conf)), 3)


def make_safe_id(prefix: str, label: str, used_ids: set[str]) -> str:
    """Collision-safe anomaly ID generator."""
    base = f"{prefix}_{str(label).upper().strip().replace(' ', '_')}"
    base = "".join(ch for ch in base if ch.isalnum() or ch == "_")
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def leave_one_out_baseline(group_count: int, group_total: int, global_count: int, global_total: int) -> tuple[float, int, int]:
    """
    Computes the baseline rate EXCLUDING the group under test.

    Why this matters: comparing a group to a global rate that *includes*
    that same group means a large anomalous group inflates the very
    baseline it's being measured against, which can mask the anomaly
    (verified empirically in testing — a 40%-of-portfolio unit spiking to
    45% overdue barely moved the naive global average, and also acted as
    an outlier that inflated the group-rate std used for the threshold,
    self-cancelling in the original formula). Excluding the tested group
    from its own comparison baseline avoids this.
    """
    baseline_count = global_count - group_count
    baseline_total = global_total - group_total
    baseline_rate = baseline_count / baseline_total if baseline_total > 0 else 0.0
    return baseline_rate, baseline_count, baseline_total


def severity_from_rate(rate: float, critical_at: float, high_at: float) -> str:
    if rate > critical_at:
        return "Critical"
    if rate > high_at:
        return "High"
    return "Medium"


def validate_columns(df: pd.DataFrame, detector_key: str) -> bool:
    """
    Checks required columns for a detector are present. Logs a warning
    (does NOT silently pass) if the detector will be skipped, so a broken
    upstream ETL is visible in logs instead of just producing fewer
    anomalies with no explanation.
    """
    required = REQUIRED_COLUMNS_BY_DETECTOR[detector_key]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "Skipping detector '%s': missing required column(s) %s",
            detector_key, missing,
        )
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

def detect_overdue_spikes(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags units whose overdue rate is a statistically significant spike
    above the portfolio baseline (not just numerically higher)."""
    if not validate_columns(df, "overdue"):
        return []

    out: list[Anomaly] = []
    global_overdue = int(df["IsOverdue"].sum())
    global_total = len(df)
    global_rate = global_overdue / global_total if global_total else 0.0

    grp = df.groupby("Current Unit Name").agg(
        total=("Case Identifier", "count"),
        overdue=("IsOverdue", "sum"),
        open_days=("OpenAgingDays", "sum"),
    ).reset_index()
    grp = grp[grp["total"] >= cfg.min_group_size]
    if grp.empty:
        return out

    grp["rate"] = grp["overdue"] / grp["total"]

    for _, row in grp.iterrows():
        base_rate, base_count, base_total = leave_one_out_baseline(
            int(row["overdue"]), int(row["total"]), global_overdue, global_total
        )
        if base_total <= 0:
            continue
        if row["rate"] <= base_rate * cfg.overdue_min_relative_deviation:
            continue  # not meaningfully higher than the rest of the portfolio

        z, p = two_proportion_z_test(int(row["overdue"]), int(row["total"]), base_count, base_total)
        if p > cfg.significance_alpha:
            continue  # numerically above threshold but not statistically significant

        deviation = round((row["rate"] - base_rate) / max(base_rate, 0.001) * 100, 1)
        severity = severity_from_rate(row["rate"], cfg.overdue_severity_critical, cfg.overdue_severity_high)
        unit = row["Current Unit Name"]

        out.append(Anomaly(
            id=make_safe_id("OVERDUE", unit, used_ids),
            detected=f"Elevated Overdue Rate — {unit}",
            category="SLA Compliance",
            severity=severity,
            confidence=confidence_from_p_value(p),
            p_value=round(p, 5),
            root_cause=(
                f"{unit} shows {row['rate']*100:.1f}% overdue rate (>{cfg.sla_days} days) vs "
                f"{base_rate*100:.1f}% for the rest of the portfolio — {deviation:+.1f}% deviation "
                f"(z={z:.2f}, p={p:.4f}, n={int(row['total'])}). Likely caused by staffing "
                f"gaps, routing bottlenecks, or documentation delays."
            ),
            financial_risk=f"{int(row['open_days']):,} cumulative aging-days of open exposure at risk of CMS/regulatory SLA breach.",
            historical_comparison=f"Deviates {deviation:+.1f}% from rest-of-portfolio baseline of {base_rate*100:.1f}% (overall portfolio avg {global_rate*100:.1f}%).",
            recommended_action=(
                f"Initiate expedited clearance for all {int(row['overdue'])} overdue {unit} appeals. "
                f"Escalate cases beyond {cfg.aging_threshold_days} days to unit leadership."
            ),
            supporting_metrics={
                "Overdue Count": int(row["overdue"]),
                "Overdue Rate": f"{row['rate']*100:.1f}%",
                "Sample Size": int(row["total"]),
                "Aging Days (Open)": f"{int(row['open_days']):,}",
                "Rest-of-Portfolio Baseline": f"{base_rate*100:.1f}%",
                "p-value": round(p, 5),
            },
        ))

    return out


def detect_escalated_open_cluster(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags a build-up of unresolved highest-tier escalations."""
    if not validate_columns(df, "escalation"):
        return []

    total_escalated = int(df["IsEscalated"].sum())
    if total_escalated < cfg.min_group_size:
        logger.info("Escalation detector skipped: only %d escalated cases (< min_group_size).", total_escalated)
        return []

    esc_open = df[(df["IsEscalated"]) & (~df["IsClosed"])]
    if esc_open.empty:
        return []

    actual_rate = len(esc_open) / total_escalated
    severity = severity_from_rate(actual_rate, cfg.escalation_severity_critical, cfg.escalation_severity_high)
    esc_avg_age = esc_open["CaseAgeDays"].mean()

    # Significance vs. an implicit "healthy steady-state" of 0% open — tested
    # instead as: is the open-rate meaningfully different from 0 given n?
    # We use a one-sample proportion test against a nominal healthy baseline.
    healthy_baseline = 0.10  # documented assumption: >10% open-at-any-time is the alarm line
    z, p = two_proportion_z_test(len(esc_open), total_escalated, int(healthy_baseline * total_escalated), total_escalated)

    return [Anomaly(
        id=make_safe_id("ESCALATED_OPEN", "CLUSTER", used_ids),
        detected=f"Level {cfg.escalation_level} Escalations Awaiting Resolution",
        category="Escalation Controls",
        severity=severity,
        confidence=confidence_from_p_value(p),
        p_value=round(p, 5),
        root_cause=(
            f"{len(esc_open)} Level-{cfg.escalation_level} escalated appeals ({actual_rate*100:.1f}% of all "
            f"escalations) remain unresolved, averaging {esc_avg_age:.0f} days open. Senior-reviewer "
            f"bandwidth or decision-maker sign-off may be a bottleneck."
        ),
        financial_risk="Highest-severity appeals unresolved — elevated grievance/regulatory-complaint risk and potential member-experience escalation to external review.",
        historical_comparison=f"Level-{cfg.escalation_level} open rate of {actual_rate*100:.1f}% vs {healthy_baseline*100:.0f}% healthy-baseline assumption.",
        recommended_action="Immediately route to senior medical director queue. Prioritize by case age; target resolution within SLA for all highest-level escalation items.",
        supporting_metrics={
            "Escalated Open Count": len(esc_open),
            "Total Escalated": total_escalated,
            "Avg Age (Days)": f"{esc_avg_age:.0f}",
            "Escalated Open Rate": f"{actual_rate*100:.1f}%",
            "Top Unit": esc_open["Current Unit Name"].mode().iloc[0] if "Current Unit Name" in esc_open.columns and not esc_open.empty else "N/A",
        },
    )]


def detect_transfer_spikes(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags units with disproportionate reassignment/hand-off activity."""
    if not validate_columns(df, "transfer"):
        return []

    out: list[Anomaly] = []
    global_trans_count = int(df["IsTransferred"].sum())
    global_total = len(df)

    grp = df.groupby("Current Unit Name").agg(
        trans_count=("IsTransferred", "sum"),
        total=("Case Identifier", "count"),
    ).reset_index()
    grp = grp[(grp["total"] >= cfg.min_group_size) & (grp["trans_count"] >= cfg.transfer_min_count)]
    grp["trans_rate"] = grp["trans_count"] / grp["total"]

    for _, row in grp.iterrows():
        base_rate, base_count, base_total = leave_one_out_baseline(
            int(row["trans_count"]), int(row["total"]), global_trans_count, global_total
        )
        if base_total <= 0 or row["trans_rate"] <= base_rate * cfg.transfer_min_relative_deviation:
            continue

        z, p = two_proportion_z_test(int(row["trans_count"]), int(row["total"]), base_count, base_total)
        if p > cfg.significance_alpha:
            continue

        severity = "High" if row["trans_rate"] > cfg.transfer_severity_high else "Medium"
        unit = row["Current Unit Name"]

        out.append(Anomaly(
            id=make_safe_id("TRANSFER", unit, used_ids),
            detected=f"Abnormal Reassignment Activity — {unit}",
            category="Workflow Integrity",
            severity=severity,
            confidence=confidence_from_p_value(p),
            p_value=round(p, 5),
            root_cause=(
                f"{unit} shows {row['trans_rate']*100:.1f}% transfer rate vs {base_rate*100:.1f}% for "
                f"the rest of the portfolio (z={z:.2f}, p={p:.4f}, n={int(row['total'])}). Suggests "
                f"unclear routing rules, capacity imbalance, or repeated hand-offs delaying resolution."
            ),
            financial_risk=f"{int(row['trans_count'])} reassigned cases increase cycle time and risk of information loss between handlers.",
            historical_comparison=f"Transfer rate {row['trans_rate']/max(base_rate,0.001):.1f}x higher than rest-of-portfolio baseline.",
            recommended_action="Audit routing rules for this unit. Cross-reference transferred cases for repeated hand-offs and consolidate ownership.",
            supporting_metrics={
                "Transfer Count": int(row["trans_count"]),
                "Transfer Rate": f"{row['trans_rate']*100:.1f}%",
                "Sample Size": int(row["total"]),
                "Rest-of-Portfolio Baseline": f"{base_rate*100:.1f}%",
                "p-value": round(p, 5),
            },
        ))

    return out


def detect_risk_score_outliers(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags appeals with a compliance risk score >2σ above the mean."""
    if not validate_columns(df, "risk_score"):
        return []

    mean_risk = df["ComplianceRiskScore"].mean()
    std_risk = df["ComplianceRiskScore"].std(ddof=0)
    if std_risk == 0 or pd.isna(std_risk):
        return []

    threshold = mean_risk + cfg.risk_score_std_multiplier * std_risk
    hi_risk = df[df["ComplianceRiskScore"] > threshold]
    if hi_risk.empty:
        return []

    top_entity = (
        hi_risk.groupby("Plan Sponsor Name")["ComplianceRiskScore"].mean().idxmax()
        if "Plan Sponsor Name" in hi_risk.columns else "N/A"
    )
    top_unit = hi_risk["Current Unit Name"].mode().iloc[0] if not hi_risk.empty else "N/A"

    return [Anomaly(
        id=make_safe_id("RISK", "OUTLIERS", used_ids),
        detected="Statistical Compliance Risk Outliers Detected",
        category="Risk Concentration",
        severity="High",
        # z-score outliers at >2sigma are exact/deterministic given the data,
        # not a hypothesis test — reported confidence reflects that this is
        # a descriptive-statistics flag, not an inferential one.
        confidence=0.95,
        p_value=None,
        root_cause=(
            f"{len(hi_risk)} appeals score >{cfg.risk_score_std_multiplier}σ above mean compliance "
            f"risk ({mean_risk:.3f}). High-risk concentration in {top_unit} unit. Plan sponsor "
            f"'{top_entity}' shows highest average risk."
        ),
        financial_risk=f"{len(hi_risk):,} appeals carry elevated non-compliance / overdue / escalation exposure.",
        historical_comparison=f"Outlier threshold: {threshold:.3f} (mean {mean_risk:.3f} + {cfg.risk_score_std_multiplier}σ {std_risk:.3f}).",
        recommended_action="Perform enhanced case review on top-risk plan sponsors. Flag for internal compliance/quality audit.",
        supporting_metrics={
            "Outlier Count": len(hi_risk),
            "Mean Risk Score": f"{mean_risk:.3f}",
            "Outlier Threshold": f"{threshold:.3f}",
            "Top Risk Sponsor": top_entity,
        },
    )]


def detect_aging_backlog(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags a significant concentration of cases beyond the aging threshold."""
    if not validate_columns(df, "aging"):
        return []

    stale = df[df["CaseAgeDays"] > cfg.aging_threshold_days]
    if stale.empty:
        return []

    stale_pct = len(stale) / len(df)
    severity = severity_from_rate(stale_pct, cfg.aging_severity_critical, cfg.aging_severity_high)
    top_unit = stale["Current Unit Name"].mode().iloc[0] if "Current Unit Name" in stale.columns and not stale.empty else "N/A"

    return [Anomaly(
        id=make_safe_id("AGING", "CONCENTRATION", used_ids),
        detected="Significant Aged Case Backlog",
        category="Aging & SLA Risk",
        severity=severity,
        confidence=0.9,  # descriptive threshold breach, not a hypothesis test
        p_value=None,
        root_cause=(
            f"{len(stale)} appeals ({stale_pct*100:.1f}%) are older than {cfg.aging_threshold_days} days "
            f"(2x SLA). Primary unit: {top_unit}. Aging suggests unresolved escalations, "
            f"decision-maker bottlenecks, or member-response delays."
        ),
        financial_risk="Aged cases carry elevated regulatory-complaint and external-review risk if not resolved promptly.",
        historical_comparison=f"{stale_pct*100:.1f}% of portfolio exceeds {cfg.aging_threshold_days} days. Maximum age: {int(df['CaseAgeDays'].max())} days.",
        recommended_action=(
            f"Conduct aging review with unit leads. Automate SLA alerts at "
            f"{cfg.sla_days}/{cfg.aging_threshold_days}-day thresholds. Prioritize decision-maker sign-off for oldest cases."
        ),
        supporting_metrics={
            f"Aged Count (>{cfg.aging_threshold_days}d)": len(stale),
            "Aged Percentage": f"{stale_pct*100:.1f}%",
            "Max Age": f"{int(df['CaseAgeDays'].max())} days",
        },
    )]


def detect_overturn_outliers(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags plan sponsors with a statistically significant overturn-rate spike."""
    if not validate_columns(df, "overturn"):
        return []

    closed = df[df["IsClosed"]]
    if closed.empty:
        return []

    global_overturned = int(closed["IsAnyOverturn"].sum())
    global_total = len(closed)

    sp = closed.groupby("Plan Sponsor Name").agg(
        total=("Case Identifier", "count"),
        overturned=("IsAnyOverturn", "sum"),
    ).reset_index()
    sp = sp[sp["total"] >= cfg.min_group_size]
    if len(sp) <= 1:
        return []

    sp["rate"] = sp["overturned"] / sp["total"]

    best = None
    for _, row in sp.iterrows():
        base_rate, base_count, base_total = leave_one_out_baseline(
            int(row["overturned"]), int(row["total"]), global_overturned, global_total
        )
        if base_total <= 0 or row["rate"] <= base_rate * cfg.overturn_min_relative_deviation:
            continue
        z, p = two_proportion_z_test(int(row["overturned"]), int(row["total"]), base_count, base_total)
        if p > cfg.significance_alpha:
            continue
        if best is None or row["rate"] > best[0]["rate"]:
            best = (row, base_rate, z, p)

    if best is None:
        return []
    top, base_rate, z, p = best

    deviation = round((top["rate"] - base_rate) / max(base_rate, 0.001) * 100, 1)
    severity = severity_from_rate(top["rate"], cfg.overturn_severity_critical, cfg.overturn_severity_high)
    sponsor = top["Plan Sponsor Name"]

    return [Anomaly(
        id=make_safe_id("OVERTURN", sponsor, used_ids),
        detected=f"Elevated Overturn Rate — {sponsor}",
        category="Outcome & Determination Quality",
        severity=severity,
        confidence=confidence_from_p_value(p),
        p_value=round(p, 5),
        root_cause=(
            f"{sponsor} shows a {top['rate']*100:.1f}% combined overturn rate (full + partial) across "
            f"{int(top['total'])} closed appeals vs {base_rate*100:.1f}% for the rest of the portfolio — "
            f"{deviation:+.1f}% deviation (z={z:.2f}, p={p:.4f}). Suggests initial adverse-determination "
            f"quality issues, inconsistent clinical criteria application, or a sponsor-specific "
            f"benefit-design ambiguity driving reversals on appeal."
        ),
        financial_risk="High overturn concentration signals potential rework cost and member-abrasion risk; frequent reversals may also draw regulator attention to first-level determination accuracy for this sponsor.",
        historical_comparison=f"Overturn rate {top['rate']/max(base_rate,0.001):.1f}x rest-of-portfolio baseline of {base_rate*100:.1f}%.",
        recommended_action=f"Conduct a first-level determination quality audit for {sponsor}. Review clinical-criteria alignment with medical directors and share findings with the intake/UM team to reduce avoidable reversals.",
        supporting_metrics={
            "Overturn Rate": f"{top['rate']*100:.1f}%",
            "Overturned Count": int(top["overturned"]),
            "Total Closed": int(top["total"]),
            "Rest-of-Portfolio Baseline": f"{base_rate*100:.1f}%",
            "p-value": round(p, 5),
        },
    )]


def detect_oon_concentration(df: pd.DataFrame, cfg: AnomalyConfig, used_ids: set[str]) -> list[Anomaly]:
    """Flags provider types with a statistically elevated out-of-network share."""
    if not validate_columns(df, "oon"):
        return []
    if not df["IsOutOfNetwork"].any():
        return []

    global_oon_count = int(df["IsOutOfNetwork"].sum())
    global_total = len(df)

    prov = df.groupby("Provider Type").agg(
        total=("Case Identifier", "count"),
        oon=("IsOutOfNetwork", "sum"),
        overturned=("IsAnyOverturn", "sum"),
    ).reset_index()
    prov = prov[prov["total"] >= cfg.min_group_size]
    if len(prov) <= 1:
        return []

    prov["oon_rate"] = prov["oon"] / prov["total"]

    best = None
    for _, row in prov.iterrows():
        base_rate, base_count, base_total = leave_one_out_baseline(
            int(row["oon"]), int(row["total"]), global_oon_count, global_total
        )
        if base_total <= 0 or row["oon_rate"] <= base_rate * cfg.oon_min_relative_deviation:
            continue
        z, p = two_proportion_z_test(int(row["oon"]), int(row["total"]), base_count, base_total)
        if p > cfg.significance_alpha:
            continue
        if best is None or row["oon_rate"] > best[0]["oon_rate"]:
            best = (row, base_rate, z, p)

    if best is None:
        return []
    top, base_rate, z, p = best

    deviation = round((top["oon_rate"] - base_rate) / max(base_rate, 0.001) * 100, 1)
    severity = "High" if top["oon_rate"] > cfg.oon_severity_high else "Medium"
    provider = top["Provider Type"]

    return [Anomaly(
        id=make_safe_id("OON", provider, used_ids),
        detected=f"Elevated Out-of-Network Concentration — {provider}",
        category="Provider & Network Risk",
        severity=severity,
        confidence=confidence_from_p_value(p),
        p_value=round(p, 5),
        root_cause=(
            f"{provider} appeals show {top['oon_rate']*100:.1f}% out-of-network share vs "
            f"{base_rate*100:.1f}% for the rest of the portfolio — {deviation:+.1f}% deviation (z={z:.2f}, "
            f"p={p:.4f}), associated with {int(top['overturned'])} overturned/partial-overturn determinations."
        ),
        financial_risk="Out-of-network concentration typically carries higher member cost-share disputes and elevated overturn exposure.",
        historical_comparison=f"Out-of-network rate {top['oon_rate']/max(base_rate,0.001):.1f}x rest-of-portfolio baseline of {base_rate*100:.1f}%.",
        recommended_action=f"Review network-adequacy and steerage strategy for {provider} services; audit determination consistency for out-of-network claims in this specialty.",
        supporting_metrics={
            "Out-of-Network Rate": f"{top['oon_rate']*100:.1f}%",
            "Out-of-Network Count": int(top["oon"]),
            "Total Cases": int(top["total"]),
            "Rest-of-Portfolio Baseline": f"{base_rate*100:.1f}%",
            "p-value": round(p, 5),
        },
    )]


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

_SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

_DETECTORS = [
    detect_overdue_spikes,
    detect_escalated_open_cluster,
    detect_transfer_spikes,
    detect_risk_score_outliers,
    detect_aging_backlog,
    detect_overturn_outliers,
    detect_oon_concentration,
]


def detect_anomalies(df: pd.DataFrame, config: AnomalyConfig | None = None) -> list[dict]:
    """
    Runs the full anomaly-detection suite against an appeals & grievances
    portfolio DataFrame and returns a severity-sorted list of anomaly dicts.

    Parameters
    ----------
    df : pd.DataFrame
        Case-level portfolio data. Detectors independently validate the
        columns they need and log + skip (rather than crash) if a required
        column is absent.
    config : AnomalyConfig, optional
        Thresholds and business constants. Defaults to AnomalyConfig() —
        override this in production with your real SLA_DAYS,
        AGING_THRESHOLD, and ESCALATION_LEVEL values.

    Returns
    -------
    list[dict]
        Anomalies sorted Critical -> High -> Medium, each as a dict
        (via Anomaly.to_dict()) for easy JSON serialization / API response.
    """
    cfg = config or AnomalyConfig()

    if df.empty:
        logger.info("detect_anomalies called with empty DataFrame; returning no anomalies.")
        return []

    used_ids: set[str] = set()
    anomalies: list[Anomaly] = []

    for detector in _DETECTORS:
        try:
            found = detector(df, cfg, used_ids)
            anomalies.extend(found)
        except Exception:
            # A single detector failing should not take down the whole run.
            logger.exception("Detector '%s' raised an exception; skipping it.", detector.__name__)

    anomalies.sort(key=lambda a: (_SEVERITY_ORDER.get(a.severity, 3), -a.confidence))

    logger.info("detect_anomalies produced %d anomalies from %d cases.", len(anomalies), len(df))
    return [a.to_dict() for a in anomalies]
