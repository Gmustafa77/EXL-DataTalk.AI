"""Curated example questions shown in the AI Assistant.

Every question uses fields and certified metrics that exist in the approved appeals
table (units, sites, plan sponsors, providers, specialties, network status,
determinations, compliance, turnaround, received/closed dates). Selecting one sends
it through the normal AI Assistant workflow, exactly like a typed question.

No Streamlit imports — edit this catalogue without touching UI code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleCategory:
    name: str
    icon: str
    description: str
    questions: tuple[str, ...]


EXAMPLE_CATEGORIES: tuple[ExampleCategory, ...] = (
    ExampleCategory(
        "Performance", "🏆", "Who performs best and who needs attention.",
        (
            "Which operational unit has the best TAT performance?",
            "Which plan sponsor has high appeal volume but also a high overturn rate?",
        ),
    ),
    ExampleCategory(
        "Drill-down", "🔎", "Multi-part questions answered step by step.",
        (
            "Which sponsor has the highest number of appeals and how are those appeals being determined?",
            "Which provider has the highest appeal volume and what is its overturn rate?",
            "Show me the top 10 providers by appeal count.",
            
        ),
    ),
    ExampleCategory(
        "Trends", "📈", "How volumes move over time.",
        (
            "How many appeals were received each month?",
            "What is the trend in overturn rates over the last 12 months?",
            
        ),
    ),
)

__all__ = ["ExampleCategory", "EXAMPLE_CATEGORIES"]
