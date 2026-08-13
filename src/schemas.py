"""
Structured output schemas.

These replace the old "find the first { and last }" text scraping
(_json_from_llm) with schema-validated, typed responses. Groq handles
this via native JSON-mode/tool-calling under the hood, so malformed
JSON, stray preambles, or markdown fences from the model no longer
break parsing.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Result of checking whether a request belongs to travel planning."""

    allowed: bool = Field(
        description="True if the request is a valid travel-planning request."
    )
    reason: str = Field(
        default="",
        description=(
            "Short explanation, especially why the request was blocked "
            "when allowed is false."
        ),
    )


class TripConstraints(BaseModel):
    """Structured trip details extracted from the user's request."""

    destination: str = ""
    origin: str = ""
    duration: str = ""
    budget: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(default_factory=list)


class SupervisorPlan(BaseModel):
    """The supervisor's routing decision and extracted constraints."""

    selected_agents: list[str] = Field(default_factory=list)
    trip_constraints: TripConstraints = Field(default_factory=TripConstraints)
    reasoning: str = ""


class BudgetAssessment(BaseModel):
    """Structured feasibility check produced before the itinerary is drafted."""

    risk_level: Literal["low", "medium", "high"] = "low"
    feasible: bool = True
    cost_summary: str = ""
    risk_areas: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ReplanAdjustment(BaseModel):
    """Adjusted trip constraints produced when the budget risk is high."""

    adjusted_constraints: TripConstraints
    explanation: str = ""


class BudgetReview(BaseModel):
    """Feasibility check run against the drafted itinerary itself."""

    feasible: bool = True
    issues: list[str] = Field(default_factory=list)
    notes: str = ""