"""
Shared graph state, constants, and small helpers used across agents.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage

# Caps the budget <-> itinerary negotiation loop so it can never run forever.
MAX_NEGOTIATION_ROUNDS = 2


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    # Dynamic replanning state
    budget_risk_level: str
    budget_feasible: bool
    replanned: bool

    # Agent-to-agent negotiation state (budget_review_agent <-> itinerary_agent)
    negotiation_rounds: int
    budget_review_notes: str

    # llm_calls uses an additive reducer because flight_agent, hotel_agent, and
    # weather_agent can now run concurrently. Each node must return only the
    # calls IT made (a delta), never a recomputed running total - otherwise
    # concurrent branches would overwrite each other's counts.
    llm_calls: Annotated[int, operator.add]


KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]