"""
Dynamic supervisor / graph routing.

Conditional-edge functions that decide where the graph goes next, plus the
route maps LangGraph needs alongside them.
"""

from langgraph.types import Send

from .state import MAX_NEGOTIATION_ROUNDS, TravelState, selected_agents

SUPERVISOR_ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "budget_gate": "budget_gate",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
}


def route_from_supervisor(state: TravelState):
    """Fan out to flight/hotel/weather agents in parallel via Send.

    flight_agent, hotel_agent, and weather_agent don't depend on each
    other's output, so whichever of them the supervisor selected are
    dispatched together instead of chained sequentially. budget_agent still
    runs afterward at budget_gate, since it depends on their results.
    """
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = selected_agents(state)
    parallel_targets = [
        agent
        for agent in ("flight_agent", "hotel_agent", "weather_agent")
        if agent in selected
    ]

    if not parallel_targets:
        # Nothing to parallelize (e.g. only budget_agent/itinerary_agent were
        # selected) - skip straight to the join point.
        return "budget_gate"

    return [Send(agent, state) for agent in parallel_targets]


BUDGET_GATE_ROUTE_MAP = {
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def route_from_budget_gate(state: TravelState) -> str:
    selected = selected_agents(state)
    return "budget_agent" if "budget_agent" in selected else "itinerary_agent"


REPLAN_ROUTE_MAP = {
    "replan_agent": "replan_agent",
    "itinerary_agent": "itinerary_agent",
}


def route_after_budget(state: TravelState) -> str:
    """Dynamic replanning: loop back through budget_agent once if risk is high."""
    if state.get("budget_risk_level") == "high" and not state.get("replanned", False):
        return "replan_agent"
    return "itinerary_agent"


NEGOTIATION_ROUTE_MAP = {
    "human_approval": "human_approval",
    "itinerary_agent": "itinerary_agent",
}


def route_after_budget_review(state: TravelState) -> str:
    """Agent-to-agent negotiation: itinerary_agent <-> budget_review_agent.

    Loops back to itinerary_agent for a revision when the draft isn't
    feasible, capped at MAX_NEGOTIATION_ROUNDS so it can't run forever.
    """
    feasible = state.get("budget_feasible", True)
    rounds = state.get("negotiation_rounds", 0)

    if feasible or rounds >= MAX_NEGOTIATION_ROUNDS:
        return "human_approval"
    return "itinerary_agent"