from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import guardrail_llm, supervisor_llm
from ..state import AGENT_ORDER, KNOWN_AGENTS, TravelState, empty_constraints


def supervisor_agent(state: TravelState):
    query = state["user_query"]
    # With the llm_calls reducer, every node returns only the calls IT made,
    # not a running total - so this starts at 0 regardless of prior state.
    llm_calls = 0

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

User request:
{query}
"""

    # Fail open on model/schema errors so a temporary provider issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_result = guardrail_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the input guardrail for a travel-planning "
                        "application."
                    )
                ),
                HumanMessage(content=guardrail_prompt),
            ]
        )
        allowed = guardrail_result.allowed
        guardrail_reason = guardrail_result.reason.strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

User request:
{query}
"""

    try:
        plan = supervisor_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You route work to travel specialist agents."
                    )
                ),
                HumanMessage(content=supervisor_prompt),
            ]
        )
        selected_agents = [
            name
            for name in AGENT_ORDER
            if name in plan.selected_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = plan.trip_constraints.model_dump()
        reasoning = plan.reasoning.strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }