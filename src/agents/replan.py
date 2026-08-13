from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import replan_llm
from ..state import TravelState, empty_constraints


def replan_agent(state: TravelState):
    prompt = f"""
The current trip plan may not be feasible for the user's budget.

User Query:
{state['user_query']}

Current Trip Constraints:
{state.get('trip_constraints', {})}

Budget Assessment:
{state.get('budget_results', '')}

Adjust the trip constraints to make the trip more affordable - for example a
shorter duration, a more budget-friendly travel style, or fewer special
requests. Keep the destination and origin unchanged unless clearly necessary.
"""

    # Fail open: if replanning fails, keep the original constraints and let
    # the pipeline continue rather than getting stuck.
    try:
        result = replan_llm.invoke(
            [
                SystemMessage(
                    content="You adjust trip plans to fit a traveler's budget."
                ),
                HumanMessage(content=prompt),
            ]
        )
        constraints = result.adjusted_constraints.model_dump()
        note = result.explanation.strip() or "Constraints adjusted for budget."
        llm_calls_delta = 1
    except Exception as exc:
        print(f"Replan agent fallback used: {exc}")
        constraints = state.get("trip_constraints", empty_constraints())
        note = f"Replanning skipped: {exc}"
        llm_calls_delta = 0

    reasoning = state.get("supervisor_reasoning", "")
    reasoning = f"{reasoning}\n\nReplanned for budget: {note}".strip()

    return {
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        # Marks that a replan already happened, so route_after_budget won't
        # loop back here again after the re-check.
        "replanned": True,
        "messages": [AIMessage(content=f"Trip constraints adjusted for budget: {note}")],
        "llm_calls": llm_calls_delta,
    }