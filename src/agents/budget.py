from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import budget_llm
from ..state import TravelState


def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Assess: estimated cost categories, budget risk areas, money-saving
suggestions, and overall feasibility. If exact live prices are unavailable,
clearly label estimates as approximate.
"""

    # Fail open: if the structured call errors, treat the trip as low-risk
    # rather than blocking the graph, and simply skip replanning this round.
    try:
        assessment = budget_llm.invoke(
            [
                SystemMessage(content="You are a practical travel budget analyst."),
                HumanMessage(content=prompt),
            ]
        )
        risk_level = assessment.risk_level
        feasible = assessment.feasible
        budget_text = (
            f"Cost Summary: {assessment.cost_summary}\n"
            f"Risk Level: {risk_level}\n"
            f"Risk Areas: {', '.join(assessment.risk_areas) or 'None noted'}\n"
            f"Suggestions: {', '.join(assessment.suggestions) or 'None'}"
        )
        llm_calls_delta = 1
    except Exception as exc:
        print(f"Budget agent fallback used: {exc}")
        risk_level = "low"
        feasible = True
        budget_text = f"Budget analysis unavailable: {exc}"
        llm_calls_delta = 0

    return {
        "budget_results": budget_text,
        "budget_risk_level": risk_level,
        "budget_feasible": feasible,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": llm_calls_delta,
    }