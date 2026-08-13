from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import llm
from ..state import TravelState


def itinerary_agent(state: TravelState):
    negotiation_rounds = state.get("negotiation_rounds", 0)
    revision_feedback = ""

    if negotiation_rounds > 0 and state.get("budget_review_notes"):
        revision_feedback = f"""

Budget Reviewer Feedback (this is a revision - address these points):
{state.get('budget_review_notes')}
"""

    prompt = f"""
Create a complete travel itinerary.

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

Budget Results:
{state.get('budget_results', '')}
{revision_feedback}
Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": 1,
    }