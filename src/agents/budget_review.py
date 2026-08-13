from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import budget_review_llm
from ..state import TravelState


def budget_review_agent(state: TravelState):
    prompt = f"""
Review this draft itinerary for budget feasibility.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Budget Assessment:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Decide whether the draft itinerary is realistically affordable given the
constraints and budget assessment. If it is not, list concrete issues and
notes the itinerary agent should address in a revision.
"""

    # Fail open: if the review call errors, treat the draft as approved so
    # the pipeline doesn't get stuck waiting on a broken negotiation loop.
    try:
        review = budget_review_llm.invoke(
            [
                SystemMessage(
                    content="You review draft itineraries for budget feasibility."
                ),
                HumanMessage(content=prompt),
            ]
        )
        feasible = review.feasible
        notes = review.notes.strip() or "; ".join(review.issues)
        llm_calls_delta = 1
    except Exception as exc:
        print(f"Budget review fallback used: {exc}")
        feasible = True
        notes = f"Budget review skipped: {exc}"
        llm_calls_delta = 0

    rounds = state.get("negotiation_rounds", 0) + 1
    status = "approved" if feasible else "requested a revision"

    return {
        "budget_feasible": feasible,
        "budget_review_notes": notes,
        "negotiation_rounds": rounds,
        "messages": [
            AIMessage(content=f"Budget review round {rounds}: {status}.")
        ],
        "llm_calls": llm_calls_delta,
    }