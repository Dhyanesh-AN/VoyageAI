import asyncio

from langchain_core.messages import AIMessage
from mcp_client import tavily_mcp_search

from ..state import TravelState


def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        # Delta, not a total (see flight_agent). hotel_agent only calls
        # Tavily, not the LLM, so its delta is 0 - this also corrects a
        # pre-existing miscount in the original code.
        "llm_calls": 0,
    }