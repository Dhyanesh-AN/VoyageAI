from ..state import TravelState


def budget_gate(state: TravelState):
    # Pure pass-through node. Its only purpose is to give flight_agent,
    # hotel_agent, and weather_agent a common downstream node to join on,
    # so the graph waits for whichever of them ran before continuing.
    return {}