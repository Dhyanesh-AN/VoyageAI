"""
Graph assembly.

Builds the LangGraph StateGraph out of the node functions in `app.agents`
and the routing functions in `app.routing`, then wires up the PostgreSQL
checkpointer and compiles it into `travel_graph`.
"""

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row

from . import config
from .agents import (
    budget_agent,
    budget_gate,
    budget_review_agent,
    final_agent,
    flight_agent,
    guardrail_blocked_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    replan_agent,
    supervisor_agent,
    weather_agent,
)
from .routing import (
    BUDGET_GATE_ROUTE_MAP,
    NEGOTIATION_ROUTE_MAP,
    REPLAN_ROUTE_MAP,
    SUPERVISOR_ROUTE_MAP,
    route_after_budget,
    route_after_budget_review,
    route_from_budget_gate,
    route_from_supervisor,
)
from .state import TravelState

# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_gate", budget_gate)
graph.add_node("budget_agent", budget_agent)
graph.add_node("replan_agent", replan_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("budget_review_agent", budget_review_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")

# Fan-out: supervisor dispatches to flight_agent/hotel_agent/weather_agent in
# parallel via Send. Each of them then edges into the shared join point.
graph.add_conditional_edges("supervisor", route_from_supervisor, SUPERVISOR_ROUTE_MAP)
graph.add_edge("flight_agent", "budget_gate")
graph.add_edge("hotel_agent", "budget_gate")
graph.add_edge("weather_agent", "budget_gate")

# Fan-in: budget_gate waits for whichever parallel agents ran, then routes to
# budget_agent (if selected) or straight to itinerary_agent.
graph.add_conditional_edges("budget_gate", route_from_budget_gate, BUDGET_GATE_ROUTE_MAP)

# Dynamic replanning loop: high budget risk sends the trip back through
# replan_agent, which adjusts constraints and re-runs budget_agent once.
graph.add_conditional_edges("budget_agent", route_after_budget, REPLAN_ROUTE_MAP)
graph.add_edge("replan_agent", "budget_agent")

# Agent-to-agent negotiation loop: itinerary_agent and budget_review_agent go
# back and forth (capped at MAX_NEGOTIATION_ROUNDS) before human approval.
graph.add_edge("itinerary_agent", "budget_review_agent")
graph.add_conditional_edges(
    "budget_review_agent", route_after_budget_review, NEGOTIATION_ROUTE_MAP
)

graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = config.get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)