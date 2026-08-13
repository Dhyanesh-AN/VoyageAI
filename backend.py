"""
Backwards-compatible entry point.

Keeps `from backend import run_travel_agent, resume_travel_agent, travel_graph`
working for anything (FastAPI routes, scripts, tests) that imported the old
monolithic backend.py, while the real implementation now lives in `app/`.
"""

from src.graph import travel_graph
from src.service import resume_travel_agent, run_travel_agent

__all__ = ["run_travel_agent", "resume_travel_agent", "travel_graph"]