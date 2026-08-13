"""
LLM client setup.

A single ChatGroq instance is shared across all agents, plus one
`.with_structured_output(...)` variant per schema so each agent gets back a
validated Pydantic instance instead of a raw string to parse.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from . import config
from .schemas import (
    BudgetAssessment,
    BudgetReview,
    GuardrailResult,
    ReplanAdjustment,
    SupervisorPlan,
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=config.GROQ_API_KEY,
)

guardrail_llm = llm.with_structured_output(GuardrailResult)
supervisor_llm = llm.with_structured_output(SupervisorPlan)
budget_llm = llm.with_structured_output(BudgetAssessment)
replan_llm = llm.with_structured_output(ReplanAdjustment)
budget_review_llm = llm.with_structured_output(BudgetReview)


def llm_text(system_prompt: str, user_prompt: str) -> str:
    """Plain-text (non-structured) LLM call helper."""
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)