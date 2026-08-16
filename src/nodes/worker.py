"""Action node: calls Claude with structured output, catches errors, increments retry_count.

Failure handling: any exception (validation, network, rate limit) is caught;
retry_count++ ; the error message is fed back to the LLM on the next attempt
via `last_error` and an appended HumanMessage.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import settings
from src.state import AgentState


class ExtractionResult(BaseModel):
    """Generic extraction shape for a free-text input."""

    summary: str = Field(description="One-sentence summary of the input")
    entities: list[str] = Field(description="Named entities detected")
    sentiment: str = Field(description="positive | neutral | negative")
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = (
    "You are a structured extraction agent. Analyse the input and return JSON "
    "STRICTLY conforming to the schema provided. No text outside the JSON."
)


def _build_llm():
    """Factory isolated to make mocking trivial in tests."""
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
        thinking={"type": "adaptive"},
    ).with_structured_output(ExtractionResult)


def worker_node(state: AgentState) -> dict:
    """Run one extraction attempt. Router decides what to do next."""
    llm = _build_llm()

    msgs = [SystemMessage(SYSTEM_PROMPT)]
    if state.get("last_error"):
        msgs.append(HumanMessage(
            f"Previous attempt invalid: {state['last_error']}. "
            f"Correct it and try again."
        ))
    msgs.append(HumanMessage(state["input"]))

    try:
        result = llm.invoke(msgs)
        return {
            "extracted_data": result.model_dump(),
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "extracted_data": {},
            "messages": [HumanMessage(f"Extraction failed: {e}")],
        }
