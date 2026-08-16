"""Test that the worker increments retry_count on failure,
and that the router forces HITL at max_retries."""
from unittest.mock import MagicMock

from src.config import settings
from src.nodes import worker as worker_module
from src.nodes.router import route
from src.state import initial_state


class FakeFailingLLM:
    def invoke(self, messages):
        raise ValueError("simulated structured output validation error")


class FakeSucceedingLLM:
    def invoke(self, messages):
        result = MagicMock()
        result.model_dump.return_value = {
            "summary": "ok",
            "entities": ["foo"],
            "sentiment": "neutral",
            "confidence": 0.9,
        }
        return result


def test_worker_increments_retry_on_failure(monkeypatch):
    monkeypatch.setattr(worker_module, "_build_llm", lambda: FakeFailingLLM())
    state = initial_state("anything")

    update = worker_module.worker_node(state)
    assert update["retry_count"] == 1
    assert "simulated" in update["last_error"]
    assert update["extracted_data"] == {}


def test_three_failures_trigger_circuit_breaker(monkeypatch):
    monkeypatch.setattr(worker_module, "_build_llm", lambda: FakeFailingLLM())
    state = initial_state("anything")

    for _ in range(settings.max_retries):
        update = worker_module.worker_node(state)
        state.update(update)

    assert state["retry_count"] == settings.max_retries
    assert route(state) == "hitl"


def test_worker_success_populates_extracted_data(monkeypatch):
    monkeypatch.setattr(worker_module, "_build_llm", lambda: FakeSucceedingLLM())
    state = initial_state("Marie bought 3 apples")

    update = worker_module.worker_node(state)
    assert update["extracted_data"]["summary"] == "ok"
    assert update["last_error"] == ""
    assert "retry_count" not in update
