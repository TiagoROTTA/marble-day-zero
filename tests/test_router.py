"""Router guarantees for the generic retry/approval path.

The router once defaulted unconditionally to `"worker"` and decided success by
inspecting `extracted_data`; both were replaced by the `_NEXT_STAGE` dispatch,
so the assertions that named those two outcomes now name their stage
equivalents. The reject and circuit-breaker assertions are untouched — those two
rules are non-negotiable.
"""
from src.config import settings
from src.nodes.router import route
from src.state import initial_state


def _state(**overrides):
    s = initial_state("hello")
    s.update(overrides)
    return s


def test_default_goes_to_ingest():
    # A fresh state has no `stage`, which _NEXT_STAGE maps to the pipeline entry.
    assert route(_state()) == "ingest"


def test_circuit_breaker_at_max_retries():
    assert route(_state(retry_count=settings.max_retries)) == "hitl"


def test_below_max_retries_still_advances():
    assert route(_state(retry_count=settings.max_retries - 1)) == "ingest"


def test_terminal_stage_terminates():
    # Was: a truthy `extracted_data` ended the run. The pipeline now ends on the
    # terminal stage instead, so a populated field mid-pipeline cannot short-circuit it.
    assert route(_state(stage="po_approved")) == "end"


def test_extracted_data_no_longer_short_circuits_the_pipeline():
    assert route(_state(stage="ingested", extracted_data={"summary": "ok"})) == "extract_menu"


def test_human_reject_terminates():
    assert route(_state(human_decision="reject", retry_count=2)) == "end"


def test_human_approve_after_hitl_terminates_at_terminal_stage():
    s = _state(
        stage="po_approved",
        human_decision="approve",
        needs_human=False,
    )
    assert route(s) == "end"


def test_needs_human_routes_to_hitl_even_mid_pipeline():
    assert route(_state(stage="ingested", needs_human=True)) == "hitl"
