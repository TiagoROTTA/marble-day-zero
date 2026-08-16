"""Stage dispatch, precedence, and the review/hitl split.

The assertion that earns its keep is `test_circuit_breaker_outranks_every_stage`:
the stage dispatch once sat above the breaker, so a node that populated its
output field *and* failed on a later chunk could route forward past a tripped
breaker. The breaker is meant to be absolute; these tests are what make that
claim checkable.
"""
import pytest

from src.config import settings
from src.nodes.review_gate import MAX_REVIEW_ITEMS, review_gate_node
from src.nodes.router import _NEXT_STAGE, route
from src.state import CONF_AUTO_ACCEPT, CONF_REVIEW_FLOOR

# The pipeline in order, as a stage -> next-node table independent of the
# implementation dict, so a typo in _NEXT_STAGE fails here rather than at invoke time.
EXPECTED = {
    "": "ingest",
    "ingested": "extract_menu",
    "menu_extracted": "decompose_recipes",
    "recipes_decomposed": "canonicalize",
    "canonicalized": "cost_plates",
    "costed": "review_gate",
    "reviewed": "forecast",
    "forecast": "draft_po",
    "po_drafted": "po_approval",
    "po_approved": "end",
}


@pytest.mark.parametrize("stage,expected", sorted(EXPECTED.items()))
def test_each_stage_maps_to_its_next_node(stage, expected):
    assert route({"stage": stage, "retry_count": 0}) == expected


def test_dispatch_table_matches_expected_exactly():
    assert _NEXT_STAGE == EXPECTED


def test_unknown_stage_ends_rather_than_raising():
    assert route({"stage": "not_a_stage", "retry_count": 0}) == "end"


def test_missing_stage_key_behaves_like_empty_stage():
    assert route({"retry_count": 0}) == "ingest"


# --- precedence -----------------------------------------------------------


@pytest.mark.parametrize("stage", sorted(EXPECTED))
def test_circuit_breaker_outranks_every_stage(stage):
    """A tripped breaker wins even when `stage` names a valid forward step."""
    assert route({"stage": stage, "retry_count": settings.max_retries}) == "hitl"


def test_circuit_breaker_outranks_needs_human_false_and_a_full_queue():
    state = {
        "stage": "costed",
        "retry_count": settings.max_retries + 50,
        "needs_human": False,
        "review_queue": [{"kind": "plate_cost", "ref": "x", "confidence": 0.7}],
    }
    assert route(state) == "hitl"


@pytest.mark.parametrize("stage", sorted(EXPECTED))
def test_reject_outranks_everything(stage):
    state = {
        "stage": stage,
        "retry_count": settings.max_retries + 1,
        "needs_human": True,
        "review_queue": [{"kind": "menu_item", "ref": "y", "confidence": 0.7}],
        "human_decision": "reject",
    }
    assert route(state) == "end"


def test_needs_human_outranks_stage_dispatch():
    assert route({"stage": "canonicalized", "needs_human": True, "retry_count": 0}) == "hitl"


# --- review vs hitl split (the graph's _ROUTES depends on this) -----------


def test_needs_human_with_queue_routes_to_review():
    state = {
        "stage": "costed",
        "retry_count": 0,
        "needs_human": True,
        "review_queue": [{"kind": "plate_cost", "ref": "Margherita", "confidence": 0.7}],
    }
    assert route(state) == "review"


def test_needs_human_with_empty_queue_routes_to_generic_hitl():
    state = {"stage": "costed", "retry_count": 0, "needs_human": True, "review_queue": []}
    assert route(state) == "hitl"


def test_queue_without_needs_human_does_not_divert():
    """A leftover queue must not hijack a run nobody flagged."""
    state = {
        "stage": "costed",
        "retry_count": 0,
        "needs_human": False,
        "review_queue": [{"kind": "plate_cost", "ref": "z", "confidence": 0.7}],
    }
    assert route(state) == "review_gate"


# --- purity ---------------------------------------------------------------


def test_router_source_has_no_io_or_exceptions():
    import inspect

    import src.nodes.router as router_module

    source = inspect.getsource(router_module)
    for forbidden in ("try:", "open(", "print(", "raise "):
        assert forbidden not in source, f"router must stay pure: found {forbidden!r}"


def test_router_imports_only_settings_and_state():
    import inspect

    import src.nodes.router as router_module

    imports = [
        line.strip()
        for line in inspect.getsource(router_module).splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert imports == [
        "from src.config import settings",
        "from src.state import AgentState",
    ]


def test_route_never_raises_on_junk_state():
    for junk in ({}, {"stage": None}, {"stage": 12}, {"stage": ""}):
        assert isinstance(route(junk), str)


# --- review_gate_node: the policy the router deliberately does not own -------


def _entry(ref: str, confidence: float) -> dict:
    return {
        "kind": "plate_cost",
        "ref": ref,
        "confidence": confidence,
        "question": f"check {ref}",
        "payload": {},
    }


def _record(update: dict) -> dict:
    return update["extracted_data"]["review_gate"]


def test_gate_keeps_only_the_uncertain_band():
    update = review_gate_node({
        "review_queue": [
            _entry("auto", 0.95),          # >= CONF_AUTO_ACCEPT -> accepted silently
            _entry("edge_accept", CONF_AUTO_ACCEPT),
            _entry("ask", 0.70),
            _entry("edge_ask", CONF_REVIEW_FLOOR),
            _entry("gap", 0.20),
        ],
    })
    assert [e["ref"] for e in update["review_queue"]] == ["edge_ask", "ask"]
    record = _record(update)
    assert record["auto_accepted"] == 2
    assert [g["ref"] for g in record["gaps"]] == ["gap"]
    assert record["below_floor"] == 1
    assert record["dropped"] == 1


def test_gate_flags_for_human_and_leaves_stage_at_costed():
    update = review_gate_node({"stage": "costed", "review_queue": [_entry("a", 0.7)]})
    assert update["needs_human"] is True
    assert update["stage"] == "costed"
    assert route({**update, "retry_count": 0}) == "review"


def test_gate_advances_when_nothing_is_in_the_band():
    update = review_gate_node({
        "stage": "costed",
        "review_queue": [_entry("auto", 0.99), _entry("gap", 0.1)],
    })
    assert update["needs_human"] is False
    assert update["review_queue"] == []
    assert update["stage"] == "reviewed"
    assert _record(update)["dropped"] == 1
    assert route({**update, "retry_count": 0}) == "forecast"


def test_gate_reentry_after_human_closes_the_loop():
    """review_wait_node clears the queue on approve; the gate must then advance, not re-flag."""
    update = review_gate_node({"stage": "costed", "review_queue": []})
    assert update == {"needs_human": False, "stage": "reviewed"}
    assert route({**update, "retry_count": 0}) == "forecast"


def test_gate_caps_the_queue_and_counts_what_it_left_out():
    queue = [_entry(f"i{n:02d}", 0.60 + n * 0.001) for n in range(21)]
    update = review_gate_node({"stage": "costed", "review_queue": queue + [_entry("gap", 0.1)]})

    assert len(update["review_queue"]) == MAX_REVIEW_ITEMS == 12
    # Ascending by confidence: the shakiest items surface first.
    confidences = [e["confidence"] for e in update["review_queue"]]
    assert confidences == sorted(confidences)
    assert [e["ref"] for e in update["review_queue"]][0] == "i00"

    record = _record(update)
    assert record["truncated"] == 9
    assert record["below_floor"] == 1
    assert record["dropped"] == 10


def test_gate_preserves_existing_extracted_data():
    update = review_gate_node({
        "review_queue": [_entry("a", 0.7)],
        "extracted_data": {"summary": "keep me"},
    })
    assert update["extracted_data"]["summary"] == "keep me"


def test_gate_missing_confidence_is_a_gap_not_a_question():
    update = review_gate_node({"review_queue": [{"kind": "x", "ref": "no_conf"}]})
    assert update["review_queue"] == []
    assert _record(update)["below_floor"] == 1


def test_gate_bad_entry_increments_retry_instead_of_raising():
    update = review_gate_node({"retry_count": 1, "review_queue": [{"confidence": "abc"}]})
    assert update["retry_count"] == 2
    assert "ValueError" in update["last_error"]


def test_gate_makes_no_llm_or_network_call():
    import inspect

    import src.nodes.review_gate as gate_module

    source = inspect.getsource(gate_module)
    for forbidden in ("ChatAnthropic", "httpx", "requests", "invoke("):
        assert forbidden not in source, f"gate must stay offline: found {forbidden!r}"
