"""Build-time guards on the graph topology.

LangGraph only discovers a missing route key at *invoke* time, and it surfaces
as an opaque error deep inside a run. These tests turn that into a build-time
failure: cheap to run, and it saves a confusing debugging session mid-run.
"""
from langgraph.graph import END

from src.config import settings
from src.graph import _ADVANCING, _ROUTES, build_graph
from src.nodes.router import _NEXT_STAGE, route


def _compiled_nodes() -> set[str]:
    return set(build_graph().compile().get_graph().nodes)


# --- compiles at all -------------------------------------------------------


def test_graph_compiles():
    build_graph().compile()


def test_all_day_zero_nodes_registered():
    nodes = _compiled_nodes()
    expected = {
        "ingest",
        "extract_menu",
        "decompose_recipes",
        "canonicalize",
        "cost_plates",
        "review_gate",
        "review_notify",
        "review_wait",
        "forecast",
        "draft_po",
        "po_notify",
        "po_wait",
    }
    assert expected <= nodes, f"missing nodes: {sorted(expected - nodes)}"


def test_generic_retry_path_survives():
    """worker/hitl_notify/hitl are what the circuit breaker falls back to."""
    assert {"worker", "hitl_notify", "hitl"} <= _compiled_nodes()


def test_start_edge_is_ingest():
    edges = build_graph().compile().get_graph().edges
    starts = {e.target for e in edges if e.source == "__start__"}
    assert starts == {"ingest"}


def test_notify_nodes_pair_to_their_wait_nodes():
    """Each notify hands off unconditionally to exactly one wait node."""
    edges = build_graph().compile().get_graph().edges
    plain = {(e.source, e.target) for e in edges if not e.conditional}
    assert ("hitl_notify", "hitl") in plain
    assert ("review_notify", "review_wait") in plain
    assert ("po_notify", "po_wait") in plain


# --- route() and _ROUTES agree --------------------------------------------


def test_every_next_stage_value_is_a_route_key():
    missing = sorted(set(_NEXT_STAGE.values()) - set(_ROUTES))
    assert not missing, f"_NEXT_STAGE values absent from _ROUTES: {missing}"


def test_every_route_return_value_is_a_route_key():
    """Drive route() across every stage and every control-flow branch."""
    stages = list(_NEXT_STAGE) + ["nonsense_stage", "reviewed", ""]
    states = []
    for stage in stages:
        states.append({"stage": stage})
        states.append({"stage": stage, "needs_human": True})
        states.append({"stage": stage, "needs_human": True, "review_queue": [{"ref": "sku"}]})
        states.append({"stage": stage, "retry_count": settings.max_retries})
        states.append({"stage": stage, "human_decision": "reject"})
        states.append({"stage": stage, "human_decision": "approve"})
        states.append({"stage": stage, "human_decision": "skip"})
    states.append({})

    for state in states:
        target = route(state)
        assert target in _ROUTES, f"route() returned {target!r} for {state!r}, not in _ROUTES"


def test_route_targets_are_real_nodes():
    nodes = _compiled_nodes()
    for value, target in _ROUTES.items():
        if target is END:
            continue
        assert target in nodes, f"_ROUTES[{value!r}] -> {target!r} is not a registered node"


def test_advancing_nodes_are_registered():
    nodes = _compiled_nodes()
    assert set(_ADVANCING) <= nodes


# --- the review dispatch, the bug this step exists to prevent --------------


def test_review_queue_routes_to_the_review_card_not_the_breaker_card():
    """A pending review queue must reach review_notify, never hitl_notify.

    Both are entered via needs_human=True. Sending the review queue to the
    generic hitl_notify posts the circuit-breaker card instead of the review
    card — a wiring bug invisible until the wrong card shows up in Slack.
    """
    state = {"needs_human": True, "review_queue": [{"ref": "sku"}], "stage": "costed"}
    assert route(state) == "review"
    assert _ROUTES["review"] == "review_notify"


def test_empty_review_queue_falls_back_to_the_generic_hitl_card():
    for state in (
        {"needs_human": True, "review_queue": [], "stage": "costed"},
        {"needs_human": True, "stage": "costed"},
    ):
        assert route(state) == "hitl"
    assert _ROUTES["hitl"] == "hitl_notify"


def test_circuit_breaker_beats_a_pending_review_queue():
    """The breaker is absolute — it outranks the review card."""
    state = {
        "retry_count": settings.max_retries,
        "needs_human": True,
        "review_queue": [{"ref": "sku"}],
        "stage": "costed",
    }
    assert route(state) == "hitl"
