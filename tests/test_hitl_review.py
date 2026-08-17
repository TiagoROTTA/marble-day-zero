"""Tests for the review notify/wait pair in src/nodes/hitl.py.

Two things can silently break this pair, and both are covered here:

  1. The notify/wait split. If the Slack send lived in the interrupting node it
     would fire again on every resume — one card per button press. The notify
     test asserts send_review_queue is called exactly once.
  2. The meaning of "skip". A skip that behaves like an approve puts unreviewed
     numbers into a total presented as approved, so the tests assert skip
     records `skipped_refs` and never claims human_decision == "approve".

The third test walks the review_gate loop end to end (flag -> approve -> gate
re-entry -> "reviewed"), because a gate that re-flags the same items forever is
the failure mode this node exists to prevent.
"""
from unittest.mock import patch

import pytest

from src.nodes.hitl import review_notify_node, review_wait_node
from src.nodes.review_gate import review_gate_node

THREAD = "thread-abc"
CONFIG = {"configurable": {"thread_id": THREAD}}


def _queue() -> list[dict]:
    return [
        {
            "kind": "sku_match",
            "ref": "PROD-TOMATO-ROMA",
            "confidence": 0.62,
            "question": "Is 'san marzano' the Roma tomato case?",
        },
        {
            "kind": "plate_cost",
            "ref": "Margherita",
            "confidence": 0.71,
            "question": "Margherita plate cost looks low — check?",
        },
    ]


def _state(**overrides) -> dict:
    state = {
        "input": "test-trattoria",
        "restaurant": {"name": "Test Trattoria"},
        "review_queue": _queue(),
        "needs_human": True,
        "stage": "costed",
        "retry_count": 0,
        "extracted_data": {"review_gate": {"dropped": 3}},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# notify — the side effect, exactly once
# ---------------------------------------------------------------------------


def test_notify_sends_exactly_one_card():
    with patch("src.nodes.hitl.get_config", return_value=CONFIG), patch(
        "src.nodes.hitl.send_review_queue"
    ) as send:
        out = review_notify_node(_state())

    assert out == {}, "notify must contribute no state; its job is the side effect"
    assert send.call_count == 1

    thread_id, name, items, dropped = send.call_args.args
    assert thread_id == THREAD
    assert name == "Test Trattoria"
    assert len(items) == 2
    assert dropped == 3, "the dropped count comes from extracted_data['review_gate']"


def test_notify_falls_back_when_restaurant_and_gate_record_are_missing():
    state = _state(restaurant={}, extracted_data={})
    with patch("src.nodes.hitl.get_config", return_value=CONFIG), patch(
        "src.nodes.hitl.send_review_queue"
    ) as send:
        review_notify_node(state)

    _, name, _, dropped = send.call_args.args
    assert name == "test-trattoria", "falls back to the slug rather than an empty title"
    assert dropped == 0


def test_wait_never_sends_slack():
    """The whole point of the split: the re-executed node has no side effect."""
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "approve"}), patch(
        "src.nodes.hitl.send_review_queue"
    ) as send:
        review_wait_node(_state())

    assert send.call_count == 0


def test_wait_interrupt_payload_describes_the_pause():
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "approve"}) as itr:
        review_wait_node(_state())

    assert itr.call_args.args[0] == {"awaiting": "review_queue", "count": 2}


# ---------------------------------------------------------------------------
# wait — the three decisions
# ---------------------------------------------------------------------------


def test_approve_clears_the_queue_and_advances_the_stage():
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "approve"}):
        out = review_wait_node(_state())

    assert out["human_decision"] == "approve"
    assert out["needs_human"] is False
    assert out["review_queue"] == []
    assert out["stage"] == "reviewed"
    assert out["retry_count"] == 0
    assert "skipped_refs" not in out, "approve accepts the items, it does not drop them"


def test_skip_records_refs_and_is_not_an_approve():
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "skip"}):
        out = review_wait_node(_state())

    assert out["human_decision"] == "skip"
    assert out["human_decision"] != "approve"
    assert out["skipped_refs"] == ["PROD-TOMATO-ROMA", "Margherita"]
    assert out["review_queue"] == []
    assert out["stage"] == "reviewed"
    assert out["needs_human"] is False
    assert out["retry_count"] == 0


def test_skip_ignores_entries_without_a_ref():
    state = _state(review_queue=[{"kind": "sku_match", "confidence": 0.6}, *_queue()])
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "skip"}):
        out = review_wait_node(state)

    assert out["skipped_refs"] == ["PROD-TOMATO-ROMA", "Margherita"]


def test_skip_on_an_empty_queue_yields_no_refs():
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "skip"}):
        out = review_wait_node(_state(review_queue=[]))

    assert out["skipped_refs"] == []
    assert out["stage"] == "reviewed"


def test_reject_is_terminal_and_does_not_advance_the_stage():
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "reject"}):
        out = review_wait_node(_state())

    assert out["human_decision"] == "reject"
    assert out["needs_human"] is False
    assert out["retry_count"] == 0
    assert out.get("stage") != "reviewed", "a rejected run was never reviewed"
    assert "skipped_refs" not in out


@pytest.mark.parametrize("payload", [{}, {"decision": "wat"}, {"decision": ""}])
def test_unrecognised_decision_degrades_to_reject(payload):
    with patch("src.nodes.hitl.interrupt", return_value=payload):
        out = review_wait_node(_state())

    assert out["human_decision"] == "reject"


# ---------------------------------------------------------------------------
# the loop actually closes
# ---------------------------------------------------------------------------


def test_gate_flag_then_approve_then_gate_advances_to_reviewed():
    """flag -> pause -> approve -> re-enter gate -> "reviewed". No re-flagging."""
    state = _state(review_queue=_queue(), needs_human=False, stage="canonicalized")

    first = review_gate_node(state)
    assert first["needs_human"] is True
    assert first["stage"] == "costed", "gate holds the stage so the resume re-enters it"
    assert len(first["review_queue"]) == 2
    state.update(first)

    with patch("src.nodes.hitl.interrupt", return_value={"decision": "approve"}):
        state.update(review_wait_node(state))

    assert state["review_queue"] == []

    second = review_gate_node(state)
    assert second["needs_human"] is False
    assert second["stage"] == "reviewed", "the queue is empty, so the gate lets it through"

    state.update(second)
    third = review_gate_node(state)
    assert third["stage"] == "reviewed", "idempotent: nothing gets re-flagged"


def test_router_sends_an_approved_run_forward_not_back_to_review():
    from src.nodes.router import route

    state = _state()
    assert route(state) == "review", "flagged: the review card"

    with patch("src.nodes.hitl.interrupt", return_value={"decision": "approve"}):
        state.update(review_wait_node(state))

    assert route(state) == "forecast", "approved: the run moves on, it does not loop"


def test_router_ends_a_rejected_run():
    from src.nodes.router import route

    state = _state()
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "reject"}):
        state.update(review_wait_node(state))

    assert route(state) == "end"


def test_router_sends_a_skipped_run_forward():
    from src.nodes.router import route

    state = _state()
    with patch("src.nodes.hitl.interrupt", return_value={"decision": "skip"}):
        state.update(review_wait_node(state))

    assert route(state) == "forecast"


# ---------------------------------------------------------------------------
# skipped_refs survives a COMPILED graph, not just a direct node call
# ---------------------------------------------------------------------------
#
# Every skip test above calls review_wait_node() directly and inspects the dict
# it returned, which is exactly why they all passed while the feature was broken
# in the real pipeline: LangGraph silently DISCARDS any key a node returns that
# is not declared on the state schema. `skipped_refs` was returned by
# review_wait_node and read by draft_po._consumption, but was missing from
# AgentState, so in the compiled graph it never made the trip between them and
# "Skip flagged" degraded into a silent approve.
#
# A direct-call assertion structurally cannot catch that -- the loss happens in
# the channel-write step, downstream of the node's return -- so this test goes
# through compile()/invoke(). A one-node graph is enough: the drop is a property
# of the schema, not of the Day Zero topology, and keeping it minimal keeps it
# fast and free of LLM/Slack mocking.
def test_skipped_refs_survives_a_compiled_graph():
    from langgraph.graph import END, START, StateGraph

    from src.state import AgentState

    graph = StateGraph(AgentState)
    graph.add_node(
        "skip", lambda state: {"skipped_refs": ["PROD-TOMATO-ROMA"], "stage": "reviewed"}
    )
    graph.add_edge(START, "skip")
    graph.add_edge("skip", END)

    out = graph.compile().invoke({"stage": "costed"})

    assert "skipped_refs" in out, (
        "AgentState must declare skipped_refs; LangGraph drops undeclared keys "
        "and the reviewer's skip silently becomes an approve"
    )
    assert out["skipped_refs"] == ["PROD-TOMATO-ROMA"]
    assert out["stage"] == "reviewed"
