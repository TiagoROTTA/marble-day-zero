"""Human-in-the-loop nodes. Split into two to avoid double Slack sends.

In LangGraph 0.2.x, a node is re-executed from the beginning when resumed after
interrupt(). Splitting notify/wait ensures send_approval runs exactly once:
  hitl_notify  — sends the Slack message, completes normally (never re-executed).
  hitl_wait    — calls interrupt(), safe to re-execute (no side effects).
"""
from langgraph.config import get_config
from langgraph.types import interrupt

from src.slack.client import send_approval, send_purchase_order, send_review_queue
from src.state import AgentState


def hitl_notify_node(state: AgentState) -> dict:
    """Send the Slack approval message. Runs once, never re-executed on resume."""
    cfg = get_config()
    thread_id = cfg["configurable"]["thread_id"]

    title = f"Agent stuck after {state.get('retry_count', 0)} retries — needs review"
    context = {
        "Last error": state.get("last_error", "n/a") or "n/a",
        "Input excerpt": (state.get("input", "") or "")[:200],
        "Retry count": str(state.get("retry_count", 0)),
    }

    send_approval(thread_id, title, context)
    return {}


def hitl_node(state: AgentState) -> dict:
    """Pause and wait for human decision. Safe to re-execute on resume."""
    decision_payload = interrupt({"awaiting": "human_approval"})

    return {
        "human_decision": decision_payload.get("decision", "reject"),
        "needs_human": False,
        "retry_count": 0,
    }


def review_notify_node(state: AgentState) -> dict:
    """Post the low-confidence review card. Runs once, never re-executed.

    Same notify/wait split as hitl_notify_node/hitl_node, for the same reason:
    on resume LangGraph re-runs the interrupting node from the top, so the send
    has to live in a node that never interrupts. Returns {} — this node
    contributes nothing to state, its whole job is the side effect.
    """
    cfg = get_config()
    thread_id = cfg["configurable"]["thread_id"]

    restaurant_name = (state.get("restaurant") or {}).get("name", "") or state.get("input", "")
    items = state.get("review_queue") or []

    # The gate namespaces its bookkeeping under extracted_data because LangGraph
    # discards node-return keys that are not AgentState fields.
    gate = (state.get("extracted_data") or {}).get("review_gate") or {}
    dropped = int(gate.get("dropped", 0) or 0)

    send_review_queue(thread_id, restaurant_name, items, dropped)
    return {}


def review_wait_node(state: AgentState) -> dict:
    """Pause on the review queue and translate the decision into state.

    Safe to re-execute on resume: no side effects, only interrupt().

    Clearing `review_queue` and resetting `retry_count` is what closes the
    review_gate loop. The gate deliberately leaves `stage` at "costed" while it
    has questions pending, so the resumed run routes back into the gate; finding
    an empty queue there, it advances to "reviewed" and the pipeline moves on.
    Leaving a single entry in the queue here would re-flag the same items
    forever.

    "skip" is not a quiet approve. The flagged refs are recorded in
    `skipped_refs` so draft_po drops those SKUs from the order and the food-cost
    chart drops those plates: unreviewed numbers must not be folded into a total
    presented as approved.
    """
    decision_payload = interrupt(
        {"awaiting": "review_queue", "count": len(state.get("review_queue", []))}
    )
    decision = decision_payload.get("decision", "reject")

    if decision == "approve":
        return {
            "human_decision": "approve",
            "needs_human": False,
            "review_queue": [],
            "stage": "reviewed",
            "retry_count": 0,
        }

    if decision == "skip":
        skipped_refs = [
            str(entry.get("ref"))
            for entry in (state.get("review_queue") or [])
            if entry.get("ref")
        ]
        return {
            "human_decision": "skip",
            "needs_human": False,
            "review_queue": [],
            "skipped_refs": skipped_refs,
            "stage": "reviewed",
            "retry_count": 0,
        }

    # Anything else — including an unrecognised value — is a rejection. The
    # router's first check turns "reject" into "end", so `stage` is deliberately
    # left where it was: a rejected run was never reviewed.
    return {
        "human_decision": "reject",
        "needs_human": False,
        "review_queue": [],
        "retry_count": 0,
    }


def po_notify_node(state: AgentState) -> dict:
    """Post the draft opening order to Slack. Runs once, never re-executed."""
    cfg = get_config()
    thread_id = cfg["configurable"]["thread_id"]

    restaurant_name = (state.get("restaurant") or {}).get("name", "") or state.get("input", "")

    send_purchase_order(thread_id, restaurant_name, state.get("purchase_order") or {})
    return {}


def po_wait_node(state: AgentState) -> dict:
    """Pause for approval of the purchase order. Safe to re-execute on resume.

    A rejection is terminal on purpose: the router's first check ends the run, and
    an opening order a human refused should stop rather than quietly proceed.
    """
    decision_payload = interrupt({"awaiting": "po_approval"})

    return {
        "human_decision": decision_payload.get("decision", "reject"),
        "needs_human": False,
        "stage": "po_approved",
        "retry_count": 0,
    }
