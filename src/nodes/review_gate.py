"""Confidence gate: decides whether a human needs to look at anything.

**No LLM call, no network call.** It is a filter over `review_queue` and nothing
else, and keeping the "should a human see this?" policy in one small readable
file is the whole point — the router stays a lookup table, this node owns the
policy.

Three bands, straight from the contract in `src/state.py`:

  * `confidence >= CONF_AUTO_ACCEPT` (0.85) — auto-accepted. Not asked about.
  * `CONF_REVIEW_FLOOR .. CONF_AUTO_ACCEPT` — uncertain enough to be worth
    asking about, plausible enough to be worth keeping. This is the queue.
  * `confidence < CONF_REVIEW_FLOOR` (0.60) — recorded as a **gap**, not a
    question. Asking a human to adjudicate something the system has no real
    guess about wastes the one scarce resource in this loop: reviewer attention.

The queue is capped at `MAX_REVIEW_ITEMS` and sorted ascending by confidence so
the shakiest items surface first. A Slack message with forty buttons is unusable
and Block Kit has its own limits. Everything left out is counted in `dropped` and
surfaced by the Slack card ("N further items fell below the review threshold"),
because silently truncating while implying full coverage would make the card
actively misleading.

`stage` is deliberately left at `"costed"` when items are flagged: when the human
approves and the graph resumes, the router sends the run back through this gate,
which now finds an empty queue and advances to `"reviewed"`. That is what closes
the loop.

Where the counts live: LangGraph silently discards node-return keys that are not
fields of `AgentState`, so the gate's own bookkeeping is namespaced under
`extracted_data["review_gate"]` — `{dropped, below_floor, truncated,
auto_accepted, gaps}`. `review_notify_node` in `src/nodes/hitl.py` reads
`dropped` from there.

Failure handling: any exception (a non-numeric confidence, a malformed entry) is
caught; retry_count++ ; the error is fed back through `last_error`.
"""
from langchain_core.messages import HumanMessage

from src.state import CONF_AUTO_ACCEPT, CONF_REVIEW_FLOOR, AgentState

# Twelve is enough to cover the genuinely shaky items and small enough to stay
# readable as a Slack card. Block Kit caps blocks per message; forty questions is
# not a review, it is a punishment.
MAX_REVIEW_ITEMS = 12


def review_gate_node(state: AgentState) -> dict:
    """Split the review queue into questions, gaps and auto-accepts."""
    try:
        queue = state.get("review_queue") or []

        # Nothing pending: either a fresh run with a clean queue or the resume
        # pass after a human answered. Advance and leave the first pass's
        # bookkeeping in extracted_data untouched.
        if not queue:
            return {"needs_human": False, "stage": "reviewed"}

        band: list[dict] = []
        gaps: list[dict] = []
        auto_accepted = 0

        for entry in queue:
            confidence = float(entry.get("confidence") or 0.0)
            if confidence >= CONF_AUTO_ACCEPT:
                auto_accepted += 1
            elif confidence >= CONF_REVIEW_FLOOR:
                band.append(entry)
            else:
                gaps.append(entry)

        band.sort(key=lambda e: float(e.get("confidence") or 0.0))
        kept = band[:MAX_REVIEW_ITEMS]
        truncated = len(band) - len(kept)
        dropped = len(gaps) + truncated

        record = {
            "dropped": dropped,
            "below_floor": len(gaps),
            "truncated": truncated,
            "auto_accepted": auto_accepted,
            "gaps": gaps,
        }
        extracted = dict(state.get("extracted_data") or {})
        extracted["review_gate"] = record

        print(
            f"review_gate: {len(queue)} flagged -> {len(kept)} queued, "
            f"{len(gaps)} below floor, {truncated} truncated, "
            f"{auto_accepted} auto-accepted"
        )

        if kept:
            return {
                "needs_human": True,
                "review_queue": kept,
                # Unchanged on purpose: the resume pass re-enters this gate.
                "stage": "costed",
                "extracted_data": extracted,
                "last_error": "",
            }

        return {
            "needs_human": False,
            "review_queue": [],
            "stage": "reviewed",
            "extracted_data": extracted,
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "messages": [HumanMessage(f"Review queue filtering failed: {e}")],
        }
