"""Pure routing function. No I/O, no side effects, no exceptions.

Reads state, returns the name of the next node. The graph wires the names
to actual nodes via add_conditional_edges.

Order of checks is load-bearing and deliberately explicit:

  1. `human_decision == "reject"` wins over everything. A human who said no
     must not be overruled by a pipeline that still has work queued.
  2. The circuit breaker (`retry_count >= settings.max_retries`) comes
     *second*, above the stage dispatch. In a multi-stage pipeline a node can
     populate its output field and still fail on a later chunk, so a stage-based
     dispatch placed above the breaker would happily route forward past a
     tripped breaker. CLAUDE.md calls the breaker absolute; this ordering is
     what actually makes it so.
  3. `needs_human` → a human pause. Which pause depends on whether there is a
     review queue to show: a non-empty queue means the review card, an empty
     one means the generic circuit-breaker card. Reading a field's emptiness is
     still a pure function of state.
  4. Otherwise dispatch on the last completed stage through `_NEXT_STAGE`.

An unknown stage returns "end" rather than raising — the no-exceptions rule
means a typo degrades to a finished run, never to a crashed one.
"""
from src.config import settings
from src.state import AgentState

# Last completed stage -> next node name. The graph's _ROUTES must cover every
# value here; tests/test_graph_build.py enforces that.
_NEXT_STAGE = {
    "":                    "ingest",
    "ingested":            "extract_menu",
    "menu_extracted":      "decompose_recipes",
    "recipes_decomposed":  "canonicalize",
    "canonicalized":       "cost_plates",
    "costed":              "review_gate",
    "reviewed":            "forecast",
    "forecast":            "draft_po",
    "po_drafted":          "po_approval",
    "po_approved":         "end",
}


def route(state: AgentState) -> str:
    """Decide next node based on state."""
    human_decision = state.get("human_decision", "")
    retry_count = state.get("retry_count", 0)
    needs_human = state.get("needs_human", False)
    review_queue = state.get("review_queue") or []
    stage = state.get("stage", "")

    # Human said reject → terminate immediately
    if human_decision == "reject":
        return "end"

    # Circuit breaker: too many failures → forced HITL, whatever the stage says
    if retry_count >= settings.max_retries:
        return "hitl"

    # A node flagged the run for a human. A pending review queue gets the review
    # card; anything else gets the generic approval card.
    if needs_human:
        return "review" if review_queue else "hitl"

    # Pipeline dispatch: unknown stage ends the run rather than raising
    return _NEXT_STAGE.get(stage, "end")
