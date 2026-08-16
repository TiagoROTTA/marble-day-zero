"""Ingest node: loads the frozen snapshot for `state["input"]` (a restaurant slug).

Failure handling: any exception (missing directory, incomplete meta.json,
malformed popular_times_index) is caught; retry_count++ and the error is fed
back through `last_error`. A missing snapshot is exactly the kind of failure
that should trip the circuit breaker and route to a human rather than crash
the run — and reusing the established shape means the router needs no special
case.
"""
from src.state import AgentState
from src.tools.snapshot import load_restaurant


def ingest_node(state: AgentState) -> dict:
    """Load `<slug>/meta.json` into state. Router decides what to do next."""
    try:
        meta = load_restaurant(state["input"])
        return {
            "restaurant": meta,
            "stage": "ingested",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
        }
