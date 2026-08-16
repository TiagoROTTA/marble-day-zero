"""Graph topology. Wires nodes together with conditional edges.

Touch only when adding or removing nodes. Routing logic lives in src/nodes/router.py.
"""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.nodes.canonicalize import canonicalize_node
from src.nodes.cost_plates import cost_plates_node
from src.nodes.decompose_recipes import decompose_recipes_node
from src.nodes.draft_po import draft_po_node
from src.nodes.extract_menu import extract_menu_node
from src.nodes.forecast import forecast_node
from src.nodes.hitl import (
    hitl_node,
    hitl_notify_node,
    po_notify_node,
    po_wait_node,
    review_notify_node,
    review_wait_node,
)
from src.nodes.ingest import ingest_node
from src.nodes.review_gate import review_gate_node
from src.nodes.router import route
from src.nodes.worker import worker_node
from src.state import AgentState

# Every value route() can return must be a key here, mapped to a real node name.
# The two human pauses map to their *notify* nodes, not their wait nodes: notify
# sends the Slack card once, wait calls interrupt() and is safe to re-execute.
# build_graph() below registers the nodes themselves; the names must agree.
_ROUTES = {
    "ingest": "ingest",
    "extract_menu": "extract_menu",
    "decompose_recipes": "decompose_recipes",
    "canonicalize": "canonicalize",
    "cost_plates": "cost_plates",
    "review_gate": "review_gate",
    "forecast": "forecast",
    "draft_po": "draft_po",
    "po_approval": "po_notify",
    "review": "review_notify",
    "worker": "worker",
    "hitl": "hitl_notify",
    "end": END,
}


# Nodes that can advance the run: each gets the same conditional edge on route().
# A loop rather than twelve near-identical calls — same call, applied over a list.
# The three *_notify nodes are absent on purpose: they hand off unconditionally to
# their paired wait node, which is where interrupt() lives.
_ADVANCING = (
    "ingest",
    "extract_menu",
    "decompose_recipes",
    "canonicalize",
    "cost_plates",
    "review_gate",
    "review_wait",
    "forecast",
    "draft_po",
    "po_wait",
    "worker",
    "hitl",
)


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Day Zero pipeline. Names match src/nodes/router.py's _NEXT_STAGE one for one.
    g.add_node("ingest", ingest_node)
    g.add_node("extract_menu", extract_menu_node)
    g.add_node("decompose_recipes", decompose_recipes_node)
    g.add_node("canonicalize", canonicalize_node)
    g.add_node("cost_plates", cost_plates_node)
    g.add_node("review_gate", review_gate_node)
    g.add_node("review_notify", review_notify_node)
    g.add_node("review_wait", review_wait_node)
    g.add_node("forecast", forecast_node)
    g.add_node("draft_po", draft_po_node)
    g.add_node("po_notify", po_notify_node)
    g.add_node("po_wait", po_wait_node)

    # The generic retry/approval path the circuit breaker falls back to. It is
    # also what scripts/run_agent.py drives directly for a plain text input.
    g.add_node("worker", worker_node)
    g.add_node("hitl_notify", hitl_notify_node)
    g.add_node("hitl", hitl_node)

    g.add_edge(START, "ingest")

    # notify → wait pairs: send the card once, then pause in a node that is safe
    # to re-execute when interrupt() replays it on resume.
    g.add_edge("hitl_notify", "hitl")
    g.add_edge("review_notify", "review_wait")
    g.add_edge("po_notify", "po_wait")

    for name in _ADVANCING:
        g.add_conditional_edges(name, route, _ROUTES)

    return g


def get_checkpointer() -> SqliteSaver:
    db_path = Path(settings.checkpoint_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def get_compiled_graph():
    """Return a compiled graph backed by SQLite. Caller owns the lifetime."""
    return build_graph().compile(checkpointer=get_checkpointer())
