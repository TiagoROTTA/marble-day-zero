from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Confidence contract, shared by every node that emits a `confidence` float in [0.0, 1.0].
# >= CONF_AUTO_ACCEPT            -> auto-accept
# CONF_REVIEW_FLOOR .. CONF_AUTO_ACCEPT -> human review queue
# < CONF_REVIEW_FLOOR            -> auto-rejected, recorded as a gap rather than guessed at
CONF_AUTO_ACCEPT = 0.85
CONF_REVIEW_FLOOR = 0.60


class AgentState(TypedDict, total=False):
    # I/O — the run's input and a free-form bag for per-node bookkeeping
    input: str
    extracted_data: dict[str, Any]

    # Conversation history (LLM sees its own past errors)
    messages: Annotated[list[BaseMessage], add_messages]

    # Control flow / autocorrection
    retry_count: int
    last_error: str

    # Human-in-the-loop
    human_decision: str   # "approve" | "reject" | ""
    needs_human: bool

    # --- Day Zero pipeline (plain dicts/lists of primitives, deliberately) ---
    restaurant: dict[str, Any]          # the loaded meta.json plus resolved snapshot path
    menu_items: list[dict[str, Any]]    # {name, section, price, description, confidence}
    recipes: list[dict[str, Any]]       # {item_name, yield_qty, yield_uom, components:[...]}
    sku_matches: list[dict[str, Any]]   # {raw_name, sku_id, method, confidence}
    # plate_cost, cost_per_serving and food_cost_pct are None when costable is
    # False: nothing in the plate could be costed, and 0.0 there would read as a
    # real 0% food cost. plate_cost is the cost of the whole MENU LINE — the same
    # sold unit menu_price buys, so that food_cost_pct means something;
    # cost_per_serving is that divided by yield_qty.
    # {item_name, plate_cost, cost_per_serving, yield_qty, menu_price,
    #  food_cost_pct, coverage, costable, confidence}
    plate_costs: list[dict[str, Any]]
    demand_forecast: dict[str, Any]     # {covers_per_day:[7 floats], item_mix:{name: share}, assumptions:[str]}
    par_levels: list[dict[str, Any]]    # {sku_id, par_qty, uom, days_cover, rationale}
    purchase_order: dict[str, Any]      # {vendor_lines:[...], total_cost, generated_at_stage}
    review_queue: list[dict[str, Any]]  # {kind, ref, confidence, question, detail, payload}
    # Refs (sku_id or raw_name) the reviewer chose to skip rather than approve,
    # written by `review_wait_node`; `draft_po._consumption` withholds those SKUs
    # from the order. It MUST be declared here: LangGraph silently discards any
    # key a node returns that is not a field of the state schema, so without this
    # line "Skip flagged" reaches draft_po as an empty list and behaves as a
    # silent approve — the exact failure hitl.py's docstring says must not happen.
    skipped_refs: list[str]
    stage: str                          # last completed pipeline stage name


def initial_state(user_input: str) -> AgentState:
    """Fresh state for a new run. Use this in CLI / API entrypoints."""
    return {
        "input": user_input,
        "extracted_data": {},
        "messages": [],
        "retry_count": 0,
        "last_error": "",
        "human_decision": "",
        "needs_human": False,
    }


def initial_dayzero_state(slug: str) -> AgentState:
    """Fresh state for a Day Zero pipeline run, keyed by restaurant slug.

    Same base keys as initial_state() so the retry / HITL machinery works
    untouched, plus every Day Zero field initialised empty.
    """
    return {
        "input": slug,
        "extracted_data": {},
        "messages": [],
        "retry_count": 0,
        "last_error": "",
        "human_decision": "",
        "needs_human": False,
        "restaurant": {},
        "menu_items": [],
        "recipes": [],
        "sku_matches": [],
        "plate_costs": [],
        "demand_forecast": {},
        "par_levels": [],
        "purchase_order": {},
        "review_queue": [],
        "skipped_refs": [],
        "stage": "",
    }
