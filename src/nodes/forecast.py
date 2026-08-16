"""Cold-start demand forecast node: a defensible PRIOR, not a forecast.

Read this before quoting any number this node produces.

On day one of a new customer there is no POS history, and everything downstream
— par levels, purchase orders, prep sheets — needs a demand number anyway. So
this node produces one from public metadata. Any accuracy figure worth quoting
belongs to a warmed-up system with weeks of real sales behind it. This node has
none of that, and its output must never be presented as if it did.

What this is:
  * A point estimate of covers per day/week derived from public metadata
    (seats, service style, price tier, hand-entered popular-times shape,
    review volume) plus an LLM judgement about item mix.
  * Enough to unblock day-one par levels, purchase orders and prep sheets,
    which are otherwise blocked on data that does not exist yet.

What this is NOT, and cannot be made into by any amount of prompt work:
  * It is unvalidated and unvalidatable. There is no held-out set, because
    there are no actuals. No accuracy figure can honestly be attached to it.
  * It has no weather, event, seasonality or promotion signal.
  * It is a level, not a distribution: no interval, no error bars that mean
    anything.

Its purpose is to be REPLACED. Once real sales arrive the prior should be
blended away within weeks; `src/tools/blend.py` is that convergence mechanism.
A node that overstates what it knows is worse than one that does less.

Failure handling: any exception is caught; retry_count++ ; the error is fed back
through `last_error`.
"""
import math

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import settings
from src.state import AgentState

# --------------------------------------------------------------------------
# Assumptions, named. Every constant below is a judgement call, and a judgement
# call you can point at is a design decision; the same number buried in a
# literal is a bug waiting to happen. Change these, not the output.
# --------------------------------------------------------------------------

# Seat turns per operating day by service style.
# Basis: a seat in a counter/slice shop is a poor capacity proxy — a large share
# of covers are takeaway and never occupy one at all — so its turn figure has to
# absorb that traffic. Calibrated against the 20-restaurant NYC corpus in
# data/restaurants/: counter was raised from a seat-bound 6.0 to 12.0 because
# 6.0 put Joe's Pizza (18 seats, Carmine St) at ~570 covers/week, roughly half
# of a credible 1,200-2,000 for that store. Full-service and fine-dining seats
# genuinely are the constraint, so those stay at plausible dining-room turns.
# Known failure mode: large counter venues (Tacombi, 64 seats) come out high,
# because the model assumes every seat turns 12x rather than that the takeaway
# volume is roughly fixed per storefront.
TURNS_PER_DAY = {
    "counter": 12.0,      # slice/falafel/taco counters: mostly takeaway, all-day trade
    "fast_casual": 4.5,   # order at till, sit down, ~2h dwell across lunch + dinner
    "full_service": 2.5,  # one lunch turn plus one and a half dinner turns
    "fine_dining": 1.6,   # tasting-menu pacing, dinner only for most of the corpus
}

# Fraction of nominal capacity actually filled, by price tier.
# Basis: cheaper rooms fill closer to the brim and empty faster; expensive rooms
# hold tables, take fewer walk-ins and run thinner mid-week. These are averages
# over an ordinary week, not peak-service occupancy.
UTILISATION = {
    "$": 0.75,
    "$$": 0.65,
    "$$$": 0.55,
    "$$$$": 0.45,
}

# Average number of menu items ordered per cover, by service style.
# Basis: a slice-counter guest buys roughly one item and sometimes a drink; a
# full-service guest orders an appetiser-plus-main, often a dessert or a side.
# Handed to the model so item shares are anchored to something real rather than
# forced to sum to 1.0.
ITEMS_PER_COVER = {
    "counter": 1.2,
    "fast_casual": 1.5,
    "full_service": 2.4,
    "fine_dining": 3.2,
}

# Fallbacks when meta.json carries a style or tier we have no calibration for.
# Deliberately the mid of the corpus rather than an exception: an unknown style
# should degrade the estimate, not block the pipeline.
DEFAULT_TURNS_PER_DAY = 3.0
DEFAULT_UTILISATION = 0.65
DEFAULT_ITEMS_PER_COVER = 1.8

# Median review_count across the 20-restaurant corpus in data/restaurants/
# (computed 2026-08-08). Used as the denominator so a median-reviewed
# restaurant gets a correction of exactly 1.0 and the constants above stay
# readable as "covers for a typical store".
CORPUS_MEDIAN_REVIEW_COUNT = 2750

# Review volume is a crude but real proxy for footfall. It is taken in log space
# and clamped so a 20,000-review tourist landmark cannot triple the estimate and
# a brand-new room with 30 reviews cannot zero it out.
REVIEW_CORRECTION_MIN = 0.7
REVIEW_CORRECTION_MAX = 1.4


def _review_correction(
    review_count: float,
    median_review_count: float = CORPUS_MEDIAN_REVIEW_COUNT,
) -> float:
    """log10(n+1) / log10(median+1), clamped to [0.7, 1.4]. No I/O, no model."""
    denominator = math.log10(float(median_review_count) + 1.0)
    if denominator <= 0.0:
        return 1.0

    raw = math.log10(max(float(review_count), 0.0) + 1.0) / denominator
    return min(max(raw, REVIEW_CORRECTION_MIN), REVIEW_CORRECTION_MAX)


def _covers_per_day(restaurant: dict) -> list[float]:
    """Seven covers figures, Monday -> Sunday, from metadata alone.

    Deterministic and model-free by design: this half of the node must be
    unit-testable without an API key. The arithmetic is

        base    = seats * turns_per_day(service_style) * utilisation(price_tier)
        weights = popular_times_index, rescaled so its mean is 1.0
        day_i   = base * weights_i * review_correction(review_count)

    so the mean of the seven returned values is exactly
    `base * review_correction`, whatever shape the week has.
    """
    seats = float(restaurant.get("seats") or 0.0)
    # meta.json writes "full-service"; the constants above key on "full_service".
    style = str(restaurant.get("service_style") or "").strip().lower().replace("-", "_")
    tier = str(restaurant.get("price_tier") or "").strip()

    turns = TURNS_PER_DAY.get(style, DEFAULT_TURNS_PER_DAY)
    utilisation = UTILISATION.get(tier, DEFAULT_UTILISATION)
    base = seats * turns * utilisation

    index = [float(v) for v in (restaurant.get("popular_times_index") or [])]
    if len(index) != 7:
        raise ValueError(
            f"popular_times_index must hold 7 values (Mon..Sun), got {len(index)}"
        )

    mean_index = sum(index) / 7.0
    if mean_index <= 0.0:
        raise ValueError("popular_times_index sums to zero: cannot shape a week from it")

    weights = [v / mean_index for v in index]
    correction = _review_correction(restaurant.get("review_count") or 0)

    return [base * w * correction for w in weights]


# --------------------------------------------------------------------------
# LLM half: item mix. Not derivable from arithmetic — it is judgement about
# what people actually order at this kind of restaurant.
# --------------------------------------------------------------------------


class ItemShare(BaseModel):
    item_name: str
    share: float = Field(ge=0.0, le=1.0, description="Fraction of total covers ordering this item")
    reasoning: str = Field(description="One clause: why this share")


class ItemMix(BaseModel):
    shares: list[ItemShare]
    assumptions: list[str] = Field(description="Every assumption you made, stated plainly")
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = (
    "You estimate the item mix of a restaurant that has no sales history at all. This is "
    "day one: no POS data exists, and the kitchen still has to be told what to order "
    "tomorrow. Your job is to produce the best defensible prior an experienced restaurant "
    "operator would produce from the menu and the room, and to be explicit about what you "
    "assumed.\n"
    "\n"
    "GIVE EVERY MENU ITEM A SHARE. `share` is the fraction of covers that order that item: "
    "0.35 means roughly one guest in three orders it. Shares are independent per item, they "
    "are NOT a probability distribution and they do NOT have to sum to 1.0. Across all items "
    "they should sum to approximately the average items-per-cover figure you are given, "
    "because that is how many things a typical guest actually orders.\n"
    "\n"
    "THINK LIKE AN OPERATOR, NOT LIKE A UNIFORM DISTRIBUTION. The signature item, the "
    "cheapest item and the item the restaurant is known for carry far more volume than the "
    "tail. Price tier, neighbourhood and service style all move the mix: a counter takes "
    "one-item orders at speed, a full-service dining room sells appetisers and sides "
    "alongside mains, an expensive room sells fewer covers of more things. Sections matter "
    "too: a large section is usually a large share of the business, and beverages and sides "
    "attach on top of mains rather than replacing them.\n"
    "\n"
    "POPULATE `assumptions` — THIS IS NOT OPTIONAL. Each entry is one plain sentence naming "
    "something you assumed rather than knew: what you took the signature item to be, how you "
    "split a section you could not distinguish, what you assumed about takeaway versus "
    "dine-in, anything you inferred from the neighbourhood. These sentences are shown "
    "verbatim to the restaurant's own manager next to the purchase order they justify, so "
    "write them for that reader. An unstated assumption is the one that costs someone money.\n"
    "\n"
    "`reasoning` is one clause per item, not a paragraph. `confidence` is your honest "
    "self-assessment of the whole mix: a menu with a clear signature dish deserves a higher "
    "score than a sprawling one where any split is a guess. An honest 0.5 is more useful to "
    "us than an optimistic 0.9, because low scores route to a human instead of straight to a "
    "purchase order."
)


def _build_llm():
    """Factory isolated to make mocking trivial in tests."""
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
        thinking={"type": "adaptive"},
    ).with_structured_output(ItemMix)


def _menu_block(menu_items: list[dict]) -> str:
    """Menu rendered for the prompt: section, name, price. No model, no I/O."""
    lines = []
    for item in menu_items:
        price = item.get("price")
        printed = f"${float(price):.2f}" if isinstance(price, (int, float)) else "no printed price"
        section = item.get("section") or "(no section)"
        lines.append(f"- [{section}] {item.get('name', '')} — {printed}")
    return "\n".join(lines)


def forecast_node(state: AgentState) -> dict:
    """Deterministic covers + LLM item mix -> state['demand_forecast']."""
    restaurant = state.get("restaurant") or {}
    menu_items = state.get("menu_items") or []
    if not restaurant or not menu_items:
        # Paying for an Opus call that cannot succeed is the cheapest bug to prevent.
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": "forecast needs both restaurant and menu_items",
            "demand_forecast": {},
        }

    llm = _build_llm()

    msgs = [SystemMessage(SYSTEM_PROMPT)]
    if state.get("last_error"):
        msgs.append(HumanMessage(
            f"Previous attempt invalid: {state['last_error']}. "
            f"Correct it and try again."
        ))

    try:
        covers = _covers_per_day(restaurant)

        style = str(restaurant.get("service_style") or "").strip().lower().replace("-", "_")
        target_items_per_cover = ITEMS_PER_COVER.get(style, DEFAULT_ITEMS_PER_COVER)

        msgs.append(HumanMessage(
            f"Restaurant: {restaurant.get('name', '')}\n"
            f"Cuisine: {restaurant.get('cuisine', '')}\n"
            f"Neighbourhood: {restaurant.get('neighborhood', '')}\n"
            f"Service style: {restaurant.get('service_style', '')}\n"
            f"Price tier: {restaurant.get('price_tier', '')}\n"
            f"Seats: {restaurant.get('seats', '')}\n"
            f"Estimated covers per week: {round(sum(covers))}\n"
            f"Average items ordered per cover at this kind of restaurant: "
            f"{target_items_per_cover}\n"
            f"\n"
            f"Menu ({len(menu_items)} items):\n"
            f"{_menu_block(menu_items)}\n"
            f"\n"
            f"Give every item above a share of covers. The shares should sum to roughly "
            f"{target_items_per_cover} across the whole menu, not to 1.0. Populate "
            f"assumptions with one plain sentence per assumption you made."
        ))

        result = llm.invoke(msgs)

        if not result.assumptions:
            raise ValueError("assumptions list came back empty and it is required")

        raw = {s.item_name: max(float(s.share), 0.0) for s in result.shares}
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("every item share came back zero: no mix to normalise")

        # Normalise in Python rather than trusting the model's arithmetic: the
        # shares are guaranteed to sum to the target items-per-cover, so
        # `src/nodes/draft_po.py` can multiply straight through to consumption
        # without re-checking.
        scale = target_items_per_cover / total
        item_mix = {name: share * scale for name, share in raw.items()}

        return {
            "demand_forecast": {
                "covers_per_day": [round(c, 1) for c in covers],
                "covers_per_week": round(sum(covers), 1),
                "item_mix": item_mix,
                "assumptions": list(result.assumptions),
                "confidence": result.confidence,
                "method": "cold_start_prior",
            },
            "stage": "forecast",
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "demand_forecast": {},
        }
