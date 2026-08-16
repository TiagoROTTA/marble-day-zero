"""Recipe decomposition: turn each menu item into an ingredient list with quantities.

The single largest token consumer in the pipeline (a 60-item menu is 60 items of
work), so it carries the two cost levers:

  1. `settings.llm_model_cheap` instead of `llm_model`. Left at its
     `claude-opus-5` default for best quality; setting
     `LLM_MODEL_CHEAP=claude-haiku-4-5` in `.env` cuts this node ~5x with no
     code change.
  2. Menu items are batched 10 per call, and the SKU catalog display-name list
     rides in the system prompt inside a `cache_control: ephemeral` block. That
     block is byte-identical (and deterministically ordered) across chunks, so
     it is written to cache once and read back at ~a tenth of input price on
     every later chunk. Seeing sibling dishes together also makes the model use
     the same `raw_name` for the mozzarella on the Margherita and on the
     Marinara, which is what makes canonicalization tractable.

Failure handling: any exception (validation, network, rate limit) is caught;
retry_count++ ; the error message is fed back to the LLM on the next attempt via
`last_error` and an appended HumanMessage. A chunk that fails does not wipe the
chunks that succeeded — partial progress plus a trippable circuit breaker beats
an all-or-nothing wipe on a 60-item menu.
"""
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import settings
from src.state import AgentState

CATALOG_PATH = str(Path(__file__).resolve().parents[2] / "data" / "catalog" / "skus.json")

# The eight units the SKU catalog is written in. The unit converter in
# `src/nodes/cost_plates.py` knows these and nothing else, so anything outside
# the set is an uncostable line.
VALID_UOMS = ("lb", "oz", "gal", "qt", "fl_oz", "each", "dozen", "bunch")

# One call per 10 menu items: 10 calls for a 100-item menu instead of 100.
CHUNK_SIZE = 10


class Component(BaseModel):
    raw_name: str = Field(description="Ingredient as you would write it, e.g. 'roma tomatoes'")
    qty: float = Field(gt=0, description="Quantity per single serving")
    uom: str = Field(description="One of: lb, oz, gal, qt, fl_oz, each, dozen, bunch")
    confidence: float = Field(ge=0.0, le=1.0)


class Recipe(BaseModel):
    item_name: str
    yield_qty: float = Field(default=1.0, description="Servings this recipe produces")
    yield_uom: str = Field(default="serving")
    components: list[Component]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the whole decomposition")


class RecipeBatch(BaseModel):
    recipes: list[Recipe]


SYSTEM_PROMPT = (
    "You are a restaurant costing agent. You are given a batch of menu items and you return, "
    "for each one, the recipe a kitchen would actually use: the ingredients and how much of "
    "each goes into the dish.\n"
    "\n"
    "QUANTITIES ARE PER SINGLE SERVING, NEVER PER BATCH. One pizza is one serving. One pasta "
    "dish is one serving. One cocktail is one serving. Do not return the 40-portion prep-batch "
    "quantity a kitchen would make on a Tuesday morning: return the share of it that lands on "
    "one plate. A 12-inch Margherita carries roughly 0.25 lb of mozzarella, not 4 lb. If a "
    "dish is genuinely sold as a shareable platter, still describe one sold unit and say so via "
    "yield_qty / yield_uom.\n"
    "\n"
    "INCLUDE EVERYTHING THAT COSTS MONEY. Oil, butter, salt, the garnish, the lemon wedge, the "
    "flour the dough is made from — these are real line items on a food-cost report and leaving "
    "them out is how a plate cost comes out 15% low. Do not itemise trivial amounts below "
    "roughly a gram: a pinch of dried oregano is noise, half an ounce of olive oil is not.\n"
    "\n"
    "UNITS. Every uom MUST be exactly one of: lb, oz, gal, qt, fl_oz, each, dozen, bunch. "
    "Nothing else exists downstream — 'g', 'ml', 'tsp', 'cup', 'slice' and 'portion' are all "
    "rejected and the line becomes uncostable. Convert before answering: use oz or lb for "
    "solids, fl_oz / qt / gal for liquids, each for countable items (eggs, lemons, buns), "
    "bunch for herbs sold by the bunch.\n"
    "\n"
    "CONFIDENCE IS YOUR OWN HONEST SELF-ASSESSMENT, AND A LOW SCORE IS A CORRECT ANSWER, NOT A "
    "FAILURE. Downstream stages route low-confidence recipes to a human instead of guessing. "
    "Score the recipe under 0.6 whenever the composition is genuinely ambiguous from the menu "
    "text alone: 'Chef's Special', 'Market Fish', 'Soup of the Day', or any dish with no "
    "description whose name does not pin down what is in it. An honest 0.4 on a Chef's Special "
    "is worth far more to us than a confident invention. Reserve above 0.85 for dishes whose "
    "composition is standard and unambiguous.\n"
    "\n"
    "BE CONSISTENT ACROSS THE BATCH. If two dishes use the same ingredient, write its raw_name "
    "identically in both. Return exactly one recipe per menu item you are given, with "
    "item_name copied verbatim from the input, and no extras."
)

CATALOG_HEADING = (
    "Prefer these ingredient names when they fit. They are the purchasing catalog this "
    "restaurant buys from; using this vocabulary in raw_name makes the ingredient resolve "
    "cleanly to a real price. If an ingredient genuinely is not in this list, write it "
    "naturally instead of forcing it onto the nearest entry.\n\n"
)

INSTRUCTION = (
    "Decompose each of these menu items into a per-serving recipe. "
    "Return one recipe per item, item_name copied verbatim."
)


def _load_catalog_names() -> list[str]:
    """Newline-joinable list of catalog display_names, deterministically ordered.

    Sorted, and sorted by sku_id rather than by display_name, so the block is
    byte-identical on every chunk call. A non-deterministic order here silently
    invalidates the prompt cache and you pay full input price on all ten chunks.
    """
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    names = []
    seen = set()
    for sku in sorted(catalog, key=lambda s: s["sku_id"]):
        name = (sku.get("display_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _chunk(items: list, size: int = CHUNK_SIZE) -> list[list]:
    """Split a list into consecutive groups of at most `size`."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _build_llm():
    """Factory isolated to make mocking trivial in tests."""
    return ChatAnthropic(
        model=settings.llm_model_cheap,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
        # Thinking ON, and this was measured rather than assumed. Turning it off
        # to save tokens made this node stop differentiating its confidence: on
        # joes-pizza-carmine the component confidences collapsed from a real
        # spread (0.45-0.80, mode 0.75) to 0.50 for 46% of components -- a lazy
        # default rather than a judgement. That is not a cosmetic problem. The
        # confidence it emits drives the review gate, so a pile of 0.50s pushes
        # every item under CONF_REVIEW_FLOOR, empties the review queue, and the
        # human-in-the-loop step silently stops having anything to ask about.
        #
        # Estimating how much of an ingredient a dish contains, and how sure you
        # are of that, is the one genuinely judgement-heavy call in this pipeline.
        # It is worth the thinking tokens. The truncation this was meant to fix
        # is handled by `llm_max_tokens` (32k) plus the 10-item batching above.
        thinking={"type": "adaptive"},
    ).with_structured_output(RecipeBatch)


def decompose_recipes_node(state: AgentState) -> dict:
    """Turn state['menu_items'] into recipes, ten items per LLM call."""
    menu_items = state.get("menu_items") or []
    if not menu_items:
        # No point paying for a call that cannot produce anything.
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": "menu extraction did not populate menu_items",
            "recipes": [],
        }

    try:
        catalog_names = _load_catalog_names()
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "recipes": [],
            "messages": [HumanMessage(f"Decomposition failed: {e}")],
        }

    # Two system blocks: the instructions, then the catalog. The cache breakpoint
    # sits on the catalog block, so the whole prefix is written once and read
    # back on the other chunk calls.
    system_blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {
            "type": "text",
            "text": CATALOG_HEADING + "\n".join(catalog_names),
            "cache_control": {"type": "ephemeral"},
        },
    ]

    llm = _build_llm()

    all_recipes: list[dict] = []
    new_review_items: list[dict] = []
    failed_items: list[str] = []
    chunk_errors: list[str] = []

    for chunk in _chunk(menu_items, CHUNK_SIZE):
        payload = [
            {
                "name": item.get("name", ""),
                "section": item.get("section", ""),
                "description": item.get("description", ""),
                "price": item.get("price"),
            }
            for item in chunk
        ]

        msgs = [SystemMessage(content=system_blocks)]
        if state.get("last_error"):
            msgs.append(HumanMessage(
                f"Previous attempt invalid: {state['last_error']}. "
                f"Correct it and try again."
            ))
        msgs.append(HumanMessage(
            f"{INSTRUCTION}\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        ))

        try:
            result = llm.invoke(msgs)
        except Exception as e:
            # Keep what the other chunks produced; name the casualties.
            failed_items.extend(item.get("name", "") for item in chunk)
            chunk_errors.append(f"{type(e).__name__}: {e}")
            continue

        for recipe in result.recipes:
            kept: list[dict] = []
            for component in recipe.components:
                if component.uom not in VALID_UOMS:
                    # Dropped rather than kept: cost_plates cannot convert it, and a
                    # silent unconvertible line is a plate cost that is quietly wrong.
                    new_review_items.append({
                        "kind": "recipe_uom",
                        "ref": f"{recipe.item_name} / {component.raw_name}",
                        "confidence": component.confidence,
                        "question": (
                            f"'{component.raw_name}' came back as "
                            f"{component.qty} {component.uom}, which is not a costable unit. "
                            f"Restate it in one of: {', '.join(VALID_UOMS)}."
                        ),
                        "payload": {
                            "item_name": recipe.item_name,
                            "raw_name": component.raw_name,
                            "qty": component.qty,
                            "uom": component.uom,
                        },
                    })
                    continue
                kept.append(component.model_dump())

            dumped = recipe.model_dump()
            dumped["components"] = kept
            all_recipes.append(dumped)

    if failed_items:
        return {
            "recipes": all_recipes,
            "review_queue": state.get("review_queue", []) + new_review_items,
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": (
                f"decomposition failed for {len(failed_items)} item(s): "
                f"{', '.join(failed_items)} | {' ; '.join(chunk_errors)}"
            ),
            "messages": [HumanMessage(
                f"Partial decomposition: {len(failed_items)} item(s) failed."
            )],
        }

    return {
        "recipes": all_recipes,
        "review_queue": state.get("review_queue", []) + new_review_items,
        "stage": "recipes_decomposed",
        "last_error": "",
    }
