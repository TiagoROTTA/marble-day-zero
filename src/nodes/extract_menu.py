"""Menu extraction node: reads the frozen snapshot and transcribes it into items.

First LLM node of the Day Zero pipeline and the pattern the next four copy:
Pydantic schema declared here, `_build_llm()` factory isolated for mocking,
and the try/except -> retry_count++ -> feed `last_error` back idiom copied
verbatim in behaviour from `src/nodes/worker.py`.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import settings
from src.state import AgentState
from src.tools.snapshot import load_menu_source


class MenuItem(BaseModel):
    name: str = Field(description="Dish name exactly as printed on the menu")
    section: str = Field(description="Menu section, e.g. Appetizers, Pizzas, Desserts")
    price: float | None = Field(description="Price in USD, null if not printed")
    description: str = Field(description="Menu description verbatim, empty string if none")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your confidence this item and price were read correctly",
    )


class MenuExtraction(BaseModel):
    items: list[MenuItem]
    menu_notes: list[str] = Field(
        description="Anything ambiguous: unreadable prices, market-price items, sections you skipped"
    )


SYSTEM_PROMPT = (
    "You are a menu transcription agent for a restaurant back-office system. You read a "
    "single restaurant's menu and return it as structured data.\n"
    "\n"
    "TRANSCRIBE, DO NOT INVENT. Every item you return must be printed on the menu in front "
    "of you. Do not add dishes that a restaurant of this type 'would' serve, do not complete "
    "a section that looks truncated, and do not guess a price that is not printed. If a dish "
    "is on the menu but its price is not legible or not printed, return the item with "
    "price=null rather than a plausible number. A missing item is a smaller error than an "
    "invented one: everything downstream (recipe costing, purchase orders) is priced off "
    "this output.\n"
    "\n"
    "WHAT COUNTS AS AN ITEM. An item is one thing a guest orders and receives as its own line "
    "on the check: a dish, a shared plate, a side sold on its own, a dessert, or a drink. "
    "Return every single one that is printed. This is a transcription, not a summary or a "
    "selection: never return only the well-known dishes, the highest-priced ones, or the first "
    "few sections. If the menu prints forty items, return forty.\n"
    "\n"
    "Three kinds of line are NOT items, however the menu lays them out:\n"
    "- OPTION AND INGREDIENT BLOCKS: a 'Toppings', 'Choice of', 'Add-ons', 'Sauces' or "
    "'Mix-ins' list naming things a guest adds to some other dish. These stay out of items "
    "even when printed under their own bold heading, and even if that empties the section.\n"
    "- MODIFIER LINES: a line whose name does not by itself name something a guest receives "
    "-- '+add any one topping', 'add chicken', 'sub fries', 'large', 'extra cheese', "
    "'make it a combo'. The price on such a line is an upcharge or an alternative price for a "
    "neighbouring dish, not the price of a dish of its own. Note it, or fold it into that "
    "dish's description; do not give it its own entry.\n"
    "- PAGE FURNITURE: address, phone, opening hours, delivery minimums, website and ordering "
    "links, taglines, catering blurbs, allergy disclaimers.\n"
    "Add one menu_notes line for each modifier line and each option block you left out, so "
    "that nothing you dropped is invisible to the human reading your output.\n"
    "\n"
    "Excluding those is NOT the same as skipping a dish, and having a price is NOT what makes "
    "something an item. Plenty of menus print no prices at all; a real dish with no printed "
    "price still belongs in items with price=null. The test is whether the line names a thing "
    "a guest can order on its own, never whether a number sits next to it.\n"
    "\n"
    "CONFIDENCE IS YOUR OWN HONEST SELF-ASSESSMENT, AND A LOW SCORE IS A USEFUL ANSWER, NOT "
    "A FAILURE. Downstream stages route low-confidence items to a human instead of guessing, "
    "so an honest 0.4 is worth more to us than an optimistic 0.9. Score under 0.6, and add a "
    "line to menu_notes, whenever: the price is rendered as an image or is otherwise hard to "
    "read; the price is 'MP', 'market price', 'AQ' or similar; the item is a handwritten or "
    "chalkboard special; the item or section is cut off, cropped, blurred, or partially "
    "hidden by the snapshot; or you are unsure whether a number is the price of this item or "
    "of the one next to it. Reserve scores above 0.85 for items whose name and price are both "
    "plainly legible.\n"
    "\n"
    "PRESERVE THE MENU'S OWN SECTION NAMES. Use the heading exactly as the restaurant wrote "
    "it ('Zakuski', 'From the Wood Oven', 'Bo Luc Lac'), not a normalised taxonomy like "
    "'Appetizers' or 'Entrees'. The section names themselves are a signal about how this "
    "restaurant is shaped and later stages read them. If an item sits under no heading at "
    "all, use an empty string rather than inventing one.\n"
    "\n"
    "Prices are plain USD numbers: 18.5, not '$18.50'. Copy descriptions verbatim; use an "
    "empty string when a dish has none. Use menu_notes for anything ambiguous, including "
    "sections you could not read and had to skip."
)

INSTRUCTION = (
    "Transcribe every item on this restaurant menu into the structured schema. "
    "Keep the menu's own section names. Record anything ambiguous in menu_notes. "
    "Return every orderable dish and drink that is printed, and leave option blocks "
    "and modifier lines out of items -- note them in menu_notes instead."
)


def _build_llm():
    """Factory isolated to make mocking trivial in tests."""
    return ChatAnthropic(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
        # Thinking OFF -- and this node is the deliberate exception. The other
        # LLM nodes (decompose_recipes, canonicalize, forecast, worker) all keep
        # thinking ON, because disabling it there collapsed their emitted
        # confidences into a lazy 0.50 default. Three things make this node
        # different:
        #  1. Extraction is transcription, not judgement: menu text -> schema.
        #     There is nothing here worth reasoning about.
        #  2. It is one large single call, not a batched loop. `max_tokens` caps
        #     thinking AND output together, so the reasoning ate the budget the
        #     JSON needed and the run died with "Output parser received a
        #     `max_tokens` stop reason" -- fonda-park-slope and junoon both
        #     failed exactly this way against the old 16k ceiling.
        #     (`llm_max_tokens` is now 32000, see `src/config.py`.)
        #  3. It emits no confidence, so nothing it returns drives the review
        #     gate. The lazy-default failure mode simply does not apply.
        thinking={"type": "disabled"},
    ).with_structured_output(MenuExtraction)


def extract_menu_node(state: AgentState) -> dict:
    """Read the snapshot for state['restaurant'] and write menu_items."""
    restaurant = state.get("restaurant") or {}
    if not restaurant:
        # Paying for an Opus call that cannot succeed is the cheapest bug to prevent.
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": "ingest did not populate restaurant",
            "menu_items": [],
        }

    llm = _build_llm()

    msgs = [SystemMessage(SYSTEM_PROMPT)]
    if state.get("last_error"):
        msgs.append(HumanMessage(
            f"Previous attempt invalid: {state['last_error']}. "
            f"Correct it and try again."
        ))

    try:
        kind, payload = load_menu_source(restaurant)

        if kind == "image_b64":
            # Image block first, then the text instruction: documented ordering,
            # and it measurably matters for transcription quality.
            msgs.append(HumanMessage(content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": payload,
                    },
                },
                {"type": "text", "text": INSTRUCTION},
            ]))
        else:
            msgs.append(HumanMessage(f"{INSTRUCTION}\n\n---\n\n{payload}"))

        result = llm.invoke(msgs)
        return {
            "menu_items": [i.model_dump() for i in result.items],
            "stage": "menu_extracted",
            "last_error": "",
        }
    except Exception as e:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "last_error": f"{type(e).__name__}: {e}",
            "menu_items": [],
        }
