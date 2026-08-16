"""Menu extraction node: happy path writes menu_items + stage, any exception
increments retry_count and leaves menu_items empty."""
from src.config import settings
from src.nodes import extract_menu as extract_menu_module
from src.nodes.extract_menu import (
    INSTRUCTION,
    SYSTEM_PROMPT,
    MenuExtraction,
    MenuItem,
    extract_menu_node,
)
from src.state import initial_dayzero_state

FIXED_EXTRACTION = MenuExtraction(
    items=[
        MenuItem(
            name="Cheese Slice",
            section="Slices",
            price=3.75,
            description="Plain cheese",
            confidence=0.95,
        ),
        MenuItem(
            name="Whole Fish",
            section="Specials",
            price=None,
            description="",
            confidence=0.4,
        ),
    ],
    menu_notes=["Whole Fish is market price"],
)

RESTAURANT = {
    "slug": "test-slug",
    "name": "Test Pizza",
    "menu_format": "html",
    "snapshot_path": "/nonexistent/source.html",
}


class FakeSucceedingLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FIXED_EXTRACTION


class FakeFailingLLM:
    def invoke(self, messages):
        raise ValueError("simulated truncated JSON payload")


def _state(**overrides):
    state = initial_dayzero_state("test-slug")
    state["restaurant"] = dict(RESTAURANT)
    state.update(overrides)
    return state


def test_happy_path_writes_menu_items_and_stage(monkeypatch):
    monkeypatch.setattr(extract_menu_module, "_build_llm", lambda: FakeSucceedingLLM())
    monkeypatch.setattr(
        extract_menu_module, "load_menu_source", lambda r: ("text", "Cheese Slice 3.75")
    )

    update = extract_menu_node(_state())

    assert update["stage"] == "menu_extracted"
    assert update["last_error"] == ""
    assert len(update["menu_items"]) == 2
    assert update["menu_items"][0] == {
        "name": "Cheese Slice",
        "section": "Slices",
        "price": 3.75,
        "description": "Plain cheese",
        "confidence": 0.95,
    }
    assert update["menu_items"][1]["price"] is None
    assert "retry_count" not in update


def test_image_snapshot_sends_image_block_before_text(monkeypatch):
    fake = FakeSucceedingLLM()
    monkeypatch.setattr(extract_menu_module, "_build_llm", lambda: fake)
    monkeypatch.setattr(
        extract_menu_module, "load_menu_source", lambda r: ("image_b64", "QUJD")
    )

    update = extract_menu_node(_state())

    assert update["stage"] == "menu_extracted"
    blocks = fake.messages[-1].content
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["data"] == "QUJD"
    assert blocks[1]["type"] == "text"


def test_exception_increments_retry_and_empties_menu_items(monkeypatch):
    monkeypatch.setattr(extract_menu_module, "_build_llm", lambda: FakeFailingLLM())
    monkeypatch.setattr(
        extract_menu_module, "load_menu_source", lambda r: ("text", "whatever")
    )

    update = extract_menu_node(_state())

    assert update["retry_count"] == 1
    assert "simulated truncated JSON payload" in update["last_error"]
    assert update["menu_items"] == []
    assert "stage" not in update


def test_last_error_is_fed_back_on_retry(monkeypatch):
    fake = FakeSucceedingLLM()
    monkeypatch.setattr(extract_menu_module, "_build_llm", lambda: fake)
    monkeypatch.setattr(
        extract_menu_module, "load_menu_source", lambda r: ("text", "Cheese Slice 3.75")
    )

    extract_menu_node(_state(last_error="ValidationError: truncated"))

    feedback = fake.messages[1].content
    assert "truncated" in feedback


def test_prompt_defines_an_item_and_demands_exhaustiveness():
    """The 16-vs-6 swing on joes-pizza-carmine was one undefined boundary:
    whether an unpriced 'Toppings' block counts as menu items. Naming the
    boundary is the fix, so assert the naming stays in the prompt."""
    lowered = SYSTEM_PROMPT.lower()

    # Exhaustive, not a highlights reel.
    assert "return every single one that is printed" in lowered
    assert "not a summary" in lowered

    # Option / ingredient blocks are not items, even under their own heading.
    assert "toppings" in lowered
    assert "choice of" in lowered
    assert "even when printed under their own bold heading" in lowered

    # Modifier lines are not items.
    assert "+add any one topping" in lowered
    assert "do not give it its own entry" in lowered

    # Page furniture is not an item.
    assert "opening hours" in lowered

    # What was dropped stays visible to the human.
    assert "menu_notes" in lowered


def test_prompt_still_keeps_unpriced_dishes_as_items():
    """Guard against over-correcting into 'only priced lines are items'.
    The corpus includes menus that print no prices at all; those must not
    come back empty just because the modifier rule was added."""
    lowered = SYSTEM_PROMPT.lower()

    assert "having a price is not what makes something an item" in lowered
    assert "price=null" in lowered
    assert "never whether a number sits next to it" in lowered


def test_instruction_repeats_the_item_boundary():
    lowered = INSTRUCTION.lower()

    assert "every orderable dish and drink" in lowered
    assert "modifier lines out of items" in lowered


def test_prompts_are_ascii_only():
    """These strings are safe to echo on the Windows cp1252 console."""
    SYSTEM_PROMPT.encode("ascii")
    INSTRUCTION.encode("ascii")


def test_build_llm_passes_the_token_ceiling_and_no_sampling_params(monkeypatch):
    """max_tokens must come from settings, and temperature / top_p / top_k /
    budget_tokens must never be sent: this API configuration 400s on them,
    so determinism has to come from the prompt instead."""
    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema):
            captured["schema"] = schema
            return self

    monkeypatch.setattr(extract_menu_module, "ChatAnthropic", FakeChatAnthropic)

    extract_menu_module._build_llm()

    assert captured["max_tokens"] == settings.llm_max_tokens
    assert captured["model"] == settings.llm_model
    assert captured["schema"] is MenuExtraction
    for banned in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in captured


def test_missing_restaurant_guard_skips_the_llm(monkeypatch):
    def explode():
        raise AssertionError("_build_llm must not be called without a restaurant")

    monkeypatch.setattr(extract_menu_module, "_build_llm", explode)
    state = initial_dayzero_state("test-slug")

    update = extract_menu_node(state)

    assert update["retry_count"] == 1
    assert update["last_error"] == "ingest did not populate restaurant"
    assert update["menu_items"] == []
