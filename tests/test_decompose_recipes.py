"""Recipe decomposition: chunking, uom filtering, and partial-chunk survival.

The LLM is always stubbed via `_build_llm`; nothing here touches the network.
"""
import json

import pytest

from src.nodes import decompose_recipes as decompose_module
from src.nodes.decompose_recipes import (
    CHUNK_SIZE,
    VALID_UOMS,
    Component,
    Recipe,
    RecipeBatch,
    _chunk,
    _load_catalog_names,
    decompose_recipes_node,
)
from src.state import initial_dayzero_state


def _menu(n: int) -> list[dict]:
    return [
        {
            "name": f"Dish {i}",
            "section": "Mains",
            "price": 10.0 + i,
            "description": f"description {i}",
            "confidence": 0.9,
        }
        for i in range(n)
    ]


def _state(menu_items, **overrides):
    state = initial_dayzero_state("test-slug")
    state["menu_items"] = menu_items
    state.update(overrides)
    return state


def _recipe(item_name: str, components=None) -> Recipe:
    return Recipe(
        item_name=item_name,
        components=components
        or [Component(raw_name="olive oil", qty=0.5, uom="fl_oz", confidence=0.9)],
        confidence=0.9,
    )


class FakeLLM:
    """Returns one recipe per item in the chunk it was handed."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        names = _names_in(messages)
        return RecipeBatch(recipes=[_recipe(n) for n in names])


class FakeFailingLLM:
    def invoke(self, messages):
        raise ValueError("simulated truncated JSON payload")


class FakeSecondChunkFailsLLM:
    """Succeeds on the first chunk, blows up on every later one."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("simulated rate limit")
        return RecipeBatch(recipes=[_recipe(n) for n in _names_in(messages)])


def _names_in(messages) -> list[str]:
    """Pull the dish names back out of the JSON payload in the last message."""
    text = messages[-1].content
    return [item["name"] for item in json.loads(text[text.index("["):])]


# --- chunking -------------------------------------------------------------


def test_chunk_splits_into_groups_of_ten():
    assert [len(c) for c in _chunk(list(range(25)))] == [10, 10, 5]
    assert _chunk([]) == []
    assert [len(c) for c in _chunk(list(range(10)))] == [10]


def test_twentyfive_items_make_three_calls(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: fake)

    update = decompose_recipes_node(_state(_menu(25)))

    assert len(fake.calls) == 3
    assert CHUNK_SIZE == 10
    assert update["stage"] == "recipes_decomposed"
    assert update["last_error"] == ""
    assert len(update["recipes"]) == 25
    assert [r["item_name"] for r in update["recipes"]][:3] == ["Dish 0", "Dish 1", "Dish 2"]
    assert "retry_count" not in update


def test_recipes_are_plain_dicts_with_yield_defaults(monkeypatch):
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: FakeLLM())

    update = decompose_recipes_node(_state(_menu(1)))

    recipe = update["recipes"][0]
    assert recipe["yield_qty"] == 1.0
    assert recipe["yield_uom"] == "serving"
    assert recipe["components"][0] == {
        "raw_name": "olive oil",
        "qty": 0.5,
        "uom": "fl_oz",
        "confidence": 0.9,
    }


# --- cached catalog block -------------------------------------------------


def test_catalog_block_is_deterministic_and_cache_marked(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: fake)

    decompose_recipes_node(_state(_menu(25)))

    blocks = [msgs[0].content for msgs in fake.calls]
    # Byte-identical across chunks, or the cache is invalidated every call.
    assert blocks[0] == blocks[1] == blocks[2]
    catalog_block = blocks[0][-1]
    assert catalog_block["cache_control"] == {"type": "ephemeral"}
    assert "Prefer these ingredient names" in catalog_block["text"]
    # The instructions block carries no breakpoint of its own: one cache write.
    assert "cache_control" not in blocks[0][0]


def test_catalog_names_are_stable_across_loads():
    assert _load_catalog_names() == _load_catalog_names()
    assert len(_load_catalog_names()) > 50


# --- uom filtering --------------------------------------------------------


def test_invalid_uom_is_dropped_and_queued_for_review(monkeypatch):
    bad = Recipe(
        item_name="Dish 0",
        components=[
            Component(raw_name="mozzarella", qty=4.0, uom="oz", confidence=0.9),
            Component(raw_name="basil", qty=5.0, uom="g", confidence=0.5),
        ],
        confidence=0.8,
    )

    class OneBadComponentLLM:
        def invoke(self, messages):
            return RecipeBatch(recipes=[bad])

    monkeypatch.setattr(decompose_module, "_build_llm", lambda: OneBadComponentLLM())

    update = decompose_recipes_node(_state(_menu(1)))

    components = update["recipes"][0]["components"]
    assert [c["raw_name"] for c in components] == ["mozzarella"]
    assert all(c["uom"] in VALID_UOMS for c in components)

    queued = update["review_queue"]
    assert len(queued) == 1
    assert queued[0]["kind"] == "recipe_uom"
    assert queued[0]["ref"] == "Dish 0 / basil"
    assert queued[0]["payload"]["uom"] == "g"
    assert queued[0]["confidence"] == 0.5
    # Still a success: a dropped component is not a failed run.
    assert update["stage"] == "recipes_decomposed"
    assert "retry_count" not in update


def test_review_queue_is_appended_not_replaced(monkeypatch):
    class OneBadComponentLLM:
        def invoke(self, messages):
            return RecipeBatch(recipes=[Recipe(
                item_name="Dish 0",
                components=[Component(raw_name="basil", qty=5.0, uom="ml", confidence=0.5)],
                confidence=0.8,
            )])

    monkeypatch.setattr(decompose_module, "_build_llm", lambda: OneBadComponentLLM())
    existing = {"kind": "menu_item", "ref": "Whole Fish", "confidence": 0.4,
                "question": "market price?", "payload": {}}

    update = decompose_recipes_node(_state(_menu(1), review_queue=[existing]))

    assert update["review_queue"][0] == existing
    assert len(update["review_queue"]) == 2


# --- failure handling -----------------------------------------------------


def test_partial_chunk_failure_keeps_earlier_recipes(monkeypatch):
    fake = FakeSecondChunkFailsLLM()
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: fake)

    update = decompose_recipes_node(_state(_menu(25)))

    assert fake.calls == 3  # every chunk is attempted, not aborted on first error
    assert len(update["recipes"]) == 10
    assert [r["item_name"] for r in update["recipes"]] == [f"Dish {i}" for i in range(10)]
    # Failed chunk item names are named in last_error, and the breaker still trips.
    assert update["retry_count"] == 1
    assert "failed for 15 item(s)" in update["last_error"]
    assert "Dish 10" in update["last_error"]
    assert "Dish 24" in update["last_error"]
    assert "simulated rate limit" in update["last_error"]
    assert "stage" not in update


def test_total_failure_increments_retry_and_returns_no_recipes(monkeypatch):
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: FakeFailingLLM())

    update = decompose_recipes_node(_state(_menu(3)))

    assert update["recipes"] == []
    assert update["retry_count"] == 1
    assert "simulated truncated JSON payload" in update["last_error"]
    assert "stage" not in update


def test_retry_count_builds_on_existing_count(monkeypatch):
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: FakeFailingLLM())

    update = decompose_recipes_node(_state(_menu(3), retry_count=2))

    assert update["retry_count"] == 3


def test_last_error_is_fed_back_on_retry(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(decompose_module, "_build_llm", lambda: fake)

    decompose_recipes_node(_state(_menu(1), last_error="ValidationError: truncated"))

    feedback = fake.calls[0][1].content
    assert "truncated" in feedback


def test_empty_menu_items_skips_the_llm(monkeypatch):
    def explode():
        raise AssertionError("_build_llm must not be called without menu_items")

    monkeypatch.setattr(decompose_module, "_build_llm", explode)

    update = decompose_recipes_node(initial_dayzero_state("test-slug"))

    assert update["retry_count"] == 1
    assert update["recipes"] == []
    assert "menu_items" in update["last_error"]


def test_component_schema_rejects_nonpositive_qty():
    with pytest.raises(Exception):
        Component(raw_name="salt", qty=0.0, uom="oz", confidence=0.5)
