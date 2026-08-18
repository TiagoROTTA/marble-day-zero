"""Canonicalization: the cheap passes must run first, and the LLM pass must never
be trusted blindly.

The important test here is `test_exact_alias_resolves_without_any_llm_call`:
`_build_llm` is stubbed to RAISE, so the node can only pass if pass 1 resolved
the name without ever reaching the model. Same trick for the normalised pass.
"""
import json

import pytest

from src.nodes import canonicalize as canon
from src.nodes.canonicalize import SkuMatch, SkuMatchBatch, _norm
from src.state import initial_dayzero_state

# Same shape as data/catalog/skus.json (extra pricing keys omitted: this node
# only ever reads sku_id / display_name / aliases / category).
CATALOG = [
    {
        "sku_id": "PROD-TOMATO-ROMA",
        "display_name": "Roma tomato",
        "aliases": ["roma tomatoes", "plum tomato", "tomatoes, roma"],
        "category": "produce",
    },
    {
        "sku_id": "DAIR-MOZZ-FRESHBALL",
        "display_name": "Fresh mozzarella ball",
        "aliases": ["mozzarella", "fior di latte", "mozzarella cheese"],
        "category": "dairy",
    },
]


class ExplodingLLM:
    """Any call to this is a bug: the cheap passes should have resolved everything."""

    def invoke(self, messages):
        raise AssertionError("LLM was called for a name the cheap passes should have matched")


class FakeLLM:
    def __init__(self, batch: SkuMatchBatch):
        self._batch = batch
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self._batch


@pytest.fixture
def catalog_path(tmp_path, monkeypatch):
    path = tmp_path / "skus.json"
    path.write_text(json.dumps(CATALOG), encoding="utf-8")
    monkeypatch.setattr(canon, "CATALOG_PATH", str(path))
    return path


def _state_with(*raw_names: str):
    state = initial_dayzero_state("test-slug")
    state["recipes"] = [{
        "item_name": "Margherita",
        "yield_qty": 1.0,
        "yield_uom": "serving",
        "components": [
            {"raw_name": n, "qty": 1.0, "uom": "oz", "confidence": 0.9} for n in raw_names
        ],
        "confidence": 0.9,
    }]
    return state


def test_norm_collapses_adjective_order_and_plurals():
    assert _norm("Fresh Roma Tomatoes") == _norm("tomatoes, roma")
    # and the catalog's singular display_name lands on the same key
    assert _norm("Roma tomato") == _norm("tomatoes, roma")


def test_exact_alias_resolves_without_any_llm_call(catalog_path, monkeypatch):
    monkeypatch.setattr(canon, "_build_llm", lambda: ExplodingLLM())
    state = _state_with("roma tomatoes")

    update = canon.canonicalize_node(state)

    assert update["stage"] == "canonicalized"
    assert update["last_error"] == ""
    assert update["sku_matches"] == [{
        "raw_name": "roma tomatoes",
        "sku_id": "PROD-TOMATO-ROMA",
        "method": "alias",
        "confidence": 1.0,
    }]
    assert update["review_queue"] == []


def test_normalized_match_also_avoids_the_llm(catalog_path, monkeypatch):
    monkeypatch.setattr(canon, "_build_llm", lambda: ExplodingLLM())
    state = _state_with("Fresh Roma Tomatoes")

    update = canon.canonicalize_node(state)

    match = update["sku_matches"][0]
    assert match["sku_id"] == "PROD-TOMATO-ROMA"
    assert match["method"] == "normalized"
    assert match["confidence"] == 0.9
    assert update["review_queue"] == []


def test_repeated_raw_name_is_resolved_once(catalog_path, monkeypatch):
    monkeypatch.setattr(canon, "_build_llm", lambda: ExplodingLLM())
    state = _state_with("mozzarella", "roma tomatoes", "mozzarella")

    update = canon.canonicalize_node(state)

    assert [m["raw_name"] for m in update["sku_matches"]] == ["mozzarella", "roma tomatoes"]


def test_casing_duplicate_costs_one_llm_lookup_and_one_question(catalog_path, monkeypatch):
    """'roasted peanuts' / 'Roasted peanuts' are one ingredient, one question."""
    batch = SkuMatchBatch(matches=[SkuMatch(
        raw_name="roasted peanuts",
        sku_id=None,
        confidence=0.0,
        reasoning="catalog carries no peanuts",
    )])
    fake = FakeLLM(batch)
    monkeypatch.setattr(canon, "_build_llm", lambda: fake)
    state = _state_with("roasted peanuts", "Roasted peanuts")

    update = canon.canonicalize_node(state)

    assert fake.calls == 1
    # One question, not two.
    assert [i["ref"] for i in update["review_queue"]] == ["roasted peanuts"]
    # But both spellings still carry the answer.
    assert [m["raw_name"] for m in update["sku_matches"]] == ["roasted peanuts", "Roasted peanuts"]
    assert all(m["sku_id"] is None and m["method"] == "llm" for m in update["sku_matches"])


def test_cheap_pass_answer_reaches_every_spelling(catalog_path, monkeypatch):
    """A group resolved by pass 1 fans its sku_id out to all its spellings."""
    monkeypatch.setattr(canon, "_build_llm", lambda: ExplodingLLM())
    state = _state_with("roma tomatoes", "Roma Tomatoes", "Fresh roma tomatoes")

    update = canon.canonicalize_node(state)

    assert [m["raw_name"] for m in update["sku_matches"]] == [
        "roma tomatoes", "Roma Tomatoes", "Fresh roma tomatoes",
    ]
    assert all(m["sku_id"] == "PROD-TOMATO-ROMA" for m in update["sku_matches"])


def test_every_raw_name_survives_into_sku_matches(catalog_path, monkeypatch):
    """cost_plates looks matches up by raw_name: a missing key drops an
    ingredient from every plate that uses it, silently. Deduplication is of the
    work, never of the rows."""
    batch = SkuMatchBatch(matches=[
        SkuMatch(raw_name="nuoc mam", sku_id=None, confidence=0.0, reasoning="no fish sauce"),
        SkuMatch(raw_name="lump crab meat", sku_id=None, confidence=0.0, reasoning="no crab"),
    ])
    monkeypatch.setattr(canon, "_build_llm", lambda: FakeLLM(batch))
    raw_names = [
        "roma tomatoes", "Roma Tomatoes", "roma tomatoes",      # dupes, pass 1
        "Fresh Roma Tomatoes",                                  # same group again
        "mozzarella", "Mozzarella",                             # dupes, pass 1
        "nuoc mam", "Nuoc Mam",                                 # dupes, pass 3
        "lump crab meat", "canned lump crab meat",              # distinct, pass 3
    ]
    state = _state_with(*raw_names)

    update = canon.canonicalize_node(state)

    resolved = {m["raw_name"] for m in update["sku_matches"]}
    assert resolved == set(raw_names)
    # One row per distinct raw string, no duplicates.
    assert len(update["sku_matches"]) == len(set(raw_names))
    # "canned lump crab meat" is a different ingredient: not merged away.
    assert _norm("lump crab meat") != _norm("canned lump crab meat")
    assert {i["ref"] for i in update["review_queue"]} == {"nuoc mam", "lump crab meat", "canned lump crab meat"}


def test_stopword_only_names_do_not_merge(catalog_path, monkeypatch):
    """Names that normalise to nothing are kept apart rather than piled into one
    group by an empty key."""
    batch = SkuMatchBatch(matches=[
        SkuMatch(raw_name="fresh", sku_id=None, confidence=0.0, reasoning="not an ingredient"),
        SkuMatch(raw_name="whole", sku_id=None, confidence=0.0, reasoning="not an ingredient"),
    ])
    monkeypatch.setattr(canon, "_build_llm", lambda: FakeLLM(batch))
    state = _state_with("fresh", "whole")

    update = canon.canonicalize_node(state)

    assert [m["raw_name"] for m in update["sku_matches"]] == ["fresh", "whole"]


def test_hallucinated_sku_id_is_demoted_to_none(catalog_path, monkeypatch):
    batch = SkuMatchBatch(matches=[SkuMatch(
        raw_name="nuoc mam",
        sku_id="COND-FISHSAUCE-VN",  # not in the catalog
        confidence=0.95,
        reasoning="confidently wrong",
    )])
    monkeypatch.setattr(canon, "_build_llm", lambda: FakeLLM(batch))
    state = _state_with("nuoc mam")

    update = canon.canonicalize_node(state)

    match = update["sku_matches"][0]
    assert match["method"] == "llm"
    assert match["sku_id"] is None
    assert match["confidence"] == 0.0
    item = update["review_queue"][0]
    assert item["payload"]["suggested"] is None
    assert "COND-FISHSAUCE-VN" in item["payload"]["reasoning"]


def test_none_match_reaches_the_review_queue(catalog_path, monkeypatch):
    batch = SkuMatchBatch(matches=[SkuMatch(
        raw_name="nuoc mam",
        sku_id=None,
        confidence=0.0,
        reasoning="catalog carries no fish sauce",
    )])
    monkeypatch.setattr(canon, "_build_llm", lambda: FakeLLM(batch))
    state = _state_with("roma tomatoes", "nuoc mam")
    state["review_queue"] = [{"kind": "menu_item", "ref": "pre-existing"}]

    update = canon.canonicalize_node(state)

    assert len(update["sku_matches"]) == 2
    assert update["review_queue"][0]["ref"] == "pre-existing"  # preserved
    item = update["review_queue"][1]
    assert item == {
        "kind": "sku_match",
        "ref": "nuoc mam",
        "confidence": 0.0,
        "question": "*nuoc mam* → nothing in the catalog matches it",
        "detail": "catalog carries no fish sauce",
        "payload": {"suggested": None, "reasoning": "catalog carries no fish sauce"},
    }


def test_low_confidence_llm_match_reaches_the_review_queue(catalog_path, monkeypatch):
    batch = SkuMatchBatch(matches=[SkuMatch(
        raw_name="buffalo mozzarella di bufala",
        sku_id="DAIR-MOZZ-FRESHBALL",
        confidence=0.7,
        reasoning="closest dairy SKU, but not the same product",
    )])
    monkeypatch.setattr(canon, "_build_llm", lambda: FakeLLM(batch))
    state = _state_with("buffalo mozzarella di bufala")

    update = canon.canonicalize_node(state)

    assert update["sku_matches"][0]["sku_id"] == "DAIR-MOZZ-FRESHBALL"
    item = update["review_queue"][0]
    assert item["confidence"] == 0.7
    # A reviewer with two buttons can only confirm or refuse a named proposal,
    # so the entry has to name the SKU it landed on and say why.
    assert item["question"] == (
        "*buffalo mozzarella di bufala* → Fresh mozzarella ball  `DAIR-MOZZ-FRESHBALL`"
    )
    assert item["detail"] == "closest dairy SKU, but not the same product"


def test_confident_llm_match_skips_the_review_queue(catalog_path, monkeypatch):
    batch = SkuMatchBatch(matches=[SkuMatch(
        raw_name="fior di latte, hand pulled",
        sku_id="DAIR-MOZZ-FRESHBALL",
        confidence=0.95,
        reasoning="same product",
    )])
    monkeypatch.setattr(canon, "_build_llm", lambda: FakeLLM(batch))
    state = _state_with("fior di latte, hand pulled")

    update = canon.canonicalize_node(state)

    assert update["sku_matches"][0]["confidence"] == 0.95
    assert update["review_queue"] == []


def test_llm_is_called_once_for_all_survivors(catalog_path, monkeypatch):
    batch = SkuMatchBatch(matches=[
        SkuMatch(raw_name="nuoc mam", sku_id=None, confidence=0.0, reasoning="no fish sauce"),
        SkuMatch(raw_name="gochujang", sku_id=None, confidence=0.0, reasoning="no korean paste"),
    ])
    fake = FakeLLM(batch)
    monkeypatch.setattr(canon, "_build_llm", lambda: fake)
    state = _state_with("roma tomatoes", "nuoc mam", "gochujang")

    update = canon.canonicalize_node(state)

    assert fake.calls == 1
    assert len(update["sku_matches"]) == 3
    assert len(update["review_queue"]) == 2


def test_missing_catalog_takes_the_failure_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(canon, "CATALOG_PATH", str(tmp_path / "nope.json"))
    state = _state_with("roma tomatoes")

    update = canon.canonicalize_node(state)

    assert update["retry_count"] == 1
    assert "nope.json" in update["last_error"]
    assert update["sku_matches"] == []


def test_real_catalog_resolves_common_names_without_the_llm(monkeypatch):
    """Guards against catalog shape drift: these must all hit passes 1 or 2."""
    monkeypatch.setattr(canon, "_build_llm", lambda: ExplodingLLM())
    state = _state_with(
        "roma tomatoes", "mozzarella", "olive oil", "yellow onion", "garlic", "kosher salt",
    )

    update = canon.canonicalize_node(state)

    assert update["stage"] == "canonicalized"
    assert all(m["sku_id"] for m in update["sku_matches"])
    assert all(m["method"] in ("alias", "normalized") for m in update["sku_matches"])
