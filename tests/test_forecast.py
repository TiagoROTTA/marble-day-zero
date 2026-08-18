"""Cold-start forecast node.

The deterministic half (covers) is tested with no model call at all — that is
the point of splitting it out. The LLM half is tested through a stub that
returns deliberately unnormalised shares, so the normalisation is proved to
happen in Python rather than be trusted to the model.
"""
import pytest

from src.nodes import forecast as forecast_module
from src.nodes.forecast import (
    CORPUS_MEDIAN_REVIEW_COUNT,
    ITEMS_PER_COVER,
    REVIEW_CORRECTION_MAX,
    REVIEW_CORRECTION_MIN,
    TURNS_PER_DAY,
    UTILISATION,
    ItemMix,
    ItemShare,
    _covers_per_day,
    _review_correction,
    forecast_node,
)
from src.state import initial_dayzero_state

# A median-reviewed restaurant, so the review correction is exactly 1.0 and the
# arithmetic under test is not entangled with it.
RESTAURANT = {
    "slug": "test-slug",
    "name": "Test Pizza",
    "cuisine": "italian-pizza",
    "neighborhood": "West Village",
    "price_tier": "$",
    "seats": 20,
    "service_style": "counter",
    "popular_times_index": [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9],
    "review_count": CORPUS_MEDIAN_REVIEW_COUNT,
    "menu_format": "image",
}

MENU_ITEMS = [
    {"name": "Cheese Slice", "section": "Slices", "price": 3.75,
     "description": "Plain cheese", "confidence": 0.95},
    {"name": "Sicilian Slice", "section": "Slices", "price": 4.5,
     "description": "", "confidence": 0.9},
    {"name": "Fountain Soda", "section": "Drinks", "price": 2.0,
     "description": "", "confidence": 0.9},
]

# Deliberately unnormalised: these sum to 2.3, while a counter's target
# items-per-cover is 1.2. The node must rescale them.
UNNORMALISED_MIX = ItemMix(
    shares=[
        ItemShare(item_name="Cheese Slice", share=1.0, reasoning="the signature item"),
        ItemShare(item_name="Sicilian Slice", share=0.3, reasoning="secondary slice"),
        ItemShare(item_name="Fountain Soda", share=1.0, reasoning="attaches to most orders"),
    ],
    assumptions=["Assumed takeaway dominates dine-in at a slice counter."],
    confidence=0.7,
)


class FakeSucceedingLLM:
    def __init__(self, result=UNNORMALISED_MIX):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


class FakeFailingLLM:
    def invoke(self, messages):
        raise ValueError("simulated truncated JSON payload")


def _state(**overrides):
    state = initial_dayzero_state("test-slug")
    state["restaurant"] = dict(RESTAURANT)
    state["menu_items"] = [dict(i) for i in MENU_ITEMS]
    state.update(overrides)
    return state


# --------------------------------------------------------------------------
# Deterministic half — no model call anywhere below.
# --------------------------------------------------------------------------


def test_covers_per_day_returns_seven_floats_whose_mean_is_the_base_estimate():
    covers = _covers_per_day(dict(RESTAURANT))

    assert len(covers) == 7
    assert all(isinstance(c, float) for c in covers)

    base = RESTAURANT["seats"] * TURNS_PER_DAY["counter"] * UTILISATION["$"]
    assert sum(covers) / 7 == pytest.approx(base)


def test_flat_popular_times_index_produces_seven_equal_days():
    restaurant = dict(RESTAURANT, popular_times_index=[1.0] * 7)

    covers = _covers_per_day(restaurant)

    assert covers == pytest.approx([covers[0]] * 7)


def test_busy_days_keep_the_shape_of_the_popular_times_index():
    restaurant = dict(RESTAURANT, popular_times_index=[0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    covers = _covers_per_day(restaurant)

    # Saturday is twice Monday, and the mean is still the base estimate.
    assert covers[5] == pytest.approx(2 * covers[0])
    base = RESTAURANT["seats"] * TURNS_PER_DAY["counter"] * UTILISATION["$"]
    assert sum(covers) / 7 == pytest.approx(base)


def test_hyphenated_service_style_from_meta_json_matches_the_constant_dict():
    # meta.json writes "full-service"; TURNS_PER_DAY keys on "full_service".
    hyphenated = _covers_per_day(
        dict(RESTAURANT, service_style="full-service", popular_times_index=[1.0] * 7)
    )
    underscored = _covers_per_day(
        dict(RESTAURANT, service_style="full_service", popular_times_index=[1.0] * 7)
    )

    assert hyphenated == pytest.approx(underscored)
    assert hyphenated[0] == pytest.approx(
        RESTAURANT["seats"] * TURNS_PER_DAY["full_service"] * UTILISATION["$"]
    )


def test_malformed_popular_times_index_raises():
    with pytest.raises(ValueError):
        _covers_per_day(dict(RESTAURANT, popular_times_index=[1.0, 1.0]))

    with pytest.raises(ValueError):
        _covers_per_day(dict(RESTAURANT, popular_times_index=[0.0] * 7))


def test_review_correction_is_one_at_the_corpus_median():
    assert _review_correction(CORPUS_MEDIAN_REVIEW_COUNT) == pytest.approx(1.0)


def test_review_correction_clamps_at_both_ends():
    # A brand-new room with a handful of reviews must not zero the estimate out.
    assert _review_correction(0) == pytest.approx(REVIEW_CORRECTION_MIN)
    assert _review_correction(3) == pytest.approx(REVIEW_CORRECTION_MIN)
    # No review count, however large, may produce an absurd multiplier.
    assert _review_correction(1_000_000) == pytest.approx(REVIEW_CORRECTION_MAX)
    assert _review_correction(10_000_000) == pytest.approx(REVIEW_CORRECTION_MAX)
    # And a 20,000-review tourist landmark is already well inside the ceiling,
    # which is the log's doing rather than the clamp's.
    assert _review_correction(20_000) < REVIEW_CORRECTION_MAX


def test_review_correction_is_monotonic_between_the_clamps():
    assert _review_correction(500) < _review_correction(1500) < _review_correction(4000)


# --------------------------------------------------------------------------
# LLM half.
# --------------------------------------------------------------------------


def test_shares_are_normalised_to_the_target_items_per_cover(monkeypatch):
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM())

    forecast = forecast_node(_state())["demand_forecast"]

    target = ITEMS_PER_COVER["counter"]
    assert sum(forecast["item_mix"].values()) == pytest.approx(target)
    # Rescaling only: the ratio between two items is untouched.
    assert forecast["item_mix"]["Cheese Slice"] == pytest.approx(1.0 * target / 2.3)
    assert forecast["item_mix"]["Cheese Slice"] == pytest.approx(
        forecast["item_mix"]["Sicilian Slice"] / 0.3
    )


def test_happy_path_writes_the_whole_forecast_and_stage(monkeypatch):
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM())

    update = forecast_node(_state())
    forecast = update["demand_forecast"]

    assert update["stage"] == "forecast"
    assert update["last_error"] == ""
    assert "retry_count" not in update
    assert len(forecast["covers_per_day"]) == 7
    assert forecast["covers_per_week"] == pytest.approx(sum(_covers_per_day(RESTAURANT)), abs=0.5)
    assert forecast["assumptions"] == UNNORMALISED_MIX.assumptions
    assert forecast["confidence"] == 0.7
    assert forecast["method"] == "cold_start_prior"


def test_prompt_carries_the_menu_and_the_items_per_cover_target(monkeypatch):
    fake = FakeSucceedingLLM()
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: fake)

    forecast_node(_state())

    prompt = fake.messages[-1].content
    assert "Cheese Slice" in prompt
    assert "$3.75" in prompt
    assert "Slices" in prompt
    assert "West Village" in prompt
    assert str(ITEMS_PER_COVER["counter"]) in prompt


def test_empty_assumptions_is_an_error_not_a_silent_pass(monkeypatch):
    naked = ItemMix(
        shares=[ItemShare(item_name="Cheese Slice", share=1.0, reasoning="signature")],
        assumptions=[],
        confidence=0.9,
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(naked))

    update = forecast_node(_state())

    assert update["retry_count"] == 1
    assert "assumptions" in update["last_error"]
    assert update["demand_forecast"] == {}


def test_all_zero_shares_is_an_error_not_a_division_by_zero(monkeypatch):
    zeroed = ItemMix(
        shares=[ItemShare(item_name="Cheese Slice", share=0.0, reasoning="none")],
        assumptions=["Assumed nothing sells."],
        confidence=0.1,
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(zeroed))

    update = forecast_node(_state())

    assert update["retry_count"] == 1
    assert update["demand_forecast"] == {}


def test_exception_increments_retry_and_empties_the_forecast(monkeypatch):
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeFailingLLM())

    update = forecast_node(_state())

    assert update["retry_count"] == 1
    assert "simulated truncated JSON payload" in update["last_error"]
    assert update["demand_forecast"] == {}
    assert "stage" not in update


def test_last_error_is_fed_back_on_retry(monkeypatch):
    fake = FakeSucceedingLLM()
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: fake)

    forecast_node(_state(last_error="ValidationError: truncated"))

    assert "truncated" in fake.messages[1].content


def test_missing_menu_items_guard_skips_the_llm(monkeypatch):
    def explode():
        raise AssertionError("_build_llm must not be called without a menu")

    monkeypatch.setattr(forecast_module, "_build_llm", explode)

    update = forecast_node(_state(menu_items=[]))

    assert update["retry_count"] == 1
    assert update["demand_forecast"] == {}


def test_missing_restaurant_guard_skips_the_llm(monkeypatch):
    def explode():
        raise AssertionError("_build_llm must not be called without a restaurant")

    monkeypatch.setattr(forecast_module, "_build_llm", explode)

    update = forecast_node(initial_dayzero_state("test-slug"))

    assert update["retry_count"] == 1
    assert update["last_error"] == "forecast needs both restaurant and menu_items"
    assert update["demand_forecast"] == {}


# --------------------------------------------------------------------------
# Item-name resolution. Joe's Pizza drafted a $0.00 purchase order because the
# model returned "Classic Cheese Pie" for a menu line printed "Classic Cheese
# Pie 8 slices", and draft_po's exact recipe lookup missed all three pies.
# --------------------------------------------------------------------------

JOES_MENU = [
    {"name": "Classic Cheese Pie 8 slices", "section": "Pies", "price": 27.0},
    {"name": "Fresh Mozzarella Pie 8 Slices", "section": "Pies", "price": 32.0},
    {"name": "Sicilian Square Pie 8 Slices", "section": "Pies", "price": 33.0},
    {"name": "Sodas, Snapple, Stewarts", "section": "Drinks", "price": 3.0},
]


def _forecast_with(shares, menu_items=None, **overrides):
    """Run the node against a stubbed mix and return the demand_forecast."""
    mix = ItemMix(
        shares=[ItemShare(item_name=n, share=s, reasoning="stub") for n, s in shares],
        assumptions=["Assumed something worth saying."],
        confidence=0.7,
    )
    state = _state(**overrides)
    if menu_items is not None:
        state["menu_items"] = [dict(i) for i in menu_items]
    return mix, state


def test_shortened_item_name_resolves_back_onto_the_printed_menu_name(monkeypatch):
    mix, state = _forecast_with(
        [("Classic Cheese Pie", 1.0), ("Sodas, Snapple, Stewarts", 0.5)],
        menu_items=JOES_MENU,
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    # The key draft_po will look the recipe up by, not the model's paraphrase.
    assert "Classic Cheese Pie 8 slices" in forecast["item_mix"]
    assert "Classic Cheese Pie" not in forecast["item_mix"]
    assert "Sodas, Snapple, Stewarts" in forecast["item_mix"]


def test_exact_menu_name_is_passed_through_untouched(monkeypatch):
    mix, state = _forecast_with(
        [("Fresh Mozzarella Pie 8 Slices", 1.0)], menu_items=JOES_MENU
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    assert list(forecast["item_mix"]) == ["Fresh Mozzarella Pie 8 Slices"]


def test_casing_and_whitespace_differences_still_resolve(monkeypatch):
    mix, state = _forecast_with(
        [("  fresh mozzarella pie 8 slices.  ", 1.0)], menu_items=JOES_MENU
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    assert list(forecast["item_mix"]) == ["Fresh Mozzarella Pie 8 Slices"]


def test_ambiguous_name_refuses_to_join_and_is_reported(monkeypatch):
    # "Pie" is contained in three menu lines. Two candidates is already ambiguity,
    # and this project refuses rather than guesses.
    mix, state = _forecast_with([("Pie", 1.0)], menu_items=JOES_MENU)
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    assert list(forecast["item_mix"]) == ["Pie"]
    assert any("Pie" in a and "matched no menu item" in a for a in forecast["assumptions"])


def test_unmatchable_name_keeps_the_models_wording_and_is_reported(monkeypatch):
    mix, state = _forecast_with(
        [("Classic Cheese Pie", 1.0), ("Garlic Knots", 0.4)], menu_items=JOES_MENU
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    assert "Garlic Knots" in forecast["item_mix"]
    assert any("Garlic Knots" in a for a in forecast["assumptions"])


def test_two_model_names_landing_on_one_menu_item_have_their_shares_summed(monkeypatch):
    mix, state = _forecast_with(
        [("Classic Cheese Pie", 0.6), ("Classic Cheese Pie 8 slices", 0.4)],
        menu_items=JOES_MENU,
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    # One key, and the whole target items-per-cover on it: nothing was lost to
    # the merge, which a dict overwrite would have done silently.
    assert list(forecast["item_mix"]) == ["Classic Cheese Pie 8 slices"]
    assert sum(forecast["item_mix"].values()) == pytest.approx(ITEMS_PER_COVER["counter"])


def test_resolution_does_not_disturb_the_assumptions_when_everything_joins(monkeypatch):
    mix, state = _forecast_with(
        [("Classic Cheese Pie", 1.0)], menu_items=JOES_MENU
    )
    monkeypatch.setattr(forecast_module, "_build_llm", lambda: FakeSucceedingLLM(mix))

    forecast = forecast_node(state)["demand_forecast"]

    assert forecast["assumptions"] == ["Assumed something worth saying."]


def test_item_name_field_tells_the_model_to_copy_the_name_exactly():
    # Defence in depth for the same bug: a prompt can drift, the join above cannot.
    description = ItemShare.model_fields["item_name"].description or ""
    assert "exactly" in description.lower()
