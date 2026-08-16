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
