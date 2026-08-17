"""Plate costing: unit conversion, the join, the guarded divisions, the sane band.

Nothing here touches the network — this node makes no LLM call at all, which is
exactly why every number below is asserted against arithmetic done by hand in the
comments rather than against a recorded fixture.
"""
import json
import math

from src.nodes import cost_plates as cost_module
from src.nodes.cost_plates import (
    SANE_BAND_HIGH,
    SANE_BAND_LOW,
    _convert,
    _load_catalog,
    cost_plates_node,
)
from src.state import CONF_REVIEW_FLOOR, initial_dayzero_state

# Fixture catalog. Round prices so the expected costs can be checked mentally.
FIXTURE_SKUS = [
    {
        "sku_id": "DAIRY-MOZZ",
        "display_name": "Mozzarella",
        "aliases": ["mozzarella"],
        "category": "dairy",
        "price_per_uom": 3.20,
        "uom": "lb",
        "vendor": "Restaurant Depot",
    },
    {
        "sku_id": "PROD-TOMATO-ROMA",
        "display_name": "Roma tomato",
        "aliases": ["roma tomatoes"],
        "category": "produce",
        "price_per_uom": 1.30,
        "uom": "lb",
        "vendor": "Baldor Specialty Foods",
    },
    {
        "sku_id": "OIL-OLIVE-XV",
        "display_name": "Extra virgin olive oil",
        "aliases": ["olive oil"],
        "category": "oils",
        "price_per_uom": 25.60,
        "uom": "gal",
        "vendor": "Restaurant Depot",
    },
]


def _use_fixture_catalog(monkeypatch, tmp_path, skus=None):
    path = tmp_path / "skus.json"
    path.write_text(json.dumps(skus if skus is not None else FIXTURE_SKUS), encoding="utf-8")
    monkeypatch.setattr(cost_module, "CATALOG_PATH", str(path))
    return path


def _state(recipes, sku_matches, menu_items, **overrides):
    state = initial_dayzero_state("test-slug")
    state["recipes"] = recipes
    state["sku_matches"] = sku_matches
    state["menu_items"] = menu_items
    state.update(overrides)
    return state


def _component(raw_name, qty, uom, confidence=1.0):
    return {"raw_name": raw_name, "qty": qty, "uom": uom, "confidence": confidence}


def _recipe(item_name, components, confidence=1.0, yield_qty=1.0):
    return {
        "item_name": item_name,
        "yield_qty": yield_qty,
        "yield_uom": "serving",
        "components": components,
        "confidence": confidence,
    }


def _match(raw_name, sku_id, confidence=1.0, method="alias"):
    return {
        "raw_name": raw_name,
        "sku_id": sku_id,
        "method": method,
        "confidence": confidence,
    }


# --- _convert -------------------------------------------------------------


def test_convert_uses_the_real_units_table():
    assert _convert(16, "oz", "lb") == 1.0
    assert _convert(4, "oz", "lb") == 0.25
    assert _convert(1, "lb", "oz") == 16.0
    assert _convert(128, "fl_oz", "gal") == 1.0


def test_convert_identity_returns_qty_unchanged():
    assert _convert(2.5, "lb", "lb") == 2.5
    # Even a unit the table has never heard of converts to itself.
    assert _convert(3.0, "sprig", "sprig") == 3.0


def test_convert_returns_none_when_no_conversion_exists():
    # Weight to volume is not a conversion, it is a density lookup we do not have.
    assert _convert(1, "lb", "gal") is None
    assert _convert(1, "each", "lb") is None
    assert _convert(1, "bunch", "oz") is None


# --- the happy path -------------------------------------------------------


def test_two_ingredient_plate_costs_correctly(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 4 oz mozzarella  -> 4 * 0.0625 = 0.25 lb  * 3.20 = 0.8000
    # 6 oz roma tomato -> 6 * 0.0625 = 0.375 lb * 1.30 = 0.4875
    #                                              total = 1.2875
    state = _state(
        recipes=[_recipe("Margherita", [
            _component("mozzarella", 4.0, "oz"),
            _component("roma tomatoes", 6.0, "oz"),
        ])],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA"),
        ],
        menu_items=[{"name": "Margherita", "section": "Pizza", "price": 4.00,
                     "description": "", "confidence": 0.95}],
    )

    update = cost_plates_node(state)

    plate = update["plate_costs"][0]
    assert plate["item_name"] == "Margherita"
    assert plate["plate_cost"] == 1.2875
    assert plate["menu_price"] == 4.00
    assert plate["food_cost_pct"] == round(1.2875 / 4.00, 4) == 0.3219
    assert plate["costed_components"] == 2
    assert plate["total_components"] == 2
    assert plate["coverage"] == 1.0
    assert plate["uncosted"] == []
    assert plate["confidence"] == 1.0
    assert plate["costable"] is True
    assert update["stage"] == "costed"
    assert update["last_error"] == ""
    assert "retry_count" not in update
    # In the sane band, so nothing queued.
    assert update["review_queue"] == []


def test_yield_qty_does_not_divide_the_plate_cost(monkeypatch, tmp_path):
    """THE INVARIANT: plate_cost and menu_price describe the same sold unit.

    This assertion used to read `plate_cost == 3.20` — 25.60 / 8 — because the
    node divided by yield_qty. That was the bug: the recipe describes one SOLD
    UNIT (what menu_price buys) and yield_qty only REPORTS how many servings that
    unit feeds, so dividing left a one-serving numerator over a whole-unit
    denominator. The per-serving figure is still available, on cost_per_serving.
    """
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 1 gal olive oil at 25.60 is the whole sold unit; it happens to feed 8.
    state = _state(
        recipes=[_recipe("Aglio e Olio", [_component("olive oil", 1.0, "gal")], yield_qty=8.0)],
        sku_matches=[_match("olive oil", "OIL-OLIVE-XV")],
        menu_items=[{"name": "Aglio e Olio", "price": 12.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 25.60
    assert plate["cost_per_serving"] == 3.20
    assert plate["yield_qty"] == 8.0


def test_yield_qty_of_one_leaves_the_two_figures_equal(monkeypatch, tmp_path):
    """The safety property: 844 of the 879 cached recipes are yield_qty 1.0."""
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Aglio e Olio", [_component("olive oil", 1.0, "gal")])],
        sku_matches=[_match("olive oil", "OIL-OLIVE-XV")],
        menu_items=[{"name": "Aglio e Olio", "price": 64.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 25.60
    assert plate["cost_per_serving"] == 25.60
    assert plate["food_cost_pct"] == 0.4


def test_missing_yield_defaults_to_one_serving(monkeypatch, tmp_path):
    """yield_qty 0 or None must not divide by zero on cost_per_serving."""
    _use_fixture_catalog(monkeypatch, tmp_path)

    for bad_yield in (0.0, None):
        recipe = _recipe("Aglio e Olio", [_component("olive oil", 1.0, "gal")])
        recipe["yield_qty"] = bad_yield
        state = _state(
            recipes=[recipe],
            sku_matches=[_match("olive oil", "OIL-OLIVE-XV")],
            menu_items=[{"name": "Aglio e Olio", "price": 64.00}],
        )

        plate = cost_plates_node(state)["plate_costs"][0]

        assert plate["yield_qty"] == 1.0
        assert plate["cost_per_serving"] == 25.60


def test_uncostable_plate_has_no_per_serving_figure_either(monkeypatch, tmp_path):
    """None, not 0.0 — the same rule plate_cost follows, for the same reason."""
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Mystery", [_component("unobtainium", 1.0, "gal")], yield_qty=4.0)],
        sku_matches=[],
        menu_items=[{"name": "Mystery", "price": 12.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["costable"] is False
    assert plate["plate_cost"] is None
    assert plate["cost_per_serving"] is None


def test_joes_pizza_whole_pie_regression(monkeypatch, tmp_path):
    """The real shape that exposed the bug: a whole pie priced as a whole pie.

    joes-pizza-carmine's "Classic Cheese Pie 8 slices" carries yield_qty 8 and
    WHOLE-PIE components (14 oz flour, 12 oz mozzarella), against a $24.00 menu
    price that buys the whole pie. Dividing the cost by 8 reported 2.4% food
    cost and put a fake near-zero on the food-cost distribution chart; the honest
    ratio is ~19%.
    """
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Whole-pie quantities, priced off the fixture catalog to land on the real
    # run's totals:
    #   12 oz mozzarella  = 0.75 lb   x 3.20  = 2.40
    #    8 fl_oz olive oil = 0.0625 gal x 25.60 = 1.60
    #    8 oz roma tomato  = 0.5 lb    x 1.30  = 0.65
    #                                    total = 4.65   (the real run: 4.62)
    state = _state(
        recipes=[
            _recipe(
                "Classic Cheese Pie 8 slices",
                [
                    _component("mozzarella", 12.0, "oz"),
                    _component("olive oil", 8.0, "fl_oz"),
                    _component("roma tomatoes", 8.0, "oz"),
                ],
                yield_qty=8.0,
            )
        ],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("olive oil", "OIL-OLIVE-XV"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA"),
        ],
        menu_items=[{"name": "Classic Cheese Pie 8 slices", "price": 24.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 4.65
    # 4.65 / 24.00 = 19.4% — an ordinary pizzeria food cost, inside the band.
    assert plate["food_cost_pct"] == 0.1938
    assert SANE_BAND_LOW <= plate["food_cost_pct"] <= SANE_BAND_HIGH

    # The old divide-by-yield behaviour, reproduced explicitly so the regression
    # is unmistakable: it reported 4.65 / 8 = $0.58 against the $24.00 whole-pie
    # price, i.e. 2.4% food cost. That number is now on cost_per_serving, where
    # it is true, instead of on food_cost_pct, where it was a fake near-zero.
    assert plate["cost_per_serving"] == 0.5813
    assert plate["food_cost_pct"] != 0.0242
    assert round(plate["cost_per_serving"] / 24.00, 4) == 0.0242


def test_cross_unit_conversion_into_the_sku_uom(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 32 fl_oz of oil -> 32 * 0.0078125 = 0.25 gal * 25.60 = 6.40
    state = _state(
        recipes=[_recipe("Confit", [_component("olive oil", 32.0, "fl_oz")])],
        sku_matches=[_match("olive oil", "OIL-OLIVE-XV")],
        menu_items=[{"name": "Confit", "price": 20.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 6.40
    assert plate["uncosted"] == []


# --- guarded divisions ----------------------------------------------------


def test_menu_price_none_yields_none_pct(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Market Fish", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Market Fish", "price": None}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    assert plate["menu_price"] is None
    assert plate["food_cost_pct"] is None
    assert plate["plate_cost"] == 0.80
    # A missing price is not a costing failure and not a sane-band outlier.
    assert update["review_queue"] == []
    assert "retry_count" not in update


def test_menu_price_zero_yields_none_pct_not_zerodivision(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Staff Meal", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Staff Meal", "price": 0.0}],
    )

    update = cost_plates_node(state)

    assert update["plate_costs"][0]["food_cost_pct"] is None
    assert "retry_count" not in update


def test_missing_menu_item_leaves_price_none(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Off-Menu Special", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["menu_price"] is None
    assert plate["food_cost_pct"] is None


def test_menu_items_without_a_recipe_are_skipped_entirely(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Margherita", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[
            {"name": "Margherita", "price": 18.00},
            {"name": "Tiramisu", "price": 9.00},
        ],
    )

    update = cost_plates_node(state)

    # Not costed at zero, not present at all.
    assert [p["item_name"] for p in update["plate_costs"]] == ["Margherita"]


def test_recipe_with_no_components_has_zero_coverage(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Chef's Special", [], confidence=0.4)],
        sku_matches=[],
        menu_items=[{"name": "Chef's Special", "price": 25.00}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    assert plate["total_components"] == 0
    assert plate["coverage"] == 0.0
    # An empty recipe has no costable component either, so it is uncostable for
    # the same reason an all-excluded one is: there is nothing to add up.
    assert plate["costable"] is False
    assert plate["plate_cost"] is None
    assert plate["food_cost_pct"] is None
    assert plate["confidence"] == 0.0
    assert "retry_count" not in update


# --- uncosted components --------------------------------------------------


def test_unmatched_component_is_uncosted_with_a_reason(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Bun Cha", [
            _component("mozzarella", 4.0, "oz"),
            _component("fish sauce", 2.0, "fl_oz"),   # matched to nothing
            _component("shiso leaf", 1.0, "bunch"),   # no match record at all
        ])],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("fish sauce", None, confidence=0.0, method="llm"),
        ],
        menu_items=[{"name": "Bun Cha", "price": 18.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["costed_components"] == 1
    assert plate["total_components"] == 3
    assert plate["coverage"] == round(1 / 3, 4)
    assert len(plate["uncosted"]) == 2
    assert "fish sauce" in plate["uncosted"][0]
    assert "shiso leaf" in plate["uncosted"][1]


def test_unconvertible_unit_is_uncosted_never_assumed_one(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # "each" mozzarella cannot become lb: there is no conversion, so the line is
    # a visible gap rather than 1 lb of cheese quietly priced at 3.20.
    state = _state(
        recipes=[_recipe("Caprese", [_component("mozzarella", 1.0, "each")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Caprese", "price": 14.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    # Nothing survived the join, so there is no cost to report — not 0.0.
    assert plate["plate_cost"] is None
    assert plate["costable"] is False
    assert plate["costed_components"] == 0
    assert plate["coverage"] == 0.0
    assert "no conversion from each to lb" in plate["uncosted"][0]


def test_hallucinated_sku_id_is_uncosted(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    state = _state(
        recipes=[_recipe("Mystery", [_component("nduja", 2.0, "oz")])],
        sku_matches=[_match("nduja", "MEAT-NDUJA-INVENTED", confidence=0.9, method="llm")],
        menu_items=[{"name": "Mystery", "price": 16.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["costed_components"] == 0
    assert "not in the catalog" in plate["uncosted"][0]


# --- below-floor components are not costed --------------------------------
#
# `src/nodes/draft_po.py` withholds a component whose own confidence is below
# CONF_REVIEW_FLOOR from the purchase order. These tests pin the matching
# behaviour here, because a plate cost computed from ingredients the order does
# not buy is a number nobody can trace. The exclusion deliberately runs through
# `coverage`, so the under-costing shows up as lower plate confidence.


def test_below_floor_component_is_costed_and_recorded_as_low_confidence(
    monkeypatch, tmp_path
):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Both components are costed: mozzarella 4 oz -> 0.25 lb * 3.20 = 0.80, and
    # the 0.40-confidence tomato 6 oz -> 0.375 lb * 1.30 = 0.4875.
    #
    # This used to drop the tomato and report 0.80. That is the bias this test
    # now guards against: the hedged component is disproportionately the
    # expensive one, so deleting it drags food cost down every time. It is
    # costed and recorded instead.
    state = _state(
        recipes=[_recipe("Hallucinated Caprese", [
            _component("mozzarella", 4.0, "oz", confidence=0.95),
            _component("roma tomatoes", 6.0, "oz", confidence=0.40),
        ])],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA"),
        ],
        menu_items=[{"name": "Hallucinated Caprese", "price": 8.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 1.2875
    assert plate["costed_components"] == 2
    assert plate["total_components"] == 2
    # `uncosted` is for genuine gaps only. A hedged quantity is not a gap.
    assert plate["uncosted"] == []
    assert len(plate["low_confidence"]) == 1
    # The reason names the ingredient, its confidence and the floor.
    assert "roma tomatoes" in plate["low_confidence"][0]
    assert "0.40" in plate["low_confidence"][0]
    assert "0.60" in plate["low_confidence"][0]


def test_below_floor_component_drags_confidence_down_but_not_the_cost(
    monkeypatch, tmp_path
):
    _use_fixture_catalog(monkeypatch, tmp_path)

    def build(tomato_conf):
        return _state(
            recipes=[_recipe("Caprese", [
                _component("mozzarella", 4.0, "oz", confidence=0.95),
                _component("roma tomatoes", 6.0, "oz", confidence=tomato_conf),
            ], confidence=0.9)],
            sku_matches=[
                _match("mozzarella", "DAIRY-MOZZ"),
                _match("roma tomatoes", "PROD-TOMATO-ROMA"),
            ],
            menu_items=[{"name": "Caprese", "price": 8.00}],
        )

    confident = cost_plates_node(build(0.95))["plate_costs"][0]
    guessed = cost_plates_node(build(0.40))["plate_costs"][0]

    # Each component contributes match_conf * quantity_conf to the mean, so
    # identity and quantity both have to be sound for a line to read as certain.
    # confident: mean (1.0*0.95, 1.0*0.95) = 0.95 -> sqrt(0.9 * 0.95) * 1.0
    assert confident["coverage"] == 1.0
    assert confident["confidence"] == round(math.sqrt(0.9 * 0.95) * 1.0, 4)
    # guessed: mean (1.0*0.95, 1.0*0.40) = 0.675 -> sqrt(0.9 * 0.675) * 1.0
    assert guessed["coverage"] == 1.0
    assert guessed["confidence"] == round(math.sqrt(0.9 * 0.675) * 1.0, 4)

    # The uncertainty now lands where it belongs: the plate says it is less sure,
    # rather than quietly costing less. Coverage and cost are unchanged, because
    # nothing was actually missing.
    assert guessed["confidence"] < confident["confidence"]
    assert guessed["coverage"] == confident["coverage"]
    assert guessed["plate_cost"] == confident["plate_cost"]


def test_confidence_exactly_at_the_floor_is_still_costed(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # CONF_REVIEW_FLOOR is a floor, not a bar to clear: the predicate is `<`,
    # matching draft_po.py exactly. 0.60 is reviewable, not rejected.
    state = _state(
        recipes=[_recipe("At The Floor", [
            _component("mozzarella", 4.0, "oz", confidence=CONF_REVIEW_FLOOR),
        ])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "At The Floor", "price": 8.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 0.80
    assert plate["costed_components"] == 1
    assert plate["coverage"] == 1.0
    assert plate["uncosted"] == []


def test_confidence_a_hair_under_the_floor_is_costed_but_flagged(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    conf = CONF_REVIEW_FLOOR - 0.01
    state = _state(
        recipes=[_recipe("Under The Floor", [
            _component("mozzarella", 4.0, "oz", confidence=conf),
        ])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Under The Floor", "price": 8.00}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    # Costed, with full coverage — nothing was missing, only uncertain.
    assert plate["plate_cost"] == 0.80
    assert plate["costable"] is True
    assert plate["costed_components"] == 1
    assert plate["coverage"] == 1.0
    # sqrt(recipe 1.0 * mean(match 1.0 * quantity 0.59)) * coverage 1.0
    assert plate["confidence"] == round(math.sqrt(1.0 * conf) * 1.0, 4)
    assert plate["uncosted"] == []
    assert len(plate["low_confidence"]) == 1
    # Crossing the floor still reaches a human — it just no longer alters the money.
    assert [q["kind"] for q in update["review_queue"]] == ["plate_quantity"]


def test_component_with_no_stated_confidence_is_still_costed(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Absence of evidence is not a bad guess. draft_po.py keeps this component
    # too, and the two nodes must agree on what they are describing.
    state = _state(
        recipes=[_recipe("Silent", [
            {"raw_name": "mozzarella", "qty": 4.0, "uom": "oz"},
            {"raw_name": "roma tomatoes", "qty": 6.0, "uom": "oz", "confidence": None},
        ])],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA"),
        ],
        menu_items=[{"name": "Silent", "price": 8.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["plate_cost"] == 1.2875
    assert plate["costed_components"] == 2
    assert plate["coverage"] == 1.0
    assert plate["uncosted"] == []


def test_unparseable_confidence_is_treated_as_absent_and_costed(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # A non-numeric confidence is no stated confidence, not a zero. Costing it is
    # the same call draft_po.py makes, and it must not raise into the retry path.
    state = _state(
        recipes=[_recipe("Garbled", [
            _component("mozzarella", 4.0, "oz", confidence="high"),
        ])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Garbled", "price": 8.00}],
    )

    update = cost_plates_node(state)

    assert "retry_count" not in update
    assert update["plate_costs"][0]["plate_cost"] == 0.80
    assert update["plate_costs"][0]["coverage"] == 1.0


def test_quantity_confidence_multiplies_match_confidence_in_the_mean(
    monkeypatch, tmp_path
):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Identity and quantity are separate questions and both bear on the plate.
    # Matching "Parmigiano Reggiano" to the Parmigiano SKU can be a perfect 1.0
    # while saying nothing about whether that much of it belongs in the dish, so
    # a strong match must not paper over a weak quantity — or the reverse.
    state = _state(
        recipes=[_recipe("Mixed", [
            _component("mozzarella", 4.0, "oz", confidence=0.9),
            _component("roma tomatoes", 6.0, "oz", confidence=0.40),
        ], confidence=1.0)],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ", confidence=0.5, method="llm"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA", confidence=1.0),
        ],
        menu_items=[{"name": "Mixed", "price": 8.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    # mean of (0.5 * 0.9) and (1.0 * 0.40) = (0.45 + 0.40) / 2 = 0.425
    assert plate["confidence"] == round(math.sqrt(1.0 * 0.425) * 1.0, 4)
    # A weakly-matched but confidently-measured line and a well-matched but
    # hedged line both pull the mean down; neither is silently discarded.
    assert plate["costed_components"] == 2
    assert plate["coverage"] == 1.0


# --- a plate with nothing left is not costable ------------------------------
#
# A plate where every component is a genuine gap — nothing matched, or no
# conversion exists — still produces a number from the arithmetic: 0.0, which
# reads as a real 0% food cost. scripts/plot_foodcost.py plots that distribution
# against the 28-33% industry band, so a handful of fake zeros would move the chart and the
# finding drawn from it. `plate_cost` is None and `costable` is False instead.
#
# A low-confidence quantity is NOT one of those gaps: the component is costed,
# so a plate can no longer become uncostable merely by being hedged.


def test_plate_with_every_component_uncostable_is_not_costable(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # "each" cannot be converted to the SKU's "lb" — a real gap, not a hedge.
    state = _state(
        recipes=[_recipe("Imported Soda", [
            _component("mozzarella", 1.0, "each"),
        ])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Imported Soda", "price": 3.00}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    assert plate["costable"] is False
    assert plate["plate_cost"] is None
    # A ratio of a cost we do not have is the same fiction as the cost.
    assert plate["food_cost_pct"] is None
    assert plate["menu_price"] == 3.00
    assert plate["costed_components"] == 0
    assert plate["coverage"] == 0.0
    assert len(plate["uncosted"]) == 1
    assert "retry_count" not in update


def test_plate_whose_only_component_is_hedged_is_still_costed(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Exactly the live case: "+add any one topping", one 0.40-confidence topping.
    # Under the old rule this plate lost its only component and became
    # uncostable. It is a real, cheap plate with an uncertain quantity.
    state = _state(
        recipes=[_recipe("+add any one topping", [
            _component("mozzarella", 4.0, "oz", confidence=0.40),
        ])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "+add any one topping", "price": 3.00}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    assert plate["costable"] is True
    assert plate["plate_cost"] == 0.80
    assert plate["coverage"] == 1.0
    assert len(plate["low_confidence"]) == 1
    assert "0.40" in plate["low_confidence"][0]


def test_uncostable_plate_is_not_flagged_as_outside_the_sane_band(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 0.0 / 3.00 would be a 0% food cost and would be queued as an outlier
    # alongside the genuinely broken 80% plate. It is not an outlier, it is a gap.
    state = _state(
        recipes=[
            _recipe("Imported Soda", [_component("mozzarella", 1.0, "each")]),
            _recipe("Broken Pizza", [_component("mozzarella", 2.5, "lb")]),
        ],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Imported Soda", "price": 3.00},
                    {"name": "Broken Pizza", "price": 10.00}],
    )

    update = cost_plates_node(state)

    assert [q["ref"] for q in update["review_queue"]] == ["Broken Pizza"]


def test_uncostable_plates_are_kept_out_of_the_mean_and_reported_separately(
    monkeypatch, tmp_path, capsys
):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Two real dishes and one plate where nothing could be costed. Averaging the
    # zero in would report a costing quality nobody measured — that is how
    # joes-pizza-carmine read 0.43 instead of 0.85 across three sodas and three
    # genuine pizzas. "Cheap Soda" is hedged but real, so it counts.
    state = _state(
        recipes=[
            _recipe("Margherita", [
                _component("mozzarella", 4.0, "oz"),
                _component("roma tomatoes", 6.0, "oz"),
            ]),
            _recipe("Cheap Soda", [
                _component("mozzarella", 1.0, "oz", confidence=0.45),
            ]),
            _recipe("Imported Soda", [_component("mozzarella", 1.0, "each")]),
        ],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA"),
        ],
        menu_items=[{"name": "Margherita", "price": 4.00},
                    {"name": "Cheap Soda", "price": 3.00},
                    {"name": "Imported Soda", "price": 4.00}],
    )

    update = cost_plates_node(state)
    out = capsys.readouterr().out

    assert [p["costable"] for p in update["plate_costs"]] == [True, True, False]
    # Mean coverage over the two costable plates, not dragged toward zero by the
    # plate that could not be costed at all.
    assert "mean coverage 1.00" in out
    assert "3 plates" in out
    assert "2 costed" in out
    assert "1 not costable" in out
    assert out.isascii(), "printed on a cp1252 console; non-ASCII trips the breaker"


def test_a_partially_costed_plate_is_still_costable(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # One of two components is a genuine gap — no SKU matched the tomato — so
    # this is under-costed rather than uncostable, and it must keep reporting a
    # real number with a real (lowered) coverage.
    state = _state(
        recipes=[_recipe("Half Caprese", [
            _component("mozzarella", 4.0, "oz", confidence=0.95),
            _component("roma tomatoes", 6.0, "oz", confidence=0.95),
        ])],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ"),
            _match("roma tomatoes", None),
        ],
        menu_items=[{"name": "Half Caprese", "price": 8.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["costable"] is True
    assert plate["plate_cost"] == 0.80
    assert plate["food_cost_pct"] == 0.10
    assert plate["coverage"] == 0.5


def test_a_genuinely_cheap_plate_is_still_a_real_number(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # The distinction the whole change exists for: a plate that really does cost
    # almost nothing keeps its number and its outlier flag. Only "we could not
    # cost this" becomes None. 0.1 oz of mozzarella = 0.00625 lb * 3.20 = 0.02.
    state = _state(
        recipes=[_recipe("Bread Service", [_component("mozzarella", 0.1, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Bread Service", "price": 6.00}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    assert plate["costable"] is True
    assert plate["plate_cost"] == 0.02
    assert plate["food_cost_pct"] == round(0.02 / 6.00, 4)
    assert [q["ref"] for q in update["review_queue"]] == ["Bread Service"]


# --- confidence is a product ----------------------------------------------


def test_partial_coverage_drags_confidence_below_the_component_mean(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # Recipe confidence 1.0, the one match we used is 1.0, coverage 0.5.
    # Averaging would give ~0.83; multiplying gives 0.5.
    state = _state(
        recipes=[_recipe("Half Known", [
            _component("mozzarella", 4.0, "oz"),
            _component("fish sauce", 2.0, "fl_oz"),
        ], confidence=1.0)],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ", confidence=1.0),
            _match("fish sauce", None, confidence=0.0, method="llm"),
        ],
        menu_items=[{"name": "Half Known", "price": 10.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    component_mean = 1.0
    assert plate["coverage"] == 0.5
    assert plate["confidence"] == 0.5
    assert plate["confidence"] < component_mean


def test_confidence_is_a_geometric_mean_of_the_two_opinions(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # sqrt(recipe 0.8 * mean match (0.9 + 0.7)/2 = 0.8) * coverage 1.0 -> 0.8
    state = _state(
        recipes=[_recipe("Pasta", [
            _component("mozzarella", 4.0, "oz"),
            _component("roma tomatoes", 4.0, "oz"),
        ], confidence=0.8)],
        sku_matches=[
            _match("mozzarella", "DAIRY-MOZZ", confidence=0.9, method="normalized"),
            _match("roma tomatoes", "PROD-TOMATO-ROMA", confidence=0.7, method="llm"),
        ],
        menu_items=[{"name": "Pasta", "price": 10.00}],
    )

    plate = cost_plates_node(state)["plate_costs"][0]

    assert plate["confidence"] == round(math.sqrt(0.8 * 0.8) * 1.0, 4) == 0.8
    # The raw triple product would have said 0.64, which is below the 0.60 review
    # floor's neighbourhood for no reason a reviewer could defend: two 0.8-ish
    # opinions about a fully-covered plate are not evidence of a shaky plate.
    assert plate["confidence"] > round(0.8 * 0.8 * 1.0, 4)


def test_coverage_scales_confidence_linearly(monkeypatch, tmp_path):
    """Coverage multiplies outright, so half a plate reports at most half.

    This is the property the geometric mean must NOT dilute: coverage is a
    measured fact, not a model opinion, and averaging it in would let a
    confident model talk its way past a plate it only half priced.
    """
    _use_fixture_catalog(monkeypatch, tmp_path)

    def build(second_component):
        return _state(
            recipes=[_recipe("Pasta", [
                _component("mozzarella", 4.0, "oz"),
                second_component,
            ], confidence=0.9)],
            sku_matches=[
                _match("mozzarella", "DAIRY-MOZZ", confidence=0.9),
                _match("roma tomatoes", "PROD-TOMATO-ROMA", confidence=0.9),
            ],
            menu_items=[{"name": "Pasta", "price": 10.00}],
        )

    full = cost_plates_node(
        build(_component("roma tomatoes", 4.0, "oz"))
    )["plate_costs"][0]
    # "tsp" has no conversion, so this component cannot be costed: coverage 0.5.
    half = cost_plates_node(
        build(_component("roma tomatoes", 4.0, "tsp"))
    )["plate_costs"][0]

    assert full["coverage"] == 1.0
    assert half["coverage"] == 0.5
    assert half["confidence"] == round(full["confidence"] * 0.5, 4)


# --- the sane band --------------------------------------------------------


def test_eighty_percent_food_cost_lands_in_review_queue(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 2.5 lb mozzarella * 3.20 = 8.00 against a 10.00 menu price -> 80%.
    state = _state(
        recipes=[_recipe("Broken Pizza", [_component("mozzarella", 2.5, "lb")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Broken Pizza", "price": 10.00}],
    )

    update = cost_plates_node(state)
    plate = update["plate_costs"][0]

    assert plate["plate_cost"] == 8.00
    assert plate["food_cost_pct"] == 0.8

    assert len(update["review_queue"]) == 1
    flagged = update["review_queue"][0]
    assert flagged["kind"] == "plate_cost"
    assert flagged["ref"] == "Broken Pizza"
    assert flagged["confidence"] == plate["confidence"]
    assert "80% food cost" in flagged["question"]
    assert flagged["payload"]["plate_cost"] == 8.00
    assert flagged["payload"]["menu_price"] == 10.00


def test_four_percent_food_cost_also_flagged(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 0.25 lb mozzarella = 0.80 against a 20.00 price -> 4%, a missed ingredient.
    state = _state(
        recipes=[_recipe("Suspiciously Cheap", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Suspiciously Cheap", "price": 20.00}],
    )

    update = cost_plates_node(state)

    assert update["plate_costs"][0]["food_cost_pct"] == 0.04
    assert [q["ref"] for q in update["review_queue"]] == ["Suspiciously Cheap"]
    assert "4% food cost" in update["review_queue"][0]["question"]


def test_band_edges_are_inclusive(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    # 0.80 / 8.00 = 0.10 exactly, and 0.80 / 1.4545... is not worth faking:
    # use the low edge, which is the one a real menu lands on.
    state = _state(
        recipes=[_recipe("Edge", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Edge", "price": 8.00}],
    )

    update = cost_plates_node(state)

    assert update["plate_costs"][0]["food_cost_pct"] == SANE_BAND_LOW == 0.10
    assert update["review_queue"] == []
    assert SANE_BAND_HIGH == 0.55


def test_review_queue_is_appended_not_replaced(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path)

    existing = {"kind": "sku_match", "ref": "fish sauce", "confidence": 0.0,
                "question": "Which SKU is 'fish sauce'?", "payload": {}}
    state = _state(
        recipes=[_recipe("Broken Pizza", [_component("mozzarella", 2.5, "lb")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Broken Pizza", "price": 10.00}],
        review_queue=[existing],
    )

    update = cost_plates_node(state)

    assert update["review_queue"][0] == existing
    assert len(update["review_queue"]) == 2


# --- failure handling -----------------------------------------------------


def test_malformed_catalog_increments_retry_count(monkeypatch, tmp_path):
    _use_fixture_catalog(monkeypatch, tmp_path, skus=[{"display_name": "no sku_id here"}])

    state = _state(
        recipes=[_recipe("Margherita", [_component("mozzarella", 4.0, "oz")])],
        sku_matches=[_match("mozzarella", "DAIRY-MOZZ")],
        menu_items=[{"name": "Margherita", "price": 18.00}],
        retry_count=2,
    )

    update = cost_plates_node(state)

    assert update["plate_costs"] == []
    assert update["retry_count"] == 3
    assert "KeyError" in update["last_error"]
    assert "stage" not in update


def test_missing_catalog_file_increments_retry_count(monkeypatch, tmp_path):
    monkeypatch.setattr(cost_module, "CATALOG_PATH", str(tmp_path / "nope.json"))

    update = cost_plates_node(_state([], [], []))

    assert update["retry_count"] == 1
    assert "FileNotFoundError" in update["last_error"]
    assert "stage" not in update


# --- the real catalog -----------------------------------------------------


def test_real_catalog_loads_and_every_sku_has_a_price_and_uom():
    catalog = _load_catalog()

    assert len(catalog) > 50
    for sku_id, sku in catalog.items():
        assert isinstance(sku["price_per_uom"], (int, float)), sku_id
        assert sku["price_per_uom"] > 0, sku_id
        # Every catalog uom must be convertible to itself at minimum.
        assert _convert(1.0, sku["uom"], sku["uom"]) == 1.0, sku_id
