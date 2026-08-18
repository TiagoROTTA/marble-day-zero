"""Tests for the draft purchase order node.

The point of this node is that a founder can be walked through any line of the
order live, so the tests trace the same path a human would: consumption ->
par level -> packs -> line cost -> total. No LLM is involved, so every assertion
below is exact arithmetic rather than a tolerance.
"""
import math

import pytest

from src.nodes.draft_po import (
    DAYS_COVER,
    DEFAULT_DAYS_COVER,
    SAFETY_FACTOR,
    _consumption,
    _load_catalog,
    _par_levels,
    _to_packs,
    draft_po_node,
)

# Real catalog entries, so the arithmetic in these tests is the arithmetic the
# pipeline runs. Roma tomato: 25 lb case at $32.50, produce (2 days cover).
ROMA = "PROD-TOMATO-ROMA"
BEEF = "PROT-BEEF-GROUND8020"      # 40 lb case at $178.00, protein (3 days cover)
FLOUR = "DRY-FLOUR-ALLPURPOSE"     # 50 lb sack at $24.50, dry_goods (14 days cover)
# The SKU from the live incident: a 20 lb wedge at $388.00, dairy (3 days cover),
# bought for a Vietnamese restaurant off a 0.40-confidence hallucination.
PARM = "DAIR-PARM-REGGIANO"


def _state(**overrides) -> dict:
    """A minimal but complete Day Zero state: one dish, three components."""
    state = {
        "restaurant": {"name": "Test Trattoria"},
        "menu_items": [{"name": "Margherita", "section": "Pizza", "price": 20.0}],
        "recipes": [
            {
                "item_name": "Margherita",
                "yield_qty": 1,
                "yield_uom": "each",
                "components": [
                    {"raw_name": "roma tomatoes", "qty": 4.0, "uom": "oz"},
                    {"raw_name": "ground beef", "qty": 2.0, "uom": "oz"},
                    {"raw_name": "flour", "qty": 8.0, "uom": "oz"},
                ],
            }
        ],
        "sku_matches": [
            {"raw_name": "roma tomatoes", "sku_id": ROMA, "confidence": 0.9},
            {"raw_name": "ground beef", "sku_id": BEEF, "confidence": 0.9},
            {"raw_name": "flour", "sku_id": FLOUR, "confidence": 0.9},
        ],
        "demand_forecast": {
            "covers_per_week": 700.0,
            "item_mix": {"Margherita": 1.0},
            "assumptions": ["Forecast assumption carried from the forecast node."],
        },
        "review_queue": [],
    }
    state.update(overrides)
    return state


# --------------------------------------------------------------------------
# Pass 3 — the rounding rule, tested on the exact case from the step file.
# --------------------------------------------------------------------------


def test_four_point_two_lb_of_a_twenty_five_lb_case_orders_exactly_one_pack():
    """A restaurant cannot buy 4.2 lb of a 25 lb case; it buys one case."""
    skus = _load_catalog()
    lines = _to_packs([{"sku_id": ROMA, "par_qty": 4.2, "uom": "lb"}], skus)

    assert len(lines) == 1
    assert lines[0]["packs"] == 1
    assert lines[0]["pack_qty"] == 25.0
    assert lines[0]["line_cost"] == 32.5
    # The theoretical par survives alongside the rounded pack: the ~20.8 lb gap
    # is real week-one overstock and has to stay visible.
    assert lines[0]["par_qty"] == 4.2


def test_packs_round_up_not_to_nearest():
    """26 lb of a 25 lb case is two cases, not one."""
    skus = _load_catalog()
    lines = _to_packs([{"sku_id": ROMA, "par_qty": 26.0, "uom": "lb"}], skus)

    assert lines[0]["packs"] == 2
    assert lines[0]["line_cost"] == 65.0


def test_zero_consumption_sku_is_omitted_not_ordered_at_zero():
    """A SKU with no projected consumption produces no line at all."""
    skus = _load_catalog()
    lines = _to_packs([{"sku_id": ROMA, "par_qty": 0.0, "uom": "lb"}], skus)

    assert lines == []


def test_zero_share_item_never_reaches_the_order():
    """An item nobody is forecast to order contributes no consumption."""
    state = _state(
        demand_forecast={
            "covers_per_week": 700.0,
            "item_mix": {"Margherita": 0.0},
            "assumptions": [],
        }
    )
    consumption, _, _ = _consumption(state, _load_catalog())
    assert consumption == {}

    result = draft_po_node(state)
    assert result["purchase_order"]["vendor_lines"] == {}
    assert result["purchase_order"]["total_cost"] == 0.0


# --------------------------------------------------------------------------
# Pass 1 — consumption, including the unit conversion borrowed from cost_plates.
# --------------------------------------------------------------------------


def test_consumption_converts_component_units_into_the_sku_uom():
    """700 covers x 4 oz of tomato = 2800 oz = 175 lb, in the SKU's own uom."""
    consumption, gaps, _ = _consumption(_state(), _load_catalog())

    assert consumption[ROMA] == pytest.approx(175.0)
    assert consumption[BEEF] == pytest.approx(87.5)
    assert consumption[FLOUR] == pytest.approx(350.0)
    assert gaps == []


def test_consumption_does_not_divide_by_recipe_yield():
    """THE INVARIANT: one ORDER consumes the recipe as written, undivided.

    `item_mix` shares are "the fraction of covers that order that item"
    (src/nodes/forecast.py), so `share x covers` counts ORDERS of a menu item,
    not diners. A recipe describes one SOLD UNIT — what one order buys — and
    yield_qty only reports how many servings that unit feeds.

    This assertion used to read `175.0 / 4`, because _consumption divided the
    components by yield_qty. That under-ordered every multi-serving line by
    exactly its yield: 1.75 oz of flour for a pie that takes 14. Same bug as the
    one in cost_plates.py, fixed the same way so plate cost and order quantity
    still describe the same thing.
    """
    state = _state()
    state["recipes"][0]["yield_qty"] = 4
    consumption, _, _ = _consumption(state, _load_catalog())

    assert consumption[ROMA] == pytest.approx(175.0)


def test_consumption_is_unchanged_by_any_yield_value():
    """The yield is now purely reporting: it must not move a purchase quantity."""
    baseline, _, _ = _consumption(_state(), _load_catalog())

    for yield_qty in (0, None, 1, 8, 0.5):
        state = _state()
        state["recipes"][0]["yield_qty"] = yield_qty
        consumption, _, _ = _consumption(state, _load_catalog())

        assert consumption[ROMA] == pytest.approx(baseline[ROMA])


def test_a_menu_item_with_no_recipe_blames_the_decomposer():
    """The dish is on the menu and nobody wrote it down: that is a missing recipe."""
    state = _state(recipes=[])
    _, gaps, _ = _consumption(state, _load_catalog())

    assert gaps == ["Margherita (no decomposed recipe; not ordered for)"]


def test_a_forecast_name_that_is_not_a_menu_item_says_so_instead():
    """Joe's Pizza: the mix said "Classic Cheese Pie", the menu prints "... 8 slices".

    The old wording accused the decomposer of not producing a recipe that in fact
    existed, which sends the next reader to the wrong node entirely.
    """
    state = _state(
        demand_forecast={
            "covers_per_week": 700.0,
            "item_mix": {"Margherit": 1.0},
            "assumptions": [],
        }
    )
    _, gaps, _ = _consumption(state, _load_catalog())

    assert gaps == ["Margherit (not a menu item name; matched no recipe; not ordered for)"]


def test_unconvertible_component_becomes_a_visible_gap():
    """No conversion exists from `bunch` to `lb`: the quantity is dropped loudly."""
    state = _state()
    state["recipes"][0]["components"][0]["uom"] = "bunch"
    consumption, gaps, _ = _consumption(state, _load_catalog())

    assert ROMA not in consumption
    assert any("no conversion from bunch to lb" in g for g in gaps)


# --------------------------------------------------------------------------
# skipped_refs — a skip must genuinely remove the quantity.
# --------------------------------------------------------------------------


def test_skipped_sku_id_is_absent_from_vendor_lines():
    state = _state(skipped_refs=[ROMA])
    result = draft_po_node(state)
    po = result["purchase_order"]

    ordered = {line["sku_id"] for lines in po["vendor_lines"].values() for line in lines}
    assert ROMA not in ordered
    assert BEEF in ordered

    # excluded_skus is a defensible record now, not a bare list of refs: the
    # reason, the dish and the money not spent all have to be on it.
    assert [e["sku_id"] for e in po["excluded_skus"]] == [ROMA]
    excluded = po["excluded_skus"][0]
    assert excluded["reason"] == "skipped_in_review"
    assert excluded["dishes"] == ["Margherita"]
    assert excluded["cost_not_spent"] == 97.5
    assert po["excluded_cost_total"] == 97.5


def test_skipped_raw_name_is_also_excluded():
    """review_wait_node records the flagged `ref`, which for a SKU match is the raw name."""
    state = _state(skipped_refs=["ground beef"])
    result = draft_po_node(state)

    ordered = {
        line["sku_id"]
        for lines in result["purchase_order"]["vendor_lines"].values()
        for line in lines
    }
    assert BEEF not in ordered
    assert ROMA in ordered


def test_skipping_everything_yields_an_empty_order_not_a_crash():
    state = _state(skipped_refs=[ROMA, BEEF, FLOUR])
    result = draft_po_node(state)

    assert result["purchase_order"]["vendor_lines"] == {}
    assert result["purchase_order"]["total_cost"] == 0.0
    # No lines means no cover policy was applied, so the card is given nothing
    # to print rather than a number describing an order that does not exist.
    assert result["purchase_order"]["days_cover_label"] == ""
    assert result["stage"] == "po_drafted"


# --------------------------------------------------------------------------
# The confidence floor. The live Madame Vo run bought a $388 wheel of
# Parmigiano Reggiano for a Vietnamese restaurant off a component the
# decomposer itself scored 0.40 — below the 0.60 review floor, classified as a
# gap by review_gate, so no human was ever asked. These tests pin the rule that
# a gap must not become a line, and the rule that the withholding is loud.
# --------------------------------------------------------------------------


def _low_confidence_state(confidence: float = 0.40) -> dict:
    """Margherita, plus one hallucinated component the decomposer distrusts."""
    state = _state()
    state["recipes"][0]["components"].append(
        {"raw_name": "parmesan", "qty": 0.3, "uom": "oz", "confidence": confidence}
    )
    state["sku_matches"].append(
        # Matched correctly and with total certainty: the canonicalizer is not at
        # fault here, and its confidence must not launder the decomposer's guess.
        {"raw_name": "parmesan", "sku_id": PARM, "method": "alias", "confidence": 1.0}
    )
    return state


def test_component_below_the_review_floor_never_becomes_a_line():
    result = draft_po_node(_low_confidence_state())
    po = result["purchase_order"]

    ordered = {line["sku_id"] for lines in po["vendor_lines"].values() for line in lines}
    assert PARM not in ordered
    assert ROMA in ordered, "the confident components must still be bought"


def test_the_withheld_component_is_recorded_with_enough_detail_to_defend():
    po = draft_po_node(_low_confidence_state())["purchase_order"]

    excluded = next(e for e in po["excluded_skus"] if e["sku_id"] == PARM)
    assert excluded["reason"] == "confidence_below_floor"
    assert excluded["confidence"] == 0.40
    assert excluded["dishes"] == ["Margherita"]
    assert excluded["cost_not_spent"] > 0.0
    assert excluded["qty_withheld"] > 0.0
    assert excluded["display_name"]


def test_the_withheld_spend_is_totalled_on_the_purchase_order():
    """`excluded_cost_total` must be the order's own difference, not an estimate."""
    withheld = draft_po_node(_low_confidence_state())["purchase_order"]
    kept = draft_po_node(_state())["purchase_order"]

    # Same state minus the low-confidence component: the withheld total is
    # exactly the money the two orders differ by.
    assert withheld["total_cost"] == kept["total_cost"]
    assert withheld["excluded_cost_total"] == pytest.approx(
        sum(e["cost_not_spent"] for e in withheld["excluded_skus"])
    )
    assert withheld["excluded_cost_total"] > 0.0


@pytest.mark.parametrize("confidence", [0.60, 0.61, 0.85, 1.0])
def test_a_component_at_or_above_the_floor_is_bought(confidence):
    """0.60 is the floor, not the first rejected value: `>=` keeps it."""
    po = draft_po_node(_low_confidence_state(confidence))["purchase_order"]

    ordered = {line["sku_id"] for lines in po["vendor_lines"].values() for line in lines}
    assert PARM in ordered
    assert po["excluded_skus"] == []
    assert po["excluded_cost_total"] == 0.0


def test_a_component_with_no_stated_confidence_is_bought():
    """Absent evidence is not evidence of a bad guess. Dropping it would be the
    same silent-loss bug, pointed the other way."""
    state = _state()
    state["recipes"][0]["components"].append({"raw_name": "parmesan", "qty": 0.3, "uom": "oz"})
    state["sku_matches"].append({"raw_name": "parmesan", "sku_id": PARM, "confidence": 1.0})

    po = draft_po_node(state)["purchase_order"]
    ordered = {line["sku_id"] for lines in po["vendor_lines"].values() for line in lines}
    assert PARM in ordered
    assert po["excluded_skus"] == []


def test_a_sku_used_confidently_elsewhere_is_still_bought_for_that_dish():
    """Only the shaky quantity is withheld, not the whole SKU.

    Jasmine rice reaching 0.5 confidence in one dish is no reason to send a
    Vietnamese restaurant no rice at all. `cost_not_spent` is then the delta
    between the two orders, pack rounding included:

      confident dish alone: 175 lb -> 25 lb/day -> 57.5 par -> 3 cases  $97.50
      both dishes:          350 lb -> 50 lb/day -> 115  par -> 5 cases  $162.50
    """
    state = _state()
    state["menu_items"].append({"name": "Tomato salad", "section": "Sides", "price": 12.0})
    state["recipes"].append({
        "item_name": "Tomato salad",
        "yield_qty": 1,
        "yield_uom": "each",
        "components": [
            {"raw_name": "roma tomatoes", "qty": 4.0, "uom": "oz", "confidence": 0.40}
        ],
    })
    state["demand_forecast"]["item_mix"]["Tomato salad"] = 1.0

    po = draft_po_node(state)["purchase_order"]

    line = next(
        line
        for lines in po["vendor_lines"].values()
        for line in lines
        if line["sku_id"] == ROMA
    )
    assert line["packs"] == 3, "the confident dish still gets its tomatoes"

    excluded = next(e for e in po["excluded_skus"] if e["sku_id"] == ROMA)
    assert excluded["dishes"] == ["Tomato salad"]
    assert excluded["cost_not_spent"] == pytest.approx(162.50 - 97.50)


def test_the_lowest_confidence_is_the_one_reported():
    """Two shaky dishes share a SKU: the number quoted is the worst one."""
    state = _low_confidence_state(0.55)
    state["menu_items"].append({"name": "Cheese plate", "section": "Sides", "price": 14.0})
    state["recipes"].append({
        "item_name": "Cheese plate",
        "yield_qty": 1,
        "yield_uom": "each",
        "components": [
            {"raw_name": "parmesan", "qty": 1.0, "uom": "oz", "confidence": 0.30}
        ],
    })
    state["demand_forecast"]["item_mix"]["Cheese plate"] = 1.0

    po = draft_po_node(state)["purchase_order"]
    excluded = next(e for e in po["excluded_skus"] if e["sku_id"] == PARM)

    assert excluded["confidence"] == 0.30
    assert set(excluded["dishes"]) == {"Margherita", "Cheese plate"}


def test_the_exclusion_is_announced_on_stdout_in_pure_ascii(capsys):
    """A withheld line nobody is told about is the same bug as a silent buy.

    ASCII is load-bearing: a non-ASCII dash in a node's print() raises
    UnicodeEncodeError inside that node's own try/except on the cp1252 Windows
    console, which trips the circuit breaker into a retry loop.
    """
    draft_po_node(_low_confidence_state())
    out = capsys.readouterr().out

    assert "WITHHELD" in out
    assert "confidence floor" in out
    assert out.isascii(), f"non-ASCII in draft_po stdout: {out!r}"


def test_the_assumptions_say_what_was_withheld_and_why():
    po = draft_po_node(_low_confidence_state())["purchase_order"]
    joined = " ".join(po["assumptions"])

    assert "NOT on this order" in joined
    assert "60%" in joined
    assert "confidence_below_floor" in joined


def test_the_slack_card_shows_the_money_not_spent():
    """End of the chain: the withholding has to reach the human, not just state."""
    from src.slack.blocks import purchase_order_blocks

    po = draft_po_node(_low_confidence_state())["purchase_order"]
    blocks = purchase_order_blocks("t-1", "Test Trattoria", po)

    text = " ".join(
        b["text"]["text"] for b in blocks if isinstance(b.get("text"), dict)
    )
    assert "Withheld" in text
    assert "confidence below review floor" in text
    assert f"${po['excluded_cost_total']:,.2f}" in text


# --------------------------------------------------------------------------
# Pass 2 — par levels and the sentence that explains them.
# --------------------------------------------------------------------------


def test_par_level_arithmetic_and_days_cover_by_category():
    levels = {lvl["sku_id"]: lvl for lvl in _par_levels(
        {ROMA: 175.0, FLOUR: 350.0}, _load_catalog()
    )}

    roma = levels[ROMA]
    assert roma["days_cover"] == DAYS_COVER["produce"] == 2
    assert roma["daily_consumption"] == pytest.approx(25.0)
    assert roma["par_qty"] == pytest.approx(25.0 * 2 * 1.15)
    assert roma["uom"] == "lb"

    flour = levels[FLOUR]
    assert flour["days_cover"] == DAYS_COVER["dry_goods"] == 14
    assert flour["par_qty"] == pytest.approx(50.0 * 14 * 1.15)


def test_days_cover_label_reports_the_range_the_order_actually_used():
    """The Slack card gets a range, because one order carries several policies.

    ROMA is produce (2 days) and FLOUR is dry_goods (14), so no single number
    describes the order. The label is derived from the lines that survived pack
    rounding, not from the whole policy table.
    """
    po = draft_po_node(_state())["purchase_order"]

    assert po["days_cover_label"] == "2-14 days cover by category"


def test_days_cover_label_is_singular_when_one_policy_covers_the_order():
    state = _state()
    state["recipes"] = [{
        "item_name": "Margherita",
        "yield_qty": 1,
        "components": [
            {"raw_name": "roma tomatoes", "qty": 6.0, "uom": "oz", "confidence": 0.9}
        ],
    }]
    po = draft_po_node(state)["purchase_order"]

    assert po["days_cover_label"] == "2 days cover"


def test_every_catalog_category_has_a_days_cover_policy():
    """No real SKU should reach the fallback: that would be an unstated policy."""
    categories = {sku.get("category") for sku in _load_catalog().values()}
    assert categories, "expected a non-empty catalog"
    assert categories <= set(DAYS_COVER), f"uncovered categories: {categories - set(DAYS_COVER)}"


def test_unknown_category_falls_back_to_default_cover():
    levels = _par_levels({"MADE-UP-SKU": 70.0}, {})
    assert levels[0]["days_cover"] == DEFAULT_DAYS_COVER
    assert levels[0]["par_qty"] == pytest.approx(10.0 * DEFAULT_DAYS_COVER * 1.15)


def test_rationale_names_every_input():
    levels = _par_levels({ROMA: 175.0}, _load_catalog())
    rationale = levels[0]["rationale"]

    assert "2 days cover" in rationale
    assert "produce" in rationale
    assert "25.0 lb/day" in rationale
    assert "+15% safety" in rationale
    assert SAFETY_FACTOR == 0.15


# --------------------------------------------------------------------------
# Pass 4 — grouping, ordering, and the total that has to add up.
# --------------------------------------------------------------------------


def test_total_cost_equals_the_sum_of_every_line_cost():
    po = draft_po_node(_state())["purchase_order"]

    every_line = [line for lines in po["vendor_lines"].values() for line in lines]
    assert every_line, "expected a non-empty order"
    assert po["total_cost"] == pytest.approx(round(sum(l["line_cost"] for l in every_line), 2))


def test_lines_are_grouped_by_vendor_and_sorted_by_descending_cost():
    po = draft_po_node(_state())["purchase_order"]

    # Roma tomato and ground beef are both Restaurant Depot in the catalog.
    depot = po["vendor_lines"]["Restaurant Depot"]
    assert {line["sku_id"] for line in depot} >= {ROMA, BEEF, FLOUR}
    costs = [line["line_cost"] for line in depot]
    assert costs == sorted(costs, reverse=True)


def test_one_line_traced_end_to_end_by_hand():
    """The walkthrough a founder gets on the call, asserted step by step.

    700 covers/week x share 1.0        = 700 servings
    700 x 4 oz                          = 2800 oz = 175 lb/week
    175 / 7                             = 25 lb/day
    25 x 2 days cover x 1.15            = 57.5 lb par
    ceil(57.5 / 25 lb case)             = 3 cases
    3 x $32.50                          = $97.50
    """
    result = draft_po_node(_state())

    par = {lvl["sku_id"]: lvl for lvl in result["par_levels"]}[ROMA]
    assert par["daily_consumption"] == pytest.approx(25.0)
    assert par["par_qty"] == pytest.approx(57.5)

    line = next(
        line
        for lines in result["purchase_order"]["vendor_lines"].values()
        for line in lines
        if line["sku_id"] == ROMA
    )
    assert line["packs"] == math.ceil(57.5 / 25.0) == 3
    assert line["line_cost"] == 97.5


def test_node_returns_the_shape_slack_and_the_graph_expect():
    result = draft_po_node(_state())
    po = result["purchase_order"]

    assert result["stage"] == "po_drafted"
    assert result["last_error"] == ""
    assert po["generated_at_stage"] == "po_drafted"
    assert po["covers_per_week"] == 700.0
    assert isinstance(po["total_cost"], float)

    line = next(iter(next(iter(po["vendor_lines"].values()))))
    assert set(line) == {
        "sku_id", "display_name", "packs", "pack_unit",
        "pack_qty", "pack_uom", "line_cost", "par_qty",
    }


def test_assumptions_carry_the_forecast_and_add_this_nodes_own():
    po = draft_po_node(_state())["purchase_order"]
    joined = " ".join(po["assumptions"])

    assert "Forecast assumption carried from the forecast node." in po["assumptions"]
    assert "days of cover" in joined.lower()
    assert "15%" in joined
    assert "packs" in joined.lower()
    assert "catalog" in joined.lower()


def test_prints_the_food_cost_ratio(capsys):
    draft_po_node(_state())
    out = capsys.readouterr().out

    assert "food-cost ratio" in out
    assert "%" in out


# --------------------------------------------------------------------------
# The approval pair added to hitl.py: notify sends, wait pauses. Split so the
# Slack message is not re-sent when the node re-executes on resume.
# --------------------------------------------------------------------------


def test_po_notify_sends_the_order_once_and_returns_no_state(monkeypatch):
    import src.nodes.hitl as hitl

    sent = []
    monkeypatch.setattr(hitl, "get_config", lambda: {"configurable": {"thread_id": "t-1"}})
    monkeypatch.setattr(
        hitl, "send_purchase_order", lambda *args: sent.append(args) or "ts-1"
    )

    po = draft_po_node(_state())["purchase_order"]
    out = hitl.po_notify_node({"restaurant": {"name": "Test Trattoria"}, "purchase_order": po})

    assert out == {}
    assert len(sent) == 1
    assert sent[0][0] == "t-1"
    assert sent[0][1] == "Test Trattoria"
    assert sent[0][2]["total_cost"] == po["total_cost"]


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_po_wait_returns_the_decision_and_clears_the_retry_counter(monkeypatch, decision):
    import src.nodes.hitl as hitl

    monkeypatch.setattr(hitl, "interrupt", lambda payload: {"decision": decision})
    out = hitl.po_wait_node({"retry_count": 2})

    assert out == {
        "human_decision": decision,
        "needs_human": False,
        "stage": "po_approved",
        "retry_count": 0,
    }


def test_failure_increments_retry_and_feeds_the_error_back():
    """A malformed forecast must not raise: it must route through the breaker."""
    result = draft_po_node({"demand_forecast": {"item_mix": "not a dict"}, "retry_count": 2})

    assert result["retry_count"] == 3
    assert result["last_error"]
    assert result["purchase_order"] == {}
