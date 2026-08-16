from src.state import (
    CONF_AUTO_ACCEPT,
    CONF_REVIEW_FLOOR,
    initial_dayzero_state,
    initial_state,
)


def test_initial_dayzero_state_keeps_base_keys():
    s = initial_dayzero_state("joes-pizza-carmine")
    assert s["input"] == "joes-pizza-carmine"
    assert s["extracted_data"] == {}
    assert s["messages"] == []
    assert s["retry_count"] == 0
    assert s["last_error"] == ""
    assert s["human_decision"] == ""
    assert s["needs_human"] is False


def test_initial_dayzero_state_empty_dayzero_fields():
    s = initial_dayzero_state("x")
    assert s["restaurant"] == {}
    assert s["demand_forecast"] == {}
    assert s["purchase_order"] == {}
    assert s["menu_items"] == []
    assert s["recipes"] == []
    assert s["sku_matches"] == []
    assert s["plate_costs"] == []
    assert s["par_levels"] == []
    assert s["review_queue"] == []
    assert s["stage"] == ""


def test_initial_dayzero_state_field_types():
    s = initial_dayzero_state("x")
    for key in ("restaurant", "demand_forecast", "purchase_order"):
        assert isinstance(s[key], dict), key
    for key in (
        "menu_items",
        "recipes",
        "sku_matches",
        "plate_costs",
        "par_levels",
        "review_queue",
    ):
        assert isinstance(s[key], list), key
    assert isinstance(s["stage"], str)


def test_initial_dayzero_state_is_superset_of_initial_state():
    base = set(initial_state("x"))
    assert base <= set(initial_dayzero_state("x"))


def test_initial_dayzero_state_instances_are_independent():
    a = initial_dayzero_state("a")
    b = initial_dayzero_state("b")
    a["menu_items"].append({"name": "pie"})
    a["restaurant"]["slug"] = "a"
    assert b["menu_items"] == []
    assert b["restaurant"] == {}


def test_initial_state_unchanged_by_extension():
    s = initial_state("hello")
    assert set(s) == {
        "input",
        "extracted_data",
        "messages",
        "retry_count",
        "last_error",
        "human_decision",
        "needs_human",
    }


def test_confidence_thresholds():
    assert CONF_AUTO_ACCEPT == 0.85
    assert CONF_REVIEW_FLOOR == 0.60
    assert 0.0 < CONF_REVIEW_FLOOR < CONF_AUTO_ACCEPT < 1.0
